"""Agentic naturalisation: rewrite a question with the repository open.

The plain naturaliser (`naturalize.py`) rewrites from the template alone, so it
can only keep or shorten names. Keeping a fully qualified name leaks answers --
`laya.evals_cli._cmd_run` spells out `laya/evals_cli.py` -- and shortening is
only allowed when the short name is unique, so the leaky template survives
exactly where the repository is most confusing. A writer that can read the
code can instead refer to an entity the way a colleague would: its short name
plus the context that pins it down.

That freedom brings two risks the plain naturaliser avoids by construction,
and each has its own gate:

  writer (repo tools, answer withheld)
    -> deterministic checks  no qualified names, no answer tokens, no invented
                             names, every bare name the query uses kept
    -> resolver (repo tools, sees ONLY the rewrite)
                             must recover exactly the entities the ground-truth
                             Cypher is parameterised on. A description that
                             drifts to a different entity fails here.
    -> judge (no tools, sees the answer)
                             same answer as the template, and no detail that
                             narrows it -- describing a function by its callees
                             in a question about its callees, say.

Ground truth never moves: the answer is still the graph's. The agents only
choose the wording, and a rewrite that fails any gate falls back to the
template.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..eval.agent import REPO_TOOLS, RepoSandbox, run_tool_loop
from .base import QuestionRecord
from .personas import personas_for_family
from .validate import RepoVocabulary, _answer_tokens, _mentions, check_invented_entities

PROMPT_VERSION = "agentic-v2"


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False}}}


SUBMIT_REWRITE = _tool(
    "submit_rewrite", "Submit the rewritten question. Call this exactly once.",
    {"question": {"type": "string"},
     "references": {"type": "array", "description": "How the question refers to each entity you were given.",
                    "items": {"type": "object", "additionalProperties": False,
                              "required": ["entity", "referred_to_as"],
                              "properties": {"entity": {"type": "string"},
                                             "referred_to_as": {"type": "string"}}}}},
    ["question", "references"])

SUBMIT_ENTITIES = _tool(
    "submit_entities", "Submit what each phrase denotes. Call this exactly once.",
    {"entities": {"type": "array", "description": "One entry per phrase, in the order given.",
                  "items": {"type": "object", "additionalProperties": False,
                            "required": ["phrase", "qualified_name"],
                            "properties": {"phrase": {"type": "string"},
                                           "qualified_name": {"type": "string"}}}}},
    ["entities"])

SUBMIT_VERDICT = _tool(
    "submit_verdict", "Submit your verdict. Call this exactly once.",
    {"same_answer": {"type": "boolean"},
     "narrows_answer": {"type": "boolean",
                        "description": "True if the rewrite contains any detail that hints at or narrows the answer."},
     "reason": {"type": "string"}},
    ["same_answer", "narrows_answer", "reason"])

WRITER_SYSTEM = """\
You rewrite a templated question about a Python repository into the question \
a developer on the team would actually ask. You can read the repository with \
`list_dir`, `read_file` and `grep`. Use them to understand what each entity \
is and where it lives, so you can refer to it the way a colleague would. The \
package root is `{package_root}`.

The answer has been withheld from you. Do not work it out, and never hint at it.

How to refer to entities:
- Never write a fully qualified dotted name, such as `pkg.module.Class.method`, \
and never a dotted module name such as `pkg.module`. To mention a file, write its \
path, such as `pkg/module.py`.
- Use its short name, plus just enough context to make it unambiguous in this \
repository: its enclosing class, the file it lives in, or what it is for.
- Never describe an entity by the thing the question asks about. If the question \
asks what a function calls, do not mention what it calls; if it asks where \
something is defined, do not mention its file; if it asks about a class's bases, \
do not mention them. Describe identity, never the answer.
- Mention only the entities listed. Any bare name the question uses, such as a \
method or attribute name, must appear exactly as written.

Keep the meaning exact: the same scope (direct or transitive, declared or \
inherited, in-repository only) and the same kind of answer (yes/no, a count, \
a list, a name, a path). Do not add instructions about answer format.

Voice: {persona}. At most three sentences. Submit with `submit_rewrite`.\
"""

RESOLVER_SYSTEM = """\
You are given a question about a Python repository and a few phrases taken \
from it. Each phrase names one class, function or method in the repository. \
Find which one, using `list_dir`, `read_file` and `grep`. The package root is \
`{package_root}`; a fully qualified name is the module's dotted path relative to \
it, followed by the qualified name inside the module, e.g. `pkg.mod.Class.method`.

You are NOT answering the question. For each phrase, give only the entity that \
phrase itself denotes. If a phrase could denote more than one entity, give the \
one the question's wording points to. Submit with `submit_entities`.\
"""

JUDGE_SYSTEM = """\
You check a rewritten question against the original it came from, for a \
benchmark whose answers are fixed in advance. You are given the original, the \
rewrite, and the correct answer to the original.

same_answer: true only if a competent engineer with full knowledge of the \
repository would give exactly this answer to the rewrite. False if the rewrite \
changes the entity, broadens or narrows the scope (direct vs transitive, \
declared vs inherited, one hop vs many), or changes the kind of answer.

narrows_answer: true if the rewrite contains any detail that hints at the \
answer or rules part of it out -- describing an entity by its callers, callees, \
bases, fields or file when that is what is being asked, or presuming a yes or a no.

Submit with `submit_verdict`.\
"""


@dataclass
class AgenticConfig:
    writer: str
    resolver: str
    judge: str
    attempts: int = 2
    writer_steps: int = 12
    resolver_steps: int = 12
    max_seconds: float = 180


@dataclass
class Outcome:
    rec: QuestionRecord
    accepted: bool
    cost_usd: float = 0.0
    attempts: list[dict] = field(default_factory=list)


def _expected_entities(rec: QuestionRecord, vocab: RepoVocabulary) -> tuple[set[str], set[str]]:
    """(qualified names the resolver must recover, bare names the rewrite must keep)."""
    entities, names = set(), set()
    for key, value in rec.params.items():
        if key == "repo" or not isinstance(value, str):
            continue
        if value in vocab.qnames:
            entities.add(value)
        elif "." in value:
            # An entity that does not exist (t4_absent_entity's "no" side): its
            # namespace is real and must be recovered; its name must be kept.
            head, _, tail = value.rpartition(".")
            if head in vocab.qnames:
                entities.add(head)
            names.add(tail)
        elif value not in vocab.file_paths:
            names.add(value)
    return entities, names


def check_answer_leak_strict(rec: QuestionRecord, rewrite: str, vocab: RepoVocabulary) -> list[str]:
    """No answer token in the rewrite, whatever the template contained.

    The plain naturaliser exempts tokens the template already held, because it
    can only shorten names. Here that exemption is the leak: "the `delete`
    method on `Collector`" passed because the template's qualified name
    contained `Collector` -- which is the answer. Only the question's own
    referents may be named.
    """
    entities, names = _expected_entities(rec, vocab)
    own = names | {e.rsplit(".", 1)[-1] for e in entities} | entities
    return [f"names {token!r}, which is part of the answer"
            for token in _answer_tokens(rec)
            if token and token not in own and _mentions(rewrite, token)]


def check_rewrite(rec: QuestionRecord, rewrite: str, vocab: RepoVocabulary) -> list[str]:
    """Deterministic gates. The qualified-name rule replaces the plain
    naturaliser's referent check: a description is verified by the resolver."""
    problems = []
    for token in set(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b", rewrite)):
        if token in vocab.qnames:
            problems.append(f"uses the fully qualified name {token!r}")
    problems += check_answer_leak_strict(rec, rewrite, vocab)
    problems += check_invented_entities(rewrite, vocab)
    _, names = _expected_entities(rec, vocab)
    for name in names:
        if not re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", rewrite):
            problems.append(f"drops the name {name!r} that the ground-truth query depends on")
    return problems


def normalize_qname(token: str, vocab: RepoVocabulary) -> str:
    """Map a resolver's answer onto a graph qname when it is unambiguous.

    Accepts the qname itself, a unique dotted suffix (`DecisionModel.__init__`),
    and `path/to/file.py:Name` forms. Anything else comes back unchanged and
    simply fails to match.
    """
    text = str(token).strip().strip("`").strip()
    text = text[:-2] if text.endswith("()") else text
    if text in vocab.qnames:
        return text
    if ".py" in text:
        path, _, rest = text.partition(".py")
        text = path.replace("/", ".") + ("." + rest.lstrip(":. ") if rest.strip(":. ") else "")
    hits = [q for q in vocab.qnames if q == text or q.endswith("." + text)]
    if len(hits) == 1:
        return hits[0]
    # A path longer than the package root (`backend/app/x.py`) as a dotted
    # string: drop leading components until something matches uniquely.
    parts = text.split(".")
    for i in range(1, len(parts) - 1):
        hits = [q for q in vocab.qnames if q == ".".join(parts[i:])]
        if len(hits) == 1:
            return hits[0]
    return text


def _entity_list(rec: QuestionRecord, vocab: RepoVocabulary) -> str:
    entities, names = _expected_entities(rec, vocab)
    lines = [f"- {e}" for e in sorted(entities)] + [f"- the name `{n}` (keep exactly)" for n in sorted(names)]
    return "\n".join(lines) or "- (none)"


class AgenticNaturalizer:
    def __init__(self, client, config: AgenticConfig, repo_path: Path, package_root: str,
                 vocab: RepoVocabulary, rng: random.Random):
        self.client, self.config, self.vocab, self.rng = client, config, vocab, rng
        self.sandbox = RepoSandbox(repo_path)
        self.package_root = package_root

    def _write(self, rec: QuestionRecord, persona, feedback: str | None):
        user = (f"Templated question:\n{rec.question}\n\n"
                f"Entities this question is about:\n{_entity_list(rec, self.vocab)}")
        if feedback:
            user += f"\n\nYour previous rewrite was rejected:\n{feedback}\nWrite a new one that fixes this."
        return run_tool_loop(
            self.client, self.config.writer,
            system=WRITER_SYSTEM.format(package_root=self.package_root,
                                        persona=f"{persona.who}. {persona.situation}"),
            user=user, tools=REPO_TOOLS + [SUBMIT_REWRITE], final_tool="submit_rewrite",
            required_key="question", sandbox=self.sandbox,
            max_steps=self.config.writer_steps, max_seconds=self.config.max_seconds)

    def _resolve(self, rewrite: str, phrases: list[str]):
        user = f"Question:\n{rewrite}\n\nPhrases:\n" + "\n".join(f"- {p}" for p in phrases)
        return run_tool_loop(
            self.client, self.config.resolver,
            system=RESOLVER_SYSTEM.format(package_root=self.package_root),
            user=user, tools=REPO_TOOLS + [SUBMIT_ENTITIES], final_tool="submit_entities",
            required_key="entities", sandbox=self.sandbox,
            max_steps=self.config.resolver_steps, max_seconds=self.config.max_seconds)

    def _judge(self, rec: QuestionRecord, rewrite: str):
        user = (f"Original question:\n{rec.question}\n\nRewrite:\n{rewrite}\n\n"
                f"Correct answer to the original:\n{json.dumps(rec.answer)}")
        return run_tool_loop(
            self.client, self.config.judge, system=JUDGE_SYSTEM, user=user,
            tools=[SUBMIT_VERDICT], final_tool="submit_verdict", required_key="same_answer",
            sandbox=None, max_steps=3, max_seconds=self.config.max_seconds)

    def naturalize(self, rec: QuestionRecord) -> Outcome:
        persona = self.rng.choice(personas_for_family(rec.family))
        expected, _ = _expected_entities(rec, self.vocab)
        outcome, feedback = Outcome(rec=rec, accepted=False), None
        for _ in range(self.config.attempts):
            attempt: dict = {"persona": persona.id}
            outcome.attempts.append(attempt)
            wrote = self._write(rec, persona, feedback)
            outcome.cost_usd += wrote.cost_usd
            attempt["writer_tools"] = wrote.tool_calls
            if not wrote.answered:
                attempt["rejected_by"], attempt["problems"] = "writer", [wrote.error]
                continue
            rewrite = str(wrote.submitted["question"]).strip()
            attempt["question"], attempt["references"] = rewrite, wrote.submitted.get("references")

            problems = check_rewrite(rec, rewrite, self.vocab)
            if problems:
                attempt["rejected_by"], attempt["problems"] = "checks", problems
                feedback = "; ".join(problems)
                continue

            refs = [r for r in (wrote.submitted.get("references") or []) if isinstance(r, dict)]
            # Only the phrases for entities go to the resolver. A bare name the
            # query uses (the `device` in "does Agent define `device`") is
            # checked verbatim above; handed over as a phrase, the resolver
            # maps it to the field itself and the rewrite fails for nothing.
            own = [r for r in refs if normalize_qname(r.get("entity", ""), self.vocab) in expected]
            phrases = [str(r.get("referred_to_as", "")).strip() for r in (own or refs)]
            flat = re.sub(r"[`'\"]", "", rewrite).lower()
            phrases = [ph for ph in phrases if ph and re.sub(r"[`'\"]", "", ph).lower() in flat]
            if len(phrases) < len(expected):
                attempt["rejected_by"] = "checks"
                attempt["problems"] = [f"reported {len(phrases)} phrase(s) found verbatim in the "
                                       f"question for {len(expected)} entities"]
                feedback = ("List, in `references`, the exact words the question uses for each "
                            "entity -- copied verbatim from your question.")
                continue

            resolved = self._resolve(rewrite, phrases)
            outcome.cost_usd += resolved.cost_usd
            rows = (resolved.submitted or {}).get("entities") or []
            found = {normalize_qname(r.get("qualified_name", ""), self.vocab) for r in rows if isinstance(r, dict)}
            attempt["resolved"] = sorted(found)
            missing, extra = expected - found, {e for e in found - expected if e in self.vocab.qnames}
            if not resolved.answered or missing or extra:
                attempt["rejected_by"] = "resolver"
                attempt["problems"] = ([resolved.error] if not resolved.answered else []) + \
                    [f"could not recover {m!r}" for m in sorted(missing)] + \
                    [f"read as {e!r} instead" for e in sorted(extra)]
                feedback = ("A reader could not tell which entities you meant: "
                            + "; ".join(attempt["problems"]))
                continue

            verdict = self._judge(rec, rewrite)
            outcome.cost_usd += verdict.cost_usd
            v = verdict.submitted or {}
            attempt["judge"] = v
            if not verdict.answered or not v.get("same_answer") or v.get("narrows_answer"):
                attempt["rejected_by"] = "judge"
                attempt["problems"] = [v.get("reason") or verdict.error or "no verdict"]
                feedback = f"A reviewer rejected it: {attempt['problems'][0]}"
                continue

            rec.question_template = rec.question_template or rec.question
            rec.question = rewrite
            outcome.accepted = True
            break

        rec.question_template = rec.question_template or rec.question
        rec.naturalization = {
            "method": "agentic", "prompt_version": PROMPT_VERSION,
            "status": "accepted" if outcome.accepted else "fell_back_to_template",
            "persona": persona.id if outcome.accepted else None,
            "writer": self.config.writer, "resolver": self.config.resolver, "judge": self.config.judge,
            "attempts": outcome.attempts, "cost_usd": round(outcome.cost_usd, 5),
        }
        return outcome
