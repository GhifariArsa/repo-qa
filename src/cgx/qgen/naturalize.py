"""Rewrite templated questions into natural developer questions with Claude.

**The model rewrites the question. The graph still owns the answer.** The
answer is never sent to the model, never regenerated, and never edited here —
only the surface form changes. That is what keeps the benchmark's contribution
intact: ground truth is still derived by Cypher over the extracted graph, not
produced by an LLM.

Pipeline per question:

    templated question ─▶ Claude (persona rewrite, answer withheld)
                       ─▶ deterministic validators (validate.py)
                       ─▶ Claude judge (same answer? adversarial)
                       ─▶ accepted rewrite, or fall back to the template

A rewrite that fails any gate is discarded and the record keeps its templated
phrasing, with the failure recorded. Losing naturalness is cheap; shipping a
question whose stored answer no longer matches it is not.

Both a synchronous path and a Batches path are provided — at 600 questions the
batch path halves the cost and is the intended one for a full run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .base import QuestionRecord
from .personas import Persona, personas_for_family
from .providers import LLMProvider, get_provider
from .validate import RepoVocabulary, load_vocabulary, validate_rewrite

#: Model spec, "<provider>:<model>". Either provider works; see providers.py
#: for why writing questions with a different family than you evaluate on is
#: worth doing deliberately.
DEFAULT_MODEL = "anthropic:claude-opus-5"
PROMPT_VERSION = "v2-families"


# --------------------------------------------------------------------------
# Structured output shapes
# --------------------------------------------------------------------------


class Phrasing(BaseModel):
    persona: str
    question: str


class Rewrites(BaseModel):
    phrasings: list[Phrasing]


class Judgement(BaseModel):
    same_answer: bool
    reason: str


#: Hand-written schema for the Batches path, where `messages.parse()` is not
#: available. Structured outputs require `additionalProperties: false`.
REWRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "phrasings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"persona": {"type": "string"}, "question": {"type": "string"}},
                "required": ["persona", "question"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["phrasings"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

REWRITE_SYSTEM = """\
You rewrite questions for a code-comprehension benchmark. Each question is about \
the structure of one Python repository, and was generated from a template, so it \
reads like a database query rather than something a person would say. Your job is \
to rewrite it as the question a working engineer would actually ask.

The answer has deliberately been withheld from you. You do not need it, and you \
must not guess at it.

Rules, in order of importance:

1. Do not change what is being asked. The rewrite must have exactly the same \
answer as the original, against this repository. If you cannot preserve the \
meaning in a given persona's voice, return the plain phrasing for that persona \
instead of forcing it.

2. Keep every named entity. Fully qualified names may be shortened to their last \
segment ONLY when the shortened name is marked unambiguous below. Never rename, \
abbreviate, or paraphrase an identifier.

3. Never name a class, function, module, or file that was not given to you. Do \
not invent plausible-sounding neighbours.

4. Do not state or imply the answer, and do not narrow it. "Does anything call \
this, or is it dead code?" is fine. "Is this dead code?" is not — it presumes.

5. Sound like a person, not like documentation. Contractions are fine. A short \
sentence of situation before the question is fine. Do not add pleasantries, do \
not explain the codebase back to the reader, and do not exceed three sentences.

Return one rewrite per requested persona."""

JUDGE_SYSTEM = """\
You check whether two questions about the same Python repository have the same \
answer. You are looking for reasons they DIFFER — assume they differ until the \
evidence says otherwise.

Answer false if the rewrite: asks about a different entity; broadens or narrows \
the scope (direct callers vs all transitive callers, defined vs defined-or- \
inherited, one hop vs any number); changes a count into a list or vice versa; \
presumes an answer; or is ambiguous about which entity it means when the original \
was not.

Answer true only if any competent engineer with full knowledge of the repository \
would give the identical answer to both."""


@dataclass
class NaturalizationConfig:
    model: str = DEFAULT_MODEL     # "<provider>:<model>"
    max_tokens: int = 4096
    judge: bool = True
    per_question: int = 1          # how many personas to request per question
    seed_note: str | None = None


def _entity_context(rec: QuestionRecord, vocab: RepoVocabulary) -> str:
    """The entities the rewrite may name, with disambiguation guidance."""
    lines = []
    seen: set[str] = set()
    for key, value in rec.params.items():
        if key == "repo" or not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        if value in vocab.qnames:
            short = value.rsplit(".", 1)[-1]
            tag = "unambiguous" if vocab.is_unambiguous(short) else "AMBIGUOUS — keep it fully qualified"
            lines.append(f"- {value}  (short name `{short}`: {tag})")
        else:
            lines.append(f"- {value}")
    return "\n".join(lines) if lines else "- (none)"


FAMILY_INTENT = {
    "definition_organisation": "where entities are defined and how code is organised",
    "dependency_interaction": "relationships between callers, callees and paths in the static call graph",
    "inheritance_specialisation": "ancestry and explicit member declarations, without assuming runtime dispatch",
    "state_access": "which code reads or writes fields or globals, including through static call paths",
    "structural_comparison": "the overlap or difference between two explicitly defined sets of relationships",
}


def build_rewrite_prompt(rec: QuestionRecord, vocab: RepoVocabulary, personas: tuple[Persona, ...]) -> str:
    persona_block = "\n".join(
        f"- {p.id}: {p.who}. {p.situation}" for p in personas
    )
    return f"""\
Repository: {rec.repo_id}

What this question is fundamentally asking: {FAMILY_INTENT[rec.family]}

Preserve all traversal limits, direct/transitive distinctions, and static scope.

Templated question:
{rec.question}

Entities this question is about (these, and only these, may be named):
{_entity_context(rec, vocab)}

Write one rewrite for each of these personas:
{persona_block}"""


def build_judge_prompt(rec: QuestionRecord, rewrite: str) -> str:
    return f"""\
Repository: {rec.repo_id}

Original question:
{rec.question}

Rewritten question:
{rewrite}

Do these two questions have the same answer against this repository?"""


# --------------------------------------------------------------------------
# Synchronous path
# --------------------------------------------------------------------------


class Naturalizer:
    """Rewrites questions one at a time. Use the batch path for full runs."""

    def __init__(self, config: NaturalizationConfig | None = None,
                 provider: LLMProvider | None = None, client=None):
        self.config = config or NaturalizationConfig()
        self.provider = provider or get_provider(self.config.model, client=client)

    @property
    def client(self):
        """Underlying SDK client — the Batches path is Anthropic-only."""
        return self.provider.client

    def rewrite(self, rec: QuestionRecord, vocab: RepoVocabulary) -> list[Phrasing]:
        personas = personas_for_family(rec.family)[: self.config.per_question]
        parsed = self.provider.parse(
            system=REWRITE_SYSTEM,
            user=build_rewrite_prompt(rec, vocab, personas),
            schema=Rewrites,
            max_tokens=self.config.max_tokens,
        )
        return list(parsed.phrasings) if parsed else []

    def judge(self, rec: QuestionRecord, rewrite: str) -> Judgement:
        # Same budget as the rewrite: on a thinking model the reasoning tokens
        # count against max_tokens, and 1024 truncated the verdict mid-JSON.
        # A judge that cannot answer rejects the rewrite -- one question falls
        # back to its template instead of the whole run crashing.
        try:
            parsed = self.provider.parse(
                system=JUDGE_SYSTEM,
                user=build_judge_prompt(rec, rewrite),
                schema=Judgement,
                max_tokens=self.config.max_tokens,
            )
        except Exception as exc:                      # truncation, network, refusal
            return Judgement(same_answer=False, reason=f"judge failed: {type(exc).__name__}")
        return parsed or Judgement(same_answer=False, reason="no parsed output / refusal")

    def naturalize(self, rec: QuestionRecord, vocab: RepoVocabulary) -> QuestionRecord:
        """Return `rec` with a natural phrasing, or unchanged if none survived."""
        attempts: list[dict] = []
        try:
            phrasings = self.rewrite(rec, vocab)
        except Exception as exc:                      # network, parse, refusal
            return _record_failure(rec, self.config, [{"error": str(exc)}])

        for phrasing in phrasings:
            problems = validate_rewrite(rec, phrasing.question, vocab)
            if problems:
                attempts.append({"persona": phrasing.persona, "question": phrasing.question,
                                 "rejected_by": "validators", "problems": problems})
                continue
            if self.config.judge:
                verdict = self.judge(rec, phrasing.question)
                if not verdict.same_answer:
                    attempts.append({"persona": phrasing.persona, "question": phrasing.question,
                                     "rejected_by": "judge", "problems": [verdict.reason]})
                    continue
            return _record_success(rec, self.config, phrasing, attempts)

        return _record_failure(rec, self.config, attempts)


def _base_meta(config: NaturalizationConfig) -> dict:
    return {"model": config.model, "prompt_version": PROMPT_VERSION, "judged": config.judge}


def _record_success(rec, config, phrasing: Phrasing, attempts: list[dict]) -> QuestionRecord:
    rec.question_template = rec.question_template or rec.question
    rec.question = phrasing.question
    rec.naturalization = {
        **_base_meta(config), "status": "accepted", "persona": phrasing.persona,
        "rejected_attempts": attempts,
    }
    return rec


def _record_failure(rec, config, attempts: list[dict]) -> QuestionRecord:
    rec.question_template = rec.question_template or rec.question
    rec.naturalization = {
        **_base_meta(config), "status": "fell_back_to_template", "persona": None,
        "rejected_attempts": attempts,
    }
    return rec


# --------------------------------------------------------------------------
# Batch path — 50% cheaper, the intended path for a full 10-repo run
# --------------------------------------------------------------------------


def build_batch_requests(records: list[QuestionRecord], vocab: RepoVocabulary,
                         config: NaturalizationConfig) -> list[dict]:
    """One Batches request per question. Results are keyed by `custom_id`."""
    requests = []
    for rec in records:
        personas = personas_for_family(rec.family)[: config.per_question]
        requests.append({
            "custom_id": rec.id.replace("::", "__")[:64],
            "params": {
                "model": config.model,
                "max_tokens": config.max_tokens,
                "system": [{"type": "text", "text": REWRITE_SYSTEM,
                            "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user",
                              "content": build_rewrite_prompt(rec, vocab, personas)}],
                "output_config": {"format": {"type": "json_schema", "schema": REWRITE_SCHEMA}},
            },
        })
    return requests


def parse_batch_result(text: str) -> list[Phrasing]:
    data = json.loads(text)
    return [Phrasing(**p) for p in data.get("phrasings", [])]


def submit_batch(client, records: list[QuestionRecord], vocab: RepoVocabulary,
                 config: NaturalizationConfig) -> str:
    """Submit all rewrites as one batch. Returns the batch id.

    Anthropic only — this uses the Messages Batches API shape. OpenAI's batch
    API takes an uploaded JSONL file instead and is not wired up; use the
    synchronous path for OpenAI.
    """
    if not config.model.startswith(("anthropic:", "claude")):
        raise SystemExit(
            f"--batch is Anthropic-only; {config.model!r} must use the synchronous path"
        )
    from .naturalize import build_batch_requests as _  # noqa: F401  (keeps import local)
    batch = client.messages.batches.create(
        requests=build_batch_requests(records, vocab, config)
    )
    return batch.id


def collect_batch(client, batch_id: str, records: list[QuestionRecord],
                  vocab: RepoVocabulary, config: NaturalizationConfig,
                  naturalizer: "Naturalizer | None" = None) -> list[QuestionRecord]:
    """Apply a finished batch's results, validating each rewrite as usual.

    Results arrive in arbitrary order, so they are keyed by `custom_id` rather
    than by position. The judge pass still runs synchronously per surviving
    candidate -- it is a small fraction of the calls, and only on rewrites that
    already cleared the deterministic validators.
    """
    by_custom_id = {rec.id.replace("::", "__")[:64]: rec for rec in records}

    for result in client.messages.batches.results(batch_id):
        rec = by_custom_id.get(result.custom_id)
        if rec is None:
            continue
        if result.result.type != "succeeded":
            _record_failure(rec, config, [{"error": f"batch result: {result.result.type}"}])
            continue
        text = next(
            (b.text for b in result.result.message.content if b.type == "text"), ""
        )
        try:
            phrasings = parse_batch_result(text)
        except Exception as exc:                      # malformed despite the schema
            _record_failure(rec, config, [{"error": f"unparseable batch output: {exc}"}])
            continue

        attempts: list[dict] = []
        accepted = False
        for phrasing in phrasings:
            problems = validate_rewrite(rec, phrasing.question, vocab)
            if problems:
                attempts.append({"persona": phrasing.persona, "question": phrasing.question,
                                 "rejected_by": "validators", "problems": problems})
                continue
            if config.judge and naturalizer is not None:
                verdict = naturalizer.judge(rec, phrasing.question)
                if not verdict.same_answer:
                    attempts.append({"persona": phrasing.persona, "question": phrasing.question,
                                     "rejected_by": "judge", "problems": [verdict.reason]})
                    continue
            _record_success(rec, config, phrasing, attempts)
            accepted = True
            break
        if not accepted:
            _record_failure(rec, config, attempts)

    return records


__all__ = [
    "Naturalizer", "NaturalizationConfig", "Phrasing", "Rewrites", "Judgement",
    "build_batch_requests", "parse_batch_result", "load_vocabulary",
    "submit_batch", "collect_batch", "get_provider",
    "REWRITE_SYSTEM", "JUDGE_SYSTEM", "DEFAULT_MODEL",
]
