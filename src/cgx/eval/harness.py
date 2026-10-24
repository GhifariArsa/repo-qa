"""Phase 5 -- minimal evaluation harness (family and operation scoring).

Deliberately thin. It exists so the pipeline can be validated on real output
rather than only inspected by hand. Scores are grouped by subject family,
operation, answer outcome, and template.

Two models ship:
  * `stub`     -- answers "Yes" to every yes/no question and gives an empty or
                  zero answer otherwise. It should score zero on negative
                  boolean questions, independently of their subject family.
  * `oracle`   -- replays the stored ground truth. Should score 1.0 everywhere;
                  a failure means the scorer, not the model, is broken.

Wiring a real LLM or coding agent means implementing one function with the
signature of `stub_model` and registering it in MODELS.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..qgen.taxonomy import HARD_TEMPLATES
from ..qgen.base import QuestionRecord, read_questions


def stub_model(rec: QuestionRecord):
    if rec.answer_type == "bool":
        return True
    if rec.answer_type == "count":
        return 0
    if rec.answer_type in ("list", "path"):
        return []
    return ""


def oracle_model(rec: QuestionRecord):
    return rec.answer


MODELS = {"stub": stub_model, "oracle": oracle_model}


_EMPTY_WORDS = {"none", "n/a", "na", "no", "nothing", "empty", "null", "[]"}


def _atom(value) -> str:
    """One name, path or label, compared on content rather than formatting:
    `./laya/x.py` is `laya/x.py`, `global variable` is `GLOBAL_VARIABLE`, and a
    name wrapped in backticks is the name."""
    text = str(value).strip().strip("`'\"").strip()
    if text.startswith("./"):
        text = text[2:]
    return re.sub(r"[\s-]+", "_", text).casefold()


def _norm(value, answer_type: str):
    if answer_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in {"yes", "true", "y"}
        return bool(value)
    if answer_type == "count":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, str) and answer_type in ("list", "path"):
        value = [value] if value.strip() else []
    # The questions themselves say "or say none", so `["none"]` is a model
    # following the question, not a wrong answer: it means the empty list.
    if answer_type in ("list", "path") and value and all(_atom(v) in _EMPTY_WORDS for v in value):
        value = []
    if answer_type == "list":
        return sorted(_atom(v) for v in (value or []))
    if answer_type == "path":
        return [_atom(v) for v in (value or [])]
    return _atom(value or "")


def score(rec: QuestionRecord, prediction, calls: set[tuple[str, str]] | None = None) -> float:
    """Exact match, except lists score by F1 so partial recall is visible.

    A shortest call chain is often not unique. With the repository's CALLS
    edges in hand, any chain with the right endpoints, the gold length and a
    real edge at every hop is correct, not just the one the graph returned.
    """
    gold = _norm(rec.answer, rec.answer_type)
    pred = _norm(prediction, rec.answer_type)
    if rec.answer_type == "path" and calls is not None and pred and gold and pred != gold:
        valid = (len(pred) == len(gold) and pred[0] == gold[0] and pred[-1] == gold[-1]
                 and all((a, b) in calls for a, b in zip(pred, pred[1:])))
        return 1.0 if valid else 0.0
    if rec.answer_type == "list":
        gset, pset = set(gold), set(pred)
        if not gset and not pset:
            return 1.0
        if not gset or not pset:
            return 0.0
        tp = len(gset & pset)
        if tp == 0:
            return 0.0
        precision, recall = tp / len(pset), tp / len(gset)
        return 2 * precision * recall / (precision + recall)
    return 1.0 if gold == pred else 0.0


def run_evaluation(questions_path: Path, out_path: Path, model: str, limit: int | None = None):
    if model not in MODELS:
        raise SystemExit(f"unknown model {model!r}; available: {', '.join(MODELS)}")
    fn = MODELS[model]
    records = read_questions(questions_path)
    if limit:
        records = records[:limit]

    per_family: dict[str, list[float]] = defaultdict(list)
    per_operation: dict[str, list[float]] = defaultdict(list)
    per_outcome: dict[str, list[float]] = defaultdict(list)
    per_template: dict[str, list[float]] = defaultdict(list)
    results = []
    for rec in records:
        prediction = fn(rec)
        s = score(rec, prediction)
        per_family[rec.family].append(s)
        for operation in rec.operations:
            per_operation[operation].append(s)
        per_outcome[rec.outcome].append(s)
        per_template[rec.template_id].append(s)
        results.append({"id": rec.id, "family": rec.family, "operations": rec.operations,
                        "outcome": rec.outcome, "score": s, "prediction": prediction})

    summary = {
        "model": model,
        "n": len(records),
        "overall": round(sum(r["score"] for r in results) / len(results), 4) if results else 0.0,
        "by_family": {f: round(sum(v) / len(v), 4) for f, v in sorted(per_family.items())},
        "by_operation": {op: round(sum(v) / len(v), 4) for op, v in sorted(per_operation.items())},
        "by_outcome": {o: round(sum(v) / len(v), 4) for o, v in sorted(per_outcome.items())},
        "family_counts": {f: len(v) for f, v in sorted(per_family.items())},
        "by_template": {t: round(sum(v) / len(v), 4) for t, v in sorted(per_template.items())},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    return summary


# --------------------------------------------------------------------------
# Real models: the read-only agent in `agent.py`, over OpenRouter.
# --------------------------------------------------------------------------

CONDITIONS = ("repo", "closed")
PHRASINGS = ("template", "natural")


# --------------------------------------------------------------------------
# Scoring v2: the same rules, with five answer-form ambiguities resolved.
# `score` above is what every row was scored with during the run and is kept
# unchanged; v2 is applied afterwards to the stored predictions so both can be
# reported. Each rule fixes a case where a model was right and marked wrong:
#
#   1. annotations  `recipes/gepa/method.py (GEPAPlugin)` is the path.
#   2. src layouts  `packages.artemis-client.src.artemis_client.errors.X` is
#                   importable as `artemis_client.errors.X`; either is accepted.
#   3. overriders   the question asks which classes override method M, and
#                   `pkg.C.M` names class `pkg.C` and its override.
#   4. implicit     calling a class runs `__init__`; when the class defines no
#      __init__     `__init__` in the repository, the graph has no node for it,
#                   so naming `C.__init__` is the convention, not a fabrication.
#   5. none         unchanged from v1 (`["none"]` is the empty list).
# --------------------------------------------------------------------------

_ANNOTATION = re.compile(r"\s*\([^()]*\)\s*$")


def _atom_v2(value) -> str:
    text = _ANNOTATION.sub("", str(value).strip().strip("`'\"").strip())
    atom = _atom(text)
    return atom.rsplit(".src.", 1)[-1] if ".src." in atom else atom


@dataclass
class GraphFacts:
    """What v2 needs to know about the repository beyond the question."""
    calls: set[tuple[str, str]]
    classes_without_init: set[str]


def load_graph_facts(graph_path: Path) -> GraphFacts:
    graph = json.loads(Path(graph_path).read_text())
    qnames = {n["qname"] for n in graph["nodes"]}
    classes = {n["qname"] for n in graph["nodes"] if n["label"] == "CLASS"}
    return GraphFacts(calls={(_atom_v2(a), _atom_v2(b)) for a, b in _calls_by_name(graph)},
                      classes_without_init={_atom_v2(c) for c in classes if f"{c}.__init__" not in qnames})


def _calls_by_name(graph: dict) -> list[tuple[str, str]]:
    qname = {n["id"]: n["qname"] for n in graph["nodes"]}
    return [(qname[e["source"]], qname[e["target"]]) for e in graph["edges"] if e["type"] == "CALLS"]


def score_v2(rec: QuestionRecord, prediction, facts: GraphFacts | None = None) -> float:
    gold = _norm_v2(rec.answer, rec, None)
    pred = _norm_v2(prediction, rec, facts)
    if rec.answer_type == "path" and facts is not None and pred and gold and pred != gold:
        valid = (len(pred) == len(gold) and pred[0] == gold[0] and pred[-1] == gold[-1]
                 and all((a, b) in facts.calls for a, b in zip(pred, pred[1:])))
        return 1.0 if valid else 0.0
    return _compare(gold, pred, rec.answer_type)


def _norm_v2(value, rec: QuestionRecord, facts: GraphFacts | None):
    kind = rec.answer_type
    if kind in ("bool", "count"):
        return _norm(value, kind)
    if isinstance(value, str) and kind in ("list", "path"):
        value = [value] if value.strip() else []
    if kind in ("list", "path"):
        items = [_atom_v2(v) for v in (value or [])]
        if items and all(i in _EMPTY_WORDS for i in items):
            items = []
        method = rec.params.get("method") if rec.template_id == "inheritance_overriders" else None
        if method:
            suffix = "." + _atom(method)
            items = [i[: -len(suffix)] if i.endswith(suffix) else i for i in items]
        if facts is not None:
            items = [i for i in items
                     if not (i.endswith(".__init__") and i[: -len(".__init__")] in facts.classes_without_init)]
        return sorted(set(items)) if kind == "list" else items
    return _atom_v2(value or "")


def _compare(gold, pred, answer_type: str) -> float:
    if answer_type == "list":
        gset, pset = set(gold), set(pred)
        if not gset and not pset:
            return 1.0
        if not gset or not pset:
            return 0.0
        tp = len(gset & pset)
        if tp == 0:
            return 0.0
        precision, recall = tp / len(pset), tp / len(gset)
        return 2 * precision * recall / (precision + recall)
    return 1.0 if gold == pred else 0.0


def load_calls(graph_path: Path) -> set[tuple[str, str]]:
    """The repository's CALLS edges as normalised (caller, callee) names."""
    graph = json.loads(Path(graph_path).read_text())
    qname = {n["id"]: n["qname"] for n in graph["nodes"]}
    return {(_atom(qname[e["source"]]), _atom(qname[e["target"]]))
            for e in graph["edges"] if e["type"] == "CALLS"}


def select_questions(records: list[QuestionRecord], per_family: int | None, seed: int,
                     sampling: str = "round_robin") -> list[QuestionRecord]:
    """Every question, or a seeded sample of `per_family` from each family.

    `round_robin` spreads a family's sample evenly over its templates, which
    suits checking every template works. `proportional` samples in proportion
    to the dataset itself, which suits measuring the benchmark as it is: once
    hard templates hold most of a family, round-robin over-represents lookups.
    """
    if not per_family:
        return records
    import random

    if sampling == "proportional":
        by_family_rows: dict[str, list[QuestionRecord]] = defaultdict(list)
        for rec in records:
            by_family_rows[rec.family].append(rec)
        chosen = []
        for family in sorted(by_family_rows):
            pool = sorted(by_family_rows[family], key=lambda r: r.id)
            repo = pool[0].repo_id
            chosen += random.Random(f"select:{seed}:{repo}:{family}").sample(pool, min(per_family, len(pool)))
        return chosen
    if sampling != "round_robin":
        raise ValueError(f"sampling is round_robin or proportional, not {sampling!r}")

    # Round-robin over templates, in a per-repo seeded order. Shuffling each
    # family's pool with one shared seed picked the same positions in every
    # repo, so the same few templates filled every sample.
    by_family: dict[str, dict[str, list[QuestionRecord]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        by_family[rec.family][rec.template_id].append(rec)
    chosen = []
    for family in sorted(by_family):
        templates = by_family[family]
        repo = next(iter(templates.values()))[0].repo_id
        rng = random.Random(f"select:{seed}:{repo}:{family}")
        order = sorted(templates)
        rng.shuffle(order)
        pools = {t: rng.sample(sorted(templates[t], key=lambda r: r.id), len(templates[t])) for t in order}
        picked: list[QuestionRecord] = []
        while len(picked) < per_family and any(pools.values()):
            for t in order:
                if pools[t] and len(picked) < per_family:
                    picked.append(pools[t].pop())
        chosen += picked
    return chosen


#: One answer-format line per template, appended to the question in EVERY arm.
#: Rewrites drop format instructions (the agentic writer is told to), and in
#: the first phrasing pilot about half the accuracy drop was a model answering
#: correctly in prose where the template had asked for a bare name. The same
#: line in both arms leaves wording as the only difference between them.
ANSWER_FORMAT = {
    "t1_defining_file": "Give the file path relative to the repository root.",
    "t1_entity_type": "Answer with one of: module, class, function, method, field, global variable.",
    "t1_method_owner": "Give the fully qualified class name.",
    "t3_common_ancestor": "Give the fully qualified class name.",
    "comparison_shared_members": "Give short method names, or say none.",
    "t2_files_touched": "Give file paths relative to the repository root.",
    "organisation_subclass_files": "Give file paths relative to the repository root.",
}
_FORMAT_BY_TYPE = {
    "bool": "Answer yes or no.",
    "count": "Answer with a single integer.",
    "list": "Give fully qualified names, or say none.",
    "path": "List every function or method on the chain in order, as fully qualified names, first to last.",
    "string": "Give the fully qualified name.",
}


def answer_format(rec: QuestionRecord) -> str:
    return ANSWER_FORMAT.get(rec.template_id) or _FORMAT_BY_TYPE[rec.answer_type]


def question_text(rec: QuestionRecord, phrasing: str) -> str:
    return f"{_question_body(rec, phrasing)}\n\n{answer_format(rec)}"


def _question_body(rec: QuestionRecord, phrasing: str) -> str:
    if phrasing == "template":
        return rec.question_template or rec.question
    # A question the naturaliser could not rephrase keeps its template wording
    # in the natural set too -- that is what the rephrased dataset contains.
    # Rows record `rephrased`, so phrasing is compared on accepted rewrites only.
    if not (rec.naturalization or {}).get("status"):
        raise SystemExit(
            f"{rec.id} has not been through naturalization; run `cgx naturalize` first, or use --phrasing template"
        )
    return rec.question


def _error_kind(error: str) -> str:
    if "prompt injection" in error:
        return "blocked_by_gateway_guardrail"
    return error.split(":", 1)[0]


def is_final(row: dict) -> bool:
    """Whether a result row is the model's outcome rather than an outage.

    A model that runs out of steps without answering has failed the question:
    that row is final, and re-running it on resume would hand an unreliable
    model extra attempts (nemotron3 got 83 of them before this rule) and let
    the latest attempt overwrite the failure. Only an infrastructure error --
    a dropped connection, a gateway refusal -- is retried.
    """
    return not row["error"] or row["error"] == "no answer submitted"


def summarise(results: list[dict]) -> dict:
    """Accuracy is over questions that got a model answer or ran out of steps.

    A request refused by infrastructure -- a gateway guardrail, a provider
    outage -- says nothing about the model, so it is counted under `errors`
    and kept out of every accuracy figure rather than scored as a wrong answer.
    """
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    errored = [r for r in results if r["error"] and r["error"] != "no answer submitted"]
    results = [r for r in results if r not in errored]
    for r in results:
        groups["by_family"][r["family"]].append(r["score"])
        groups["by_outcome"][r["outcome"]].append(r["score"])
        groups["by_template"][r["template_id"]].append(r["score"])
        for op in r["operations"]:
            groups["by_operation"][op].append(r["score"])
    mean = lambda v: round(sum(v) / len(v), 4) if v else 0.0  # noqa: E731
    return {
        "n": len(results),
        "overall": mean([r["score"] for r in results]),
        "answered": sum(r["answered"] for r in results),
        "errors": len(errored),
        "error_kinds": dict(sorted(Counter(_error_kind(r["error"]) for r in errored).items())),
        "cost_usd": round(sum(r["cost_usd"] for r in results), 4),
        "mean_tool_calls": mean([r["tool_calls"] for r in results]),
        "forced_answers": sum(r.get("forced", False) for r in results),
        "text_submissions": sum(r.get("text_submission", False) for r in results),
        "mean_seconds": mean([r["seconds"] for r in results]),
        "mean_steps": mean([r["steps"] for r in results]),
        "mean_prompt_tokens": mean([r["prompt_tokens"] for r in results]),
        "mean_completion_tokens": mean([r["completion_tokens"] for r in results]),
        "max_peak_prompt_tokens": max((r.get("peak_prompt_tokens", 0) for r in results), default=0),
        "invalid_tool_calls": sum(r.get("invalid_tool_calls", 0) for r in results),
        "hard": mean([r["score"] for r in results if r.get("hard")]),
        "lookup": mean([r["score"] for r in results if r.get("hard") is False]),
        **{g: {k: mean(v) for k, v in sorted(d.items())} for g, d in sorted(groups.items())},
        "family_counts": {k: len(v) for k, v in sorted(groups["by_family"].items())},
    }


def run_agent_evaluation(*, repo_id: str, repo_path: Path, package_root: Path, questions_path: Path,
                         graph_path: Path, model: str, condition: str, phrasing: str,
                         per_family: int | None, limit: int | None, seed: int, concurrency: int,
                         max_cost: float | None, out_dir: Path, client=None) -> dict:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from ..qgen.providers import client_for, parse_model_spec
    from .agent import MAX_SECONDS, MAX_SECONDS_LOCAL, RepoSandbox, run_agent

    provider, model_id = parse_model_spec(model)
    if provider not in ("openrouter", "ollama"):
        raise SystemExit("agent evaluation needs --model openrouter:<vendor>/<model> or ollama:<model>")
    if condition not in CONDITIONS or phrasing not in PHRASINGS:
        raise SystemExit(f"--condition is one of {CONDITIONS}; --phrasing one of {PHRASINGS}")

    by_id = {r.id: r for r in read_questions(questions_path)}
    chosen = select_questions(list(by_id.values()), per_family, seed)[:limit]
    questions = {rec.id: question_text(rec, phrasing) for rec in chosen}
    calls = load_calls(graph_path)

    slug = re.sub(r"[^A-Za-z0-9.-]+", "_", model_id)
    out_path = Path(out_dir) / repo_id / f"{slug}__{condition}__{phrasing}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():   # resume: a finished question is never paid for twice
        for line in out_path.read_text().splitlines():
            row = json.loads(line)
            if row["id"] not in done and is_final(row):
                done[row["id"]] = row
    todo = [rec for rec in chosen if rec.id not in done]

    client = client or client_for(provider)
    sandbox = RepoSandbox(repo_path) if condition == "repo" else None
    root_rel = str(Path(package_root).resolve().relative_to(Path(repo_path).resolve())) \
        if Path(package_root).resolve().is_relative_to(Path(repo_path).resolve()) else "."
    lock, spent = threading.Lock(), [sum(r["cost_usd"] for r in done.values())]
    print(f"{repo_id}: {len(chosen)} question(s), {len(done)} already done, {len(todo)} to run "
          f"| {model_id} | {condition} | {phrasing}")

    def work(rec: QuestionRecord):
        if max_cost is not None and spent[0] >= max_cost:
            return None
        res = run_agent(client, model_id, rec, condition=condition, question=questions[rec.id],
                        sandbox=sandbox, package_root=root_rel,
                        max_seconds=MAX_SECONDS_LOCAL if provider == "ollama" else MAX_SECONDS)
        row = {"id": rec.id, "family": rec.family, "operations": rec.operations,
               "outcome": rec.outcome, "template_id": rec.template_id,
               "question": questions[rec.id], "gold": rec.answer, "prediction": res.prediction,
               "rephrased": (rec.naturalization or {}).get("status") == "accepted",
               "score": score(rec, res.prediction, calls) if res.answered else 0.0,
               "answered": res.answered, "error": res.error, "steps": res.steps,
               "tool_calls": res.tool_calls, "tools_used": res.tools_used,
               "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
               "cost_usd": res.cost_usd, "seconds": res.seconds, "forced": res.forced,
               "text_submission": res.text_submission, "model": model_id, "condition": condition,
               "hard": rec.template_id in HARD_TEMPLATES,
               "peak_prompt_tokens": res.peak_prompt_tokens, "reasoning_tokens": res.reasoning_tokens,
               "invalid_tool_calls": res.invalid_tool_calls, "no_tool_turns": res.no_tool_turns,
               "files_read": len(res.files_read), "model_seconds": round(res.model_seconds, 2)}
        with lock:
            spent[0] += res.cost_usd
            with out_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        return row

    finished = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for fut in as_completed([pool.submit(work, rec) for rec in todo]):
            row = fut.result()
            if row is None:
                continue
            finished += 1
            if row["error"]:
                status = f"ERROR {row['error'][:80]}"
            else:
                status = f"score {row['score']:.2f}"
            print(f"  [{finished}/{len(todo)}] {row['id']}  {status}  "
                  f"({row['tool_calls']} tools, {row['seconds']}s, ${row['cost_usd']:.4f}, total ${spent[0]:.3f})")
    if max_cost is not None and spent[0] >= max_cost:
        print(f"stopped: spend ${spent[0]:.3f} reached --max-cost ${max_cost:.2f}")

    rows = {}
    for line in out_path.read_text().splitlines():
        row = json.loads(line)
        # The first finished attempt counts; only an infrastructure failure is
        # replaced by a later attempt (see `is_final`).
        if row["id"] in questions and (row["id"] not in rows or not is_final(rows[row["id"]])):
            rows[row["id"]] = row
    summary = {"repo": repo_id, "model": model_id, "condition": condition, "phrasing": phrasing,
               **summarise(list(rows.values()))}
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("n", "overall", "answered", "errors", "cost_usd",
                                               "mean_tool_calls", "by_family")}, indent=2))
    return summary
