"""Question records and the generator registry.

Every question carries the Cypher that derived its ground truth. That is the
whole point of the benchmark: the answer is not mined from an issue tracker or
produced by an agent exploring the repo, it is read off the graph, so it is
reproducible and auditable. Keep `cypher` + `params` populated -- a question
without them cannot be re-verified when the repo or extractor changes.

Tier labels are carried per record and must not be flattened away: reporting is
per-tier accuracy, not one aggregate number.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

TIER_NAMES = {
    1: "existence_lookup",
    2: "path_traversal",
    3: "aggregation",
    4: "counterfactual_negative",
    5: "cross_graph_composition",
}


@dataclass
class QuestionRecord:
    id: str
    tier: int
    tier_name: str
    template_id: str
    repo_id: str
    question: str                   # the phrasing an evaluated model sees
    answer: Any                     # canonical, machine-comparable
    answer_type: str                # bool | count | string | list | path
    answer_text: str                # rendered form, for display and scoring
    cypher: str                     # the query that produced `answer`
    params: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(
        default_factory=lambda: {"status": "pending", "note": None, "reviewer": None, "at": None}
    )
    #: The generated template phrasing, retained when an LLM rewrite replaces
    #: `question`. Keeping both makes every rewrite auditable and reversible,
    #: and lets the harness score templated vs natural phrasings separately.
    question_template: str | None = None
    #: Provenance of the rewrite: model, prompt version, persona, and every
    #: candidate a validator or the judge rejected.
    naturalization: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_question(
    *,
    tier: int,
    template_id: str,
    repo_id: str,
    index: int,
    question: str,
    answer: Any,
    answer_type: str,
    cypher: str,
    params: dict | None = None,
    evidence: dict | None = None,
    answer_text: str | None = None,
) -> QuestionRecord:
    return QuestionRecord(
        id=f"{repo_id}::t{tier}::{template_id}::{index:04d}",
        tier=tier,
        tier_name=TIER_NAMES[tier],
        template_id=template_id,
        repo_id=repo_id,
        question=question,
        answer=answer,
        answer_type=answer_type,
        answer_text=answer_text if answer_text is not None else render_answer(answer, answer_type),
        cypher=cypher.strip(),
        params=params or {},
        evidence=evidence or {},
    )


def render_answer(answer: Any, answer_type: str) -> str:
    if answer_type == "bool":
        return "Yes" if answer else "No"
    if answer_type == "list":
        return ", ".join(map(str, answer)) if answer else "none"
    if answer_type == "path":
        return " -> ".join(map(str, answer)) if answer else "no path"
    if answer is None:
        return "none"
    return str(answer)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

#: template_id -> (tier, generator callable)
Generator = Callable[..., list[QuestionRecord]]
_REGISTRY: dict[str, tuple[int, Generator]] = {}


def template(template_id: str, tier: int) -> Callable[[Generator], Generator]:
    def wrap(fn: Generator) -> Generator:
        _REGISTRY[template_id] = (tier, fn)
        return fn
    return wrap


def registry() -> dict[str, tuple[int, Generator]]:
    return dict(_REGISTRY)


def templates_for_tier(tier: int) -> dict[str, Generator]:
    return {tid: fn for tid, (t, fn) in _REGISTRY.items() if t == tier}


# --------------------------------------------------------------------------
# Dataset IO -- JSONL, one record per line, so Phase 3 can rewrite in place.
# --------------------------------------------------------------------------


def write_questions(records: list[QuestionRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict()) + "\n")
    return path


def read_questions(path: Path) -> list[QuestionRecord]:
    records = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(QuestionRecord(**json.loads(line)))
    return records
