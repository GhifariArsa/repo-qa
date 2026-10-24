"""Question records and the generator registry.

Every question carries the Cypher that derived its ground truth. That is the
whole point of the benchmark: the answer is not mined from an issue tracker or
produced by an agent exploring the repo, it is read off the graph, so it is
reproducible and auditable. Keep `cypher` + `params` populated -- a question
without them cannot be re-verified when the repo or extractor changes.

Subject families and operations are independent; neither implies difficulty.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .taxonomy import TAXONOMY_VERSION, answer_outcome, template_spec

@dataclass
class QuestionRecord:
    id: str
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
    family: str = ""
    operations: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    outcome: str = ""
    taxonomy_version: int = TAXONOMY_VERSION
    legacy_taxonomy: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        spec = template_spec(self.template_id)
        self.family = spec.family
        self.operations = list(spec.operations)
        self.relations = list(spec.relations)
        self.outcome = answer_outcome(self.answer, self.answer_type)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_question(
    *,
    tier: int | None = None,
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
    # Existing templates retain their IDs and seeds. The optional tier argument
    # supports old callers only; family assignment always comes from the spec.
    prefix = template_id.split("_", 1)[0]
    legacy = int(prefix[1:]) if prefix in {"t1", "t2", "t3", "t4", "t5"} else None
    if tier is not None and tier != legacy:
        raise ValueError("legacy tier does not match the template identifier")
    identity = f"t{legacy}" if legacy else template_spec(template_id).family
    return QuestionRecord(
        id=f"{repo_id}::{identity}::{template_id}::{index:04d}",
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

#: template_id -> (family, generator callable)
Generator = Callable[..., list[QuestionRecord]]
_REGISTRY: dict[str, tuple[str, Generator]] = {}


def template(template_id: str, tier: int | None = None) -> Callable[[Generator], Generator]:
    def wrap(fn: Generator) -> Generator:
        _REGISTRY[template_id] = (template_spec(template_id).family, fn)
        return fn
    return wrap


def registry() -> dict[str, tuple[str, Generator]]:
    return dict(_REGISTRY)


def templates_for_family(family: str) -> dict[str, Generator]:
    return {tid: fn for tid, (f, fn) in _REGISTRY.items() if f == family}


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
                data = json.loads(line)
                version = data.get("taxonomy_version", 1)
                if version not in (1, TAXONOMY_VERSION):
                    raise ValueError(f"unsupported taxonomy version {version}")
                legacy = {key: data.pop(key) for key in ("tier", "tier_name") if key in data}
                if legacy:
                    data["legacy_taxonomy"] = {**data.get("legacy_taxonomy", {}), **legacy}
                data["taxonomy_version"] = TAXONOMY_VERSION
                records.append(QuestionRecord(**data))
    return records
