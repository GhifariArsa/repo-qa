"""Phase 5 -- minimal evaluation harness (per-tier scoring).

Deliberately thin. It exists so the pipeline can be validated on real output
rather than only inspected by hand, and so per-tier accuracy -- the number this
benchmark is built to report -- has somewhere to land.

Two models ship:
  * `stub`     -- answers "Yes" to every yes/no question and gives an empty or
                  zero answer otherwise. A majority-class baseline: it should
                  score near zero on tier 4, which is a live check that the
                  negative tier is doing its job.
  * `oracle`   -- replays the stored ground truth. Should score 1.0 everywhere;
                  a failure means the scorer, not the model, is broken.

Wiring a real LLM or coding agent means implementing one function with the
signature of `stub_model` and registering it in MODELS.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

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
    if answer_type == "list":
        return sorted(str(v).strip() for v in (value or []))
    if answer_type == "path":
        return [str(v).strip() for v in (value or [])]
    return str(value or "").strip()


def score(rec: QuestionRecord, prediction) -> float:
    """Exact match, except lists score by F1 so partial recall is visible."""
    gold = _norm(rec.answer, rec.answer_type)
    pred = _norm(prediction, rec.answer_type)
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

    per_tier: dict[int, list[float]] = defaultdict(list)
    per_template: dict[str, list[float]] = defaultdict(list)
    results = []
    for rec in records:
        prediction = fn(rec)
        s = score(rec, prediction)
        per_tier[rec.tier].append(s)
        per_template[rec.template_id].append(s)
        results.append({"id": rec.id, "tier": rec.tier, "score": s, "prediction": prediction})

    summary = {
        "model": model,
        "n": len(records),
        "overall": round(sum(r["score"] for r in results) / len(results), 4) if results else 0.0,
        "by_tier": {t: round(sum(v) / len(v), 4) for t, v in sorted(per_tier.items())},
        "by_template": {t: round(sum(v) / len(v), 4) for t, v in sorted(per_template.items())},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    return summary
