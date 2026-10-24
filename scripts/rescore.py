"""Re-score the final run: corrected gold answers and the v2 answer-form rules.

    uv run python scripts/rescore.py [--results data/eval/final] [--questions data/questions_v4_eval]
                                     [--graphs data/graphs_v2]

Needs Neo4j loaded with the graphs in --graphs (`cgx ingest <repo> --graph ...`).

For every question in the evaluation sample, the stored Cypher is re-run on
the current graph; where the stored answer no longer appears (the aliased-
import fix restored call edges), the gold is replaced by the query's value of
the same type. Each run's counted attempt is the first final one
(`harness.is_final`). Every counted row is then scored two ways:

  v1  the stored score: original gold, the scorer used during the run
  v2  corrected gold, `harness.score_v2` with the repaired graph's facts

Writes <results>/rescored.jsonl (one row per model x condition x question)
and <results>/rescored_summary.json, and prints the per-model table.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
from collections import defaultdict
from pathlib import Path

from cgx.config import load_config
from cgx.eval.harness import is_final, load_graph_facts, score_v2
from cgx.graph.client import client as graph_client
from cgx.qgen.base import read_questions
from cgx.qgen.taxonomy import HARD_TEMPLATES
from cgx.verify.cli import _matches, _normalize


def corrected_gold(client, rec):
    """The stored answer if its Cypher still yields it, else the new value of the same type."""
    rows = client.run(rec.cypher, **rec.params) or []
    expected = _normalize(rec.answer)
    values = [v for row in rows for v in row.values()]
    if any(_matches(expected, _normalize(v)) for v in values):
        return rec.answer, False
    if expected in (("list", ()), ("none", None)) and not rows:
        return rec.answer, False
    same_type = [v for v in values if _normalize(v)[0] == expected[0]]
    if not same_type:
        raise SystemExit(f"{rec.id}: no value of type {expected[0]} in {rows!r}")
    return same_type[0], True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/eval/final")
    ap.add_argument("--questions", default="data/questions_v4_eval")
    ap.add_argument("--graphs", default="data/graphs_v2")
    args = ap.parse_args()
    results, qdir, gdir = Path(args.results), Path(args.questions), Path(args.graphs)

    cfg = load_config(Path("config/repos.yaml"))
    questions, changed = {}, []
    with graph_client(cfg.neo4j) as c:
        for qfile in sorted(qdir.glob("*.jsonl")):
            for rec in read_questions(qfile):
                gold, moved = corrected_gold(c, rec)
                if moved:
                    changed.append({"id": rec.id, "old": rec.answer, "new": gold})
                questions[rec.id] = (rec, gold)
    facts = {p.stem: load_graph_facts(p) for p in gdir.glob("*.json")}

    out_rows = []
    for f in sorted(results.glob("*/*__natural.jsonl")):
        repo = f.parent.name
        if repo not in facts:              # e.g. backup_pre_parse_rerun/
            continue
        slug, condition, _ = f.name.split("__")
        counted = {}
        for line in f.open():
            r = json.loads(line)
            if r["id"] in questions and (r["id"] not in counted or not is_final(counted[r["id"]])):
                counted[r["id"]] = r
        for qid, r in counted.items():
            if not is_final(r):
                continue                                   # unresolved outage: excluded, as in v1
            rec, gold = questions[qid]
            fixed = dataclasses.replace(rec, answer=gold)
            v2 = score_v2(fixed, r["prediction"], facts[repo]) if r["answered"] else 0.0
            out_rows.append({
                "model": slug, "condition": condition, "repo": repo, "id": qid,
                "family": rec.family, "template_id": rec.template_id,
                "hard": rec.template_id in HARD_TEMPLATES, "outcome": rec.outcome,
                "answer_type": rec.answer_type, "gold_v1": rec.answer, "gold_v2": gold,
                "prediction": r["prediction"], "answered": r["answered"],
                "score_v1": r["score"], "score_v2": v2,
                **{k: r.get(k) for k in ("steps", "tool_calls", "seconds", "model_seconds", "prompt_tokens",
                                         "completion_tokens", "peak_prompt_tokens", "invalid_tool_calls",
                                         "no_tool_turns", "files_read", "forced", "text_submission")},
            })

    with (results / "rescored.jsonl").open("w") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, default=str) + "\n")

    groups = defaultdict(list)
    for row in out_rows:
        groups[(row["model"], row["condition"])].append(row)
    mean = lambda xs: round(statistics.mean(xs), 4) if xs else None  # noqa: E731
    summary = {"gold_changed": changed, "runs": {}}
    for (model, cond), rows in sorted(groups.items()):
        hard = [r for r in rows if r["hard"]]
        summary["runs"][f"{model}__{cond}"] = {
            "n": len(rows),
            "v1": mean([r["score_v1"] for r in rows]), "v2": mean([r["score_v2"] for r in rows]),
            "hard_v2": mean([r["score_v2"] for r in hard]),
            "lookup_v2": mean([r["score_v2"] for r in rows if not r["hard"]]),
            "no_answer": sum(not r["answered"] for r in rows),
            "forced": sum(bool(r["forced"]) for r in rows),
            "mean_steps": mean([r["steps"] for r in rows]),
            "mean_seconds": mean([r["seconds"] for r in rows]),
            "mean_prompt_tokens": mean([r["prompt_tokens"] for r in rows]),
            "mean_completion_tokens": mean([r["completion_tokens"] for r in rows]),
            "invalid_tool_calls": sum(r["invalid_tool_calls"] or 0 for r in rows),
            "near_context_cap": sum((r["peak_prompt_tokens"] or 0) > 0.9 * 40960 for r in rows),
        }
    (results / "rescored_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"gold answers corrected: {len(changed)}")
    print(f"{'model':30s} {'cond':6s} {'n':>4s} {'v1':>6s} {'v2':>6s} {'hard':>6s} {'lookup':>6s} {'noans':>5s}")
    for key, s in sorted(summary["runs"].items(), key=lambda kv: (kv[0].split("__")[1], -kv[1]["v2"])):
        model, cond = key.split("__")
        print(f"{model:30s} {cond:6s} {s['n']:4d} {s['v1']:6.3f} {s['v2']:6.3f} {s['hard_v2']:6.3f} "
              f"{s['lookup_v2'] if s['lookup_v2'] is not None else float('nan'):6.3f} {s['no_answer']:5d}")


if __name__ == "__main__":
    main()
