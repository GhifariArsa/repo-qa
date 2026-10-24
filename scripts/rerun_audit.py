"""How much did re-running finished questions change each run's score?

    cd data/eval/final && python ../../../scripts/rerun_audit.py [questions_dir]

Before `harness.is_final`, a resume re-ran questions whose model gave no
answer and kept the latest attempt. This compares that ("latest") with the
rule now in force ("first final": the first attempt that is a model outcome;
only infrastructure errors are replaced). Prints runs with any re-run, and
lists the question ids whose counted attempt changes.
"""
import collections
import glob
import json
import sys

qdir = sys.argv[1] if len(sys.argv) > 1 else "../../questions_v4_eval"
ids = {json.loads(l)["id"] for f in glob.glob(f"{qdir}/*.jsonl") for l in open(f)}


def final(r):
    return not r["error"] or r["error"] == "no answer submitted"


runs = collections.defaultdict(lambda: {"latest": {}, "first": {}, "reruns": 0})
for f in sorted(glob.glob("*/*__natural.jsonl")):
    key = f.replace("__natural.jsonl", "")
    seen = set()
    for line in open(f):
        r = json.loads(line)
        if r["id"] not in ids:
            continue
        d = runs[key]
        d["reruns"] += r["id"] in seen
        seen.add(r["id"])
        d["latest"][r["id"]] = r
        if r["id"] not in d["first"] or not final(d["first"][r["id"]]):
            d["first"][r["id"]] = r


def acc(rows):
    v = [r for r in rows.values() if final(r)]
    return (sum(r["score"] for r in v) / len(v) if v else 0.0), len(v)


by_model = collections.defaultdict(lambda: [0, 0.0, 0.0])
for key, d in sorted(runs.items()):
    (a, n), (b, m) = acc(d["latest"]), acc(d["first"])
    model = key.split("/")[1]
    by_model[model][0] += m
    by_model[model][1] += a * n
    by_model[model][2] += b * m
    if d["reruns"]:
        print(f"{key:48s} reruns={d['reruns']:3d}  latest={a:.3f} (n={n})  first-final={b:.3f} (n={m})  diff={b - a:+.3f}")
print()
for model, (n, a, b) in sorted(by_model.items()):
    if n:
        print(f"{model:36s} latest={a / n:.3f}  first-final={b / n:.3f}  diff={(b - a) / n:+.3f}")
