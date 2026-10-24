"""Pool per-repo run summaries into one score per model and condition.

    cd data/eval/final && python ../../../scripts/pool_scores.py
"""
import glob
import json
import re

MODELS = ["qwen3.8:27b", "gpt-oss:20b", "gemma4:31b", "deepseek-r1:32b", "qwen3-coder:30b", "qwen3.5:27b",
          "nemotron3:33b", "ornith:35b", "muse-glimmer:30b", "nemotron-3.5-lightning:30b",
          "qwen2.5-coder:32b", "qwen3:8b", "llama3.1:8b"]

for m in MODELS:
    slug = re.sub(r"[^A-Za-z0-9.-]+", "_", m)
    out = []
    for c in ("repo", "closed"):
        n = s = 0
        for f in glob.glob(f"*/{slug}__{c}__natural.summary.json"):
            d = json.load(open(f))
            n += d["n"]
            s += d["overall"] * d["n"]
        out.append(f"{c} {s / n:.3f} (n={n})" if n else f"{c} -")
    print(f"{m:28s}", " | ".join(out))
