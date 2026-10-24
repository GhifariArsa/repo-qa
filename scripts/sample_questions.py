"""Draw a fixed evaluation sample, proportional across repo x family x template.

    python scripts/sample_questions.py <in_dir> <out_dir> <n> [seed]

Every (repo, family, template) cell keeps its share of `n` (largest-remainder
rounding), so the sample has the dataset's mix of repositories, families and
templates -- including its hard/lookup split -- rather than evening them out.
Within a cell the draw is seeded, and a yes/no cell takes alternate answers so
it stays balanced. Writes one JSONL per repo plus a manifest of what was kept.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def main(in_dir: str, out_dir: str, n: int, seed: int = 0) -> None:
    src, dst = Path(in_dir), Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    rows = {p.stem: [json.loads(l) for l in p.open()] for p in sorted(src.glob("*.jsonl"))}
    cells: dict[tuple, list] = defaultdict(list)
    for repo, recs in rows.items():
        for r in recs:
            cells[(repo, r["family"], r["template_id"])].append(r)
    total = sum(len(v) for v in cells.values())
    # Allocate top-down -- repo, then family, then template -- each level by
    # largest remainder with seeded tie-breaks. Rounding the tiny leaf cells
    # directly left most of them tied, and the tie-break then decided which
    # repositories got the spare questions.
    take = _allocate(cells, n, seed)
    kept: dict[str, list] = defaultdict(list)
    for key in sorted(cells):
        recs = cells[key][:]
        rng = random.Random(f"{seed}:{':'.join(key)}")
        rng.shuffle(recs)
        if recs and recs[0]["answer_type"] == "bool":
            yes = [r for r in recs if r["answer"] is True]
            no = [r for r in recs if r["answer"] is False]
            first, second = (yes, no) if rng.random() < 0.5 else (no, yes)
            recs = [r for pair in zip(first, second) for r in pair] + first[len(second):] + second[len(first):]
        kept[key[0]] += recs[: take[key]]

    manifest = {"source": str(src), "n_requested": n, "n_source": total, "seed": seed, "by_repo": {}}
    for repo in rows:
        order = {r["id"]: i for i, r in enumerate(rows[repo])}
        chosen = sorted(kept[repo], key=lambda r: order[r["id"]])
        with (dst / f"{repo}.jsonl").open("w") as fh:
            for r in chosen:
                fh.write(json.dumps(r) + "\n")
        manifest["by_repo"][repo] = [r["id"] for r in chosen]
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"kept {sum(len(v) for v in kept.values())} of {total}")


def _split(sizes: dict, n: int, rng: random.Random) -> dict:
    total = sum(sizes.values())
    exact = {k: v * n / total for k, v in sizes.items()}
    out = {k: int(x) for k, x in exact.items()}
    keys = sorted(exact, key=lambda k: (exact[k] - out[k], rng.random()), reverse=True)
    for k in keys[: n - sum(out.values())]:
        out[k] += 1
    return out


def _allocate(cells: dict, n: int, seed: int) -> dict:
    rng = random.Random(f"alloc:{seed}")
    take = {}
    repos = defaultdict(int)
    for (repo, _, _), v in cells.items():
        repos[repo] += len(v)
    for repo, n_repo in _split(dict(repos), n, rng).items():
        fams = defaultdict(int)
        for (r, fam, _), v in cells.items():
            if r == repo:
                fams[fam] += len(v)
        for fam, n_fam in _split(dict(fams), n_repo, rng).items():
            tpls = {k: len(v) for k, v in cells.items() if k[0] == repo and k[1] == fam}
            take.update(_split(tpls, n_fam, rng))
    return take


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0)
