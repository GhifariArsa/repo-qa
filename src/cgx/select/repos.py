"""Phase 4 -- contamination-aware candidate repo selection.

Filters GitHub search results by creation date, biasing toward repositories
created after current model training cutoffs, on the assumption that a repo the
model has never seen cannot be answered from memory.

Caveats worth stating in the paper rather than hiding here:
  * Creation date is a weak proxy. A repo created recently can vendor, fork, or
    copy older code that *is* in training data.
  * Cutoffs are approximate and vendor-specific, and they move.
  * Popularity and recency are correlated with being scraped, so a high-star
    recent repo is a worse bet than the star filter suggests.

FLAGGED, NOT BUILT: a fully synthetic repository -- generated code with a known
graph, never published -- is the only contamination-proof source, since it
cannot appear in any training set. It also trades away ecological validity:
questions over generated code may not resemble questions over real code. The
intended use is as a control arm alongside the real repos, not a replacement.
See `synthetic_repo_note()`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

#: Rough public training-cutoff dates, for the `post_cutoff` flag only.
KNOWN_CUTOFFS = {
    "claude-opus-4.5": "2025-02-01",
    "gpt-5": "2024-10-01",
    "gemini-2.5-pro": "2025-01-01",
}


def find_candidates(
    out: Path,
    min_created: str,
    min_stars: int,
    limit: int,
    query: str,
) -> list[dict]:
    """Query the GitHub search API via `gh` and write filtered candidates."""
    search = f"{query} stars:>={min_stars} created:>={min_created} archived:false"
    cmd = [
        "gh", "api", "-X", "GET", "search/repositories",
        "-f", f"q={search}", "-f", "sort=stars", "-f", "order=desc",
        "-F", f"per_page={min(limit, 100)}",
    ]
    try:
        raw = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        raise SystemExit("`gh` CLI not found; install it or run the search manually")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"gh search failed: {exc.stderr.strip()}")

    payload = json.loads(raw)
    candidates = []
    for item in payload.get("items", [])[:limit]:
        candidates.append({
            "id": item["full_name"].replace("/", "__"),
            "full_name": item["full_name"],
            "url": item["html_url"],
            "created_at": item["created_at"],
            "pushed_at": item["pushed_at"],
            "stars": item["stargazers_count"],
            "language": item.get("language"),
            "license": (item.get("license") or {}).get("spdx_id"),
            "post_cutoff": {
                model: item["created_at"][:10] >= cutoff
                for model, cutoff in KNOWN_CUTOFFS.items()
            },
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"query": search, "candidates": candidates}, indent=2))
    print(f"{len(candidates)} candidate(s) -> {out}")
    for c in candidates[:10]:
        flags = "".join("*" if v else "." for v in c["post_cutoff"].values())
        print(f"  [{flags}] {c['full_name']:<45} created {c['created_at'][:10]}  {c['stars']}*")
    if candidates:
        print("\n  flags = post-cutoff vs " + ", ".join(KNOWN_CUTOFFS))
    return candidates


def synthetic_repo_note() -> str:
    return (
        "Synthetic-repo option (flagged, not implemented): generate a repository with a "
        "known target graph -- prescribed inheritance depth, call-chain lengths, and "
        "fan-in distribution -- so ground truth is known by construction and contamination "
        "is impossible. Cost: ecological validity. Recommended as a control arm, not a "
        "replacement for real repos."
    )
