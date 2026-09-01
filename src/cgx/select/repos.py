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

#: Published training-data cutoffs, checked 2026-09-01.
#:
#: These are SECONDARY sources -- aggregator sites and vendor help pages -- and
#: they disagree with each other on several models (see CONTESTED_CUTOFFS).
#: Before any of this reaches the paper, re-verify each entry against the
#: vendor's own model card or API docs, and record the date you checked: these
#: values move with every point release.
#:
#: Note the distinction the vendors themselves draw: the *training* cutoff is
#: the last date any data was included, while the *reliable knowledge* cutoff
#: is the date through which the model answers dependably, and is usually
#: earlier because the final months are thin. For contamination purposes the
#: training cutoff is the conservative choice, and it is what is listed here.
KNOWN_CUTOFFS = {
    # Anthropic
    "claude-opus-5": "2026-05-01",
    "claude-sonnet-5": "2026-01-01",
    "claude-fable-5": "2026-01-01",
    # OpenAI -- the GPT-5.6 family shares one cutoff (officially 2026-02-16)
    "gpt-5.6-sol": "2026-02-16",
    "gpt-5.6-terra": "2026-02-16",
    "gpt-5.6-luna": "2026-02-16",
    "gpt-5.5": "2025-12-01",
    # Open weights
    "qwen3.6-27b": "2025-10-01",
    "qwen3": "2025-01-01",
    "deepseek-v4-pro": "2025-06-01",
    "llama-4-maverick": "2024-08-01",
}

#: Models whose published cutoff could not be pinned down to a single value.
#: Deliberately kept OUT of KNOWN_CUTOFFS: a contamination filter built on a
#: disputed date gives false assurance, which is worse than no filter at all.
CONTESTED_CUTOFFS = {
    "gemini-3-pro": ["2025-01", "2025-11"],      # sources disagree by 10 months
    "gemini-3.6-flash": ["2025-11", "2026-03"],
    "grok-4.3": ["2024-11", "2025-12"],          # sources disagree by 13 months
    "grok-4.5": ["2026-02-01"],                  # single source only
    "qwen3.8": [],                               # no published cutoff found at all
}


def strictest_cutoff() -> str:
    """The latest cutoff among known models.

    A repository is only contamination-resistant with respect to a model if it
    postdates THAT model's cutoff, so a benchmark evaluated on several models
    must clear the latest one. Filtering on an average or an older model's date
    would let contaminated repos through for the newest evaluator.
    """
    return max(KNOWN_CUTOFFS.values())


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
