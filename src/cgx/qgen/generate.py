"""Driver: run tier generators against a Neo4j graph and emit a dataset."""

from __future__ import annotations

import random
import time
from collections import Counter

from ..logconfig import get_logger

from . import tier1_existence, tier2_path, tier3_aggregate, tier4_negative, tier5_composition  # noqa: F401
from .base import QuestionRecord, registry

log = get_logger(__name__)

DEFAULT_TIERS = (1, 2, 3, 4)


def generate(
    client,
    repo_id: str,
    tiers=DEFAULT_TIERS,
    per_template: int = 5,
    seed: int = 0,
) -> tuple[list[QuestionRecord], dict]:
    """Generate questions for `tiers`, `per_template` each, deterministically."""
    records: list[QuestionRecord] = []
    per_template_counts: dict[str, int] = {}
    active = [(tid, t, fn) for tid, (t, fn) in sorted(registry().items()) if t in tiers]
    log.info("generating from %d template(s), %d question(s) each", len(active), per_template)

    for n, (template_id, tier, fn) in enumerate(active, 1):
        # Seed per template so adding a template does not reshuffle the others.
        rng = random.Random(f"{seed}:{repo_id}:{template_id}")
        started = time.monotonic()
        produced = fn(client, repo_id, per_template, rng)
        elapsed = time.monotonic() - started
        per_template_counts[template_id] = len(produced)
        records.extend(produced)
        log.info("  [%2d/%d] %-26s %6.1fs  n=%d%s", n, len(active), template_id,
                 elapsed, len(produced),
                 "  (short: repo has fewer eligible candidates)"
                 if len(produced) < per_template else "")

    stats = {
        "total": len(records),
        "by_tier": dict(sorted(Counter(r.tier for r in records).items())),
        "by_template": dict(sorted(per_template_counts.items())),
        "seed": seed,
        "per_template_target": per_template,
    }
    return records, stats
