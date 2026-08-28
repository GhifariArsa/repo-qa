"""Driver: run tier generators against a Neo4j graph and emit a dataset."""

from __future__ import annotations

import random
from collections import Counter

from . import tier1_existence, tier2_path, tier3_aggregate, tier4_negative, tier5_composition  # noqa: F401
from .base import QuestionRecord, registry

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

    for template_id, (tier, fn) in sorted(registry().items()):
        if tier not in tiers:
            continue
        # Seed per template so adding a template does not reshuffle the others.
        rng = random.Random(f"{seed}:{repo_id}:{template_id}")
        produced = fn(client, repo_id, per_template, rng)
        per_template_counts[template_id] = len(produced)
        records.extend(produced)

    stats = {
        "total": len(records),
        "by_tier": dict(sorted(Counter(r.tier for r in records).items())),
        "by_template": dict(sorted(per_template_counts.items())),
        "seed": seed,
        "per_template_target": per_template,
    }
    return records, stats
