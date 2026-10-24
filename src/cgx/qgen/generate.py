"""Generate graph-grounded questions across subject families."""

from __future__ import annotations

import random
import time
from collections import Counter

from ..logconfig import get_logger

from . import tier1_existence, tier2_path, tier3_aggregate, tier4_negative, tier5_composition, comparison, inheritance, repo_wide  # noqa: F401
from ._util import rebalance
from .base import QuestionRecord, registry
from .completeness import CompletenessFilter
from .taxonomy import DEFAULT_FAMILIES, HARD_TEMPLATES, TAXONOMY_VERSION, TRIVIAL_TEMPLATES, validate_families

log = get_logger(__name__)

#: Over-sampling factor for templates the completeness filter can shrink.
_OVERSAMPLE = 3


def _take(records: list[QuestionRecord], n: int, rng: random.Random) -> list[QuestionRecord]:
    """The first `n`, keeping a yes/no template's answers balanced.

    Which answer leads the interleave is a seeded coin. Always leading with
    "yes" made every quota of one -- the common case per repo -- ship a "yes",
    and two templates pooled to 100% "yes".
    """
    if records and records[0].answer_type == "bool":
        records = rebalance(records)
        yes = [r for r in records if r.answer is True]
        no = [r for r in records if r.answer is False]
        first, second = (yes, no) if rng.random() < 0.5 else (no, yes)
        interleaved = [r for pair in zip(first, second) for r in pair] + first[len(second):] + second[len(first):]
        return interleaved[:n]
    return records[:n]

def generate(
    client,
    repo_id: str,
    families=DEFAULT_FAMILIES,
    per_template: int = 5,
    seed: int = 0,
    per_family: int | None = None,
    include_trivial: bool = False,
    completeness: str = "default",
    hard_share: float = 0.8,
) -> tuple[list[QuestionRecord], dict]:
    """Generate all five families; optionally cap each family with a budget.

    With `per_family`, each family's budget is split between hard templates
    (repo-wide or transitive; taxonomy.HARD_TEMPLATES) and lookups by
    `hard_share`. Every template draws a pool large enough to fill its whole
    tier, and the tier is then filled round-robin across templates, so a hard
    template the repository cannot supply leaves its slots to the other hard
    templates rather than to nothing. Fixed per-template quotas lost those
    slots, which held the pilot set at 66% hard against a 3:1 weighting.
    A short lookup tier is topped up from hard pools; a short hard tier is not
    topped up with lookups -- the family is reported short instead.

    `completeness` ("default", "strict" or "off") drops questions whose answer
    reads a part of the call graph with possibly missed edges (qgen/completeness.py).
    Those templates are over-sampled so the filter costs coverage, not quota,
    where the repository allows.
    """
    if completeness not in ("default", "strict", "off"):
        raise ValueError(f"completeness must be default, strict or off, not {completeness!r}")
    if not 0.0 <= hard_share <= 1.0:
        raise ValueError("hard_share must be between 0 and 1")
    gate = None if completeness == "off" else CompletenessFilter(client, repo_id, strict=completeness == "strict")
    dropped_incomplete: dict[str, int] = {}
    families = validate_families(families)
    if per_template < 1 or (per_family is not None and per_family < 1):
        raise ValueError("question quotas must be positive")
    active = [(tid, family, fn) for tid, (family, fn) in sorted(registry().items())
              if family in families and (include_trivial or tid not in TRIVIAL_TEMPLATES)]

    # How many questions to ask each template for, and each tier's budget.
    tier_budget: dict[tuple[str, str], int] = {}
    if per_family is None:
        requests = {tid: per_template for tid, _, _ in active}
    else:
        requests = {}
        for family in families:
            tiers = {t: [tid for tid, f, _ in active if f == family and _tier(tid) == t]
                     for t in ("hard", "lookup")}
            hard = round(per_family * hard_share)
            if not tiers["lookup"]:
                hard = per_family
            elif not tiers["hard"]:
                hard = 0
            tier_budget[(family, "hard")], tier_budget[(family, "lookup")] = hard, per_family - hard
            for t, tids in tiers.items():
                # The lookup tier may borrow from hard pools, so those cover both.
                need = per_family if t == "hard" else tier_budget[(family, t)]
                for tid in tids:
                    requests[tid] = need
    log.info("generating from %d template(s)", len(active))

    pools: dict[str, list[QuestionRecord]] = {}
    for n, (template_id, family, fn) in enumerate(active, 1):
        # Seed per template so adding a template does not reshuffle the others.
        rng = random.Random(f"{seed}:{repo_id}:{template_id}")
        started = time.monotonic()
        target = requests[template_id]
        gated = gate is not None and gate.depends(template_id)
        produced = fn(client, repo_id, target * _OVERSAMPLE if gated else target, rng) if target else []
        if gated and produced:
            kept = [r for r in produced if gate.ok(r)]
            dropped_incomplete[template_id] = len(produced) - len(kept)
            produced = kept
        # Ordered so any prefix keeps a yes/no template balanced.
        pools[template_id] = _take(produced, len(produced), random.Random(f"{seed}:{repo_id}:{template_id}:take"))
        log.info("  [%2d/%d] %-32s %6.1fs  pool=%d", n, len(active), template_id,
                 time.monotonic() - started, len(pools[template_id]))

    if per_family is None:
        chosen = {tid: pool[:per_template] for tid, pool in pools.items()}
    else:
        chosen = {tid: [] for tid in pools}
        for family in families:
            order = [tid for tid, f, _ in active if f == family]
            random.Random(f"{seed}:{repo_id}:{family}:quotas").shuffle(order)
            hard = [tid for tid in order if _tier(tid) == "hard"]
            lookup = [tid for tid in order if _tier(tid) == "lookup"]
            _round_robin(pools, chosen, hard, tier_budget[(family, "hard")])
            got_lookup = _round_robin(pools, chosen, lookup, tier_budget[(family, "lookup")])
            _round_robin(pools, chosen, hard, tier_budget[(family, "lookup")] - got_lookup)
    records = [r for tid, _, _ in active for r in chosen[tid]]

    by_family = {f: sum(r.family == f for r in records) for f in families}
    stats = {
        "total": len(records),
        "taxonomy_version": TAXONOMY_VERSION,
        "by_family": by_family,
        "hard_by_family": {f: sum(r.family == f and _tier(r.template_id) == "hard" for r in records)
                           for f in families},
        "by_operation": dict(sorted(Counter(op for r in records for op in r.operations).items())),
        "by_outcome": dict(sorted(Counter(r.outcome for r in records).items())),
        "by_template": {tid: len(chosen[tid]) for tid in sorted(chosen)},
        "pool_sizes": {tid: len(pools[tid]) for tid in sorted(pools)},
        "seed": seed,
        "per_template_target": per_template if per_family is None else None,
        "per_family_target": per_family,
        "hard_share": hard_share if per_family is not None else None,
        "tier_budget": {f"{f}:{t}": b for (f, t), b in sorted(tier_budget.items())},
        "shortfalls": ({f: per_family - c for f, c in by_family.items() if c < per_family}
                       if per_family is not None else
                       {tid: per_template - len(c) for tid, c in chosen.items() if len(c) < per_template}),
        "completeness": completeness,
        "dropped_incomplete": {k: v for k, v in sorted(dropped_incomplete.items()) if v},
    }
    return records, stats


def _tier(template_id: str) -> str:
    return "hard" if template_id in HARD_TEMPLATES else "lookup"


def _round_robin(pools, chosen, tids: list[str], budget: int) -> int:
    """Take one question per template in turn until `budget` is met or the
    pools run dry; returns how many were taken."""
    taken = 0
    while taken < budget:
        progressed = False
        for tid in tids:
            if taken >= budget:
                break
            if len(chosen[tid]) < len(pools[tid]):
                chosen[tid].append(pools[tid][len(chosen[tid])])
                taken += 1
                progressed = True
        if not progressed:
            break
    return taken
