"""Phase 3 -- manual verification workflow.

Verification status lives on each question record in the JSONL dataset rather
than in a separate ledger, so a record always travels with its own provenance
and the file stays the single artifact. The `verification` block is a dict, so
extra fields (a second reviewer, a confidence score) can be added later without
migrating anything.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..qgen.base import QuestionRecord, read_questions, write_questions


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_review(path: Path, tier: int | None, sample: int, seed: int, status: str) -> None:
    records = read_questions(path)
    pool = [
        r for r in records
        if (tier is None or r.tier == tier) and r.verification.get("status") == status
    ]
    if not pool:
        print(f"nothing to review (tier={tier}, status={status})")
        return

    rng = random.Random(seed)
    chosen = pool if sample >= len(pool) else rng.sample(pool, sample)
    by_id = {r.id: r for r in records}
    reviewed = 0

    print(f"reviewing {len(chosen)} of {len(pool)} '{status}' records\n")
    print("  [v] verify   [r] reject   [s] skip   [q] save and quit\n")

    for n, rec in enumerate(chosen, 1):
        print("=" * 78)
        print(f"[{n}/{len(chosen)}]  {rec.id}")
        print(f"tier {rec.tier} ({rec.tier_name})   template {rec.template_id}")
        print(f"\nQ: {rec.question}")
        print(f"\nA: {rec.answer_text}")
        if rec.evidence:
            print(f"\nevidence: {json.dumps(rec.evidence)}")
        print(f"\ncypher:{rec.cypher}")
        print(f"params: {json.dumps(rec.params)}")

        choice = ""
        while choice not in {"v", "r", "s", "q"}:
            choice = input("\n[v/r/s/q] > ").strip().lower()

        if choice == "q":
            break
        if choice == "s":
            continue

        note = input("note (optional) > ").strip() or None
        target = by_id[rec.id]
        target.verification = {
            "status": "verified" if choice == "v" else "rejected",
            "note": note,
            "reviewer": "cli",
            "at": _now(),
        }
        reviewed += 1

    write_questions(records, path)
    print(f"\nsaved {reviewed} decision(s) to {path}")


def report(path: Path) -> None:
    records = read_questions(path)
    per_tier: dict[int, Counter] = defaultdict(Counter)
    for rec in records:
        per_tier[rec.tier][rec.verification.get("status", "pending")] += 1

    print(f"{path}  ({len(records)} questions)\n")
    header = f"{'tier':<6}{'total':>7}{'verified':>10}{'rejected':>10}{'pending':>9}"
    print(header)
    print("-" * len(header))
    for tier in sorted(per_tier):
        counts = per_tier[tier]
        total = sum(counts.values())
        print(
            f"{tier:<6}{total:>7}{counts['verified']:>10}"
            f"{counts['rejected']:>10}{counts['pending']:>9}"
        )


def _normalize(value):
    """Type-tagged so that 0 does not compare equal to False."""
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (list, tuple)):
        return ("list", tuple(str(v) for v in value))
    if isinstance(value, int):
        return ("int", value)
    if value is None:
        return ("none", None)
    return ("str", str(value))


def _matches(expected, actual) -> bool:
    """Equal as-is, or equal disregarding order (list answers are unordered)."""
    if expected == actual:
        return True
    if expected[0] == "list" and actual[0] == "list":
        return sorted(expected[1]) == sorted(actual[1])
    return False


def recheck_answers(client, records: list[QuestionRecord]) -> list[str]:
    """Re-run each stored Cypher and confirm the stored answer still appears.

    This is a consistency check, not a re-derivation: it asserts that the stored
    answer is still present in the result of its own query. It catches graph
    drift, a changed extractor, and copy-paste errors in a template -- which is
    what it is for.
    """
    mismatches: list[str] = []
    for rec in records:
        try:
            rows = client.run(rec.cypher, **rec.params)
        except Exception as exc:                       # pragma: no cover - defensive
            mismatches.append(f"{rec.id}: cypher failed: {exc}")
            continue

        expected = _normalize(rec.answer)
        found = False
        for row in rows or [{}]:
            for value in row.values():
                if _matches(expected, _normalize(value)):
                    found = True
                    break
            if found:
                break
        # An empty-list answer is legitimately produced by an empty result set.
        if not found and expected in (("list", ()), ("none", None)) and not rows:
            found = True
        if not found:
            mismatches.append(
                f"{rec.id} ({rec.template_id}): stored {rec.answer!r} not found in {rows!r}"
            )
    return mismatches
