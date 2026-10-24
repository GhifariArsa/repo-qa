"""Shared sampling helpers for the tier generators."""

from __future__ import annotations

import random
from typing import Any


def pick(rng: random.Random, rows: list[Any], limit: int) -> list[Any]:
    """Deterministic sample without replacement, capped at `limit`."""
    if limit >= len(rows):
        return list(rows)
    return rng.sample(rows, limit)


#: Every callable (FUNCTION or METHOD) in a repo, ordered for determinism.
CALLABLES = """
MATCH (n:CodeNode {repo_id: $repo})
WHERE n:FUNCTION OR n:METHOD
RETURN n.qname AS qname, n.name AS name, n.class_qname AS class_qname
ORDER BY n.qname
"""


def callables(client, repo_id: str) -> list[dict]:
    return client.run(CALLABLES, repo=repo_id)


def split_quota(limit: int, rng: random.Random) -> tuple[int, int]:
    """(positives, negatives) for a yes/no template: as even as `limit` allows.

    A yes/no template whose answer is always the same is answered by its wording
    alone -- the pilot's closed-book arm scored 1.00 on every such template.
    An odd count's extra question goes to either side by a seeded coin: always
    giving it to "yes" balanced each repo but tilted the pooled set 10 to 1,
    because per-repo counts of one or three are the common case.
    """
    half, extra = divmod(limit, 2)
    yes_gets_extra = extra and rng.random() < 0.5
    return half + (1 if yes_gets_extra else 0), half + (extra if not yes_gets_extra else 0)


def rebalance(records: list) -> list:
    """Trim the majority answer to at most one more than the minority.

    When one side cannot be filled -- a tiny repo with no unreachable pair, a
    module whose every function reads its global -- shipping the surplus would
    make the template's answer guessable again. Fewer questions is the right
    trade; the shortfall shows up in the generation stats.
    """
    yes = [r for r in records if r.answer is True]
    no = [r for r in records if r.answer is False]
    keep = min(len(yes), len(no)) + 1
    kept = {id(r) for r in yes[:keep] + no[:keep]}
    return [r for r in records if id(r) in kept]
