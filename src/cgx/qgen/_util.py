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
