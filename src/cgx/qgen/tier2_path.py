"""Tier 2 -- multi-hop path traversal along CALLS and USES.

These are the questions the CALLS edge exists to make askable. None of them are
answerable from a single edge lookup, and none of them are cheap to answer by
reading one file: the shortest chain from `app.run` to `utils.strip_ws` crosses
three modules.
"""

from __future__ import annotations

import random

from .base import make_question, template
from ._util import callables, pick

# Neo4j only allows a shortestPath lower bound of 0 or 1, so the "at least two
# hops" constraint -- which is what keeps tier 2 from degenerating into tier 1
# single-edge lookups -- is applied after the fact on path length.
_SAMPLE_PATHS = """
MATCH p = shortestPath(
    (a:CodeNode {repo_id: $repo})-[:CALLS*1..8]->(b:CodeNode {repo_id: $repo})
)
WHERE (a:FUNCTION OR a:METHOD) AND (b:FUNCTION OR b:METHOD)
  AND a <> b AND length(p) >= 2
RETURN a.qname AS source, b.qname AS target, length(p) AS hops
ORDER BY hops DESC, source, target
LIMIT 400
"""

# A question's stored Cypher must itself yield the stored answer -- that is the
# provenance claim the benchmark rests on. So a yes/no question stores a query
# returning a boolean, not one returning a path that happens to imply it.
_GT_PATH_EXISTS = """
OPTIONAL MATCH p = shortestPath(
    (a:CodeNode {repo_id: $repo, qname: $source})-[:CALLS*1..12]->(b:CodeNode {repo_id: $repo, qname: $target})
)
RETURN p IS NOT NULL AS reachable
"""

_GT_SHORTEST_PATH = """
MATCH p = shortestPath(
    (a:CodeNode {repo_id: $repo, qname: $source})-[:CALLS*1..12]->(b:CodeNode {repo_id: $repo, qname: $target})
)
RETURN [n IN nodes(p) | n.qname] AS chain, length(p) AS hops
"""


def _shortest(client, repo_id, source, target):
    rows = client.run(_GT_SHORTEST_PATH, repo=repo_id, source=source, target=target)
    return (rows[0]["chain"], rows[0]["hops"]) if rows else (None, None)


@template("t2_path_exists", 2)
def t2_path_exists(client, repo_id, limit, rng: random.Random):
    rows = pick(rng, client.run(_SAMPLE_PATHS, repo=repo_id), limit)
    out = []
    for i, row in enumerate(rows):
        reachable = client.run(
            _GT_PATH_EXISTS, repo=repo_id, source=row["source"], target=row["target"]
        )[0]["reachable"]
        chain, hops = _shortest(client, repo_id, row["source"], row["target"])
        out.append(make_question(
            tier=2, template_id="t2_path_exists", repo_id=repo_id, index=i,
            question=(
                f"Following function and method calls only, can execution starting in "
                f"`{row['source']}` ever reach `{row['target']}`?"
            ),
            answer=bool(reachable), answer_type="bool",
            cypher=_GT_PATH_EXISTS,
            params={"repo": repo_id, "source": row["source"], "target": row["target"]},
            evidence={"chain": chain, "hops": hops},
        ))
    return out


@template("t2_shortest_chain", 2)
def t2_shortest_chain(client, repo_id, limit, rng: random.Random):
    rows = pick(rng, client.run(_SAMPLE_PATHS, repo=repo_id), limit)
    out = []
    for i, row in enumerate(rows):
        chain, hops = _shortest(client, repo_id, row["source"], row["target"])
        out.append(make_question(
            tier=2, template_id="t2_shortest_chain", repo_id=repo_id, index=i,
            question=(
                f"What is the shortest chain of calls from `{row['source']}` to `{row['target']}`? "
                f"List every callable on the path in order, starting with `{row['source']}`."
            ),
            answer=chain, answer_type="path",
            cypher=_GT_SHORTEST_PATH,
            params={"repo": repo_id, "source": row["source"], "target": row["target"]},
            evidence={"hops": hops},
        ))
    return out


@template("t2_hop_count", 2)
def t2_hop_count(client, repo_id, limit, rng: random.Random):
    rows = pick(rng, client.run(_SAMPLE_PATHS, repo=repo_id), limit)
    out = []
    for i, row in enumerate(rows):
        chain, hops = _shortest(client, repo_id, row["source"], row["target"])
        out.append(make_question(
            tier=2, template_id="t2_hop_count", repo_id=repo_id, index=i,
            question=(
                f"How many call steps are on the shortest path from `{row['source']}` to "
                f"`{row['target']}`? Answer with a single integer."
            ),
            answer=hops, answer_type="count",
            cypher=_GT_SHORTEST_PATH,
            params={"repo": repo_id, "source": row["source"], "target": row["target"]},
            evidence={"chain": chain},
        ))
    return out


_GT_REACHABLE_K = """
MATCH (a:CodeNode {repo_id: $repo, qname: $source})-[:CALLS*1..2]->(b:CodeNode)
WHERE b <> a
RETURN collect(DISTINCT b.qname) AS reachable
"""


@template("t2_reachable_within_2", 2)
def t2_reachable_within_2(client, repo_id, limit, rng: random.Random):
    rows = [r for r in callables(client, repo_id)]
    rows = pick(rng, rows, limit * 3)
    out = []
    for row in rows:
        res = client.run(_GT_REACHABLE_K, repo=repo_id, source=row["qname"])
        reachable = sorted(res[0]["reachable"]) if res else []
        if not reachable:
            continue                      # empty answers belong to tier 4
        out.append(make_question(
            tier=2, template_id="t2_reachable_within_2", repo_id=repo_id, index=len(out),
            question=(
                f"Which functions or methods in this repository can be reached from "
                f"`{row['qname']}` in at most two call steps? List their fully qualified names."
            ),
            answer=reachable, answer_type="list",
            cypher=_GT_REACHABLE_K, params={"repo": repo_id, "source": row["qname"]},
            evidence={"count": len(reachable)},
        ))
        if len(out) >= limit:
            break
    return out


_SAMPLE_USES_CHAIN = """
MATCH (f:CodeNode {repo_id: $repo})-[:USES]->(g:GLOBAL_VARIABLE)
WHERE f:FUNCTION OR f:METHOD
RETURN g.qname AS global, f.qname AS reader
ORDER BY g.qname, f.qname
"""

_GT_CALLERS_READING_GLOBAL = """
MATCH (caller:CodeNode {repo_id: $repo, qname: $source})-[:CALLS*0..6]->(f:CodeNode)-[:USES]->(g:GLOBAL_VARIABLE {qname: $global})
RETURN count(DISTINCT f) > 0 AS reaches, collect(DISTINCT f.qname) AS readers
"""


@template("t2_reaches_global_read", 2)
def t2_reaches_global_read(client, repo_id, limit, rng: random.Random):
    """Does a call starting here end up reading a particular global?"""
    pairs = client.run(_SAMPLE_USES_CHAIN, repo=repo_id)
    calls = callables(client, repo_id)
    if not pairs or not calls:
        return []
    candidates = [(c["qname"], p["global"]) for c in calls for p in pairs]
    seen, uniq = set(), []
    for src, glob in candidates:
        if (src, glob) not in seen:
            seen.add((src, glob))
            uniq.append((src, glob))
    out = []
    for src, glob in pick(rng, uniq, limit * 4):
        res = client.run(_GT_CALLERS_READING_GLOBAL, repo=repo_id, source=src, **{"global": glob})
        reaches = bool(res and res[0]["reaches"])
        if not reaches:
            continue                      # negatives belong to tier 4
        out.append(make_question(
            tier=2, template_id="t2_reaches_global_read", repo_id=repo_id, index=len(out),
            question=(
                f"Starting from `{src}` and following calls, is the global variable "
                f"`{glob}` read anywhere along the way (including in `{src}` itself)?"
            ),
            answer=True, answer_type="bool",
            cypher=_GT_CALLERS_READING_GLOBAL,
            params={"repo": repo_id, "source": src, "global": glob},
            evidence={"readers": sorted(res[0]["readers"])},
        ))
        if len(out) >= limit:
            break
    return out
