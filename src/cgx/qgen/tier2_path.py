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

#: Hop-length buckets for tier 2. Sampling is stratified across these rather
#: than drawn from one pool, for two reasons.
#:
#: Correctness: the old query ordered ALL pairs by `hops DESC` and took 400
#: before sampling. On the neo4j driver that admitted only 5-8 hop pairs --
#: 2-4 hop pairs are 83.5% of the population and NONE were eligible. The bias
#: was invisible on the fixture, which has 26 pairs in total.
#:
#: Scale: it also computed shortestPath() over every callable pair. Django has
#: 9225 callables, so 85,100,625 path computations before the LIMIT applied,
#: and tier 2 simply never finished.
#:
#: Both are fixed by anchoring on a bounded random sample of sources and
#: expanding outward from each, then filling buckets. Anchors are drawn with
#: the seeded RNG, so generation stays reproducible (Cypher's `rand()` would
#: not be).
HOP_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("2", 2, 2),
    ("3", 3, 3),
    ("4+", 4, 12),
)

#: How many source anchors to expand before giving up on filling every bucket.
MAX_ANCHORS = 60

#: Seconds any single candidate check may take before it is abandoned.
#: Variable-length traversal cost is wildly pair-dependent on a large graph --
#: milliseconds for one pair, minutes for another in a dense region -- so a
#: candidate that cannot be decided quickly is skipped rather than allowed to
#: stall the run.
CANDIDATE_TIMEOUT = 5.0

#: Seconds one source anchor's outward expansion may take.
ANCHOR_TIMEOUT = 10.0

_ALL_CALLABLES = """
MATCH (n:CodeNode {repo_id: $repo})
WHERE n:FUNCTION OR n:METHOD
RETURN n.qname AS qname
ORDER BY n.qname
"""

#: Everything reachable from ONE source, with its shortest distance. Bounded by
#: the graph rather than by the pair space, so this is linear per anchor.
#: The `hops >= 2` filter MUST come after a WITH. Putting it in the same WHERE
#: as the shortestPath lets Neo4j push the predicate into the path search, so
#: it returns the shortest path *satisfying the predicate* rather than the true
#: shortest -- reporting a genuinely 1-hop pair as 2 hops. That silently breaks
#: tier 2's defining property (multi-hop) and mislabels the hop bucket.
_PATHS_FROM_SOURCE = """
MATCH p = shortestPath(
    (a:CodeNode {repo_id: $repo, qname: $source})-[:CALLS*1..12]->(b:CodeNode {repo_id: $repo})
)
WHERE (b:FUNCTION OR b:METHOD) AND b.qname <> $source
WITH b, length(p) AS hops
WHERE hops >= 2
RETURN b.qname AS target, hops
ORDER BY b.qname
"""


def _bucket_for(hops: int) -> str | None:
    for label, lo, hi in HOP_BUCKETS:
        if lo <= hops <= hi:
            return label
    return None


def sample_paths_stratified(client, repo_id, limit, rng: random.Random) -> list[dict]:
    """Sample (source, target, hops) triples spread across hop buckets.

    Returns up to `limit` rows, drawn as evenly as the repo allows. A bucket the
    repo cannot fill contributes what it has -- the shortfall is not silently
    made up from another bucket, since that is the confound stratifying is
    meant to remove.
    """
    callables = [r["qname"] for r in client.run(_ALL_CALLABLES, repo=repo_id)]
    if not callables:
        return []
    anchors = pick(rng, callables, min(MAX_ANCHORS, len(callables)))

    per_bucket = max(1, -(-limit // len(HOP_BUCKETS)))     # ceil
    found: dict[str, list[dict]] = {label: [] for label, _, _ in HOP_BUCKETS}

    for source in anchors:
        rows = client.run_with_timeout(
            _PATHS_FROM_SOURCE, ANCHOR_TIMEOUT, repo=repo_id, source=source
        )
        if rows is None:
            continue                      # pathological anchor -- try the next
        for row in rows:
            label = _bucket_for(row["hops"])
            if label is None:
                continue
            found[label].append(
                {"source": source, "target": row["target"], "hops": row["hops"]}
            )
        # Oversample before stopping so the per-bucket draw still has a choice.
        if all(len(v) >= per_bucket * 4 for v in found.values()):
            break

    out: list[dict] = []
    for label, _, _ in HOP_BUCKETS:
        out.extend(pick(rng, found[label], per_bucket))
    return out[:limit]


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

#: `shortestPath()` returns ONE arbitrary path when several tie for shortest.
#: For a question whose answer *is* the path, that makes the stored chain only
#: one of several correct answers -- a model naming an equally short alternative
#: would be scored wrong. Pairs with more than one shortest path are therefore
#: not eligible for t2_shortest_chain. `t2_path_exists` and `t2_hop_count` are
#: unaffected: a boolean and a length are identical across tied paths.
_GT_COUNT_SHORTEST = """
MATCH p = allShortestPaths(
    (a:CodeNode {repo_id: $repo, qname: $source})-[:CALLS*1..12]->(b:CodeNode {repo_id: $repo, qname: $target})
)
RETURN count(p) AS n_shortest
"""

#: Cheaper equivalent when the shortest distance is already known, which it is
#: after stratified sampling. Any path of exactly that length IS a shortest
#: path, so counting fixed-length paths counts shortest paths -- without making
#: the planner search a 1..12 range. `allShortestPaths` over that range is what
#: made this template hang on Django for some randomly drawn pairs.
_GT_COUNT_PATHS_AT_LENGTH = """
MATCH p = (a:CodeNode {repo_id: $repo, qname: $source})-[:CALLS*%d]->(b:CodeNode {repo_id: $repo, qname: $target})
RETURN count(p) AS n_shortest
"""


def _n_shortest(client, repo_id, source, target, hops: int | None = None) -> int | None:
    """Number of shortest paths, or None if the check could not be completed."""
    if hops is not None:
        rows = client.run_with_timeout(
            _GT_COUNT_PATHS_AT_LENGTH % hops, CANDIDATE_TIMEOUT,
            repo=repo_id, source=source, target=target,
        )
    else:
        rows = client.run_with_timeout(
            _GT_COUNT_SHORTEST, CANDIDATE_TIMEOUT,
            repo=repo_id, source=source, target=target,
        )
    if rows is None:
        return None
    return rows[0]["n_shortest"] if rows else 0


def _shortest(client, repo_id, source, target):
    """Shortest chain, or (None, None) if absent or too expensive to compute."""
    rows = client.run_with_timeout(
        _GT_SHORTEST_PATH, CANDIDATE_TIMEOUT, repo=repo_id, source=source, target=target
    )
    if not rows:
        return (None, None)
    return (rows[0]["chain"], rows[0]["hops"])


@template("t2_path_exists", 2)
def t2_path_exists(client, repo_id, limit, rng: random.Random):
    rows = sample_paths_stratified(client, repo_id, limit, rng)
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
            evidence={"chain": chain, "hops": hops, "hop_bucket": _bucket_for(hops)},
        ))
    return out


@template("t2_shortest_chain", 2)
def t2_shortest_chain(client, repo_id, limit, rng: random.Random):
    # Oversample, because pairs with tied shortest paths get skipped. Accept
    # round-robin across buckets: taking the first N survivors in bucket order
    # filled the whole template from the 2-hop bucket, since short paths are
    # far likelier to have a unique shortest path than long ones.
    rows = sample_paths_stratified(client, repo_id, limit * 4, rng)
    by_bucket: dict[str, list[dict]] = {label: [] for label, _, _ in HOP_BUCKETS}
    for row in rows:
        label = _bucket_for(row["hops"])
        if label is not None:
            by_bucket[label].append(row)
    ordered: list[dict] = []
    for i in range(max((len(v) for v in by_bucket.values()), default=0)):
        for label, _, _ in HOP_BUCKETS:
            if i < len(by_bucket[label]):
                ordered.append(by_bucket[label][i])

    out = []
    skipped_ambiguous = 0
    quota = max(1, -(-limit // len(HOP_BUCKETS)))
    taken: dict[str, int] = {label: 0 for label, _, _ in HOP_BUCKETS}
    for row in ordered:
        bucket = _bucket_for(row["hops"])
        if taken.get(bucket, 0) >= quota and len(out) < limit:
            continue                      # this bucket has had its share
        # `hops` is already known from stratified sampling -- use the cheap
        # fixed-length form, and skip the pair entirely if even that is slow.
        n = _n_shortest(client, repo_id, row["source"], row["target"], row["hops"])
        if n != 1:
            skipped_ambiguous += 1
            continue                      # tied, or too expensive to decide
        chain, hops = _shortest(client, repo_id, row["source"], row["target"])
        if chain is None:
            skipped_ambiguous += 1
            continue
        out.append(make_question(
            tier=2, template_id="t2_shortest_chain", repo_id=repo_id, index=len(out),
            question=(
                f"What is the shortest chain of calls from `{row['source']}` to `{row['target']}`? "
                f"List every callable on the path in order, starting with `{row['source']}`."
            ),
            answer=chain, answer_type="path",
            cypher=_GT_SHORTEST_PATH,
            params={"repo": repo_id, "source": row["source"], "target": row["target"]},
            evidence={"hops": hops, "hop_bucket": _bucket_for(hops),
                      "unique_shortest_path": True},
        ))
        taken[bucket] = taken.get(bucket, 0) + 1
        if len(out) >= limit:
            break
    if skipped_ambiguous:
        for rec in out:
            rec.evidence["skipped_tied_pairs_in_sample"] = skipped_ambiguous
    return out


@template("t2_hop_count", 2)
def t2_hop_count(client, repo_id, limit, rng: random.Random):
    rows = sample_paths_stratified(client, repo_id, limit, rng)
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
            evidence={"chain": chain, "hop_bucket": _bucket_for(hops)},
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


#: Build these backwards. Pairing arbitrary callables with arbitrary globals
#: and testing each is both a cross product AND a poor hit rate -- on Django it
#: spent 3.4s to return zero questions, because a random (callable, global) pair
#: almost never has a path between them.
#:
#: Instead: start from a global that is genuinely read by some function, then
#: walk BACKWARDS to the callers that reach it. Every candidate found this way
#: is a true positive by construction, so the template cannot come up empty.
_SAMPLE_GLOBAL_READERS = """
MATCH (f:CodeNode {repo_id: $repo})-[u:USES]->(g:GLOBAL_VARIABLE)
WHERE (f:FUNCTION OR f:METHOD)
RETURN f.qname AS reader, g.qname AS global
ORDER BY g.qname, f.qname
"""

#: Callers that reach `reader` within 6 hops. Bounded: anchored on one node.
_CALLERS_REACHING = """
MATCH (caller:CodeNode {repo_id: $repo})-[:CALLS*0..6]->(f:CodeNode {repo_id: $repo, qname: $reader})
WHERE caller:FUNCTION OR caller:METHOD
RETURN DISTINCT caller.qname AS caller
ORDER BY caller.qname
LIMIT 50
"""

_GT_CALLERS_READING_GLOBAL = """
MATCH (caller:CodeNode {repo_id: $repo, qname: $source})-[:CALLS*0..6]->(f:CodeNode)-[:USES]->(g:GLOBAL_VARIABLE {qname: $global})
RETURN count(DISTINCT f) > 0 AS reaches, collect(DISTINCT f.qname) AS readers
"""


@template("t2_reaches_global_read", 2)
def t2_reaches_global_read(client, repo_id, limit, rng: random.Random):
    """Does a call starting here end up reading a particular global?"""
    readers = client.run(_SAMPLE_GLOBAL_READERS, repo=repo_id)
    if not readers:
        return []
    out = []
    for row in pick(rng, readers, max(limit * 4, 20)):
        if len(out) >= limit:
            break
        callers = [r["caller"] for r in client.run(
            _CALLERS_REACHING, repo=repo_id, reader=row["reader"])]
        if not callers:
            continue
        src = rng.choice(callers)
        res = client.run(_GT_CALLERS_READING_GLOBAL, repo=repo_id,
                         source=src, **{"global": row["global"]})
        if not res or not res[0]["reaches"]:
            continue                      # negatives belong to tier 4
        out.append(make_question(
            tier=2, template_id="t2_reaches_global_read", repo_id=repo_id, index=len(out),
            question=(
                f"Starting from `{src}` and following calls, is the global variable "
                f"`{row['global']}` read anywhere along the way (including in `{src}` itself)?"
            ),
            answer=True, answer_type="bool",
            cypher=_GT_CALLERS_READING_GLOBAL,
            params={"repo": repo_id, "source": src, "global": row["global"]},
            evidence={"readers": sorted(res[0]["readers"])},
        ))
    return out
