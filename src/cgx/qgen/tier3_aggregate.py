"""Tier 3 -- aggregation over graph shape.

Fan-in, fan-out, transitive reachability size, nearest common ancestor. These
require holding the whole graph in view rather than following one path, which
is precisely where file-at-a-time retrieval degrades.
"""

from __future__ import annotations

import random

from .base import make_question, template
from ._util import callables, pick

#: Minimum metric value for a callable to be worth asking about.
#:
#: Sampling callables uniformly makes this tier mostly trivial, because the
#: metrics are heavily zero-inflated. Measured over all callables:
#:
#:     repo          fan_in>=2   fan_out>=2   reach>=2
#:     minirepo (21)    14.3%       14.3%       28.6%
#:     neo4j    (1765)  17.5%       23.3%       32.8%
#:
#: Class method counts are less skewed but degenerate often enough to matter --
#: 30% of Django's 1957 classes define no methods of their own at all (marker
#: subclasses, lookups, `Meta`-style attribute holders), and another 19% define
#: exactly one:
#:
#:     repo               methods>=2
#:     minirepo (5)          80.0%
#:     neo4j    (302)        57.9%
#:     django   (1957)       51.3%
#:
#: On the real repo, ~65% of callables have fan-in 0 -- so two thirds of a
#: uniform sample would ask "how many callers?" about something nothing calls.
#: A threshold of 2 still leaves 308-579 candidates there, which is ample.
#: Zero-valued cases are not lost: they are tier 4's `t4_no_callers`, where
#: "nothing calls this" is the point rather than filler.
#:
#: `t3_common_ancestor` already filters this way -- its sampling query only
#: matches class pairs that genuinely share an ancestor.
MIN_FAN_IN = 2
MIN_FAN_OUT = 2
MIN_REACH = 2
MIN_METHOD_COUNT = 2

#: Candidates whose metric clears a threshold, computed in one pass rather than
#: per sampled node.
_NONTRIVIAL_FAN_IN = """
MATCH (n:CodeNode {repo_id: $repo})
WHERE n:FUNCTION OR n:METHOD
OPTIONAL MATCH (c:CodeNode)-[:CALLS]->(n)
WITH n, count(DISTINCT c) AS metric
WHERE metric >= $threshold
RETURN n.qname AS qname, metric
ORDER BY n.qname
"""

_NONTRIVIAL_FAN_OUT = """
MATCH (n:CodeNode {repo_id: $repo})
WHERE n:FUNCTION OR n:METHOD
OPTIONAL MATCH (n)-[:CALLS]->(t:CodeNode)
WITH n, count(DISTINCT t) AS metric
WHERE metric >= $threshold
RETURN n.qname AS qname, metric
ORDER BY n.qname
"""

#: Counts methods defined in the class's own body, matching what the question
#: asks -- inherited methods are deliberately not counted.
_NONTRIVIAL_METHOD_COUNT = """
MATCH (c:CLASS {repo_id: $repo})
OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:METHOD)
WITH c, count(DISTINCT m) AS metric
WHERE metric >= $threshold
RETURN c.qname AS qname, metric
ORDER BY c.qname
"""

_NONTRIVIAL_REACH = """
MATCH (n:CodeNode {repo_id: $repo})
WHERE n:FUNCTION OR n:METHOD
OPTIONAL MATCH (n)-[:CALLS*1..12]->(t:CodeNode)
WHERE t <> n
WITH n, count(DISTINCT t) AS metric
WHERE metric >= $threshold
RETURN n.qname AS qname, metric
ORDER BY n.qname
"""


def _nontrivial(client, repo_id, cypher, threshold, rng, limit):
    """Sample from candidates clearing `threshold`.

    Returns fewer than `limit` when the repo simply has that few non-trivial
    candidates -- it does not quietly relax the threshold, since falling back
    to trivial nodes is the behaviour this filter exists to prevent.
    """
    rows = client.run(cypher, repo=repo_id, threshold=threshold)
    return pick(rng, rows, limit)

_GT_FAN_IN = """
MATCH (t:CodeNode {repo_id: $repo, qname: $target})
OPTIONAL MATCH (c:CodeNode)-[:CALLS]->(t)
RETURN count(DISTINCT c) AS fan_in, collect(DISTINCT c.qname) AS callers
"""


@template("t3_fan_in", 3)
def t3_fan_in(client, repo_id, limit, rng: random.Random):
    rows = _nontrivial(client, repo_id, _NONTRIVIAL_FAN_IN, MIN_FAN_IN, rng, limit)
    out = []
    for i, row in enumerate(rows):
        res = client.run(_GT_FAN_IN, repo=repo_id, target=row["qname"])[0]
        out.append(make_question(
            tier=3, template_id="t3_fan_in", repo_id=repo_id, index=i,
            question=(
                f"How many distinct functions or methods in this repository call "
                f"`{row['qname']}` directly? Answer with a single integer."
            ),
            answer=res["fan_in"], answer_type="count",
            cypher=_GT_FAN_IN, params={"repo": repo_id, "target": row["qname"]},
            evidence={"callers": sorted(c for c in res["callers"] if c),
                      "sampled_from": f"fan_in >= {MIN_FAN_IN}"},
        ))
    return out


_GT_FAN_OUT = """
MATCH (s:CodeNode {repo_id: $repo, qname: $source})
OPTIONAL MATCH (s)-[:CALLS]->(t:CodeNode)
RETURN count(DISTINCT t) AS fan_out, collect(DISTINCT t.qname) AS callees
"""


@template("t3_fan_out", 3)
def t3_fan_out(client, repo_id, limit, rng: random.Random):
    rows = _nontrivial(client, repo_id, _NONTRIVIAL_FAN_OUT, MIN_FAN_OUT, rng, limit)
    out = []
    for i, row in enumerate(rows):
        res = client.run(_GT_FAN_OUT, repo=repo_id, source=row["qname"])[0]
        out.append(make_question(
            tier=3, template_id="t3_fan_out", repo_id=repo_id, index=i,
            question=(
                f"How many distinct functions or methods defined in this repository does "
                f"`{row['qname']}` call directly? Answer with a single integer."
            ),
            answer=res["fan_out"], answer_type="count",
            cypher=_GT_FAN_OUT, params={"repo": repo_id, "source": row["qname"]},
            evidence={"callees": sorted(c for c in res["callees"] if c),
                      "sampled_from": f"fan_out >= {MIN_FAN_OUT}"},
        ))
    return out


_GT_REACH_SET = """
MATCH (s:CodeNode {repo_id: $repo, qname: $source})
OPTIONAL MATCH (s)-[:CALLS*1..12]->(t:CodeNode)
WHERE t <> s
RETURN count(DISTINCT t) AS reach, collect(DISTINCT t.qname) AS reachable
"""


@template("t3_reach_set_size", 3)
def t3_reach_set_size(client, repo_id, limit, rng: random.Random):
    rows = _nontrivial(client, repo_id, _NONTRIVIAL_REACH, MIN_REACH, rng, limit)
    out = []
    for i, row in enumerate(rows):
        res = client.run(_GT_REACH_SET, repo=repo_id, source=row["qname"])[0]
        out.append(make_question(
            tier=3, template_id="t3_reach_set_size", repo_id=repo_id, index=i,
            question=(
                f"Counting every function or method reachable from `{row['qname']}` through any "
                f"number of call steps, how many distinct callables can it reach? "
                f"Answer with a single integer."
            ),
            answer=res["reach"], answer_type="count",
            cypher=_GT_REACH_SET, params={"repo": repo_id, "source": row["qname"]},
            evidence={"reachable": sorted(c for c in res["reachable"] if c),
                      "sampled_from": f"reach >= {MIN_REACH}"},
        ))
    return out


_SAMPLE_CLASS_PAIRS = """
MATCH (a:CLASS {repo_id: $repo})-[:INHERITS*1..8]->(anc:CLASS)<-[:INHERITS*1..8]-(b:CLASS {repo_id: $repo})
WHERE a.qname < b.qname
RETURN DISTINCT a.qname AS a, b.qname AS b
ORDER BY a, b
"""

_GT_COMMON_ANCESTOR = """
MATCH pa = (a:CLASS {repo_id: $repo, qname: $a})-[:INHERITS*1..8]->(anc:CLASS)
MATCH pb = (b:CLASS {repo_id: $repo, qname: $b})-[:INHERITS*1..8]->(anc)
WITH anc, length(pa) + length(pb) AS dist
ORDER BY dist ASC, anc.qname ASC
RETURN collect(anc.qname)[0] AS ancestor, min(dist) AS distance
"""


@template("t3_common_ancestor", 3)
def t3_common_ancestor(client, repo_id, limit, rng: random.Random):
    rows = pick(rng, client.run(_SAMPLE_CLASS_PAIRS, repo=repo_id), limit)
    out = []
    for i, row in enumerate(rows):
        res = client.run(_GT_COMMON_ANCESTOR, repo=repo_id, a=row["a"], b=row["b"])
        if not res or res[0]["ancestor"] is None:
            continue
        out.append(make_question(
            tier=3, template_id="t3_common_ancestor", repo_id=repo_id, index=len(out),
            question=(
                f"What is the nearest class that both `{row['a']}` and `{row['b']}` inherit from, "
                f"directly or indirectly?"
            ),
            answer=res[0]["ancestor"], answer_type="string",
            cypher=_GT_COMMON_ANCESTOR, params={"repo": repo_id, "a": row["a"], "b": row["b"]},
            evidence={"combined_distance": res[0]["distance"]},
        ))
    return out


_GT_MAX_FAN_IN = """
MATCH (c:CodeNode {repo_id: $repo})-[:CALLS]->(t:CodeNode)
WITH t, count(DISTINCT c) AS fan_in
WITH max(fan_in) AS top
MATCH (c2:CodeNode {repo_id: $repo})-[:CALLS]->(t2:CodeNode)
WITH t2, count(DISTINCT c2) AS fan_in, top
WHERE fan_in = top
RETURN collect(DISTINCT t2.qname) AS most_called, top AS fan_in
"""


@template("t3_max_fan_in", 3)
def t3_max_fan_in(client, repo_id, limit, rng: random.Random):
    res = client.run(_GT_MAX_FAN_IN, repo=repo_id)
    if not res or not res[0]["most_called"]:
        return []
    answer = sorted(res[0]["most_called"])
    return [make_question(
        tier=3, template_id="t3_max_fan_in", repo_id=repo_id, index=0,
        question=(
            "Which function or method in this repository is called directly by the largest "
            "number of distinct callers? If several tie, list all of them."
        ),
        answer=answer, answer_type="list",
        cypher=_GT_MAX_FAN_IN, params={"repo": repo_id},
        evidence={"fan_in": res[0]["fan_in"]},
    )]


_GT_CLASS_METHOD_COUNT = """
MATCH (c:CLASS {repo_id: $repo, qname: $class})
OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:METHOD)
RETURN count(DISTINCT m) AS n, collect(DISTINCT m.name) AS methods
"""

@template("t3_class_method_count", 3)
def t3_class_method_count(client, repo_id, limit, rng: random.Random):
    rows = _nontrivial(client, repo_id, _NONTRIVIAL_METHOD_COUNT, MIN_METHOD_COUNT, rng, limit)
    out = []
    for i, row in enumerate(rows):
        res = client.run(_GT_CLASS_METHOD_COUNT, repo=repo_id, **{"class": row["qname"]})[0]
        out.append(make_question(
            tier=3, template_id="t3_class_method_count", repo_id=repo_id, index=i,
            question=(
                f"How many methods does the class `{row['qname']}` define in its own body "
                f"(not counting methods it inherits)? Answer with a single integer."
            ),
            answer=res["n"], answer_type="count",
            cypher=_GT_CLASS_METHOD_COUNT, params={"repo": repo_id, "class": row["qname"]},
            evidence={"methods": sorted(m for m in res["methods"] if m),
                      "sampled_from": f"method_count >= {MIN_METHOD_COUNT}"},
        ))
    return out
