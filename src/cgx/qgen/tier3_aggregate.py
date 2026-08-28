"""Tier 3 -- aggregation over graph shape.

Fan-in, fan-out, transitive reachability size, nearest common ancestor. These
require holding the whole graph in view rather than following one path, which
is precisely where file-at-a-time retrieval degrades.
"""

from __future__ import annotations

import random

from .base import make_question, template
from ._util import callables, pick

_GT_FAN_IN = """
MATCH (t:CodeNode {repo_id: $repo, qname: $target})
OPTIONAL MATCH (c:CodeNode)-[:CALLS]->(t)
RETURN count(DISTINCT c) AS fan_in, collect(DISTINCT c.qname) AS callers
"""


@template("t3_fan_in", 3)
def t3_fan_in(client, repo_id, limit, rng: random.Random):
    rows = pick(rng, callables(client, repo_id), limit)
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
            evidence={"callers": sorted(c for c in res["callers"] if c)},
        ))
    return out


_GT_FAN_OUT = """
MATCH (s:CodeNode {repo_id: $repo, qname: $source})
OPTIONAL MATCH (s)-[:CALLS]->(t:CodeNode)
RETURN count(DISTINCT t) AS fan_out, collect(DISTINCT t.qname) AS callees
"""


@template("t3_fan_out", 3)
def t3_fan_out(client, repo_id, limit, rng: random.Random):
    rows = pick(rng, callables(client, repo_id), limit)
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
            evidence={"callees": sorted(c for c in res["callees"] if c)},
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
    rows = pick(rng, callables(client, repo_id), limit)
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
            evidence={"reachable": sorted(c for c in res["reachable"] if c)},
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

_SAMPLE_CLASSES = """
MATCH (c:CLASS {repo_id: $repo}) RETURN c.qname AS qname ORDER BY c.qname
"""


@template("t3_class_method_count", 3)
def t3_class_method_count(client, repo_id, limit, rng: random.Random):
    rows = pick(rng, client.run(_SAMPLE_CLASSES, repo=repo_id), limit)
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
            evidence={"methods": sorted(m for m in res["methods"] if m)},
        ))
    return out
