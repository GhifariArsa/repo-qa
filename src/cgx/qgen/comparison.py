"""Compare explicit sets, with bounded candidate sampling and query-backed gold.

No repository-wide pair cross product is materialised. Candidate selection
attempts a roughly even split of empty and non-empty answers; shortages are
visible in generation counts. Sets are compared by fully qualified callable
name, or by short method name for declarations on two different classes.
"""

import random

from .base import make_question, template

_CALL_SETS = """
MATCH (a:CodeNode {repo_id: $repo})
WHERE a:FUNCTION OR a:METHOD
OPTIONAL MATCH (a)-[:CALLS]->(t:CodeNode {repo_id: $repo})
RETURN a.qname AS qname, collect(DISTINCT t.qname) AS members
ORDER BY qname
"""
_MEMBER_SETS = """
MATCH (a:CLASS {repo_id: $repo})
OPTIONAL MATCH (a)-[:HAS_METHOD]->(m:METHOD {repo_id: $repo})
RETURN a.qname AS qname, collect(DISTINCT m.name) AS members
ORDER BY qname
"""

_CALLEE_QUERY = """
MATCH (a:CodeNode {repo_id: $repo, qname: $a})
WHERE a:FUNCTION OR a:METHOD
MATCH (b:CodeNode {repo_id: $repo, qname: $b})
WHERE b:FUNCTION OR b:METHOD
OPTIONAL MATCH (a)-[:CALLS]->(x:CodeNode {repo_id: $repo})
WITH b, collect(DISTINCT x.qname) AS left_members
OPTIONAL MATCH (b)-[:CALLS]->(y:CodeNode {repo_id: $repo})
WITH left_members, collect(DISTINCT y.qname) AS right_members
RETURN [name IN left_members WHERE %s] AS answer,
       left_members, right_members
"""
_MEMBER_QUERY = """
MATCH (a:CLASS {repo_id: $repo, qname: $a}), (b:CLASS {repo_id: $repo, qname: $b})
OPTIONAL MATCH (a)-[:HAS_METHOD]->(x:METHOD)
WITH b, collect(DISTINCT x.name) AS left_members
OPTIONAL MATCH (b)-[:HAS_METHOD]->(y:METHOD)
WITH left_members, collect(DISTINCT y.name) AS right_members
RETURN [name IN left_members WHERE name IN right_members] AS answer,
       left_members, right_members
"""


def _candidate_pairs(rows, limit, rng, difference=False):
    """Bound work by the requested sample size, not n-squared storage."""
    if len(rows) < 2 or limit < 1:
        return []
    positive, negative, seen = [], [], set()
    target_positive, target_negative = (limit + 1) // 2, limit // 2
    for _ in range(max(200, limit * 50)):
        a, b = rng.sample(rows, 2)
        if not difference and a["qname"] > b["qname"]:
            a, b = b, a
        key = (a["qname"], b["qname"])
        if key in seen:
            continue
        seen.add(key)
        left, right = set(a["members"]), set(b["members"])
        # Empty comparisons should still involve real non-empty sets.
        if not left or not right:
            continue
        answer = left - right if difference else left & right
        bucket, quota = (positive, target_positive) if answer else (negative, target_negative)
        if len(bucket) < quota:
            bucket.append(key)
        if len(positive) == target_positive and len(negative) == target_negative:
            break
    # Random pairs from a sparse graph almost never share a member, and almost
    # always differ, so the rare outcome -- a non-empty intersection, or an
    # empty difference -- starves. Left there, the family skews toward one
    # answer and "none" becomes a winning guess. Build the rare bucket from an
    # inverted member -> owners index instead: O(total members), still no
    # pair cross product.
    rare, rare_quota = (positive, target_positive) if not difference else (negative, target_negative)
    if len(rare) < rare_quota:
        by_qname = {r["qname"]: set(r["members"]) for r in rows if r["members"]}
        owners: dict[str, list[str]] = {}
        for qname, members in by_qname.items():
            for m in members:
                owners.setdefault(m, []).append(qname)
        shared = sorted(m for m, os in owners.items() if len(os) >= 2)
        for _ in range(max(200, limit * 50)):
            if len(rare) >= rare_quota or not shared:
                break
            if not difference:
                a, b = sorted(rng.sample(owners[rng.choice(shared)], 2))
            else:
                a = rng.choice(owners[rng.choice(shared)])
                pool = sorted({o for m in by_qname[a] for o in owners[m]} - {a})
                if not pool:
                    continue
                b = rng.choice(pool)
                if not by_qname[a] <= by_qname[b]:
                    continue
            key = (a, b)
            if key in seen or a == b:
                continue
            seen.add(key)
            rare.append(key)
    pairs = positive + negative
    rng.shuffle(pairs)
    return pairs


def _generate(client, repo_id, limit, rng, tid, sample_query, query, wording, difference=False):
    rows = client.run(sample_query, repo=repo_id)
    out = []
    for a, b in _candidate_pairs(rows, limit, rng, difference):
        params = {"repo": repo_id, "a": a, "b": b}
        result = client.run(query, **params)
        if not result:
            continue
        row = result[0]
        out.append(make_question(
            template_id=tid, repo_id=repo_id, index=len(out),
            question=wording.format(a=a, b=b),
            answer=sorted(row["answer"]), answer_type="list",
            cypher=query, params=params,
            evidence={"left_members": sorted(row["left_members"]),
                      "right_members": sorted(row["right_members"]),
                      "comparison": "difference" if difference else "intersection",
                      "scope": "in-repository direct relationships",
                      "sampled_from": "both input sets non-empty; balanced outcome targets"},
        ))
    return out


@template("comparison_shared_callees")
def shared_callees(client, repo_id, limit, rng: random.Random):
    return _generate(client, repo_id, limit, rng, "comparison_shared_callees", _CALL_SETS,
                     _CALLEE_QUERY % "name IN right_members",
                     "Which in-repository functions or methods are called directly by both `{a}` "
                     "and `{b}`? List their fully qualified names, or say none.")


@template("comparison_exclusive_callees")
def exclusive_callees(client, repo_id, limit, rng: random.Random):
    return _generate(client, repo_id, limit, rng, "comparison_exclusive_callees", _CALL_SETS,
                     _CALLEE_QUERY % "NOT (name IN right_members)",
                     "Which in-repository functions or methods does `{a}` call directly that "
                     "`{b}` does not call directly? List their fully qualified names, or say none.",
                     difference=True)


@template("comparison_shared_members")
def shared_members(client, repo_id, limit, rng: random.Random):
    return _generate(client, repo_id, limit, rng, "comparison_shared_members", _MEMBER_SETS,
                     _MEMBER_QUERY,
                     "Which method names are declared directly in both `{a}` and `{b}`? "
                     "Exclude inherited methods. Give short method names, or say none.")
