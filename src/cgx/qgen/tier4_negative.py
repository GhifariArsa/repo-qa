"""Tier 4 -- counterfactual and negative questions.

The correct answer here is "no", "none", or "not reachable". This tier exists
because a model that hallucinates plausible structure scores well on tiers 1-3
by pattern-matching names, and only a negative tier catches it.

The negatives are deliberately *plausible*: absent entities are built by
recombining a real short name with a real namespace from elsewhere in the same
repo (`minirepo.models.Guest.grant` -- both halves real, the pair is not), and
unreachable pairs are drawn from callables that genuinely exist. A negative
built from a nonsense identifier would be answerable without consulting the
repository at all, which would make the tier worthless.

Every negative is verified against the graph before it is emitted -- if the
Cypher says the entity or path exists, the candidate is discarded rather than
shipped with a wrong label.
"""

from __future__ import annotations

import random

from .base import make_question, template
from ._util import callables, pick

# --------------------------------------------------------------------------
# Absent entities, built by recombining real names
# --------------------------------------------------------------------------

_SAMPLE_NAMESPACES = """
MATCH (n:CodeNode {repo_id: $repo})
WHERE n:MODULE OR n:CLASS
RETURN n.qname AS namespace, [l IN labels(n) WHERE l <> 'CodeNode'][0] AS kind
ORDER BY n.qname
"""

_SAMPLE_SHORT_NAMES = """
MATCH (n:CodeNode {repo_id: $repo})
WHERE n:FUNCTION OR n:METHOD
RETURN DISTINCT n.name AS name
ORDER BY n.name
"""

_GT_ENTITY_EXISTS = """
MATCH (n:CodeNode {repo_id: $repo, qname: $qname})
RETURN count(n) > 0 AS exists
"""


@template("t4_absent_entity", 4)
def t4_absent_entity(client, repo_id, limit, rng: random.Random):
    namespaces = client.run(_SAMPLE_NAMESPACES, repo=repo_id)
    names = [r["name"] for r in client.run(_SAMPLE_SHORT_NAMES, repo=repo_id) if not r["name"].startswith("__")]
    if not namespaces or not names:
        return []
    combos = [(ns["namespace"], ns["kind"], nm) for ns in namespaces for nm in names]
    out = []
    for namespace, kind, name in pick(rng, combos, max(limit * 6, 30)):
        qname = f"{namespace}.{name}"
        res = client.run(_GT_ENTITY_EXISTS, repo=repo_id, qname=qname)
        if res and res[0]["exists"]:
            continue                      # it really exists -- not a negative
        noun = "class" if kind == "CLASS" else "module"
        out.append(make_question(
            tier=4, template_id="t4_absent_entity", repo_id=repo_id, index=len(out),
            question=(
                f"Does this repository define anything named `{qname}` -- that is, a callable "
                f"`{name}` belonging to the {noun} `{namespace}`?"
            ),
            answer=False, answer_type="bool",
            cypher=_GT_ENTITY_EXISTS, params={"repo": repo_id, "qname": qname},
            evidence={"namespace": namespace, "borrowed_name": name,
                      "note": "both halves occur in the repo; the combination does not"},
        ))
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# Unreachable call pairs
# --------------------------------------------------------------------------

_GT_PATH_EXISTS = """
OPTIONAL MATCH p = shortestPath(
    (a:CodeNode {repo_id: $repo, qname: $source})-[:CALLS*1..12]->(b:CodeNode {repo_id: $repo, qname: $target})
)
RETURN p IS NOT NULL AS reachable
"""


@template("t4_no_call_path", 4)
def t4_no_call_path(client, repo_id, limit, rng: random.Random):
    calls = callables(client, repo_id)
    if len(calls) < 2:
        return []
    # Draw pairs at random instead of materialising the full pair space:
    # Django has 9225 callables, so that space is 85,100,625 tuples.
    names = [c["qname"] for c in calls]
    pairs, seen = [], set()
    for _ in range(max(limit * 20, 200)):
        a, b = rng.choice(names), rng.choice(names)
        if a != b and (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b))
    out = []
    for source, target in pairs:
        res = client.run(_GT_PATH_EXISTS, repo=repo_id, source=source, target=target)
        if not res or res[0]["reachable"]:
            continue
        out.append(make_question(
            tier=4, template_id="t4_no_call_path", repo_id=repo_id, index=len(out),
            question=(
                f"Following function and method calls only, can execution starting in "
                f"`{source}` ever reach `{target}`?"
            ),
            answer=False, answer_type="bool",
            cypher=_GT_PATH_EXISTS, params={"repo": repo_id, "source": source, "target": target},
            evidence={"note": "both callables exist; no CALLS path connects them"},
        ))
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# Method attributed to the wrong class
# --------------------------------------------------------------------------

#: Classes and method names are drawn separately and paired in Python. The
#: joined form (every class x every method defined elsewhere) is 12,150,672
#: rows on Django -- ten seconds merely to COUNT, and all of it shipped to the
#: client so ~20 could be kept.
_SAMPLE_CLASSES = """
MATCH (c:CLASS {repo_id: $repo}) RETURN c.qname AS qname ORDER BY c.qname
"""

_SAMPLE_METHOD_NAMES = """
MATCH (:CLASS {repo_id: $repo})-[:HAS_METHOD]->(m:METHOD)
WHERE NOT m.name STARTS WITH '__'
RETURN DISTINCT m.name AS name ORDER BY m.name
"""

_GT_CLASS_HAS_METHOD = """
MATCH (c:CLASS {repo_id: $repo, qname: $class})
OPTIONAL MATCH (c)-[:INHERITS*0..8]->(anc:CLASS)-[:HAS_METHOD]->(m:METHOD {name: $method})
RETURN count(m) > 0 AS has_method
"""


@template("t4_wrong_method_owner", 4)
def t4_wrong_method_owner(client, repo_id, limit, rng: random.Random):
    classes = [r["qname"] for r in client.run(_SAMPLE_CLASSES, repo=repo_id)]
    names = [r["name"] for r in client.run(_SAMPLE_METHOD_NAMES, repo=repo_id)]
    if not classes or not names:
        return []
    out, seen = [], set()
    for _ in range(max(limit * 20, 200)):
        if len(out) >= limit:
            break
        row = {"class": rng.choice(classes), "method": rng.choice(names)}
        key = (row["class"], row["method"])
        if key in seen:
            continue
        seen.add(key)
        res = client.run(_GT_CLASS_HAS_METHOD, repo=repo_id, **{"class": row["class"], "method": row["method"]})
        if not res or res[0]["has_method"]:
            continue                      # it does have it, by definition or inheritance
        out.append(make_question(
            tier=4, template_id="t4_wrong_method_owner", repo_id=repo_id, index=len(out),
            question=(
                f"Can you call `{row['method']}()` on an instance of `{row['class']}` -- that is, "
                f"does that class define or inherit a method named `{row['method']}`?"
            ),
            answer=False, answer_type="bool",
            cypher=_GT_CLASS_HAS_METHOD,
            params={"repo": repo_id, "class": row["class"], "method": row["method"]},
            evidence={"note": "the method name exists on a different class in this repo"},
        ))
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# Dead code: callables nothing calls
# --------------------------------------------------------------------------

_GT_CALLERS = """
MATCH (t:CodeNode {repo_id: $repo, qname: $target})
OPTIONAL MATCH (c:CodeNode)-[:CALLS]->(t)
RETURN collect(DISTINCT c.qname) AS callers
"""


@template("t4_no_callers", 4)
def t4_no_callers(client, repo_id, limit, rng: random.Random):
    rows = callables(client, repo_id)
    out = []
    for row in pick(rng, rows, max(limit * 4, 20)):
        res = client.run(_GT_CALLERS, repo=repo_id, target=row["qname"])[0]
        callers = sorted(c for c in res["callers"] if c)
        if callers:
            continue
        out.append(make_question(
            tier=4, template_id="t4_no_callers", repo_id=repo_id, index=len(out),
            question=(
                f"Which functions or methods in this repository call `{row['qname']}`? "
                f"If nothing calls it, say so explicitly."
            ),
            answer=[], answer_type="list",
            cypher=_GT_CALLERS, params={"repo": repo_id, "target": row["qname"]},
            evidence={"note": "no incoming CALLS edge in the extracted graph"},
        ))
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# Non-existent inheritance
# --------------------------------------------------------------------------

_GT_INHERITS = """
OPTIONAL MATCH p = (a:CLASS {repo_id: $repo, qname: $a})-[:INHERITS*1..8]->(b:CLASS {repo_id: $repo, qname: $b})
RETURN p IS NOT NULL AS inherits
"""


@template("t4_not_inherits", 4)
def t4_not_inherits(client, repo_id, limit, rng: random.Random):
    classes = [r["qname"] for r in client.run(
        "MATCH (c:CLASS {repo_id: $repo}) RETURN c.qname AS qname ORDER BY c.qname", repo=repo_id)]
    pairs = [(a, b) for a in classes for b in classes if a != b]
    out = []
    for a, b in pick(rng, pairs, max(limit * 4, 20)):
        res = client.run(_GT_INHERITS, repo=repo_id, a=a, b=b)
        if not res or res[0]["inherits"]:
            continue
        out.append(make_question(
            tier=4, template_id="t4_not_inherits", repo_id=repo_id, index=len(out),
            question=f"Is `{a}` a subclass of `{b}`, directly or through any chain of base classes?",
            answer=False, answer_type="bool",
            cypher=_GT_INHERITS, params={"repo": repo_id, "a": a, "b": b},
            evidence={"note": "both classes exist; no INHERITS chain connects them"},
        ))
        if len(out) >= limit:
            break
    return out
