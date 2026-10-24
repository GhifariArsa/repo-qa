"""Tier 4 -- counterfactual questions, answered yes or no in equal measure.

These templates exist to catch a model that hallucinates plausible structure,
which scores well elsewhere by pattern-matching names. They used to be
negative-only, which made every answer "no" and every question answerable from
its wording alone: the closed-book pilot scored 1.00 on all of them. Each
yes/no template now splits its quota evenly (`split_quota`), and the "yes" side
is chosen to be as hard as the "no" side -- a name shared by several entities, a
path with no direct call, a transitive subclass, an inherited method -- so that
neither answer can be read off the surface of the question. `t4_no_callers`
lists callers, so an empty answer carries no such tell, and stays as it was.

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
from ._util import callables, pick, rebalance, split_quota

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


#: Real callables whose short name is shared by at least one other entity, so
#: "that name exists somewhere in the repo" does not imply "yes".
_SAMPLE_SHARED_NAME_ENTITIES = """
MATCH (n:CodeNode {repo_id: $repo})
WHERE (n:FUNCTION OR n:METHOD) AND NOT n.name STARTS WITH '__'
WITH n.name AS name, collect(n) AS owners
WHERE size(owners) >= 2
UNWIND owners AS n
WITH n, name, CASE WHEN n:METHOD THEN n.class_qname ELSE n.module_qname END AS namespace
WHERE namespace IS NOT NULL AND n.qname = namespace + '.' + name
RETURN n.qname AS qname, name, namespace, CASE WHEN n:METHOD THEN 'CLASS' ELSE 'MODULE' END AS kind
ORDER BY qname
"""


def _absent_entity_question(client, repo_id, index, namespace, kind, name):
    qname = f"{namespace}.{name}"
    exists = client.run(_GT_ENTITY_EXISTS, repo=repo_id, qname=qname)[0]["exists"]
    noun = "class" if kind == "CLASS" else "module"
    return make_question(
        tier=4, template_id="t4_absent_entity", repo_id=repo_id, index=index,
        question=(
            f"Does this repository define anything named `{qname}` -- that is, a callable "
            f"`{name}` belonging to the {noun} `{namespace}`?"
        ),
        answer=bool(exists), answer_type="bool",
        cypher=_GT_ENTITY_EXISTS, params={"repo": repo_id, "qname": qname},
        evidence={"namespace": namespace, "name": name,
                  "polarity": "real_entity_with_shared_name" if exists else "recombined_near_miss"},
    )


@template("t4_absent_entity", 4)
def t4_absent_entity(client, repo_id, limit, rng: random.Random):
    n_yes, n_no = split_quota(limit, rng)
    out = []
    for row in pick(rng, client.run(_SAMPLE_SHARED_NAME_ENTITIES, repo=repo_id), n_yes):
        rec = _absent_entity_question(client, repo_id, len(out), row["namespace"], row["kind"], row["name"])
        if rec.answer is True:
            out.append(rec)

    namespaces = client.run(_SAMPLE_NAMESPACES, repo=repo_id)
    names = [r["name"] for r in client.run(_SAMPLE_SHORT_NAMES, repo=repo_id) if not r["name"].startswith("__")]
    if not namespaces or not names:
        return out
    combos = [(ns["namespace"], ns["kind"], nm) for ns in namespaces for nm in names]
    added = 0
    for namespace, kind, name in pick(rng, combos, max(limit * 6, 30)):
        if added >= n_no:
            break
        rec = _absent_entity_question(client, repo_id, len(out), namespace, kind, name)
        if rec.answer is False:           # both halves real, the combination not
            out.append(rec)
            added += 1
    return rebalance(out)


# --------------------------------------------------------------------------
# Unreachable call pairs
# --------------------------------------------------------------------------

_GT_PATH_EXISTS = """
OPTIONAL MATCH p = shortestPath(
    (a:CodeNode {repo_id: $repo, qname: $source})-[:CALLS*1..12]->(b:CodeNode {repo_id: $repo, qname: $target})
)
RETURN p IS NOT NULL AS reachable
"""


#: Targets 2-4 calls away from a source with no direct call between them, so
#: a single grep of the source's body does not settle "yes".
_REACHABLE_MULTI_HOP = """
MATCH (a:CodeNode {repo_id: $repo, qname: $source})-[:CALLS*2..4]->(b:CodeNode)
WHERE (b:FUNCTION OR b:METHOD) AND b <> a AND NOT (a)-[:CALLS]->(b)
RETURN DISTINCT b.qname AS target
ORDER BY target
LIMIT 50
"""


#: Is everything reachable from this source free of possibly missed calls?
#: "No path exists" is only a trustworthy answer when it is (qgen/completeness.py).
_CLOSURE_COMPLETE = """
MATCH (s:CodeNode {repo_id: $repo, qname: $q})-[:CALLS*0..12]->(n)
WHERE coalesce(n.unresolved_out, 0) > 0
RETURN count(DISTINCT n) = 0 AS complete
"""


#: Callables in the files the source's call closure passes through that the
#: source still cannot reach: the near-miss "no". A target drawn from anywhere
#: in the repo usually sits in an unrelated module, and closed-book models
#: scored 0.60-0.67 on yes/no from the names alone.
_NEAR_MISS_TARGETS = """
MATCH (a:CodeNode {repo_id: $repo, qname: $q})-[:CALLS*0..12]->(n)
WITH collect(DISTINCT n.file_path) AS files, collect(DISTINCT n) AS reach
MATCH (b:CodeNode {repo_id: $repo})
WHERE (b:FUNCTION OR b:METHOD) AND b.file_path IN files AND NOT b IN reach
RETURN b.qname AS target ORDER BY target LIMIT 200
"""


def sample_unreachable_pairs(client, repo_id, n, rng: random.Random) -> list[tuple[str, str]]:
    """(source, target) callables with no CALLS path between them.

    Targets are near misses where possible: defined in a file the source's
    calls already pass through, so the names suggest a connection and only
    following the calls shows there is none. After half the attempts the
    target may come from anywhere, so a repo of leaf functions still yields
    its quota. Evidence `polarity` records which kind each pair is.

    Drawn at random rather than from the full pair space: Django has 9225
    callables, so that space is 85,100,625 tuples. Sources are restricted to
    those whose whole call closure is free of possibly missed calls; otherwise
    the completeness filter drops nearly every "no" and the template is all
    "yes" again.
    """
    return [(a, b) for a, b, _ in sample_unreachable_pairs_tagged(client, repo_id, n, rng)]


def sample_unreachable_pairs_tagged(client, repo_id, n, rng: random.Random) -> list[tuple[str, str, str]]:
    names = [c["qname"] for c in callables(client, repo_id)]
    out, seen = [], set()
    if len(names) < 2 or n < 1:
        return out
    complete: dict[str, bool] = {}
    near: dict[str, list[str]] = {}
    attempts = max(n * 40, 400)
    for attempt in range(attempts):
        if len(out) >= n:
            break
        a = rng.choice(names)
        if a not in complete:
            res = client.run_with_timeout(_CLOSURE_COMPLETE, 5.0, repo=repo_id, q=a)
            complete[a] = bool(res and res[0]["complete"])
        if not complete[a]:
            continue
        kind = "near_miss"
        if attempt < attempts // 2:
            if a not in near:
                res = client.run_with_timeout(_NEAR_MISS_TARGETS, 5.0, repo=repo_id, q=a)
                near[a] = [r["target"] for r in res or []]
            if not near[a]:
                continue
            b = rng.choice(near[a])
        else:
            b, kind = rng.choice(names), "unreachable"
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        res = client.run(_GT_PATH_EXISTS, repo=repo_id, source=a, target=b)
        if res and not res[0]["reachable"]:
            out.append((a, b, kind))
    return out


def sample_multi_hop_pairs(client, repo_id, n, rng: random.Random) -> list[tuple[str, str]]:
    """(source, target) callables connected only through at least one intermediary."""
    names = [c["qname"] for c in callables(client, repo_id)]
    out, seen = [], set()
    if n < 1:
        return out
    for source in pick(rng, names, max(n * 30, 300)):
        if len(out) >= n:
            break
        targets = [r["target"] for r in client.run(_REACHABLE_MULTI_HOP, repo=repo_id, source=source)]
        if not targets:
            continue
        target = rng.choice(targets)
        if (source, target) not in seen:
            seen.add((source, target))
            out.append((source, target))
    return out


def reachability_question(client, repo_id, template_id, tier, index, source, target, evidence=None):
    reachable = client.run(_GT_PATH_EXISTS, repo=repo_id, source=source, target=target)[0]["reachable"]
    return make_question(
        tier=tier, template_id=template_id, repo_id=repo_id, index=index,
        question=(
            f"Following function and method calls only, can execution starting in "
            f"`{source}` ever reach `{target}`?"
        ),
        answer=bool(reachable), answer_type="bool",
        cypher=_GT_PATH_EXISTS, params={"repo": repo_id, "source": source, "target": target},
        evidence=evidence or {"polarity": "multi_hop_reachable" if reachable else "unreachable"},
    )


@template("t4_no_call_path", 4)
def t4_no_call_path(client, repo_id, limit, rng: random.Random):
    n_yes, n_no = split_quota(limit, rng)
    pairs = [(s, t, "multi_hop_reachable") for s, t in sample_multi_hop_pairs(client, repo_id, n_yes, rng)]
    pairs += sample_unreachable_pairs_tagged(client, repo_id, n_no, rng)
    return rebalance([reachability_question(client, repo_id, "t4_no_call_path", 4, i, s, t,
                                            evidence={"polarity": kind})
                      for i, (s, t, kind) in enumerate(pairs)])


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


#: Methods a class inherits but does not declare itself: "yes", but grepping
#: the class body for the name finds nothing.
_INHERITED_ONLY_METHODS = """
MATCH (c:CLASS {repo_id: $repo})-[:INHERITS*1..8]->(:CLASS)-[:HAS_METHOD]->(m:METHOD)
WHERE NOT m.name STARTS WITH '__'
  AND NOT EXISTS { MATCH (c)-[:HAS_METHOD]->(own:METHOD) WHERE own.name = m.name }
RETURN DISTINCT c.qname AS class, m.name AS method
ORDER BY class, method
LIMIT 5000
"""


def _method_owner_question(client, repo_id, index, cls, method, polarity):
    res = client.run(_GT_CLASS_HAS_METHOD, repo=repo_id, **{"class": cls, "method": method})
    return make_question(
        tier=4, template_id="t4_wrong_method_owner", repo_id=repo_id, index=index,
        question=(
            f"Can you call `{method}()` on an instance of `{cls}` -- that is, "
            f"does that class define or inherit a method named `{method}`?"
        ),
        answer=bool(res and res[0]["has_method"]), answer_type="bool",
        cypher=_GT_CLASS_HAS_METHOD,
        params={"repo": repo_id, "class": cls, "method": method},
        evidence={"polarity": polarity},
    )


@template("t4_wrong_method_owner", 4)
def t4_wrong_method_owner(client, repo_id, limit, rng: random.Random):
    n_yes, n_no = split_quota(limit, rng)
    out = []
    for row in pick(rng, client.run(_INHERITED_ONLY_METHODS, repo=repo_id), n_yes):
        rec = _method_owner_question(client, repo_id, len(out), row["class"], row["method"],
                                     "inherited_not_declared")
        if rec.answer is True:
            out.append(rec)
    limit = len(out) + n_no
    classes = [r["qname"] for r in client.run(_SAMPLE_CLASSES, repo=repo_id)]
    names = [r["name"] for r in client.run(_SAMPLE_METHOD_NAMES, repo=repo_id)]
    if not classes or not names:
        return out
    seen = set()
    for _ in range(max(limit * 20, 200)):
        if len(out) >= limit:
            break
        row = {"class": rng.choice(classes), "method": rng.choice(names)}
        key = (row["class"], row["method"])
        if key in seen:
            continue
        seen.add(key)
        rec = _method_owner_question(client, repo_id, len(out), row["class"], row["method"],
                                     "name_exists_on_another_class")
        if rec.answer is False:           # it does not have it, by definition or inheritance
            out.append(rec)
    return rebalance(out)


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


#: Subclasses reached only through an intermediate base, so the answer is not
#: sitting in the class's own `class A(B)` line. Direct pairs are the fallback
#: for repositories whose hierarchies are all one level deep.
_TRANSITIVE_SUBCLASSES = """
MATCH (a:CLASS {repo_id: $repo})-[:INHERITS*2..8]->(b:CLASS {repo_id: $repo})
WHERE a <> b AND NOT (a)-[:INHERITS]->(b)
RETURN DISTINCT a.qname AS a, b.qname AS b ORDER BY a, b LIMIT 5000
"""
_DIRECT_SUBCLASSES = """
MATCH (a:CLASS {repo_id: $repo})-[:INHERITS]->(b:CLASS {repo_id: $repo})
RETURN DISTINCT a.qname AS a, b.qname AS b ORDER BY a, b LIMIT 5000
"""


def _inherits_question(client, repo_id, index, a, b):
    inherits = client.run(_GT_INHERITS, repo=repo_id, a=a, b=b)[0]["inherits"]
    return make_question(
        tier=4, template_id="t4_not_inherits", repo_id=repo_id, index=index,
        question=f"Is `{a}` a subclass of `{b}`, directly or through any chain of base classes?",
        answer=bool(inherits), answer_type="bool",
        cypher=_GT_INHERITS, params={"repo": repo_id, "a": a, "b": b},
        evidence={"polarity": "subclass" if inherits else "unrelated_classes"},
    )


@template("t4_not_inherits", 4)
def t4_not_inherits(client, repo_id, limit, rng: random.Random):
    n_yes, n_no = split_quota(limit, rng)
    positives = client.run(_TRANSITIVE_SUBCLASSES, repo=repo_id)
    if len(positives) < n_yes:
        positives += client.run(_DIRECT_SUBCLASSES, repo=repo_id)
    out = []
    for row in pick(rng, positives, n_yes):
        rec = _inherits_question(client, repo_id, len(out), row["a"], row["b"])
        if rec.answer is True:
            out.append(rec)

    classes = [r["qname"] for r in client.run(
        "MATCH (c:CLASS {repo_id: $repo}) RETURN c.qname AS qname ORDER BY c.qname", repo=repo_id)]
    added, seen = 0, set()
    for _ in range(max(n_no * 20, 200)):
        if added >= n_no or len(classes) < 2:
            break
        a, b = rng.choice(classes), rng.choice(classes)
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        rec = _inherits_question(client, repo_id, len(out), a, b)
        if rec.answer is False:
            out.append(rec)
            added += 1
    return rebalance(out)
