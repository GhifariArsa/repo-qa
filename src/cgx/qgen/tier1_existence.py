"""Tier 1 -- existence and lookup.

Single-node or single-edge questions: does X exist, what kind of thing is X,
which class owns X, where is X defined. These are the floor of the benchmark;
an agent that cannot answer these cannot answer anything above them.
"""

from __future__ import annotations

import random

from .base import make_question, template
from ._util import pick

_SAMPLE_ENTITIES = """
MATCH (n:CodeNode {repo_id: $repo})
WHERE NOT n:REPOSITORY
RETURN n.qname AS qname
ORDER BY n.qname
"""

_GT_ENTITY_TYPE = """
MATCH (n:CodeNode {repo_id: $repo, qname: $qname})
RETURN [l IN labels(n) WHERE l <> 'CodeNode'][0] AS kind
"""


@template("t1_entity_type", 1)
def t1_entity_type(client, repo_id, limit, rng: random.Random):
    rows = pick(rng, client.run(_SAMPLE_ENTITIES, repo=repo_id), limit)
    out = []
    for i, row in enumerate(rows):
        res = client.run(_GT_ENTITY_TYPE, repo=repo_id, qname=row["qname"])
        kind = res[0]["kind"]
        out.append(make_question(
            tier=1, template_id="t1_entity_type", repo_id=repo_id, index=i,
            question=(
                f"In this repository, what kind of code entity is `{row['qname']}`? "
                f"Answer with one of: module, class, function, method, field, global variable."
            ),
            answer=kind, answer_type="string", answer_text=kind.lower().replace("_", " "),
            cypher=_GT_ENTITY_TYPE, params={"repo": repo_id, "qname": row["qname"]},
            evidence={"entity": row["qname"]},
        ))
    return out


_SAMPLE_METHODS = """
MATCH (c:CLASS {repo_id: $repo})-[:HAS_METHOD]->(m:METHOD)
RETURN m.qname AS method, c.qname AS class
ORDER BY m.qname
"""

_GT_METHOD_OWNER = """
MATCH (c:CLASS {repo_id: $repo})-[:HAS_METHOD]->(m:METHOD {qname: $method})
RETURN c.qname AS owner
"""


@template("t1_method_owner", 1)
def t1_method_owner(client, repo_id, limit, rng: random.Random):
    rows = pick(rng, client.run(_SAMPLE_METHODS, repo=repo_id), limit)
    out = []
    for i, row in enumerate(rows):
        owner = client.run(_GT_METHOD_OWNER, repo=repo_id, method=row["method"])[0]["owner"]
        out.append(make_question(
            tier=1, template_id="t1_method_owner", repo_id=repo_id, index=i,
            question=f"Which class defines the method `{row['method']}`? Give the fully qualified class name.",
            answer=owner, answer_type="string",
            cypher=_GT_METHOD_OWNER, params={"repo": repo_id, "method": row["method"]},
            evidence={"method": row["method"], "class": owner},
        ))
    return out


_SAMPLE_DEFINED = """
MATCH (n:CodeNode {repo_id: $repo})
WHERE (n:CLASS OR n:FUNCTION OR n:METHOD) AND n.file_path IS NOT NULL
RETURN n.qname AS qname
ORDER BY n.qname
"""

_GT_FILE = """
MATCH (n:CodeNode {repo_id: $repo, qname: $qname})
RETURN n.file_path AS file_path
"""


@template("t1_defining_file", 1)
def t1_defining_file(client, repo_id, limit, rng: random.Random):
    rows = pick(rng, client.run(_SAMPLE_DEFINED, repo=repo_id), limit)
    out = []
    for i, row in enumerate(rows):
        path = client.run(_GT_FILE, repo=repo_id, qname=row["qname"])[0]["file_path"]
        out.append(make_question(
            tier=1, template_id="t1_defining_file", repo_id=repo_id, index=i,
            question=f"In which source file is `{row['qname']}` defined? Give the path relative to the repository root.",
            answer=path, answer_type="string",
            cypher=_GT_FILE, params={"repo": repo_id, "qname": row["qname"]},
            evidence={"entity": row["qname"]},
        ))
    return out


_SAMPLE_INHERITS = """
MATCH (c:CLASS {repo_id: $repo})-[:INHERITS]->(b:CLASS)
RETURN c.qname AS child
ORDER BY c.qname
"""

_GT_DIRECT_BASE = """
MATCH (c:CLASS {repo_id: $repo, qname: $child})-[:INHERITS]->(b:CLASS)
RETURN collect(DISTINCT b.qname) AS bases
"""


@template("t1_direct_base", 1)
def t1_direct_base(client, repo_id, limit, rng: random.Random):
    rows = pick(rng, client.run(_SAMPLE_INHERITS, repo=repo_id), limit)
    out = []
    for i, row in enumerate(rows):
        bases = sorted(client.run(_GT_DIRECT_BASE, repo=repo_id, child=row["child"])[0]["bases"])
        out.append(make_question(
            tier=1, template_id="t1_direct_base", repo_id=repo_id, index=i,
            question=f"Which class or classes does `{row['child']}` directly inherit from, within this repository?",
            answer=bases, answer_type="list",
            cypher=_GT_DIRECT_BASE, params={"repo": repo_id, "child": row["child"]},
            evidence={"child": row["child"]},
        ))
    return out


_SAMPLE_FIELDS = """
MATCH (c:CLASS {repo_id: $repo})-[:HAS_FIELD]->(f:FIELD)
RETURN c.qname AS class, f.name AS field
ORDER BY c.qname, f.name
"""

_GT_HAS_FIELD = """
MATCH (c:CLASS {repo_id: $repo, qname: $class})
RETURN exists((c)-[:HAS_FIELD]->(:FIELD {name: $field})) AS has_field
"""

#: Candidate negatives: a field name that is real *somewhere else* in the repo,
#: paired with a class that neither defines nor inherits it.
#:
#: Sampling only from existing HAS_FIELD edges made this template answer True
#: every time, which meant the answer was predictable from the template alone --
#: a model could score on it without consulting the graph, and any proxy that
#: correlates with template identity (phrasing, persona, tier) leaked it too.
#: The negatives are built on the same near-miss principle as
#: `t4_absent_entity`: both halves occur in the repo, the pairing does not.
#:
#: Inheritance is excluded when *selecting* a negative, so the question is never
#: confusing about defines-vs-inherits. The stored ground-truth query is still
#: the plain direct-definition check the question asks.
#: Anchors for negatives, drawn separately and combined in Python.
#:
#: The obvious query -- every (class, field-name-from-elsewhere) pair, filtered
#: by a NOT-inherits check -- is a cross product: 10,423,524 rows on Django,
#: each carrying a variable-length inheritance traversal. It never finished.
#: Sampling a bounded set of classes and a bounded set of field names, then
#: testing only those combinations, is the same distribution at a fraction of
#: the cost.
_SAMPLE_CLASSES_FOR_FIELDS = """
MATCH (c:CLASS {repo_id: $repo})
RETURN c.qname AS qname
ORDER BY c.qname
"""

_SAMPLE_FIELD_NAMES = """
MATCH (:CLASS {repo_id: $repo})-[:HAS_FIELD]->(f:FIELD)
RETURN DISTINCT f.name AS name
ORDER BY f.name
"""

#: Absent means neither defined nor inherited, so the question is never
#: ambiguous about defines-vs-inherits. The stored ground-truth query stays the
#: plain direct-definition check the question actually asks.
_GT_FIELD_ABSENT_INCL_INHERITED = """
MATCH (c:CLASS {repo_id: $repo, qname: $class})
OPTIONAL MATCH (c)-[:INHERITS*0..8]->(:CLASS)-[:HAS_FIELD]->(f:FIELD {name: $field})
RETURN count(f) = 0 AS absent
"""


def _sample_absent_field_pairs(client, repo_id, rng, wanted, attempts=200):
    classes = [r["qname"] for r in client.run(_SAMPLE_CLASSES_FOR_FIELDS, repo=repo_id)]
    names = [r["name"] for r in client.run(_SAMPLE_FIELD_NAMES, repo=repo_id)]
    if not classes or not names:
        return []
    out, seen = [], set()
    for _ in range(attempts):
        if len(out) >= wanted:
            break
        cand = (rng.choice(classes), rng.choice(names))
        if cand in seen:
            continue
        seen.add(cand)
        res = client.run(_GT_FIELD_ABSENT_INCL_INHERITED, repo=repo_id,
                         **{"class": cand[0], "field": cand[1]})
        if res and res[0]["absent"]:
            out.append({"class": cand[0], "field": cand[1]})
    return out


def _has_field_question(client, repo_id, index, cls, field):
    ans = client.run(_GT_HAS_FIELD, repo=repo_id, **{"class": cls, "field": field})[0]["has_field"]
    return make_question(
        tier=1, template_id="t1_has_field", repo_id=repo_id, index=index,
        question=f"Does the class `{cls}` define an attribute named `{field}`?",
        answer=bool(ans), answer_type="bool",
        cypher=_GT_HAS_FIELD, params={"repo": repo_id, "class": cls, "field": field},
        evidence={"class": cls, "field": field,
                  "polarity": "positive" if ans else "near_miss_negative"},
    )


@template("t1_has_field", 1)
def t1_has_field(client, repo_id, limit, rng: random.Random):
    """Roughly balanced True/False, so the answer is not predictable from the template."""
    n_positive = (limit + 1) // 2
    n_negative = limit - n_positive

    out = []
    for row in pick(rng, client.run(_SAMPLE_FIELDS, repo=repo_id), n_positive):
        out.append(_has_field_question(client, repo_id, len(out), row["class"], row["field"]))

    for row in _sample_absent_field_pairs(client, repo_id, rng, n_negative):
        rec = _has_field_question(client, repo_id, len(out), row["class"], row["field"])
        if rec.answer is False:           # the selection query should guarantee this
            out.append(rec)
    return out
