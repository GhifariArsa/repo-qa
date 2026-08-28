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


@template("t1_has_field", 1)
def t1_has_field(client, repo_id, limit, rng: random.Random):
    rows = pick(rng, client.run(_SAMPLE_FIELDS, repo=repo_id), limit)
    out = []
    for i, row in enumerate(rows):
        ans = client.run(_GT_HAS_FIELD, repo=repo_id, **{"class": row["class"], "field": row["field"]})[0]["has_field"]
        out.append(make_question(
            tier=1, template_id="t1_has_field", repo_id=repo_id, index=i,
            question=f"Does the class `{row['class']}` define an attribute named `{row['field']}`?",
            answer=bool(ans), answer_type="bool",
            cypher=_GT_HAS_FIELD, params={"repo": repo_id, "class": row["class"], "field": row["field"]},
            evidence={"class": row["class"], "field": row["field"]},
        ))
    return out
