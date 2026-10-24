"""Declaration relationships; no claim about runtime dispatch or Python MRO."""

from .base import make_question, template
from ._util import pick

_CLASSES = """
MATCH (c:CLASS {repo_id: $repo})-[:INHERITS]->(:CLASS)
RETURN DISTINCT c.qname AS qname ORDER BY qname
"""
_REDECLARED = """
MATCH (c:CLASS {repo_id: $repo, qname: $class})-[:HAS_METHOD]->(m:METHOD)
MATCH (c)-[:INHERITS*1..8]->(a:CLASS)-[:HAS_METHOD]->(base:METHOD)
WHERE m.name = base.name
RETURN collect(DISTINCT m.qname) AS answer,
       collect(DISTINCT base.qname) AS ancestor_declarations
"""


@template("inheritance_redeclared_methods")
def redeclared_methods(client, repo_id, limit, rng):
    out = []
    for row in pick(rng, client.run(_CLASSES, repo=repo_id), limit):
        params = {"repo": repo_id, "class": row["qname"]}
        result = client.run(_REDECLARED, **params)[0]
        out.append(make_question(
            template_id="inheritance_redeclared_methods", repo_id=repo_id, index=len(out),
            question=(f"Which methods declared directly in `{row['qname']}` have the same name "
                      "as a method declared in an in-repository ancestor within eight inheritance "
                      "steps? List the subclass methods' fully qualified names, or say none."),
            answer=sorted(result["answer"]), answer_type="list", cypher=_REDECLARED,
            params=params, evidence={"ancestor_declarations": sorted(result["ancestor_declarations"]),
                                     "max_inheritance_hops": 8,
                                     "semantics": "same-name declarations, not dispatch resolution"},
        ))
    return out
