"""Tier 5 -- cross-graph composition (reduced scope).

The full stretch goal was to compose the CALLS graph with a *data-flow* graph.
This repo has no data-flow extraction, and building one is a separate project,
so tier 5 here composes the two graphs that do exist: the call graph (CALLS)
and the data-access graph (USES, over FIELD and GLOBAL_VARIABLE, carrying a
read/write `access` attribute).

That is genuinely two-graph composition -- a question here cannot be answered
from either relation alone -- but it is *not* data-flow, and it should not be
reported as such. Tier 5 is disabled by default for that reason; pass
--include-tier5 to generate it.
"""

from __future__ import annotations

import random

from .base import make_question, template
from ._util import callables, pick

_GT_CLOSURE_READS = """
MATCH (s:CodeNode {repo_id: $repo, qname: $source})
OPTIONAL MATCH (s)-[:CALLS*0..8]->(f:CodeNode)-[u:USES]->(v:CodeNode)
WHERE v:GLOBAL_VARIABLE AND u.access = 'read'
RETURN collect(DISTINCT v.qname) AS globals_read
"""


@template("t5_closure_globals_read", 5)
def t5_closure_globals_read(client, repo_id, limit, rng: random.Random):
    rows = pick(rng, callables(client, repo_id), max(limit * 3, 15))
    out = []
    for row in rows:
        res = client.run(_GT_CLOSURE_READS, repo=repo_id, source=row["qname"])[0]
        answer = sorted(g for g in res["globals_read"] if g)
        if not answer:
            continue
        out.append(make_question(
            tier=5, template_id="t5_closure_globals_read", repo_id=repo_id, index=len(out),
            question=(
                f"Consider `{row['qname']}` and everything it can call, directly or transitively. "
                f"Which module-level global variables are read anywhere in that call closure?"
            ),
            answer=answer, answer_type="list",
            cypher=_GT_CLOSURE_READS, params={"repo": repo_id, "source": row["qname"]},
            evidence={"composition": "CALLS closure x USES(read) over GLOBAL_VARIABLE"},
        ))
        if len(out) >= limit:
            break
    return out


_GT_CLOSURE_WRITES_FIELD = """
MATCH (s:CodeNode {repo_id: $repo, qname: $source})
OPTIONAL MATCH (s)-[:CALLS*0..8]->(f:CodeNode)-[u:USES]->(fl:FIELD {qname: $field})
WHERE u.access = 'write'
RETURN count(u) > 0 AS written, collect(DISTINCT f.qname) AS writers
"""

_SAMPLE_WRITTEN_FIELDS = """
MATCH (:CodeNode {repo_id: $repo})-[u:USES {access: 'write'}]->(f:FIELD)
RETURN DISTINCT f.qname AS field
ORDER BY f.qname
"""


@template("t5_closure_writes_field", 5)
def t5_closure_writes_field(client, repo_id, limit, rng: random.Random):
    fields = [r["field"] for r in client.run(_SAMPLE_WRITTEN_FIELDS, repo=repo_id)]
    calls = callables(client, repo_id)
    if not fields or not calls:
        return []
    pairs = [(c["qname"], f) for c in calls for f in fields]
    out = []
    for source, fld in pick(rng, pairs, max(limit * 5, 30)):
        res = client.run(_GT_CLOSURE_WRITES_FIELD, repo=repo_id, source=source, field=fld)[0]
        out.append(make_question(
            tier=5, template_id="t5_closure_writes_field", repo_id=repo_id, index=len(out),
            question=(
                f"If `{source}` runs, can the attribute `{fld}` be assigned to -- either in "
                f"`{source}` itself or in anything it calls, directly or transitively?"
            ),
            answer=bool(res["written"]), answer_type="bool",
            cypher=_GT_CLOSURE_WRITES_FIELD, params={"repo": repo_id, "source": source, "field": fld},
            evidence={"writers": sorted(w for w in res["writers"] if w),
                      "composition": "CALLS closure x USES(write) over FIELD"},
        ))
        if len(out) >= limit:
            break
    return out
