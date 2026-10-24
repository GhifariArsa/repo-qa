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
from ._util import callables, pick, rebalance, split_quota

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

#: A field's writers, and the callers that reach each writer: every candidate
#: found walking backwards from a real write is a "yes" by construction.
_FIELD_WRITERS = """
MATCH (w:CodeNode {repo_id: $repo})-[:USES {access: 'write'}]->(f:FIELD)
WHERE w:FUNCTION OR w:METHOD
RETURN w.qname AS writer, f.qname AS field
ORDER BY field, writer
"""
_CALLERS_REACHING_WRITER = """
MATCH (c:CodeNode {repo_id: $repo})-[:CALLS*0..6]->(:CodeNode {repo_id: $repo, qname: $writer})
WHERE c:FUNCTION OR c:METHOD
RETURN DISTINCT c.qname AS caller ORDER BY caller LIMIT 50
"""


def _writes_field_question(client, repo_id, index, source, fld, polarity):
    res = client.run(_GT_CLOSURE_WRITES_FIELD, repo=repo_id, source=source, field=fld)[0]
    return make_question(
        tier=5, template_id="t5_closure_writes_field", repo_id=repo_id, index=index,
        question=(
            f"If `{source}` runs, can the attribute `{fld}` be assigned to -- either in "
            f"`{source}` itself or in anything it calls, directly or transitively?"
        ),
        answer=bool(res["written"]), answer_type="bool",
        cypher=_GT_CLOSURE_WRITES_FIELD, params={"repo": repo_id, "source": source, "field": fld},
        evidence={"writers": sorted(w for w in res["writers"] if w), "polarity": polarity,
                  "composition": "CALLS closure x USES(write) over FIELD"},
    )


@template("t5_closure_writes_field", 5)
def t5_closure_writes_field(client, repo_id, limit, rng: random.Random):
    """Half yes, half no.

    Random (callable, field) pairs almost never write, so this template came
    out "no" on all 40 questions and a constant guess scored perfectly. The
    "yes" side is now built backwards from a real write; the "no" side draws a
    callable from the field's own class or module, so proximity is no tell.
    """
    writes = client.run(_FIELD_WRITERS, repo=repo_id)
    if not writes:
        return []
    n_yes, n_no = split_quota(limit, rng)
    out = []
    for row in pick(rng, writes, max(n_yes * 4, 20)):
        if len(out) >= n_yes:
            break
        callers = [r["caller"] for r in client.run(_CALLERS_REACHING_WRITER, repo=repo_id, writer=row["writer"])]
        if not callers:
            continue
        rec = _writes_field_question(client, repo_id, len(out), rng.choice(callers), row["field"],
                                     "reaches_writer")
        if rec.answer is True:
            out.append(rec)

    names = [c["qname"] for c in callables(client, repo_id)]
    added, seen = 0, set()
    candidates = pick(rng, writes, max(n_no * 10, 50))
    for attempt, row in enumerate(candidates * 2):
        if added >= n_no:
            break
        owner = row["field"].rsplit(".", 1)[0]          # the field's class
        module = owner.rsplit(".", 1)[0]
        # Near misses from the field's own class or module first; once those
        # run dry, any callable.
        near = ([q for q in names if q.startswith(owner + ".")]
                or [q for q in names if q.startswith(module + ".")]) if attempt < len(candidates) else []
        near = near or names
        src = rng.choice(near)
        if (src, row["field"]) in seen:
            continue
        seen.add((src, row["field"]))
        rec = _writes_field_question(client, repo_id, len(out), src, row["field"], "nearby_non_writer")
        if rec.answer is False:
            out.append(rec)
            added += 1
    return rebalance(out)
