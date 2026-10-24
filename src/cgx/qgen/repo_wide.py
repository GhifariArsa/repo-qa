"""Repo-wide, multi-hop questions: the ones that still separate models.

In the pilot, questions answerable from one file or one hop were solved almost
every time, with or without the repository, while questions that forced a
search across the codebase -- everything reachable from a function, every
caller at any depth -- were where accuracy fell. These templates are built
around that property: none can be answered by reading one file, because the
answer is spread across several and found only by following edges repeatedly.

  t2_transitive_callers              every function that can reach a target, at any depth
  t2_files_touched                   every file whose code can run when a function runs
  t3_all_subclasses                  every class below a base, at any depth
  comparison_reachable_difference    what A can reach that B cannot, at any depth
  comparison_reachable_shared        what both A and B can reach, at any depth
  inheritance_overriders             which descendants, at any depth, redeclare a method
  organisation_subclass_files        which files define a base's descendants, at any depth

A weak and a strong model scored within four points of each other on the
one-hop templates, and apart only on templates like these.

Candidates must have an answer of workable size (a question whose answer is 400
names is a data dump, not a question) and must genuinely need more than one
hop: at least one caller or subclass that is NOT a direct one.
"""

from __future__ import annotations

import random

from ._util import callables, pick
from .base import make_question, template

#: Seconds one candidate's traversal may take before it is skipped. Transitive
#: queries are cheap on most nodes and explosive on a few hubs.
CANDIDATE_TIMEOUT = 8.0
#: At least five members: with three, a model that explored half the answer
#: still scores respectably, and partial exploration is what separates models.
MIN_ANSWER, MAX_ANSWER = 5, 25
MIN_FILES = 4
#: Overriders are a filtered subset of the descendants, so a smaller floor.
MIN_OVERRIDERS = 2
MAX_CANDIDATES = 400

_GT_TRANSITIVE_CALLERS = """
MATCH (c:CodeNode)-[:CALLS*1..12]->(t:CodeNode {repo_id: $repo, qname: $target})
WHERE (c:FUNCTION OR c:METHOD) AND c <> t
RETURN collect(DISTINCT c.qname) AS callers
"""
_DIRECT_CALLERS = """
MATCH (c:CodeNode)-[:CALLS]->(t:CodeNode {repo_id: $repo, qname: $target})
RETURN collect(DISTINCT c.qname) AS callers
"""


@template("t2_transitive_callers")
def t2_transitive_callers(client, repo_id, limit, rng: random.Random):
    out = []
    for row in pick(rng, callables(client, repo_id), MAX_CANDIDATES):
        if len(out) >= limit:
            break
        res = client.run_with_timeout(_GT_TRANSITIVE_CALLERS, CANDIDATE_TIMEOUT,
                                      repo=repo_id, target=row["qname"])
        if not res:
            continue
        callers = sorted(res[0]["callers"])
        if not MIN_ANSWER <= len(callers) <= MAX_ANSWER:
            continue
        direct = set(client.run(_DIRECT_CALLERS, repo=repo_id, target=row["qname"])[0]["callers"])
        if not set(callers) - direct:
            continue                      # every caller is direct: a one-hop question
        out.append(make_question(
            template_id="t2_transitive_callers", repo_id=repo_id, index=len(out),
            question=(
                f"Which functions or methods in this repository can end up calling "
                f"`{row['qname']}`, either directly or through any chain of intermediate calls?"
            ),
            answer=callers, answer_type="list",
            cypher=_GT_TRANSITIVE_CALLERS, params={"repo": repo_id, "target": row["qname"]},
            evidence={"direct_callers": sorted(direct), "indirect": len(set(callers) - direct)},
        ))
    return out


_GT_FILES_TOUCHED = """
MATCH (s:CodeNode {repo_id: $repo, qname: $source})-[:CALLS*0..12]->(n:CodeNode)
WHERE n.file_path IS NOT NULL
RETURN collect(DISTINCT n.file_path) AS files
"""


@template("t2_files_touched")
def t2_files_touched(client, repo_id, limit, rng: random.Random):
    out = []
    for row in pick(rng, callables(client, repo_id), MAX_CANDIDATES):
        if len(out) >= limit:
            break
        res = client.run_with_timeout(_GT_FILES_TOUCHED, CANDIDATE_TIMEOUT,
                                      repo=repo_id, source=row["qname"])
        if not res:
            continue
        files = sorted(res[0]["files"])
        if not MIN_FILES <= len(files) <= 15:
            continue
        out.append(make_question(
            template_id="t2_files_touched", repo_id=repo_id, index=len(out),
            question=(
                f"When `{row['qname']}` runs, which source files in this repository contain "
                f"code that can execute as a result -- its own file, and the file of anything "
                f"it calls directly or transitively?"
            ),
            answer=files, answer_type="list",
            cypher=_GT_FILES_TOUCHED, params={"repo": repo_id, "source": row["qname"]},
            evidence={"file_count": len(files)},
        ))
    return out


_GT_ALL_SUBCLASSES = """
MATCH (d:CLASS)-[:INHERITS*1..8]->(c:CLASS {repo_id: $repo, qname: $base})
RETURN collect(DISTINCT d.qname) AS subclasses
"""
_DIRECT_SUBCLASSES = """
MATCH (d:CLASS)-[:INHERITS]->(c:CLASS {repo_id: $repo, qname: $base})
RETURN collect(DISTINCT d.qname) AS subclasses
"""
_BASES_WITH_GRANDCHILDREN = """
MATCH (:CLASS)-[:INHERITS]->(:CLASS)-[:INHERITS]->(c:CLASS {repo_id: $repo})
RETURN DISTINCT c.qname AS qname ORDER BY qname
"""


@template("t3_all_subclasses")
def t3_all_subclasses(client, repo_id, limit, rng: random.Random):
    out = []
    for row in pick(rng, client.run(_BASES_WITH_GRANDCHILDREN, repo=repo_id), MAX_CANDIDATES):
        if len(out) >= limit:
            break
        subs = sorted(client.run(_GT_ALL_SUBCLASSES, repo=repo_id, base=row["qname"])[0]["subclasses"])
        if not MIN_ANSWER <= len(subs) <= MAX_ANSWER:
            continue
        direct = set(client.run(_DIRECT_SUBCLASSES, repo=repo_id, base=row["qname"])[0]["subclasses"])
        out.append(make_question(
            template_id="t3_all_subclasses", repo_id=repo_id, index=len(out),
            question=(
                f"Which classes in this repository inherit from `{row['qname']}`, "
                f"directly or through any chain of intermediate base classes?"
            ),
            answer=subs, answer_type="list",
            cypher=_GT_ALL_SUBCLASSES, params={"repo": repo_id, "base": row["qname"]},
            evidence={"direct_subclasses": sorted(direct), "indirect": len(set(subs) - direct)},
        ))
    return out


# --------------------------------------------------------------------------
# Closure comparisons
# --------------------------------------------------------------------------

_REACH = """
MATCH (s:CodeNode {repo_id: $repo, qname: $q})-[:CALLS*1..12]->(n:CodeNode)
WHERE (n:FUNCTION OR n:METHOD) AND n <> s
RETURN collect(DISTINCT n.qname) AS reach
"""

#: Both closures, then the set operation; `a` and `b` themselves are left out
#: of the answer so "does A reach B" never decides a member.
_GT_REACH_COMPARE = """
MATCH (a:CodeNode {repo_id: $repo, qname: $a})
OPTIONAL MATCH (a)-[:CALLS*1..12]->(x:CodeNode) WHERE (x:FUNCTION OR x:METHOD)
WITH a, collect(DISTINCT x.qname) AS ra
MATCH (b:CodeNode {repo_id: $repo, qname: $b})
OPTIONAL MATCH (b)-[:CALLS*1..12]->(y:CodeNode) WHERE (y:FUNCTION OR y:METHOD)
WITH ra, collect(DISTINCT y.qname) AS rb
RETURN [n IN ra WHERE %s AND n <> $a AND n <> $b] AS answer
"""
_GT_REACH_DIFFERENCE = _GT_REACH_COMPARE % "NOT n IN rb"
_GT_REACH_SHARED = _GT_REACH_COMPARE % "n IN rb"


def _closures(client, repo_id, rng, want: int) -> dict[str, set[str]]:
    """Reachable sets of a sample of callables, keeping those big enough to compare."""
    out: dict[str, set[str]] = {}
    for row in pick(rng, callables(client, repo_id), MAX_CANDIDATES):
        if len(out) >= want:
            break
        res = client.run_with_timeout(_REACH, CANDIDATE_TIMEOUT, repo=repo_id, q=row["qname"])
        if res and len(res[0]["reach"]) >= MIN_ANSWER:
            out[row["qname"]] = set(res[0]["reach"])
    return out


def _closure_pairs(client, repo_id, limit, rng, shared: bool):
    """Pairs whose closures overlap -- an unrelated pair makes the difference
    trivially "everything A reaches" and the intersection trivially empty."""
    reach = _closures(client, repo_id, rng, want=60)
    names = sorted(reach)
    pairs = [(a, b) for a in names for b in names if a != b and reach[a] & reach[b]]
    rng.shuffle(pairs)
    out = []
    for a, b in pairs:
        members = (reach[a] & reach[b]) if shared else (reach[a] - reach[b])
        members -= {a, b}
        if MIN_ANSWER <= len(members) <= MAX_ANSWER:
            out.append((a, b))
            if len(out) >= limit:
                break
    return out


def _closure_question(client, repo_id, template_id, index, a, b, query, wording):
    answer = sorted(client.run(query, repo=repo_id, a=a, b=b)[0]["answer"])
    return make_question(
        template_id=template_id, repo_id=repo_id, index=index,
        question=wording.format(a=a, b=b), answer=answer, answer_type="list",
        cypher=query, params={"repo": repo_id, "a": a, "b": b},
        evidence={"comparison": "closure", "size": len(answer)},
    )


@template("comparison_reachable_difference")
def comparison_reachable_difference(client, repo_id, limit, rng: random.Random):
    return [_closure_question(
        client, repo_id, "comparison_reachable_difference", i, a, b, _GT_REACH_DIFFERENCE,
        "Following calls to any depth, which functions or methods can `{a}` reach that `{b}` "
        "cannot? Leave out the two functions themselves.")
        for i, (a, b) in enumerate(_closure_pairs(client, repo_id, limit, rng, shared=False))]


@template("comparison_reachable_shared")
def comparison_reachable_shared(client, repo_id, limit, rng: random.Random):
    return [_closure_question(
        client, repo_id, "comparison_reachable_shared", i, a, b, _GT_REACH_SHARED,
        "Following calls to any depth, which functions or methods can both `{a}` and `{b}` "
        "reach? Leave out the two functions themselves.")
        for i, (a, b) in enumerate(_closure_pairs(client, repo_id, limit, rng, shared=True))]


# --------------------------------------------------------------------------
# Inheritance, at any depth
# --------------------------------------------------------------------------

_BASE_METHODS = """
MATCH (c:CLASS {repo_id: $repo, qname: $base})-[:HAS_METHOD]->(m:METHOD)
WHERE NOT m.name STARTS WITH '__'
RETURN m.name AS name ORDER BY name
"""
_GT_OVERRIDERS = """
MATCH (d:CLASS)-[:INHERITS*1..8]->(c:CLASS {repo_id: $repo, qname: $base})
WHERE EXISTS { MATCH (d)-[:HAS_METHOD]->(m:METHOD) WHERE m.name = $method }
RETURN collect(DISTINCT d.qname) AS answer
"""
_DESCENDANTS = """
MATCH (d:CLASS)-[:INHERITS*1..8]->(c:CLASS {repo_id: $repo, qname: $base})
RETURN count(DISTINCT d) AS n
"""


@template("inheritance_overriders")
def inheritance_overriders(client, repo_id, limit, rng: random.Random):
    """Needs every descendant found AND each one's own body checked. At least
    one descendant must NOT redeclare the method, or "all of them" is the answer."""
    out = []
    for row in pick(rng, client.run(_BASES_WITH_GRANDCHILDREN, repo=repo_id), MAX_CANDIDATES):
        if len(out) >= limit:
            break
        n_desc = client.run(_DESCENDANTS, repo=repo_id, base=row["qname"])[0]["n"]
        methods = [r["name"] for r in client.run(_BASE_METHODS, repo=repo_id, base=row["qname"])]
        rng.shuffle(methods)
        for method in methods:
            answer = sorted(client.run(_GT_OVERRIDERS, repo=repo_id, base=row["qname"],
                                       method=method)[0]["answer"])
            if MIN_OVERRIDERS <= len(answer) <= MAX_ANSWER and len(answer) < n_desc:
                out.append(make_question(
                    template_id="inheritance_overriders", repo_id=repo_id, index=len(out),
                    question=(
                        f"Among the classes that inherit from `{row['qname']}`, directly or through "
                        f"any chain of base classes, which ones declare their own method named "
                        f"`{method}` rather than only inheriting it?"
                    ),
                    answer=answer, answer_type="list",
                    cypher=_GT_OVERRIDERS,
                    params={"repo": repo_id, "base": row["qname"], "method": method},
                    evidence={"descendants": n_desc},
                ))
                break
    return out


_GT_SUBCLASS_FILES = """
MATCH (d:CLASS)-[:INHERITS*1..8]->(c:CLASS {repo_id: $repo, qname: $base})
WHERE d.file_path IS NOT NULL
RETURN collect(DISTINCT d.file_path) AS answer
"""


@template("organisation_subclass_files")
def organisation_subclass_files(client, repo_id, limit, rng: random.Random):
    out = []
    for row in pick(rng, client.run(_BASES_WITH_GRANDCHILDREN, repo=repo_id), MAX_CANDIDATES):
        if len(out) >= limit:
            break
        files = sorted(client.run(_GT_SUBCLASS_FILES, repo=repo_id, base=row["qname"])[0]["answer"])
        if not MIN_FILES <= len(files) <= 15:
            continue
        out.append(make_question(
            template_id="organisation_subclass_files", repo_id=repo_id, index=len(out),
            question=(
                f"In which source files are the classes that inherit from `{row['qname']}` "
                f"defined -- counting subclasses at any depth, not just direct ones?"
            ),
            answer=files, answer_type="list",
            cypher=_GT_SUBCLASS_FILES, params={"repo": repo_id, "base": row["qname"]},
            evidence={"file_count": len(files)},
        ))
    return out
