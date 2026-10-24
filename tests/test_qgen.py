"""Integration tests for family-based question generation. Require live Neo4j."""

import random

import pytest

REPO = "minirepo"
from cgx.qgen.taxonomy import DEFAULT_FAMILIES


def test_all_families_produce_questions(loaded):
    from cgx.qgen.generate import generate

    records, stats = generate(loaded, REPO, per_template=5, seed=0)
    assert set(stats["by_family"]) == set(DEFAULT_FAMILIES)
    assert all(n > 0 for n in stats["by_family"].values())
    assert all(r.family and r.operations and r.cypher and r.params for r in records)


def test_generation_is_deterministic(loaded):
    from cgx.qgen.generate import generate

    a, _ = generate(loaded, REPO, per_template=4, seed=7)
    b, _ = generate(loaded, REPO, per_template=4, seed=7)
    assert [(r.id, r.question, r.answer_text) for r in a] == \
           [(r.id, r.question, r.answer_text) for r in b]


def test_every_answer_reproduces_from_its_own_cypher(loaded):
    """The provenance claim: each stored query yields its stored answer."""
    from cgx.qgen.generate import generate
    from cgx.verify.cli import recheck_answers

    records, _ = generate(loaded, REPO, per_template=5, seed=0)
    assert recheck_answers(loaded, records) == []


def test_yes_no_templates_are_balanced(loaded):
    """A yes/no template with one fixed answer is answered by its wording alone:
    the closed-book pilot scored 1.00 on every such template. Each must now
    produce both answers, as evenly as the requested count allows."""
    from collections import Counter

    from cgx.qgen.generate import generate

    records, _ = generate(loaded, REPO, per_template=6, seed=0)
    by_template: dict[str, Counter] = {}
    for rec in records:
        if rec.answer_type == "bool":
            by_template.setdefault(rec.template_id, Counter())[rec.answer] += 1
    assert by_template
    for tid, counts in by_template.items():
        assert counts[True] and counts[False], (tid, dict(counts))
        assert abs(counts[True] - counts[False]) <= 1, (tid, dict(counts))

def test_tier4_negatives_are_built_from_real_names(loaded):
    """Negatives must be plausible, else they need no repo knowledge to reject:
    both halves of a "no" are real names; only the combination is not."""
    from cgx.qgen.tier4_negative import t4_absent_entity

    records = t4_absent_entity(loaded, REPO, 5, random.Random(0))
    assert records
    known_names = {
        r["name"] for r in loaded.run(
            "MATCH (n:CodeNode {repo_id:$repo}) WHERE n:FUNCTION OR n:METHOD "
            "RETURN DISTINCT n.name AS name", repo=REPO)
    }
    known_ns = {
        r["q"] for r in loaded.run(
            "MATCH (n:CodeNode {repo_id:$repo}) WHERE n:MODULE OR n:CLASS "
            "RETURN n.qname AS q", repo=REPO)
    }
    negatives = [r for r in records if r.answer is False]
    assert negatives
    for rec in negatives:
        assert rec.evidence["name"] in known_names
        assert rec.evidence["namespace"] in known_ns


def test_tier2_paths_are_genuinely_multi_hop(loaded):
    """Tier 2 must not collapse into tier-1 single-edge lookups."""
    from cgx.qgen.tier2_path import t2_hop_count

    records = t2_hop_count(loaded, REPO, 5, random.Random(0))
    assert records
    assert all(r.answer >= 2 for r in records)


def test_known_ground_truth_values(loaded):
    """Spot-check answers that are checkable by reading the fixture."""
    q = lambda c, **p: loaded.run(c, repo=REPO, **p)

    hops = q("""MATCH p = shortestPath((a:CodeNode {repo_id:$repo, qname:'minirepo.app.run'})
                -[:CALLS*1..12]->(b:CodeNode {repo_id:$repo, qname:'minirepo.utils.strip_ws'}))
                RETURN length(p) AS hops""")[0]["hops"]
    assert hops == 5      # run -> main -> create_user -> normalize -> sanitize -> strip_ws

    fan_in = q("""MATCH (c)-[:CALLS]->(t:CodeNode {repo_id:$repo, qname:'minirepo.models.Base.__init__'})
                  RETURN count(DISTINCT c) AS n""")[0]["n"]
    assert fan_in == 2    # User.__init__ and Guest.__init__

    ancestor = q("""MATCH (a:CLASS {repo_id:$repo, qname:'minirepo.models.Admin'})-[:INHERITS*1..8]->(x:CLASS)
                    <-[:INHERITS*1..8]-(b:CLASS {repo_id:$repo, qname:'minirepo.models.Guest'})
                    RETURN collect(DISTINCT x.qname) AS a""")[0]["a"]
    assert ancestor == ["minirepo.models.Base"]

    callers = q("""MATCH (t:CodeNode {repo_id:$repo, qname:'minirepo.utils.unused_helper'})
                   OPTIONAL MATCH (c)-[:CALLS]->(t) RETURN collect(c.qname) AS c""")[0]["c"]
    assert callers == []


def test_evaluation_scores_per_family(loaded, tmp_path):
    """Oracle scores 1.0 everywhere; the yes-baseline must fail tier 4."""
    from cgx.eval.harness import run_evaluation
    from cgx.qgen.base import write_questions
    from cgx.qgen.generate import generate

    records, _ = generate(loaded, REPO, per_template=5, seed=0)
    path = write_questions(records, tmp_path / "q.jsonl")

    oracle = run_evaluation(path, tmp_path / "oracle.json", "oracle")
    assert oracle["overall"] == 1.0
    assert set(oracle["by_family"]) == set(DEFAULT_FAMILIES)

    stub = run_evaluation(path, tmp_path / "stub.json", "stub")
    assert stub["by_outcome"]["negative"] == 0


def test_comparison_answers_against_fixture_source(loaded):
    from cgx.qgen.comparison import _CALLEE_QUERY, _MEMBER_QUERY
    result = loaded.run(_CALLEE_QUERY % "name IN right_members", repo=REPO,
                        a="minirepo.models.User.__init__", b="minirepo.models.Guest.__init__")
    assert result[0]["answer"] == ["minirepo.models.Base.__init__"]
    result = loaded.run(_CALLEE_QUERY % "NOT (name IN right_members)", repo=REPO,
                        a="minirepo.models.User.__init__", b="minirepo.models.Guest.__init__")
    assert result[0]["answer"] == []
    result = loaded.run(_MEMBER_QUERY, repo=REPO,
                        a="minirepo.models.User", b="minirepo.models.Guest")
    assert set(result[0]["answer"]) == {"__init__"}


def test_family_budget_and_filter(loaded):
    from cgx.qgen.generate import generate
    records, stats = generate(loaded, REPO, families=("structural_comparison",), per_family=6)
    assert records
    assert {r.family for r in records} == {"structural_comparison"}
    assert len(records) <= 6
    assert sum(stats["tier_budget"].values()) == 6


def test_redeclared_methods_against_fixture(loaded):
    from cgx.qgen.inheritance import _REDECLARED
    rows = loaded.run(_REDECLARED, repo=REPO, **{"class": "minirepo.models.User"})
    assert set(rows[0]["answer"]) == {"minirepo.models.User.__init__", "minirepo.models.User.validate"}


def test_repo_wide_templates_on_known_ground_truth(loaded, monkeypatch):
    """minirepo: strip_ws <- sanitize <- normalize <- {create_user, promote} <- main <- run.
    The fixture is tiny, so the answer-size floors are lowered for the test."""
    from cgx.qgen import repo_wide
    from cgx.qgen.repo_wide import t2_files_touched, t2_transitive_callers, t3_all_subclasses

    monkeypatch.setattr(repo_wide, "MIN_ANSWER", 3)
    monkeypatch.setattr(repo_wide, "MIN_FILES", 3)

    callers = {r.params["target"]: r.answer for r in t2_transitive_callers(loaded, REPO, 50, random.Random(0))}
    assert callers["minirepo.utils.strip_ws"] == sorted([
        "minirepo.utils.sanitize", "minirepo.utils.normalize",
        "minirepo.service.UserService.create_user", "minirepo.service.UserService.promote",
        "minirepo.app.main", "minirepo.app.run"])
    subs = {r.params["base"]: r.answer for r in t3_all_subclasses(loaded, REPO, 50, random.Random(0))}
    assert subs["minirepo.models.Base"] == ["minirepo.models.Admin", "minirepo.models.Guest", "minirepo.models.User"]
    files = {r.params["source"]: r.answer for r in t2_files_touched(loaded, REPO, 50, random.Random(0))}
    assert files["minirepo.app.run"] == ["app.py", "config.py", "models.py", "service.py", "utils.py"]


def test_completeness_filter_blocks_exactly_the_affected_region(loaded):
    """Mark `sanitize` as possibly missing outgoing calls: anything whose answer
    reads its outgoing edges -- directly or through a closure -- must be
    blocked, and nothing else."""
    from cgx.qgen.base import make_question
    from cgx.qgen.completeness import CompletenessFilter

    def q(tid, **params):
        return make_question(template_id=tid, repo_id=REPO, index=0, question="?", answer=[],
                             answer_type="list", cypher="RETURN 1", params={"repo": REPO, **params})

    loaded.run("MATCH (n:CodeNode {repo_id:$repo, qname:'minirepo.utils.sanitize'}) SET n.unresolved_out = 1", repo=REPO)
    try:
        gate = CompletenessFilter(loaded, REPO)
        assert not gate.ok(q("t3_reach_set_size", source="minirepo.app.run"))       # sanitize in closure
        assert not gate.ok(q("t3_fan_out", source="minirepo.utils.sanitize"))
        assert gate.ok(q("t3_fan_out", source="minirepo.utils.normalize"))          # its own calls are fine
        assert gate.ok(q("t3_reach_set_size", source="minirepo.config.get_timeout"))  # does not reach it
        assert gate.ok(q("t3_all_subclasses", base="minirepo.models.Base"))           # no CALLS: untouched
    finally:
        loaded.run("MATCH (n:CodeNode {repo_id:$repo, qname:'minirepo.utils.sanitize'}) SET n.unresolved_out = 0", repo=REPO)


def test_completeness_checks_only_what_could_change_the_answer(loaded):
    """A witnessed "yes" needs no check; a k-hop shortest path needs only the
    k-2 window. run -> main -> create_user -> normalize -> sanitize -> strip_ws."""
    from cgx.qgen.base import make_question
    from cgx.qgen.completeness import CompletenessFilter

    def q(tid, answer, answer_type, **params):
        return make_question(template_id=tid, repo_id=REPO, index=0, question="?", answer=answer,
                             answer_type=answer_type, cypher="RETURN 1", params={"repo": REPO, **params})

    loaded.run("MATCH (n:CodeNode {repo_id:$repo, qname:'minirepo.utils.sanitize'}) SET n.unresolved_out = 1", repo=REPO)
    try:
        gate = CompletenessFilter(loaded, REPO)
        # sanitize is 4 hops from run: a "yes" still stands, a "no" does not
        assert gate.ok(q("t2_path_exists", True, "bool", source="minirepo.app.run", target="minirepo.utils.strip_ws"))
        assert not gate.ok(q("t2_path_exists", False, "bool", source="minirepo.app.run", target="minirepo.config.get_retries"))
        # 3-hop answer from run: window is 1 hop (run, main) -- sanitize is outside it
        assert gate.ok(q("t2_hop_count", 3, "count", source="minirepo.app.run", target="minirepo.utils.normalize"))
        # 5-hop answer from run: window is 3 hops, still short of sanitize (4 hops)
        assert gate.ok(q("t2_hop_count", 5, "count", source="minirepo.app.run", target="minirepo.utils.strip_ws"))
        # from main, sanitize is 3 hops away -- inside a 4-hop answer's 2-hop window? no: 3 > 2
        assert gate.ok(q("t2_hop_count", 4, "count", source="minirepo.app.main", target="minirepo.utils.strip_ws"))
        # from create_user, sanitize is 2 hops away: a 3-hop answer's window (1) misses it,
        # but a 4-hop answer's window (2) contains it
        assert not gate.ok(q("t2_hop_count", 4, "count", source="minirepo.service.UserService.create_user",
                             target="minirepo.utils.strip_ws"))
    finally:
        loaded.run("MATCH (n:CodeNode {repo_id:$repo, qname:'minirepo.utils.sanitize'}) SET n.unresolved_out = 0", repo=REPO)


def test_closure_comparisons_match_independent_set_algebra(loaded, monkeypatch):
    """Every stored answer must equal the set operation computed here, in Python,
    from the reachable sets -- not merely reproduce its own Cypher."""
    from cgx.qgen import repo_wide

    monkeypatch.setattr(repo_wide, "MIN_ANSWER", 2)
    reach = lambda q: set(loaded.run(repo_wide._REACH, repo=REPO, q=q)[0]["reach"])
    for fn, op in ((repo_wide.comparison_reachable_difference, lambda x, y: x - y),
                   (repo_wide.comparison_reachable_shared, lambda x, y: x & y)):
        recs = fn(loaded, REPO, 10, random.Random(0))
        assert recs, fn.__name__
        for rec in recs:
            a, b = rec.params["a"], rec.params["b"]
            assert rec.answer == sorted(op(reach(a), reach(b)) - {a, b}), rec.id


def test_inheritance_depth_templates(loaded, monkeypatch):
    """Base <- User <- Admin, Base <- Guest; only User redeclares `validate`;
    all four classes live in models.py."""
    from cgx.qgen import repo_wide

    monkeypatch.setattr(repo_wide, "MIN_OVERRIDERS", 1)
    monkeypatch.setattr(repo_wide, "MIN_FILES", 1)
    over = {(r.params["base"], r.params["method"]): r.answer
            for r in repo_wide.inheritance_overriders(loaded, REPO, 10, random.Random(0))}
    assert over[("minirepo.models.Base", "validate")] == ["minirepo.models.User"]
    files = {r.params["base"]: r.answer
             for r in repo_wide.organisation_subclass_files(loaded, REPO, 10, random.Random(0))}
    assert files["minirepo.models.Base"] == ["models.py"]
