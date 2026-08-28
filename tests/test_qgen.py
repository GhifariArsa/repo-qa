"""Integration tests for tiered question generation. Require a live Neo4j."""

import random

import pytest

REPO = "minirepo"


@pytest.fixture(scope="module")
def loaded(neo4j_client, minigraph):
    from cgx.graph.ingest import ingest

    graph, _ = minigraph
    neo4j_client.apply_schema()
    ingest(neo4j_client, graph.to_dict(), replace=True)
    return neo4j_client


def test_all_tiers_produce_questions(loaded):
    from cgx.qgen.generate import generate

    records, stats = generate(loaded, REPO, tiers=(1, 2, 3, 4), per_template=5, seed=0)
    assert set(stats["by_tier"]) == {1, 2, 3, 4}
    for tier in (1, 2, 3, 4):
        assert stats["by_tier"][tier] > 0
    assert all(r.tier_name and r.cypher and r.params for r in records)


def test_generation_is_deterministic(loaded):
    from cgx.qgen.generate import generate

    a, _ = generate(loaded, REPO, tiers=(1, 2, 3, 4), per_template=4, seed=7)
    b, _ = generate(loaded, REPO, tiers=(1, 2, 3, 4), per_template=4, seed=7)
    assert [(r.id, r.question, r.answer_text) for r in a] == \
           [(r.id, r.question, r.answer_text) for r in b]


def test_every_answer_reproduces_from_its_own_cypher(loaded):
    """The provenance claim: each stored query yields its stored answer."""
    from cgx.qgen.generate import generate
    from cgx.verify.cli import recheck_answers

    records, _ = generate(loaded, REPO, tiers=(1, 2, 3, 4, 5), per_template=5, seed=0)
    assert recheck_answers(loaded, records) == []


def test_tier4_answers_are_all_negative(loaded):
    """A negative tier full of positives would be silently useless."""
    from cgx.qgen.generate import generate

    records, _ = generate(loaded, REPO, tiers=(4,), per_template=5, seed=0)
    assert records
    for rec in records:
        if rec.answer_type == "bool":
            assert rec.answer is False, rec.id
        elif rec.answer_type == "list":
            assert rec.answer == [], rec.id


def test_tier4_negatives_are_built_from_real_names(loaded):
    """Negatives must be plausible, else they need no repo knowledge to reject."""
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
    for rec in records:
        assert rec.evidence["borrowed_name"] in known_names
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


def test_evaluation_scores_per_tier(loaded, tmp_path):
    """Oracle scores 1.0 everywhere; the yes-baseline must fail tier 4."""
    from cgx.eval.harness import run_evaluation
    from cgx.qgen.base import write_questions
    from cgx.qgen.generate import generate

    records, _ = generate(loaded, REPO, tiers=(1, 2, 3, 4), per_template=5, seed=0)
    path = write_questions(records, tmp_path / "q.jsonl")

    oracle = run_evaluation(path, tmp_path / "oracle.json", "oracle")
    assert oracle["overall"] == 1.0
    assert set(oracle["by_tier"]) == {1, 2, 3, 4}

    stub = run_evaluation(path, tmp_path / "stub.json", "stub")
    assert stub["by_tier"][4] < 0.5
