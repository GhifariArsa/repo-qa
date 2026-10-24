"""Migration, classification, quotas and prompt regressions without Neo4j."""

import json
import random

import pytest
from typer.testing import CliRunner

from cgx.cli import app
from cgx.qgen.base import make_question, read_questions, registry, write_questions
from cgx.qgen.generate import generate
from cgx.qgen.taxonomy import DEFAULT_FAMILIES, OPERATIONS, TEMPLATE_SPECS


def question(tid="t4_no_call_path", answer=False, answer_type="bool"):
    return make_question(template_id=tid, repo_id="r", index=0,
                         question="Does `pkg.a` reach `pkg.b`?", answer=answer,
                         answer_type=answer_type, cypher="RETURN false AS reachable",
                         params={"repo": "r", "source": "pkg.a", "target": "pkg.b"})


def test_every_registered_template_has_one_family_and_valid_operations():
    assert set(registry()) == set(TEMPLATE_SPECS)
    assert {f for f, _ in registry().values()} == set(DEFAULT_FAMILIES)
    for spec in TEMPLATE_SPECS.values():
        assert spec.family in DEFAULT_FAMILIES
        assert spec.operations and set(spec.operations) <= OPERATIONS


def test_negative_outcome_is_not_a_family():
    positive = question("t2_path_exists", True)
    negative = question()
    assert positive.family == negative.family == "dependency_interaction"
    assert positive.operations == negative.operations == ["traversal"]
    assert (positive.outcome, negative.outcome) == ("positive", "negative")
    assert "tier" not in negative.to_dict()


def test_legacy_migration_preserves_research_data(tmp_path):
    rec = question()
    old = rec.to_dict()
    for key in ("family", "operations", "relations", "outcome", "taxonomy_version", "legacy_taxonomy"):
        old.pop(key)
    old.update(tier=4, tier_name="counterfactual_negative")
    old["verification"] = {"status": "verified", "reviewer": "A", "note": "checked source", "at": "2026-09-01"}
    old["naturalization"] = {"status": "accepted", "model": "test"}
    old["question_template"] = "original question"
    source = tmp_path / "old.jsonl"
    source.write_text(json.dumps(old) + "\n")
    before = source.read_bytes()
    migrated = read_questions(source)[0]
    for key in ("id", "answer", "answer_text", "cypher", "params", "evidence", "verification", "naturalization", "question_template"):
        assert migrated.to_dict()[key] == old[key]
    assert migrated.legacy_taxonomy == {"tier": 4, "tier_name": "counterfactual_negative"}
    assert source.read_bytes() == before
    target = tmp_path / "new.jsonl"
    result = CliRunner().invoke(app, ["migrate-taxonomy", str(source), "--out", str(target)])
    assert result.exit_code == 0, result.output
    assert read_questions(target)[0].to_dict() == migrated.to_dict()
    assert CliRunner().invoke(app, ["migrate-taxonomy", str(source), "--out", str(source)]).exit_code != 0
    assert source.read_bytes() == before


def test_unknown_template_and_future_taxonomy_fail_loudly(tmp_path):
    with pytest.raises(ValueError, match="unknown template"):
        question("custom_unknown")
    source = tmp_path / "future.jsonl"
    source.write_text(json.dumps({**question().to_dict(), "taxonomy_version": 999}) + "\n")
    with pytest.raises(ValueError, match="unsupported taxonomy"):
        read_questions(source)


def test_rewrite_prompt_does_not_reveal_negative_outcome():
    from cgx.qgen.naturalize import build_rewrite_prompt
    from cgx.qgen.personas import personas_for_family
    from cgx.qgen.validate import RepoVocabulary
    rec = question()
    vocab = RepoVocabulary(qnames={"pkg.a", "pkg.b"}, short_name_counts={"a": 1, "b": 1})
    prompt = build_rewrite_prompt(rec, vocab, personas_for_family(rec.family))
    assert "true answer is negative" not in prompt
    assert "counterfactual_negative" not in prompt
    assert "False" not in prompt


def test_comparison_sampling_is_bounded_deterministic_and_balances_outcomes():
    from cgx.qgen.comparison import _candidate_pairs
    rows = [{"qname": "a", "members": ["x"]}, {"qname": "b", "members": ["x", "y"]},
            {"qname": "c", "members": ["z"]}, {"qname": "d", "members": []}]
    pairs = _candidate_pairs(rows, 2, random.Random(7))
    assert pairs == _candidate_pairs(rows, 2, random.Random(7))
    members = {r["qname"]: set(r["members"]) for r in rows}
    assert sorted(bool(members[a] & members[b]) for a, b in pairs) == [False, True]
    assert all(a != "d" and b != "d" for a, b in pairs)
    assert _candidate_pairs(rows[:1], 5, random.Random(7)) == []


def test_quota_allocation_and_invalid_filters(monkeypatch):
    calls = []
    def fake(client, repo, limit, rng):
        calls.append(limit)
        return []
    monkeypatch.setattr("cgx.qgen.generate.registry", lambda: {
        "a": ("structural_comparison", fake), "b": ("structural_comparison", fake),
        "c": ("state_access", fake)})
    _, stats = generate(None, "r", families=("structural_comparison",), per_family=5)
    # Both are lookups and the family has no hard tier: each pool covers the budget.
    assert sorted(calls) == [5, 5]
    assert stats["shortfalls"] == {"structural_comparison": 5}
    with pytest.raises(ValueError, match="unknown families"):
        generate(None, "r", families=("typo",))
    with pytest.raises(ValueError, match="positive"):
        generate(None, "r", per_family=0)


def test_hard_tier_redistributes_shortfall(monkeypatch):
    """A hard template the repo cannot supply leaves its slots to other hard
    templates, not to lookups and not to nothing."""
    from cgx.qgen.base import make_question

    def maker(tid, supply):
        def fn(client, repo, limit, rng):
            return [make_question(template_id=tid, repo_id=repo, index=i, question=f"{tid} {i}",
                                  answer=["x"], answer_type="list", cypher="", params={})
                    for i in range(min(limit, supply))]
        return fn
    monkeypatch.setattr("cgx.qgen.generate.registry", lambda: {
        "comparison_reachable_difference": ("structural_comparison", maker("comparison_reachable_difference", 1)),
        "comparison_reachable_shared": ("structural_comparison", maker("comparison_reachable_shared", 50)),
        "comparison_shared_callees": ("structural_comparison", maker("comparison_shared_callees", 50))})
    records, stats = generate(None, "r", families=("structural_comparison",), per_family=10,
                              completeness="off", hard_share=0.8)
    by = stats["by_template"]
    assert by == {"comparison_reachable_difference": 1, "comparison_reachable_shared": 7,
                  "comparison_shared_callees": 2}
    assert stats["hard_by_family"]["structural_comparison"] == 8
    assert not stats["shortfalls"]


def test_cli_preserves_existing_dataset(tmp_path):
    path = tmp_path / "existing.jsonl"
    write_questions([question()], path)
    before = path.read_bytes()
    result = CliRunner().invoke(app, ["generate", "r", "--out", str(path)])
    assert result.exit_code != 0
    assert "exists" in result.output
    assert path.read_bytes() == before


def test_evaluation_and_review_use_families(tmp_path, capsys):
    from cgx.eval.harness import run_evaluation
    from cgx.verify.cli import report
    path = tmp_path / "questions.jsonl"
    write_questions([question(), question("t5_closure_globals_read", ["g"], "list")], path)
    summary = run_evaluation(path, tmp_path / "scores.json", "oracle")
    assert summary["by_family"] == {"dependency_interaction": 1.0, "state_access": 1.0}
    assert summary["by_outcome"]["negative"] == 1.0
    assert "composition" in summary["by_operation"]
    report(path)
    assert "state_access" in capsys.readouterr().out
