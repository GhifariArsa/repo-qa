"""Deterministic gates of the agentic naturaliser (no model calls)."""

from cgx.qgen.agentic_naturalize import _expected_entities, check_rewrite
from cgx.qgen.base import make_question
from cgx.qgen.validate import RepoVocabulary

VOCAB = RepoVocabulary(
    qnames={"laya", "laya.evals_cli", "laya.evals_cli._cmd_run", "laya.models", "laya.models.Guest",
            "laya.models.Admin", "laya.models.Admin.grant"},
    file_paths={"laya/evals_cli.py", "laya/models.py"},
    short_name_counts={"_cmd_run": 1, "Guest": 1, "Admin": 1, "grant": 1, "evals_cli": 1, "models": 1},
)


def _rec(template_id, question, params, answer, answer_type="string"):
    return make_question(template_id=template_id, repo_id="laya", index=0, question=question,
                         answer=answer, answer_type=answer_type, cypher="RETURN 1",
                         params={"repo": "laya", **params})


FILE_Q = _rec("t1_defining_file", "In which source file is `laya.evals_cli._cmd_run` defined?",
              {"qname": "laya.evals_cli._cmd_run"}, "laya/evals_cli.py")


def test_qualified_name_is_rejected_even_though_the_template_had_it():
    problems = check_rewrite(FILE_Q, "Where does `laya.evals_cli._cmd_run` live?", VOCAB)
    assert any("fully qualified" in p for p in problems)


def test_short_name_with_context_passes():
    assert check_rewrite(FILE_Q, "Where is the `_cmd_run` handler for the evals CLI defined?", VOCAB) == []


def test_a_bare_file_name_is_a_real_reference_not_an_invention():
    rec = _rec("t4_wrong_method_owner", "Does `laya.models.Admin` define or inherit `grant`?",
               {"class": "laya.models.Admin", "method": "grant"}, True, "bool")
    assert check_rewrite(rec, "Can I call `grant` on the `Admin` model in `models.py`?", VOCAB) == []


def test_the_bare_name_the_query_uses_must_survive():
    rec = _rec("t4_wrong_method_owner", "Does `laya.models.Admin` define or inherit `grant`?",
               {"class": "laya.models.Admin", "method": "grant"}, True, "bool")
    assert any("drops the name 'grant'" in p for p in
               check_rewrite(rec, "Can I call that permission method on `Admin`?", VOCAB))


def test_absent_entity_expects_its_real_namespace_and_keeps_its_name():
    rec = _rec("t4_absent_entity", "Does this repository define `laya.models.Guest.grant`?",
               {"qname": "laya.models.Guest.grant"}, False, "bool")
    assert _expected_entities(rec, VOCAB) == ({"laya.models.Guest"}, {"grant"})


def test_naming_the_answer_is_caught_even_when_the_template_contained_it():
    """The pilot passed "the `delete` method on `Collector`" for "which class
    defines `...Collector.delete`" -- the template's qname held the answer."""
    vocab = RepoVocabulary(qnames={"app.deletion.Collector", "app.deletion.Collector.delete"},
                           file_paths={"app/deletion.py"}, short_name_counts={"Collector": 1, "delete": 1})
    rec = _rec("t1_method_owner", "Which class defines `app.deletion.Collector.delete`?",
               {"qname": "app.deletion.Collector.delete"}, "app.deletion.Collector")
    assert any("part of the answer" in p for p in
               check_rewrite(rec, "Which class defines the `delete` method on `Collector`?", vocab))
    assert check_rewrite(rec, "Which class owns the `delete` that runs the cascade in `app/deletion.py`?",
                         vocab) == []


def test_resolver_answers_are_matched_leniently_but_not_loosely():
    from cgx.qgen.agentic_naturalize import normalize_qname

    assert normalize_qname("`laya.evals_cli._cmd_run`", VOCAB) == "laya.evals_cli._cmd_run"
    assert normalize_qname("evals_cli._cmd_run()", VOCAB) == "laya.evals_cli._cmd_run"
    assert normalize_qname("laya/evals_cli.py:_cmd_run", VOCAB) == "laya.evals_cli._cmd_run"
    assert normalize_qname("_cmd_run", VOCAB) == "laya.evals_cli._cmd_run"
    two = RepoVocabulary(qnames={"a.main", "b.main"})
    assert normalize_qname("main", two) == "main"          # ambiguous: no match, not a guess
