"""The agent harness, driven by a scripted fake client: no network, no spend."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cgx.eval.agent import RepoSandbox, run_agent
from cgx.eval.harness import load_calls, run_agent_evaluation, score
from cgx.qgen.base import read_questions

ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    """Plays back a script of turns; each turn is a list of (tool, args)."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # The agent keeps appending to one messages list; record what the
        # model saw at this call, not what the list becomes later.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        turn = self.script.pop(0) if self.script else [("submit_answer", {"answer": None})]
        calls = []
        for i, (name, args) in enumerate(turn):
            fn = SimpleNamespace(name=name, arguments=json.dumps(args))
            call = SimpleNamespace(id=f"call_{len(self.requests)}_{i}", function=fn, type="function")
            call.model_dump = lambda c=call: {"id": c.id, "type": "function",
                                              "function": {"name": c.function.name,
                                                           "arguments": c.function.arguments}}
            calls.append(call)
        message = SimpleNamespace(content="", tool_calls=calls)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10, cost=0.001)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _rec(**overrides):
    recs = read_questions(ROOT / "data/questions/minirepo.jsonl")
    return next(r for r in recs if all(getattr(r, k) == v for k, v in overrides.items()))


@pytest.fixture
def sandbox():
    return RepoSandbox(ROOT / "fixtures/minirepo")


def test_sandbox_refuses_to_leave_the_repository(sandbox):
    assert sandbox.call("read_file", {"path": "../../pyproject.toml"}).startswith("error:")
    assert sandbox.call("list_dir", {"path": "/etc"}).startswith("error:")
    assert "def normalize" in sandbox.call("read_file", {"path": "utils.py"})
    assert "utils.py:" in sandbox.call("grep", {"pattern": r"def sanitize"})


def test_agent_explores_then_submits(sandbox):
    rec = _rec(answer_type="bool")
    client = FakeClient([[("grep", {"pattern": "class"})], [("submit_answer", {"answer": rec.answer})]])
    res = run_agent(client, "m", rec, condition="repo", question=rec.question,
                    sandbox=sandbox, package_root=".")
    assert res.answered and res.prediction == rec.answer
    assert res.tool_calls == 1 and res.steps == 2
    tool_result = client.requests[1]["messages"][-1]
    assert tool_result["role"] == "tool" and "class" in tool_result["content"]


def test_closed_book_offers_no_repository_tools():
    rec = _rec(answer_type="count")
    client = FakeClient([[("submit_answer", {"answer": 3})]])
    run_agent(client, "m", rec, condition="closed", question=rec.question,
              sandbox=None, package_root=".")
    names = [t["function"]["name"] for t in client.requests[0]["tools"]]
    assert names == ["submit_answer"]
    assert client.requests[0]["tools"][0]["function"]["parameters"]["properties"]["answer"]["type"] == "integer"


def test_a_repo_tool_called_in_closed_book_is_refused():
    rec = _rec(answer_type="bool")
    client = FakeClient([[("read_file", {"path": "utils.py"})], [("submit_answer", {"answer": True})]])
    res = run_agent(client, "m", rec, condition="closed", question=rec.question,
                    sandbox=None, package_root=".")
    assert res.tool_calls == 0
    assert client.requests[1]["messages"][-1]["content"].startswith("error: no tool")


def test_last_step_forces_an_answer(sandbox):
    from cgx.eval import agent

    rec = _rec(answer_type="bool")
    client = FakeClient([[("list_dir", {"path": "."})]] * (agent.MAX_STEPS - 1)
                        + [[("submit_answer", {"answer": False})]])
    res = run_agent(client, "m", rec, condition="repo", question=rec.question,
                    sandbox=sandbox, package_root=".")
    assert res.answered and res.steps == agent.MAX_STEPS
    assert client.requests[-1]["tool_choice"] == {"type": "function", "function": {"name": "submit_answer"}}


def test_scorer_normalises_formatting_not_content():
    rec = _rec(template_id="t1_entity_type")
    assert score(rec, rec.answer.lower().replace("_", " ")) == 1.0
    assert score(rec, "`" + rec.answer + "`") == 1.0
    assert score(rec, "definitely not a kind") == 0.0


def test_any_real_shortest_chain_scores(tmp_path):
    rec = _rec(answer_type="path")
    calls = load_calls(ROOT / "data/graphs/minirepo.json")
    assert score(rec, rec.answer, calls) == 1.0
    a, b = rec.answer[0], rec.answer[-1]
    assert score(rec, [a] + ["not.a.real.hop"] * (len(rec.answer) - 2) + [b], calls) == 0.0
    assert score(rec, list(reversed(rec.answer)), calls) == 0.0


def test_run_is_resumable_and_never_repays(tmp_path):
    def always_gold():
        recs = {r.question: r for r in read_questions(ROOT / "data/questions/minirepo.jsonl")}

        def create(**kw):
            rec = recs[kw["messages"][1]["content"].split("\n\n")[0]]   # drop the format line
            return FakeClient([[("submit_answer", {"answer": rec.answer})]])._create(**kw)
        client = FakeClient([])
        client.chat.completions.create = create
        return client

    kwargs = dict(repo_id="minirepo", repo_path=ROOT / "fixtures/minirepo",
                  package_root=ROOT / "fixtures", questions_path=ROOT / "data/questions/minirepo.jsonl",
                  graph_path=ROOT / "data/graphs/minirepo.json", model="openrouter:fake/model",
                  condition="closed", phrasing="template", per_family=2, limit=None, seed=0,
                  concurrency=2, max_cost=None, out_dir=tmp_path)
    first = run_agent_evaluation(client=always_gold(), **kwargs)
    assert first["n"] == 10 and first["overall"] == 1.0 and first["errors"] == 0

    counting = always_gold()
    calls = []
    original = counting.chat.completions.create
    counting.chat.completions.create = lambda **kw: (calls.append(1), original(**kw))[1]
    second = run_agent_evaluation(client=counting, **kwargs)
    assert calls == [] and second["overall"] == 1.0


def test_saying_none_for_an_empty_list_is_correct():
    rec = next(r for r in read_questions(ROOT / "data/questions/minirepo.jsonl")
               if r.answer_type == "list" and r.answer == [])
    for said in (["none"], ["None"], "none", [], ["n/a"]):
        assert score(rec, said) == 1.0, said
    assert score(rec, ["none", "minirepo.app.run"]) == 0.0


def test_time_budget_forces_an_answer(sandbox):
    rec = _rec(answer_type="bool")
    client = FakeClient([[("list_dir", {"path": "."})], [("submit_answer", {"answer": True})]])
    res = run_agent(client, "m", rec, condition="repo", question=rec.question,
                    sandbox=sandbox, package_root=".", max_seconds=-1)
    assert client.requests[0]["tool_choice"]["function"]["name"] == "submit_answer"
    assert res.answered and res.forced


def test_sampler_rotates_templates_and_differs_by_repo():
    from cgx.eval.harness import select_questions

    recs = read_questions(ROOT / "data/questions/minirepo.jsonl")
    picked = select_questions(recs, per_family=4, seed=0)
    by_family = {}
    for r in picked:
        by_family.setdefault(r.family, []).append(r.template_id)
    for family, templates in by_family.items():
        n_templates = len({r.template_id for r in recs if r.family == family})
        assert len(set(templates)) == min(4, n_templates), (family, templates)


def test_both_arms_end_with_the_same_format_line():
    from cgx.eval.harness import answer_format, question_text

    for rec in read_questions(ROOT / "data/questions/minirepo.jsonl"):
        rec.naturalization = {"status": "accepted"}
        rec.question_template, rec.question = rec.question, "A rephrased version."
        line = answer_format(rec)
        assert question_text(rec, "template").endswith(line)
        assert question_text(rec, "natural").endswith(line)


def test_final_turn_offers_only_the_submit_tool(sandbox):
    """Ollama ignores `tool_choice`, so forcing must not depend on it."""
    from cgx.eval import agent

    rec = _rec(answer_type="bool")
    client = FakeClient([[("list_dir", {"path": "."})]] * (agent.MAX_STEPS - 1)
                        + [[("submit_answer", {"answer": False})]])
    run_agent(client, "m", rec, condition="repo", question=rec.question, sandbox=sandbox, package_root=".")
    assert [t["function"]["name"] for t in client.requests[-1]["tools"]] == ["submit_answer"]
    assert len(client.requests[0]["tools"]) == 4


def test_ollama_spec_parses_without_credentials():
    from cgx.qgen.providers import parse_model_spec

    assert parse_model_spec("ollama:qwen3:8b") == ("ollama", "qwen3:8b")


def test_text_submission_is_accepted_only_when_unambiguous():
    from cgx.eval.agent import parse_text_submission as p

    assert p('submit_answer ["none"]', "submit_answer", "answer") == {"answer": ["none"]}
    assert p('`submit_answer({"answer": true})`', "submit_answer", "answer") == {"answer": True}
    assert p("submit_answer: 3", "submit_answer", "answer") == {"answer": 3}
    assert p("I will call submit_answer with 3 next.", "submit_answer", "answer") is None
    assert p("submit_answer maybe", "submit_answer", "answer") is None
    assert p("", "submit_answer", "answer") is None


def test_unparseable_tool_call_is_a_model_error_not_a_crash():
    """Ollama returns HTTP 500 for a tool call it cannot parse; the loop counts
    it as an invalid call, spends the step, and lets the model continue."""
    from types import SimpleNamespace as NS
    from cgx.eval.agent import run_tool_loop

    calls = {"n": 0}

    class Completions:
        def create(self, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Error code: 500 - error parsing tool call: raw='thinking...'")
            fn = NS(name="submit_answer", arguments='{"answer": true}')
            msg = NS(content="", tool_calls=[NS(id="c1", function=fn, model_dump=lambda: {})])
            return NS(choices=[NS(message=msg)], usage=None)

    client = NS(chat=NS(completions=Completions()))
    tool = {"type": "function", "function": {"name": "submit_answer", "parameters": {}}}
    res = run_tool_loop(client, "m", system="s", user="u", tools=[tool], final_tool="submit_answer",
                        required_key="answer", sandbox=None)
    assert res.answered and res.submitted == {"answer": True}
    assert res.invalid_tool_calls == 1 and res.steps == 2 and res.error is None


def test_xml_tool_call_parse_error_is_a_model_error():
    from cgx.eval.agent import is_tool_call_parse_error
    assert is_tool_call_parse_error(RuntimeError(
        "Error code: 500 - {'error': {'message': 'XML syntax error on line 6: unexpected EOF'}}"))
    assert not is_tool_call_parse_error(RuntimeError("APIConnectionError: Connection error."))


def test_no_answer_is_final_and_first_attempt_counts():
    from cgx.eval.harness import is_final
    assert is_final({"error": None}) and is_final({"error": "no answer submitted"})
    assert not is_final({"error": "APIConnectionError: Connection error."})


def _v2rec(template_id, answer, answer_type="list", **params):
    from cgx.qgen.base import make_question
    return make_question(template_id=template_id, repo_id="r", index=0, question="q",
                         answer=answer, answer_type=answer_type, cypher="", params=params)


def test_score_v2_resolves_answer_form_ambiguities():
    from cgx.eval.harness import GraphFacts, score, score_v2
    facts = GraphFacts(calls=set(), classes_without_init={"pkg.m.dataclassy"})

    files = _v2rec("organisation_subclass_files", ["recipes/gepa/method.py"])
    assert score(files, ["recipes/gepa/method.py (GEPAPlugin)"]) == 0.0
    assert score_v2(files, ["recipes/gepa/method.py (GEPAPlugin)"], facts) == 1.0

    src = _v2rec("t3_all_subclasses", ["packages.artemis-client.src.artemis_client.errors.ApiError"])
    assert score_v2(src, ["artemis_client.errors.ApiError"], facts) == 1.0

    over = _v2rec("inheritance_overriders", ["pkg.a.A", "pkg.b.B"], method="capabilities")
    assert score_v2(over, ["pkg.a.A.capabilities", "pkg.b.B.capabilities"], facts) == 1.0

    callers = _v2rec("comparison_reachable_difference", ["pkg.m.f"])
    assert score_v2(callers, ["pkg.m.f", "pkg.m.DataClassy.__init__"], facts) == 1.0
    # an __init__ that is neither implicit nor real still counts against the model
    assert score_v2(callers, ["pkg.m.f", "pkg.m.Other.__init__"], facts) < 1.0
    # and v2 agrees with v1 where nothing is ambiguous
    assert score_v2(callers, ["pkg.m.f"], facts) == score(callers, ["pkg.m.f"]) == 1.0
