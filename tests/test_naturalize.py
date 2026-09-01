"""Naturalizer tests. No API key needed — the provider is stubbed.

The point of these is the *gates*: a rewrite that leaks the answer, drops a
referent, or invents an entity must never reach the dataset, and the record
must fall back to its templated phrasing when nothing survives.
"""

import pytest

REPO = "minirepo"


@pytest.fixture(scope="module")
def loaded(neo4j_client, minigraph):
    from cgx.graph.ingest import ingest

    graph, _ = minigraph
    neo4j_client.apply_schema()
    ingest(neo4j_client, graph.to_dict(), replace=True)
    return neo4j_client


@pytest.fixture(scope="module")
def vocab(loaded):
    from cgx.qgen.validate import load_vocabulary

    return load_vocabulary(loaded, REPO)


@pytest.fixture
def fan_in_question(loaded):
    """A tier-3 fan-in question about Base.__init__ (answer: 2)."""
    import random
    from cgx.qgen.tier3_aggregate import t3_fan_in

    for _ in range(30):
        for rec in t3_fan_in(loaded, REPO, 20, random.Random(0)):
            if rec.params["target"] == "minirepo.models.Base.__init__":
                return rec
    pytest.skip("fixture question not sampled")


class FakeProvider:
    """Returns canned phrasings, then canned judgements."""

    name = "fake"
    model = "fake:test"

    def __init__(self, phrasings, same_answer=True):
        self.phrasings = phrasings
        self.same_answer = same_answer
        self.calls = []

    def parse(self, system, user, schema, max_tokens):
        self.calls.append({"system": system[:40], "user": user})
        if schema.__name__ == "Rewrites":
            return schema(phrasings=[{"persona": "plain", "question": q} for q in self.phrasings])
        return schema(same_answer=self.same_answer, reason="stub")


def _naturalize(rec, vocab, phrasings, same_answer=True, judge=True):
    from cgx.qgen.naturalize import NaturalizationConfig, Naturalizer

    provider = FakeProvider(phrasings, same_answer=same_answer)
    n = Naturalizer(NaturalizationConfig(judge=judge), provider=provider)
    return n.naturalize(rec, vocab), provider


def _path_record(repo="minirepo"):
    """A t2_shortest_chain-shaped record: endpoints qualified, chain of three."""
    from cgx.qgen.base import make_question

    chain = ["pkg.a.Alpha.start", "pkg.b.Middle.hop", "pkg.c.Omega.finish"]
    return make_question(
        tier=2, template_id="t2_shortest_chain", repo_id=repo, index=0,
        question=("What is the shortest chain of calls from `pkg.a.Alpha.start` to "
                  "`pkg.c.Omega.finish`? List every callable on the path in order."),
        answer=chain, answer_type="path",
        cypher="RETURN 1", params={"repo": repo, "source": chain[0], "target": chain[-1]},
    )


def test_path_endpoints_are_not_treated_as_leaked_answer():
    """The endpoints ARE the question -- naming them cannot be a leak.

    Counting them made every t2_shortest_chain rewrite that shortened a
    qualified name fall back to the template, silently costing the tier its
    natural phrasings.
    """
    from cgx.qgen.validate import check_answer_leak

    rec = _path_record()
    rewrite = ("I'm tracing through start and trying to work out how execution could "
               "reach finish from there. What's the shortest call sequence?")
    assert check_answer_leak(rec, rewrite) == []


def test_interior_path_node_is_still_a_leak():
    """Only the interior of the chain is the answer -- it must stay protected."""
    from cgx.qgen.validate import check_answer_leak

    rec = _path_record()
    assert check_answer_leak(rec, "Does it go via `pkg.b.Middle.hop`?")
    assert check_answer_leak(rec, "Does it go via Middle.hop?")


def test_common_word_short_names_do_not_false_positive():
    """`set` is a verb as well as a method name; bare prose use is not a leak."""
    from cgx.qgen.base import make_question
    from cgx.qgen.validate import check_answer_leak

    rec = make_question(
        tier=2, template_id="t2_shortest_chain", repo_id="minirepo", index=0,
        question="What is the shortest chain from `pkg.a.Alpha.start` to `pkg.c.Omega.finish`?",
        answer=["pkg.a.Alpha.start", "pkg.b.Cache.set", "pkg.c.Omega.finish"],
        answer_type="path", cypher="RETURN 1", params={"repo": "minirepo"},
    )
    assert check_answer_leak(rec, "Give me the set of steps from start to finish.") == []
    assert check_answer_leak(rec, "Does it pass through `set`?")          # code context
    assert check_answer_leak(rec, "Does it pass through Cache.set?")      # dotted


def test_good_rewrite_is_accepted_and_template_preserved(fan_in_question, vocab):
    good = ("I'm about to change the constructor of `minirepo.models.Base` — how many "
            "functions or methods call `minirepo.models.Base.__init__` directly?")
    template = fan_in_question.question
    rec, _ = _naturalize(fan_in_question, vocab, [good])
    assert rec.naturalization["status"] == "accepted"
    assert rec.question == good
    assert rec.question_template == template     # original retained, rewrite is auditable
    assert rec.answer == 2                        # answer untouched by the model


def test_answer_leak_is_rejected(fan_in_question, vocab):
    """A rewrite that states the count makes the question self-answering."""
    leaky = ("Do exactly 2 functions call `minirepo.models.Base.__init__` directly? "
             "How many call it?")
    rec, _ = _naturalize(fan_in_question, vocab, [leaky])
    assert rec.naturalization["status"] == "fell_back_to_template"
    attempt = rec.naturalization["rejected_attempts"][0]
    assert attempt["rejected_by"] == "validators"
    assert any("leaks answer token" in p for p in attempt["problems"])


def test_dropped_referent_is_rejected(fan_in_question, vocab):
    """Losing the entity means the stored Cypher no longer answers the question."""
    vague = "How many things call that constructor directly?"
    rec, _ = _naturalize(fan_in_question, vocab, [vague])
    assert rec.naturalization["status"] == "fell_back_to_template"
    assert any("drops the referent" in p
               for p in rec.naturalization["rejected_attempts"][0]["problems"])


def test_invented_entity_is_rejected(fan_in_question, vocab):
    hallucinated = ("How many callers does `minirepo.models.Base.__init__` have, "
                    "counting `minirepo.models.Session.__init__`?")
    rec, _ = _naturalize(fan_in_question, vocab, [hallucinated])
    assert rec.naturalization["status"] == "fell_back_to_template"
    assert any("does not exist in the extracted graph" in p
               for p in rec.naturalization["rejected_attempts"][0]["problems"])


def test_judge_can_veto_a_rewrite_the_validators_passed(fan_in_question, vocab):
    """Meaning drift survives the deterministic checks; the judge is the backstop."""
    drifted = ("How many functions can reach `minirepo.models.Base.__init__` through "
               "any number of calls?")          # transitive, not direct — different answer
    rec, _ = _naturalize(fan_in_question, vocab, [drifted], same_answer=False)
    assert rec.naturalization["status"] == "fell_back_to_template"
    assert rec.naturalization["rejected_attempts"][0]["rejected_by"] == "judge"


def test_falls_back_to_first_surviving_candidate(fan_in_question, vocab):
    bad = "How many things call it?"
    good = "How many functions call `minirepo.models.Base.__init__` directly?"
    rec, _ = _naturalize(fan_in_question, vocab, [bad, good])
    assert rec.naturalization["status"] == "accepted"
    assert rec.question == good
    assert len(rec.naturalization["rejected_attempts"]) == 1


def test_answer_is_never_sent_to_the_model(fan_in_question, vocab):
    """The model must not be able to leak what it was never told."""
    _, provider = _naturalize(fan_in_question, vocab,
                              ["How many callers does `minirepo.models.Base.__init__` have?"])
    rewrite_prompt = provider.calls[0]["user"]
    # The templated question legitimately contains the word "answer" ("Answer
    # with a single integer"); what must be absent is the answer *value*.
    import re
    assert not re.search(rf"(?<!\w){re.escape(str(fan_in_question.answer))}(?!\w)", rewrite_prompt)
    assert "evidence" not in rewrite_prompt.lower()   # callers list would give it away
    for caller in fan_in_question.evidence.get("callers", []):
        assert caller not in rewrite_prompt


def test_ambiguous_short_name_is_rejected(loaded, vocab):
    """`__init__` is shared by five classes — a bare short name is not a referent."""
    import random
    from cgx.qgen.tier3_aggregate import t3_fan_in

    rec = next(r for r in t3_fan_in(loaded, REPO, 30, random.Random(1))
               if r.params["target"].endswith(".__init__"))
    rec2, _ = _naturalize(rec, vocab, ["How many functions call `__init__` directly?"])
    assert rec2.naturalization["status"] == "fell_back_to_template"


def test_model_spec_parsing_and_provider_selection():
    from cgx.qgen.providers import parse_model_spec

    assert parse_model_spec("anthropic:claude-opus-5") == ("anthropic", "claude-opus-5")
    assert parse_model_spec("openai:gpt-4.1") == ("openai", "gpt-4.1")
    assert parse_model_spec("claude-opus-5") == ("anthropic", "claude-opus-5")
    with pytest.raises(SystemExit):
        parse_model_spec("gpt-4.1")          # bare non-claude id is ambiguous
    with pytest.raises(SystemExit):
        parse_model_spec("cohere:command")   # unsupported provider
