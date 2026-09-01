"""Validators that gate an LLM rewrite before it can enter the dataset.

The benchmark's claim is that ground truth comes from the graph. An LLM that
rewrites a question can silently break that in three ways, and each has a
deterministic check here — no second model needed:

1. **Answer leakage.** The rewrite states or strongly implies the answer, so the
   question is answerable without consulting the repository. Detected by looking
   for answer tokens that appear in the rewrite but *not* in the template it came
   from -- comparing against the template is what makes this precise, since many
   templates legitimately name the entity in the answer's own namespace.

2. **Referent drift.** The rewrite drops or renames an entity the Cypher query
   was parameterised on, so the stored answer no longer answers the stored
   question. A short name is accepted only when it is unambiguous repo-wide.

3. **Invented entities.** The rewrite names a class, function, or module that
   does not exist. Every identifier in the rewrite is checked against the graph.

A fourth check -- whether the rewrite still *means* the same thing -- is not
decidable this way and is handled by an LLM judge in `naturalize.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .base import QuestionRecord

#: Backticked spans and bare dotted identifiers.
_BACKTICK = re.compile(r"`([^`]+)`")
_DOTTED = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")


@dataclass
class RepoVocabulary:
    """Every name the graph knows about, for one repo."""

    qnames: set[str] = field(default_factory=set)
    file_paths: set[str] = field(default_factory=set)
    #: short name -> how many distinct entities share it
    short_name_counts: dict[str, int] = field(default_factory=dict)

    def is_unambiguous(self, short_name: str) -> bool:
        return self.short_name_counts.get(short_name, 0) == 1

    def knows(self, token: str) -> bool:
        return (
            token in self.qnames
            or token in self.file_paths
            or token in self.short_name_counts
            or any(q.endswith("." + token) for q in self.qnames)
        )


_VOCAB_CYPHER = """
MATCH (n:CodeNode {repo_id: $repo})
RETURN n.qname AS qname, n.name AS name, n.file_path AS file_path
"""


def load_vocabulary(client, repo_id: str) -> RepoVocabulary:
    vocab = RepoVocabulary()
    for row in client.run(_VOCAB_CYPHER, repo=repo_id):
        if row["qname"]:
            vocab.qnames.add(row["qname"])
        if row["file_path"]:
            vocab.file_paths.add(row["file_path"])
        if row["name"]:
            vocab.short_name_counts[row["name"]] = vocab.short_name_counts.get(row["name"], 0) + 1
    return vocab


def _answer_tokens(rec: QuestionRecord) -> list[str]:
    """Strings whose presence would give the answer away.

    Booleans are excluded: a yes/no answer has no surface form to leak, and the
    words "yes"/"no" appear in ordinary phrasing.
    """
    if rec.answer_type == "bool":
        return []
    if rec.answer_type == "count":
        return [str(rec.answer)]
    if rec.answer_type == "path":
        # The endpoints of a path ARE the question -- "what is the chain from A
        # to B" must name A and B. Only the interior nodes are the answer, so
        # only they can be leaked. Counting the endpoints made every
        # t2_shortest_chain rewrite that shortened a qualified name look like a
        # leak, and fall back to the template.
        chain = [str(x) for x in (rec.answer or [])]
        interior = chain[1:-1]
        tokens = []
        for text in interior:
            tokens.append(text)
            if "." in text:
                tokens.append(text.rsplit(".", 1)[-1])
        return tokens
    if rec.answer_type == "list":
        tokens = []
        for item in rec.answer or []:
            text = str(item)
            tokens.append(text)
            if "." in text:
                tokens.append(text.rsplit(".", 1)[-1])
        return tokens
    if rec.answer in (None, ""):
        return []
    text = str(rec.answer)
    tokens = [text]
    if "." in text:
        tokens.append(text.rsplit(".", 1)[-1])
    return tokens


def _template_mentions(template: str, token: str) -> bool:
    """Is `token` already information the template gave away?

    A bare short name counts as present when the template names the same entity
    in qualified form: `refresh_from_db` tells the reader nothing new when the
    template already said `django.db.models.base.Model.refresh_from_db`. Without
    this, a rewrite that legitimately shortens an unambiguous name -- which the
    persona instructions explicitly allow -- reads as a leak.
    """
    return _mentions(template, token)


def _identifier_like(token: str) -> bool:
    """Would this token be mistaken for an ordinary English word?

    `refresh_from_db` and `FileBasedCache` could only be code. `set`, `get`,
    and `name` are also verbs and nouns, so a bare occurrence proves nothing --
    those are only counted when they appear in an unambiguously code context.

    Numeric answers are always matched bare: a count question whose rewrite
    contains the count has leaked it, however it is punctuated.
    """
    if token.isdigit():
        return True
    return "_" in token or any(c.isupper() for c in token[1:])


def _mentions(text: str, token: str) -> bool:
    """Does `text` refer to the entity `token` names, in any form?

    Matches the qualified name, and a short name written either bare (when the
    short name could not be prose) or in code context -- backticked, dotted
    onto a parent, or followed by a call's parentheses.
    """
    whole = rf"(?<![\w]){re.escape(token)}(?![\w])"
    if "." in token or _identifier_like(token):
        return bool(re.search(whole, text))
    escaped = re.escape(token)
    return bool(re.search(rf"`{escaped}`|\.{escaped}(?![\w])|{escaped}\s*\(", text))


def check_answer_leak(rec: QuestionRecord, rewrite: str) -> list[str]:
    template = rec.question
    problems = []
    for token in _answer_tokens(rec):
        if not token:
            continue
        if not _mentions(rewrite, token):
            continue
        if _template_mentions(template, token):
            continue
        problems.append(f"leaks answer token {token!r} that the template did not contain")
    return problems


def check_referents(rec: QuestionRecord, rewrite: str, vocab: RepoVocabulary) -> list[str]:
    """Every entity the Cypher is parameterised on must still be identifiable."""
    problems = []
    for key, value in rec.params.items():
        if key == "repo" or not isinstance(value, str):
            continue
        if value not in vocab.qnames and value not in vocab.file_paths:
            continue                      # not an entity reference (e.g. a bare field name)
        if value in rewrite:
            continue
        short = value.rsplit(".", 1)[-1]
        if short in rewrite:
            if vocab.is_unambiguous(short):
                continue
            problems.append(
                f"refers to {short!r} by short name, but {vocab.short_name_counts.get(short)} "
                f"entities share it — the question is ambiguous"
            )
        else:
            problems.append(f"drops the referent {value!r} that the ground-truth query depends on")
    return problems


def check_invented_entities(rewrite: str, vocab: RepoVocabulary) -> list[str]:
    """Any code identifier in the rewrite must exist in the graph."""
    problems = []
    candidates = set(_BACKTICK.findall(rewrite)) | set(_DOTTED.findall(rewrite))
    for token in candidates:
        token = token.strip().rstrip("()")
        if not token or token.isdigit():
            continue
        if vocab.knows(token):
            continue
        # A dotted path whose head is known but whose tail is not is the worst
        # case: it looks real and is not.
        problems.append(f"names {token!r}, which does not exist in the extracted graph")
    return problems


def validate_rewrite(rec: QuestionRecord, rewrite: str, vocab: RepoVocabulary) -> list[str]:
    """All deterministic checks. Empty list means the rewrite is admissible."""
    if not rewrite or not rewrite.strip():
        return ["empty rewrite"]
    return (
        check_answer_leak(rec, rewrite)
        + check_referents(rec, rewrite, vocab)
        + check_invented_entities(rewrite, vocab)
    )
