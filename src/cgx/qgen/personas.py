"""Personas — the register a real developer would ask a question in.

A templated question ("How many distinct functions or methods call `X`
directly?") is unambiguous but nobody talks like that. A persona reframes the
same graph fact as the situation that would actually prompt someone to ask it:
a refactor impact check, an onboarding question, a debugging trace.

The persona changes *only the framing*. The referent entities and the answer
are fixed by the graph before the model ever sees the question, and the
validators in `validate.py` enforce that the rewrite preserved them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    id: str
    who: str
    situation: str


PERSONAS: dict[str, Persona] = {
    "plain": Persona(
        id="plain",
        who="a developer who knows this codebase",
        situation=(
            "Asking a colleague a direct, factual question about the code. No preamble, "
            "no explanation of why. One or two sentences."
        ),
    ),
    "newcomer": Persona(
        id="newcomer",
        who="an engineer who joined the team this week",
        situation=(
            "Reading the codebase for the first time and orienting. May mention being new, "
            "or say they are trying to understand how a piece fits together."
        ),
    ),
    "refactorer": Persona(
        id="refactorer",
        who="an engineer about to change this code",
        situation=(
            "Assessing blast radius before editing something — who depends on this, what "
            "breaks if the signature changes, how much work is this. Frames the question "
            "around the change they intend to make."
        ),
    ),
    "debugger": Persona(
        id="debugger",
        who="an engineer tracing a bug",
        situation=(
            "Following execution to work out how control reaches somewhere, or whether it "
            "can at all. Frames the question around the symptom or the trace they are chasing."
        ),
    ),
    "reviewer": Persona(
        id="reviewer",
        who="an engineer reviewing a pull request in this repo",
        situation=(
            "Checking a claim about the structure of the code before approving — verifying "
            "something exists, is reachable, or is actually unused."
        ),
    ),
}

#: Personas follow the subject family, independently of answer polarity.
FAMILY_PERSONAS: dict[str, tuple[str, ...]] = {
    "definition_organisation": ("plain", "newcomer", "reviewer"),
    "dependency_interaction": ("debugger", "refactorer", "plain"),
    "inheritance_specialisation": ("newcomer", "reviewer", "plain"),
    "state_access": ("debugger", "refactorer", "plain"),
    "structural_comparison": ("refactorer", "reviewer", "plain"),
}


def personas_for_family(family: str) -> tuple[Persona, ...]:
    return tuple(PERSONAS[p] for p in FAMILY_PERSONAS.get(family, ("plain",)))
