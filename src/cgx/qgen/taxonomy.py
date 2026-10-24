"""Subject families and orthogonal operations for graph-grounded repository QA.

Template identifiers are stable provenance keys, not difficulty labels. Old
``t1_`` ... ``t5_`` identifiers remain valid so existing datasets can migrate
without changing answers, evidence, IDs, or human verification decisions.
"""

from dataclasses import dataclass

TAXONOMY_VERSION = 2
FAMILY_NAMES = {
    "definition_organisation": "Definition and organisation",
    "dependency_interaction": "Dependency and interaction",
    "inheritance_specialisation": "Inheritance and specialisation",
    "state_access": "State access",
    "structural_comparison": "Structural comparison",
}
DEFAULT_FAMILIES = tuple(FAMILY_NAMES)
OPERATIONS = frozenset({
    "lookup", "traversal", "filtering", "intersection", "difference",
    "counting", "ranking", "composition",
})


@dataclass(frozen=True)
class TemplateSpec:
    family: str
    operations: tuple[str, ...]
    relations: tuple[str, ...]


def _spec(family, operations, relations=()):
    return TemplateSpec(family, tuple(operations.split()), tuple(relations))


TEMPLATE_SPECS = {
    "t1_entity_type": _spec("definition_organisation", "lookup"),
    "t1_method_owner": _spec("definition_organisation", "lookup", ("HAS_METHOD",)),
    "t1_defining_file": _spec("definition_organisation", "lookup"),
    "t1_direct_base": _spec("inheritance_specialisation", "lookup", ("INHERITS",)),
    "t1_has_field": _spec("definition_organisation", "lookup", ("HAS_FIELD",)),
    "t2_path_exists": _spec("dependency_interaction", "traversal", ("CALLS",)),
    "t2_shortest_chain": _spec("dependency_interaction", "traversal", ("CALLS",)),
    "t2_hop_count": _spec("dependency_interaction", "traversal counting", ("CALLS",)),
    "t2_reachable_within_2": _spec("dependency_interaction", "traversal", ("CALLS",)),
    "t2_reaches_global_read": _spec("state_access", "traversal filtering composition", ("CALLS", "USES")),
    "t3_fan_in": _spec("dependency_interaction", "lookup counting", ("CALLS",)),
    "t3_fan_out": _spec("dependency_interaction", "lookup counting", ("CALLS",)),
    "t3_reach_set_size": _spec("dependency_interaction", "traversal counting", ("CALLS",)),
    "t3_common_ancestor": _spec("inheritance_specialisation", "traversal intersection ranking", ("INHERITS",)),
    "t3_max_fan_in": _spec("dependency_interaction", "counting ranking", ("CALLS",)),
    "t3_class_method_count": _spec("definition_organisation", "lookup counting", ("HAS_METHOD",)),
    "t4_absent_entity": _spec("definition_organisation", "lookup"),
    "t4_no_call_path": _spec("dependency_interaction", "traversal", ("CALLS",)),
    "t4_wrong_method_owner": _spec("inheritance_specialisation", "traversal filtering composition", ("INHERITS", "HAS_METHOD")),
    "t4_no_callers": _spec("dependency_interaction", "lookup", ("CALLS",)),
    "t4_not_inherits": _spec("inheritance_specialisation", "traversal", ("INHERITS",)),
    "t5_closure_globals_read": _spec("state_access", "traversal filtering composition", ("CALLS", "USES")),
    "t5_closure_writes_field": _spec("state_access", "traversal filtering composition", ("CALLS", "USES")),
    "comparison_shared_callees": _spec("structural_comparison", "lookup intersection", ("CALLS",)),
    "comparison_exclusive_callees": _spec("structural_comparison", "lookup difference", ("CALLS",)),
    "comparison_shared_members": _spec("structural_comparison", "lookup intersection", ("HAS_METHOD",)),
    "inheritance_redeclared_methods": _spec("inheritance_specialisation", "traversal filtering composition", ("INHERITS", "HAS_METHOD")),
    "t2_transitive_callers": _spec("dependency_interaction", "traversal", ("CALLS",)),
    "t2_files_touched": _spec("dependency_interaction", "traversal composition", ("CALLS", "CONTAINS")),
    "t3_all_subclasses": _spec("inheritance_specialisation", "traversal", ("INHERITS",)),
    "comparison_reachable_difference": _spec("structural_comparison", "traversal difference", ("CALLS",)),
    "comparison_reachable_shared": _spec("structural_comparison", "traversal intersection", ("CALLS",)),
    "inheritance_overriders": _spec("inheritance_specialisation", "traversal filtering", ("INHERITS", "HAS_METHOD")),
    "organisation_subclass_files": _spec("definition_organisation", "traversal composition", ("INHERITS", "CONTAINS")),
}

#: Templates that need a repo-wide or multi-hop search: they get two slots in a
#: family's round-robin quota, one-hop lookups one. In the pilot, lookups were
#: solved almost every time and only these separated models.
HARD_TEMPLATES = frozenset({
    "t2_shortest_chain", "t2_hop_count", "t2_reachable_within_2", "t2_reaches_global_read",
    "t2_transitive_callers", "t2_files_touched", "t3_reach_set_size", "t3_common_ancestor",
    "t3_all_subclasses", "t5_closure_globals_read", "t5_closure_writes_field",
    "inheritance_redeclared_methods", "comparison_reachable_difference",
    "comparison_reachable_shared", "inheritance_overriders", "organisation_subclass_files",
})


def template_weight(template_id: str) -> int:
    """Three slots for a repo-wide or transitive template, one for a lookup.
    The lookups stay as a calibration tier; they no longer carry the score."""
    return 3 if template_id in HARD_TEMPLATES else 1


#: Templates whose answer is written on the question itself, so no wording can
#: hide it: `t1_entity_type` (the name's shape -- `PRESET_STATE_KEYS`,
#: `EmailServers` -- gives the kind) and `t1_method_owner` (naming which method
#: requires naming its class, which is the answer). Both scored 1.00 closed-book
#: in the pilot, and the agentic naturaliser correctly refused to rewrite them.
#: Off by default; `include_trivial=True` keeps them as a labelled control.
TRIVIAL_TEMPLATES = frozenset({"t1_entity_type", "t1_method_owner"})


def template_spec(template_id: str) -> TemplateSpec:
    try:
        return TEMPLATE_SPECS[template_id]
    except KeyError:
        raise ValueError(f"unknown template {template_id!r}; declare its taxonomy first") from None


def validate_families(families):
    families = tuple(families)
    unknown = set(families) - FAMILY_NAMES.keys()
    if unknown:
        raise ValueError(f"unknown families: {', '.join(sorted(unknown))}; choose from {', '.join(FAMILY_NAMES)}")
    return families


def answer_outcome(answer, answer_type: str) -> str:
    """Polarity is an answer property, never a family or difficulty level."""
    if answer is None:
        return "unknown"
    if answer_type == "bool":
        return "positive" if answer else "negative"
    if answer_type in {"list", "path"}:
        return "non_empty" if answer else "empty"
    if answer_type == "count":
        return "zero" if answer == 0 else "non_zero"
    return "not_applicable"
