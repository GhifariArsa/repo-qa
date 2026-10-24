# Repository QA taxonomy — version 2

Questions have one **subject family**, independently annotated **operations**,
relations, answer form and outcome. Families are not difficulty tiers.
`src/cgx/qgen/taxonomy.py` supplies the executable mapping.

## Five core families

| Family slug | Scope and assignment rule | Templates |
| --- | --- | --- |
| `definition_organisation` | Definition, location, direct ownership and declared membership | `t1_entity_type`, `t1_method_owner`, `t1_defining_file`, `t1_has_field`, `t3_class_method_count`, `t4_absent_entity` |
| `dependency_interaction` | Callers, callees, connectivity and call-graph summaries | `t2_path_exists`, `t2_shortest_chain`, `t2_hop_count`, `t2_reachable_within_2`, `t3_fan_in`, `t3_fan_out`, `t3_reach_set_size`, `t3_max_fan_in`, `t4_no_call_path`, `t4_no_callers` |
| `inheritance_specialisation` | Ancestry and declarations shared or redefined along inheritance | `t1_direct_base`, `t3_common_ancestor`, `t4_wrong_method_owner`, `t4_not_inherits`, `inheritance_redeclared_methods` |
| `state_access` | Field/global reads and writes, including through call paths | `t2_reaches_global_read`, `t5_closure_globals_read`, `t5_closure_writes_field` |
| `structural_comparison` | Explicit overlap or difference between two sets | `comparison_shared_callees`, `comparison_exclusive_callees`, `comparison_shared_members` |

All 27 templates across the five families are enabled by default. Old
`t1_` … `t5_` identifiers and filenames remain stable provenance names;
their prefixes no longer classify a record. Direct member counts concern
organisation; questions requiring ancestry concern inheritance. Explicit
two-set overlap/difference reports are structural comparisons. Finding a
common ancestor remains an inheritance question.

## Independent annotations

- `operations`: lookup, traversal, filtering, intersection, difference,
  counting, ranking, composition. Multiple tags are allowed.
- `relations`: edge types used to derive the answer.
- `answer_type`: bool, count, string, list (unordered set), path (ordered).
- `outcome`: positive/negative for booleans; empty/non_empty for lists/paths;
  zero/non_zero for counts; not_applicable for strings; unknown for missing
  answers. Unknown is not negative.
- `evidence`: existing paths, callers/callees, readers/writers and sampling
  metadata; comparisons add both input sets and redeclaration questions add
  matching ancestor declarations.
- `verification`: status, reviewer, note and timestamp.

Outcome is never supplied to the rewriting model. Personas depend only on
family. Comprehensive source-span and file/module-spread annotations remain
future work; the current change does not claim these are complete.

## New templates and contracts

| Template | Question | Answer |
| --- | --- | --- |
| Shared callees | Which in-repository callables do A and B both call directly? | Intersection of qualified callee names |
| Exclusive callees | Which callables does A call directly that B does not? | A's callee set minus B's; direction matters |
| Shared declared members | Which method names are declared directly in both classes? | Shared short names, excluding inherited methods |
| Redeclared methods | Which methods declared directly in C have the same name as an ancestor declaration within eight inheritance steps? | Qualified subclass method names |

Matching names do not establish matching behaviour or runtime dispatch.
Comparison sampling uses bounded random pairs with non-empty input sets and
targets roughly equal empty/non-empty answers. Shortfalls are reported, not
filled with another outcome. Bounded sampling may miss eligible cases.
Each stored comparison query returns the answer and supporting sets.

## Negatives and limitations

Negative generators now belong to their subject families. Existing template
IDs are preserved, so reclassification alone does not balance every template.
The field-membership and new comparison templates target both outcomes.
Other positive-only/negative-only generators still need matched sampling.

Graph absence is not proof of runtime impossibility. Legacy queries retain
their original bounds (calls 6/8/12, inheritance 8); some original wording
is broader than the query. Migration preserves this wording and does not
retroactively certify labels. Human source verification is still required.

Change/structural-impact questions are **not implemented**. A future extension
must define a graph intervention and before/after answer. Runtime breakage,
design rationale and full data-flow are outside the current core scope.

## Generation and reporting

```bash
cgx generate minirepo --per-family 12 --out data/questions/minirepo-v2.jsonl
cgx generate minirepo --family structural_comparison --per-template 6 --out data/questions/comparison.jsonl
cgx verify minirepo --questions data/questions/minirepo-v2.jsonl --family state_access
cgx evaluate minirepo --questions data/questions/minirepo-v2.jsonl --model oracle
```

Repeat `--family` to select several. `--per-family` overrides `--per-template`
and allocates a seeded round-robin budget across templates before queries run.
It is a target: singleton templates and candidate shortages can underfill it.
Targets and shortfalls are reported. With five families, a quota of 12 targets
at most 60 questions. The default without this option is five per template.

Evaluation reports family, operation, outcome and template scores. Operation
groups overlap. Lists use set F1; other answers use normalized exact match,
so these are mixed scores, not uniformly accuracy. Only stub/oracle evaluation
adapters currently ship.

## Legacy dataset migration

The reader classifies known old template IDs without changing the file.
IDs, answers, questions, Cypher, evidence, rewriting history and human review
are preserved. Old tier fields move to `legacy_taxonomy` on serialization.
Unknown templates and unsupported future taxonomy versions fail explicitly.

```bash
cgx migrate-taxonomy data/questions/minirepo.jsonl --out data/questions/minirepo-migrated-v2.jsonl
```

Migration writes only to a new file. Generation refuses existing output unless
`--overwrite` is supplied. Explicit regeneration creates fresh pending records;
ordinal IDs are not enough to safely merge reviews after graph/template changes.

## Research status

This is a proposed project taxonomy. CodeQueries motivates query-derived
labels, evidence and plausible negatives; broader QA literature motivates
separating intent, answer form and evidence location. Category clarity and
coverage still require a pilot with independent annotation and real developer
questions.
