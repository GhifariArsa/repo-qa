# Repo-QA

A repository-level code-QA benchmark whose ground truth is **derived from a
codebase's graph structure**, not mined from GitHub issues (SWE-QA), Stack
Overflow (StackRepoQA), or agent exploration (CoReQA). Every question ships
with the Cypher query that produced its answer, so the label is reproducible
and auditable rather than adjudicated.

Pipeline: `pyan3 static analysis -> CodexGraph-schema graph (+ CALLS) -> Neo4j
-> family-classified question generator -> optional LLM rephrasing`.

## Schema

Follows the CodexGraph schema (Liu et al., 2024, [arXiv:2408.03910](https://arxiv.org/abs/2408.03910))
with two deliberate additions, both marked below.

| Nodes             | Attributes                                      |
| ----------------- | ----------------------------------------------- |
| `REPOSITORY`      | **our addition** — scopes a multi-repo database |
| `MODULE`          | `name` (dotted), `file_path`                    |
| `CLASS`           | `name`, `file_path`, `signature`, `code_ref`    |
| `FUNCTION`        | `name`, `file_path`, `signature`, `code_ref`    |
| `METHOD`          | as FUNCTION, plus `class_qname`                 |
| `FIELD`           | `name`, `file_path`, `class_qname`              |
| `GLOBAL_VARIABLE` | `name`, `file_path`, `code_ref`                 |

| Edges        | Direction                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------- |
| `CONTAINS`   | `REPOSITORY -> MODULE` (**our addition**), `MODULE -> CLASS \| FUNCTION \| GLOBAL_VARIABLE` |
| `HAS_METHOD` | `CLASS -> METHOD`                                                                           |
| `HAS_FIELD`  | `CLASS -> FIELD`                                                                            |
| `INHERITS`   | `CLASS -> CLASS`                                                                            |
| `USES`       | `FUNCTION \| METHOD -> GLOBAL_VARIABLE \| FIELD`, with `access` = read/write                |
| `CALLS`      | `FUNCTION \| METHOD -> FUNCTION \| METHOD` (**our addition**)                               |

`CALLS` is the substantive extension. CodexGraph's own authors flag its absence:
_"adding edges related to function calls will enhance the broader applicability
of CodexGraph."_ Call traversal, dependency aggregation and state-access composition depend on it.

Following the paper, `code_ref` stores a **pointer** into the source
(`path:start-end`), not the snippet text.

`src/cgx/schema.py` is the single source of truth. `validate_graph()` rejects
illegal endpoint pairs at extraction time — a schema violation fails the run
rather than silently poisoning a question's ground truth. This is not
theoretical: it is what caught the qname-collision bug below on the first real
repository we pointed it at.

**A qname is not a unique key.** `EDGE_IDENTITY_ATTRS` and node ids both key on
_(label, qname)_, because real Python routinely gives one qualified name two
different kinds of node — see Extraction and Ingest below.

## Extraction: why two passes

pyan3 alone cannot produce this schema, and taking its output at face value
produces **wrong** ground truth. Two failure modes, both handled:

1. **Inheritance leaks into `uses_edges`.** pyan emits `CLASS Admin -> CLASS User`
   as a "uses" edge. That is `INHERITS`. Every edge with a `CLASS` source is dropped.

2. **Field access is misattributed to the defining method.** pyan has no field
   nodes, so `self.permissions` inside `Admin.grant` resolves to `Admin.__init__`
   — the method where the attribute is first bound. Read naively, pyan claims
   `Admin.grant` calls `Admin.__init__`. It does not.

The fix for (2) is to treat pyan as a **resolution** oracle, not a **relation**
oracle. A resolved edge becomes `CALLS` only if the AST pass recorded a call
site in the source body whose syntactic callee matches the target. Constructor
calls are handled explicitly, since `User(clean)` is written as a class name but
resolves to `User.__init__`.

So: the **AST pass** (`extract/ast_pass.py`) owns nodes, attributes, `CONTAINS`,
`HAS_METHOD`, `HAS_FIELD`, `INHERITS`, and `USES -> FIELD`; the **pyan pass**
(`extract/pyan_pass.py`) owns cross-file resolution into `CALLS` and
`USES -> GLOBAL_VARIABLE`. On the fixture this yields all 15 real call edges and
rejects exactly the 2 artifacts (`tests/test_extract.py` asserts both sets exactly).

### Three things real repositories broke

Each of these passed on the fixture and failed on the first real codebase.

**Property/attribute qname collisions.** `self._socket = sock` in `__init__`
defines a FIELD; `@property def _socket` defines a METHOD. Same qualified name,
two node kinds. The node index was keyed by qname alone, so the second
registration silently returned the first node and a `HAS_METHOD` edge ended up
aimed at a FIELD. `NodeIndex` now keys on `(label, qname)` — mirroring the id
scheme the schema always used — and every resolution site states which kind it
expects (`get(qname, CLASS)`, `get(qname, FUNCTION, METHOD)`), so nothing is
resolved by guesswork. The neo4j driver has 6 such collisions; Django has more.

**Files that do not parse.** The AST pass skipped them, then handed the
_original_ file list to pyan, which builds a symbol table with
`symtable.symtable()` and raises on anything that will not compile. Django ships
`tests/test_runner_apps/tagged/tests_syntax_error.py` deliberately, to exercise
its own test runner. `AstResult.parsed_files` is now the single source of truth
for what reaches pyan, and dropped files are logged as warnings rather than
swallowed.

**Test directories.** `tests`/`test`/`testing` are excluded by default
(`--include-tests` keeps them). On Django that is 2929 → 912 files. Test trees
dominate file counts on real projects, so including them would skew every
sampled question toward fixtures; they also deliberately contain pathological
code. Exclusions match **repo-relative** path components — matching absolute
components meant a repo that merely lived under a directory named `tests/`
extracted to an empty graph, silently, depending only on where it sat on disk.

## Ingest: two correctness bugs worth knowing about

**`CodeNode.id` must be indexed.** Edge ingest resolves endpoints with
`MATCH (s:CodeNode {id: …})`. The per-label uniqueness constraints index
`(:METHOD, id)`, `(:CLASS, id)` and so on, but nothing covered `(:CodeNode, id)`
— so the planner fell back to `NodeByLabelScan` and edge ingest was
O(nodes × edges). Django took an extrapolated ~23 minutes; it now takes ~5s.

**`apply_schema` was silently dropping statements.** It split `schema.cypher` on
`;` and discarded chunks starting with `//` — but after splitting, each chunk
_begins_ with the comment block that preceded it, so every documented statement
was skipped. Three never ran, including the index above. It now strips comment
lines per statement instead: 16 statements declared, 16 applied.

**Edge identity vs edge payload.** `MERGE (s)-[r:USES]->(t)` keeps one
relationship per pair, so a method that both reads _and_ writes the same field
collapsed to one edge with `access` overwritten by whichever arrived last — on
Django, 510 edges, silently corrupting the read/write distinction that state access
filters on. `EDGE_IDENTITY_ATTRS` in `schema.py` declares which attributes are
part of an edge's identity, and those go into the MERGE pattern.

Nodes carry two labels: their schema label and a shared `:CodeNode`, so
traversals that do not care about the kind stay simple and still hit an index.

## Question families (taxonomy v2)

Questions have one subject family plus independent operation and outcome tags.
Families are not difficulty levels.

| Family | Coverage |
| --- | --- |
| `definition_organisation` | Definitions, files, direct ownership and members |
| `dependency_interaction` | Callers/callees, paths, reachability and counts |
| `inheritance_specialisation` | Ancestry and same-name member declarations |
| `state_access` | Field/global reads and writes through call paths |
| `structural_comparison` | Shared/exclusive callees and shared declared methods |

All five families are enabled by default: 23 existing templates plus four new
ones. Records carry `family`, `operations`, `relations`, `outcome`,
`taxonomy_version`, the existing QA/query/evidence fields, and review history.
Old template IDs and module filenames remain provenance keys. See
[QUESTION_TAXONOMY.md](QUESTION_TAXONOMY.md) for mapping and answer contracts.

Use repeatable `--family <slug>` to select families and `--per-family 12`
to target at most 60 questions across five families. Budgets are allocated
across templates before queries execute; shortages are reported explicitly.

```bash
cgx generate minirepo --per-family 12 --out data/questions/minirepo-v2.jsonl
cgx migrate-taxonomy data/questions/minirepo.jsonl --out data/questions/minirepo-migrated-v2.jsonl
```

Old JSONL is readable directly. Migration writes a new file and preserves IDs,
answers, evidence and reviews; original tiers are retained in `legacy_taxonomy`.
Generation protects existing output unless `--overwrite` is explicitly given.
Evaluation reports family, operation, outcome and template scores; operation
groups overlap. Scores combine list F1 and exact match, not uniform accuracy.

## Sampling: what gets turned into a question

Deriving ground truth from a graph is only half the method; **which instances
are selected** is the other half, and it decides whether a question family is discriminative
or trivial. Uniform random sampling over a code graph produces a benchmark
dominated by boilerplate, because the underlying distributions are heavily
zero-inflated.

Measured fraction of **callables** clearing each threshold:

|                          | fan_in ≥ 2 | fan_out ≥ 2 | reach ≥ 2 |
| ------------------------ | ---------- | ----------- | --------- |
| minirepo (21 callables)  | 14.3%      | 14.3%       | 28.6%     |
| neo4j driver (1765)      | 17.5%      | 23.3%       | 32.8%     |
| Django (9225)            | 16.9%      | 21.6%       | 35.4%     |

~65% of real-repo callables have fan-in 0, so two-thirds of a uniform aggregation
sample would ask "how many callers?" about something nothing calls.

Class method counts are less skewed but degenerate often enough to matter —
note the denominator here is **classes**, not callables:

|                       | 0 methods | 1 method | method_count ≥ 2 |
| --------------------- | --------- | -------- | ---------------- |
| minirepo (5 classes)  | 0         | 1        | 80.0%            |
| neo4j driver (302)    | 76        | 51       | 57.9%            |
| Django (1957)         | **580 (30%)** | 373 (19%) | 51.3%       |

Nearly a third of Django's classes define no methods of their own — marker
subclasses, lookups, `Meta`-style attribute holders. Uniform sampling made
`t3_class_method_count` answer 0 for 7 of 12 questions.

**Aggregation templates sample the non-trivial tail.** Each metric is computed across all
candidates in one pass, and sampling draws only from those clearing a threshold
— currently 2 for all four: `MIN_FAN_IN`, `MIN_FAN_OUT`, `MIN_REACH`,
`MIN_METHOD_COUNT`. Every such record carries `evidence.sampled_from` recording
the threshold it was drawn under, so the filter is visible in the dataset rather
than implicit in the code. When a repo has fewer non-trivial candidates than
requested it returns fewer questions rather than quietly relaxing the threshold.
Zero-valued cases are not lost — they are `t4_no_callers` in the dependency family, where
"nothing calls this" is the point.

The remaining two aggregation templates need no threshold: `t3_common_ancestor` is
structurally pre-filtered (its sampling query only matches class pairs that
genuinely share an ancestor) and `t3_max_fan_in` computes one repo-wide answer
without sampling.

**Path templates are stratified by hop length.** Sampling draws a quota per hop bucket
(`2`, `3`, `4+`) rather than from one pool, so accuracy-by-hop-length is a
reportable, unconfounded result instead of an artefact of graph shape. Each
record carries `evidence.hop_bucket`. Candidates come from bounded random source
anchors expanded outward, not from the full pair space — Django has 9225
callables, so that space is 85,100,625 pairs.

One trap worth knowing: a `WHERE ... length(p) >= 2` in the same clause as
`shortestPath()` gets pushed into the path search, so Neo4j returns the shortest
path *satisfying the predicate* rather than the true shortest — silently
labelling genuinely 1-hop pairs as 2-hop. The filter must come after a `WITH`.

**Shortest-chain questions require a unique shortest path.** `shortestPath()` returns one
arbitrary path when several tie, which would make the stored chain only one of
several correct answers. `t2_shortest_chain` checks with `allShortestPaths()`
and skips tied pairs. `t2_path_exists` and `t2_hop_count` are unaffected: a
boolean and a length are identical across tied paths.

**Answer polarity is not predictable from the template.** `t1_has_field` used to
sample only from existing `HAS_FIELD` edges, so its answer was always `True` —
a model could score on it via any proxy that correlates with template identity
(phrasing, persona, family) without consulting the graph. It now splits its quota
between positives and near-miss negatives: a real field name from elsewhere in
the repo, paired with a class that neither defines nor inherits it.

**Absent-entity negatives are deliberately plausible.** Absent entities recombine a
real short name with a real namespace from the same repo
(`minirepo.models.Base.strip_ws` — both halves real, the pair is not). A
negative built from a nonsense identifier would be rejectable without consulting
the repository. Every negative is verified against the graph before emission.

Sampling is seeded per template, so a given `--seed` reproduces the same
dataset, and adding a template does not reshuffle the others.

## Natural phrasing (optional)

Templated questions are unambiguous but read like database queries. The
naturalizer rewrites them as questions a working engineer would actually ask,
under one invariant:

> **The model rewrites the question. The graph still owns the answer.**

The answer is never sent to the model, never regenerated, and never edited —
only the surface form changes. Personas (`newcomer`, `refactorer`, `debugger`,
`reviewer`, `plain`) supply the register; a fan-in question becomes "I'm about
to change this signature — how many places would I need to update?"

Four gates stand between a rewrite and the dataset. Three are deterministic
(`qgen/validate.py`): **answer leakage** (answer tokens present in the rewrite
but not in the template it came from), **referent drift** (an entity the Cypher
is parameterised on was dropped, or referred to by an ambiguous short name), and
**invented entities** (any identifier not in the graph). The fourth is an
adversarial LLM judge asked whether both questions have the same answer. A
rewrite failing any gate is discarded, the record keeps its templated phrasing,
and every rejected candidate is recorded in `naturalization.rejected_attempts`.

Either provider works — `anthropic:claude-opus-5` or `openai:gpt-4.1`. Beyond
convenience, writing questions with a different model family than you evaluate
on removes a self-preference confound; `naturalization.model` records which was
used. `--batch` (50% cheaper) is Anthropic-only.

## Quickstart

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev]'
docker compose up -d                      # Neo4j on bolt://localhost:7687

export PYTHONPATH=src
python -m cgx.cli extract  minirepo       # -> data/graphs/minirepo.json
python -m cgx.cli ingest   minirepo       # constraints + indexes, then load
python -m cgx.cli generate minirepo -n 6  # -> data/questions/minirepo.jsonl
python -m cgx.cli recheck  minirepo       # every answer re-derived from its Cypher
python -m cgx.cli verify   minirepo --family dependency_interaction --sample 10
python -m cgx.cli verify-report minirepo
python -m cgx.cli evaluate minirepo --model oracle
pytest -q
```

Extraction logging goes to stderr, so stdout stays pipeable:

```bash
python -m cgx.cli extract django            # INFO: phases, counts, timings
python -m cgx.cli extract django -v         # DEBUG: per-file, rejected CALLS edges
python -m cgx.cli extract django -q         # warnings and errors only
python -m cgx.cli extract django --include-tests
```

Rephrasing needs an API key (`ANTHROPIC_API_KEY`, an `ant auth login` profile,
or `OPENAI_API_KEY`):

```bash
python -m cgx.cli naturalize minirepo --dry-run          # prints prompts, no API calls
python -m cgx.cli naturalize minirepo --model openai:gpt-4.1
python -m cgx.cli naturalize minirepo --batch            # Anthropic only, 50% cheaper
python -m cgx.cli naturalize-collect minirepo <batch_id>
```

Add a repo by appending to `config/repos.yaml`. **`path` should point at the
package, not the repo root** — `path: fixtures/django/django` with
`package_root: fixtures/django` yields `django.apps.config`, while pointing at
the checkout root yields `django.django.apps.config` and pulls in `docs/` and
`scripts/`.

## Validation status

The table and tier scores below are historical pre-v2 measurements, not a
validation of the new taxonomy. Run `pytest -q` for current checks. Database
tests use a disposable repository namespace and require a running Neo4j.

| Repo                   | Files | Nodes | Edges | Notes                          |
| ---------------------- | ----- | ----- | ----- | ------------------------------ |
| `minirepo` (fixture)   | 6     | 44    | 78    | hand-built, ground truth known |
| neo4j driver           | 112   | 3379  | 8088  | 6 qname collisions             |
| Django (tests excluded)| 912   | 19694 | 44419 | ingests ~5s, 241 questions ~30s |

On minirepo, 102 questions across tiers 1–4 (27/25/20/30). The `oracle` model
scores 1.0 per tier (scorer sanity). The always-yes/always-zero `stub` baseline
scores 0.16 overall and **0.0 on tier 3** — down from 0.39 before non-trivial-tail
sampling, because the zero-answer questions it was scoring on are gone.

Django's 44419 JSON edges become 44350 in Neo4j. That 69-edge delta is correct
deduplication, not loss: a property's getter/setter/deleter assert the same
`HAS_METHOD` three times, and conditional definitions assert `CONTAINS` twice.

## Known limitations

- **Static analysis ceiling.** Dynamic dispatch, `getattr`, decorators that
  rewrap, and monkey-patching are invisible. `CALLS` is sound-ish but not
  complete; questions are ground truth _with respect to the extracted graph_,
  which is the honest claim to make in the paper.
- **Calls into third-party or stdlib code are excluded** — only in-repo targets
  become edges.
- **Sampling filters are coverage decisions, not neutral defaults.** Excluding
  test directories and trivial-valued nodes means the benchmark no longer
  samples the repo's true distribution and makes no claim about test code. This
  belongs in the paper's limitations, not buried in code.
- **State access has bounded scope.** It composes CALLS with USES (read/write),
  not full data-flow. It is enabled by default under the state-access family.
- **Property writes are modelled as both a FIELD and a METHOD.** For a property,
  `self.x = v` is really a write through the setter, but the extractor emits
  both nodes. Syntactically faithful and schema-valid, but it slightly inflates
  `HAS_FIELD` counts on property-heavy code.
- **`recheck` is a consistency check, not a re-derivation.** It asserts the
  stored answer still appears in its query's result — enough to catch graph
  drift, extractor changes, and template copy-paste errors.
- **The naturalizer has never run against a live API.** Its gates are covered by
  tests with a stubbed provider; the API path itself is unexercised.
- **Contamination filtering by creation date is a weak proxy** (see
  `select/repos.py`): recent repos can vendor older code, cutoffs are approximate,
  and high-star recent repos are likelier to have been scraped. A **fully
  synthetic repo** is flagged as the only contamination-proof source and is _not
  built_ — it trades away ecological validity, so it belongs as a control arm.

## Open decisions

- **Reachability near-miss negatives.** `t4_no_call_path` still draws random pairs,
  and ~90% of random pairs in a real repo are unreachable — so the negative is
  usually trivially distant rather than a genuine near miss. Constraining to
  pairs that share a module, class, or common caller would make these questions test
  understanding rather than distance. (`t4_absent_entity` and
  `t4_wrong_method_owner` already work this way.)
- **Pilot family quotas.** `--per-family` caps family budgets; report actual
  coverage and shortfalls rather than assuming every target was filled.
- **Lookup triviality filter.** Dunders and one-line functions are still sampled
  uniformly; ~38% of fixture methods are `__init__`.
- **Explicit regeneration creates review work.** Existing output is protected;
  `--overwrite` creates fresh pending records. Use `migrate-taxonomy` to retain
  review decisions while reclassifying existing questions. Ordinal IDs alone
  cannot safely transfer decisions to regenerated records.
- **Duplicate edges in the JSON.** The extractor emits the 69 duplicate
  assertions described above, so `edges_total` in the extract stats overcounts
  against what the graph holds. Deduping at emit time would make them agree.

## Layout

```
config/repos.yaml          repo registry + Neo4j connection
src/cgx/schema.py          node/edge definitions, identity attrs, validation
src/cgx/logconfig.py       CLI logging (-v / -q)
src/cgx/extract/           ast_pass.py (NodeIndex), pyan_pass.py, pipeline.py
src/cgx/graph/             client.py, ingest.py, cypher/schema.cypher
src/cgx/qgen/              base.py, taxonomy.py, tier1..tier5, generate.py
                           comparison.py, inheritance.py
                           naturalize.py, providers.py, personas.py, validate.py
src/cgx/verify/            Phase 3 review CLI
src/cgx/select/            Phase 4 contamination-aware repo search
src/cgx/eval/              Family/operation/outcome scoring harness
fixtures/minirepo/         hand-built repo with known ground truth
tests/                     ground truth asserted edge by edge
```
