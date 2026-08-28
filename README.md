# Repo-QA

A repository-level code-QA benchmark whose ground truth is **derived from a
codebase's graph structure**, not mined from GitHub issues (SWE-QA), Stack
Overflow (StackRepoQA), or agent exploration (CoReQA). Every question ships
with the Cypher query that produced its answer, so the label is reproducible
and auditable rather than adjudicated.

Pipeline: `pyan3 static analysis -> CodexGraph-schema graph (+ CALLS) -> Neo4j
-> tiered question generator`.

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
of CodexGraph."_ Tiers 2, 3 and 5 are largely unaskable without it.

Following the paper, `code_ref` stores a **pointer** into the source
(`path:start-end`), not the snippet text.

`src/cgx/schema.py` is the single source of truth, and `validate_graph()`
rejects illegal endpoint pairs at extraction time — a schema violation fails
the run rather than silently poisoning a question's ground truth.

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

## Question tiers

Per-tier labels are carried on every record. Report per-tier accuracy; do not
flatten to a single score.

| Tier | Name                      | Templates                                                                                 |
| ---- | ------------------------- | ----------------------------------------------------------------------------------------- |
| 1    | existence / lookup        | entity type, method owner, defining file, direct base, has field                          |
| 2    | path traversal            | path exists, shortest chain, hop count, reachable-within-2, reaches-global-read           |
| 3    | aggregation over shape    | fan-in, fan-out, reachability-set size, nearest common ancestor, max fan-in, method count |
| 4    | counterfactual / negative | absent entity, no call path, wrong method owner, no callers, not-inherits                 |
| 5    | cross-graph composition   | CALLS closure × USES — **reduced scope, off by default** (see below)                      |

**Tier 4 negatives are deliberately plausible.** Absent entities recombine a
real short name with a real namespace from the same repo
(`minirepo.models.Base.strip_ws` — both halves real, the pair is not); unreachable
pairs are drawn from callables that genuinely exist. A negative built from a
nonsense identifier would be rejectable without consulting the repository, which
would make the tier worthless. Every negative is checked against the graph
before it is emitted.

Each record carries: `tier`, `template_id`, `question`, `answer`, `answer_type`,
`cypher`, `params`, `evidence`, `verification`.

## Quickstart

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev]'
docker compose up -d                      # Neo4j on bolt://localhost:7687

export PYTHONPATH=src
python -m cgx.cli extract  minirepo       # -> data/graphs/minirepo.json
python -m cgx.cli ingest   minirepo       # constraints + indexes, then load
python -m cgx.cli generate minirepo -n 5  # -> data/questions/minirepo.jsonl
python -m cgx.cli recheck  minirepo       # every answer re-derived from its Cypher
python -m cgx.cli verify   minirepo --tier 4 --sample 10
python -m cgx.cli verify-report minirepo
python -m cgx.cli evaluate minirepo --model oracle
pytest -q
```

Add a repo by appending to `config/repos.yaml` (`path` plus `package_root`, the
directory the top-level package sits inside).

## Validation status (fixture: `fixtures/minirepo`)

44 nodes, 78 edges, 92 questions across tiers 1–4. All 92 answers reproduce from
their own stored Cypher. 20 tests pass. `oracle` scores 1.0 per tier (scorer
sanity); the always-yes `stub` baseline scores 0.20 on tier 4, confirming the
negative tier penalises yes-bias.

## Known limitations

- **Static analysis ceiling.** Dynamic dispatch, `getattr`, decorators that
  rewrap, and monkey-patching are invisible. `CALLS` is sound-ish but not
  complete; questions are ground truth _with respect to the extracted graph_,
  which is the honest claim to make in the paper.
- **Calls into third-party or stdlib code are excluded** — only in-repo targets
  become edges.
- **Tier 5 is reduced in scope.** The stretch goal was CALLS × _data-flow_. There
  is no data-flow extraction here, so tier 5 composes CALLS with the data-access
  graph (`USES`, read/write). That is genuine two-graph composition but is **not**
  data-flow and must not be reported as such. Off by default; `--include-tier5`.
- **`recheck` is a consistency check, not a re-derivation.** It asserts the
  stored answer still appears in its query's result — enough to catch graph
  drift, extractor changes, and template copy-paste errors.
- **Contamination filtering by creation date is a weak proxy** (see
  `select/repos.py`): recent repos can vendor older code, cutoffs are approximate,
  and high-star recent repos are likelier to have been scraped. A **fully
  synthetic repo** is flagged as the only contamination-proof source and is _not
  built_ — it trades away ecological validity, so it belongs as a control arm.

## Layout

```
config/repos.yaml          repo registry + Neo4j connection
src/cgx/schema.py          node/edge definitions + validation
src/cgx/extract/           ast_pass.py, pyan_pass.py, pipeline.py
src/cgx/graph/             client.py, ingest.py, cypher/schema.cypher
src/cgx/qgen/              base.py, tier1..tier5, generate.py
src/cgx/verify/            Phase 3 review CLI
src/cgx/select/            Phase 4 contamination-aware repo search
src/cgx/eval/              Phase 5 per-tier scoring harness
fixtures/minirepo/         hand-built repo with known ground truth
tests/                     ground truth asserted edge by edge
```
