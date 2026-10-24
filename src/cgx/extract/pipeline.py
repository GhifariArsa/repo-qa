"""Extraction pipeline: AST pass + pyan pass -> one validated CodeGraph."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..logconfig import get_logger

from ..schema import (
    CONTAINS,
    CodeGraph,
    Edge,
    MODULE,
    Node,
    REPOSITORY,
    make_id,
    validate_graph,
)
from .ast_pass import run_ast_pass
from .completeness import annotate as annotate_completeness
from .pyan_pass import run_pyan_pass

log = get_logger(__name__)

#: Directories never walked. Beyond the obvious build/VCS noise, test
#: directories are excluded by default for three reasons: questions about a
#: repo's structure are about its library code, not its fixtures; test trees
#: dominate file counts on real projects (Django is ~60% tests), which would
#: skew every sampled question toward them; and test suites deliberately
#: contain pathological code -- Django ships `tests_syntax_error.py` precisely
#: because it must not parse.
#:
#: This is a coverage decision, not a neutral default: the benchmark then makes
#: no claim about test code. Pass `include_tests=True` (CLI: `--include-tests`)
#: to keep them.
BUILD_EXCLUDES = {
    ".git", ".venv", "venv", "__pycache__", "build", "dist",
    ".tox", ".mypy_cache", ".pytest_cache", "node_modules",
}
#: Deliberately does NOT include "fixtures": it is too ambiguous a name to
#: exclude repo-wide, and plenty of libraries ship real code under it.
TEST_EXCLUDES = {"tests", "test", "testing"}
DEFAULT_EXCLUDES = BUILD_EXCLUDES | TEST_EXCLUDES


def discover_python_files(
    repo_root: Path,
    excludes: set[str] | None = None,
    include_tests: bool = False,
) -> list[Path]:
    if excludes is None:
        excludes = BUILD_EXCLUDES if include_tests else DEFAULT_EXCLUDES
    files, excluded = [], 0
    for path in sorted(repo_root.rglob("*.py")):
        # Match on components RELATIVE to the repo root. Matching absolute path
        # components would mean a repo that merely happens to live under a
        # directory called `tests/` (or `build/`, or `dist/`) extracts to an
        # empty graph, silently, depending only on where it sits on disk.
        # Only directory components count, so a module named `test_utils.py`
        # is kept while anything under a `tests/` directory is dropped.
        relative = path.relative_to(repo_root)
        if any(part in excludes for part in relative.parts[:-1]):
            excluded += 1
            continue
        files.append(path)
    if excluded:
        log.info("skipped %d file(s) in excluded directories (%s)",
                 excluded, ", ".join(sorted(excludes & TEST_EXCLUDES)) or "build dirs")
    return files


def extract_repo(
    repo_id: str,
    repo_path: Path,
    package_root: Path,
    include_tests: bool = False,
) -> tuple[CodeGraph, dict]:
    repo_path = Path(repo_path).resolve()
    package_root = Path(package_root).resolve()
    started = time.monotonic()
    log.info("extracting %r from %s", repo_id, repo_path)

    py_files = discover_python_files(repo_path, include_tests=include_tests)
    # A package root inside the repo (`src/`, `python/`, `backend/`) scopes what
    # gets extracted: module names are computed relative to it, so a file
    # outside it -- `scripts/`, `benchmarks/` -- has no module name at all.
    # `repo_path` stays the repo root so every `file_path` answer is still
    # relative to the repository root, which is what the questions promise.
    if package_root != repo_path and package_root.is_relative_to(repo_path):
        in_scope = [p for p in py_files if p.resolve().is_relative_to(package_root)]
        if len(in_scope) < len(py_files):
            log.info("skipped %d file(s) outside package root %s",
                     len(py_files) - len(in_scope), package_root.relative_to(repo_path))
        py_files = in_scope
    if not py_files:
        raise SystemExit(f"no Python files found under {repo_path}")
    log.info("discovered %d Python file(s) (package root: %s)", len(py_files), package_root)

    ast_result = run_ast_pass(repo_id, py_files, package_root, repo_path)

    # Only files that parsed go to pyan — see AstResult.parsed_files.
    skipped = len(py_files) - len(ast_result.parsed_files)
    if skipped:
        log.warning(
            "%d file(s) excluded from the graph because they do not parse; "
            "the extracted graph does not represent them",
            skipped,
        )
    if not ast_result.parsed_files:
        raise SystemExit(f"no parseable Python files under {repo_path}")

    pyan_result = run_pyan_pass(
        repo_id, ast_result.parsed_files, package_root,
        ast_result.nodes, ast_result.call_sites,
    )

    # REPOSITORY node + CONTAINS to every module (our addition to the schema).
    repo_node = Node(
        id=make_id(repo_id, REPOSITORY, repo_id),
        label=REPOSITORY,
        name=repo_id,
        qname=repo_id,
        repo_id=repo_id,
        file_path=str(repo_path),
    )
    nodes = [repo_node, *ast_result.nodes.values()]
    edges = [
        Edge(type=CONTAINS, source=repo_node.id, target=n.id, repo_id=repo_id)
        for n in ast_result.nodes.values()
        if n.label == MODULE
    ]
    edges += ast_result.edges
    edges += pyan_result.edges

    completeness = annotate_completeness(nodes, edges, ast_result.call_sites)
    log.info("completeness: %d of %d callable(s) may be missing outgoing calls, %d incoming",
             completeness["callables_incomplete_out"], completeness["callables"],
             completeness["callables_incomplete_in"])

    graph = CodeGraph(repo_id=repo_id, nodes=nodes, edges=edges)
    log.info("validating %d node(s) and %d edge(s) against the schema",
             len(nodes), len(edges))
    validate_graph(graph)
    log.info("extraction complete in %.1fs", time.monotonic() - started)

    stats = {
        "files": len(py_files),
        "files_parsed": len(ast_result.parsed_files),
        "tests_included": include_tests,
        "nodes_total": len(nodes),
        "edges_total": len(edges),
        "nodes_by_label": _count(n.label for n in nodes),
        "edges_by_type": _count(e.type for e in edges),
        **pyan_result.stats,
        "completeness": completeness,
    }
    return graph, {"stats": stats, "rejected_calls": pyan_result.rejected_calls}


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return dict(sorted(out.items()))


def write_graph(graph: CodeGraph, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(graph.to_dict(), indent=2))
    return out_path
