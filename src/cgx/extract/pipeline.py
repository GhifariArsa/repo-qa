"""Extraction pipeline: AST pass + pyan pass -> one validated CodeGraph."""

from __future__ import annotations

import json
from pathlib import Path

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
from .pyan_pass import run_pyan_pass

DEFAULT_EXCLUDES = {
    ".git", ".venv", "venv", "__pycache__", "build", "dist",
    ".tox", ".mypy_cache", ".pytest_cache", "node_modules",
}


def discover_python_files(repo_root: Path, excludes: set[str] | None = None) -> list[Path]:
    excludes = excludes or DEFAULT_EXCLUDES
    files = []
    for path in sorted(repo_root.rglob("*.py")):
        if any(part in excludes for part in path.parts):
            continue
        files.append(path)
    return files


def extract_repo(
    repo_id: str,
    repo_path: Path,
    package_root: Path,
) -> tuple[CodeGraph, dict]:
    repo_path = Path(repo_path).resolve()
    package_root = Path(package_root).resolve()
    py_files = discover_python_files(repo_path)
    if not py_files:
        raise SystemExit(f"no Python files found under {repo_path}")

    ast_result = run_ast_pass(repo_id, py_files, package_root, repo_path)
    pyan_result = run_pyan_pass(
        repo_id, py_files, package_root, ast_result.nodes, ast_result.call_sites
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

    graph = CodeGraph(repo_id=repo_id, nodes=nodes, edges=edges)
    validate_graph(graph)

    stats = {
        "files": len(py_files),
        "nodes_total": len(nodes),
        "edges_total": len(edges),
        "nodes_by_label": _count(n.label for n in nodes),
        "edges_by_type": _count(e.type for e in edges),
        **pyan_result.stats,
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
