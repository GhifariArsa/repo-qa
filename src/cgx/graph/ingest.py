"""Ingest an extracted CodeGraph into Neo4j.

Nodes are written with two labels: their schema label (MODULE, CLASS, ...) and
a shared :CodeNode label, so traversals that don't care about the specific kind
stay simple and still hit an index.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import EDGE_TYPES, NODE_LABELS
from .client import GraphClient

_BATCH = 1000


def load_graph_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def clear_repo(client: GraphClient, repo_id: str) -> int:
    result = client.run(
        "MATCH (n:CodeNode {repo_id: $repo_id}) "
        "WITH n LIMIT 100000 DETACH DELETE n RETURN count(n) AS deleted",
        repo_id=repo_id,
    )
    return result[0]["deleted"] if result else 0


def ingest(client: GraphClient, graph: dict, replace: bool = True) -> dict[str, int]:
    repo_id = graph["repo_id"]
    if replace:
        clear_repo(client, repo_id)

    counts = {"nodes": 0, "edges": 0}

    # Nodes, grouped by label so each MERGE can use a concrete label.
    by_label: dict[str, list[dict]] = {}
    for node in graph["nodes"]:
        by_label.setdefault(node["label"], []).append(node)

    for label, rows in by_label.items():
        if label not in NODE_LABELS:
            raise ValueError(f"refusing to ingest unknown label {label!r}")
        for chunk in _chunks(rows, _BATCH):
            client.run(
                f"""
                UNWIND $rows AS row
                MERGE (n:{label} {{id: row.id}})
                SET n += row, n:CodeNode
                """,
                rows=chunk,
            )
            counts["nodes"] += len(chunk)

    # Edges, grouped by type for the same reason.
    by_type: dict[str, list[dict]] = {}
    for edge in graph["edges"]:
        by_type.setdefault(edge["type"], []).append(edge)

    for etype, rows in by_type.items():
        if etype not in EDGE_TYPES:
            raise ValueError(f"refusing to ingest unknown edge type {etype!r}")
        # Split the structural keys from the edge's own attributes (call_kind,
        # line, access, ...) so the latter can be written with a single `+=`.
        payload = [
            {
                "source": r["source"],
                "target": r["target"],
                "props": {
                    k: v for k, v in r.items() if k not in ("type", "source", "target")
                },
            }
            for r in rows
        ]
        for chunk in _chunks(payload, _BATCH):
            client.run(
                f"""
                UNWIND $rows AS row
                MATCH (s:CodeNode {{id: row.source}})
                MATCH (t:CodeNode {{id: row.target}})
                MERGE (s)-[r:{etype}]->(t)
                SET r += row.props
                """,
                rows=chunk,
            )
            counts["edges"] += len(chunk)

    return counts


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]
