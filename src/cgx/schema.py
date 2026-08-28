"""Graph schema -- the single source of truth for node and edge shapes.

Follows the CodexGraph schema (Liu et al., 2024, arXiv:2408.03910) exactly, with
two deliberate additions:

  * REPOSITORY node  -- lets a single database hold many repos and keeps every
    query scopeable to one of them. CodexGraph assumes one repo per database.
  * CALLS edge       -- FUNCTION|METHOD -> FUNCTION|METHOD. The CodexGraph
    authors flag this as missing from their own schema: "adding edges related
    to function calls will enhance the broader applicability of CodexGraph."

Everything else -- node labels, their attributes, edge directions -- is the
paper's schema verbatim. Note in particular that FUNCTION and METHOD are
distinct labels (METHOD carries a `class` attribute) and that GLOBAL_VARIABLE
is a first-class node, because USES is defined over
FUNCTION|METHOD -> GLOBAL_VARIABLE|FIELD.

Per the paper, nodes with a `code` attribute do not inline the source text;
they store a pointer to it. We keep that property as `code_ref`
("<file_path>:<start_line>-<end_line>") alongside explicit line bounds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

REPOSITORY = "REPOSITORY"
MODULE = "MODULE"
CLASS = "CLASS"
FUNCTION = "FUNCTION"
METHOD = "METHOD"
FIELD = "FIELD"
GLOBAL_VARIABLE = "GLOBAL_VARIABLE"

NODE_LABELS = (
    REPOSITORY,
    MODULE,
    CLASS,
    FUNCTION,
    METHOD,
    FIELD,
    GLOBAL_VARIABLE,
)

#: Labels that are valid endpoints of a CALLS edge.
CALLABLE_LABELS = (FUNCTION, METHOD)

# --------------------------------------------------------------------------
# Edge types
# --------------------------------------------------------------------------

CONTAINS = "CONTAINS"
HAS_METHOD = "HAS_METHOD"
HAS_FIELD = "HAS_FIELD"
INHERITS = "INHERITS"
USES = "USES"
CALLS = "CALLS"

EDGE_TYPES = (CONTAINS, HAS_METHOD, HAS_FIELD, INHERITS, USES, CALLS)

#: Allowed (source label, target label) pairs per edge type. Enforced by
#: validate_graph(), so a schema violation fails extraction rather than
#: silently poisoning the ground truth of a generated question.
EDGE_ENDPOINTS: dict[str, set[tuple[str, str]]] = {
    CONTAINS: {
        (REPOSITORY, MODULE),            # our addition
        (MODULE, CLASS),
        (MODULE, FUNCTION),
        (MODULE, GLOBAL_VARIABLE),
    },
    HAS_METHOD: {(CLASS, METHOD)},
    HAS_FIELD: {(CLASS, FIELD)},
    INHERITS: {(CLASS, CLASS)},
    USES: {
        (FUNCTION, GLOBAL_VARIABLE),
        (FUNCTION, FIELD),
        (METHOD, GLOBAL_VARIABLE),
        (METHOD, FIELD),
    },
    CALLS: {                              # our addition
        (FUNCTION, FUNCTION),
        (FUNCTION, METHOD),
        (METHOD, FUNCTION),
        (METHOD, METHOD),
    },
}


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass
class Node:
    """A graph node.

    `qname` is the fully qualified dotted name (e.g. ``minirepo.models.User.greet``)
    and is unique per (repo, label). `id` is the globally unique key used for
    the Neo4j uniqueness constraint.
    """

    id: str
    label: str
    name: str          # short name
    qname: str         # fully qualified dotted name
    repo_id: str
    file_path: str | None = None
    signature: str | None = None
    code_ref: str | None = None     # "<file_path>:<start>-<end>", not the source text
    start_line: int | None = None
    end_line: int | None = None
    class_qname: str | None = None  # METHOD and FIELD only, per the paper
    module_qname: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Edge:
    """A directed graph edge."""

    type: str
    source: str        # Node.id
    target: str        # Node.id
    repo_id: str
    #: USES carries source/target association-type attributes in the paper; we
    #: keep a general-purpose bag so CALLS can record call-site lines too.
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "source": self.source,
            "target": self.target,
            "repo_id": self.repo_id,
            **{f"attr_{k}": v for k, v in self.attrs.items()},
        }


@dataclass
class CodeGraph:
    repo_id: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }


def make_id(repo_id: str, label: str, qname: str) -> str:
    """Stable, globally unique node key."""
    return f"{repo_id}::{label}::{qname}"


class SchemaError(ValueError):
    """Raised when an extracted graph violates the declared schema."""


def validate_graph(graph: CodeGraph) -> None:
    """Fail loudly on dangling edges, duplicate ids, or illegal endpoint pairs."""
    by_id: dict[str, Node] = {}
    for node in graph.nodes:
        if node.label not in NODE_LABELS:
            raise SchemaError(f"unknown node label {node.label!r} on {node.qname}")
        if node.id in by_id:
            raise SchemaError(f"duplicate node id {node.id!r}")
        by_id[node.id] = node

    for edge in graph.edges:
        if edge.type not in EDGE_TYPES:
            raise SchemaError(f"unknown edge type {edge.type!r}")
        src = by_id.get(edge.source)
        tgt = by_id.get(edge.target)
        if src is None:
            raise SchemaError(f"{edge.type} edge has dangling source {edge.source!r}")
        if tgt is None:
            raise SchemaError(f"{edge.type} edge has dangling target {edge.target!r}")
        pair = (src.label, tgt.label)
        if pair not in EDGE_ENDPOINTS[edge.type]:
            raise SchemaError(
                f"{edge.type} may not connect {pair[0]} -> {pair[1]} "
                f"({src.qname} -> {tgt.qname})"
            )
