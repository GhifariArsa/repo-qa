"""pyan3 pass: cross-file name resolution into CALLS and USES edges.

pyan3 is the reason this extractor can resolve a call across module boundaries
and up an inheritance chain (``self.describe()`` in ``Admin`` landing on
``Base.describe``). But its ``uses_edges`` relation is coarser than the
CodexGraph schema in two ways that matter, and both produce wrong ground truth
if taken at face value:

1. **Inheritance leaks in.** pyan emits ``CLASS Admin -> CLASS User`` as a
   "uses" edge. That is an INHERITS relation, which the AST pass already owns.
   We drop every edge whose source is a CLASS.

2. **Field access is misattributed to the defining method.** Because pyan has
   no field nodes, ``self.permissions`` inside ``Admin.grant`` resolves to
   ``Admin.__init__`` -- the method where that attribute is first bound. Read
   naively, pyan claims ``Admin.grant`` calls ``Admin.__init__``. It does not.

Fix for (2): treat pyan as a *resolution* oracle, not a *relation* oracle. A
resolved edge is promoted to CALLS only if the source body actually contains a
call site whose syntactic callee matches the target -- which the AST pass
recorded. Constructor calls are handled explicitly, since ``User(clean)`` is
spelled ``User`` at the call site but resolves to ``User.__init__``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..schema import (
    CALLS,
    Edge,
    FUNCTION,
    GLOBAL_VARIABLE,
    METHOD,
    Node,
    USES,
)

#: pyan flavors that correspond to a callable in our schema.
_CALLABLE_FLAVORS = {"FUNCTION", "METHOD", "STATICMETHOD", "CLASSMETHOD"}


@dataclass
class PyanResult:
    edges: list[Edge] = field(default_factory=list)
    #: edges pyan resolved but the call-site cross-check rejected, kept for audit
    rejected_calls: list[tuple[str, str, str]] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


def run_pyan_pass(
    repo_id: str,
    py_files: list[Path],
    package_root: Path,
    nodes: dict[str, Node],
    call_sites: dict[str, dict[str, list[int]]],
) -> PyanResult:
    from pyan.analyzer import CallGraphVisitor

    logger = logging.getLogger("cgx.pyan")
    logger.setLevel(logging.CRITICAL)      # pyan is extremely chatty at INFO

    visitor = CallGraphVisitor(
        [str(p) for p in py_files],
        root=str(package_root),
        logger=logger,
    )

    result = PyanResult()
    seen: set[tuple[str, str, str]] = set()
    n_resolved = 0

    for src, targets in visitor.uses_edges.items():
        if not src.defined:
            continue
        src_flavor = src.flavor.name
        # (1) drop inheritance and other class-body edges outright
        if src_flavor not in _CALLABLE_FLAVORS:
            continue
        src_node = nodes.get(src.get_name())
        if src_node is None:
            continue
        caller_calls = call_sites.get(src_node.qname, {})

        for tgt in targets:
            if not tgt.defined:
                continue
            n_resolved += 1
            tgt_flavor = tgt.flavor.name
            tgt_qname = tgt.get_name()

            # --- global variable reads -> USES ---------------------------
            if tgt_flavor == "NAME":
                tgt_node = nodes.get(tgt_qname)
                if tgt_node is None or tgt_node.label != GLOBAL_VARIABLE:
                    continue
                _emit(
                    result, seen, USES, src_node, tgt_node, repo_id,
                    access="read", source_type=src_node.label,
                    target_type=GLOBAL_VARIABLE, resolver="pyan",
                )
                continue

            # --- fields are owned by the AST pass ------------------------
            if tgt_flavor == "ATTRIBUTE":
                continue

            # --- constructor call: retarget CLASS -> CLASS.__init__ ------
            call_kind = "direct"
            if tgt_flavor == "CLASS":
                ctor_qname = f"{tgt_qname}.__init__"
                ctor = nodes.get(ctor_qname)
                if ctor is None or ctor.label != METHOD:
                    continue           # class has no __init__ in this repo
                tgt_node = ctor
                call_kind = "constructor"
            elif tgt_flavor in _CALLABLE_FLAVORS:
                tgt_node = nodes.get(tgt_qname)
                if tgt_node is None or tgt_node.label not in (FUNCTION, METHOD):
                    continue
                if tgt_node.name == "__init__":
                    call_kind = "constructor"
            else:
                continue

            # --- (2) call-site cross-check -------------------------------
            line = _matching_call_site(tgt_node, caller_calls)
            if line is None:
                result.rejected_calls.append((src_node.qname, tgt_node.qname, "no call site"))
                continue

            _emit(
                result, seen, CALLS, src_node, tgt_node, repo_id,
                call_kind=call_kind, line=line, resolver="pyan",
            )

    result.stats = {
        "pyan_resolved_edges": n_resolved,
        "calls_emitted": sum(1 for e in result.edges if e.type == CALLS),
        "uses_global_emitted": sum(1 for e in result.edges if e.type == USES),
        "calls_rejected": len(result.rejected_calls),
    }
    return result


def _matching_call_site(tgt_node: Node, caller_calls: dict[str, list[int]]) -> int | None:
    """First call-site line in the caller that plausibly invokes `tgt_node`.

    Returns None when the caller's body has no syntactic call matching the
    target -- meaning pyan's edge is a field-access artifact, not a call.
    """
    lines = caller_calls.get(tgt_node.name)
    if lines:
        return min(lines)
    # `User(clean)` is written as the class name but resolves to User.__init__
    if tgt_node.name == "__init__" and tgt_node.class_qname:
        class_short = tgt_node.class_qname.rsplit(".", 1)[-1]
        lines = caller_calls.get(class_short)
        if lines:
            return min(lines)
    return None


def _emit(result, seen, etype, src, tgt, repo_id, **attrs) -> None:
    key = (etype, src.id, tgt.id)
    if key in seen:
        return
    seen.add(key)
    result.edges.append(
        Edge(type=etype, source=src.id, target=tgt.id, repo_id=repo_id, attrs=attrs)
    )
