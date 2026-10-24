"""Where the call graph may be incomplete, per callable.

Static resolution misses calls through receivers whose type it cannot infer:
`agent = laya.load(...)` then `agent.accelerate()`. The edge simply never
appears, and every question that depends on it -- a fan-in, "nothing calls
this", "no path exists", a reachable set -- gets a wrong stored answer. The
pilot showed it: questions touching such call sites scored 0.79, clean ones 0.93.

A call site is POSSIBLY MISSED when its callee's short name matches a function
or method defined in this repository, yet the enclosing callable got no CALLS
edge to anything by that name. Each callable records:

  unresolved_out   possibly missed call sites in its own body
  unresolved_in    possibly missed call sites anywhere that could target it

`qgen/completeness.py` uses these to drop questions whose answer depends on a
part of the graph that may be missing edges.

The plain counts ignore names shared with ubiquitous builtin methods: if the
repo defines a `get`, every `dict.get()` in the codebase would otherwise mark
that `get` as possibly called from everywhere. That is a stated assumption --
a repo method called `get` through an untyped receiver is missed and not
flagged -- so the `*_any` counts, which make no such exception, are kept too.
"""

from __future__ import annotations

from collections import defaultdict

from ..schema import CALLS, FUNCTION, METHOD, Edge, Node

GENERIC_NAMES = frozenset("""
get set append extend pop update items keys values copy clear add remove close open read write
run start stop join split strip format replace lower upper encode decode send put load dump
loads dumps sort index count insert discard setdefault find match search sub group __init__
call apply map filter reset next iter len str repr hash wait result cancel done submit info
debug warning error exception log emit flush seek tell exists mkdir parse render save
""".split())


def annotate(nodes: list[Node], edges: list[Edge], call_sites: dict[str, dict[str, list[int]]]) -> dict:
    """Set the four unresolved_* counts on every FUNCTION and METHOD node."""
    by_id = {n.id: n for n in nodes}
    callables = [n for n in nodes if n.label in (FUNCTION, METHOD)]
    by_name: dict[str, list[Node]] = defaultdict(list)
    for n in callables:
        by_name[n.name].append(n)
        n.unresolved_out = n.unresolved_out_any = n.unresolved_in = n.unresolved_in_any = 0

    # Short names each caller DID resolve a call to. A constructor edge lands
    # on `__init__` but was written with the class's name, so count both.
    resolved: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        if e.type != CALLS:
            continue
        src, tgt = by_id[e.source], by_id[e.target]
        resolved[src.qname].add(tgt.name)
        if tgt.name == "__init__" and tgt.class_qname:
            resolved[src.qname].add(tgt.class_qname.rsplit(".", 1)[-1])

    owners = {n.qname: n for n in callables}
    missed_sites = missed_generic = 0
    for caller_qname, sites in call_sites.items():
        caller = owners.get(caller_qname)
        missed = {name for name in sites if name in by_name and name not in resolved[caller_qname]}
        if not missed:
            continue
        if caller is not None:
            caller.unresolved_out_any = len(missed)
            caller.unresolved_out = len(missed - GENERIC_NAMES)
        for name in missed:
            generic = name in GENERIC_NAMES
            missed_sites += 1
            missed_generic += generic
            for callee in by_name[name]:
                callee.unresolved_in_any += 1
                if not generic:
                    callee.unresolved_in += 1

    return {
        "possibly_missed_call_names": missed_sites,
        "of_which_generic": missed_generic,
        "callables_incomplete_out": sum(n.unresolved_out > 0 for n in callables),
        "callables_incomplete_in": sum(n.unresolved_in > 0 for n in callables),
        "callables": len(callables),
    }
