"""Drop questions whose answer depends on a possibly incomplete call graph.

Extraction flags every callable with `unresolved_out` / `unresolved_in` -- call
sites static resolution may have missed (extract/completeness.py). A question
is only as trustworthy as the part of the graph its answer reads:

  node_out      the source's own outgoing calls           fan-out, callee comparisons
  node_in       the target's own incoming calls           fan-in, "nothing calls this"
  closure_out   everything reachable from the source      paths, hop counts, reachable
                                                          sets, state reached by calls
  window_out    the source and its direct callees         reachable within two calls
  closure_in    everything that can reach the target      transitive callers
  repo_in       every callable's incoming calls           the most-called function

A question passes only if nothing in that region has an unresolved call site.

The region is the part of the graph that could change THIS answer, not
everything nearby -- checking every closure in full dropped 27 of 30
shortest-chain questions, since a long closure in real code almost always
passes through some unresolved call. Two refinements, both exact:

  * A "yes" witnessed by a path needs no check. Emitted CALLS edges passed the
    call-site cross-check, so a found path is real; only "no" (unreachable,
    never read, never written) depends on the closure being complete.
  * A shortest path of k hops can only be shortened by a missed call out of a
    node within k-2 hops of the source, so that window is all that is checked.
Relations static analysis resolves on its own -- INHERITS, HAS_METHOD,
HAS_FIELD, CONTAINS -- need no check, so those templates pass untouched.
"""

from __future__ import annotations

from .base import QuestionRecord

#: template -> (region, param keys naming the callables it is anchored on)
DEPENDENCE: dict[str, tuple[str, tuple[str, ...]]] = {
    "t3_fan_out": ("node_out", ("source",)),
    "comparison_shared_callees": ("node_out", ("a", "b")),
    "comparison_exclusive_callees": ("node_out", ("a", "b")),
    "t3_fan_in": ("node_in", ("target",)),
    "t4_no_callers": ("node_in", ("target",)),
    "t2_path_exists": ("closure_out", ("source",)),
    "t4_no_call_path": ("closure_out", ("source",)),
    "t2_hop_count": ("closure_out", ("source",)),
    "t2_shortest_chain": ("closure_out", ("source",)),
    "t3_reach_set_size": ("closure_out", ("source",)),
    "t2_reaches_global_read": ("closure_out", ("source",)),
    "t5_closure_globals_read": ("closure_out", ("source",)),
    "t5_closure_writes_field": ("closure_out", ("source",)),
    "t2_files_touched": ("closure_out", ("source",)),
    "t2_reachable_within_2": ("window_out", ("source",)),
    "t2_transitive_callers": ("closure_in", ("target",)),
    "comparison_reachable_difference": ("closure_out", ("a", "b")),
    "comparison_reachable_shared": ("closure_out", ("a", "b")),
    "t3_max_fan_in": ("repo_in", ()),
}

_QUERIES = {
    "node_out": "MATCH (n:CodeNode {{repo_id: $repo, qname: $q}}) RETURN coalesce(n.{out}, 0) AS bad",
    "node_in": "MATCH (n:CodeNode {{repo_id: $repo, qname: $q}}) RETURN coalesce(n.{inc}, 0) AS bad",
    "closure_out": """
        MATCH (s:CodeNode {{repo_id: $repo, qname: $q}})-[:CALLS*0..12]->(n)
        WHERE coalesce(n.{out}, 0) > 0 RETURN count(DISTINCT n) AS bad""",
    "window_out": """
        MATCH (s:CodeNode {{repo_id: $repo, qname: $q}})-[:CALLS*0..1]->(n)
        WHERE coalesce(n.{out}, 0) > 0 RETURN count(DISTINCT n) AS bad""",
    "closure_in": """
        MATCH (n)-[:CALLS*0..12]->(t:CodeNode {{repo_id: $repo, qname: $q}})
        WHERE coalesce(n.{inc}, 0) > 0 RETURN count(DISTINCT n) AS bad""",
    "repo_in": """
        MATCH (n:CodeNode {{repo_id: $repo}}) WHERE coalesce(n.{inc}, 0) > 0
        RETURN count(n) AS bad""",
}


#: Yes/no templates whose "yes" is witnessed by edges that exist.
WITNESSED_YES = frozenset({"t2_path_exists", "t4_no_call_path",
                           "t2_reaches_global_read", "t5_closure_writes_field"})
#: Templates whose answer is a shortest path, and how to read its hop count.
SHORTEST = {"t2_hop_count": lambda answer: int(answer),
            "t2_shortest_chain": lambda answer: len(answer) - 1}

_WINDOW_OUT = """
    MATCH (s:CodeNode {{repo_id: $repo, qname: $q}})-[:CALLS*0..{depth}]->(n)
    WHERE coalesce(n.{out}, 0) > 0 RETURN count(DISTINCT n) AS bad"""


class CompletenessFilter:
    """`strict=True` also distrusts names shared with builtin methods (`get`,
    `append`, ...), which drops far more questions -- see extract/completeness.py."""

    def __init__(self, client, repo_id: str, strict: bool = False):
        self.client, self.repo_id = client, repo_id
        suffix = "_any" if strict else ""
        self.props = {"out": f"unresolved_out{suffix}", "inc": f"unresolved_in{suffix}"}
        self._cache: dict[tuple[str, str], bool] = {}

    def depends(self, template_id: str) -> bool:
        return template_id in DEPENDENCE

    def _clean(self, region: str, qname: str, depth: int | None = None) -> bool:
        key = (region, qname, depth)
        if key not in self._cache:
            if depth is not None:
                query = _WINDOW_OUT.format(depth=depth, **self.props)
            else:
                query = _QUERIES[region].format(**self.props)
            rows = self.client.run(query, repo=self.repo_id, q=qname)
            self._cache[key] = bool(rows) and rows[0]["bad"] == 0
        return self._cache[key]

    def ok(self, rec: QuestionRecord) -> bool:
        spec = DEPENDENCE.get(rec.template_id)
        if spec is None:
            return True
        region, keys = spec
        if rec.template_id in WITNESSED_YES and rec.answer is True:
            return True
        if rec.template_id in SHORTEST:
            try:
                hops = SHORTEST[rec.template_id](rec.answer)
            except (TypeError, ValueError):
                return False
            return self._clean(region, rec.params["source"], depth=max(hops - 2, 0))
        if region == "repo_in":
            return self._clean(region, "")
        anchors = [rec.params[k] for k in keys if isinstance(rec.params.get(k), str)]
        return bool(anchors) and all(self._clean(region, q) for q in anchors)
