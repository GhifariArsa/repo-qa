"""Ground truth for fixtures/minirepo, asserted edge by edge.

The fixture was written so that every number here is checkable by reading the
six source files. If the extractor regresses, these break before any question
built on top of it can go wrong silently.
"""

import pytest

REPO = "minirepo"


def by_id(graph):
    return {n.id: n for n in graph.nodes}


def edges_of(graph, etype):
    ids = by_id(graph)
    return {(ids[e.source].qname, ids[e.target].qname) for e in graph.edges if e.type == etype}


def test_node_counts(minigraph):
    graph, _ = minigraph
    counts = {}
    for node in graph.nodes:
        counts[node.label] = counts.get(node.label, 0) + 1
    assert counts == {
        "REPOSITORY": 1,
        "MODULE": 6,
        "CLASS": 5,
        "FUNCTION": 8,
        "METHOD": 13,
        "FIELD": 8,
        "GLOBAL_VARIABLE": 3,
    }


def test_inherits_edges(minigraph):
    graph, _ = minigraph
    assert edges_of(graph, "INHERITS") == {
        ("minirepo.models.User", "minirepo.models.Base"),
        ("minirepo.models.Admin", "minirepo.models.User"),
        ("minirepo.models.Guest", "minirepo.models.Base"),
    }


def test_calls_edges_exact(minigraph):
    """Every in-repo call, and nothing else."""
    graph, _ = minigraph
    assert edges_of(graph, "CALLS") == {
        ("minirepo.app.run", "minirepo.app.main"),
        ("minirepo.app.main", "minirepo.service.UserService.__init__"),
        ("minirepo.app.main", "minirepo.service.UserService.create_user"),
        ("minirepo.app.main", "minirepo.service.UserService.promote"),
        ("minirepo.service.UserService.__init__", "minirepo.config.get_timeout"),
        ("minirepo.service.UserService.create_user", "minirepo.utils.normalize"),
        ("minirepo.service.UserService.create_user", "minirepo.models.User.__init__"),
        ("minirepo.service.UserService.promote", "minirepo.utils.normalize"),
        ("minirepo.service.UserService.promote", "minirepo.models.Admin.__init__"),
        ("minirepo.models.Admin.__init__", "minirepo.models.User.__init__"),
        ("minirepo.models.User.__init__", "minirepo.models.Base.__init__"),
        ("minirepo.models.Guest.__init__", "minirepo.models.Base.__init__"),
        ("minirepo.models.Admin.grant", "minirepo.models.Base.describe"),
        ("minirepo.utils.normalize", "minirepo.utils.sanitize"),
        ("minirepo.utils.sanitize", "minirepo.utils.strip_ws"),
    }


def test_field_access_artifacts_are_rejected(minigraph):
    """pyan resolves `self.<field>` to the method that defines the field.

    Taken at face value that yields two calls that do not exist. The call-site
    cross-check must reject exactly these two and no others.
    """
    graph, meta = minigraph
    rejected = {(src, tgt) for src, tgt, _ in meta["rejected_calls"]}
    assert rejected == {
        ("minirepo.models.Admin.grant", "minirepo.models.Admin.__init__"),
        ("minirepo.service.UserService.create_user", "minirepo.service.UserService.__init__"),
    }
    calls = edges_of(graph, "CALLS")
    for pair in rejected:
        assert pair not in calls


def test_inherited_method_call_resolves_to_base(minigraph):
    """`self.describe()` in Admin resolves to Base.describe, not to Admin."""
    graph, _ = minigraph
    assert ("minirepo.models.Admin.grant", "minirepo.models.Base.describe") in edges_of(graph, "CALLS")


def test_uses_global_variable_edges(minigraph):
    graph, _ = minigraph
    ids = by_id(graph)
    uses_global = {
        (ids[e.source].qname, ids[e.target].qname)
        for e in graph.edges
        if e.type == "USES" and ids[e.target].label == "GLOBAL_VARIABLE"
    }
    assert uses_global == {
        ("minirepo.config.get_timeout", "minirepo.config.DEFAULT_TIMEOUT"),
        ("minirepo.config.get_retries", "minirepo.config.MAX_RETRIES"),
        ("minirepo.models.Base.__init__", "minirepo.config.LOG_LEVEL"),
        ("minirepo.service.UserService.retry_count", "minirepo.config.MAX_RETRIES"),
    }


def test_uses_field_edges_carry_access_kind(minigraph):
    graph, _ = minigraph
    ids = by_id(graph)
    writes = {
        (ids[e.source].qname, ids[e.target].qname)
        for e in graph.edges
        if e.type == "USES" and e.attrs.get("access") == "write"
    }
    reads = {
        (ids[e.source].qname, ids[e.target].qname)
        for e in graph.edges
        if e.type == "USES" and e.attrs.get("access") == "read"
        and ids[e.target].label == "FIELD"
    }
    assert ("minirepo.models.User.__init__", "minirepo.models.User.name") in writes
    assert ("minirepo.models.User.greet", "minirepo.models.User.name") in reads


def test_inherited_field_access_resolves_to_base(minigraph):
    """Guest reads `self.log_level`, which Base defines."""
    graph, _ = minigraph
    ids = by_id(graph)
    assert any(
        ids[e.source].qname == "minirepo.models.Base.__init__"
        and ids[e.target].qname == "minirepo.models.Base.log_level"
        for e in graph.edges
        if e.type == "USES"
    )


def test_method_nodes_carry_owning_class(minigraph):
    graph, _ = minigraph
    greet = next(n for n in graph.nodes if n.qname == "minirepo.models.User.greet")
    assert greet.label == "METHOD"
    assert greet.class_qname == "minirepo.models.User"
    assert greet.signature == "def greet(self)"


def test_code_ref_is_a_pointer_not_source_text(minigraph):
    """The paper stores an index into the source, not the snippet."""
    graph, _ = minigraph
    node = next(n for n in graph.nodes if n.qname == "minirepo.utils.normalize")
    assert node.code_ref == f"{node.file_path}:{node.start_line}-{node.end_line}"
    assert "def normalize" not in (node.code_ref or "")


def test_schema_validation_rejects_illegal_endpoints():
    from cgx.schema import CALLS, CodeGraph, Edge, Node, SchemaError, make_id

    cls = Node(id=make_id("r", "CLASS", "a.B"), label="CLASS", name="B", qname="a.B", repo_id="r")
    fn = Node(id=make_id("r", "FUNCTION", "a.f"), label="FUNCTION", name="f", qname="a.f", repo_id="r")
    graph = CodeGraph(repo_id="r", nodes=[cls, fn],
                      edges=[Edge(type=CALLS, source=cls.id, target=fn.id, repo_id="r")])
    with pytest.raises(SchemaError, match="CALLS may not connect CLASS"):
        from cgx.schema import validate_graph
        validate_graph(graph)


def test_schema_validation_rejects_dangling_edges():
    from cgx.schema import CONTAINS, CodeGraph, Edge, Node, SchemaError, make_id, validate_graph

    mod = Node(id=make_id("r", "MODULE", "a"), label="MODULE", name="a", qname="a", repo_id="r")
    graph = CodeGraph(repo_id="r", nodes=[mod],
                      edges=[Edge(type=CONTAINS, source=mod.id, target="nope", repo_id="r")])
    with pytest.raises(SchemaError, match="dangling target"):
        validate_graph(graph)


# ---------------------------------------------------------------------------
# Regression: FIELD/METHOD qname collisions
# ---------------------------------------------------------------------------

COLLIDING_SOURCE = '''
class Connection:
    def __init__(self, sock):
        self._socket = sock          # defines a FIELD  pkg.mod.Connection._socket

    @property
    def _socket(self):               # defines a METHOD pkg.mod.Connection._socket
        return self.__sock

    @_socket.setter
    def _socket(self, value):
        self.__sock = value

    def close(self):
        self._socket.close()
'''


def _extract_source(tmp_path, source: str, repo_id="collide"):
    from cgx.extract.pipeline import extract_repo

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text(source)
    return extract_repo(repo_id, pkg, tmp_path)


def test_property_and_attribute_sharing_a_qname_both_survive(tmp_path):
    """`self._socket = x` and `@property def _socket` are one qname, two nodes.

    Keying the node index by qname alone made the second registration return
    the first node, so a HAS_METHOD edge pointed at a FIELD and extraction of
    any repo using properties failed schema validation outright.
    """
    graph, _ = _extract_source(tmp_path, COLLIDING_SOURCE)
    by_qname = {}
    for node in graph.nodes:
        by_qname.setdefault(node.qname, set()).add(node.label)

    labels = by_qname["pkg.mod.Connection._socket"]
    assert labels == {"FIELD", "METHOD"}, labels


def test_collision_edges_point_at_the_right_label(tmp_path):
    graph, _ = _extract_source(tmp_path, COLLIDING_SOURCE)
    ids = by_id(graph)
    for edge in graph.edges:
        src, tgt = ids[edge.source], ids[edge.target]
        if edge.type == "HAS_METHOD":
            assert tgt.label == "METHOD", f"HAS_METHOD -> {tgt.label} ({tgt.qname})"
        if edge.type == "HAS_FIELD":
            assert tgt.label == "FIELD", f"HAS_FIELD -> {tgt.label} ({tgt.qname})"


def test_colliding_repo_passes_schema_validation(tmp_path):
    """extract_repo() calls validate_graph() internally — this used to raise."""
    graph, _ = _extract_source(tmp_path, COLLIDING_SOURCE)
    assert len(graph.nodes) > 0 and len(graph.edges) > 0
