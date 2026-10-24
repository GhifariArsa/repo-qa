"""AST pass: node inventory, attributes, and the structural edges.

pyan3 resolves names well but exposes neither signatures, source spans, nor
class fields, so it cannot on its own emit the CodexGraph node set. This pass
owns everything that is decidable from a single file's syntax tree:

    nodes  MODULE, CLASS, FUNCTION, METHOD, FIELD, GLOBAL_VARIABLE
    edges  CONTAINS, HAS_METHOD, HAS_FIELD, INHERITS, and USES -> FIELD

It also records, per callable, the syntactic call sites in its body. The pyan
pass consumes that as a precision filter -- see pyan_pass.py for why.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..logconfig import get_logger
from ..schema import (
    CLASS,
    CONTAINS,
    Edge,
    FIELD,
    FUNCTION,
    GLOBAL_VARIABLE,
    HAS_FIELD,
    HAS_METHOD,
    INHERITS,
    METHOD,
    MODULE,
    Node,
    USES,
    make_id,
)


log = get_logger(__name__)

#: Order used when a caller looks a qname up without saying what kind of thing
#: it expects. Containers first, then callables, then values.
_RESOLUTION_ORDER = (MODULE, CLASS, FUNCTION, METHOD, GLOBAL_VARIABLE, FIELD)


class NodeIndex:
    """Nodes keyed by (label, qname), not by qname alone.

    A qname is NOT unique on its own. The commonest case in real Python is a
    property: `self._socket = sock` in `__init__` defines a FIELD, and
    `@property def _socket` defines a METHOD -- same qualified name, two
    different kinds of node. Keying by qname alone meant the second `add()`
    silently returned the first node, so a HAS_METHOD edge could end up aimed
    at a FIELD and `validate_graph` rejected the whole repo.

    The graph schema already distinguishes them: `make_id()` keys on label and
    qname, so both nodes coexist happily once they are indexed separately.
    """

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], Node] = {}

    def add(self, node: Node) -> Node:
        """Register `node`; first writer wins for a given (label, qname)."""
        return self._by_key.setdefault((node.label, node.qname), node)

    def get(self, qname: str, *labels: str) -> Node | None:
        """Look up `qname`, preferring the given labels in order.

        Callers that know what they expect (a base class, a call target, a
        global) should say so -- that is what makes the lookup unambiguous.
        """
        for label in labels or _RESOLUTION_ORDER:
            hit = self._by_key.get((label, qname))
            if hit is not None:
                return hit
        return None

    def has(self, qname: str, label: str) -> bool:
        return (label, qname) in self._by_key

    def values(self):
        return self._by_key.values()

    def __len__(self) -> int:
        return len(self._by_key)


@dataclass
class AstResult:
    nodes: NodeIndex = field(default_factory=NodeIndex)
    edges: list[Edge] = field(default_factory=list)
    #: callable qname -> {callee short name -> [line numbers]}
    call_sites: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    #: class qname -> list of base class qnames (only those defined in-repo)
    bases: dict[str, list[str]] = field(default_factory=dict)
    #: class qname -> {field short name -> field qname}
    fields_by_class: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Files that actually parsed. The pyan pass must be given THIS list, not
    #: the discovered one: pyan builds a symbol table with `symtable.symtable()`,
    #: which raises on a file that does not compile and takes the whole
    #: extraction down with it. Real repos ship such files on purpose -- Django
    #: has `tests_syntax_error.py` to exercise its own test runner.
    parsed_files: list[Path] = field(default_factory=list)


def module_qname(py_file: Path, package_root: Path) -> str:
    """Dotted module name of `py_file` relative to `package_root`."""
    rel = py_file.resolve().relative_to(package_root.resolve())
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def _callee_name(node: ast.expr) -> str | None:
    """Short name of a call target: ``a.b.c()`` -> ``c``, ``f()`` -> ``f``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):        # e.g. factory()()
        return _callee_name(node.func)
    return None


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    try:
        args = ast.unparse(node.args)
    except Exception:                      # pragma: no cover - defensive
        args = "..."
    returns = ""
    if node.returns is not None:
        try:
            returns = f" -> {ast.unparse(node.returns)}"
        except Exception:                  # pragma: no cover - defensive
            returns = ""
    return f"{prefix} {node.name}({args}){returns}"


def _class_signature(node: ast.ClassDef) -> str:
    parts = []
    for base in node.bases:
        try:
            parts.append(ast.unparse(base))
        except Exception:                  # pragma: no cover - defensive
            parts.append("?")
    for kw in node.keywords:
        try:
            parts.append(f"{kw.arg}={ast.unparse(kw.value)}")
        except Exception:                  # pragma: no cover - defensive
            pass
    inner = f"({', '.join(parts)})" if parts else ""
    return f"class {node.name}{inner}"


class _ModuleVisitor(ast.NodeVisitor):
    """Walks one module, emitting nodes/edges into the shared AstResult."""

    def __init__(self, result: AstResult, repo_id: str, mod_qname: str, file_path: str):
        self.r = result
        self.repo_id = repo_id
        self.mod = mod_qname
        self.file_path = file_path
        #: alias -> fully qualified name, from imports in this module
        self.imports: dict[str, str] = {}
        #: (kind, qname) stack; kind in {"class", "func"}
        self.stack: list[tuple[str, str]] = []
        #: raw base expressions per class, resolved after the whole repo is walked
        self.raw_bases: dict[str, list[ast.expr]] = {}

    # -- helpers ---------------------------------------------------------

    def _add(self, node: Node) -> Node:
        return self.r.nodes.add(node)

    def _edge(self, etype: str, src: Node, tgt: Node, **attrs) -> None:
        self.r.edges.append(
            Edge(type=etype, source=src.id, target=tgt.id, repo_id=self.repo_id, attrs=attrs)
        )

    def _code_ref(self, node: ast.AST) -> tuple[str, int, int]:
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", start) or start
        return f"{self.file_path}:{start}-{end}", start, end

    @property
    def _enclosing_class(self) -> str | None:
        for kind, qname in reversed(self.stack):
            if kind == "class":
                return qname
        return None

    @property
    def _enclosing_callable(self) -> str | None:
        for kind, qname in reversed(self.stack):
            if kind == "func":
                return qname
        return None

    @property
    def _at_module_level(self) -> bool:
        return not self.stack

    # -- imports ---------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports[alias.asname or alias.name.split(".")[0]] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            # Relative import: walk up from this module's package.
            pkg_parts = self.mod.split(".")[:-1] if "." in self.mod else [self.mod]
            base_parts = pkg_parts[: len(pkg_parts) - (node.level - 1)] if node.level > 1 else pkg_parts
            base = ".".join(base_parts)
            module = f"{base}.{node.module}" if node.module else base
        else:
            module = node.module or ""
        for alias in node.names:
            local = alias.asname or alias.name
            self.imports[local] = f"{module}.{alias.name}" if module else alias.name
        self.generic_visit(node)

    # -- definitions -----------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qname = f"{self._enclosing_scope_prefix()}.{node.name}"
        code_ref, start, end = self._code_ref(node)
        cls = self._add(
            Node(
                id=make_id(self.repo_id, CLASS, qname),
                label=CLASS,
                name=node.name,
                qname=qname,
                repo_id=self.repo_id,
                file_path=self.file_path,
                signature=_class_signature(node),
                code_ref=code_ref,
                start_line=start,
                end_line=end,
                module_qname=self.mod,
            )
        )
        if self._at_module_level:
            self._edge(CONTAINS, self.r.nodes.get(self.mod, MODULE), cls)
        self.raw_bases[qname] = list(node.bases)
        self.r.fields_by_class.setdefault(qname, {})

        # Class-body assignments are fields too (`role = "user"`, `x: int`).
        for stmt in node.body:
            targets: list[str] = []
            if isinstance(stmt, ast.Assign):
                targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                targets = [stmt.target.id]
            for name in targets:
                self._add_field(cls, name, stmt)

        self.stack.append(("class", qname))
        self.generic_visit(node)
        self.stack.pop()

    def _enclosing_scope_prefix(self) -> str:
        return self.stack[-1][1] if self.stack else self.mod

    def _add_field(self, cls: Node, name: str, at: ast.AST) -> Node:
        qname = f"{cls.qname}.{name}"
        code_ref, start, end = self._code_ref(at)
        fld = self._add(
            Node(
                id=make_id(self.repo_id, FIELD, qname),
                label=FIELD,
                name=name,
                qname=qname,
                repo_id=self.repo_id,
                file_path=self.file_path,
                code_ref=code_ref,
                start_line=start,
                end_line=end,
                class_qname=cls.qname,
                module_qname=self.mod,
            )
        )
        known = self.r.fields_by_class.setdefault(cls.qname, {})
        if name not in known:
            known[name] = qname
            self._edge(HAS_FIELD, cls, fld)
        return fld

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        cls_qname = self._enclosing_class
        # A def directly inside a class body is a METHOD; anything else (module
        # level, or nested inside another def) is a FUNCTION.
        is_method = bool(self.stack) and self.stack[-1][0] == "class"
        label = METHOD if is_method else FUNCTION
        qname = f"{self._enclosing_scope_prefix()}.{node.name}"
        code_ref, start, end = self._code_ref(node)
        fn = self._add(
            Node(
                id=make_id(self.repo_id, label, qname),
                label=label,
                name=node.name,
                qname=qname,
                repo_id=self.repo_id,
                file_path=self.file_path,
                signature=_signature(node),
                code_ref=code_ref,
                start_line=start,
                end_line=end,
                class_qname=cls_qname if is_method else None,
                module_qname=self.mod,
            )
        )
        if is_method:
            self._edge(HAS_METHOD, self.r.nodes.get(cls_qname, CLASS), fn)
        elif self._at_module_level:
            self._edge(CONTAINS, self.r.nodes.get(self.mod, MODULE), fn)

        self.r.call_sites.setdefault(qname, defaultdict(list))
        self.stack.append(("func", qname))
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    # -- module-level variables -----------------------------------------

    def _visit_assign(self, node: ast.Assign | ast.AnnAssign) -> None:
        if self._at_module_level:
            targets = (
                [t for t in node.targets if isinstance(t, ast.Name)]
                if isinstance(node, ast.Assign)
                else ([node.target] if isinstance(node.target, ast.Name) else [])
            )
            for target in targets:
                qname = f"{self.mod}.{target.id}"
                code_ref, start, end = self._code_ref(node)
                gv = self._add(
                    Node(
                        id=make_id(self.repo_id, GLOBAL_VARIABLE, qname),
                        label=GLOBAL_VARIABLE,
                        name=target.id,
                        qname=qname,
                        repo_id=self.repo_id,
                        file_path=self.file_path,
                        code_ref=code_ref,
                        start_line=start,
                        end_line=end,
                        module_qname=self.mod,
                    )
                )
                self._edge(CONTAINS, self.r.nodes.get(self.mod, MODULE), gv)
        self.generic_visit(node)

    visit_Assign = _visit_assign
    visit_AnnAssign = _visit_assign

    # -- field definition and access ------------------------------------

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # `self.x` inside a method: a store defines the field, a load uses it.
        cls_qname = self._enclosing_class
        if (
            cls_qname is not None
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            cls = self.r.nodes.get(cls_qname, CLASS)
            if cls is not None:
                if isinstance(node.ctx, ast.Store):
                    self._add_field(cls, node.attr, node)
                # Loads are resolved to a field (possibly inherited) in a later
                # pass, once every class in the repo is known.
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        caller = self._enclosing_callable
        name = _callee_name(node.func)
        if caller is not None and name is not None:
            sites = self.r.call_sites.setdefault(caller, defaultdict(list))
            sites[name].append(node.lineno)
            # `from .joiner import cli as remote_cli; remote_cli()` calls `cli`.
            # Recording only the local alias made the call-site cross-check
            # reject pyan's correct edge to `joiner.cli` -- and the completeness
            # annotation, which matches by the same short names, never flagged
            # it as missed either. Record the imported name as well.
            if isinstance(node.func, ast.Name) and name in self.imports:
                original = self.imports[name].rsplit(".", 1)[-1]
                if original != name:
                    sites[original].append(node.lineno)
        self.generic_visit(node)


class _FieldUseVisitor(ast.NodeVisitor):
    """Second walk: emit USES edges to fields, resolving inherited fields."""

    def __init__(self, result: AstResult, repo_id: str, mod_qname: str, resolve_field):
        self.r = result
        self.repo_id = repo_id
        self.mod = mod_qname
        self.resolve_field = resolve_field
        self.stack: list[tuple[str, str]] = []
        self.seen: set[tuple[str, str, str]] = set()

    def _prefix(self) -> str:
        return self.stack[-1][1] if self.stack else self.mod

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(("class", f"{self._prefix()}.{node.name}"))
        self.generic_visit(node)
        self.stack.pop()

    def _visit_function(self, node) -> None:
        self.stack.append(("func", f"{self._prefix()}.{node.name}"))
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Attribute(self, node: ast.Attribute) -> None:
        cls_qname = next((q for k, q in reversed(self.stack) if k == "class"), None)
        fn_qname = next((q for k, q in reversed(self.stack) if k == "func"), None)
        if (
            cls_qname
            and fn_qname
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            target_q = self.resolve_field(cls_qname, node.attr)
            fn = self.r.nodes.get(fn_qname, FUNCTION, METHOD)
            fld = self.r.nodes.get(target_q, FIELD) if target_q else None
            if fn is not None and fld is not None:
                access = "write" if isinstance(node.ctx, ast.Store) else "read"
                key = (fn_qname, target_q, access)
                if key not in self.seen:
                    self.seen.add(key)
                    self.r.edges.append(
                        Edge(
                            type=USES,
                            source=fn.id,
                            target=fld.id,
                            repo_id=self.repo_id,
                            attrs={
                                "access": access,
                                "source_type": fn.label,
                                "target_type": FIELD,
                            },
                        )
                    )
        self.generic_visit(node)


def run_ast_pass(
    repo_id: str,
    py_files: list[Path],
    package_root: Path,
    repo_root: Path,
) -> AstResult:
    """Parse every file, emit structural nodes/edges, resolve bases and fields."""
    result = AstResult()
    parsed: list[tuple[str, ast.Module, _ModuleVisitor]] = []
    unparseable: list[str] = []
    log.info("AST pass: parsing %d file(s)", len(py_files))

    for n, py_file in enumerate(py_files, 1):
        mod = module_qname(py_file, package_root)
        rel_path = str(py_file.resolve().relative_to(repo_root.resolve()))
        source = py_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError as exc:
            # Previously swallowed. A file that does not parse contributes no
            # nodes or edges, so every question derived from this repo is built
            # on a graph that is silently missing it -- say so.
            unparseable.append(rel_path)
            log.warning("skipping %s — does not parse: %s (line %s)",
                        rel_path, exc.msg, exc.lineno)
            continue
        result.nodes.add(
            Node(
                id=make_id(repo_id, MODULE, mod),
                label=MODULE,
                name=mod.split(".")[-1],
                qname=mod,
                repo_id=repo_id,
                file_path=rel_path,
                module_qname=mod,
            )
        )
        visitor = _ModuleVisitor(result, repo_id, mod, rel_path)
        visitor.visit(tree)
        parsed.append((mod, tree, visitor))
        result.parsed_files.append(py_file)
        log.debug("  parsed %s (%s)", mod, rel_path)
        if n % 50 == 0:
            log.info("  ...%d/%d files, %d nodes so far", n, len(py_files), len(result.nodes))

    log.info("AST pass: %d file(s) parsed, %d node(s)%s",
             len(parsed), len(result.nodes),
             f", {len(unparseable)} unparseable" if unparseable else "")

    # --- resolve INHERITS once every class in the repo is known ---------
    n_inherits = 0
    unresolved_bases = 0
    for mod, _tree, visitor in parsed:
        for cls_qname, base_exprs in visitor.raw_bases.items():
            resolved: list[str] = []
            for base in base_exprs:
                target = _resolve_base(base, mod, visitor.imports, result.nodes)
                if target is None:
                    # Almost always a third-party or stdlib base class, which is
                    # out of scope by design rather than a failure.
                    unresolved_bases += 1
                    continue
                resolved.append(target)
                n_inherits += 1
                result.edges.append(
                    Edge(
                        type=INHERITS,
                        source=result.nodes.get(cls_qname, CLASS).id,
                        target=result.nodes.get(target, CLASS).id,
                        repo_id=repo_id,
                    )
                )
            result.bases[cls_qname] = resolved

    # --- resolve self.<attr> reads/writes to (possibly inherited) fields -
    def resolve_field(cls_qname: str, attr: str) -> str | None:
        seen: set[str] = set()
        queue = [cls_qname]
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            hit = result.fields_by_class.get(current, {}).get(attr)
            if hit:
                return hit
            queue.extend(result.bases.get(current, []))
        return None

    log.info("AST pass: %d INHERITS edge(s) resolved in-repo, %d base(s) outside the repo",
             n_inherits, unresolved_bases)

    before = len(result.edges)
    for mod, tree, _visitor in parsed:
        _FieldUseVisitor(result, repo_id, mod, resolve_field).visit(tree)
    log.info("AST pass: %d USES edge(s) to fields", len(result.edges) - before)

    return result


def _resolve_base(
    base: ast.expr,
    mod: str,
    imports: dict[str, str],
    nodes: "NodeIndex",
) -> str | None:
    """Map a base-class expression to an in-repo class qname, or None."""
    if isinstance(base, ast.Name):
        candidates = [f"{mod}.{base.id}", imports.get(base.id, "")]
    elif isinstance(base, ast.Attribute):
        try:
            dotted = ast.unparse(base)
        except Exception:                  # pragma: no cover - defensive
            return None
        head, _, tail = dotted.partition(".")
        candidates = [dotted, f"{imports.get(head, head)}.{tail}" if tail else dotted]
    else:
        return None
    for cand in candidates:
        if nodes.has(cand, CLASS):
            return cand
    return None
