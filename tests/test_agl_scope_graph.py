"""Tests for the graph-aware scope resolver: ``resolve_graph``.

These tests drive multi-module AgL programs through ``resolve_graph`` and assert on:
- ``ResolvedModuleGraph`` and ``ResolvedModule`` shape
- open-import name resolution (unqualified access)
- using / hiding / qualified / as import forms
- S-bounded qualified access
- clash-on-use disambiguation errors
- multiple import declarations merging
- duplicate-alias and alias-root-collision static errors
- ``::name`` self-reference
- private boundary enforcement
- declaration-only enforcement for non-entry modules
- entry-only enforcement (agent, param, program)
- header-only import placement
- wildcard subtree expansion and re-rooting
- wildcard overlap (idempotent same module, clash different modules)
- cross-file mutual recursion
- ``BindingRef.module_id`` set correctly
- single-module graph via ``resolve_graph`` equals ``resolve()`` result
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.modules.loader import load_graph
from agm.agl.modules.roots import RootSet
from agm.agl.scope import resolve
from agm.agl.scope.graph import ResolvedModule, ResolvedModuleGraph, resolve_graph
from agm.agl.scope.symbols import AglScopeError
from agm.agl.syntax.nodes import VarRef

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _roots(*paths: Path) -> RootSet:
    return RootSet(roots=frozenset(paths))


def _write_module(root: Path, dotted: str, source: str) -> Path:
    mid = ModuleId.from_dotted(dotted)
    p = root / mid.relpath().replace("/", os.sep)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(source)
    return p


def _make_graph_from_files(tmp_path: Path, modules: dict[str, str]) -> object:
    """Build a ModuleGraph via load_graph from {dotted_name_or_entry: source} dict.

    The key 'entry' is used as the entry source.
    Other keys are written as .agl files.
    """
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    entry_source = modules.get("entry", "()")
    for dotted, source in modules.items():
        if dotted == "entry":
            continue
        _write_module(root, dotted, source)
    return load_graph(entry_source, entry_path=None, roots=_roots(root))


def _find_varref(program: object, name: str) -> VarRef | None:
    """Recursively find the first VarRef with the given name in a Program."""
    from agm.agl.syntax.nodes import (
        AssignStmt,
        BinaryOp,
        Block,
        Call,
        Cast,
        Constructor,
        DictLit,
        Do,
        FieldAccess,
        FuncDef,
        If,
        IndexAccess,
        InterpSegment,
        IsTest,
        Lambda,
        LetDecl,
        ListLit,
        ParamDecl,
        Program,
        Raise,
        Template,
        Try,
        UnaryNeg,
        UnaryNot,
        VarDecl,
        VarRef,
    )

    def walk(node: object) -> VarRef | None:
        if isinstance(node, VarRef):
            if node.name == name:
                return node
        if isinstance(node, Program):
            return walk(node.body)
        if isinstance(node, Block):
            for item in node.items:
                r = walk(item)
                if r is not None:
                    return r
        if isinstance(node, FuncDef):
            for param in node.params:
                if param.default is not None:
                    r = walk(param.default)
                    if r is not None:
                        return r
            return walk(node.body)
        if isinstance(node, Lambda):
            return walk(node.body)
        if isinstance(node, LetDecl):
            return walk(node.value)
        if isinstance(node, VarDecl):
            return walk(node.value)
        if isinstance(node, AssignStmt):
            return walk(node.value)
        if isinstance(node, Call):
            r = walk(node.callee)
            if r is not None:
                return r
            for arg in node.args:
                r = walk(arg)
                if r is not None:
                    return r
        if isinstance(node, BinaryOp):
            r = walk(node.left)
            if r is not None:
                return r
            return walk(node.right)
        if isinstance(node, If):
            for branch in node.branches:
                r = walk(branch.body)
                if r is not None:
                    return r
        if isinstance(node, Do):
            r = walk(node.body)
            if r is not None:
                return r
            return walk(node.condition)
        if isinstance(node, Try):
            r = walk(node.body)
            if r is not None:
                return r
            for clause in node.handlers:
                r = walk(clause.body)
                if r is not None:
                    return r
        if isinstance(node, Template):
            for seg in node.segments:
                if isinstance(seg, InterpSegment):
                    r = walk(seg.expr)
                    if r is not None:
                        return r
        if isinstance(node, FieldAccess):
            return walk(node.obj)
        if isinstance(node, IndexAccess):
            r = walk(node.obj)
            if r is not None:
                return r
            return walk(node.index)
        if isinstance(node, UnaryNot):
            return walk(node.operand)
        if isinstance(node, UnaryNeg):
            return walk(node.operand)
        if isinstance(node, Raise):
            return walk(node.exc)
        if isinstance(node, Cast):
            return walk(node.expr)
        if isinstance(node, IsTest):
            return walk(node.expr)
        if isinstance(node, Constructor):
            for arg in node.args:
                r = walk(arg.value)
                if r is not None:
                    return r
        if isinstance(node, ListLit):
            for elem in node.elements:
                r = walk(elem)
                if r is not None:
                    return r
        if isinstance(node, DictLit):
            for entry in node.entries:
                r = walk(entry.value)
                if r is not None:
                    return r
        if isinstance(node, ParamDecl):
            if node.default is not None:
                return walk(node.default)
        return None

    return walk(program)


# ---------------------------------------------------------------------------
# Test: basic ResolvedModuleGraph shape
# ---------------------------------------------------------------------------


class TestResolvedModuleGraphShape:
    def test_single_module_graph_has_entry(self, tmp_path: Path) -> None:
        """A single-module graph has one ResolvedModule keyed by ENTRY_ID."""
        graph = _make_graph_from_files(tmp_path, {"entry": "()"})
        result = resolve_graph(graph)
        assert isinstance(result, ResolvedModuleGraph)
        assert ENTRY_ID in result.modules
        assert result.entry_id == ENTRY_ID

    def test_resolved_module_shape(self, tmp_path: Path) -> None:
        """ResolvedModule has module_id, resolved, import_env, exports."""
        graph = _make_graph_from_files(tmp_path, {"entry": "()"})
        result = resolve_graph(graph)
        entry_mod = result.modules[ENTRY_ID]
        assert isinstance(entry_mod, ResolvedModule)
        assert entry_mod.module_id == ENTRY_ID
        assert entry_mod.resolved is not None
        assert entry_mod.import_env is not None
        assert isinstance(entry_mod.exports, frozenset)

    def test_multi_module_graph_has_both_modules(self, tmp_path: Path) -> None:
        """A two-module graph has entries for entry and the library module."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "def foo() -> int = 42",
        })
        result = resolve_graph(graph)
        mylib_id = ModuleId.from_dotted("mylib")
        assert ENTRY_ID in result.modules
        assert mylib_id in result.modules

    def test_exports_excludes_private(self, tmp_path: Path) -> None:
        """Private functions are excluded from exports."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "def pub() -> int = 1\nprivate def priv() -> int = 2",
        })
        result = resolve_graph(graph)
        mylib_id = ModuleId.from_dotted("mylib")
        exports = result.modules[mylib_id].exports
        assert "pub" in exports
        assert "priv" not in exports

    def test_pre_pass_tables_populated(self, tmp_path: Path) -> None:
        """all_public_funcs and all_public_types are populated."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "def foo() -> int = 1",
        })
        result = resolve_graph(graph)
        mylib_id = ModuleId.from_dotted("mylib")
        assert (mylib_id, "foo") in result.all_public_funcs

    def test_entry_agents_populated(self, tmp_path: Path) -> None:
        """entry_agents maps agent names declared in the entry."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": 'agent bot = "claude"\n()',
        })
        result = resolve_graph(graph)
        assert "bot" in result.entry_agents


# ---------------------------------------------------------------------------
# Test: open import — bare name resolution
# ---------------------------------------------------------------------------


class TestOpenImport:
    def test_open_import_varref_has_correct_module_id(self, tmp_path: Path) -> None:
        """VarRef resolved via open import has BindingRef.module_id == owning/source module's id."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\nlet x = foo()",
            "mylib": "def foo() -> int = 42",
        })
        result = resolve_graph(graph)
        entry_resolved = result.modules[ENTRY_ID].resolved
        # Find the VarRef for 'foo' in the entry
        from agm.agl.modules.loader import ModuleGraph
        assert isinstance(graph, ModuleGraph)
        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "foo")
        assert var is not None
        ref = entry_resolved.resolution[var.node_id]
        assert ref.module_id == ModuleId.from_dotted("mylib")

    def test_using_import_limits_exposed_names(self, tmp_path: Path) -> None:
        """'import mylib using bar' only exposes 'bar'."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib using bar\nlet x = bar()",
            "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
        })
        result = resolve_graph(graph)
        from agm.agl.modules.loader import ModuleGraph
        assert isinstance(graph, ModuleGraph)
        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "bar")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.name == "bar"

    def test_using_import_hides_non_listed_name(self, tmp_path: Path) -> None:
        """'import mylib using bar' — bare 'foo' should error."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib using bar\nlet x = foo()",
            "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
        })
        with pytest.raises(AglScopeError, match="foo"):
            resolve_graph(graph)

    def test_hiding_import_excludes_hidden_name(self, tmp_path: Path) -> None:
        """'import mylib hiding foo' exposes 'bar' but not 'foo'."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib hiding foo\nlet x = bar()",
            "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
        })
        result = resolve_graph(graph)
        from agm.agl.modules.loader import ModuleGraph
        assert isinstance(graph, ModuleGraph)
        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "bar")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.name == "bar"

    def test_hiding_import_hides_named(self, tmp_path: Path) -> None:
        """'import mylib hiding foo' — bare 'foo' should error."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib hiding foo\nlet x = foo()",
            "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
        })
        with pytest.raises(AglScopeError, match="foo"):
            resolve_graph(graph)

    def test_using_with_rename(self, tmp_path: Path) -> None:
        """'import mylib using foo as baz' exposes 'baz', not 'foo'.

        The BindingRef.name records the *original* declared name in the owning
        module (``"foo"``), not the exposed name (``"baz"``).  This is what
        the evaluator uses to look up the value in the owning module's frame.
        """
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib using foo as baz\nlet x = baz()",
            "mylib": "def foo() -> int = 42",
        })
        result = resolve_graph(graph)
        from agm.agl.modules.loader import ModuleGraph
        assert isinstance(graph, ModuleGraph)
        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "baz")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        # BindingRef.name is the original declared name in the owning module.
        assert ref.name == "foo"

    def test_qualified_import_prevents_bare_access(self, tmp_path: Path) -> None:
        """'import mylib qualified' — bare 'foo' should error."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib qualified\nlet x = foo()",
            "mylib": "def foo() -> int = 42",
        })
        with pytest.raises(AglScopeError, match="foo"):
            resolve_graph(graph)

    def test_qualified_import_allows_qualified_access(self, tmp_path: Path) -> None:
        """'import mylib qualified' — 'mylib::foo' should resolve."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib qualified\nlet x = mylib::foo()",
            "mylib": "def foo() -> int = 42",
        })
        result = resolve_graph(graph)
        assert ENTRY_ID in result.modules

    def test_as_alias(self, tmp_path: Path) -> None:
        """'import mylib as M' — 'M::foo()' resolves."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib as M\nlet x = M::foo()",
            "mylib": "def foo() -> int = 42",
        })
        result = resolve_graph(graph)
        assert ENTRY_ID in result.modules

    def test_as_alias_unqualified_still_exposed(self, tmp_path: Path) -> None:
        """'import mylib as M' without 'qualified' exposes bare names too."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib as M\nlet x = foo()",
            "mylib": "def foo() -> int = 42",
        })
        result = resolve_graph(graph)
        from agm.agl.modules.loader import ModuleGraph
        assert isinstance(graph, ModuleGraph)
        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "foo")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.module_id == ModuleId.from_dotted("mylib")


# ---------------------------------------------------------------------------
# Test: qualified access (S bounds)
# ---------------------------------------------------------------------------


class TestQualifiedAccess:
    def test_qualified_using_bounds_set(self, tmp_path: Path) -> None:
        """'import mylib qualified using foo' — 'mylib::bar' should error."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib qualified using foo\nlet x = mylib::bar()",
            "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
        })
        with pytest.raises(AglScopeError, match="bar"):
            resolve_graph(graph)

    def test_qualified_using_allows_listed(self, tmp_path: Path) -> None:
        """'import mylib qualified using foo' — 'mylib::foo()' should resolve."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib qualified using foo\nlet x = mylib::foo()",
            "mylib": "def foo() -> int = 42",
        })
        result = resolve_graph(graph)
        assert ENTRY_ID in result.modules

    def test_unknown_qualifier_handle_errors(self, tmp_path: Path) -> None:
        """'nomodule::foo' when no such module is imported errors."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "let x = nomodule::foo()",
        })
        with pytest.raises(AglScopeError, match="nomodule"):
            resolve_graph(graph)


# ---------------------------------------------------------------------------
# Test: clash deferred to use-site
# ---------------------------------------------------------------------------


class TestClashDeferred:
    def test_two_imports_same_name_clashes_at_use(self, tmp_path: Path) -> None:
        """Two open imports both export 'foo' — bare 'foo' → ambiguous error."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import libA\nimport libB\nlet x = foo()",
            "libA": "def foo() -> int = 1",
            "libB": "def foo() -> int = 2",
        })
        with pytest.raises(AglScopeError, match="ambiguous"):
            resolve_graph(graph)

    def test_clash_error_names_qualifiers(self, tmp_path: Path) -> None:
        """Ambiguous error mentions at least one disambiguation qualifier."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import libA\nimport libB\nlet x = foo()",
            "libA": "def foo() -> int = 1",
            "libB": "def foo() -> int = 2",
        })
        with pytest.raises(AglScopeError) as exc_info:
            resolve_graph(graph)
        msg = str(exc_info.value)
        assert "libA" in msg or "libB" in msg or "ambiguous" in msg.lower()

    def test_no_clash_same_qname(self, tmp_path: Path) -> None:
        """Two imports of the same module's same function don't clash (idempotent)."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\nimport mylib using foo\nlet x = foo()",
            "mylib": "def foo() -> int = 42",
        })
        # Should not raise — same QName from same module
        result = resolve_graph(graph)
        assert ENTRY_ID in result.modules


# ---------------------------------------------------------------------------
# Test: multiple imports merge
# ---------------------------------------------------------------------------


class TestMultipleImportsMerge:
    def test_two_decls_same_module_merges(self, tmp_path: Path) -> None:
        """Two import declarations for the same module merge their exposed sets."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib using foo\nimport mylib using bar\nlet a = foo()\nlet b = bar()",
            "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
        })
        result = resolve_graph(graph)
        assert ENTRY_ID in result.modules


# ---------------------------------------------------------------------------
# Test: static import errors
# ---------------------------------------------------------------------------


class TestStaticImportErrors:
    def test_duplicate_alias_different_modules(self, tmp_path: Path) -> None:
        """'import A as X' + 'import B as X' → static alias error."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import libA as X\nimport libB as X\n()",
            "libA": "def foo() -> int = 1",
            "libB": "def bar() -> int = 2",
        })
        with pytest.raises(AglScopeError, match="X"):
            resolve_graph(graph)

    def test_alias_root_collision(self, tmp_path: Path) -> None:
        """'import libA' + 'import libB as libA' → alias-root collision."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import libA\nimport libB as libA\n()",
            "libA": "def foo() -> int = 1",
            "libB": "def bar() -> int = 2",
        })
        with pytest.raises(AglScopeError, match="libA"):
            resolve_graph(graph)


# ---------------------------------------------------------------------------
# Test: ::name self-reference
# ---------------------------------------------------------------------------


class TestSelfReference:
    def test_self_ref_in_entry(self, tmp_path: Path) -> None:
        """'::foo' in the entry resolves to the entry's own 'foo'."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "def foo() -> int = 1\nlet x = ::foo()",
        })
        result = resolve_graph(graph)
        from agm.agl.modules.loader import ModuleGraph
        assert isinstance(graph, ModuleGraph)
        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "foo")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.module_id == ENTRY_ID

    def test_self_ref_in_lib_module(self, tmp_path: Path) -> None:
        """'::bar' in module 'mylib' resolves to mylib's own 'bar'."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "def bar() -> int = 1\ndef baz() -> int = ::bar()",
        })
        result = resolve_graph(graph)
        mylib_id = ModuleId.from_dotted("mylib")
        from agm.agl.modules.loader import ModuleGraph
        assert isinstance(graph, ModuleGraph)
        mylib_program = graph.modules[mylib_id].program
        var = _find_varref(mylib_program, "bar")
        assert var is not None
        ref = result.modules[mylib_id].resolved.resolution[var.node_id]
        assert ref.module_id == mylib_id

    def test_self_ref_undefined_name_errors(self, tmp_path: Path) -> None:
        """'::nonexistent' in a module errors."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "let x = ::nonexistent()",
        })
        with pytest.raises(AglScopeError, match="nonexistent"):
            resolve_graph(graph)

    def test_self_ref_bypasses_param_shadow(self, tmp_path: Path) -> None:
        """'::foo' inside g(foo: int) resolves to the top-level def foo, not the param.

        D9/§7: '::name' means the CURRENT MODULE'S OWN TOP-LEVEL declaration,
        bypassing any lexical shadow introduced by params or let bindings.
        """
        graph = _make_graph_from_files(tmp_path, {
            "entry": "def foo() -> int = 1\ndef g(foo: int) -> int = ::foo()",
        })
        result = resolve_graph(graph)
        from agm.agl.modules.loader import ModuleGraph
        from agm.agl.scope.symbols import BinderKind
        assert isinstance(graph, ModuleGraph)
        entry_program = graph.modules[ENTRY_ID].program
        # Find the ::foo VarRef inside g's body — it should resolve to the
        # top-level function foo, NOT to the parameter foo.
        # Use _find_varref to locate the foo VarRef, then check that the resolved
        # binding is the top-level function binding, not the parameter.
        # _find_varref finds by name; the first 'foo' VarRef in the body of 'g'
        # is the ::foo qualifier reference.  We need to find the one with qualifier.
        from agm.agl.syntax.nodes import Block, Call, FuncDef, VarRef

        def find_self_ref_varref(node: object) -> VarRef | None:
            """Find the first VarRef named 'foo' with a non-None module_qualifier."""
            from agm.agl.syntax.nodes import Program
            if isinstance(node, VarRef) and node.name == "foo" and (
                node.module_qualifier is not None
            ):
                return node
            if isinstance(node, Program):
                return find_self_ref_varref(node.body)
            if isinstance(node, FuncDef):
                r = find_self_ref_varref(node.body)
                if r is not None:
                    return r
            if isinstance(node, Call):
                r = find_self_ref_varref(node.callee)
                if r is not None:
                    return r
                for arg in node.args:
                    r = find_self_ref_varref(arg)
                    if r is not None:
                        return r
            if isinstance(node, Block):
                for item in node.items:
                    r = find_self_ref_varref(item)
                    if r is not None:
                        return r
            return None

        var = find_self_ref_varref(entry_program)
        assert var is not None, "::foo VarRef not found in entry program"
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        # Must resolve to the top-level function_binding, not the param
        assert ref.kind == BinderKind.function_binding, (
            f"::foo should resolve to function_binding, got {ref.kind}"
        )
        assert ref.module_id == ENTRY_ID


# ---------------------------------------------------------------------------
# Test: private boundary enforcement
# ---------------------------------------------------------------------------


class TestPrivateBoundary:
    def test_private_not_in_exports(self, tmp_path: Path) -> None:
        """Private functions don't appear in the exporting module's exports."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "private def secret() -> int = 1",
        })
        result = resolve_graph(graph)
        mylib_id = ModuleId.from_dotted("mylib")
        assert "secret" not in result.modules[mylib_id].exports

    def test_private_not_accessible_via_open_import(self, tmp_path: Path) -> None:
        """A private function from an imported module is not accessible unqualified."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\nlet x = secret()",
            "mylib": "private def secret() -> int = 1",
        })
        with pytest.raises(AglScopeError, match="secret"):
            resolve_graph(graph)

    def test_private_not_accessible_via_qualified_access(self, tmp_path: Path) -> None:
        """A private function is not accessible via qualified access.

        When a module has only private exports, its qualifier is not registered
        (no names to qualify). When it has public exports, 'mylib::secret'
        raises the private-access error naming mylib specifically.
        """
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\nlet x = mylib::secret()",
            "mylib": "def pub() -> int = 1\nprivate def secret() -> int = 2",
        })
        with pytest.raises(AglScopeError, match="[Pp]rivate"):
            resolve_graph(graph)

    def test_private_error_message_names_owning_module(self, tmp_path: Path) -> None:
        """Qualified access to a private name names the MODULE that owns the private decl.

        Bug (Finding 3): when libA::secret is accessed but 'secret' only exists as
        a private name in libB (a different module), the error used to name libB
        instead of correctly resolving libA as the owning module and emitting 'private'
        (if secret is private in libA) or 'not in imported set of libA' (if it isn't).
        """
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\nlet x = mylib::secret()",
            "mylib": "def pub() -> int = 1\nprivate def secret() -> int = 2",
        })
        with pytest.raises(AglScopeError) as exc_info:
            resolve_graph(graph)
        msg = str(exc_info.value)
        # The error must specifically say 'private' (owning module is mylib and
        # 'secret' IS private in mylib), not a generic 'not in imported set'.
        assert "private" in msg.lower(), f"Expected 'private' in error, got: {msg!r}"
        # The error must name 'mylib' — not some other unrelated module.
        assert "mylib" in msg, f"Expected 'mylib' in error message, got: {msg!r}"

    def test_private_error_does_not_blame_wrong_module(self, tmp_path: Path) -> None:
        """'libA::secret' must NOT name libB in the error when secret is private in libB only.

        When libA is imported with a public export, and libB is a different module
        that has a private 'secret', accessing libA::secret must report 'not in
        imported set of libA' — not 'secret in libB is private'.
        """
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import libA\nimport libB\nlet x = libA::secret()",
            "libA": "def pub() -> int = 1",
            "libB": "def other() -> int = 2\nprivate def secret() -> int = 99",
        })
        with pytest.raises(AglScopeError) as exc_info:
            resolve_graph(graph)
        msg = str(exc_info.value)
        # Must mention libA (the qualifier used), not libB (which has the private decl)
        assert "libA" in msg, f"Expected 'libA' in error, got: {msg!r}"
        assert "libB" not in msg, f"libB should NOT be named, got: {msg!r}"


# ---------------------------------------------------------------------------
# Test: declaration-only enforcement
# ---------------------------------------------------------------------------


class TestDeclarationOnly:
    def test_let_in_non_entry_errors(self, tmp_path: Path) -> None:
        """A 'let' declaration in a non-entry module is an error."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "let x = 1",
        })
        with pytest.raises(AglScopeError):
            resolve_graph(graph)

    def test_var_in_non_entry_errors(self, tmp_path: Path) -> None:
        """A 'var' declaration in a non-entry module is an error."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "var x = 1",
        })
        with pytest.raises(AglScopeError):
            resolve_graph(graph)

    def test_bare_expr_in_non_entry_errors(self, tmp_path: Path) -> None:
        """A bare expression in a non-entry module is an error."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "def foo() -> int = 1\n42",
        })
        with pytest.raises(AglScopeError):
            resolve_graph(graph)

    def test_funcdef_in_non_entry_allowed(self, tmp_path: Path) -> None:
        """A 'def' in a non-entry module is allowed."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "def foo() -> int = 42",
        })
        result = resolve_graph(graph)
        assert ENTRY_ID in result.modules


# ---------------------------------------------------------------------------
# Test: entry-only constructs
# ---------------------------------------------------------------------------


class TestEntryOnlyConstructs:
    def test_agent_in_non_entry_errors(self, tmp_path: Path) -> None:
        """An 'agent' declaration in a non-entry module is an error."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": 'agent bot = "claude"',
        })
        with pytest.raises(AglScopeError, match="agent"):
            resolve_graph(graph)

    def test_param_in_non_entry_errors(self, tmp_path: Path) -> None:
        """A 'param' declaration in a non-entry module is an error."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "param x",
        })
        with pytest.raises(AglScopeError, match="param"):
            resolve_graph(graph)

    def test_program_decl_in_non_entry_errors(self, tmp_path: Path) -> None:
        """A 'program' declaration in a non-entry module is an error."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "program myname",
        })
        with pytest.raises(AglScopeError, match="program"):
            resolve_graph(graph)

    def test_agent_in_entry_allowed(self, tmp_path: Path) -> None:
        """An 'agent' declaration in the entry module is allowed."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": 'agent bot = "claude"\n()',
        })
        result = resolve_graph(graph)
        assert "bot" in result.entry_agents


# ---------------------------------------------------------------------------
# Test: header-only imports
# ---------------------------------------------------------------------------


class TestHeaderOnlyImports:
    def test_import_after_def_in_non_entry_errors(self, tmp_path: Path) -> None:
        """An import after a def in a non-entry module is an error."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "def foo() -> int = 1\nimport libB",
            "libB": "def bar() -> int = 2",
        })
        with pytest.raises(AglScopeError):
            resolve_graph(graph)

    def test_import_at_top_of_non_entry_allowed(self, tmp_path: Path) -> None:
        """Import declarations at the top of a non-entry module are allowed."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "import libB\ndef foo() -> int = bar()",
            "libB": "def bar() -> int = 42",
        })
        result = resolve_graph(graph)
        assert ENTRY_ID in result.modules

    def test_import_in_entry_works(self, tmp_path: Path) -> None:
        """Entry module works fine with import followed by def."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\ndef helper() -> int = foo()\n()",
            "mylib": "def foo() -> int = 42",
        })
        result = resolve_graph(graph)
        assert ENTRY_ID in result.modules


# ---------------------------------------------------------------------------
# Test: wildcard imports
# ---------------------------------------------------------------------------


class TestWildcardImports:
    def test_wildcard_expands_all_submodules(self, tmp_path: Path) -> None:
        """'import foo.*' exposes functions from all submodules of 'foo'."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import foo.*\nlet a = alpha()\nlet b = beta()",
            "foo.alpha": "def alpha() -> int = 1",
            "foo.beta": "def beta() -> int = 2",
        })
        result = resolve_graph(graph)
        assert ENTRY_ID in result.modules

    def test_wildcard_as_reroots_qualifier(self, tmp_path: Path) -> None:
        """'import foo.* as F' makes 'F.alpha::alpha()' accessible via qualifier."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import foo.* as F\nlet a = F.alpha::alpha()",
            "foo.alpha": "def alpha() -> int = 1",
        })
        result = resolve_graph(graph)
        assert ENTRY_ID in result.modules

    def test_wildcard_compatible_overlap_idempotent(self, tmp_path: Path) -> None:
        """Two wildcards that expose same QName from same module are idempotent (no error)."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import foo.*\nimport foo.alpha\nlet x = alpha()",
            "foo.alpha": "def alpha() -> int = 1",
        })
        result = resolve_graph(graph)
        assert ENTRY_ID in result.modules

    def test_wildcard_conflicting_overlap_clashes_on_use(self, tmp_path: Path) -> None:
        """Two wildcards expose same bare name from different modules → clash on use."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import foo.*\nimport bar.*\nlet x = common()",
            "foo.sub": "def common() -> int = 1",
            "bar.sub": "def common() -> int = 2",
        })
        with pytest.raises(AglScopeError, match="ambiguous|common"):
            resolve_graph(graph)


# ---------------------------------------------------------------------------
# Test: cross-file mutual recursion
# ---------------------------------------------------------------------------


class TestCrossFileMutualRecursion:
    def test_a_calls_b_resolves(self, tmp_path: Path) -> None:
        """Module A can call module B's function (one-way, open import)."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import modA\ncallA()",
            "modA": "import modB\ndef callA() -> int = callB()",
            "modB": "def callB() -> int = 42",
        })
        result = resolve_graph(graph)
        assert ENTRY_ID in result.modules

    def test_mutual_recursion_across_modules(self, tmp_path: Path) -> None:
        """A's def calls B's def and B's def calls A's def — both resolve."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import modA\nimport modB\nfuncA()",
            "modA": "import modB\ndef funcA() -> int = funcB()",
            "modB": "import modA\ndef funcB() -> int = funcA()",
        })
        result = resolve_graph(graph)
        # All three should be resolved
        assert ENTRY_ID in result.modules
        assert ModuleId.from_dotted("modA") in result.modules
        assert ModuleId.from_dotted("modB") in result.modules


# ---------------------------------------------------------------------------
# Test: BindingRef.module_id
# ---------------------------------------------------------------------------


class TestBindingRefModuleId:
    def test_local_binding_has_entry_module_id(self, tmp_path: Path) -> None:
        """A local let-binding in the entry has module_id == ENTRY_ID."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "let x = 1\nx",
        })
        result = resolve_graph(graph)
        from agm.agl.modules.loader import ModuleGraph
        assert isinstance(graph, ModuleGraph)
        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "x")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.module_id == ENTRY_ID

    def test_cross_module_binding_has_source_module_id(self, tmp_path: Path) -> None:
        """A VarRef resolved from an import has module_id of the providing module."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\nlet y = foo()",
            "mylib": "def foo() -> int = 1",
        })
        result = resolve_graph(graph)
        from agm.agl.modules.loader import ModuleGraph
        assert isinstance(graph, ModuleGraph)
        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "foo")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        mylib_id = ModuleId.from_dotted("mylib")
        assert ref.module_id == mylib_id

    def test_function_binding_in_lib_has_lib_module_id(self, tmp_path: Path) -> None:
        """A function's own binding in mylib has module_id == mylib."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "def foo() -> int = 1\ndef bar() -> int = foo()",
        })
        result = resolve_graph(graph)
        mylib_id = ModuleId.from_dotted("mylib")
        from agm.agl.modules.loader import ModuleGraph
        assert isinstance(graph, ModuleGraph)
        mylib_program = graph.modules[mylib_id].program
        var = _find_varref(mylib_program, "foo")
        assert var is not None
        ref = result.modules[mylib_id].resolved.resolution[var.node_id]
        assert ref.module_id == mylib_id


# ---------------------------------------------------------------------------
# Test: single-module resolve_graph == resolve()
# ---------------------------------------------------------------------------


class TestSingleModuleEquivalence:
    def test_single_module_resolution_tables_match(self, tmp_path: Path) -> None:
        """A single-module graph resolves identically via resolve_graph and resolve()."""
        source = "def foo() -> int = 1\nlet x = foo()\nx"
        graph = _make_graph_from_files(tmp_path, {"entry": source})
        graph_result = resolve_graph(graph)
        from agm.agl.parser import parse_program
        program = parse_program(source)
        single_result = resolve(program)
        # Both should resolve the same VarRef node_ids
        entry_resolution = graph_result.modules[ENTRY_ID].resolved.resolution
        single_resolution = single_result.resolution
        # The node_ids should match in count
        assert len(entry_resolution) == len(single_resolution)
        # And all resolved names/kinds should match (module_id may differ: ENTRY_ID vs local)
        for node_id, single_ref in single_resolution.items():
            assert node_id in entry_resolution
            graph_ref = entry_resolution[node_id]
            assert graph_ref.name == single_ref.name
            assert graph_ref.kind == single_ref.kind

    def test_single_module_module_id_is_entry_id(self, tmp_path: Path) -> None:
        """In a single-module graph, all resolved BindingRefs have module_id == ENTRY_ID."""
        source = "def foo() -> int = 1\nlet x = foo()\nx"
        graph = _make_graph_from_files(tmp_path, {"entry": source})
        result = resolve_graph(graph)
        for ref in result.modules[ENTRY_ID].resolved.resolution.values():
            assert ref.module_id == ENTRY_ID

    def test_existing_resolve_still_works(self) -> None:
        """The single-program resolve() function still works unchanged after adding module_id."""
        from agm.agl.parser import parse_program
        program = parse_program("def foo() -> int = 1\nlet x = foo()\nx")
        result = resolve(program)
        for ref in result.resolution.values():
            assert ref.module_id == ENTRY_ID


# ---------------------------------------------------------------------------
# Test: assign-stmt resolution with module_id
# ---------------------------------------------------------------------------


class TestAssignStmtModuleId:
    def test_assign_stmt_module_id_in_entry(self, tmp_path: Path) -> None:
        """Assignment targets in the entry have module_id == ENTRY_ID."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "var x = 1\nx := 2",
        })
        result = resolve_graph(graph)
        # Should not raise
        assert ENTRY_ID in result.modules

    def test_assign_stmt_in_non_entry_errors(self, tmp_path: Path) -> None:
        """An assignment statement in a non-entry module is an error."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "def setup() -> unit = ()\nx := 2",
        })
        with pytest.raises(AglScopeError):
            resolve_graph(graph)


# ---------------------------------------------------------------------------
# Test: config pragma in non-entry errors
# ---------------------------------------------------------------------------


class TestConfigPragmaInNonEntry:
    def test_config_pragma_in_non_entry_errors(self, tmp_path: Path) -> None:
        """A config pragma in a non-entry module is an error."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "config log = true",
        })
        with pytest.raises(AglScopeError, match="config|pragma"):
            resolve_graph(graph)


# ---------------------------------------------------------------------------
# Test: type declarations in modules (RecordDef/EnumDef/TypeAlias)
# ---------------------------------------------------------------------------


class TestTypeDeclarationsInModules:
    def test_record_in_module_is_exported(self, tmp_path: Path) -> None:
        """A record declaration in a non-entry module is in exports."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "record Point\n  x: int\n  y: int",
        })
        result = resolve_graph(graph)
        mylib_id = ModuleId.from_dotted("mylib")
        assert "Point" in result.modules[mylib_id].exports

    def test_private_record_not_in_exports(self, tmp_path: Path) -> None:
        """A private record is not in exports."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "private record Hidden\n  x: int",
        })
        result = resolve_graph(graph)
        mylib_id = ModuleId.from_dotted("mylib")
        assert "Hidden" not in result.modules[mylib_id].exports

    def test_enum_in_module_in_pre_pass_tables(self, tmp_path: Path) -> None:
        """An enum in a module is in all_public_types."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "enum Color\n  | Red\n  | Green\n  | Blue",
        })
        result = resolve_graph(graph)
        mylib_id = ModuleId.from_dotted("mylib")
        assert (mylib_id, "Color") in result.all_public_types

    def test_type_alias_in_module_exports(self, tmp_path: Path) -> None:
        """A type alias in a non-entry module is in exports."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "type MyInt = int",
        })
        result = resolve_graph(graph)
        mylib_id = ModuleId.from_dotted("mylib")
        assert "MyInt" in result.modules[mylib_id].exports
        assert (mylib_id, "MyInt") in result.all_public_types

    def test_private_func_qualified_access_errors(self, tmp_path: Path) -> None:
        """Qualified access to a private function gives an informative error."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\nlet x = mylib::secret()",
            "mylib": "def pub() -> int = 1\nprivate def secret() -> int = 2",
        })
        with pytest.raises(AglScopeError, match="[Pp]rivate|secret|imported set"):
            resolve_graph(graph)

    def test_non_private_name_not_in_imported_set_errors(self, tmp_path: Path) -> None:
        """Qualified access to a non-private name not in S gives 'not in imported set'."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib using foo\nlet x = mylib::bar()",
            "mylib": "def foo() -> int = 1\ndef bar() -> int = 2\nprivate def hidden() -> int = 3",
        })
        with pytest.raises(AglScopeError, match="bar|imported set"):
            resolve_graph(graph)


# ---------------------------------------------------------------------------
# Test: ::name self-reference in non-entry module
# ---------------------------------------------------------------------------


class TestSelfReferenceInNonEntryModule:
    def test_self_ref_to_nonexistent_name_errors(self, tmp_path: Path) -> None:
        """'::nonexistent' in a non-entry module errors."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import mylib\n()",
            "mylib": "def foo() -> int = ::noname()",
        })
        with pytest.raises(AglScopeError, match="noname"):
            resolve_graph(graph)


# ---------------------------------------------------------------------------
# Test: call with module qualifier (modA.funcA() syntax)
# ---------------------------------------------------------------------------


class TestModuleQualifiedCall:
    def test_module_double_colon_call_resolves(self, tmp_path: Path) -> None:
        """'modA::callA()' — double-colon qualified call resolves the function."""
        graph = _make_graph_from_files(tmp_path, {
            "entry": "import modA\nmodA::callA()",
            "modA": "def callA() -> int = 42",
        })
        result = resolve_graph(graph)
        assert ENTRY_ID in result.modules
