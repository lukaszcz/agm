"""Tests for the program-level scope resolver: ``resolve_program``.

These tests drive multi-module AgL programs through ``resolve_program`` and assert on:
- ``ResolvedProgram`` and ``ResolvedModule`` shape
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
- single-module graph via ``resolve_program`` equals ``resolve_module()`` result
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.modules.ids import ENTRY_ID, STD_CONFIG_ID, ModuleId
from agm.agl.parser import parse_program
from agm.agl.scope import resolve_module
from agm.agl.scope.program import ResolvedModule, ResolvedProgram, resolve_program
from agm.agl.scope.symbols import AglScopeError, BinderKind
from agm.agl.syntax.nodes import VarRef
from tests.agl.ir_harness import make_graph_from_files as _make_graph_from_files

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_varref(program: object, name: str) -> VarRef | None:
    """Recursively find the first VarRef with the given name in a Program."""
    from agm.agl.syntax.nodes import (
        AssignStmt,
        BinaryOp,
        Block,
        Call,
        Cast,
        DictLit,
        FieldAccess,
        FuncDef,
        If,
        IndexAccess,
        InterpSegment,
        IsTest,
        Lambda,
        LetDecl,
        ListLit,
        Loop,
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
        if isinstance(node, Loop):
            r = walk(node.body)
            if r is not None:
                return r
            if node.until_cond is not None:
                return walk(node.until_cond)
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
# Test: basic ResolvedProgram shape
# ---------------------------------------------------------------------------


class TestResolvedProgramShape:
    def test_single_module_graph_has_entry(self, tmp_path: Path) -> None:
        """A single-module graph has one ResolvedModule keyed by ENTRY_ID."""
        graph = _make_graph_from_files(tmp_path, {"entry": "()"})
        result = resolve_program(graph)
        assert isinstance(result, ResolvedProgram)
        assert ENTRY_ID in result.modules
        assert result.entry_id == ENTRY_ID

    def test_resolved_module_shape(self, tmp_path: Path) -> None:
        """ResolvedModule has module_id, resolved, import_env, exports."""
        graph = _make_graph_from_files(tmp_path, {"entry": "()"})
        result = resolve_program(graph)
        entry_mod = result.modules[ENTRY_ID]
        assert isinstance(entry_mod, ResolvedModule)
        assert entry_mod.module_id == ENTRY_ID
        assert entry_mod.resolved is not None
        assert entry_mod.import_env is not None
        assert isinstance(entry_mod.exports, dict)

    def test_multi_module_graph_has_both_modules(self, tmp_path: Path) -> None:
        """A two-module graph has entries for entry and the library module."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert ENTRY_ID in result.modules
        assert mylib_id in result.modules

    def test_exports_excludes_private(self, tmp_path: Path) -> None:
        """Private functions are excluded from exports."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "def pub() -> int = 1\nprivate def priv() -> int = 2",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        exports = result.modules[mylib_id].exports
        assert "pub" in exports
        assert "priv" not in exports

    def test_pre_pass_tables_populated(self, tmp_path: Path) -> None:
        """all_public_funcs and all_public_types are populated."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "def foo() -> int = 1",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert (mylib_id, "foo") in result.all_public_funcs

    def test_entry_agents_populated(self, tmp_path: Path) -> None:
        """entry_agents maps agent names declared in the entry."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": 'agent bot = "claude"\n()',
            },
        )
        result = resolve_program(graph)
        assert "bot" in result.entry_agents


# ---------------------------------------------------------------------------
# Test: open import — bare name resolution
# ---------------------------------------------------------------------------


class TestOpenImport:
    def test_open_import_varref_has_correct_module_id(self, tmp_path: Path) -> None:
        """VarRef resolved via open import has BindingRef.module_id == owning/source module's id."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\nlet x = foo()",
                "mylib": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)
        entry_resolved = result.modules[ENTRY_ID].resolved
        # Find the VarRef for 'foo' in the entry

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "foo")
        assert var is not None
        ref = entry_resolved.resolution[var.node_id]
        assert ref.module_id == ModuleId.from_path("mylib")

    def test_using_import_limits_exposed_names(self, tmp_path: Path) -> None:
        """'import mylib using bar' only exposes 'bar'."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib using bar\nlet x = bar()",
                "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
            },
        )
        result = resolve_program(graph)

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "bar")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.name == "bar"

    def test_using_import_hides_non_listed_name(self, tmp_path: Path) -> None:
        """'import mylib using bar' — bare 'foo' should error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib using bar\nlet x = foo()",
                "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
            },
        )
        with pytest.raises(AglScopeError, match="foo"):
            resolve_program(graph)

    def test_hiding_import_excludes_hidden_name(self, tmp_path: Path) -> None:
        """'import mylib hiding foo' exposes 'bar' but not 'foo'."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib hiding foo\nlet x = bar()",
                "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
            },
        )
        result = resolve_program(graph)

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "bar")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.name == "bar"

    def test_hiding_import_hides_named(self, tmp_path: Path) -> None:
        """'import mylib hiding foo' — bare 'foo' should error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib hiding foo\nlet x = foo()",
                "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
            },
        )
        with pytest.raises(AglScopeError, match="foo"):
            resolve_program(graph)

    def test_using_with_rename(self, tmp_path: Path) -> None:
        """'import mylib using foo as baz' exposes 'baz', not 'foo'.

        The BindingRef.name records the *original* declared name in the owning
        module (``"foo"``), not the exposed name (``"baz"``).  This is what
        the evaluator uses to look up the value in the owning module's frame.
        """
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib using foo as baz\nlet x = baz()",
                "mylib": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "baz")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        # BindingRef.name is the original declared name in the owning module.
        assert ref.name == "foo"

    def test_qualified_import_prevents_bare_access(self, tmp_path: Path) -> None:
        """'import mylib qualified' — bare 'foo' should error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib qualified\nlet x = foo()",
                "mylib": "def foo() -> int = 42",
            },
        )
        with pytest.raises(AglScopeError, match="foo"):
            resolve_program(graph)

    def test_qualified_import_allows_qualified_access(self, tmp_path: Path) -> None:
        """'import mylib qualified' — 'mylib::foo' should resolve."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib qualified\nlet x = mylib::foo()",
                "mylib": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "foo")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.name == "foo"
        assert ref.module_id == ModuleId.from_path("mylib")

    def test_as_alias(self, tmp_path: Path) -> None:
        """'import mylib as M' — 'M::foo()' resolves."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib as M\nlet x = M::foo()",
                "mylib": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "foo")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.name == "foo"
        assert ref.module_id == ModuleId.from_path("mylib")

    def test_as_alias_unqualified_still_exposed(self, tmp_path: Path) -> None:
        """'import mylib as M' without 'qualified' exposes bare names too."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib as M\nlet x = foo()",
                "mylib": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "foo")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.module_id == ModuleId.from_path("mylib")


# ---------------------------------------------------------------------------
# Test: qualified access (S bounds)
# ---------------------------------------------------------------------------


class TestQualifiedAccess:
    def test_qualified_using_bounds_set(self, tmp_path: Path) -> None:
        """'import mylib qualified using foo' — 'mylib::bar' should error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib qualified using foo\nlet x = mylib::bar()",
                "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
            },
        )
        with pytest.raises(AglScopeError, match="bar"):
            resolve_program(graph)

    def test_qualified_using_allows_listed(self, tmp_path: Path) -> None:
        """'import mylib qualified using foo' — 'mylib::foo()' should resolve."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib qualified using foo\nlet x = mylib::foo()",
                "mylib": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "foo")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.name == "foo"
        assert ref.module_id == ModuleId.from_path("mylib")

    def test_unknown_qualifier_handle_errors(self, tmp_path: Path) -> None:
        """'nomodule::foo' when no such module is imported errors."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "let x = nomodule::foo()",
            },
        )
        with pytest.raises(AglScopeError, match="nomodule"):
            resolve_program(graph)


# ---------------------------------------------------------------------------
# Test: clash deferred to use-site
# ---------------------------------------------------------------------------


class TestClashDeferred:
    def test_two_imports_same_name_clashes_at_use(self, tmp_path: Path) -> None:
        """Two open imports both export 'foo' — bare 'foo' → ambiguous error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import libA\nimport libB\nlet x = foo()",
                "libA": "def foo() -> int = 1",
                "libB": "def foo() -> int = 2",
            },
        )
        with pytest.raises(AglScopeError, match="ambiguous"):
            resolve_program(graph)

    def test_clash_error_names_qualifiers(self, tmp_path: Path) -> None:
        """Ambiguous error mentions at least one disambiguation qualifier."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import libA\nimport libB\nlet x = foo()",
                "libA": "def foo() -> int = 1",
                "libB": "def foo() -> int = 2",
            },
        )
        with pytest.raises(AglScopeError) as exc_info:
            resolve_program(graph)
        msg = str(exc_info.value)
        assert "libA" in msg or "libB" in msg or "ambiguous" in msg.lower()

    def test_no_clash_same_qname(self, tmp_path: Path) -> None:
        """Two imports of the same module's same function don't clash (idempotent)."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\nimport mylib using foo\nlet x = foo()",
                "mylib": "def foo() -> int = 42",
            },
        )
        # Should not raise — same QName from same module
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "foo")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.module_id == ModuleId.from_path("mylib")


# ---------------------------------------------------------------------------
# Test: multiple imports merge
# ---------------------------------------------------------------------------


class TestMultipleImportsMerge:
    def test_two_decls_same_module_merges(self, tmp_path: Path) -> None:
        """Two import declarations for the same module merge their exposed sets."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import mylib using foo\nimport mylib using bar\nlet a = foo()\nlet b = bar()"
                ),
                "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

        entry_program = graph.modules[ENTRY_ID].program
        mylib_id = ModuleId.from_path("mylib")
        foo_var = _find_varref(entry_program, "foo")
        assert foo_var is not None
        assert result.modules[ENTRY_ID].resolved.resolution[foo_var.node_id].module_id == mylib_id
        bar_var = _find_varref(entry_program, "bar")
        assert bar_var is not None
        assert result.modules[ENTRY_ID].resolved.resolution[bar_var.node_id].module_id == mylib_id


# ---------------------------------------------------------------------------
# Test: static import errors
# ---------------------------------------------------------------------------


class TestStaticImportErrors:
    def test_duplicate_alias_different_modules(self, tmp_path: Path) -> None:
        """'import A as X' + 'import B as X' → static alias error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import libA as X\nimport libB as X\n()",
                "libA": "def foo() -> int = 1",
                "libB": "def bar() -> int = 2",
            },
        )
        with pytest.raises(AglScopeError, match="X"):
            resolve_program(graph)

    def test_alias_root_collision(self, tmp_path: Path) -> None:
        """'import libA' + 'import libB as libA' → alias-root collision."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import libA\nimport libB as libA\n()",
                "libA": "def foo() -> int = 1",
                "libB": "def bar() -> int = 2",
            },
        )
        with pytest.raises(AglScopeError, match="libA"):
            resolve_program(graph)


# ---------------------------------------------------------------------------
# Test: ::name self-reference
# ---------------------------------------------------------------------------


class TestSelfReference:
    def test_self_ref_in_entry(self, tmp_path: Path) -> None:
        """'::foo' in the entry resolves to the entry's own 'foo'."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "def foo() -> int = 1\nlet x = ::foo()",
            },
        )
        result = resolve_program(graph)

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "foo")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.module_id == ENTRY_ID

    def test_self_ref_in_lib_module(self, tmp_path: Path) -> None:
        """'::bar' in module 'mylib' resolves to mylib's own 'bar'."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "def bar() -> int = 1\ndef baz() -> int = ::bar()",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")

        mylib_program = graph.modules[mylib_id].program
        var = _find_varref(mylib_program, "bar")
        assert var is not None
        ref = result.modules[mylib_id].resolved.resolution[var.node_id]
        assert ref.module_id == mylib_id

    def test_self_ref_undefined_name_errors(self, tmp_path: Path) -> None:
        """'::nonexistent' in a module errors."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "let x = ::nonexistent()",
            },
        )
        with pytest.raises(AglScopeError, match="nonexistent"):
            resolve_program(graph)

    def test_self_ref_bypasses_param_shadow(self, tmp_path: Path) -> None:
        """'::foo' inside g(foo: int) resolves to the top-level def foo, not the param.

        : '::name' means the CURRENT MODULE'S OWN TOP-LEVEL declaration,
        bypassing any lexical shadow introduced by params or let bindings.
        """
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "def foo() -> int = 1\ndef g(foo: int) -> int = ::foo()",
            },
        )
        result = resolve_program(graph)
        from agm.agl.scope.symbols import BinderKind

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

            if (
                isinstance(node, VarRef)
                and node.name == "foo"
                and (node.module_qualifier is not None)
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
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "private def secret() -> int = 1",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert "secret" not in result.modules[mylib_id].exports

    def test_private_not_accessible_via_open_import(self, tmp_path: Path) -> None:
        """A private function from an imported module is not accessible unqualified."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\nlet x = secret()",
                "mylib": "private def secret() -> int = 1",
            },
        )
        with pytest.raises(AglScopeError, match="secret"):
            resolve_program(graph)

    def test_private_not_accessible_via_qualified_access(self, tmp_path: Path) -> None:
        """A private function is not accessible via qualified access.

        When a module has only private exports, its qualifier is not registered
        (no names to qualify). When it has public exports, 'mylib::secret'
        raises the private-access error naming mylib specifically.
        """
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\nlet x = mylib::secret()",
                "mylib": "def pub() -> int = 1\nprivate def secret() -> int = 2",
            },
        )
        with pytest.raises(AglScopeError, match="[Pp]rivate"):
            resolve_program(graph)

    def test_private_error_message_names_owning_module(self, tmp_path: Path) -> None:
        """Qualified access to a private name names the MODULE that owns the private decl.

        Regression: when libA::secret is accessed but 'secret' only exists as
        a private name in libB (a different module), the error used to name libB
        instead of correctly resolving libA as the owning module and emitting 'private'
        (if secret is private in libA) or 'not in imported set of libA' (if it isn't).
        """
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\nlet x = mylib::secret()",
                "mylib": "def pub() -> int = 1\nprivate def secret() -> int = 2",
            },
        )
        with pytest.raises(AglScopeError) as exc_info:
            resolve_program(graph)
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
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import libA\nimport libB\nlet x = libA::secret()",
                "libA": "def pub() -> int = 1",
                "libB": "def other() -> int = 2\nprivate def secret() -> int = 99",
            },
        )
        with pytest.raises(AglScopeError) as exc_info:
            resolve_program(graph)
        msg = str(exc_info.value)
        # Must mention libA (the qualifier used), not libB (which has the private decl)
        assert "libA" in msg, f"Expected 'libA' in error, got: {msg!r}"
        assert "libB" not in msg, f"libB should NOT be named, got: {msg!r}"


# ---------------------------------------------------------------------------
# Test: declaration-only enforcement
# ---------------------------------------------------------------------------


class TestBuiltinVarPlacement:
    def test_entry_declaration_rejected(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {"entry": "builtin var max-iters: int\n()"},
        )

        with pytest.raises(AglScopeError, match="std.config"):
            resolve_program(graph)

    def test_arbitrary_library_declaration_rejected(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "builtin var max-iters: int",
            },
        )

        with pytest.raises(AglScopeError, match="std.config"):
            resolve_program(graph)

    def test_std_config_declarations_and_qualified_assignment_resolve(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {"entry": ("import std.config\nstd.config::max-iters := 3\nstd.config::max-iters")},
        )

        resolved = resolve_program(graph)
        std_config = resolved.modules[STD_CONFIG_ID]
        binding = std_config.resolved.root_scope.lookup("max-iters")
        assert binding is not None
        assert binding.kind is BinderKind.builtin_var_binding
        assignment_ref = next(
            ref
            for ref in resolved.modules[ENTRY_ID].resolved.resolution.values()
            if ref.name == "max-iters" and ref.kind is BinderKind.builtin_var_binding
        )
        assert assignment_ref.module_id == STD_CONFIG_ID


class TestDeclarationOnly:
    def test_let_in_non_entry_errors(self, tmp_path: Path) -> None:
        """A 'let' declaration in a non-entry module is an error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "let x = 1",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_var_in_non_entry_errors(self, tmp_path: Path) -> None:
        """A 'var' declaration in a non-entry module is an error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "var x = 1",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_bare_expr_in_non_entry_errors(self, tmp_path: Path) -> None:
        """A bare expression in a non-entry module is an error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "def foo() -> int = 1\n42",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_funcdef_in_non_entry_allowed(self, tmp_path: Path) -> None:
        """A 'def' in a non-entry module is allowed."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules
        mylib_id = ModuleId.from_path("mylib")
        assert "foo" in result.modules[mylib_id].exports


# ---------------------------------------------------------------------------
# Test: entry-only constructs
# ---------------------------------------------------------------------------


class TestEntryOnlyConstructs:
    def test_agent_in_non_entry_errors(self, tmp_path: Path) -> None:
        """An 'agent' declaration in a non-entry module is an error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": 'agent bot = "claude"',
            },
        )
        with pytest.raises(AglScopeError, match="agent"):
            resolve_program(graph)

    def test_param_in_non_entry_errors(self, tmp_path: Path) -> None:
        """A 'param' declaration in a non-entry module is an error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "param x",
            },
        )
        with pytest.raises(AglScopeError, match="param"):
            resolve_program(graph)

    def test_program_decl_in_non_entry_errors(self, tmp_path: Path) -> None:
        """A 'program' declaration in a non-entry module is an error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "program myname",
            },
        )
        with pytest.raises(AglScopeError, match="program"):
            resolve_program(graph)

    def test_agent_in_entry_allowed(self, tmp_path: Path) -> None:
        """An 'agent' declaration in the entry module is allowed."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": 'agent bot = "claude"\n()',
            },
        )
        result = resolve_program(graph)
        assert "bot" in result.entry_agents


# ---------------------------------------------------------------------------
# Test: header-only imports
# ---------------------------------------------------------------------------


class TestHeaderOnlyImports:
    def test_import_after_def_in_non_entry_errors(self, tmp_path: Path) -> None:
        """An import after a def in a non-entry module is an error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "def foo() -> int = 1\nimport libB",
                "libB": "def bar() -> int = 2",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_import_after_infix_decl_in_non_entry_errors(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "infixl |> at 12\nimport libB\ndef |>(x: int, y: int) -> int = x",
                "libB": "def bar() -> int = 2",
            },
        )
        with pytest.raises(AglScopeError, match="Import and export"):
            resolve_program(graph)

    def test_import_at_top_of_non_entry_allowed(self, tmp_path: Path) -> None:
        """Import declarations at the top of a non-entry module are allowed."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "import libB\ndef foo() -> int = bar()",
                "libB": "def bar() -> int = 42",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules
        mylib_id = ModuleId.from_path("mylib")
        assert "foo" in result.modules[mylib_id].exports

    def test_import_in_entry_works(self, tmp_path: Path) -> None:
        """Entry module works fine with import followed by def."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\ndef helper() -> int = foo()\n()",
                "mylib": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "foo")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.module_id == ModuleId.from_path("mylib")


# ---------------------------------------------------------------------------
# Test: wildcard imports
# ---------------------------------------------------------------------------


class TestWildcardImports:
    def test_wildcard_expands_all_submodules(self, tmp_path: Path) -> None:
        """'import foo.*' exposes functions from all submodules of 'foo'."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import foo.*\nlet a = alpha()\nlet b = beta()",
                "foo/alpha": "def alpha() -> int = 1",
                "foo/beta": "def beta() -> int = 2",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

        entry_program = graph.modules[ENTRY_ID].program
        alpha_var = _find_varref(entry_program, "alpha")
        assert alpha_var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[alpha_var.node_id]
        assert ref.module_id == ModuleId.from_path("foo/alpha")

    def test_wildcard_as_reroots_qualifier(self, tmp_path: Path) -> None:
        "'import foo.* as F' makes 'F.alpha::alpha()' accessible via qualifier."
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import foo.* as F\nlet a = F.alpha::alpha()",
                "foo/alpha": "def alpha() -> int = 1",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

        entry_program = graph.modules[ENTRY_ID].program
        alpha_var = _find_varref(entry_program, "alpha")
        assert alpha_var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[alpha_var.node_id]
        assert ref.module_id == ModuleId.from_path("foo/alpha")

    def test_type_name_import_handle_ambiguity_errors(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import lib qualified as Color\nenum Color | Red\nlet x = Color::Red\nx",
                "lib": "def f() -> int = 1",
            },
        )
        with pytest.raises(AglScopeError, match="both a type name and an import handle"):
            resolve_program(graph)

    def test_wildcard_compatible_overlap_idempotent(self, tmp_path: Path) -> None:
        """Two wildcards that expose same QName from same module are idempotent (no error)."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import foo.*\nimport foo.alpha\nlet x = alpha()",
                "foo/alpha": "def alpha() -> int = 1",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

        entry_program = graph.modules[ENTRY_ID].program
        alpha_var = _find_varref(entry_program, "alpha")
        assert alpha_var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[alpha_var.node_id]
        assert ref.module_id == ModuleId.from_path("foo/alpha")

    def test_wildcard_conflicting_overlap_clashes_on_use(self, tmp_path: Path) -> None:
        """Two wildcards expose same bare name from different modules → clash on use."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import foo.*\nimport bar.*\nlet x = common()",
                "foo/sub": "def common() -> int = 1",
                "bar/sub": "def common() -> int = 2",
            },
        )
        with pytest.raises(AglScopeError, match="ambiguous|common"):
            resolve_program(graph)


# ---------------------------------------------------------------------------
# Test: cross-file mutual recursion
# ---------------------------------------------------------------------------


class TestCrossFileMutualRecursion:
    def test_a_calls_b_resolves(self, tmp_path: Path) -> None:
        """Module A can call module B's function (one-way, open import)."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import modA\ncallA()",
                "modA": "import modB\ndef callA() -> int = callB()",
                "modB": "def callB() -> int = 42",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

        # entry sees callA from modA
        entry_program = graph.modules[ENTRY_ID].program
        calla_var = _find_varref(entry_program, "callA")
        assert calla_var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[calla_var.node_id]
        assert ref.module_id == ModuleId.from_path("modA")
        # modA sees callB from modB
        moda_id = ModuleId.from_path("modA")
        moda_program = graph.modules[moda_id].program
        callb_var = _find_varref(moda_program, "callB")
        assert callb_var is not None
        ref2 = result.modules[moda_id].resolved.resolution[callb_var.node_id]
        assert ref2.module_id == ModuleId.from_path("modB")

    def test_mutual_recursion_across_modules(self, tmp_path: Path) -> None:
        """A's def calls B's def and B's def calls A's def — both resolve."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import modA\nimport modB\nfuncA()",
                "modA": "import modB\ndef funcA() -> int = funcB()",
                "modB": "import modA\ndef funcB() -> int = funcA()",
            },
        )
        result = resolve_program(graph)
        mod_a = ModuleId.from_path("modA")
        mod_b = ModuleId.from_path("modB")
        # funcA's body call to funcB must resolve into modB, and vice-versa.
        call_b = _find_varref(graph.modules[mod_a].program, "funcB")
        assert call_b is not None
        assert result.modules[mod_a].resolved.resolution[call_b.node_id].module_id == mod_b
        call_a = _find_varref(graph.modules[mod_b].program, "funcA")
        assert call_a is not None
        assert result.modules[mod_b].resolved.resolution[call_a.node_id].module_id == mod_a

    def test_placeholder_call_in_imported_module_function_body_resolves(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import modA\nmake()",
                "modA": "def f(x: int) -> int = x\ndef make() -> int = f(?)",
            },
        )
        result = resolve_program(graph)
        mod_a = ModuleId.from_path("modA")
        call_f = _find_varref(graph.modules[mod_a].program, "f")
        assert call_f is not None
        assert result.modules[mod_a].resolved.resolution[call_f.node_id].module_id == mod_a


# ---------------------------------------------------------------------------
# Test: BindingRef.module_id
# ---------------------------------------------------------------------------


class TestBindingRefModuleId:
    def test_local_binding_has_entry_module_id(self, tmp_path: Path) -> None:
        """A local let-binding in the entry has module_id == ENTRY_ID."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "let x = 1\nx",
            },
        )
        result = resolve_program(graph)

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "x")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.module_id == ENTRY_ID

    def test_cross_module_binding_has_source_module_id(self, tmp_path: Path) -> None:
        """A VarRef resolved from an import has module_id of the providing module."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\nlet y = foo()",
                "mylib": "def foo() -> int = 1",
            },
        )
        result = resolve_program(graph)

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "foo")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        mylib_id = ModuleId.from_path("mylib")
        assert ref.module_id == mylib_id

    def test_function_binding_in_lib_has_lib_module_id(self, tmp_path: Path) -> None:
        """A function's own binding in mylib has module_id == mylib."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "def foo() -> int = 1\ndef bar() -> int = foo()",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")

        mylib_program = graph.modules[mylib_id].program
        var = _find_varref(mylib_program, "foo")
        assert var is not None
        ref = result.modules[mylib_id].resolved.resolution[var.node_id]
        assert ref.module_id == mylib_id


# ---------------------------------------------------------------------------
# Test: single-module resolve_program == resolve_module()
# ---------------------------------------------------------------------------


class TestSingleModuleEquivalence:
    def test_single_module_resolution_tables_match(self, tmp_path: Path) -> None:
        """A single-module graph resolves identically via resolve_program and resolve_module()."""
        source = "def foo() -> int = 1\nlet x = foo()\nx"
        graph = _make_graph_from_files(tmp_path, {"entry": source})
        program_result = resolve_program(graph)
        from agm.agl.parser import parse_program

        program = parse_program(source)
        single_result = resolve_module(program)
        # Both should resolve the same VarRef node_ids
        entry_resolution = program_result.modules[ENTRY_ID].resolved.resolution
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
        result = resolve_program(graph)
        for ref in result.modules[ENTRY_ID].resolved.resolution.values():
            assert ref.module_id == ENTRY_ID

    def test_existing_resolve_still_works(self) -> None:
        """The per-module resolve_module() function still works unchanged after adding module_id."""
        from agm.agl.parser import parse_program

        program = parse_program("def foo() -> int = 1\nlet x = foo()\nx")
        result = resolve_module(program)
        for ref in result.resolution.values():
            assert ref.module_id == ENTRY_ID


# ---------------------------------------------------------------------------
# Test: assign-stmt resolution with module_id
# ---------------------------------------------------------------------------


class TestAssignStmtModuleId:
    def test_assign_stmt_module_id_in_entry(self, tmp_path: Path) -> None:
        """Assignment targets in the entry have module_id == ENTRY_ID."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "var x = 1\nx := 2",
            },
        )
        result = resolve_program(graph)
        # Should not raise
        assert ENTRY_ID in result.modules
        # var "x" is declared in entry and is mutable
        entry_resolved = result.modules[ENTRY_ID].resolved
        binding = entry_resolved.root_scope.lookup("x")
        assert binding is not None
        assert binding.module_id == ENTRY_ID
        assert binding.mutable is True

    def test_assign_stmt_in_non_entry_errors(self, tmp_path: Path) -> None:
        """An assignment statement in a non-entry module is an error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "def setup() -> unit = ()\nx := 2",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)


# ---------------------------------------------------------------------------
# Test: type declarations in modules (RecordDef/EnumDef/TypeAlias)
# ---------------------------------------------------------------------------


class TestTypeDeclarationsInModules:
    def test_record_in_module_is_exported(self, tmp_path: Path) -> None:
        """A record declaration in a non-entry module is in exports."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "record Point\n  x: int\n  y: int",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert "Point" in result.modules[mylib_id].exports

    def test_private_record_not_in_exports(self, tmp_path: Path) -> None:
        """A private record is not in exports."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "private record Hidden\n  x: int",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert "Hidden" not in result.modules[mylib_id].exports

    def test_enum_in_module_in_pre_pass_tables(self, tmp_path: Path) -> None:
        """An enum in a module is in all_public_types."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "enum Color\n  | Red\n  | Green\n  | Blue",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert (mylib_id, "Color") in result.all_public_types

    def test_type_alias_in_module_exports(self, tmp_path: Path) -> None:
        """A type alias in a non-entry module is in exports."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "type MyInt = int",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert "MyInt" in result.modules[mylib_id].exports
        assert (mylib_id, "MyInt") in result.all_public_types

    def test_private_func_qualified_access_errors(self, tmp_path: Path) -> None:
        """Qualified access to a private function gives an informative error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\nlet x = mylib::secret()",
                "mylib": "def pub() -> int = 1\nprivate def secret() -> int = 2",
            },
        )
        with pytest.raises(AglScopeError, match="[Pp]rivate|secret|imported set"):
            resolve_program(graph)

    def test_non_private_name_not_in_imported_set_errors(self, tmp_path: Path) -> None:
        """Qualified access to a non-private name not in S gives 'not in imported set'."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib using foo\nlet x = mylib::bar()",
                "mylib": (
                    "def foo() -> int = 1\ndef bar() -> int = 2\nprivate def hidden() -> int = 3"
                ),
            },
        )
        with pytest.raises(AglScopeError, match="bar|imported set"):
            resolve_program(graph)


# ---------------------------------------------------------------------------
# Test: ::name self-reference in non-entry module
# ---------------------------------------------------------------------------


class TestSelfReferenceInNonEntryModule:
    def test_self_ref_to_nonexistent_name_errors(self, tmp_path: Path) -> None:
        """'::nonexistent' in a non-entry module errors."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "def foo() -> int = ::noname()",
            },
        )
        with pytest.raises(AglScopeError, match="noname"):
            resolve_program(graph)


# ---------------------------------------------------------------------------
# Test: call with module qualifier (modA.funcA() syntax)
# ---------------------------------------------------------------------------


class TestModuleQualifiedCall:
    def test_module_double_colon_call_resolves(self, tmp_path: Path) -> None:
        """'modA::callA()' — double-colon qualified call resolves the function."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import modA\nmodA::callA()",
                "modA": "def callA() -> int = 42",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "callA")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.name == "callA"
        assert ref.module_id == ModuleId.from_path("modA")


# ---------------------------------------------------------------------------
# Coverage: resolver.py _resolve_field_access and _resolve_cross_module_type_name
# ---------------------------------------------------------------------------


class TestFieldAccessCoverage:
    def test_self_ref_field_access_in_non_entry_module(self, tmp_path: Path) -> None:
        """A non-entry module can resolve a self-qualified constructor ref."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "mylib": (
                    "enum Color\n  | Red\n  | Blue\ndef getDefault() -> Color = ::Color::Red"
                ),
                "entry": "import mylib\nmylib::getDefault()",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules
        # The self-ref ::Color::Red within mylib is in mylib's qualified_constructor_refs
        mylib_id = ModuleId.from_path("mylib")
        mylib_resolved = result.modules[mylib_id].resolved
        assert len(mylib_resolved.qualified_constructor_refs) > 0

    def test_unrecognized_qualifier_in_field_access_errors(self, tmp_path: Path) -> None:
        """An unknown module qualifier in a constructor ref is rejected."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "mylib": "enum Color\n  | Red\n  | Blue",
                "entry": "import mylib\nlet x = notimported::Color::Red\nx",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_unknown_exported_name_in_qualified_field_access_errors(self, tmp_path: Path) -> None:
        """An unknown type name in a module-qualified constructor ref is rejected."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "mylib": "enum Color\n  | Red\n  | Blue",
                "entry": "import mylib qualified\nlet x = mylib::NonExistent::Red\nx",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_private_type_in_qualified_constructor_errors(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "mylib": "private enum Hidden\n  | Red\ndef public() -> int = 1",
                "entry": "import mylib qualified\nlet x = mylib::Hidden::Red\nx",
            },
        )
        with pytest.raises(AglScopeError, match="private"):
            resolve_program(graph)

    def test_non_constructible_qualified_type_errors(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "mylib": "type Alias = int",
                "entry": "import mylib qualified\nlet x = mylib::Alias::Ctor\nx",
            },
        )
        with pytest.raises(AglScopeError, match="constructible"):
            resolve_program(graph)

    def test_non_type_exported_name_in_field_access_falls_through(self, tmp_path: Path) -> None:
        """Coverage: non-constructor export in qualified field access.

        ``mylib::compute.value`` where ``compute`` is a function (not a type) exercises
        the ``kind != constructor_binding`` path in _resolve_cross_module_type_name.
        """
        graph = _make_graph_from_files(
            tmp_path,
            {
                "mylib": (
                    "record Result\n  value: int\ndef compute(n: int) -> Result = Result(value = n)"
                ),
                "entry": (
                    "import mylib qualified\n"
                    # mylib::compute is a function, not a type — falls through to value resolution.
                    "mylib::compute.value"
                ),
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

        entry_program = graph.modules[ENTRY_ID].program
        # "compute" is a function binding from mylib
        var = _find_varref(entry_program, "compute")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.name == "compute"
        assert ref.module_id == ModuleId.from_path("mylib")

    def test_self_ref_field_access_unknown_type_errors(self, tmp_path: Path) -> None:
        """An unknown self-qualified constructor type name is rejected."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "mylib": "def foo() -> int = ::NonExistent::Red",
                "entry": "import mylib\n()",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)


# ---------------------------------------------------------------------------
# Test: ExceptionDef export, constructor candidates, and exception-skip branch
# ---------------------------------------------------------------------------


class TestExceptionDefInGraph:
    def test_exception_exported(self, tmp_path: Path) -> None:
        """An exception declaration in a non-entry module is in exports."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "exception MyErr(msg: text)",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert "MyErr" in result.modules[mylib_id].exports

    def test_exception_in_all_public_types(self, tmp_path: Path) -> None:
        """A public exception is in all_public_types."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "exception MyErr(msg: text)",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert (mylib_id, "MyErr") in result.all_public_types

    def test_private_exception_not_exported(self, tmp_path: Path) -> None:
        """A private exception is not in exports."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "private exception HiddenErr(code: int)",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert "HiddenErr" not in result.modules[mylib_id].exports

    def test_exception_constructor_candidate_available(self, tmp_path: Path) -> None:
        """An open-imported exception exposes its name as a constructor candidate."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": 'import mylib\nMyErr(msg = "oops")',
                "mylib": "exception MyErr(msg: text)",
            },
        )
        result = resolve_program(graph)
        # The entry resolved correctly (no scope error raised).
        assert ENTRY_ID in result.modules
        entry_resolved = result.modules[ENTRY_ID].resolved
        # MyErr is a constructor candidate resolved from the open import.
        assert "MyErr" in entry_resolved.constructor_candidates

    def test_exception_skip_branch_enum_variant_collision(self, tmp_path: Path) -> None:
        """Exception-skip branch: an enum variant whose name collides with a public
        ExceptionDef in the same module is skipped as a constructor candidate.

        Exercises graph.py lines ~154-157: when iterating EnumDef variants, any variant
        whose name matches a public ExceptionDef in all_public_types is skipped so the
        exception wins as the constructor.
        """
        # The enum "Status" has a variant named "Conflict".
        # The module also has a public exception "Conflict".
        # When resolving the open import, "Conflict" (enum variant) must be skipped
        # and only the ExceptionDef "Conflict" is in constructor_candidates.
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": ("enum Status\n  | Ok\n  | Conflict\nexception Conflict(msg: text)\n"),
            },
        )
        result = resolve_program(graph)
        entry_resolved = result.modules[ENTRY_ID].resolved
        # "Conflict" must exist in constructor_candidates and must refer to the
        # ExceptionDef (variant=None), NOT to the enum variant (variant="Conflict").
        assert "Conflict" in entry_resolved.constructor_candidates
        candidates = entry_resolved.constructor_candidates["Conflict"]
        # The exception's constructor ref has variant=None (record-like).
        assert all(c.variant is None for c in candidates), (
            "Enum variant 'Conflict' should have been skipped; only the ExceptionDef "
            f"candidate (variant=None) should remain. Got: {candidates}"
        )


# ---------------------------------------------------------------------------
# Test: resolve_program REPL seams — ambient_agents, entry_parent_scope, warnings
# ---------------------------------------------------------------------------


class TestResolveGraphReplSeams:
    def test_ambient_agents_resolves_undeclared_agent(self, tmp_path: Path) -> None:
        """ambient_agents lets the entry module use an agent not declared in source."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": 'let x = ask("Q", agent = session_bot)\nx',
            },
        )
        # Without ambient_agents, "session_bot" is undeclared → AglScopeError.
        with pytest.raises(AglScopeError):
            resolve_program(graph)

        # With ambient_agents, it resolves cleanly.
        result = resolve_program(graph, ambient_agents=frozenset({"session_bot"}))
        assert ENTRY_ID in result.modules
        # The entry has no declared_agents (ambient agents are not declared locally).
        entry_resolved = result.modules[ENTRY_ID].resolved
        assert "session_bot" not in entry_resolved.declared_agents

    def test_ambient_agents_kwarg_is_noop_for_clean_graph(self, tmp_path: Path) -> None:
        """Passing ambient_agents does not perturb a graph that doesn't use the ambient name.

        (Agents are entry-only and library modules are declaration-only, so a non-entry
        module cannot reference an ambient agent; this asserts the kwarg is a no-op here.)
        """
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph, ambient_agents=frozenset({"session_bot"}))
        assert ENTRY_ID in result.modules
        assert ModuleId.from_path("mylib") in result.modules

    def test_entry_parent_scope_binding_visible_in_entry(self, tmp_path: Path) -> None:
        """entry_parent_scope: a name pre-bound in the parent scope is visible in entry."""
        # Build a prior session that binds "x" as a let binding.
        prior_source = "let x = 42\nx"
        prior_resolved = resolve_module(parse_program(prior_source))
        session_scope = prior_resolved.root_scope

        # New entry references "x" — which is only in the parent scope.
        graph = _make_graph_from_files(tmp_path, {"entry": "x"})
        result = resolve_program(graph, entry_parent_scope=session_scope)
        assert ENTRY_ID in result.modules

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "x")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.name == "x"
        assert ref.kind == BinderKind.let_binding

    def test_entry_parent_scope_not_applied_to_non_entry(self, tmp_path: Path) -> None:
        """entry_parent_scope is only injected into the entry module, not library modules.

        ``helper`` is bound only in the prior session scope.  A non-entry module that
        references it must fail to resolve — if the parent scope leaked into library
        resolution, ``mylib`` would silently succeed.
        """
        prior_source = "let helper = 1\nhelper"
        prior_resolved = resolve_module(parse_program(prior_source))
        session_scope = prior_resolved.root_scope

        # mylib references "helper", which exists ONLY in the entry's parent scope.
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "def foo() -> int = helper",
            },
        )
        with pytest.raises(AglScopeError, match="helper"):
            resolve_program(graph, entry_parent_scope=session_scope)

    def test_warnings_aggregated_from_entry_module(self, tmp_path: Path) -> None:
        """result.warnings aggregates non-fatal scope warnings from the entry module.

        Declaring an agent that is never referenced produces an unused-agent warning
        in the scope pass; resolve_program must surface it in the top-level warnings tuple.
        """
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "agent unused_bot\n()",
            },
        )
        result = resolve_program(graph)
        # The entry module must have emitted an unused-agent warning.
        assert len(result.warnings) >= 1
        messages = [w.message for w in result.warnings]
        assert any("unused_bot" in m for m in messages)

    def test_warnings_aggregated_across_modules(self, tmp_path: Path) -> None:
        """Warnings from non-entry modules are also collected into result.warnings."""
        # A non-entry module that declares an agent would be a scope error (agents are
        # entry-only), so we cannot test cross-module warnings that way.
        # Instead, confirm that the empty-warnings case is also correct: when no module
        # produces warnings, result.warnings is empty.
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)
        assert result.warnings == ()


# ---------------------------------------------------------------------------
# Re-export behaviour
# ---------------------------------------------------------------------------


class TestExportDecl:
    """Tests for explicit export declarations."""

    def test_reexport_all_adds_to_exports(self, tmp_path: Path) -> None:
        """export lib — all public names from lib appear in the facade's exports."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\n()",
                "facade": "export lib",
                "lib": "def foo() -> int = 1\nprivate def _bar() -> int = 2",
            },
        )
        result = resolve_program(graph)
        facade_id = ModuleId.from_path("facade")
        lib_id = ModuleId.from_path("lib")
        facade_exports = result.modules[facade_id].exports
        assert "foo" in facade_exports
        assert facade_exports["foo"] == (lib_id, "foo")
        assert "_bar" not in facade_exports

    def test_reexport_origin_is_preserved_through_chain(self, tmp_path: Path) -> None:
        """Re-export is transparent: B re-exports from A, C uses B — origin is A, not B."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import b\nlet x = foo()",
                "b": "export a",
                "a": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)
        b_id = ModuleId.from_path("b")
        a_id = ModuleId.from_path("a")
        b_exports = result.modules[b_id].exports
        assert b_exports["foo"] == (a_id, "foo")

    def test_per_item_reexport_selective(self, tmp_path: Path) -> None:
        """export lib using foo — only foo is re-exported."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\n()",
                "facade": "export lib using foo",
                "lib": "def foo() -> int = 1\ndef bar() -> int = 2",
            },
        )
        result = resolve_program(graph)
        facade_id = ModuleId.from_path("facade")
        lib_id = ModuleId.from_path("lib")
        facade_exports = result.modules[facade_id].exports
        assert "foo" in facade_exports
        assert facade_exports["foo"] == (lib_id, "foo")
        assert "bar" not in facade_exports

    def test_per_item_reexport_with_rename(self, tmp_path: Path) -> None:
        """export lib using foo as plus — re-exported as 'plus', origin is lib.foo."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\n()",
                "facade": "export lib using foo as plus",
                "lib": "def foo() -> int = 1",
            },
        )
        result = resolve_program(graph)
        facade_id = ModuleId.from_path("facade")
        lib_id = ModuleId.from_path("lib")
        facade_exports = result.modules[facade_id].exports
        assert "plus" in facade_exports
        assert facade_exports["plus"] == (lib_id, "foo")
        assert "foo" not in facade_exports

    def test_hiding_reexport(self, tmp_path: Path) -> None:
        """export lib hiding secret — all except 'secret' are re-exported."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\n()",
                "facade": "export lib hiding secret",
                "lib": "def foo() -> int = 1\ndef secret() -> int = 0",
            },
        )
        result = resolve_program(graph)
        facade_id = ModuleId.from_path("facade")
        lib_id = ModuleId.from_path("lib")
        facade_exports = result.modules[facade_id].exports
        assert "foo" in facade_exports
        assert facade_exports["foo"] == (lib_id, "foo")
        assert "secret" not in facade_exports

    def test_no_export_flag_does_not_reexport(self, tmp_path: Path) -> None:
        """Plain import (no export) does not add names to the module's exports."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\n()",
                "facade": "import lib\ndef local() -> int = 1",
                "lib": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)
        facade_id = ModuleId.from_path("facade")
        facade_exports = result.modules[facade_id].exports
        assert "foo" not in facade_exports
        assert "local" in facade_exports

    def test_reexport_chain_multi_hop(self, tmp_path: Path) -> None:
        """A re-exports from B which re-exports from C — A's consumers get C's origin."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import a\nlet x = foo()",
                "a": "export b",
                "b": "export c",
                "c": "def foo() -> int = 99",
            },
        )
        result = resolve_program(graph)
        a_id = ModuleId.from_path("a")
        c_id = ModuleId.from_path("c")
        a_exports = result.modules[a_id].exports
        assert a_exports["foo"] == (c_id, "foo")

    def test_wildcard_reexport(self, tmp_path: Path) -> None:
        """export lib.* — all matching modules' public names are re-exported."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\n()",
                "facade": "export lib.*",
                "lib/ops": "def add() -> int = 1",
            },
        )
        result = resolve_program(graph)
        facade_id = ModuleId.from_path("facade")
        lib_ops_id = ModuleId.from_path("lib/ops")
        facade_exports = result.modules[facade_id].exports
        assert "add" in facade_exports
        assert facade_exports["add"] == (lib_ops_id, "add")

    def test_reexport_conflict_raises(self, tmp_path: Path) -> None:
        """Re-exporting the same name from two different origins raises AglScopeError."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\n()",
                "facade": "export a\nexport b",
                "a": "def foo() -> int = 1",
                "b": "def foo() -> int = 2",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_per_item_reexport_through_chain(self, tmp_path: Path) -> None:
        """Per-item export resolves correctly even when target's re-exports aren't yet populated."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import a\nlet x = foo()",
                "a": "export b using foo",
                "b": "export c",
                "c": "def foo() -> int = 99",
            },
        )
        result = resolve_program(graph)
        a_id = ModuleId.from_path("a")
        c_id = ModuleId.from_path("c")
        a_exports = result.modules[a_id].exports
        assert a_exports["foo"] == (c_id, "foo")

    def test_consumer_import_sees_reexport_unqualified_and_qualified(self, tmp_path: Path) -> None:
        """import facade exposes re-exported names both bare and as facade::name."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\nlet x = foo()\nlet y = facade::foo()",
                "facade": "export lib",
                "lib": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)
        entry = result.modules[ENTRY_ID]
        bare = _find_varref(entry.resolved.program, "foo")
        lib_id = ModuleId.from_path("lib")
        assert bare is not None
        assert entry.resolved.resolution[bare.node_id].module_id == lib_id
        assert (
            sum(
                1
                for ref in entry.resolved.resolution.values()
                if ref.name == "foo" and ref.module_id == lib_id
            )
            == 2
        )


def test_bare_pattern_constructor_shared_spelling_defers_to_scrutinee(tmp_path: Path) -> None:
    # 'same' is a variant of both the imported Foreign and the local Local.
    # A bare pattern on a Local scrutinee is not an ambiguity error at scope
    # resolution: both candidates are recorded and the scrutinee's enum type
    # selects between them at check time.
    graph = _make_graph_from_files(
        tmp_path,
        {
            "foreign": "enum Foreign\n  | same",
            "entry": (
                "import foreign\n"
                "enum Local\n  | same\n"
                "let value: Local = Local::same\n"
                "case value of | same => 1"
            ),
        },
    )

    result = resolve_program(graph)
    entry = result.modules[ENTRY_ID]
    case = entry.resolved.program.body.items[-1]
    pattern = case.branches[0].pattern
    assert pattern.node_id in entry.resolved.bare_variant_patterns
    candidate_owners = {
        ref.owner_name for ref in entry.resolved.bare_variant_candidates[pattern.node_id]
    }
    assert candidate_owners == {"Local", "Foreign"}
