"""Tests for the program-level scope resolver: ``resolve_program``.

These tests drive multi-module AgL programs through ``resolve_program`` and assert on:
- ``ResolvedProgram`` and ``ResolvedModule`` shape
- glob-import name resolution (unqualified access)
- brace-tail / hiding / qualified / as import forms
- full-surface qualified access
- clash-on-use disambiguation errors
- multiple import declarations merging
- duplicate-alias and alias-root-collision static errors
- ``::name`` self-reference
- static-root enforcement for entry and library modules
- parameter scope legality in entry and library modules
- header-only import placement
- wildcard subtree expansion and re-rooting
- wildcard overlap (idempotent same module, clash different modules)
- cross-file mutual recursion
- ``BindingRef.module_id`` set correctly
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import AglError, HiddenMemberError
from agm.agl.modules.ids import ENTRY_ID, STD_CONFIG_ID, ModuleId
from agm.agl.parser import AglSyntaxError, parse_program_seeded
from agm.agl.repl import ReplSession
from agm.agl.scope.program import ResolvedModule, ResolvedProgram, resolve_program
from agm.agl.scope.symbols import (
    AglScopeError,
    AmbiguousQualificationError,
    BinderKind,
    DuplicateDeclarationError,
    MissRepair,
    ReceiverOwner,
    UnknownMemberError,
    UnknownQualifierError,
)
from agm.agl.semantics.values import BoolValue, IntValue
from agm.agl.syntax.nodes import (
    AssignStmt,
    Block,
    Case,
    ConstructorPattern,
    FuncDef,
    VarPattern,
    VarRef,
)
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck.program import check_program
from tests._timeouts import fail_if_slow
from tests.agl.ir_harness import (
    base_caps,
    evaluate_ir_graph,
    make_file_graph_from_files,
)
from tests.agl.ir_harness import (
    make_inline_graph_from_files as _make_graph_from_files,
)
from tests.agl.module_graph import resolve_repl_entry
from tests.agl.qualifier_support import span_text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rejection(
    tmp_path: Path, modules: dict[str, str], module: str
) -> tuple[type[AglError], str | None]:
    """Resolve *modules* inline; the rejection's class and the text of *module* its span covers."""
    with pytest.raises(AglError) as exc_info:
        resolve_program(_make_graph_from_files(tmp_path, modules))
    return type(exc_info.value), span_text(modules[module], exc_info.value.span)


def _find_varref(program: object, name: str) -> VarRef | None:
    """Recursively find the first VarRef with the given name in a Program."""
    from agm.agl.syntax.nodes import (
        ArrayLit,
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
        Loop,
        Program,
        Raise,
        ScopeRegion,
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
        if isinstance(node, ScopeRegion):
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
        if isinstance(node, ArrayLit):
            for elem in node.elements:
                r = walk(elem)
                if r is not None:
                    return r
        if isinstance(node, DictLit):
            for entry in node.entries:
                r = walk(entry.value)
                if r is not None:
                    return r
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
                "entry": "import mylib::*\n()",
                "mylib": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert ENTRY_ID in result.modules
        assert mylib_id in result.modules

    def test_pre_pass_tables_populated(self, tmp_path: Path) -> None:
        """all_public_funcs and all_public_types are populated."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::*\n()",
                "mylib": "def foo() -> int = 1",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert (mylib_id, "foo") in result.all_public_funcs

    @pytest.mark.parametrize(
        "modules",
        (
            {
                "entry": "import alpha\n()",
                "alpha": "import beta\ndef alpha() -> int = 1",
                "beta": "def beta() -> int = 1",
            },
            {
                "entry": "import alpha\n()",
                "alpha": "import beta\ndef alpha() -> int = 1",
                "beta": "import alpha\ndef beta() -> int = 1",
            },
        ),
    )
    def test_preserves_loader_import_sccs(self, tmp_path: Path, modules: dict[str, str]) -> None:
        """Resolved programs retain the loader's ordered immutable import SCCs."""
        graph = _make_graph_from_files(tmp_path, modules)

        result = resolve_program(graph)

        assert result.import_sccs == graph.sccs
        assert isinstance(result.import_sccs, tuple)
        assert all(isinstance(component, tuple) for component in result.import_sccs)


# ---------------------------------------------------------------------------
# Test: glob import — bare name resolution
# ---------------------------------------------------------------------------


class TestGlobImport:
    def test_glob_import_varref_has_correct_module_id(self, tmp_path: Path) -> None:
        """A glob import resolves a VarRef to the owning module id."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::*\nlet x = foo()",
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

    def test_brace_tail_import_limits_bare_names(self, tmp_path: Path) -> None:
        """A brace-tail import exposes its selected bare member."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::{bar}\nlet x = bar()",
                "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
            },
        )
        result = resolve_program(graph)

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "bar")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.name == "bar"

    def test_brace_tail_import_omits_unselected_bare_name(self, tmp_path: Path) -> None:
        """A brace-tail import does not inject unselected bare members."""
        modules = {
            "entry": "import mylib::{bar}\nlet x = foo()",
            "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
        }
        assert _rejection(tmp_path, modules, "entry") == (AglScopeError, "foo")

    def test_hiding_import_excludes_hidden_name(self, tmp_path: Path) -> None:
        """'import mylib hiding foo' exposes 'bar' but not 'foo'."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::* hiding foo\nlet x = bar()",
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
        modules = {
            "entry": "import mylib::* hiding foo\nlet x = foo()",
            "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
        }
        assert _rejection(tmp_path, modules, "entry") == (AglScopeError, "foo")

    def test_brace_tail_rename(self, tmp_path: Path) -> None:
        """A brace-tail rename adds a bare spelling for its selected member.

        The BindingRef.name records the *original* declared name in the owning
        module (``"foo"``), not the exposed name (``"baz"``).  This is what
        the evaluator uses to look up the value in the owning module's frame.
        """
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::{foo as baz}\nlet x = baz()",
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

    def test_brace_tail_rename_of_scoped_record_keeps_its_owner_path(self, tmp_path: Path) -> None:
        """A renamed scoped-record import keeps the source's named scope.

        Constructor identity is structured: the candidate is exposed at the
        importing module's root under its alias, while its ``owner_path`` still
        carries the declaring scope (``A``) rather than defaulting to the module
        root, or downstream field-type lookup targets the wrong declaration.
        """
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::{A::Token as T}\nlet t = T(n = 1)",
                "mylib": "scope A\n  record Token\n    n: int\nend A",
            },
        )
        result = resolve_program(graph)
        entry_resolved = result.modules[ENTRY_ID].resolved
        (candidate,) = entry_resolved.constructor_refs.values()
        assert candidate.owner_name == "Token"
        assert candidate.owner_path == ("A",)

    def test_brace_tail_rename_of_scoped_enum_variant_is_a_constructor_candidate(
        self, tmp_path: Path
    ) -> None:
        """An individually-imported, renamed scoped enum variant resolves as a constructor.

        ``all_public_types`` is keyed by the owning enum's QName, not the
        variant's, so the candidate must be recovered from the cross-module
        constructor-ref table instead.  It is exposed at the importing
        module's root under its alias, and must still carry the owning enum's
        selected member record's name and enum-qualified scope.
        """
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::{A::Status::Good as X}\nlet x = X",
                "mylib": "scope A\n  enum Status\n    | Good\n    | Bad\nend A",
            },
        )
        result = resolve_program(graph)
        entry_resolved = result.modules[ENTRY_ID].resolved
        (candidate,) = entry_resolved.constructor_refs.values()
        assert candidate.owner_name == "Good"
        assert candidate.owner_path == ("A", "Status")

    def test_qualified_import_prevents_bare_access(self, tmp_path: Path) -> None:
        """'import mylib qualified' — bare 'foo' should error."""
        modules = {
            "entry": "import mylib\nlet x = foo()",
            "mylib": "def foo() -> int = 42",
        }
        assert _rejection(tmp_path, modules, "entry") == (AglScopeError, "foo")

    def test_qualified_import_allows_qualified_access(self, tmp_path: Path) -> None:
        """'import mylib qualified' — 'mylib::foo' should resolve."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\nlet x = mylib::foo()",
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

    def test_alias_does_not_inject_bare_members(self, tmp_path: Path) -> None:
        """An import alias creates only its canonical qualified route."""
        modules = {
            "entry": "import mylib as M\nlet x = foo()",
            "mylib": "def foo() -> int = 42",
        }
        assert _rejection(tmp_path, modules, "entry") == (AglScopeError, "foo")


# ---------------------------------------------------------------------------
# Test: qualified access
# ---------------------------------------------------------------------------


class TestQualifiedAccess:
    def test_brace_tail_keeps_the_full_qualified_surface(self, tmp_path: Path) -> None:
        """A positive import tail does not narrow qualified access."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::{foo}\nlet x = mylib::bar()",
                "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
            },
        )

        result = resolve_program(graph)

        bar = _find_varref(graph.modules[ENTRY_ID].program, "bar")
        assert bar is not None
        assert result.modules[ENTRY_ID].resolved.resolution[bar.node_id].name == "bar"

    def test_qualified_brace_tail_allows_selected_member(self, tmp_path: Path) -> None:
        """A brace-tail member remains available through its module route."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::{foo}\nlet x = mylib::foo()",
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
        modules = {
            "entry": "let x = nomodule::foo()",
        }
        assert _rejection(tmp_path, modules, "entry") == (UnknownQualifierError, "nomodule::foo")

    def test_local_scope_beats_a_same_named_module_route(self, tmp_path: Path) -> None:
        """``mylib::foo`` is the own scope's; ``/mylib::foo`` reaches the module."""
        result = evaluate_ir_graph(
            "import mylib\n\nscope mylib\n  def foo() -> int = 1\nend mylib\n\n"
            "let own = mylib::foo()\nlet routed = /mylib::foo()",
            {"mylib": "def foo() -> int = 2"},
            tmp_path,
        )

        assert (result["own"], result["routed"]) == (IntValue(1), IntValue(2))

    @pytest.mark.parametrize(
        ("declaration", "use"),
        [
            (
                "record Box\n  v: int",
                "let b = ::mylib::Box(v = 1)\nlet r = case b of | mylib::Box(v) => v == 1",
            ),
            (
                "exception Oops",
                'let e: Exception = ::mylib::Oops(message = "m")\nlet r = e is mylib::Oops',
            ),
        ],
        ids=["pattern", "is-test"],
    )
    def test_local_scope_beats_a_same_named_module_route_in_patterns_and_is_tests(
        self, tmp_path: Path, declaration: str, use: str
    ) -> None:
        local = "\n".join(f"  {line}" for line in declaration.splitlines())
        result = evaluate_ir_graph(
            f"import mylib\n\nscope mylib\n{local}\nend mylib\n\n{use}",
            {"mylib": declaration},
            tmp_path,
        )

        assert result["r"] == BoolValue(True)

    def test_qualified_assign_writes_the_local_scope_beside_a_same_named_module_route(
        self, tmp_path: Path
    ) -> None:
        """Assignment decides the spelling exactly as a read does."""
        result = evaluate_ir_graph(
            "import mylib\n\nscope mylib\n  var counter = 0\nend mylib\n\n"
            "mylib::counter := 1\nlet r = mylib::counter",
            {"mylib": "def counter() -> int = 2"},
            tmp_path,
        )

        assert result["r"] == IntValue(1)

    def test_nested_scope_sharing_only_a_run_of_segments_with_an_import_route_is_not_a_clash(
        self, tmp_path: Path
    ) -> None:
        """The route is the leading segment alone, so ``alpha::beta`` naming ``import alpha/beta``

        only through a two-segment run is not a route at all: the local nested scope wins outright.
        """
        entry_source = (
            "import alpha/beta\n"
            "\n"
            "scope alpha\n"
            "\n"
            "  scope beta\n"
            "    def member() -> int = 1\n"
            "  end beta\n"
            "end alpha\n"
            "\n"
            "let result = alpha::beta::member()"
        )

        result = evaluate_ir_graph(
            entry_source, {"alpha/beta": "def member() -> int = 2"}, tmp_path
        )

        assert result["result"] == IntValue(1)

    def test_import_route_is_not_a_suffix_matched_local_scope(self, tmp_path: Path) -> None:
        entry_source = (
            "import beta\n"
            "\n"
            "scope alpha\n"
            "\n"
            "  scope beta\n"
            "    def local() -> int = 1\n"
            "  end beta\n"
            "end alpha\n"
            "\n"
            "let result = beta::remote()"
        )

        result = evaluate_ir_graph(entry_source, {"beta": "def remote() -> int = 2"}, tmp_path)

        assert result["result"] == IntValue(2)

    def test_anchored_nested_scope_beside_a_complete_import_route_is_accepted(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import alpha/beta\n"
                    "\n"
                    "scope alpha\n"
                    "\n"
                    "  scope beta\n"
                    "    def member() -> int = 1\n"
                    "  end beta\n"
                    "end alpha\n"
                    "\n"
                    "::alpha::beta::member()"
                ),
                "alpha/beta": "def member() -> int = 2",
            },
        )

        assert resolve_program(graph).entry_id == ENTRY_ID

    def test_constructor_path_sharing_only_a_run_of_segments_with_an_import_route_is_not_a_clash(
        self, tmp_path: Path
    ) -> None:
        """As the value case, a three-segment run naming ``import alpha/beta/Color`` is not a route.

        The local enum constructor is selected outright, never merged with the imported function.
        """
        entry_source = (
            "import alpha/beta/Color\n"
            "\n"
            "scope alpha\n"
            "\n"
            "  scope beta\n"
            "    enum Color\n"
            "      | red\n"
            "  end beta\n"
            "end alpha\n"
            "\n"
            "let value = alpha::beta::Color::red\n"
            "let result = case value of\n"
            "  | alpha::beta::Color::red => true"
        )

        result = evaluate_ir_graph(
            entry_source, {"alpha/beta/Color": "def red() -> int = 2"}, tmp_path
        )

        assert result["result"] == BoolValue(True)

    def test_anchored_constructor_path_beside_a_complete_import_route_is_accepted(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import alpha/beta/Color\n"
                    "\n"
                    "scope alpha\n"
                    "\n"
                    "  scope beta\n"
                    "    enum Color\n"
                    "      | red\n"
                    "  end beta\n"
                    "end alpha\n"
                    "\n"
                    "::alpha::beta::Color::red"
                ),
                "alpha/beta/Color": "def red() -> int = 2",
            },
        )

        assert resolve_program(graph).entry_id == ENTRY_ID

    def test_current_module_anchor_selects_the_own_scope_beside_a_route(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import mylib\n"
                    "\n"
                    "scope mylib\n"
                    "  def foo() -> int = 1\n"
                    "end mylib\n"
                    "\n"
                    "::mylib::foo()"
                ),
                "mylib": "def foo() -> int = 2",
            },
        )

        assert resolve_program(graph).entry_id == ENTRY_ID


# ---------------------------------------------------------------------------
# Test: clash deferred to use-site
# ---------------------------------------------------------------------------


class TestClashDeferred:
    def test_two_imports_same_name_clashes_at_use(self, tmp_path: Path) -> None:
        """Two glob imports both expose 'foo', making its bare use ambiguous."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import libA::*\nimport libB::*\nlet x = foo()",
                "libA": "def foo() -> int = 1",
                "libB": "def foo() -> int = 2",
            },
        )
        with pytest.raises(AmbiguousQualificationError):
            resolve_program(graph)

    def test_clash_error_points_at_the_use_site(self, tmp_path: Path) -> None:
        """The ambiguity is reported at the bare use, not at either import."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import libA::*\nimport libB::*\nlet x = foo()",
                "libA": "def foo() -> int = 1",
                "libB": "def foo() -> int = 2",
            },
        )
        with pytest.raises(AmbiguousQualificationError) as exc_info:
            resolve_program(graph)
        span = exc_info.value.span
        assert span is not None
        # ``foo`` on the third entry line, columns 9..12 -- the reference itself.
        assert (span.start_line, span.start_col, span.end_line, span.end_col) == (3, 9, 3, 12)

    def test_scopes_renamed_to_one_spelling_combine_under_a_use(self, tmp_path: Path) -> None:
        files = {
            "entry": "import lib\nuse lib::{One as X, Two as X}\nuse X::*\nfirst() + second()",
            "lib": (
                "scope One\n  def first() -> int = 1\nend One\n"
                "\n"
                "scope Two\n  def second() -> int = 2\nend Two"
            ),
        }

        check_program(resolve_program(_make_graph_from_files(tmp_path, files)), base_caps())

    def test_use_deduplicates_routes_to_same_reexport_origin(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import core::{S}\nimport facade::{S}\nuse S::*\nmember()",
                "core": "scope S\n  def member() -> int = 1\nend S",
                "facade": "export core::{S}",
            },
        )

        result = resolve_program(graph)

        assert ENTRY_ID in result.modules

    def test_use_deduplicates_type_scope_routes_to_same_reexport_origin(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import core::{E as X}\nimport facade::{X}\nuse X::*\nA",
                "core": "enum E\n  | A",
                "facade": "export core::{E as X}",
            },
        )

        result = resolve_program(graph)

        core_id = ModuleId.from_path("core")
        facade = result.modules[ModuleId.from_path("facade")]
        assert facade.scope_exports["X"] == frozenset({(core_id, "E")})

    def test_use_deduplicates_empty_scope_routes_to_same_reexport_origin(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import core::{S}\nimport facade::{S}\nuse S::*\n()",
                "core": "scope S\nend S",
                "facade": "export core::{S}",
            },
        )

        assert ENTRY_ID in resolve_program(graph).modules

    def test_use_unions_filtered_routes_to_same_reexport_origin(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": ("import core::{S}\nimport facade::{S}\nuse S::*\nalpha() + beta()"),
                "core": ("scope S\n  def alpha() -> int = 1\n  def beta() -> int = 2\nend S"),
                "facade": "export core hiding S::beta",
            },
        )

        assert ENTRY_ID in resolve_program(graph).modules

    def test_no_clash_same_qname(self, tmp_path: Path) -> None:
        """Two imports of the same module's same function don't clash (idempotent)."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\nimport mylib::{foo}\nlet x = foo()",
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
                "entry": ("import mylib::{foo}\nimport mylib::{bar}\nlet a = foo()\nlet b = bar()"),
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
        assert resolve_program(graph).entry_id == graph.entry_id

    def test_alias_root_collision(self, tmp_path: Path) -> None:
        """'import libA' + 'import libB as libA' → alias-root collision."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import libA::*\nimport libB as libA\n()",
                "libA": "def foo() -> int = 1",
                "libB": "def bar() -> int = 2",
            },
        )
        assert resolve_program(graph).entry_id == graph.entry_id

    def test_selecting_a_declaration_its_module_lacks_suggests_it_is_not_exported(
        self, tmp_path: Path
    ) -> None:
        source = "import lib::{present, missing}\n()"
        graph = _make_graph_from_files(
            tmp_path, {"entry": source, "lib": "def present() -> int = 1"}
        )
        with pytest.raises(UnknownMemberError) as caught:
            resolve_program(graph)

        error = caught.value
        assert (type(error), error.spelling, error.repair) == (
            UnknownMemberError,
            "lib::missing",
            MissRepair.NOT_EXPORTED,
        )


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
                "entry": "import mylib::*\n()",
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
        modules = {
            "entry": "let x = ::nonexistent()",
        }
        assert _rejection(tmp_path, modules, "entry") == (UnknownMemberError, "::nonexistent")

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
            """Find the first VarRef named 'foo' with a qualifier chain."""
            from agm.agl.syntax.nodes import Program

            if isinstance(node, VarRef) and node.name == "foo" and (node.qualifier is not None):
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

    @pytest.mark.parametrize(
        ("binding", "statement", "error"),
        [
            pytest.param("var x = 1", "let _ = ::x", None, id="read-var"),
            pytest.param("var x = 1", "::x := 2", None, id="assign-var"),
            pytest.param("let x = 1", "let _ = ::x", None, id="read-let"),
            pytest.param("let x = 1", "::x := 2", None, id="assign-let"),
            pytest.param("var y = 1", "let _ = ::x", UnknownMemberError, id="read-missing"),
            pytest.param("var y = 1", "::x := 2", UnknownMemberError, id="assign-missing"),
        ],
    )
    def test_anchored_root_binding_is_read_and_assigned_through_one_selection(
        self, tmp_path: Path, binding: str, statement: str, error: type[AglScopeError] | None
    ) -> None:
        """``::x := e`` selects its target exactly as the read ``::x`` does, past a local ``x``."""
        library = f"{binding}\ndef touch() -> unit =\n  let x = 5\n  {statement}\n  ()\n"
        graph = _make_graph_from_files(
            tmp_path, {"entry": "import lib\nlib::touch()", "lib": library}
        )
        if error is not None:
            with pytest.raises(error) as caught:
                resolve_program(graph)
            assert type(caught.value) is error
            return
        result = resolve_program(graph)
        lib = graph.modules[ModuleId.from_path("lib")].program
        touch = lib.body.items[-1]
        assert isinstance(touch, FuncDef) and isinstance(touch.body, Block)
        target = touch.body.items[1]
        node = target if isinstance(target, AssignStmt) else _find_varref(target, "x")
        assert node is not None
        ref = result.modules[ModuleId.from_path("lib")].resolved.resolution[node.node_id]
        assert (ref.module_id, ref.scope_path, ref.name) == (ModuleId.from_path("lib"), (), "x")

    def test_anchored_root_var_assignment_updates_the_module_binding(self, tmp_path: Path) -> None:
        """``::x := e`` in a library function writes the library's root ``var``."""
        library = "var x = 1\ndef bump() -> int =\n  let x = 40\n  ::x := x + 2\n  ::x\n"
        result = evaluate_ir_graph(
            "import lib\nlet first = lib::bump()\nlet seen = lib::x", {"lib": library}, tmp_path
        )
        assert (result["first"], result["seen"]) == (IntValue(42), IntValue(42))

    def test_inline_entry_statement_var_is_written_through_its_root_path(
        self, tmp_path: Path
    ) -> None:
        """An inline entry's top-level ``var`` is a root binding: ``::x := e`` writes it."""
        result = evaluate_ir_graph("var x = 1\n::x := ::x + 41\nlet seen = x", {}, tmp_path)
        assert result["seen"] == IntValue(42)


# ---------------------------------------------------------------------------
# Test: qualified-member diagnostics
# ---------------------------------------------------------------------------


class TestQualifiedMemberDiagnostics:
    def test_missing_member_error_names_the_qualified_module(self, tmp_path: Path) -> None:
        """'libA::secret' must name libA when 'secret' is declared only by libB.

        The qualifier chosen at the use site decides which module the error
        blames, not whichever imported module happens to declare that name.
        """
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import libA::*\nimport libB::*\nlet x = libA::secret()",
                "libA": "def pub() -> int = 1",
                "libB": "def other() -> int = 2\ndef secret() -> int = 99",
            },
        )
        with pytest.raises(UnknownMemberError) as exc_info:
            resolve_program(graph)
        assert exc_info.value.spelling == "libA::secret"


# ---------------------------------------------------------------------------
# Test: declaration-only enforcement
# ---------------------------------------------------------------------------


class TestBuiltinVarPlacement:
    def test_entry_declaration_rejected(self, tmp_path: Path) -> None:
        modules = {"entry": "builtin var strict-json: bool\n()"}
        assert _rejection(tmp_path, modules, "entry") == (
            AglScopeError,
            "builtin var strict-json: bool",
        )

    def test_arbitrary_library_declaration_rejected(self, tmp_path: Path) -> None:
        modules = {
            "entry": "import mylib::*\n()",
            "mylib": "builtin var strict-json: bool",
        }
        assert _rejection(tmp_path, modules, "mylib") == (
            AglScopeError,
            "builtin var strict-json: bool",
        )

    def test_std_config_declarations_and_qualified_assignment_resolve(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import std/config::*\nstd/config::strict-json := true\nstd/config::strict-json"
                )
            },
        )

        resolved = resolve_program(graph)
        std_config = resolved.modules[STD_CONFIG_ID]
        binding = std_config.resolved.root_scope.lookup("strict-json")
        assert binding is not None
        assert binding.kind is BinderKind.builtin_var_binding
        assignment_ref = next(
            ref
            for ref in resolved.modules[ENTRY_ID].resolved.resolution.values()
            if ref.name == "strict-json" and ref.kind is BinderKind.builtin_var_binding
        )
        assert assignment_ref.module_id == STD_CONFIG_ID


class TestExportedVarBindings:
    """An annotated simple ``var`` is exported like an annotated ``let`` and is writable."""

    def test_annotated_root_var_is_exported(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "var total: int = 0",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert "total" in result.modules[mylib_id].exports

    def test_annotated_scoped_var_is_exported(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "scope Review\n  var attempts: int = 0\nend Review",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert ("Review", "attempts") in result.modules[mylib_id].exports

    def test_unannotated_var_is_exported(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\n()",
                "mylib": "var total = 0",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert "total" in result.modules[mylib_id].exports

    def test_cross_module_var_write_is_mutable_with_source_module_id(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::*\ntotal := 1",
                "mylib": "var total: int = 0",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        entry_resolved = result.modules[ENTRY_ID].resolved
        main = entry_resolved.program.body.items[-1]
        assert isinstance(main, FuncDef)
        assignment = main.body.items[-1]
        assert isinstance(assignment, AssignStmt)
        ref = entry_resolved.resolution[assignment.node_id]
        assert ref.mutable is True
        assert ref.kind is BinderKind.var_binding
        assert ref.module_id == mylib_id


class TestStaticModuleRoots:
    def test_let_and_var_are_allowed_at_library_root_and_in_scope_regions(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib\nprogram def main() -> unit = ()",
                "mylib": (
                    "let root = 1\n"
                    "var total = 0\n"
                    "\n"
                    "scope Review\n"
                    '  let title = "review"\n'
                    "  var attempts = 0\n"
                    "end Review"
                ),
            },
        )

        assert ModuleId.from_path("mylib") in resolve_program(graph).modules

    @pytest.mark.parametrize("module", ("entry", "library"))
    def test_bare_expression_at_root_is_rejected_for_every_module(
        self, tmp_path: Path, module: str
    ) -> None:
        modules = (
            {"entry": "42"}
            if module == "entry"
            else {"entry": "import mylib\nprogram def main() -> unit = ()", "mylib": "42"}
        )

        with pytest.raises(AglScopeError):
            resolve_program(make_file_graph_from_files(tmp_path, modules))

    @pytest.mark.parametrize("module", ("entry", "library"))
    def test_assignment_at_root_is_rejected_for_every_module(
        self, tmp_path: Path, module: str
    ) -> None:
        modules = (
            {"entry": "var value = 1\nvalue := 2"}
            if module == "entry"
            else {
                "entry": "import mylib\nprogram def main() -> unit = ()",
                "mylib": "var value = 1\nvalue := 2",
            }
        )

        with pytest.raises(AglScopeError):
            resolve_program(make_file_graph_from_files(tmp_path, modules))


# ---------------------------------------------------------------------------
# Test: header-only imports
# ---------------------------------------------------------------------------


class TestHeaderOnlyImports:
    def test_import_after_def_in_non_entry_errors(self, tmp_path: Path) -> None:
        """An import after a def in a non-entry module is an error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::*\n()",
                "mylib": "def foo() -> int = 1\nimport libB::*",
                "libB": "def bar() -> int = 2",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_import_after_infix_decl_in_non_entry_errors(self, tmp_path: Path) -> None:
        modules = {
            "entry": "import mylib::*\n()",
            "mylib": "infixl |> at 12\nimport libB::*\ndef |>(x: int, y: int) -> int = x",
            "libB": "def bar() -> int = 2",
        }
        assert _rejection(tmp_path, modules, "mylib") == (AglScopeError, "import libB::*")

    def test_import_at_top_of_non_entry_allowed(self, tmp_path: Path) -> None:
        """Import declarations at the top of a non-entry module are allowed."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::*\n()",
                "mylib": "import libB::*\ndef foo() -> int = bar()",
                "libB": "def bar() -> int = 42",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules
        mylib_id = ModuleId.from_path("mylib")
        assert "foo" in result.modules[mylib_id].exports

    def test_a_region_that_is_only_an_import_is_allowed_at_the_top_of_a_non_entry_module(
        self, tmp_path: Path
    ) -> None:
        """A region's own header rule is independent of the module's root-level rule.

        A library module's root-level items are a `def` and, after it, a
        `scope` region whose *own* first item is an import -- the region's
        header rule (tracked separately for the region's own item sequence)
        must not be short-circuited by the module-root rule having already
        seen the preceding `def`.
        """
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::*\n()",
                "mylib": (
                    "def helper() -> int = 1\n"
                    "\n"
                    "scope A\n  import libB::*\n  def foo() -> int = bar()\nend A"
                ),
                "libB": "def bar() -> int = 42",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules
        mylib_id = ModuleId.from_path("mylib")
        assert ("A", "foo") in result.modules[mylib_id].exports

    def test_import_in_entry_works(self, tmp_path: Path) -> None:
        """Entry module works fine with import followed by def."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::*\ndef helper() -> int = foo()\n()",
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

    def test_import_after_def_in_entry_errors(self, tmp_path: Path) -> None:
        """An import after a def at the entry module's root is also an error."""
        entry = "def helper() -> int = 1\nimport mylib::*"
        graph = make_file_graph_from_files(
            tmp_path, {"entry": entry, "mylib": "def foo() -> int = 42"}
        )
        with pytest.raises(AglScopeError) as exc_info:
            resolve_program(graph)
        assert type(exc_info.value) is AglScopeError
        assert span_text(entry, exc_info.value.span) == "import mylib::*"


_TWO_GEOS = {
    "tl": "record Geo\nrecord Geo::Inner\n  y: int\n",
    "tl2": "record Geo\nrecord Geo::Inner\n  z: int\n",
}


def _placement_failure(tmp_path: Path, graph_kind: str, source: str) -> BaseException:
    """The error resolving *source* as an inline entry, a file entry, or one REPL entry."""
    modules = {"entry": source, **_TWO_GEOS}
    if graph_kind == "repl":
        session = ReplSession(cwd=tmp_path, default_stdlib=False)
        session.open()
        for name, text in _TWO_GEOS.items():
            (tmp_path / f"{name}.agl").write_text(text)
        failure = session.eval_entry(source).failure
        assert failure is not None
        return failure
    make = make_file_graph_from_files if graph_kind == "file" else _make_graph_from_files
    with pytest.raises(AglScopeError) as caught:
        resolve_program(make(tmp_path, modules, default_stdlib=False))
    return caught.value


class TestPlacementIsCheckedBeforeNames:
    """A misplaced header or root statement is reported before any name it precedes or follows."""

    @pytest.mark.parametrize("graph_kind", ("inline", "file"))
    def test_a_late_root_import(self, tmp_path: Path, graph_kind: str) -> None:
        """The REPL hoists an entry's imports, so only a module root rejects the late one."""
        source = "import tl::*\ndef Geo::Inner::m(self) -> int = 1\nimport tl2::*"
        failure = _placement_failure(tmp_path, graph_kind, source)
        assert type(failure) is AglScopeError
        assert (failure.span.start_line, failure.span.start_col) == (3, 1)

    @pytest.mark.parametrize("graph_kind", ("inline", "file", "repl"))
    def test_a_late_region_import(self, tmp_path: Path, graph_kind: str) -> None:
        source = (
            "import tl::*\nimport tl2::*\n\n"
            "scope A\n  let v = Geo::Inner(y = 1)\n  import tl\nend A"
        )
        failure = _placement_failure(tmp_path, graph_kind, source)
        assert type(failure) is AglScopeError
        assert (failure.span.start_line, failure.span.start_col) == (6, 3)

    def test_an_import_after_an_inline_statement(self, tmp_path: Path) -> None:
        failure = _placement_failure(
            tmp_path, "inline", "print(1)\nimport tl\ndef f() -> int = missing"
        )
        assert type(failure) is AglScopeError
        assert (failure.span.start_line, failure.span.start_col) == (2, 1)

    @pytest.mark.parametrize(
        "source",
        (
            "import tl::*\nimport tl2::*\nfn(p: Geo::Inner::Nope) => 1",
            "import tl::*\nimport tl2::*\nvar x = 1\nx := Geo::Inner::Nope",
        ),
        ids=("bare-expression", "assignment"),
    )
    def test_a_statement_at_a_static_root(self, tmp_path: Path, source: str) -> None:
        failure = _placement_failure(tmp_path, "file", source)
        assert type(failure) is AglScopeError
        assert failure.span.start_col == 1
        assert failure.span.start_line == source.count("\n") + 1

    @pytest.mark.parametrize("graph_kind", ("inline", "file", "repl"))
    @pytest.mark.parametrize(
        "nested",
        (
            "import tl",
            "export tl",
            "infixl |> at 12",
            "def h() -> int = 1",
            "record R",
            "let A::x = 1",
            "var A::x = 1",
        ),
    )
    def test_a_nested_block_declaration_after_an_unknown_name(
        self, tmp_path: Path, graph_kind: str, nested: str
    ) -> None:
        source = f"def f() -> int = missing\n\ndef g() -> int =\n  {nested}\n  1"
        failure = _placement_failure(tmp_path, graph_kind, source)
        assert type(failure) is AglScopeError
        assert (failure.span.start_line, failure.span.start_col) == (4, 3)

    @pytest.mark.parametrize("graph_kind", ("inline", "file", "repl"))
    def test_nested_misplacements_in_source_order(self, tmp_path: Path, graph_kind: str) -> None:
        """An inner block's misplaced item precedes a later one of its enclosing block."""
        source = (
            "def g() -> int =\n  if true =>\n    def k() -> int = 1\n    ()\n  | else =>\n    ()\n"
            "  record R\n  1"
        )
        failure = _placement_failure(tmp_path, graph_kind, source)
        assert type(failure) is AglScopeError
        assert (failure.span.start_line, failure.span.start_col) == (3, 5)


_OWN_SCOPE = "scope S\n  enum E\n    | Red\n  let x = 1\nend S"


class TestOwnUseSelectionsInSourceOrder:
    """A use tail or hiding naming nothing an own target reaches fails where the use is written."""

    @pytest.mark.parametrize("graph_kind", ("inline", "file", "repl"))
    @pytest.mark.parametrize(
        "use", ("use S::E::{Nope}", "use S::* hiding Nope", "use S::{E::Nope}")
    )
    def test_before_a_later_unknown_name(self, tmp_path: Path, graph_kind: str, use: str) -> None:
        source = f"{use}\n\n{_OWN_SCOPE}\n\ndef f() -> int = missing"
        failure = _placement_failure(tmp_path, graph_kind, source)
        assert type(failure) is UnknownMemberError
        assert (failure.span.start_line, failure.span.start_col) == (1, 1)

    @pytest.mark.parametrize("graph_kind", ("inline", "file", "repl"))
    def test_after_an_earlier_unknown_name(self, tmp_path: Path, graph_kind: str) -> None:
        source = f"def f() -> int = missing\n\nscope Q\n  use S::{{Nope}}\nend Q\n\n{_OWN_SCOPE}"
        failure = _placement_failure(tmp_path, graph_kind, source)
        assert type(failure) is not UnknownMemberError
        assert (failure.span.start_line, failure.span.start_col) == (1, 18)

    @pytest.mark.parametrize("use", ("use S::{x}", "use S::* hiding x", "use S::{E::Red, x}"))
    def test_a_later_scoped_binding_is_selectable(self, tmp_path: Path, use: str) -> None:
        result = evaluate_ir_graph(f"{use}\n\n{_OWN_SCOPE}\n\nlet seen = S::x", {}, tmp_path)
        assert result["seen"] == IntValue(1)


# ---------------------------------------------------------------------------
# Test: import and export as region header items
# ---------------------------------------------------------------------------


class TestRegionImportExportHeaderPlacement:
    """`import`, `export`, and `use` are region header items.

    The rule applies uniformly regardless of module kind: a region's own
    items are checked independently of whether the module is a library
    module or the entry module. The scope pass is the sole owner of this
    rule, so the violation surfaces from `resolve_program`, not from graph
    construction.
    """

    @pytest.mark.parametrize("item", ("import libB", "import libB::*", "use A::*", "export libB"))
    def test_rejected_after_a_region_item_in_a_library_module(
        self, tmp_path: Path, item: str
    ) -> None:
        with pytest.raises((AglScopeError, AglSyntaxError)):
            graph = _make_graph_from_files(
                tmp_path,
                {
                    "entry": "import mylib::*\n()",
                    "mylib": f"scope A\ndef local() -> int = 1\n{item}\nend A",
                    "libB": "def bar() -> int = 2",
                },
            )
            resolve_program(graph)

    @pytest.mark.parametrize("item", ("import libB", "import libB::*", "use A::*", "export libB"))
    def test_rejected_after_a_region_item_in_the_entry_module(
        self, tmp_path: Path, item: str
    ) -> None:
        with pytest.raises((AglScopeError, AglSyntaxError)):
            graph = _make_graph_from_files(
                tmp_path,
                {
                    "entry": f"scope A\ndef local() -> int = 1\n{item}\nend A",
                    "libB": "def bar() -> int = 2",
                },
            )
            resolve_program(graph)


# ---------------------------------------------------------------------------
# Test: scoped import bare narrowing
# ---------------------------------------------------------------------------


class TestScopedImportBareNarrowing:
    """Scoped glob and brace-tail imports narrow bare names to their region."""

    def test_wildcard_expanding_to_a_duplicate_name_is_accepted_when_unused(
        self, tmp_path: Path
    ) -> None:
        """A region-scoped wildcard doesn't hard-error on a same-named export it never uses.

        Matches the root spelling's clash-on-use policy: two modules
        exposing the same bare name through one wildcard is only ever an
        ambiguity at the name's first use, never at the import declaration.
        """
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "scope A\n  import pkg/*::*\nend A",
                "pkg/one": "def alpha() -> int = 1",
                "pkg/two": "def alpha() -> int = 2",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

    def test_wildcard_expanding_to_a_duplicate_name_is_ambiguous_at_use(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "scope A\n  import pkg/*::*\n  def show() -> int = alpha()\nend A\n\nA::show()"
                ),
                "pkg/one": "def alpha() -> int = 1",
                "pkg/two": "def alpha() -> int = 2",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_bare_name_resolves_inside_its_own_region(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "scope A\n  import mylib::*\n  def show() -> int = foo()\nend A\n\nA::show()"
                ),
                "mylib": "def foo() -> int = 1",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

    def test_bare_name_is_not_visible_at_the_module_root(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": ("scope A\n  import mylib::*\nend A\n\nfoo()"),
                "mylib": "def foo() -> int = 1",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_bare_name_does_not_leak_to_a_sibling_region(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "scope A\n  import mylib::*\nend A\n"
                    "\n"
                    "scope B\n  def show() -> int = foo()\nend B\n\nB::show()"
                ),
                "mylib": "def foo() -> int = 1",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_qualifier_route_resolves_module_wide_regardless_of_nesting(
        self, tmp_path: Path
    ) -> None:
        """A scoped import still makes the module's qualifier route available module-wide."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": ("scope A\n  import mylib\nend A\n\nlet x = mylib::foo()"),
                "mylib": "def foo() -> int = 1",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

    def test_plain_scoped_import_contributes_no_bare_names(self, tmp_path: Path) -> None:
        """A plain scoped import without a tail exposes no bare names."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "scope A\n  import mylib\n  def show() -> int = foo()\nend A\n\nA::show()"
                ),
                "mylib": "def foo() -> int = 1",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_brace_tail_narrows_a_renamed_bare_name(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "scope A\n"
                    "  import mylib::{foo as f}\n"
                    "  def show() -> int = f()\n"
                    "end A\n"
                    "\n"
                    "A::show()"
                ),
                "mylib": "def foo() -> int = 1",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

    def test_glob_import_hiding_clause_narrows_the_remaining_names(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "scope A\n  import mylib::* hiding secret\n"
                    "  def show() -> int = foo()\nend A\n\nA::show()"
                ),
                "mylib": "def foo() -> int = 1\ndef secret() -> int = 2",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

    def test_scoped_import_colliding_with_a_scoped_member_prefers_the_member(
        self, tmp_path: Path
    ) -> None:
        """A region's own member shadows a same-named bare import, like at the root."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "scope A\n  import mylib::*\n"
                    "  def foo() -> int = 2\n  def show() -> int = foo()\nend A\n\nA::show()"
                ),
                "mylib": "def foo() -> int = 1",
            },
        )
        result = resolve_program(graph)
        entry = result.modules[ENTRY_ID]
        var = _find_varref(entry.resolved.program, "foo")
        assert var is not None
        ref = entry.resolved.resolution[var.node_id]
        assert ref.module_id == ENTRY_ID
        assert ref.scope_path == ("A",)

    def test_glob_import_narrows_a_bare_constructor_to_its_region(self, tmp_path: Path) -> None:
        """A scoped glob import also narrows a bare-exposed scoped type's constructor."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "scope A\n  import mylib::*\n"
                    "  def make() -> Geo::Point = Geo::Point(x = 1)\nend A\n\nA::make()"
                ),
                "mylib": "scope Geo\n  record Point\n    x: int\nend Geo",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

    def test_glob_import_narrows_a_bare_enum_variant_reference_to_its_region(
        self, tmp_path: Path
    ) -> None:
        """A scoped glob import makes imported enum variants usable bare."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "scope Vec\n"
                    "  import lib2::*\n"
                    "  def pick() -> Color = Green\n"
                    "end Vec\n"
                    "\n"
                    "Vec::pick()"
                ),
                "lib2": "enum Color\n  | Red\n  | Green",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

    def test_glob_import_narrows_a_bare_enum_variant_pattern_to_its_region(
        self, tmp_path: Path
    ) -> None:
        """A scoped glob import also makes imported enum variants matchable bare."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "scope Vec\n  import lib2::*\n"
                    "  def classify(c: Color) -> int = case c of | Red() => 0 | Green => 1\n"
                    "  def run() -> int = classify(Green)\n"
                    "end Vec\n\nVec::run()"
                ),
                "lib2": "enum Color\n  | Red\n  | Green",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

    def test_glob_import_hiding_keeps_the_unhidden_bare_enum_variant(self, tmp_path: Path) -> None:
        """A scoped glob import's ``hiding`` clause excludes only the named variant."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "scope A\n  import lib2::* hiding Color::Red\n"
                    "  def pick() -> Color = Green\nend A"
                ),
                "lib2": "enum Color\n  | Red\n  | Green",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

    def test_glob_import_hiding_drops_the_hidden_bare_enum_variant(self, tmp_path: Path) -> None:
        """A scoped glob import's ``hiding`` clause drops the named variant's own bare spelling."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "scope A\n  import lib2::* hiding Color::Red\n"
                    "  def pick() -> Color = Red\nend A"
                ),
                "lib2": "enum Color\n  | Red\n  | Green",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_enum_variant_expansion_yields_to_a_same_named_top_level_exception(
        self, tmp_path: Path
    ) -> None:
        """A same-named top-level exception keeps its bare name; the variant is skipped.

        Expanding a bare-exposed enum into its variants checks the *plain*
        atom a standalone exception would occupy, not the variant's own
        owner-path-qualified key, so a same-named exception and enum
        variant both contributed bare do not clash.
        """
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": ('scope A\n  import lib4::*\n  let e = Red(msg = "boom")\nend A'),
                "lib4": "enum Status\n  | Red\n  | Green\n\nexception Red\n  msg: text",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

    def test_use_after_a_region_scoped_glob_import_resolves_the_unrouted_scope(
        self, tmp_path: Path
    ) -> None:
        """A `use` glob reaches a scope exposed by a region-scoped glob import."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "scope A\n  import libg::*\n  use Geo::*\n"
                    "  def show() -> int = dist()\nend A\n\nA::show()"
                ),
                "libg": "scope Geo\n  def dist() -> int = 5\nend Geo",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules

    def test_use_after_a_region_scoped_glob_import_skips_unrelated_decl_bare_entries(
        self, tmp_path: Path
    ) -> None:
        """The unrouted `use` fallback skips entries that do not match, on two axes.

        A sibling region's own scoped import is out of reach entirely (its
        own region isn't an ancestor of the resolving one); a reachable
        scoped import that doesn't expose anything under the requested path
        contributes no members. Both must be skipped without derailing the
        entry that does resolve.
        """
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "scope A\n  import libg::*\n  import otherlib::*\n  use Geo::*\n"
                    "  def show() -> int = dist()\nend A\n\nA::show()\n"
                    "\n"
                    "scope B\n  import thirdlib::*\nend B"
                ),
                "libg": "scope Geo\n  def dist() -> int = 5\nend Geo",
                "otherlib": "def helper() -> int = 9",
                "thirdlib": "def unused() -> int = 0",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules


# ---------------------------------------------------------------------------
# Test: wildcard imports
# ---------------------------------------------------------------------------


class TestWildcardImports:
    def test_wildcard_expands_all_submodules(self, tmp_path: Path) -> None:
        """'import foo.*' exposes functions from all submodules of 'foo'."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import foo/*::*\nlet a = alpha()\nlet b = beta()",
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

    def test_wildcard_as_forms_a_member_filtered_facade(self, tmp_path: Path) -> None:
        "'import foo/* as F' makes 'F::alpha()' accessible via the facade."
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import foo/* as F\nlet a = F::alpha()",
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

    def test_repeated_identical_wildcard_alias_preserves_facade(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": ("import pkg/* as F\nimport pkg/* as F\nuse F::*\nfirst() + second()"),
                "pkg/a": "def first() -> int = 1",
                "pkg/b": "def second() -> int = 2",
            },
        )

        resolve_program(graph)

    def test_wildcard_facade_accepts_duplicate_routes_to_the_same_origin(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import pkg/* as F\nlet value = F::shared()",
                "pkg/left": "export core::{shared}",
                "pkg/right": "export core::{shared}",
                "core": "def shared() -> int = 1",
            },
        )

        result = resolve_program(graph)
        entry_program = graph.modules[ENTRY_ID].program
        shared_var = _find_varref(entry_program, "shared")
        assert shared_var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[shared_var.node_id]
        assert ref.module_id == ModuleId.from_path("core")

    @pytest.mark.parametrize(
        "modules",
        (
            {
                "entry": "import alpha as F\nimport beta as F\nuse F::*\nfirst() + second()",
                "alpha": "def first() -> int = 1",
                "beta": "def second() -> int = 2",
            },
            {
                "entry": "import alpha/* as F\nimport beta/* as F\nuse F::*\nfirst() + second()",
                "alpha/one": "def first() -> int = 1",
                "beta/two": "def second() -> int = 2",
            },
            {
                "entry": "import pkg/* as F\nimport other::{F}\nuse F::*\nfirst() + third()",
                "pkg/alpha": "def first() -> int = 1",
                "pkg/beta": "def second() -> int = 2",
                "other": "scope F\n  def third() -> int = 3\nend F",
            },
        ),
        ids=("aliases", "wildcard-aliases", "wildcard-alias-and-scope"),
    )
    def test_every_scope_a_use_target_spells_combines(
        self, tmp_path: Path, modules: dict[str, str]
    ) -> None:
        check_program(resolve_program(_make_graph_from_files(tmp_path, modules)), base_caps())

    def test_own_type_beats_a_same_named_import_route(self, tmp_path: Path) -> None:
        """``Color::Red`` is the own enum member, not the aliased route's ``Red``."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import lib as Color\nenum Color | Red\nlet x: Color = Color::Red\nx",
                "lib": "def Red() -> int = 1",
            },
        )
        check_program(resolve_program(graph), base_caps())

    def test_route_supplies_a_path_a_same_named_own_type_lacks(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import lib as Color\nrecord Color\nlet y: int = Color::make()\ny",
                "lib": "def make() -> int = 1",
            },
        )
        check_program(resolve_program(graph), base_caps())

    def test_own_root_anchor_misses_an_unknown_member_of_an_own_type(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import lib\nenum E | value\n::E::missing",
                "lib": "def ignored() -> int = 1",
            },
        )
        with pytest.raises(UnknownMemberError):
            resolve_program(graph)

    def test_wildcard_compatible_overlap_idempotent(self, tmp_path: Path) -> None:
        """Two wildcards that expose same QName from same module are idempotent (no error)."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import foo/*::*\nimport foo/alpha::*\nlet x = alpha()",
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

    def test_overlapping_open_imports_dedupe_member_constructor_candidates(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import foo/*::*\nimport foo/alpha::*\nlet x = Ready()",
                "foo/alpha": "enum State\n  | Ready",
            },
        )
        result = resolve_program(graph)
        entry = result.modules[ENTRY_ID].resolved
        (candidate,) = entry.constructor_candidates["Ready"]
        assert candidate.owner_path == ("State",)
        assert candidate.owner_name == "Ready"

    def test_wildcard_conflicting_overlap_clashes_on_use(self, tmp_path: Path) -> None:
        """Two wildcards expose same bare name from different modules → clash on use."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import foo/*::*\nimport bar/*::*\nlet x = common()",
                "foo/sub": "def common() -> int = 1",
                "bar/sub": "def common() -> int = 2",
            },
        )
        with pytest.raises(AmbiguousQualificationError):
            resolve_program(graph)


# ---------------------------------------------------------------------------
# Test: cross-file mutual recursion
# ---------------------------------------------------------------------------


class TestCrossFileMutualRecursion:
    def test_a_calls_b_resolves(self, tmp_path: Path) -> None:
        """Module A can call a member made bare by a one-way glob import."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import modA::*\ncallA()",
                "modA": "import modB::*\ndef callA() -> int = callB()",
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
                "entry": "import modA::*\nimport modB::*\nfuncA()",
                "modA": "import modB::*\ndef funcA() -> int = funcB()",
                "modB": "import modA::*\ndef funcB() -> int = funcA()",
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
                "entry": "import modA::*\nmake()",
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
                "entry": "import mylib::*\nlet y = foo()",
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
                "entry": "import mylib::*\n()",
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
        main = entry_resolved.program.body.items[-1]
        assert isinstance(main, FuncDef)
        assignment = main.body.items[-1]
        assert isinstance(assignment, AssignStmt)
        binding = entry_resolved.resolution[assignment.node_id]
        assert binding.module_id == ENTRY_ID
        assert binding.mutable is True

    def test_assign_stmt_in_non_entry_errors(self, tmp_path: Path) -> None:
        """An assignment statement in a non-entry module is an error."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::*\n()",
                "mylib": "def setup() -> unit = ()\nx := 2",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    @pytest.mark.parametrize(
        "target",
        [
            pytest.param("lib::Color::Red", id="declaring-enum"),
            pytest.param("lib::Shade::Red", id="alias-owner"),
            pytest.param("lib::Red", id="module-surface"),
        ],
    )
    def test_qualified_assign_to_an_imported_enum_member_targets_its_constructor(
        self, tmp_path: Path, target: str
    ) -> None:
        """An imported enum member is an immutable constructor target, however selected."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": f"import lib\n{target} := lib::Color::Blue",
                "lib": "enum Color =\n  | Red\n  | Blue\ntype Shade = Color",
            },
        )
        resolved = resolve_program(graph).modules[ENTRY_ID].resolved
        assigns: list[AssignStmt] = []
        walk(
            resolved.program,
            lambda node: assigns.append(node) if isinstance(node, AssignStmt) else None,
        )

        ref = resolved.resolution[assigns[0].node_id]

        assert (ref.kind, ref.mutable) == (BinderKind.constructor_binding, False)


# ---------------------------------------------------------------------------
# Test: type declarations in modules (RecordDef/EnumDef/TypeAlias)
# ---------------------------------------------------------------------------


class TestTypeDeclarationsInModules:
    def test_record_in_module_is_exported(self, tmp_path: Path) -> None:
        """A record declaration in a non-entry module is in exports."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::*\n()",
                "mylib": "record Point\n  x: int\n  y: int",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert "Point" in result.modules[mylib_id].exports

    def test_enum_in_module_in_pre_pass_tables(self, tmp_path: Path) -> None:
        """An enum in a module is in all_public_types."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::*\n()",
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
                "entry": "import mylib::*\n()",
                "mylib": "type MyInt = int",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert "MyInt" in result.modules[mylib_id].exports
        assert (mylib_id, "MyInt") in result.all_public_types

    def test_brace_tail_preserves_full_qualified_module_surface(self, tmp_path: Path) -> None:
        """A brace tail limits bare names, not qualified module members."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import mylib::{foo}\nlet selected = mylib::foo()\nlet other = mylib::bar()"
                ),
                "mylib": "def foo() -> int = 1\ndef bar() -> int = 2",
            },
        )

        result = resolve_program(graph)

        entry = result.modules[ENTRY_ID].resolved
        for name in ("foo", "bar"):
            var = _find_varref(entry.program, name)
            assert var is not None
            ref = entry.resolution[var.node_id]
            assert ref.name == name
            assert ref.module_id == ModuleId.from_path("mylib")

    def test_nonconstructible_alias_name_collides_with_existing_binding(
        self, tmp_path: Path
    ) -> None:
        """A nominal alias of an enum still occupies its name for collision purposes.

        ``Palette`` has no constructor candidate (``Color`` is an enum), but
        scope still reserves the name for it, so a ``def`` of the same name
        is a duplicate declaration exactly as it would be for a constructible
        alias.
        """
        modules = {
            "entry": ("def Palette() -> int = 1\nenum Color\n  | Red\ntype Palette = Color\n()"),
        }
        assert _rejection(tmp_path, modules, "entry") == (
            DuplicateDeclarationError,
            "type Palette = Color",
        )

    def test_nonconstructible_alias_yields_to_repl_session_binding(self, tmp_path: Path) -> None:
        """A prior REPL session binding keeps its value beside a same-named nonconstructible alias.

        The alias lives only in the type namespace, so the session binding
        keeps its expression-position meaning.
        """
        prior_resolved = resolve_repl_entry("let Palette = 42\nPalette")
        session_scope = prior_resolved.root_scope

        graph = _make_graph_from_files(
            tmp_path,
            {"entry": "enum Color\n  | Red\n\ntype Palette = Color\n\nprint(Color::Red)"},
        )
        result = resolve_program(graph, entry_repl_session_scope=session_scope)
        assert ENTRY_ID in result.modules


# ---------------------------------------------------------------------------
# Test: method receiver naming a type imported from another module
# ---------------------------------------------------------------------------


class TestMethodOrphanRule:
    def test_self_on_a_type_imported_from_another_module_resolves_its_owner(
        self, tmp_path: Path
    ) -> None:
        """A bare imported record owns a method declared by the importing module."""
        shapes_id = ModuleId.from_path("shapes")
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import shapes::*\n\n"
                    "def Point::tag(self) -> int = self.x\n\n"
                    "print(Point(x = 1).tag())"
                ),
                "shapes": "record Point\n  x: int",
            },
        )

        resolved = resolve_program(graph).modules[ENTRY_ID].resolved

        assert resolved.method_declarations == {
            (ENTRY_ID, ("Point",), "tag"): ReceiverOwner(shapes_id, ("Point",)),
        }

    def test_orphan_receiver_covers_enum_members_exceptions_and_generic_records(
        self, tmp_path: Path
    ) -> None:
        """Foreign nominal owners retain their declaring module and full type path."""
        shapes_id = ModuleId.from_path("shapes")
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import shapes::*\n\n"
                    "def Tree::Node::extract[E](self) -> E = self.value\n"
                    "def Failure::tag(self) -> int = 1\n"
                    "def Box::get[E](self) -> E = self.value"
                ),
                "shapes": (
                    "enum Tree[E]\n  | Node(value: E)\n\n"
                    "exception Failure\n\n"
                    "record Box[E]\n  value: E"
                ),
            },
        )

        resolved = resolve_program(graph).modules[ENTRY_ID].resolved

        assert resolved.method_declarations == {
            (ENTRY_ID, ("Tree", "Node"), "extract"): ReceiverOwner(shapes_id, ("Tree", "Node")),
            (ENTRY_ID, ("Failure",), "tag"): ReceiverOwner(shapes_id, ("Failure",)),
            (ENTRY_ID, ("Box",), "get"): ReceiverOwner(shapes_id, ("Box",)),
        }

    def test_scoped_imported_receiver_uses_its_exposed_path(self, tmp_path: Path) -> None:
        shapes_id = ModuleId.from_path("shapes")
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import shapes::*\n\ndef Geo::Point::tag(self) -> int = self.x",
                "shapes": "scope Geo\n  record Point\n    x: int\nend Geo",
            },
        )

        resolved = resolve_program(graph).modules[ENTRY_ID].resolved

        assert resolved.method_declarations == {
            (ENTRY_ID, ("Geo", "Point"), "tag"): ReceiverOwner(shapes_id, ("Geo", "Point")),
        }

        invalid = {
            "entry": "import shapes::*\n\ndef Point::tag(self) -> int = self.x",
            "shapes": "scope Geo\n  record Point\n    x: int\nend Geo",
        }
        assert _rejection(tmp_path, invalid, "entry") == (AglScopeError, "self")

    def test_self_on_a_region_scoped_glob_imported_type_resolves_its_owner(
        self, tmp_path: Path
    ) -> None:
        """A scoped bare import reaches method declarations in its region."""
        shapes_id = ModuleId.from_path("shapes")
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "scope A\n  import shapes::*\n\n"
                    "  def Tree::Node::extract[E](self) -> E = self.value\nend A"
                ),
                "shapes": "enum Tree[E]\n  | Node(value: E)",
            },
        )

        resolved = resolve_program(graph).modules[ENTRY_ID].resolved

        assert resolved.method_declarations == {
            (ENTRY_ID, ("A", "Tree", "Node"), "extract"): ReceiverOwner(
                shapes_id, ("Tree", "Node")
            ),
        }

    def test_use_exposes_an_imported_receiver(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import shapes\nuse shapes::Geo::*\n\ndef Point::tag(self) -> int = self.x"
                ),
                "shapes": "scope Geo\n  record Point\n    x: int\nend Geo",
            },
        )

        resolved = resolve_program(graph).modules[ENTRY_ID].resolved

        assert resolved.method_declarations == {
            (ENTRY_ID, ("Point",), "tag"): ReceiverOwner(
                ModuleId.from_path("shapes"), ("Geo", "Point")
            )
        }

    def test_nearest_scoped_import_wins_for_receiver(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import far::*\n\nscope A\n  import near::*\n\n"
                    "  def Point::tag(self) -> int = 1\nend A"
                ),
                "far": "record Point",
                "near": "record Point",
            },
        )

        resolved = resolve_program(graph).modules[ENTRY_ID].resolved

        assert resolved.method_declarations == {
            (ENTRY_ID, ("A", "Point"), "tag"): ReceiverOwner(ModuleId.from_path("near"), ("Point",))
        }

    def test_import_inside_receiver_scope_does_not_supply_its_owner(self, tmp_path: Path) -> None:
        """The import contributes ``Point::Point``; no type is declared at ``Point``."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "scope Point\n  import shapes::*\n\n  def tag(self) -> int = self.x\nend Point"
                ),
                "shapes": "record Point\n  x: int",
            },
        )

        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_exact_bare_record_ignores_nested_alias(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import owners::*\nimport aliases::*\ndef Point::tag(self) -> int = 1",
                "owners": "record Point",
                "aliases": "scope Geo\n  type Point = int\nend Geo",
            },
        )

        resolved = resolve_program(graph).modules[ENTRY_ID].resolved

        assert resolved.method_declarations == {
            (ENTRY_ID, ("Point",), "tag"): ReceiverOwner(ModuleId.from_path("owners"), ("Point",))
        }

    def test_exact_bare_record_wins_over_enum_member(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import shapes::*\ndef Node::tag(self) -> int = 1",
                "shapes": "record Node\nenum Tree = Node",
            },
        )

        resolved = resolve_program(graph).modules[ENTRY_ID].resolved

        assert resolved.method_declarations == {
            (ENTRY_ID, ("Node",), "tag"): ReceiverOwner(ModuleId.from_path("shapes"), ("Node",))
        }

    def test_hidden_enum_member_cannot_own_method(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import shapes::* hiding Tree::Node\ndef Tree::Node::tag(self) -> int = 1"
                ),
                "shapes": "enum Tree = Node",
            },
        )

        with pytest.raises(HiddenMemberError):
            resolve_program(graph)

    def test_a_receiver_takes_only_the_type_declared_at_its_whole_path(
        self, tmp_path: Path
    ) -> None:
        """``scope A / scope B / def Point::tag`` declares ``A::B::Point::tag``: no type there."""
        modules = {
            "entry": (
                "record Point\n\n"
                "scope A\n"
                "  record Point\n\n"
                "  scope B\n"
                "    def Point::tag(self) -> int = 1\n"
                "  end B\n"
                "end A"
            ),
        }
        assert _rejection(tmp_path, modules, "entry") == (AglScopeError, "self")

    def test_region_path_names_the_receiver_type_declared_in_an_enclosing_region(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "record Point\n\n"
                    "scope A\n"
                    "  record Point\n\n"
                    "  scope Point\n"
                    "    def tag(self) -> int = 1\n"
                    "  end Point\n"
                    "end A"
                ),
            },
        )

        resolved = resolve_program(graph).modules[ENTRY_ID].resolved

        assert resolved.method_declarations == {
            (ENTRY_ID, ("A", "Point"), "tag"): ReceiverOwner(ENTRY_ID, ("A", "Point")),
        }

    def test_root_local_type_wins_over_a_bare_import(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import shapes::*\n\nrecord Point\n  x: int\n\n"
                    "def Point::tag(self) -> int = self.x"
                ),
                "shapes": "record Point\n  x: int",
            },
        )

        resolved = resolve_program(graph).modules[ENTRY_ID].resolved

        assert resolved.method_declarations == {
            (ENTRY_ID, ("Point",), "tag"): ReceiverOwner(ENTRY_ID, ("Point",)),
        }

    def test_selected_enum_member_path_owns_an_orphan_method(self, tmp_path: Path) -> None:
        """A selected member path is bare as that path, never as its terminal name."""
        shapes_id = ModuleId.from_path("shapes")
        files = {
            "entry": (
                "import shapes::{Tree::Node}\n\ndef Tree::Node::extract[E](self) -> E = self.value"
            ),
            "shapes": "enum Tree[E]\n  | Node(value: E)",
        }
        resolved = resolve_program(_make_graph_from_files(tmp_path, files)).modules[ENTRY_ID]

        assert resolved.resolved.method_declarations == {
            (ENTRY_ID, ("Tree", "Node"), "extract"): ReceiverOwner(shapes_id, ("Tree", "Node")),
        }

        files["entry"] = (
            "import shapes::{Tree::Node}\n\ndef Node::extract[E](self) -> E = self.value"
        )
        with pytest.raises(AglScopeError):
            resolve_program(_make_graph_from_files(tmp_path / "terminal", files))

    def test_plain_scope_named_like_an_imported_type_can_declare_an_orphan(
        self, tmp_path: Path
    ) -> None:
        """A plain scope does not mask a bare imported receiver name."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import shapes::*\n\nscope Point\n  def tag(self) -> int = self.x\nend Point"
                ),
                "shapes": "record Point\n  x: int",
            },
        )

        resolved = resolve_program(graph).modules[ENTRY_ID].resolved

        assert resolved.method_declarations == {
            (ENTRY_ID, ("Point",), "tag"): ReceiverOwner(ModuleId.from_path("shapes"), ("Point",)),
        }

    def test_qualified_only_import_does_not_supply_an_orphan_receiver(self, tmp_path: Path) -> None:
        modules = {
            "entry": "import shapes\ndef Point::tag(self) -> int = 1",
            "shapes": "record Point",
        }
        assert _rejection(tmp_path, modules, "entry") == (AglScopeError, "self")

    def test_ambiguous_bare_orphan_receiver_is_rejected(self, tmp_path: Path) -> None:
        modules = {
            "entry": "import first::*\nimport second::*\ndef Point::tag(self) -> int = 1",
            "first": "record Point",
            "second": "record Point",
        }
        assert _rejection(tmp_path, modules, "entry") == (AmbiguousQualificationError, "self")

    def test_renamed_orphan_receiver_collision_is_ambiguous(self, tmp_path: Path) -> None:
        modules = {
            "entry": (
                "import first::{Original as Point}\n"
                "import second::{Point}\n"
                "def Point::tag(self) -> int = 1"
            ),
            "first": "record Original",
            "second": "record Point",
        }
        assert _rejection(tmp_path, modules, "entry") == (AmbiguousQualificationError, "self")

    def test_foreign_renaming_alias_receiver_is_rejected(self, tmp_path: Path) -> None:
        """A receiver names its type directly, never through an imported alias."""
        modules = {
            "entry": "import shapes::*\ndef Point::tag(self) -> int = 1",
            "shapes": "record Actual\ntype Point = Actual",
        }
        assert _rejection(tmp_path, modules, "entry") == (AglScopeError, "self")

    def test_local_nested_alias_is_rejected_as_a_multi_segment_receiver(
        self, tmp_path: Path
    ) -> None:
        """Nothing is declared beneath an alias the module declares in its own type's scope."""
        modules = {
            "entry": (
                "record Outer\n"
                "  x: int\n"
                "\n"
                "scope Outer\n"
                "  type Inner = int\n"
                "end Outer\n"
                "\n"
                "def Outer::Inner::bad(self) -> int = 1"
            ),
        }
        assert _rejection(tmp_path, modules, "entry") == (AglScopeError, "Inner")

    def test_foreign_nested_alias_is_rejected_as_a_multi_segment_receiver(
        self, tmp_path: Path
    ) -> None:
        """A used nested type alias, reached without a local declaration, still rejects."""
        modules = {
            "entry": "import shapes\nuse shapes::*\ndef Outer::Inner::bad(self) -> int = 1",
            "shapes": ("record Outer\n  x: int\n\nscope Outer\n  type Inner = int\nend Outer"),
        }
        assert _rejection(tmp_path, modules, "entry") == (AglScopeError, "self")

    @pytest.mark.parametrize(
        ("module_source", "receiver"),
        (
            ("def Point() -> int = 1", "Point"),
            ("enum Tree = Node\ndef Tree::Point() -> int = 1", "Point"),
            ("enum Tree = Node\ndef Tree::Point() -> int = 1", "Tree::Point"),
            ("enum Tree = Node", "Tree::Missing"),
            ("record Point", "Point::Nested"),
        ),
    )
    def test_invalid_foreign_receiver_paths_are_not_owners(
        self, tmp_path: Path, module_source: str, receiver: str
    ) -> None:
        """Only imported nominal declarations and enum members can own methods."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": f"import shapes::*\n\ndef {receiver}::tag(self) -> int = 1",
                "shapes": module_source,
            },
        )

        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_self_with_no_enclosing_scope_at_all_reports_the_generic_rule(
        self, tmp_path: Path
    ) -> None:
        """A root function with a `self` parameter reports the generic scope rule."""
        modules = {"entry": "def helper(self) -> int = 1"}
        assert _rejection(tmp_path, modules, "entry") == (AglScopeError, "self")

    def test_orphan_check_ignores_a_scoped_import_from_an_unrelated_region(
        self, tmp_path: Path
    ) -> None:
        """A scoped import supplies receiver types only in its region and descendants."""
        modules = {
            "entry": (
                "scope A\n  import shapes::*\nend A\n\n"
                "scope B\n  def Point::tag(self) -> int = self.x\nend B"
            ),
            "shapes": "record Point\n  x: int",
        }
        assert _rejection(tmp_path, modules, "entry") == (AglScopeError, "self")


# ---------------------------------------------------------------------------
# Test: ::name self-reference in non-entry module
# ---------------------------------------------------------------------------


class TestSelfReferenceInNonEntryModule:
    def test_self_ref_to_nonexistent_name_errors(self, tmp_path: Path) -> None:
        """'::nonexistent' in a non-entry module errors."""
        modules = {
            "entry": "import mylib::*\n()",
            "mylib": "def foo() -> int = ::noname()",
        }
        assert _rejection(tmp_path, modules, "mylib") == (UnknownMemberError, "::noname")


# ---------------------------------------------------------------------------
# Test: call with module qualifier (modA.funcA() syntax)
# ---------------------------------------------------------------------------


class TestModuleQualifiedCall:
    def test_module_double_colon_call_resolves(self, tmp_path: Path) -> None:
        """'modA::callA()' — double-colon qualified call resolves the function."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import modA::*\nmodA::callA()",
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
# Qualified constructor references and field access across modules
# ---------------------------------------------------------------------------


class TestQualifiedConstructorReferences:
    def test_self_ref_field_access_in_non_entry_module(self, tmp_path: Path) -> None:
        """A non-entry module can resolve a self-qualified constructor ref."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "mylib": (
                    "enum Color\n  | Red\n  | Blue\ndef getDefault() -> Color = ::Color::Red"
                ),
                "entry": "import mylib::*\nmylib::getDefault()",
            },
        )
        result = resolve_program(graph)
        assert ENTRY_ID in result.modules
        # The self-ref ::Color::Red within mylib resolves to the Red member.
        mylib_id = ModuleId.from_path("mylib")
        mylib_resolved = result.modules[mylib_id].resolved
        assert [ref.owner_name for ref in mylib_resolved.constructor_refs.values()] == ["Red"]

    def test_unrecognized_qualifier_in_field_access_errors(self, tmp_path: Path) -> None:
        """An unknown module qualifier in a constructor ref is rejected."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "mylib": "enum Color\n  | Red\n  | Blue",
                "entry": "import mylib::*\nlet x = notimported::Color::Red\nx",
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
                "entry": "import mylib::*\nlet x = mylib::NonExistent::Red\nx",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_non_constructible_qualified_type_owns_no_member(self, tmp_path: Path) -> None:
        """A structural alias owns no member, as a local one does not."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "mylib": "type Alias = int",
                "entry": "import mylib::*\nlet x = mylib::Alias::Ctor\nx",
            },
        )
        with pytest.raises(UnknownMemberError):
            resolve_program(graph)

    def test_qualified_non_type_owns_no_constructor(self, tmp_path: Path) -> None:
        """A route member that is not a type cannot qualify a constructor."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "mylib": "def compute() -> int = 1",
                "entry": "import mylib::*\nlet x = mylib::compute::Ctor\nx",
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_non_type_exported_name_in_field_access_falls_through(self, tmp_path: Path) -> None:
        """A qualified name that exports a function resolves as a value, not a type.

        ``mylib::compute.value`` names a function, so the qualifier resolves to the
        imported binding and ``.value`` stays an ordinary field access.
        """
        graph = _make_graph_from_files(
            tmp_path,
            {
                "mylib": (
                    "record Result\n  value: int\ndef compute(n: int) -> Result = Result(value = n)"
                ),
                "entry": (
                    "import mylib::*\n"
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
                "entry": "import mylib::*\n()",
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
                "entry": "import mylib::*\n()",
                "mylib": "exception MyErr\n  msg: text",
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
                "entry": "import mylib::*\n()",
                "mylib": "exception MyErr\n  msg: text",
            },
        )
        result = resolve_program(graph)
        mylib_id = ModuleId.from_path("mylib")
        assert (mylib_id, "MyErr") in result.all_public_types

    def test_exception_constructor_candidate_available(self, tmp_path: Path) -> None:
        """A glob-imported exception exposes its name as a constructor candidate."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": 'import mylib::*\nMyErr(msg = "oops")',
                "mylib": "exception MyErr\n  msg: text",
            },
        )
        result = resolve_program(graph)
        # The entry resolved correctly (no scope error raised).
        assert ENTRY_ID in result.modules
        entry_resolved = result.modules[ENTRY_ID].resolved
        # MyErr is a constructor candidate resolved from the glob import.
        assert "MyErr" in entry_resolved.constructor_candidates

    @pytest.mark.parametrize(
        "entry",
        (
            "import mylib::{A::E as E}\nlet value: E = X",
            "scope Local\n  import mylib::{A::E as E}\n  let value: E = X\nend Local",
        ),
    )
    def test_selective_enum_import_keeps_variant_that_collides_with_unselected_exception(
        self, tmp_path: Path, entry: str
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": entry,
                "mylib": "scope A\n  enum E | X | Y\nend A\n\nexception X extends Exception",
            },
        )

        assert ENTRY_ID in resolve_program(graph).modules

    def test_scoped_exception_suppresses_sibling_enum_variant_in_imported_use(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": ('import library\nuse library::A::*\nlet error = X(message = "boom")'),
                "library": ("scope A\n  enum E | X | Y\n  exception X extends Exception\nend A"),
            },
        )

        check_program(resolve_program(graph), base_caps())

    def test_scoped_exception_suppresses_sibling_enum_variant_in_import_tail(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    'import library::{A::E as E, A::X as X}\nlet error = X(message = "boom")'
                ),
                "library": ("scope A\n  enum E | X | Y\n  exception X extends Exception\nend A"),
            },
        )

        check_program(resolve_program(graph), base_caps())

    def test_exception_skip_branch_enum_variant_collision(self, tmp_path: Path) -> None:
        """Exception-skip branch: an enum variant whose name collides with a public
        ExceptionDef in the same module is skipped as a constructor candidate.

        Exercises graph.py lines ~154-157: when iterating EnumDef variants, any variant
        whose name matches a public ExceptionDef in all_public_types is skipped so the
        exception wins as the constructor.
        """
        # The enum "Status" has a variant named "Conflict".
        # The module also has a public exception "Conflict".
        # When resolving the glob import, "Conflict" (enum variant) must be skipped
        # and only the ExceptionDef "Conflict" is in constructor_candidates.
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import mylib::*\n()",
                "mylib": ("enum Status\n  | Ok\n  | Conflict\nexception Conflict\n  msg: text\n"),
            },
        )
        result = resolve_program(graph)
        entry_resolved = result.modules[ENTRY_ID].resolved
        # "Conflict" must exist in constructor_candidates and must refer to the
        # ExceptionDef, NOT to the enum member record named "Conflict".
        assert "Conflict" in entry_resolved.constructor_candidates
        candidates = entry_resolved.constructor_candidates["Conflict"]
        assert all(c.owner_name == "Conflict" and c.owner_path == () for c in candidates), (
            "Enum member 'Conflict' should have been skipped; only the ExceptionDef "
            f"candidate should remain. Got: {candidates}"
        )


# ---------------------------------------------------------------------------
# Test: reachable declaration identities
# ---------------------------------------------------------------------------


class TestReachableDeclarations:
    def test_locals_and_import_routes_publish_declaration_identities(self, tmp_path: Path) -> None:
        metrics_id = ModuleId.from_path("metrics")
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import metrics\n"
                    "\n"
                    "scope Region\n"
                    "  import metrics::*\n"
                    "  def marker() -> int = 1\n"
                    "end Region\n"
                    "\n"
                    "record Local\n"
                    "def Local::show(self) -> int = 1"
                ),
                "metrics": "scope Point\n  def norm() -> int = 1\nend Point",
            },
            default_stdlib=False,
        )

        reachable = resolve_program(graph).modules[ENTRY_ID].resolved.reachable_declarations

        assert reachable == {
            (ENTRY_ID, (), "Local"),
            (ENTRY_ID, ("Local",), "show"),
            (ENTRY_ID, ("Region",), "marker"),
            (metrics_id, ("Point",), "norm"),
        }

    def test_region_scoped_import_is_module_wide(self, tmp_path: Path) -> None:
        metrics_id = ModuleId.from_path("metrics")
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "scope Region\n  import metrics::*\nend Region\n\n()",
                "metrics": "scope Point\n  def norm() -> int = 1\nend Point",
            },
            default_stdlib=False,
        )

        reachable = resolve_program(graph).modules[ENTRY_ID].resolved.reachable_declarations

        assert reachable == {(metrics_id, ("Point",), "norm")}

    @pytest.mark.parametrize("import_decl", ("import metrics::{Point::norm}", "import metrics::*"))
    def test_import_tails_and_wildcards_publish_declaration_identities(
        self, tmp_path: Path, import_decl: str
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": f"{import_decl}\n()",
                "metrics": "scope Point\n  def norm() -> int = 1\nend Point",
            },
            default_stdlib=False,
        )

        reachable = resolve_program(graph).modules[ENTRY_ID].resolved.reachable_declarations

        assert (ModuleId.from_path("metrics"), ("Point",), "norm") in reachable

    def test_facade_reexport_preserves_the_declaration_origin(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\n()",
                "facade": "export metrics::{Point::norm}",
                "metrics": "scope Point\n  def norm() -> int = 1\nend Point",
            },
            default_stdlib=False,
        )

        reachable = resolve_program(graph).modules[ENTRY_ID].resolved.reachable_declarations

        assert (ModuleId.from_path("metrics"), ("Point",), "norm") in reachable

    def test_hiding_removes_a_reachable_declaration(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import metrics hiding Point::norm\n()",
                "metrics": "scope Point\n  def norm() -> int = 1\nend Point",
            },
            default_stdlib=False,
        )

        reachable = resolve_program(graph).modules[ENTRY_ID].resolved.reachable_declarations

        assert (ModuleId.from_path("metrics"), ("Point",), "norm") not in reachable

    def test_use_adds_no_declaration_identities(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {"entry": "use Point::*\n\nscope Point\n  def norm() -> int = 1\nend Point\n\n()"},
            default_stdlib=False,
        )

        reachable = resolve_program(graph).modules[ENTRY_ID].resolved.reachable_declarations

        assert reachable == {(ENTRY_ID, ("Point",), "norm")}

    def test_implicit_prelude_import_publishes_stdlib_declarations(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(tmp_path, {"entry": "()"})

        reachable = resolve_program(graph).modules[ENTRY_ID].resolved.reachable_declarations

        assert (ModuleId.from_path("std/text"), ("text",), "trim") in reachable

    def test_retained_repl_declarations_persist(self, tmp_path: Path) -> None:
        prior = resolve_repl_entry("def saved() -> int = 1\nsaved()", default_stdlib=False)
        graph = _make_graph_from_files(tmp_path, {"entry": "()"}, default_stdlib=False)
        current = (
            resolve_program(graph, entry_repl_session_scope=prior.root_scope)
            .modules[ENTRY_ID]
            .resolved
        )

        assert (ENTRY_ID, (), "saved") in current.reachable_declarations

    def test_retained_repl_import_persists(self, tmp_path: Path) -> None:
        (tmp_path / "metrics.agl").write_text(
            "scope Point\n  def norm() -> int = 1\nend Point\n", encoding="utf-8"
        )
        session = ReplSession(cwd=tmp_path, default_stdlib=False)
        assert not session.open()
        assert session.eval_entry("import metrics").ok
        program, next_node_id = parse_program_seeded(
            "()", start_id=session._next_node_id, resolve_infix=False
        )

        checked = session._entry_pipeline.resolve_and_check_program(
            program, next_node_id, session._runtime.host_environment()
        )
        reachable = checked.modules[ENTRY_ID].resolved.reachable_declarations

        assert (ModuleId.from_path("metrics"), ("Point",), "norm") in reachable


# ---------------------------------------------------------------------------
# Test: resolve_program REPL seams — ambient_agents, entry_repl_session_scope, warnings
# ---------------------------------------------------------------------------


class TestResolveGraphReplSeams:
    def test_entry_repl_session_scope_binding_visible_in_entry(self, tmp_path: Path) -> None:
        """entry_repl_session_scope: a name pre-bound in the parent scope is visible in entry."""
        # Build a prior session that binds "x" as a let binding.
        prior_source = "let x = 42\nx"
        prior_resolved = resolve_repl_entry(prior_source)
        session_scope = prior_resolved.root_scope

        # New entry references "x" — which is only in the parent scope.
        graph = _make_graph_from_files(tmp_path, {"entry": "x"})
        result = resolve_program(graph, entry_repl_session_scope=session_scope)
        assert ENTRY_ID in result.modules

        entry_program = graph.modules[ENTRY_ID].program
        var = _find_varref(entry_program, "x")
        assert var is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[var.node_id]
        assert ref.name == "x"
        assert ref.kind == BinderKind.let_binding

    def test_entry_repl_session_scope_not_applied_to_non_entry(self, tmp_path: Path) -> None:
        """entry_repl_session_scope is only injected into the entry module, not library modules.

        ``helper`` is bound only in the prior session scope.  A non-entry module that
        references it must fail to resolve — if the parent scope leaked into library
        resolution, ``mylib`` would silently succeed.
        """
        prior_source = "let helper = 1\nhelper"
        prior_resolved = resolve_repl_entry(prior_source)
        session_scope = prior_resolved.root_scope

        # mylib references "helper", which exists ONLY in the entry's parent scope.
        library = "def foo() -> int = helper"
        graph = _make_graph_from_files(tmp_path, {"entry": "import mylib::*\n()", "mylib": library})
        with pytest.raises(AglScopeError) as exc_info:
            resolve_program(graph, entry_repl_session_scope=session_scope)
        assert type(exc_info.value) is AglScopeError
        assert span_text(library, exc_info.value.span) == "helper"


# ---------------------------------------------------------------------------
# Duplicate declarations
# ---------------------------------------------------------------------------


class TestDuplicateDeclarations:
    """A module declaring one name twice at one path is rejected where the second one is."""

    @pytest.mark.parametrize(
        ("modules", "duplicate"),
        [
            pytest.param({"entry": "def S() -> int = 1\n\nscope S\nend S\n\n()"}, "S", id="scope"),
            pytest.param({"entry": "record P\n  n: int\nrecord P\n  t: text\n()"}, "P", id="type"),
            pytest.param(
                {"entry": "def f() -> int = 1\ndef f() -> int = 2\n()"}, "f", id="function"
            ),
            pytest.param(
                {"entry": "def Signal::ready() -> int = 1\nenum Signal\n  | ready\n()"},
                "ready",
                id="enum-member",
            ),
            pytest.param(
                {"entry": "scope S\n  def x() -> int = 1\n  let x = 2\nend S\n\n()"},
                "x",
                id="scoped-binding",
            ),
            pytest.param(
                {
                    "entry": "import mylib::*\n()",
                    "mylib": "def x() -> int = 1\nbuiltin var x: bool",
                },
                "x",
                id="builtin-var",
            ),
            pytest.param({"entry": "let a = 1\nlet a = 2\n()"}, "a", id="binder"),
            pytest.param({"entry": "let g = fn(x: int, x: int) => x\n()"}, "x", id="parameter"),
        ],
    )
    def test_second_declaration_is_a_duplicate(
        self, tmp_path: Path, modules: dict[str, str], duplicate: str
    ) -> None:
        with pytest.raises(DuplicateDeclarationError) as caught:
            resolve_program(_make_graph_from_files(tmp_path, modules))

        assert type(caught.value) is DuplicateDeclarationError
        assert caught.value.name == duplicate

    @pytest.mark.parametrize(
        ("entries", "duplicate"),
        [
            pytest.param(("let P = 1", "record P\n  n: int"), "P", id="type-after-binding"),
            pytest.param(("record P\n  n: int", "let P = 1"), "P", id="binding-after-type"),
            pytest.param(("def S() -> int = 1", "scope S\nend S"), "S", id="scope-after-def"),
        ],
    )
    def test_later_repl_entry_redeclaring_a_name_as_another_kind_is_a_duplicate(
        self, tmp_path: Path, entries: tuple[str, str], duplicate: str
    ) -> None:
        session = ReplSession(cwd=tmp_path, default_stdlib=False)
        session.open()
        assert session.eval_entry(entries[0]).ok

        failure = session.eval_entry(entries[1]).failure

        assert type(failure) is DuplicateDeclarationError
        assert failure.name == duplicate


# ---------------------------------------------------------------------------
# Re-export behaviour
# ---------------------------------------------------------------------------


class TestExportDecl:
    """Tests for explicit export declarations."""

    def test_reexport_all_adds_to_exports(self, tmp_path: Path) -> None:
        """export lib — every name declared by lib appears in the facade's exports."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade::*\n()",
                "facade": "export lib",
                "lib": "def foo() -> int = 1\ndef bar() -> int = 2",
            },
        )
        result = resolve_program(graph)
        facade_id = ModuleId.from_path("facade")
        lib_id = ModuleId.from_path("lib")
        facade_exports = result.modules[facade_id].exports
        assert "foo" in facade_exports
        assert facade_exports["foo"] == (lib_id, "foo")
        assert facade_exports["bar"] == (lib_id, "bar")

    def test_reexport_all_preserves_an_empty_scope_for_consumers(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\nuse facade::Empty::*\n()",
                "facade": "export lib",
                "lib": "scope Empty\nend Empty",
            },
            default_stdlib=False,
        )

        result = resolve_program(graph)

        lib_id = ModuleId.from_path("lib")
        facade = result.modules[ModuleId.from_path("facade")]
        assert facade.scope_exports["Empty"] == frozenset({(lib_id, "Empty")})

    @pytest.mark.parametrize("tail", ("", "::*"), ids=("plain", "wildcard-tail"))
    def test_hiding_accepts_an_empty_scope_identity(self, tmp_path: Path, tail: str) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": f"import lib{tail} hiding Empty\n()",
                "lib": "scope Empty\nend Empty",
            },
            default_stdlib=False,
        )

        resolve_program(graph)

    def test_reexport_hiding_removes_an_empty_scope_identity(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\n()",
                "facade": "export lib hiding Empty",
                "lib": "scope Empty\nend Empty",
            },
            default_stdlib=False,
        )

        result = resolve_program(graph)

        assert "Empty" not in result.modules[ModuleId.from_path("facade")].scope_exports

    @pytest.mark.parametrize(
        "declaration",
        (
            "scope Public\n  def hidden() -> int = 1\nend Public",
            "def Public::hidden() -> int = 1",
        ),
        ids=("region", "shorthand"),
    )
    def test_scope_identity_survives_hiding_its_only_member(
        self, tmp_path: Path, declaration: str
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import lib hiding Public::hidden\nuse lib::Public::*\n()",
                "lib": declaration,
            },
            default_stdlib=False,
        )

        result = resolve_program(graph)

        lib_id = ModuleId.from_path("lib")
        assert result.modules[lib_id].scope_exports["Public"] == frozenset({(lib_id, "Public")})

    def test_import_hiding_removes_a_nonempty_scope_as_a_use_target(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import lib hiding Public\nuse lib::Public::*\n()",
                "lib": "scope Public\n  def member() -> int = 1\nend Public",
            },
            default_stdlib=False,
        )

        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_scope_hidden_on_one_import_route_remains_a_use_target_on_another(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import lib hiding Public\n"
                    "import lib as visible\n"
                    "use visible::Public::*\n"
                    "member()"
                ),
                "lib": "scope Public\n  def member() -> int = 1\nend Public",
            },
            default_stdlib=False,
        )

        resolve_program(graph)

    def test_reexport_origin_is_preserved_through_chain(self, tmp_path: Path) -> None:
        """Re-export is transparent: B re-exports from A, C uses B — origin is A, not B."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import b::*\nlet x = foo()",
                "b": "export a",
                "a": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)
        b_id = ModuleId.from_path("b")
        a_id = ModuleId.from_path("a")
        b_exports = result.modules[b_id].exports
        assert b_exports["foo"] == (a_id, "foo")

    def test_export_brace_tail_selects_members(self, tmp_path: Path) -> None:
        """An export brace tail selects only its listed member."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade::*\n()",
                "facade": "export lib::{foo}",
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

    def test_export_brace_tail_rename_preserves_origin(self, tmp_path: Path) -> None:
        """An export brace-tail rename preserves the selected member's origin."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade::*\n()",
                "facade": "export lib::{foo as plus}",
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

    def test_export_brace_tail_emits_every_alias_of_a_source(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\n()",
                "facade": "export lib::{foo, foo as first, foo as second}",
                "lib": "def foo() -> int = 1",
            },
        )

        facade_exports = resolve_program(graph).modules[ModuleId.from_path("facade")].exports
        origin = (ModuleId.from_path("lib"), "foo")
        assert facade_exports["foo"] == origin
        assert facade_exports["first"] == origin
        assert facade_exports["second"] == origin

    def test_overlapping_export_items_emit_each_destination(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\n()",
                "facade": "export lib::{Api as Public, Api::read as fetch}",
                "lib": "scope Api\n  def read() -> int = 1\n  def write() -> int = 2\nend Api",
            },
        )

        facade_exports = resolve_program(graph).modules[ModuleId.from_path("facade")].exports
        lib_id = ModuleId.from_path("lib")
        assert facade_exports[("Public", "read")] == (lib_id, ("Api", "read"))
        assert facade_exports[("Public", "write")] == (lib_id, ("Api", "write"))
        assert facade_exports["fetch"] == (lib_id, ("Api", "read"))

    def test_export_brace_tail_selects_and_renames_scoped_subtrees(self, tmp_path: Path) -> None:
        """Brace items select paths and re-root renamed paths at the facade."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\n()",
                "facade": "export lib::{Api::read as fetch, write}",
                "lib": (
                    "scope Api\n  def read() -> int = 1\n  def hidden() -> int = 2\nend Api\n"
                    "\n"
                    "def write() -> int = 3"
                ),
            },
        )

        result = resolve_program(graph)

        facade_exports = result.modules[ModuleId.from_path("facade")].exports
        lib_id = ModuleId.from_path("lib")
        assert facade_exports["fetch"] == (lib_id, ("Api", "read"))
        assert facade_exports["write"] == (lib_id, "write")
        assert ("Api", "read") not in facade_exports
        assert ("Api", "hidden") not in facade_exports

    def test_selective_nested_reexport_publishes_structural_parent_scopes(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\nuse facade::A::*\n()",
                "facade": "export lib::{A::B}",
                "lib": "scope A\n\n  scope B\n  end B\nend A",
            },
            default_stdlib=False,
        )

        result = resolve_program(graph)

        lib_id = ModuleId.from_path("lib")
        facade = result.modules[ModuleId.from_path("facade")]
        assert facade.scope_exports["A"] == frozenset({(lib_id, "A")})
        assert facade.scope_exports[("A", "B")] == frozenset({(lib_id, ("A", "B"))})

    def test_selective_deep_declaration_reexport_publishes_its_scope_chain(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": ("import facade\nuse facade::A::*\nuse B::*\nchosen()"),
                "facade": "export lib::{A::B::chosen}",
                "lib": (
                    "scope A\n\n  scope B\n    def chosen() -> int = 1\n"
                    "    def hidden() -> int = 2\n  end B\nend A"
                ),
            },
            default_stdlib=False,
        )

        result = resolve_program(graph)

        facade = result.modules[ModuleId.from_path("facade")]
        assert set(facade.scope_exports) == {"A", ("A", "B")}
        assert set(facade.exports) == {("A", "B", "chosen")}

    def test_selective_nested_reexport_scope_collides_with_local_ordinary_declaration(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\n()",
                "facade": "export lib::{A::B}\ndef A() -> int = 1",
                "lib": "scope A\n\n  scope B\n  end B\nend A",
            },
            default_stdlib=False,
        )

        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_selective_rename_and_region_reroot_only_publish_destination_scope_paths(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": ("import facade\nuse facade::Outer::*\nuse Public::*\nchosen()"),
                "facade": (
                    "scope Outer\n"
                    "  export lib::{A::B as Public, A::B::chosen as selected}\n"
                    "end Outer"
                ),
                "lib": "scope A\n\n  scope B\n    def chosen() -> int = 1\n  end B\nend A",
            },
            default_stdlib=False,
        )

        result = resolve_program(graph)

        facade_id = ModuleId.from_path("facade")
        facade = result.modules[facade_id]
        assert set(facade.scope_exports) == {"Outer", ("Outer", "Public")}
        assert set(facade.exports) == {
            ("Outer", "Public", "chosen"),
            ("Outer", "selected"),
        }

    def test_hiding_reexport(self, tmp_path: Path) -> None:
        """export lib hiding secret — all except 'secret' are re-exported."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade::*\n()",
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
                "entry": "import facade::*\n()",
                "facade": "import lib::*\ndef local() -> int = 1",
                "lib": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)
        facade_id = ModuleId.from_path("facade")
        facade_exports = result.modules[facade_id].exports
        assert "foo" not in facade_exports
        assert "local" in facade_exports

    def test_nonexpanding_reexport_cycle_converges(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import a\n()",
                "a": "export c\nexport b",
                "b": "export a",
                "c": "def value() -> int = 1",
            },
        )

        result = resolve_program(graph)

        assert result.modules[ModuleId.from_path("b")].exports["value"] == (
            ModuleId.from_path("c"),
            "value",
        )

    def test_an_imported_alias_cycle_clashes_with_another_import_of_its_name(
        self, tmp_path: Path
    ) -> None:
        modules = {
            "entry": "import m::*\nimport c::*\ntype C = A\n()",
            "m": "record A\n  x: int",
            "c": "type A = B\ntype B = A",
        }

        assert _rejection(tmp_path, modules, "entry") == (AmbiguousQualificationError, "A")

    @pytest.mark.parametrize("export", ["export m::{Id::x}", "export m hiding Id::x"])
    def test_an_export_item_beneath_an_alias_standing_for_its_parameter_names_nothing(
        self, tmp_path: Path, export: str
    ) -> None:
        modules = {"entry": "import ex::*\n()", "m": "type Id[T] = T", "ex": f"import m\n{export}"}

        assert _rejection(tmp_path, modules, "ex") == (UnknownMemberError, export)

    def test_reexport_cycle_keeps_a_hiding_of_a_module_outside_it(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import a\n()",
                "a": "export c hiding secret\nexport b",
                "b": "export a",
                "c": "def value() -> int = 1\ndef secret() -> int = 2",
            },
        )

        exports = resolve_program(graph).modules[ModuleId.from_path("b")].exports

        assert "value" in exports
        assert "secret" not in exports

    def test_cyclic_scoped_reexports_are_rejected_without_hanging(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import a\n()",
                "a": "export c\n\nscope Loop\n  export b\nend Loop",
                "b": "scope Loop\n  export a\nend Loop",
                "c": "def resource() -> int = 1",
            },
        )

        with (
            fail_if_slow("AgL re-export resolution did not terminate"),
            pytest.raises(AglScopeError),
        ):
            resolve_program(graph)

    @pytest.mark.parametrize(
        ("b_source", "query"),
        [
            pytest.param(
                "import a::*\ntype G = Base\n\ndef Base::h() -> int = 2\n", "b::G::h()", id="alias"
            ),
            pytest.param(
                "import a::*\ntype G = Base\n\ndef Base::h() -> int = 2\n",
                "b::Base::h()",
                id="target",
            ),
            # A routed spelling written in a declaration is a scope of the module's own.
            pytest.param("import a\ndef a::Base::h() -> int = 2\n", "b::a::Base::h()", id="routed"),
            pytest.param(
                "import a::*\ntype G = Base\n\nrecord Base::R\n  v: int\n"
                "def r() -> int = G::R(v=2).v",
                "b::r()",
                id="nested-record",
            ),
        ],
    )
    def test_import_cycle_with_no_export_declarations_converges(
        self, tmp_path: Path, b_source: str, query: str
    ) -> None:
        """A cycle settles what a member declares at the other's type's path in its own fixpoint."""
        result = evaluate_ir_graph(
            f"import b\nlet result = {query}",
            {"a": "import b\nrecord Base\n  x: int\n", "b": b_source},
            tmp_path,
        )
        assert result["result"] == IntValue(2)

    @pytest.mark.parametrize(
        ("af_source", "h_decl"),
        [
            pytest.param(
                "def af() -> int = Base::h()", "def Base::h() -> int = 2\n", id="declared-spelling"
            ),
            pytest.param(
                "def af() -> int = G::h()", "def Base::h() -> int = 2\n", id="alias-spelling"
            ),
            pytest.param(
                "def af(receiver: Base) -> int = receiver.h()",
                "def Base::h(self) -> int = 2\n",
                id="method-call",
            ),
        ],
    )
    def test_import_cycle_builds_the_importer_env_over_the_settled_exports(
        self, tmp_path: Path, af_source: str, h_decl: str
    ) -> None:
        """A cycle's importer environment reads the settled exports, not a stale pass."""
        call = "a::af(a::Base(x = 1))" if "receiver" in af_source else "a::af()"
        result = evaluate_ir_graph(
            f"import a\nlet result = {call}",
            {
                "a": f"import b::*\nexport b::{{G}}\nrecord Base\n  x: int\n\n{af_source}\n",
                "b": f"import a::*\ntype G = Base\n\n{h_decl}",
            },
            tmp_path,
        )
        assert result["result"] == IntValue(2)

    def test_reexport_chain_multi_hop(self, tmp_path: Path) -> None:
        """A re-exports from B which re-exports from C — A's consumers get C's origin."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import a::*\nlet x = foo()",
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

    def test_wildcard_export_brace_tail_distributes_to_each_matching_module(
        self, tmp_path: Path
    ) -> None:
        """A wildcard brace tail forwards each target's selected subtree only."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import facade\nlet first = facade::Chosen::alpha()\n"
                    "let second = facade::Chosen::beta()"
                ),
                "facade": "export source/*::{Chosen}",
                "source/a": (
                    "scope Chosen\n"
                    "  def alpha() -> int = 1\n"
                    "end Chosen\n"
                    "\n"
                    "def excluded-a() -> int = 0"
                ),
                "source/b": (
                    "scope Chosen\n"
                    "  def beta() -> int = 2\n"
                    "end Chosen\n"
                    "\n"
                    "def excluded-b() -> int = 0"
                ),
            },
            default_stdlib=False,
        )

        result = resolve_program(graph)

        facade = result.modules[ModuleId.from_path("facade")]
        a_id = ModuleId.from_path("source/a")
        b_id = ModuleId.from_path("source/b")
        assert facade.exports[("Chosen", "alpha")] == (a_id, ("Chosen", "alpha"))
        assert facade.exports[("Chosen", "beta")] == (b_id, ("Chosen", "beta"))
        assert "excluded-a" not in facade.exports
        assert "excluded-b" not in facade.exports

        entry = result.modules[ENTRY_ID].resolved
        for name, module_id in (("alpha", a_id), ("beta", b_id)):
            var = _find_varref(entry.program, name)
            assert var is not None
            assert entry.resolution[var.node_id].module_id == module_id

    @pytest.mark.parametrize("declaration", ("export lib::{missing}", "export lib hiding missing"))
    def test_unknown_brace_or_hidden_export_atom_is_rejected(
        self, tmp_path: Path, declaration: str
    ) -> None:
        """Export selections and hiding clauses reject atoms absent from the target."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\n()",
                "facade": declaration,
                "lib": "def present() -> int = 1",
            },
            default_stdlib=False,
        )

        with pytest.raises(UnknownMemberError) as excinfo:
            resolve_program(graph)
        assert type(excinfo.value) is UnknownMemberError
        assert excinfo.value.repair is MissRepair.NOT_EXPORTED

    def test_importing_consumer_resolves_a_brace_renamed_export(self, tmp_path: Path) -> None:
        """An importer's bare alias resolves to the brace export's original declaration."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade::{plus}\nlet result = plus()",
                "facade": "export lib::{add as plus}",
                "lib": "def add() -> int = 42",
            },
            default_stdlib=False,
        )

        result = resolve_program(graph)

        plus = _find_varref(result.modules[ENTRY_ID].resolved.program, "plus")
        assert plus is not None
        ref = result.modules[ENTRY_ID].resolved.resolution[plus.node_id]
        assert ref.name == "add"
        assert ref.module_id == ModuleId.from_path("lib")

    def test_wildcard_reexport(self, tmp_path: Path) -> None:
        """export lib/* — all matching modules' public names are re-exported."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade::*\n()",
                "facade": "export lib/*",
                "lib/ops": "def add() -> int = 1",
            },
        )
        result = resolve_program(graph)
        facade_id = ModuleId.from_path("facade")
        lib_ops_id = ModuleId.from_path("lib/ops")
        facade_exports = result.modules[facade_id].exports
        assert "add" in facade_exports
        assert facade_exports["add"] == (lib_ops_id, "add")

    @pytest.mark.parametrize(
        ("exports", "duplicate"),
        [
            pytest.param("export a\nexport b", "foo", id="root"),
            pytest.param("export a::{S}\nexport b::{S}", "S::foo", id="scoped"),
        ],
    )
    def test_reexport_conflict_raises(self, tmp_path: Path, exports: str, duplicate: str) -> None:
        """Re-exporting one name from two different origins declares it twice."""
        scoped = duplicate != "foo"
        library = "scope S\n  def foo() -> int = {}\nend S" if scoped else "def foo() -> int = {}"
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade::*\n()",
                "facade": exports,
                "a": library.format(1),
                "b": library.format(2),
            },
        )
        with pytest.raises(DuplicateDeclarationError) as caught:
            resolve_program(graph)

        assert type(caught.value) is DuplicateDeclarationError
        assert caught.value.name == duplicate
        assert caught.value.reexports == (f"a::{duplicate}", f"b::{duplicate}")

    def test_reexported_ordinary_name_cannot_replace_a_local_scope_identity(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\n()",
                "facade": "export lib::{member as Public}\n\nscope Public\nend Public",
                "lib": "def member() -> int = 1",
            },
            default_stdlib=False,
        )

        with pytest.raises(DuplicateDeclarationError) as caught:
            resolve_program(graph)

        assert (type(caught.value), caught.value.name) == (DuplicateDeclarationError, "Public")

    def test_reexported_scope_identity_cannot_replace_a_local_ordinary_name(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\n()",
                "facade": "export lib::{Source as Public}\ndef Public() -> int = 1",
                "lib": "scope Source\nend Source",
            },
            default_stdlib=False,
        )

        with pytest.raises(DuplicateDeclarationError) as caught:
            resolve_program(graph)

        assert (type(caught.value), caught.value.name) == (DuplicateDeclarationError, "Public")

    def test_reexported_type_can_own_a_local_scope_namespace(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import facade\nlet value = facade::Public(value = 1)\nfacade::Public::read()"
                ),
                "facade": (
                    "export lib::{Source as Public}\n"
                    "\n"
                    "scope Public\n"
                    "  def read() -> int = 1\n"
                    "end Public"
                ),
                "lib": "record Source\n  value: int",
            },
            default_stdlib=False,
        )

        resolve_program(graph)

    def test_local_type_can_own_a_reexported_scope_namespace(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import facade\nlet value = facade::Public(value = 1)\nfacade::Public::read()"
                ),
                "facade": "export lib::{Source as Public}\nrecord Public\n  value: int",
                "lib": "scope Source\n  def read() -> int = 1\nend Source",
            },
            default_stdlib=False,
        )

        resolve_program(graph)

    def test_export_brace_tail_selects_reexports_through_chain(self, tmp_path: Path) -> None:
        """A brace-tail export resolves after its target's re-exports populate."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import a::*\nlet x = foo()",
                "a": "export b::{foo}",
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
                "entry": "import facade::*\nlet x = foo()\nlet y = facade::foo()",
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

    def test_scoped_export_brace_tail_reroots_the_forwarded_atom(self, tmp_path: Path) -> None:
        """A scoped `export` re-roots its forwarded atom under the region's own path."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade::*\n()",
                "facade": "scope Geo\n  export lib::{foo}\nend Geo",
                "lib": "def foo() -> int = 1\ndef bar() -> int = 2",
            },
        )
        result = resolve_program(graph)
        facade_id = ModuleId.from_path("facade")
        lib_id = ModuleId.from_path("lib")
        facade_exports = result.modules[facade_id].exports
        assert facade_exports[("Geo", "foo")] == (lib_id, "foo")
        assert "foo" not in facade_exports
        assert ("Geo", "bar") not in facade_exports

    def test_scoped_export_all_reroots_every_forwarded_atom(self, tmp_path: Path) -> None:
        """export lib inside a region — every forwarded atom is re-rooted the same way."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade::*\n()",
                "facade": "scope Geo\n  export lib\nend Geo",
                "lib": "def foo() -> int = 1\ndef bar() -> int = 2",
            },
        )
        result = resolve_program(graph)
        facade_id = ModuleId.from_path("facade")
        lib_id = ModuleId.from_path("lib")
        facade_exports = result.modules[facade_id].exports
        assert facade_exports[("Geo", "foo")] == (lib_id, "foo")
        assert facade_exports[("Geo", "bar")] == (lib_id, "bar")

    def test_scoped_export_hiding_reroots_the_remaining_atoms(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade::*\n()",
                "facade": "scope Geo\n  export lib hiding secret\nend Geo",
                "lib": "def foo() -> int = 1\ndef secret() -> int = 0",
            },
        )
        result = resolve_program(graph)
        facade_id = ModuleId.from_path("facade")
        lib_id = ModuleId.from_path("lib")
        facade_exports = result.modules[facade_id].exports
        assert facade_exports[("Geo", "foo")] == (lib_id, "foo")
        assert ("Geo", "secret") not in facade_exports

    def test_scoped_export_brace_tail_rename_composes_with_rerooting(self, tmp_path: Path) -> None:
        """A brace-tail rename applies before a region re-roots the atom."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade::*\n()",
                "facade": "scope Geo\n  export lib::{foo as plus}\nend Geo",
                "lib": "def foo() -> int = 1",
            },
        )
        result = resolve_program(graph)
        facade_id = ModuleId.from_path("facade")
        lib_id = ModuleId.from_path("lib")
        facade_exports = result.modules[facade_id].exports
        assert facade_exports[("Geo", "plus")] == (lib_id, "foo")
        assert ("Geo", "foo") not in facade_exports

    def test_an_importer_reaches_a_scoped_brace_tail_reexport_by_its_rerooted_path(
        self, tmp_path: Path
    ) -> None:
        """A consumer of the re-exporting module reaches the atom via its re-rooted path."""
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade\nlet x = facade::Geo::foo()",
                "facade": "scope Geo\n  export lib::{foo}\nend Geo",
                "lib": "def foo() -> int = 42",
            },
        )
        result = resolve_program(graph)
        entry = result.modules[ENTRY_ID]
        ref_node = _find_varref(entry.resolved.program, "foo")
        assert ref_node is not None
        lib_id = ModuleId.from_path("lib")
        assert entry.resolved.resolution[ref_node.node_id].module_id == lib_id


@pytest.mark.parametrize(
    ("pattern", "candidate_facts"),
    (
        ("packet(mark, _ as mark)", (True, False)),
        ("packet(_ as mark, mark)", (False, True)),
    ),
)
def test_imported_nullary_variant_defers_duplicate_pattern_binders(
    tmp_path: Path, pattern: str, candidate_facts: tuple[bool, bool]
) -> None:
    graph = _make_graph_from_files(
        tmp_path,
        {
            "library": "enum Flag\n  | mark",
            "entry": (
                "import library::*\n"
                "enum Packet\n"
                "  | packet(left: int, right: int)\n"
                "let item = packet(1, 2)\n"
                f"case item of | {pattern} => mark"
            ),
        },
    )

    entry = resolve_program(graph).modules[ENTRY_ID].resolved
    main = entry.program.body.items[-1]
    assert isinstance(main, FuncDef)
    case = main.body.items[-1]
    assert isinstance(case, Case)
    assert isinstance(case.branches[0].pattern, ConstructorPattern)
    bare = next(
        pattern
        for pattern in case.branches[0].pattern.positional
        if isinstance(pattern, VarPattern)
    )
    assert entry.pattern_constructor_candidates[bare.node_id][0].can_match_bare_pattern
    slot = next(iter(entry.pattern_slots.values()))
    assert (
        tuple(candidate.can_match_bare_pattern for candidate in slot.candidates) == candidate_facts
    )


@pytest.mark.parametrize("owner", ("Token", "Alias"))
def test_plain_import_retains_qualified_constructor_pattern_candidates(
    tmp_path: Path, owner: str
) -> None:
    graph = _make_graph_from_files(
        tmp_path,
        {
            "library": "record Token\n  value: int\ntype Alias = Token",
            "entry": (
                f"import library\n"
                f"let subject: library::{owner} = library::{owner}(value = 1)\n"
                f"case subject of | library::{owner}(value) => value"
            ),
        },
    )

    entry = resolve_program(graph).modules[ENTRY_ID].resolved
    main = entry.program.body.items[-1]
    assert isinstance(main, FuncDef)
    case = main.body.items[-1]
    assert isinstance(case, Case)
    pattern = case.branches[0].pattern
    assert isinstance(pattern, ConstructorPattern)
    candidates = entry.pattern_constructor_candidates[pattern.node_id]
    assert candidates[0].owner_module_id == ModuleId.from_path("library")
    assert candidates[0].owner_name == owner


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
                "import foreign::*\n"
                "enum Local\n  | same\n"
                "let value: Local = Local::same\n"
                "case value of | same => 1"
            ),
        },
    )

    result = resolve_program(graph)
    entry = result.modules[ENTRY_ID]
    main = entry.resolved.program.body.items[-1]
    assert isinstance(main, FuncDef)
    case = main.body.items[-1]
    assert isinstance(case, Case)
    pattern = case.branches[0].pattern
    candidate_owners = {
        (ref.owner_path, ref.owner_name)
        for ref in entry.resolved.pattern_constructor_candidates[pattern.node_id]
    }
    assert candidate_owners == {(("Local",), "same"), (("Foreign",), "same")}


def test_scoped_invalid_referenced_member_does_not_create_a_constructor_candidate(
    tmp_path: Path,
) -> None:
    """Scope collection leaves an invalid referenced member to type checking."""
    graph = _make_graph_from_files(
        tmp_path,
        {
            "entry": "import lib::*\n\nscope Local\n  import lib::{E}\nend Local\n\n()",
            "lib": "import target\nenum E = target::NotRecord",
            "target": "enum NotRecord\n  | variant",
        },
    )

    result = resolve_program(graph)
    assert result.modules[ENTRY_ID].resolved.constructor_candidates.get("NotRecord") is None


# ---------------------------------------------------------------------------
# Test: diagnostic source locations
# ---------------------------------------------------------------------------


class TestDiagnosticSpans:
    """Scope diagnostics must point at the offending construct, not the file start.

    Every source below puts the offence past the first line and past the first
    column of that line, so a diagnostic that lost its location cannot pass by
    defaulting to the start of the module.
    """

    @pytest.mark.parametrize(
        ("files", "expected"),
        (
            pytest.param(
                {
                    "entry": "import libA::*\nimport libB::*\nlet x = foo()",
                    "libA": "def foo() -> int = 1",
                    "libB": "def foo() -> int = 2",
                },
                (3, 9, 3, 12),
                id="ambiguous-bare-use",
            ),
            pytest.param(
                {"entry": "let a = 1\nlet b = notdefined\n()"},
                (2, 9, 2, 19),
                id="undefined-name",
            ),
            pytest.param(
                {
                    "entry": "import lib\nlet y = lib::missing()",
                    "lib": "def foo() -> int = 1",
                },
                (2, 9, 2, 21),
                id="unknown-qualified-member",
            ),
            pytest.param(
                {"entry": "def foo() -> int = 1\ndef foo() -> int = 2\n()"},
                (2, 1, 2, 21),
                id="duplicate-declaration",
            ),
            pytest.param(
                {"entry": "let a = 1\nimport lib\n()", "lib": "def foo() -> int = 1"},
                (2, 1, 2, 11),
                id="import-after-declaration",
            ),
        ),
    )
    def test_scope_diagnostic_points_at_the_offending_construct(
        self,
        tmp_path: Path,
        files: dict[str, str],
        expected: tuple[int, int, int, int],
    ) -> None:
        graph = _make_graph_from_files(tmp_path, files)
        with pytest.raises(AglScopeError) as exc_info:
            resolve_program(graph)
        span = exc_info.value.span
        assert span is not None
        assert (span.start_line, span.start_col, span.end_line, span.end_col) == expected

    def test_scope_diagnostic_survives_conversion_to_a_diagnostic(self, tmp_path: Path) -> None:
        """The rendered diagnostic keeps the coordinates the error carried."""
        graph = _make_graph_from_files(tmp_path, {"entry": "let a = 1\nlet b = notdefined\n()"})
        with pytest.raises(AglScopeError) as exc_info:
            resolve_program(graph)
        diagnostic = exc_info.value.to_diagnostic()
        assert (diagnostic.line, diagnostic.column) == (2, 9)
        assert (diagnostic.end_line, diagnostic.end_column) == (2, 19)


class TestAppliedBuiltinReceiverScopes:
    @pytest.mark.parametrize(
        ("receiver", "source"),
        (
            ("array", "def array[E]::first(self) -> E = self[0]"),
            ("dict", 'def dict[text, V]::value(self) -> V = self["value"]'),
        ),
    )
    def test_applied_receiver_publishes_its_plain_scope(
        self, tmp_path: Path, receiver: str, source: str
    ) -> None:
        result = resolve_program(
            _make_graph_from_files(
                tmp_path,
                {"entry": "import lib\n()", "lib": source},
                default_stdlib=False,
            )
        )

        lib_id = ModuleId.from_path("lib")
        assert result.modules[lib_id].scope_exports[receiver] == frozenset({(lib_id, receiver)})

    def test_applied_receiver_names_share_one_declaration_scope(self, tmp_path: Path) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import lib\n()",
                "lib": (
                    "def array[E]::first(self) -> E = self[0]\n"
                    "def array[T]::first(self) -> T = self[0]"
                ),
            },
            default_stdlib=False,
        )

        with pytest.raises(AglScopeError):
            resolve_program(graph)

    @pytest.mark.parametrize(
        ("receiver", "method", "source"),
        (
            ("array", "first", "def array[E]::first(self) -> E = self[0]"),
            ("dict", "value", 'def dict[text, V]::value(self) -> V = self["value"]'),
        ),
    )
    def test_facade_reexports_an_applied_receiver_scope_for_import(
        self, tmp_path: Path, receiver: str, method: str, source: str
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": f"import facade::{{{receiver}}}\nuse {receiver}::*\n{method}()",
                "facade": f"export lib::{{{receiver}}}",
                "lib": source,
            },
            default_stdlib=False,
        )

        result = resolve_program(graph)

        method_ref = _find_varref(graph.modules[ENTRY_ID].program, method)
        assert method_ref is not None
        binding = result.modules[ENTRY_ID].resolved.resolution[method_ref.node_id]
        assert (binding.module_id, binding.scope_path, binding.name) == (
            ModuleId.from_path("lib"),
            (receiver,),
            method,
        )

    def test_ambiguity_origins_name_an_applied_receiver_declaration_by_its_path(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import facade\n"
                    "import other\n"
                    "use facade::array::*\n"
                    "use other::array::*\n"
                    "first()"
                ),
                "facade": "export lib::{array}",
                "lib": "def array[E]::first(self) -> E = self[0]",
                "other": "def array[T]::first(self) -> T = self[0]",
            },
            default_stdlib=False,
        )

        with pytest.raises(AmbiguousQualificationError) as exc_info:
            resolve_program(graph)

        assert [origin.declaration for origin in exc_info.value.origins] == [
            (ModuleId.from_path("lib"), ("array", "first")),
            (ModuleId.from_path("other"), ("array", "first")),
        ]
