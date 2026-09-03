"""Tests for the AgL scope/name-resolution pass.

All tests drive real AgL source through ``parse_program`` + ``resolve``,
or construct AST nodes directly for cases that are clearer to express
at the AST level.

Tests assert on user-visible behavior: ``AglScopeError`` diagnostics and
observable side-table behavior via the public ``ModuleResolution`` API.  They
deliberately do *not* pin internal implementation details.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agm.agl.modules.ids import ENTRY_ID
from agm.agl.scope import (
    AglScopeError,
    BuiltinKind,
    ModuleResolution,
)
from agm.agl.scope.symbols import BinderKind, BindingRef, ScopeNode
from agm.agl.syntax.nodes import (
    AsPattern,
    AssignStmt,
    Block,
    BoolLit,
    Call,
    Case,
    CaseBranch,
    CatchClause,
    ConstructorPattern,
    Do,
    EnumDef,
    Expr,
    FieldAccess,
    FuncDef,
    If,
    IfBranch,
    IntLit,
    Item,
    Lambda,
    LetDecl,
    NameTarget,
    Param,
    ParamKind,
    PatternField,
    Program,
    RecordDef,
    ScopeRegion,
    StringLit,
    Template,
    Try,
    UnitLit,
    VarDecl,
    VariantDef,
    VarPattern,
    VarRef,
    WildcardPattern,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import IntT
from tests.agl.module_graph import (
    resolve_entry,
    resolve_inline_entry,
    resolve_inline_program_ast,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_and_resolve(
    source: str,
    *,
    origin_path: Path | None = None,
    parent_scope: ScopeNode | None = None,
    default_stdlib: bool = True,
) -> ModuleResolution:
    """Resolve test-only inline source as ``agm exec -c`` does."""
    return resolve_inline_entry(
        source,
        origin_path=origin_path,
        parent_scope=parent_scope,
        default_stdlib=default_stdlib,
    )


def parse_and_resolve_file(source: str, *, default_stdlib: bool = True) -> ModuleResolution:
    """Resolve static test source without applying the command entry transform."""
    return resolve_entry(source, default_stdlib=default_stdlib)


def parse_and_resolve_repl(
    source: str, *, parent_scope: ScopeNode | None = None
) -> ModuleResolution:
    """Resolve a test-only REPL entry, whose root admits incremental statements."""
    return resolve_entry(
        source,
        parent_scope=parent_scope or ScopeNode(node_id=-1, parent=None, scope_path=()),
    )


def reject_scope(source: str, *, default_stdlib: bool = True) -> AglScopeError:
    """Assert that *source* fails scope resolution and return the error."""
    with pytest.raises(AglScopeError) as exc_info:
        parse_and_resolve(source, default_stdlib=default_stdlib)
    return exc_info.value


def diag(err: AglScopeError) -> tuple[int, str]:
    """Return (line, message) from an AglScopeError."""
    d = err.to_diagnostic()
    return d.line, d.message


def quoted_names(message: str) -> list[str]:
    """Return the single-quoted names a diagnostic mentions, in order."""
    return re.findall(r"'([^']+)'", message)


def _find_varref(program: object, name: str, occurrence: int = -1) -> VarRef:
    """Walk *program* depth-first; return the VarRef named *name* at *occurrence*.

    occurrence=-1 (default) returns the last match; 0 returns the first.
    Raises AssertionError if no match found.
    """
    from agm.agl.syntax.nodes import (
        ArrayLit,
        BinaryOp,
        Case,
        Cast,
        DictLit,
        IndexAccess,
        InterpSegment,
        IsTest,
        ParamDecl,
        Raise,
        ScopeRegion,
        UnaryNeg,
        UnaryNot,
    )

    results: list[VarRef] = []

    def walk(node: object) -> None:
        if isinstance(node, VarRef):
            if node.name == name:
                results.append(node)
            return
        if isinstance(node, Program):
            walk(node.body)
        elif isinstance(node, Block):
            for item in node.items:
                walk(item)
        elif isinstance(node, ScopeRegion):
            for item in node.items:
                walk(item)
        elif isinstance(node, FuncDef):
            for param in node.params:
                if param.default is not None:
                    walk(param.default)
            walk(node.body)
        elif isinstance(node, Lambda):
            for param in node.params:
                if param.default is not None:
                    walk(param.default)
            walk(node.body)
        elif isinstance(node, LetDecl):
            walk(node.value)
        elif isinstance(node, VarDecl):
            walk(node.value)
        elif isinstance(node, AssignStmt):
            walk(node.value)
        elif isinstance(node, Call):
            walk(node.callee)
            for arg in node.args:
                walk(arg)
            for na in node.named_args:
                walk(na.value)
        elif isinstance(node, BinaryOp):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, If):
            for branch in node.branches:
                walk(branch.cond)
                walk(branch.body)
        elif isinstance(node, Case):
            walk(node.subject)
            for cbranch in node.branches:
                walk(cbranch.body)
        elif isinstance(node, Do):
            walk(node.body)
            walk(node.condition)
        elif isinstance(node, Try):
            walk(node.body)
            for clause in node.handlers:
                walk(clause.body)
        elif isinstance(node, Template):
            for seg in node.segments:
                if isinstance(seg, InterpSegment):
                    walk(seg.expr)
        elif isinstance(node, FieldAccess):
            walk(node.obj)
        elif isinstance(node, IndexAccess):
            walk(node.obj)
            walk(node.index)
        elif isinstance(node, UnaryNot):
            walk(node.operand)
        elif isinstance(node, UnaryNeg):
            walk(node.operand)
        elif isinstance(node, Raise):
            walk(node.exc)
        elif isinstance(node, Cast):
            walk(node.expr)
        elif isinstance(node, IsTest):
            walk(node.expr)
        elif isinstance(node, ArrayLit):
            for elem in node.elements:
                walk(elem)
        elif isinstance(node, DictLit):
            for entry in node.entries:
                walk(entry.value)
        elif isinstance(node, ParamDecl):
            if node.default is not None:
                walk(node.default)

    walk(program)
    assert results, f"No VarRef named {name!r} found"
    return results[occurrence]


def _ref(r: ModuleResolution, name: str, occurrence: int = -1) -> BindingRef:
    """Return the BindingRef for the VarRef named *name* (at *occurrence*) in *r.program*."""
    vr = _find_varref(r.program, name, occurrence)
    return r.resolution[vr.node_id]


# ---------------------------------------------------------------------------
# AST construction helpers (for hand-built node tests)
# ---------------------------------------------------------------------------

_NID = 0


def _nid() -> int:
    global _NID
    _NID += 1
    return _NID


def _sp(line: int = 1) -> SourceSpan:
    return SourceSpan(
        start_line=line,
        start_col=1,
        end_line=line,
        end_col=2,
        start_offset=0,
        end_offset=1,
    )


def _make_intlit(val: int = 0, line: int = 1) -> IntLit:
    return IntLit(value=val, span=_sp(line), node_id=_nid())


def _make_boollit(val: bool = True, line: int = 1) -> BoolLit:
    return BoolLit(value=val, span=_sp(line), node_id=_nid())


def _make_unitlit(line: int = 1) -> UnitLit:
    return UnitLit(span=_sp(line), node_id=_nid())


def _make_varref(name: str, line: int = 1) -> VarRef:
    return VarRef(name=name, span=_sp(line), node_id=_nid())


def _make_let(name: str, value: Expr, line: int = 1) -> LetDecl:
    return LetDecl(
        pattern=VarPattern(name=name, span=_sp(line), node_id=_nid()),
        type_ann=None,
        value=value,
        span=_sp(line),
        node_id=_nid(),
    )


def _make_var(name: str, value: Expr, line: int = 1) -> VarDecl:
    return VarDecl(name=name, type_ann=None, value=value, span=_sp(line), node_id=_nid())


def _make_assign(target: str, value: Expr, line: int = 1) -> AssignStmt:
    span = _sp(line)
    return AssignStmt(
        target=NameTarget(name=target, span=span, node_id=_nid()),
        value=value,
        span=span,
        node_id=_nid(),
    )


def _make_block(*items: Item, line: int = 1) -> Block:
    return Block(items=items, span=_sp(line), node_id=_nid())


def _make_program(*items: Item) -> Program:
    block = Block(items=items, span=_sp(), node_id=_nid())
    return Program(body=block, span=_sp(), node_id=_nid())


def resolve_program(*items: Item) -> ModuleResolution:
    """Construct and resolve a Program from the given top-level items.

    Uses the explicit test-only inline-AST seam: the hand-built items model
    command source, where executable root items are synthesized into ``main``.
    """
    return resolve_inline_program_ast(_make_program(*items), next_node_id=_NID + 1)


def reject_program(*items: Item) -> AglScopeError:
    """Assert that a program built from the given items fails scope resolution."""
    with pytest.raises(AglScopeError) as exc_info:
        resolve_program(*items)
    return exc_info.value


# ---------------------------------------------------------------------------
# Scope regions
# ---------------------------------------------------------------------------


class TestScopeRegions:
    def test_regions_and_shorthand_collect_members_in_one_scope(self) -> None:
        resolved = parse_and_resolve(
            "scope Point\n"
            "  def distance() -> int = 0\n"
            "end Point\n"
            "\n"
            "def Point::length() -> int = 0\n"
            "()"
        )

        members = {
            name
            for (module_id, path, name) in resolved.declarations
            if module_id == ENTRY_ID and path == ("Point",)
        }
        assert members == {"distance", "length"}
        assert resolved.scope_nodes[("Point",)].parent is resolved.root_scope

    def test_repeated_regions_extend_the_same_scope(self) -> None:
        resolved = parse_and_resolve(
            "scope Point\n  def x() -> int = 0\nend Point\n"
            "\n"
            "scope Point\n  def y() -> int = 0\nend Point\n\n()"
        )

        assert {
            name for (_module_id, path, name) in resolved.declarations if path == ("Point",)
        } == {"x", "y"}

    def test_type_scope_merges_with_a_region_and_collects_variants(self) -> None:
        resolved = parse_and_resolve(
            "scope Result\n  def describe() -> int = 0\nend Result\n\nenum Result = ok | error\n()"
        )

        assert (ENTRY_ID, (), "Result") in resolved.declarations
        assert (ENTRY_ID, ("Result",), "describe") in resolved.declarations
        assert (ENTRY_ID, ("Result",), "ok") in resolved.declarations
        assert (ENTRY_ID, ("Result",), "error") in resolved.declarations

    def test_type_then_region_merges_into_the_type_scope(self) -> None:
        resolved = parse_and_resolve(
            "enum Result = ok | error\n"
            "\n"
            "scope Result\n"
            "  def describe() -> int = 0\n"
            "end Result\n"
            "\n"
            "()"
        )

        assert set(resolved.scope_nodes[("Result",)].members) == {"ok", "error", "describe"}

    def test_nested_type_establishes_a_nested_scope(self) -> None:
        resolved = parse_and_resolve("scope A\n  enum T = value\nend A\n\n()")

        assert (ENTRY_ID, ("A",), "T") in resolved.declarations
        assert (ENTRY_ID, ("A", "T"), "value") in resolved.declarations
        assert resolved.scope_nodes[("A", "T")].parent is resolved.scope_nodes[("A",)]

    def test_scoped_enum_variant_yields_to_an_enclosing_scope_member(self) -> None:
        parse_and_resolve("enum A::Choice = picked\ndef A::picked() -> int = 0\n()")

    def test_inline_members_establish_nested_type_scopes(self) -> None:
        resolved = parse_and_resolve("enum Tree = Leaf | Node(value: int)\n()")

        assert ("Tree", "Leaf") in resolved.declared_type_paths
        assert ("Tree", "Node") in resolved.declared_type_paths
        assert resolved.scope_nodes[("Tree", "Node")].parent is resolved.scope_nodes[("Tree",)]

    def test_scoped_members_resolve_from_their_exact_path(self) -> None:
        resolved = parse_and_resolve("def A::f() -> int = 0\nA::f()")
        assert resolved.resolution
        assert next(iter(resolved.resolution.values())).decl_node_id == next(
            declaration.decl_node_id
            for (module_id, path, name), declaration in resolved.declarations.items()
            if module_id == ENTRY_ID and path == ("A",) and name == "f"
        )

    @pytest.mark.parametrize(
        ("source", "name", "line"),
        (
            (
                "scope Point\n  def distance() -> int = 0\nend Point\n"
                "\n"
                "def Point::distance() -> int = 1",
                "distance",
                5,
            ),
            ("scope Point\nend Point\n\ndef Point() -> int = 0", "Point", 4),
            ("record Point()\ndef Point() -> int = 0", "Point", 2),
            ("enum Point = one | one", "one", 1),
        ),
    )
    def test_same_path_declaration_collisions_are_rejected(
        self, source: str, name: str, line: int
    ) -> None:
        err = reject_scope(source)

        reported_line, message = diag(err)
        assert name in message
        assert "declared" in message
        # The complaint locates the later declaration, not the first one.
        assert reported_line == line

    def test_enum_member_spelling_can_be_claimed_in_its_enclosing_scope(self) -> None:
        parse_and_resolve("enum Point = origin\ndef origin() -> int = 0\n()")


# ---------------------------------------------------------------------------
# Scoped `let`/`var` bindings — members, not lexical bindings
# ---------------------------------------------------------------------------


class TestScopedBindings:
    """A `let`/`var` with a scope path is a member of that path, not a local binding.

    Region form (``scope A / let x = 1 / end A``) and root shorthand
    (``let A::x = 1``) declare the same member and share every rule below.
    """

    def test_bare_visible_inside_its_own_region(self) -> None:
        resolved = parse_and_resolve(
            "scope A\n  let x = 1\n  def read() -> int = x\nend A\n\nA::read()"
        )
        assert _ref(resolved, "x").kind is BinderKind.let_binding
        assert _ref(resolved, "x").scope_path == ("A",)

    def test_bare_visible_from_a_nested_region_via_the_outward_walk(self) -> None:
        resolved = parse_and_resolve(
            "scope A\n"
            "  let x = 1\n"
            "\n"
            "  scope B\n"
            "    def read() -> int = x\n"
            "  end B\n"
            "end A\n"
            "\n"
            "A::B::read()"
        )
        assert _ref(resolved, "x").kind is BinderKind.let_binding

    def test_double_colon_anchors_at_the_module_root_from_inside_a_region(self) -> None:
        resolved = parse_and_resolve_file(
            "let x: int = 1\n\nscope A\n  let x = 2\n  def read() -> int = ::x\nend A"
        )
        assert _ref(resolved, "x").scope_path == ()

    def test_exact_path_reference_from_outside_the_region(self) -> None:
        resolved = parse_and_resolve("scope A\n  let x = 1\nend A\n\nlet y = A::x\ny")
        assert _ref(resolved, "x").scope_path == ("A",)
        assert _ref(resolved, "x").kind is BinderKind.let_binding

    def test_shorthand_rhs_sees_an_earlier_regions_member_bare(self) -> None:
        """A root shorthand's RHS resolves in its own scope path's layer."""
        resolved = parse_and_resolve_file(
            "scope A\n  let base = 1\nend A\n\nvar A::next = base + 1"
        )
        assert set(resolved.scope_nodes[("A",)].members) == {"base", "next"}
        assert _ref(resolved, "base").scope_path == ("A",)

    def test_region_sees_an_earlier_shorthands_member_bare(self) -> None:
        resolved = parse_and_resolve_file(
            "let A::base = 1\n\nscope A\n  def read() -> int = base\nend A"
        )
        assert set(resolved.scope_nodes[("A",)].members) == {"base", "read"}

    def test_repeated_region_blocks_extend_the_same_scope(self) -> None:
        resolved = parse_and_resolve(
            "scope A\n  let x = 1\nend A\n\nscope A\n  let y = x\nend A\n\n()"
        )
        assert set(resolved.scope_nodes[("A",)].members) == {"x", "y"}

    @pytest.mark.parametrize(
        ("source", "name"),
        (
            ("scope A\n  def f() -> int = 0\n  let f = 1\nend A\n\n()", "f"),
            ("scope A\n  let x = 1\n  let x = 2\nend A\n\n()", "x"),
            ("scope A\n  record R()\n  let R = 1\nend A\n\n()", "R"),
            ("scope A\n  var x = 1\n  var x = 2\nend A\n\n()", "x"),
        ),
        ids=("binding-vs-def", "binding-vs-binding", "binding-vs-type", "binding-vs-binding-var"),
    )
    def test_binding_duplicate_at_the_same_path_is_rejected(self, source: str, name: str) -> None:
        err = reject_scope(source)
        line, message = diag(err)
        assert name in message
        assert "declared" in message
        # The complaint locates the second declaration, on the third line.
        assert line == 3

    @pytest.mark.parametrize(
        "source",
        (
            "scope A\n  let B = 1\n\n  scope B\n    def q() -> int = 2\n  end B\nend A\n\n()",
            "scope A\n\n  scope B\n    def q() -> int = 2\n  end B\n\n  let B = 1\nend A\n\n()",
            "scope A\n"
            "  def B() -> int = 1\n"
            "\n"
            "  scope B\n"
            "    def q() -> int = 2\n"
            "  end B\n"
            "end A\n"
            "\n"
            "()",
            "scope A\n"
            "\n"
            "  scope B\n"
            "    def q() -> int = 2\n"
            "  end B\n"
            "\n"
            "  def B() -> int = 1\n"
            "end A\n"
            "\n"
            "()",
            "let A::B = 1\n\nscope A\n\n  scope B\n    def q() -> int = 2\n  end B\nend A\n\n()",
            "scope A\n\n  scope B\n    def q() -> int = 2\n  end B\nend A\n\nlet A::B = 1\n()",
        ),
        ids=(
            "region-let-before-nested-scope",
            "region-nested-scope-before-let",
            "region-def-before-nested-scope",
            "region-nested-scope-before-def",
            "shorthand-let-before-nested-scope",
            "nested-scope-before-shorthand-let",
        ),
    )
    def test_binding_collides_with_a_nested_scope_layer(self, source: str) -> None:
        """A member cannot claim a name already owned by a nested scope layer.

        ``def B`` colliding with a sibling ``scope B`` is a pre-existing,
        correctly-rejected case, included here to pin parity with the
        ``let``/``var`` binding forms across both textual orders.
        """
        with pytest.raises(AglScopeError) as exc_info:
            parse_and_resolve_file(source.removesuffix("\n()"))
        message = exc_info.value.to_diagnostic().message
        assert "B" in message
        assert "declared" in message

    def test_earlier_block_cannot_see_a_later_blocks_binding(self) -> None:
        with pytest.raises(AglScopeError):
            parse_and_resolve(
                "scope A\n  def f() -> int = A::y\nend A\n\nscope A\n  let y = 1\nend A\n\nA::f()"
            )

    def test_reverse_order_of_repeated_blocks_succeeds(self) -> None:
        resolved = parse_and_resolve(
            "scope A\n  let y = 1\nend A\n\nscope A\n  def f() -> int = A::y\nend A\n\nA::f()"
        )
        assert set(resolved.scope_nodes[("A",)].members) == {"y", "f"}

    def test_reference_textually_before_a_shorthand_binding_is_rejected(self) -> None:
        with pytest.raises(AglScopeError):
            parse_and_resolve("def f() -> int = A::x\nlet A::x = 1\nf()")

    def test_destructuring_let_in_a_region_contributes_every_selected_binder(self) -> None:
        resolved = parse_and_resolve(
            "scope A\n"
            "  record Point\n"
            "    x: int\n"
            "    y: int\n"
            "  let Point(x, y) = Point(x = 1, y = 2)\n"
            "end A\n"
            "\n"
            "A::x\n"
            "A::y\n"
        )
        members = resolved.scope_nodes[("A",)].members
        assert members["x"].kind is BinderKind.pattern_slot
        assert members["y"].kind is BinderKind.pattern_slot

    @pytest.mark.parametrize("keyword", ("let", "var"))
    def test_binder_path_in_a_nested_block_is_rejected(self, keyword: str) -> None:
        with pytest.raises(AglScopeError):
            parse_and_resolve(f"if true =>\n  {keyword} A::x = 1\n  x\n| else =>\n  0\n")

    def test_multi_segment_scoped_let_and_var(self) -> None:
        resolved = parse_and_resolve_file("let A::B::x = 1\nvar A::B::y = 2")
        assert set(resolved.scope_nodes[("A", "B")].members) == {"x", "y"}
        assert resolved.scope_nodes[("A", "B")].parent is resolved.scope_nodes[("A",)]


# ---------------------------------------------------------------------------
# Scoped `param` -- a member of its path, region-form only (no shorthand)
# ---------------------------------------------------------------------------


class TestScopedParam:
    """A `param` declared inside a scope region is a member of that path,
    following the same member/duplicate rules as every other member.
    """

    def test_bare_visible_inside_its_own_region(self) -> None:
        resolved = parse_and_resolve(
            "scope Deploy\n  param region: text\n  def read() -> text = region\nend Deploy\n"
            "\n"
            "Deploy::read()"
        )
        assert _ref(resolved, "region").kind is BinderKind.param_binding
        assert _ref(resolved, "region").scope_path == ("Deploy",)

    def test_bare_visible_from_a_nested_region_via_the_outward_walk(self) -> None:
        resolved = parse_and_resolve(
            "scope Deploy\n  param region: text\n\n  scope Inner\n"
            "    def read() -> text = region\n  end Inner\nend Deploy\n\nDeploy::Inner::read()"
        )
        assert _ref(resolved, "region").kind is BinderKind.param_binding

    def test_exact_path_reference_from_outside_the_region(self) -> None:
        resolved = parse_and_resolve(
            "scope Deploy\n  param region: text\nend Deploy\n\nDeploy::region"
        )
        assert _ref(resolved, "region").scope_path == ("Deploy",)
        assert _ref(resolved, "region").kind is BinderKind.param_binding

    def test_visible_after_use(self) -> None:
        resolved = parse_and_resolve(
            "use Deploy::*\n\nscope Deploy\n  param region: text\nend Deploy\n\nregion"
        )
        assert _ref(resolved, "region").scope_path == ("Deploy",)

    def test_repeated_region_blocks_extend_the_same_scope(self) -> None:
        resolved = parse_and_resolve(
            "scope Deploy\n  param region: text\nend Deploy\n"
            "\n"
            "scope Deploy\n  param replicas: int\nend Deploy\n\n()"
        )
        assert set(resolved.scope_nodes[("Deploy",)].members) == {"region", "replicas"}

    def test_earlier_block_cannot_see_a_later_blocks_param(self) -> None:
        with pytest.raises(AglScopeError):
            parse_and_resolve(
                "scope A\n  def f() -> text = A::region\nend A\n"
                "\n"
                "scope A\n  param region: text\nend A\n\nA::f()"
            )

    def test_reference_textually_before_the_param_is_rejected(self) -> None:
        with pytest.raises(AglScopeError):
            parse_and_resolve_file(
                "def f() -> text = Deploy::region\n\nscope Deploy\n  param region: text\nend Deploy"
            )

    def test_param_rejected_inside_a_function_body(self) -> None:
        with pytest.raises(AglScopeError, match="param"):
            parse_and_resolve("def f() =\n  param x\n  0\nf()")


class TestScopedBindingUsePrecedence:
    """A local ``use`` — glob, selected-tail, or ``hiding`` — resolves live.

    A local ``use`` records its target scope path together with its full
    selection (mode, items, renames) and resolves it against the scope tree
    at every later bare reference, never through a snapshot taken when the
    ``use`` itself is walked. A declaration (``def``/type) is fully collected
    in the pre-pass, so every ``use`` sees it regardless of order; a scoped
    binder is registered only when the walk reaches it, so a reference
    textually walked *after* the binder's own registration still reaches it
    even though the ``use`` came first -- textual precedence still governs a
    reference walked *before* the binder, with no dedicated check, for every
    selection form alike.
    """

    def test_use_before_the_binding_does_not_see_it(self) -> None:
        """The reference is walked before the binder registers inside the earlier
        ``scope B`` block, so even the live scope-use re-check finds nothing."""
        with pytest.raises(AglScopeError):
            parse_and_resolve(
                "scope B\n  use A::*\n  def get() -> int = x\nend B\n"
                "\n"
                "scope A\n  let x = 1\nend A\n\nB::get()"
            )

    def test_use_after_the_binding_sees_it(self) -> None:
        resolved = parse_and_resolve(
            "scope A\n"
            "  let x = 1\n"
            "end A\n"
            "\n"
            "scope B\n"
            "  use A::*\n"
            "  def get() -> int = x\n"
            "end B\n"
            "\n"
            "B::get()"
        )
        assert _ref(resolved, "x").scope_path == ("A",)

    def test_use_before_a_declaration_still_sees_it(self) -> None:
        """Declarations stay order-independent even when a scope use precedes their block."""
        resolved = parse_and_resolve(
            "scope B\n  use A::*\n  def get() -> int = f()\nend B\n"
            "\n"
            "scope A\n  def f() -> int = 0\nend A\n"
            "\n"
            "B::get()"
        )
        assert _ref(resolved, "f").scope_path == ("A",)

    def test_use_before_the_binding_sees_a_later_reference(self) -> None:
        """Header placement forces ``use A::*`` before ``scope A`` at the module
        root, so its own member snapshot cannot yet hold ``x``; the bare
        reference below is walked after ``scope A`` registers it, so the live
        re-check reaches it instead of falling through to an error."""
        resolved = parse_and_resolve("use A::*\n\nscope A\n  var x = 1\nend A\n\nx := x + 1\nx")
        assert _ref(resolved, "x").scope_path == ("A",)
        assert _ref(resolved, "x").kind is BinderKind.var_binding

    def test_selected_use_before_the_binding_sees_a_later_reference(self) -> None:
        """A selected ``use A::{x}`` reaches a binder declared after it too.

        Before the uniform mechanism, only the glob form was recorded for
        live lookup, so a selected tail failed at the ``use`` site for a
        member its equivalent glob form already reaches.
        """
        resolved = parse_and_resolve("use A::{x}\n\nscope A\n  var x = 1\nend A\n\nx := x + 1\nx")
        assert _ref(resolved, "x").scope_path == ("A",)
        assert _ref(resolved, "x").kind is BinderKind.var_binding

    def test_hiding_use_before_the_binding_sees_the_non_hidden_later_reference(self) -> None:
        """A filtered ``hiding`` use reaches::* its non-hidden member too, live."""
        resolved = parse_and_resolve(
            "use A::* hiding y\n\nscope A\n  var x = 1\n  var y = 2\nend A\n\nx := x + 1\nx"
        )
        assert _ref(resolved, "x").scope_path == ("A",)
        assert _ref(resolved, "x").kind is BinderKind.var_binding

    def test_hiding_use_still_hides_a_member_declared_after_it(self) -> None:
        """The hidden member stays unreachable bare even though it is
        registered after the ``use``, live, exactly as the exposed sibling
        member is reached live."""
        with pytest.raises(AglScopeError):
            parse_and_resolve("use A::* hiding y\n\nscope A\n  var x = 1\n  var y = 2\nend A\n\ny")

    def test_hiding_rejects_a_member_missing_after_the_scope_is_complete(self) -> None:
        with pytest.raises(AglScopeError):
            parse_and_resolve("use A::* hiding missing\n\nscope A\n  let x = 1\nend A\n\n()")

    def test_selected_use_before_the_binding_does_not_see_a_reference_before_it(self) -> None:
        """Textual precedence holds for selected tails too: a reference walked
        before the binder registers still fails, even though the ``use``
        textually precedes both."""
        with pytest.raises(AglScopeError):
            parse_and_resolve(
                "scope B\n  use A::{x}\n  def get() -> int = x\nend B\n"
                "\n"
                "scope A\n  let x = 1\nend A\n\nB::get()"
            )

    def test_hiding_use_before_the_binding_does_not_see_a_reference_before_it(self) -> None:
        """Textual precedence holds for ``hiding`` too."""
        with pytest.raises(AglScopeError):
            parse_and_resolve(
                "scope B\n  use A::* hiding y\n  def get() -> int = x\nend B\n"
                "\n"
                "scope A\n  let x = 1\n  let y = 2\nend A\n\nB::get()"
            )

    def test_scope_use_sees_a_member_added_in_a_later_reopening(self) -> None:
        """``use A::*`` is recorded while region A is still empty (region B's
        first block); region A only gains ``x`` afterward, and a reference
        reaches it only once region B is reopened later still. The live scope-use
        re-check must answer correctly at that later point, not from
        whatever region A looked like when the ``use`` itself was walked."""
        resolved = parse_and_resolve(
            "scope B\n  use A::*\nend B\n"
            "\n"
            "scope A\n  var x = 1\nend A\n"
            "\n"
            "scope B\n  def get() -> int = x\nend B\n"
            "\n"
            "B::get()"
        )
        assert _ref(resolved, "x").scope_path == ("A",)

    def test_scope_use_sees_a_member_registered_after_an_earlier_reference_resolved(self) -> None:
        """The target region gains a second member *between* two bare
        references that both go through the same ``use``: the first
        reference resolves against the region as it stands then, and the
        second must still reach the member registered after it."""
        resolved = parse_and_resolve(
            "scope A\n  var x = 1\nend A\n"
            "\n"
            "scope B\n  use A::*\n  def first() -> int = x\nend B\n"
            "\n"
            "scope A\n  var y = 2\nend A\n"
            "\n"
            "scope B\n  def second() -> int = y\nend B\n"
            "\n"
            "B::first() + B::second()"
        )
        assert _ref(resolved, "x").scope_path == ("A",)
        assert _ref(resolved, "y").scope_path == ("A",)

    def test_selected_tail_and_hiding_of_the_same_items_are_exact_complements(self) -> None:
        """``use A::{x, y}`` and ``use A::* hiding x, y`` select complementary
        members of the same three-member scope: the selected tail reaches
        ``x``/``y`` bare and leaves ``z`` unreachable, while ``hiding``
        reaches ``z`` bare and leaves ``x``/``y`` unreachable. Both selection
        forms share one implementation, so this pins the two branches against
        each other for an identical item set."""
        resolved_using = parse_and_resolve(
            "use A::{x, y}\n\nscope A\n  var x = 1\n  var y = 2\n  var z = 3\nend A\n"
            "\n"
            "x := x + 1\ny := y + 1\nx + y"
        )
        assert _ref(resolved_using, "x").scope_path == ("A",)
        assert _ref(resolved_using, "y").scope_path == ("A",)
        with pytest.raises(AglScopeError):
            parse_and_resolve(
                "use A::{x, y}\n\nscope A\n  var x = 1\n  var y = 2\n  var z = 3\nend A\n\nz"
            )

        resolved_hiding = parse_and_resolve(
            "use A::* hiding x, y\n\nscope A\n  var x = 1\n  var y = 2\n  var z = 3\nend A\n\nz"
        )
        assert _ref(resolved_hiding, "z").scope_path == ("A",)
        with pytest.raises(AglScopeError):
            parse_and_resolve(
                "use A::* hiding x, y\n\nscope A\n  var x = 1\n  var y = 2\n  var z = 3\nend A\n\nx"
            )

    def test_hiding_use_reaches_the_non_hidden_member_before_the_hidden_one_is_registered(
        self,
    ) -> None:
        """A ``use … hiding`` exclusion set does not depend on whether the
        excluded item currently matches anything in the target: the
        non-hidden member resolves correctly through a live reference walked
        while the hidden member has not yet been registered."""
        resolved = parse_and_resolve(
            "scope B\n  use A::* hiding y\nend B\n"
            "\n"
            "scope A\n  var x = 1\nend A\n"
            "\n"
            "scope B\n  def mid() -> int = x\nend B\n"
            "\n"
            "scope A\n  var y = 2\nend A\n"
            "\n"
            "B::mid()"
        )
        assert _ref(resolved, "x").scope_path == ("A",)


class TestScopedConstructorCandidateUnion:
    """``_owned_scope_constructor_candidates`` unions every same-named candidate.

    A scope's own directly-declared constructor and every child enum's
    variant sharing a name are all candidates, in declaration order, rather
    than whichever the resolver happens to find first while scanning a set
    -- ``_type_declarations`` is a plain list, so the result no longer
    depends on ``PYTHONHASHSEED``.
    """

    def _pattern(self, resolved: ModuleResolution) -> ConstructorPattern:
        region = resolved.program.body.items[0]
        func = next(item for item in region.items if isinstance(item, FuncDef))
        case = func.body.items[0]
        assert isinstance(case, Case)
        pattern = case.branches[0].pattern
        assert isinstance(pattern, ConstructorPattern)
        return pattern

    def test_two_enums_sharing_a_variant_name_in_one_scope_union_deterministically(
        self,
    ) -> None:
        resolved = parse_and_resolve(
            "scope S\n"
            "  enum A\n"
            "    | V(x: int)\n"
            "  enum B\n"
            "    | V(y: text)\n"
            "  def f(a: A) -> int =\n"
            "    case a of\n"
            "    | V(x) => x\n"
            "end S\n"
            "\n"
            "S::f(S::A::V(x = 1))\n"
        )
        pattern = self._pattern(resolved)
        candidates = resolved.pattern_constructor_candidates[pattern.node_id]
        owners = {candidate.owner_path for candidate in candidates}
        assert owners == {("S", "A"), ("S", "B")}

    def test_scope_own_constructor_and_child_enum_variant_both_stay_candidates(self) -> None:
        """A scope's own record and a child enum's variant sharing a name both survive."""
        resolved = parse_and_resolve(
            "scope Config\n"
            "  record V\n"
            "    n: int\n"
            "  enum E\n"
            "    | V(m: int)\n"
            "  def pick(x: V) -> int =\n"
            "    case x of\n"
            "    | V(n) => n\n"
            "end Config\n"
            "\n"
            "Config::pick(Config::V(n = 1))\n"
        )
        pattern = self._pattern(resolved)
        candidates = resolved.pattern_constructor_candidates[pattern.node_id]
        owners = {(candidate.owner_path, candidate.owner_name) for candidate in candidates}
        assert owners == {(("Config",), "V"), (("Config", "E"), "V")}

    def test_outward_walk_prefers_the_nearest_scope_layer(self) -> None:
        """A nested scope's own same-named record shadows an ancestor's."""
        resolved = parse_and_resolve(
            "scope A\n"
            "  record Item\n"
            "    label: text\n"
            "\n"
            "  scope B\n"
            "    record Item\n"
            "      value: int\n"
            "    def pick(i: Item) -> int =\n"
            "      case i of\n"
            "      | Item(value) => value\n"
            "  end B\n"
            "end A\n"
            "\n"
            "A::B::pick(A::B::Item(value = 5))\n"
        )
        region_a = resolved.program.body.items[0]
        region_b = next(item for item in region_a.items if isinstance(item, ScopeRegion))
        func = next(item for item in region_b.items if isinstance(item, FuncDef))
        case = func.body.items[0]
        assert isinstance(case, Case)
        pattern = case.branches[0].pattern
        assert isinstance(pattern, ConstructorPattern)
        candidates = resolved.pattern_constructor_candidates[pattern.node_id]
        owners = {candidate.owner_path for candidate in candidates}
        assert owners == {("A", "B")}


class TestConstructorCandidateDeduplication:
    """``dedupe_constructor_candidates`` keeps each distinct candidate once."""

    def test_keeps_first_occurrence_of_each_distinct_candidate_across_interleaving(
        self,
    ) -> None:
        """A repeated candidate does not resurface after a differing same-id candidate.

        Both candidates below share ``owner_decl_node_id`` but disagree on their
        canonical metadata, so they are genuinely distinct declarations. The
        first candidate then recurs later in the sequence; only its first
        occurrence should survive, regardless of what was seen in between.
        """
        from agm.agl.scope.symbols import ConstructorRef, dedupe_constructor_candidates

        first = ConstructorRef(owner_name="A", owner_decl_node_id=1, type_params=())
        second = ConstructorRef(owner_name="B", owner_decl_node_id=1, type_params=())
        deduped = dedupe_constructor_candidates((first, second, first))
        assert deduped == (first, second)


class TestScopedAssignment:
    """Qualified assignment to a scoped member.

    ``A::count := e`` consults the same per-path member namespace a qualified
    read consults, so a scoped ``var`` is assignable through its path while a
    scoped ``let`` -- or a ``def``, a type, or an agent sharing its path --
    reuses the immutable-binder diagnostic.
    """

    def _assign_ref(self, resolved: ModuleResolution) -> BindingRef:
        assign = next(item for item in resolved.program.body.items if isinstance(item, AssignStmt))
        return resolved.resolution[assign.node_id]

    def test_qualified_assign_to_scoped_var_at_root(self) -> None:
        resolved = parse_and_resolve("scope A\n  var count = 0\nend A\n\nA::count := 1\nA::count")
        ref = self._assign_ref(resolved)
        assert ref.mutable is True
        assert ref.scope_path == ("A",)
        assert ref.kind is BinderKind.var_binding

    def test_qualified_assign_resolves_use_alias(self) -> None:
        resolved = parse_and_resolve(
            "use A as X\n\nscope A\n  var count = 0\nend A\n\nX::count := 1\nX::count"
        )
        ref = self._assign_ref(resolved)
        assert ref.mutable is True
        assert ref.scope_path == ("A",)
        assert ref.kind is BinderKind.var_binding

    def test_qualified_assign_to_multi_segment_scoped_var(self) -> None:
        resolved = parse_and_resolve(
            "scope A\n"
            "\n"
            "  scope B\n"
            "    var count = 0\n"
            "  end B\n"
            "end A\n"
            "\n"
            "A::B::count := 1\n"
            "A::B::count"
        )
        ref = self._assign_ref(resolved)
        assert ref.scope_path == ("A", "B")

    def test_bare_assign_inside_own_region_resolves(self) -> None:
        resolved = parse_and_resolve(
            "scope A\n"
            "  var count = 0\n"
            "  def bump() -> unit =\n"
            "    count := count + 1\n"
            "    ()\n"
            "end A\n"
            "\n"
            "A::bump()"
        )
        assign = next(
            item for item in resolved.program.body.items[0].items if isinstance(item, FuncDef)
        ).body.items[0]
        assert isinstance(assign, AssignStmt)
        ref = resolved.resolution[assign.node_id]
        assert ref.scope_path == ("A",)
        assert ref.mutable is True

    def test_qualified_assign_to_unknown_member_is_a_focused_error(self) -> None:
        err = reject_scope("scope A\n  var count = 0\nend A\n\nA::missing := 2\n()")
        _, msg = diag(err)
        assert "missing" in msg
        assert "A" in msg


class TestScopeUseEnumOwners:
    """A local ``use`` must expose its enum type to `is` tests and `case`."""

    def test_is_test_resolves_enum_owner_contributed_by_a_scope_use(self) -> None:
        r = parse_and_resolve(
            "use A::*\n"
            "\n"
            "scope A\n"
            "  enum Status\n"
            "    | Good\n"
            "    | Bad\n"
            "end A\n"
            "\n"
            "let s = Status::Good\n"
            "s is Status::Good\n"
        )
        from agm.agl.syntax.nodes import IsTest
        from agm.agl.syntax.visitor import walk

        found: list[IsTest] = []
        walk(r.program, lambda node: found.append(node) if isinstance(node, IsTest) else None)
        assert len(found) == 1
        cref = r.constructor_refs[found[0].node_id]
        assert (cref.owner_path, cref.owner_name) == (("A", "Status"), "Good")

    def test_case_pattern_resolves_enum_owner_contributed_by_a_scope_use(self) -> None:
        r = parse_and_resolve(
            "use A::*\n"
            "\n"
            "scope A\n"
            "  enum Status\n"
            "    | Good\n"
            "    | Bad\n"
            "end A\n"
            "\n"
            "let s = Status::Good\n"
            "case s of\n"
            "  | Status::Good => 1\n"
            "  | Status::Bad => 2\n"
        )
        from agm.agl.syntax.nodes import Case

        case_node = r.program.body.items[-1]
        assert isinstance(case_node, Case)
        good_pattern = case_node.branches[0].pattern
        assert isinstance(good_pattern, ConstructorPattern)
        cref = r.constructor_refs[good_pattern.node_id]
        assert (cref.owner_path, cref.owner_name) == (("A", "Status"), "Good")


# ---------------------------------------------------------------------------
# Basic acceptance
# ---------------------------------------------------------------------------


class TestAcceptance:
    def test_simple_let(self) -> None:
        r = parse_and_resolve('let x = "hello"\nx')
        assert _ref(r, "x").kind == BinderKind.let_binding

    def test_let_and_print(self) -> None:
        r = parse_and_resolve('let x = "hello"\nprint x')
        assert _ref(r, "x").kind == BinderKind.let_binding

    def test_var_and_assign(self) -> None:
        r = parse_and_resolve("var n: int = 0\nn := 1\nn")
        ref = _ref(r, "n")
        assert ref.kind == BinderKind.var_binding
        assert ref.mutable

    def test_param_at_root(self) -> None:
        r = parse_and_resolve("param spec\nspec")
        assert _ref(r, "spec").kind == BinderKind.param_binding

    def test_param_with_type(self) -> None:
        r = parse_and_resolve("param spec: text\nprint spec")
        assert _ref(r, "spec").kind == BinderKind.param_binding

    def test_let_referencing_a_typed_let_binding(self) -> None:
        r = parse_and_resolve('let spec: text = "x"\nlet x = spec\nx')
        assert _ref(r, "x").kind == BinderKind.let_binding

    def test_let_with_interpolation(self) -> None:
        r = parse_and_resolve('let name: text = "x"\nlet greeting = "Hello %{name}"\ngreeting')
        assert _ref(r, "greeting").kind == BinderKind.let_binding

    def test_simple_let_captured_by_function_still_resolves(self) -> None:
        resolved = parse_and_resolve_repl("let value = 1\ndef capture() = value\ncapture()")

        captured = _find_varref(resolved.program, "value")
        assert resolved.resolution[captured.node_id].kind is BinderKind.let_binding

    def test_destructuring_let_captured_by_function_still_resolves(self) -> None:
        resolved = parse_and_resolve_repl(
            "record Pair\n"
            "  value: int\n"
            "let Pair(value) = Pair(value = 1)\n"
            "def capture() = value\n"
            "capture()"
        )

        captured = _find_varref(resolved.program, "value")
        assert resolved.resolution[captured.node_id].kind is BinderKind.pattern_slot

    def test_multiple_inputs(self) -> None:
        r = parse_and_resolve("param spec\nparam max_severity: int\nspec")
        assert _ref(r, "spec").kind == BinderKind.param_binding

    def test_unit_lit(self) -> None:
        r = parse_and_resolve("()")
        assert r.resolution == {}

    def test_type_alias_at_root(self) -> None:
        r = parse_and_resolve("type MyText = text\n()")
        assert "MyText" in r.declared_type_names

    def test_record_def_at_root(self) -> None:
        r = parse_and_resolve("record P\n  n: int\n()")
        lookup_p = r.root_scope.lookup("P")
        assert lookup_p is not None
        assert lookup_p.kind == BinderKind.constructor_binding

    def test_enum_def_at_root(self) -> None:
        r = parse_and_resolve("enum E\n  | A\n  | B\n()")
        assert "A" in r.constructor_candidates
        assert r.constructor_candidates["A"][0].owner_name == "A"

    def test_raise_expr(self) -> None:
        r = parse_and_resolve("raise 1\n")
        assert r.resolution == {}


# ---------------------------------------------------------------------------
# Block forward-scoping and isolation
# ---------------------------------------------------------------------------


class TestBlockScoping:
    def test_forward_binding_visible_after_let(self) -> None:
        """A name bound by ``let`` is visible to all subsequent items."""
        r = parse_and_resolve("let x = 1\nlet y = x\ny")
        assert _ref(r, "y").kind == BinderKind.let_binding

    def test_binding_not_visible_before_let(self) -> None:
        """A name is NOT visible before its binder in the same block."""
        err = reject_scope("let y = x\nlet x = 1\ny")
        line, msg = diag(err)
        assert line == 1
        assert "x" in msg

    def test_block_local_binding_does_not_escape(self) -> None:
        """A binding inside a nested block is not visible after the block."""
        err = reject_scope("try\n  let inner = 1\n  inner\ncatch _ =>\n  ()\ninner\n")
        line, msg = diag(err)
        assert line == 6
        assert "inner" in msg

    def test_outer_binding_visible_in_nested_block(self) -> None:
        """A binding from an outer scope is visible in nested blocks."""
        r = parse_and_resolve("let outer = 42\nif true =>\n  outer\n| else =>\n  outer\n")
        # Both VarRefs to "outer" (in then-branch and else-branch) resolve to the let_binding.
        assert _ref(r, "outer", occurrence=0).kind == BinderKind.let_binding
        assert _ref(r, "outer", occurrence=1).kind == BinderKind.let_binding

    def test_redeclaration_same_scope_let_let(self) -> None:
        err = reject_scope("let x = 1\nlet x = 2\nx")
        line, msg = diag(err)
        assert line == 2
        assert "x" in msg

    def test_redeclaration_same_scope_let_var(self) -> None:
        err = reject_scope("let twice = 1\nvar twice = 2\ntwice")
        line, msg = diag(err)
        assert line == 2
        assert "twice" in msg

    def test_redeclaration_same_scope_var_var(self) -> None:
        err = reject_scope("var a = 1\nvar a = 2\na")
        line, msg = diag(err)
        assert line == 2
        assert "a" in msg

    def test_redeclaration_input_with_let(self) -> None:
        with pytest.raises(AglScopeError) as exc_info:
            parse_and_resolve_file('param spec\nlet spec = "again"')
        err = exc_info.value
        line, msg = diag(err)
        assert line == 2
        assert "spec" in msg

    def test_redeclaration_input_with_input(self) -> None:
        err = reject_scope("param x\nparam x\nx")
        line, msg = diag(err)
        assert line == 2
        assert "x" in msg


class TestLetPatternCompatibility:
    def test_destructuring_let_creates_continuation_slots_before_later_rejection(self) -> None:
        resolved = parse_and_resolve("let value = 0\nlet Pair(left, right) = value\nleft")
        assert _ref(resolved, "left").kind is BinderKind.pattern_slot

    def test_wildcard_let_resolves_its_rhs_without_creating_a_binding(self) -> None:
        resolved = parse_and_resolve("let value = 1\nlet _ = value\n()")
        assert _ref(resolved, "value").kind is BinderKind.let_binding
        assert "_" not in resolved.root_scope.bindings


class TestLetPatternScope:
    def test_simple_let_binding_identity_is_its_pattern_node(self) -> None:
        resolved = parse_and_resolve_file("let value = 1")
        declaration = resolved.program.body.items[0]
        assert isinstance(declaration, LetDecl)
        binding = resolved.root_scope.lookup("value")
        assert binding is not None

        assert binding.decl_node_id == declaration.pattern.node_id

    def test_destructuring_let_binding_identity_is_each_binder_pattern_node(self) -> None:
        resolved = parse_and_resolve(
            "record Pair\n"
            "  left: int\n"
            "  right: int\n"
            "let Pair(left, right) = Pair(left = 1, right = 2)\n"
            "left\n"
            "right"
        )
        declaration = resolved.program.body.items[1]
        assert isinstance(declaration, LetDecl)
        assert isinstance(declaration.pattern, ConstructorPattern)
        left_pattern, right_pattern = declaration.pattern.positional
        assert isinstance(left_pattern, VarPattern)
        assert isinstance(right_pattern, VarPattern)

        left_ref = _find_varref(resolved.program, "left")
        right_ref = _find_varref(resolved.program, "right")
        assert resolved.resolution[left_ref.node_id].decl_node_id == left_pattern.node_id
        assert resolved.resolution[right_ref.node_id].decl_node_id == right_pattern.node_id

    def test_root_let_name_binds_despite_a_visible_constructor(self) -> None:
        resolved = parse_and_resolve("enum Flag\n  | on\nlet on = 1\nlet copied = on\ncopied")
        assert _ref(resolved, "on").kind is BinderKind.let_binding
        declaration = resolved.program.body.items[1]
        assert isinstance(declaration, LetDecl)
        assert resolved.pattern_constructor_candidates[declaration.pattern.node_id]

    def test_let_initializer_cannot_see_its_own_pattern(self) -> None:
        reject_scope("let value = value\nvalue")

    def test_ast_wildcard_name_does_not_create_a_binding(self) -> None:
        resolved = resolve_program(_make_let("_", _make_intlit()))
        assert "_" not in resolved.root_scope.bindings

    def test_unimported_qualified_constructor_pattern_stays_a_checker_concern(self) -> None:
        parse_and_resolve("let value = 1\ncase value of | missing::packet(_) => 1")

    def test_destructuring_let_resolves_nested_binders_for_its_continuation(self) -> None:
        resolved = parse_and_resolve(
            "let value = 0\nlet Pair(left, _ as right) = value\nleft\nright"
        )
        assert _ref(resolved, "left").kind is BinderKind.pattern_slot
        assert _ref(resolved, "right").kind is BinderKind.pattern_slot
        assert {slot.binder_kind for slot in resolved.pattern_slots.values()} == {
            BinderKind.let_binding
        }
        declaration = resolved.program.body.items[1]
        assert isinstance(declaration, LetDecl)
        assert resolved.match_site_pattern_slots[declaration.node_id]


class TestWildcardBinders:
    def test_wildcard_binders_resolve_their_rhs_without_binding_a_name(self) -> None:
        resolved = parse_and_resolve("let value = 1\nlet _ = value\nvar _ = value\n()")
        assert _ref(resolved, "value", occurrence=0).kind is BinderKind.let_binding
        assert "_" not in resolved.root_scope.bindings

    def test_wildcard_binders_are_repeatable(self) -> None:
        parse_and_resolve("let _ = 1\nlet _ = 2\nvar _ = 3\n()")

    def test_wildcard_binder_is_not_readable(self) -> None:
        err = reject_scope("let _ = 1\n_")
        _, msg = diag(err)
        assert "_" in msg
        assert "not defined" in msg

    @pytest.mark.parametrize(
        "source",
        (
            "def f(_: int) -> int = (let _ = 0; _)\nf(1)",
            "for _ in [1] do (let _ = 0; _) done",
            "try () catch _ as _ => (let _ = 0; _)",
        ),
        ids=("parameter", "loop", "catch"),
    )
    def test_wildcard_is_not_readable_through_an_enclosing_binding(self, source: str) -> None:
        err = reject_scope(source)
        _, msg = diag(err)
        assert "_" in msg
        assert "not defined" in msg


# ---------------------------------------------------------------------------
# Assignment errors
# ---------------------------------------------------------------------------


class TestAssignErrors:
    def test_assign_to_undeclared(self) -> None:
        err = reject_scope("ghost := 1")
        line, msg = diag(err)
        assert line == 1
        assert "ghost" in msg

    @pytest.mark.parametrize(
        "source",
        (
            "let stable = 1\nstable := 2\nstable",
            "param spec\nspec := 2\nspec",
            "try\n  ()\ncatch _ as err =>\n  err := 1\n",
            "def f(x: int) -> int = x\nf := 1\nf(1)",
        ),
        ids=("let", "param", "catch-binder", "function"),
    )
    def test_unqualified_assign_to_immutable_resolves_and_defers_to_typecheck(
        self, source: str
    ) -> None:
        """Scope resolves the target; only typechecking knows a slot's mutability."""
        resolved = parse_and_resolve(source)
        assert resolved.resolution

    def test_assign_to_pattern_slot_resolves_to_the_slot_binding(self) -> None:
        resolved = parse_and_resolve("let v = 1\ncase v of\n  | _ as n =>\n    n := 2\n")
        assignment = resolved.program.body.items[1].branches[0].body.items[0]
        assert isinstance(assignment, AssignStmt)
        assert resolved.resolution[assignment.node_id].kind is BinderKind.pattern_slot

    def test_assign_to_var_resolves(self) -> None:
        r = parse_and_resolve("var n = 0\nn := 1\nn")
        # Verify the assignment statement is in the resolution table.
        block = r.program.body
        assign_node = block.items[1]
        assert isinstance(assign_node, AssignStmt)
        ref = r.resolution[assign_node.node_id]
        assert ref.name == "n"
        assert ref.mutable is True


# ---------------------------------------------------------------------------
# Undefined name reads
# ---------------------------------------------------------------------------


class TestUndefinedRead:
    def test_undefined_in_assignment(self) -> None:
        err = reject_scope("let x = undeclared\nx")
        line, msg = diag(err)
        assert line == 1
        assert "undeclared" in msg

    def test_undefined_in_interpolation(self) -> None:
        err = reject_scope('let x = "Hi %{ghost}"\nx')
        line, msg = diag(err)
        assert line == 1
        assert "ghost" in msg

    def test_undefined_callee(self) -> None:
        err = reject_scope("no_such_func(1)")
        line, msg = diag(err)
        assert line == 1
        assert "no_such_func" in msg

    def test_raise_undefined_rejected(self) -> None:
        err = reject_scope("raise nope\n")
        assert "nope" in err.to_diagnostic().message


# ---------------------------------------------------------------------------
# Reserved names: built-in call names cannot be bound
# ---------------------------------------------------------------------------


class TestReservedNames:
    def test_reserve_print_let(self) -> None:
        err = reject_scope('let print = "x"\nprint')
        line, msg = diag(err)
        assert line == 1
        assert "print" in msg

    def test_reserve_render_let(self) -> None:
        err = reject_scope('let render = "x"\nrender')
        line, msg = diag(err)
        assert line == 1
        assert "render" in msg

    def test_reserve_ask_let(self) -> None:
        err = reject_scope('let ask = "not allowed"\nask')
        line, msg = diag(err)
        assert line == 1
        assert "ask" in msg

    def test_reserve_exec_var(self) -> None:
        err = reject_scope('var exec = "not allowed"\nexec')
        line, msg = diag(err)
        assert line == 1
        assert "exec" in msg

    def test_reserve_ask_input(self) -> None:
        err = reject_scope("param ask")
        line, msg = diag(err)
        assert line == 1
        assert "ask" in msg

    def test_reserve_exec_input(self) -> None:
        err = reject_scope("param exec")
        line, msg = diag(err)
        assert line == 1
        assert "exec" in msg

    def test_reserve_print_input(self) -> None:
        err = reject_scope("param print")
        line, msg = diag(err)
        assert line == 1
        assert "print" in msg

    def test_reserve_ask_def(self) -> None:
        err = reject_scope("def ask() -> int = 1\nask()")
        _, msg = diag(err)
        assert "ask" in msg

    def test_reserve_exec_def(self) -> None:
        err = reject_scope("def exec() -> int = 1\nexec()")
        _, msg = diag(err)
        assert "exec" in msg

    def test_reserve_print_def(self) -> None:
        err = reject_scope("def print() -> int = 1\nprint()")
        _, msg = diag(err)
        assert "print" in msg

    def test_reserve_print_def_still_rejected_beside_an_unrelated_type(self) -> None:
        err = reject_scope("record Point\n  x: int\ndef print() -> int = 1\n()")
        _, msg = diag(err)
        assert "print" in msg
        assert "built-in" in msg.lower()

    def test_reserve_ask_param(self) -> None:
        err = reject_scope("def f(ask: int) -> int = 1\nf(1)")
        _, msg = diag(err)
        assert "ask" in msg

    def test_reserve_exec_param(self) -> None:
        err = reject_scope("def f(exec: int) -> int = 1\nf(1)")
        _, msg = diag(err)
        assert "exec" in msg

    def test_reserve_print_param(self) -> None:
        err = reject_scope("def f(print: int) -> int = 1\nf(1)")
        _, msg = diag(err)
        assert "print" in msg

    def test_reserve_ask_catch_binder(self) -> None:
        err = reject_scope("try\n  ()\ncatch _ as ask =>\n  ()\n")
        _, msg = diag(err)
        assert "ask" in msg
        assert "reserved" in msg.lower()

    def test_reserve_exec_catch_binder(self) -> None:
        err = reject_scope("try\n  ()\ncatch _ as exec =>\n  ()\n")
        _, msg = diag(err)
        assert "exec" in msg

    def test_top_level_bare_ask_is_not_a_binder(self) -> None:
        let_x = _make_let("x", _make_intlit(1))
        pv = VarPattern(name="ask", span=_sp(2), node_id=_nid())
        branch = CaseBranch(
            pattern=pv,
            body=_make_unitlit(),
            span=_sp(2),
            node_id=_nid(),
        )
        from agm.agl.syntax.nodes import Case

        case_node = Case(
            subject=_make_varref("x"),
            branches=(branch,),
            span=_sp(2),
            node_id=_nid(),
        )
        err = reject_program(let_x, case_node)
        msg = err.to_diagnostic().message
        assert "ask" in msg
        assert "constructor" in msg.lower()

    def test_bare_ask_varref_rejected(self) -> None:
        """A bare VarRef to 'ask' (not in call position) is rejected."""
        err = reject_scope("let f = ask")
        _, msg = diag(err)
        assert "ask" in msg

    def test_bare_exec_varref_rejected(self) -> None:
        err = reject_scope("let f = exec")
        _, msg = diag(err)
        assert "exec" in msg

    def test_bare_print_varref_rejected(self) -> None:
        err = reject_scope("let f = print")
        _, msg = diag(err)
        assert "print" in msg


# ---------------------------------------------------------------------------
# Builtin-var declaration placement
# ---------------------------------------------------------------------------


class TestBuiltinVarPlacement:
    def test_entry_module_declaration_rejected(self) -> None:
        err = reject_scope("builtin var max-iters: int\n()")
        line, message = diag(err)
        assert "std/config" in message
        assert line == 1

    def test_entry_module_scoped_declaration_still_rejected(self) -> None:
        """The region relaxation lifts only the scope-path clause; the module
        restriction stands, so a scoped ``builtin var`` outside ``std/config``
        is rejected the same way as a root one."""
        err = reject_scope("scope Region\n  builtin var max-iters: int\nend Region\n\n()")
        line, message = diag(err)
        assert "std/config" in message
        assert line == 2

    def test_scoped_declaration_inside_a_function_body_still_rejected(self) -> None:
        err = reject_scope('def f() -> text =\n  builtin var runner: text\n  "x"\nf()')
        line, message = diag(err)
        assert "runner" in message, "the diagnostic names the misplaced declaration"
        assert "nested block" in message
        assert line == 2


# ---------------------------------------------------------------------------
# Scoped `builtin` declarations (def/record/enum/exception): member,
# duplicate, and visibility rules, same as every other scoped declaration.
# ---------------------------------------------------------------------------


class TestScopedBuiltinDeclarations:
    def test_builtin_def_bare_visible_inside_its_own_region(self) -> None:
        resolved = parse_and_resolve(
            "scope Host\n  builtin def native() -> int\n"
            "  def read() -> int = native()\nend Host\n\nHost::read()"
        )
        assert _ref(resolved, "native").kind is BinderKind.function_binding
        assert _ref(resolved, "native").scope_path == ("Host",)

    def test_builtin_def_exact_path_reference_from_outside(self) -> None:
        resolved = parse_and_resolve(
            "scope Host\n  builtin def native() -> int\nend Host\n\nHost::native()"
        )
        ref = resolved.resolution[
            next(
                item.callee.node_id
                for item in (resolved.program.body.items[1],)
                if isinstance(item, Call)
            )
        ]
        assert ref.scope_path == ("Host",)

    def test_builtin_def_visible_after_use(self) -> None:
        resolved = parse_and_resolve(
            "use Host::*\n\nscope Host\n  builtin def native() -> int\nend Host\n\nnative()"
        )
        call = resolved.program.body.items[2]
        assert isinstance(call, Call)
        assert isinstance(call.callee, VarRef)
        assert resolved.resolution[call.callee.node_id].scope_path == ("Host",)

    @pytest.mark.parametrize(
        ("source", "name"),
        (
            (
                "scope A\n  builtin def print[T](value: T) -> unit\n"
                "  builtin def print[T](value: T) -> unit\nend A\n\n()",
                "print",
            ),
            (
                "scope A\n"
                "  builtin\n"
                "  record ExecResult\n"
                "    x: int\n"
                "  let ExecResult = 1\n"
                "end A\n"
                "\n"
                "()",
                "ExecResult",
            ),
            (
                "scope A\n  builtin\n  record ExecResult\n    x: int\n"
                "  builtin\n  record ExecResult\n    y: int\nend A\n\n()",
                "ExecResult",
            ),
            (
                "scope A\n  builtin\n  enum ParsePolicy =\n    | Abort\n"
                "  builtin\n  enum ParsePolicy =\n    | Abort\nend A\n\n()",
                "ParsePolicy",
            ),
            (
                "scope A\n  builtin exception RangeError extends Exception()\n"
                "  builtin exception RangeError extends Exception()\nend A\n\n()",
                "RangeError",
            ),
        ),
        ids=(
            "def-vs-def",
            "record-vs-let",
            "record-vs-record",
            "enum-vs-enum",
            "exception-vs-exception",
        ),
    )
    def test_duplicate_at_the_same_path_is_rejected(self, source: str, name: str) -> None:
        err = reject_scope(source)
        message = err.to_diagnostic().message
        assert name in message
        assert "declared" in message


class TestScopedBuiltinUsedAsValueRejected:
    """A scoped ``builtin def`` referenced as a value (not called) is a clean
    scope error, matching the root-level guard for a bare unshadowed builtin
    name. Regression coverage for the reference-classification paths a
    scoped ``builtin def`` reaches (qualified chain, region-local lexical
    lookup) that a root ``builtin def`` never does, and that previously had
    no value-use guard at all."""

    def test_qualified_reference_used_as_a_value_is_rejected(self) -> None:
        err = reject_scope(
            "scope H\n"
            "  builtin def render[T](value: T) -> text\n"
            "end H\n"
            "\n"
            "let f = H::render\n"
            "print(f)"
        )
        line, message = diag(err)
        assert "render" in message
        assert "value" in message, "rejected for being used as a value, not for being unknown"
        assert line == 5

    def test_bare_reference_inside_its_own_region_used_as_a_value_is_rejected(self) -> None:
        err = reject_scope(
            "scope H\n"
            "  builtin def render[T](value: T) -> text\n"
            "  let f = render\n"
            "end H\n"
            "\n"
            "print(H::f)"
        )
        line, message = diag(err)
        assert "render" in message
        assert "value" in message, "rejected for being used as a value, not for being unknown"
        assert line == 3

    def test_qualified_reference_with_a_type_argument_used_as_a_value_is_rejected(self) -> None:
        err = reject_scope(
            "scope H\n  builtin def render[T](value: T) -> text\nend H\n"
            "\n"
            "let f = H::render[json]\nprint(f)"
        )
        line, message = diag(err)
        assert "render" in message
        assert "value" in message, "rejected for being used as a value, not for being unknown"
        assert line == 5

    def test_qualified_call_to_a_scoped_builtin_def_is_unaffected(self) -> None:
        """The value-use rejection must not reject the legitimate call form
        it is easy to conflate it with: resolution must still succeed."""
        resolved = parse_and_resolve(
            'scope H\n  builtin def render[T](value: T) -> text\nend H\n\nprint(H::render("1"))'
        )
        call_item = resolved.program.body.items[1]
        assert isinstance(call_item, Call)


class TestQualifiedMembersSharingBuiltinNames:
    """A qualified member may share a bare builtin name without shadowing it."""

    def test_qualified_scope_member_named_after_a_builtin_is_accepted(self) -> None:
        resolved = parse_and_resolve(
            "scope Codec\n  def render(value: int) -> int = value\nend Codec\n\nCodec::render(1)"
        )

        qualified = _find_varref(resolved.program, "render")
        assert qualified.qualifier is not None
        assert [segment.name for segment in qualified.qualifier.segments] == ["Codec"]
        assert qualified.qualifier.member == "render"
        # The qualified route reaches the scope's own declaration rather than
        # the builtin that shares the name.
        binding = _ref(resolved, "render")
        assert binding.scope_path == ("Codec",)
        assert binding.is_builtin is False

    def test_opened_member_does_not_intercept_the_bare_builtin(self) -> None:
        resolved = parse_and_resolve(
            "use Codec::*\n\nscope Codec\n  def render(value: int) -> int = value\nend Codec\n"
            "\n"
            "let value = render(1)\nCodec::render(1)"
        )

        bare = _find_varref(resolved.program, "render", occurrence=0)
        assert bare.qualifier is None
        assert BuiltinKind.RENDER in resolved.builtin_calls.values()


# ---------------------------------------------------------------------------
# Built-in call classification (builtin_calls side table)
# ---------------------------------------------------------------------------


class TestBuiltinCallClassification:
    def test_print_call_classified(self) -> None:
        r = parse_and_resolve("let x = 1\nprint x")
        # find the Call node in the block
        call_item = r.program.body.items[1]
        assert isinstance(call_item, Call)
        assert r.builtin_calls[call_item.node_id] == BuiltinKind.PRINT

    def test_render_call_classified(self) -> None:
        r = parse_and_resolve("let x = render 1\nx")
        let_node = r.program.body.items[0]
        assert isinstance(let_node, LetDecl)
        assert isinstance(let_node.value, Call)
        assert r.builtin_calls[let_node.value.node_id] == BuiltinKind.RENDER

    def test_exec_call_classified(self) -> None:
        r = parse_and_resolve('let x = exec "ls"\nx')
        let_node = r.program.body.items[0]
        assert isinstance(let_node, LetDecl)
        assert isinstance(let_node.value, Call)
        assert r.builtin_calls[let_node.value.node_id] == BuiltinKind.EXEC

    def test_ask_call_classified(self) -> None:
        r = parse_and_resolve('let x = ask "Q"\nx')
        let_node = r.program.body.items[0]
        assert isinstance(let_node, LetDecl)
        assert isinstance(let_node.value, Call)
        assert r.builtin_calls[let_node.value.node_id] == BuiltinKind.ASK

    def test_ask_request_with_type_arg_classified(self) -> None:
        r = parse_and_resolve('let x = ask-request::[text]("Q")\nx')
        let_node = r.program.body.items[0]
        assert isinstance(let_node, LetDecl)
        assert isinstance(let_node.value, Call)
        assert len(let_node.value.type_args) == 1
        assert r.builtin_calls[let_node.value.node_id] == BuiltinKind.ASK_REQUEST

    def test_ask_request_without_type_arg_classified(self) -> None:
        r = parse_and_resolve('let x = ask-request("Q")\nx')
        let_node = r.program.body.items[0]
        assert isinstance(let_node, LetDecl)
        assert isinstance(let_node.value, Call)
        assert let_node.value.type_args == ()
        assert r.builtin_calls[let_node.value.node_id] == BuiltinKind.ASK_REQUEST

    def test_ask_request_callee_resolves_to_its_builtin_declaration(self) -> None:
        """A bare ``ask-request`` callee resolves like any other reference —
        to std/prelude's own ``builtin def`` — before the call is classified."""
        r = parse_and_resolve('let x = ask-request("Q")\nx')
        let_node = r.program.body.items[0]
        assert isinstance(let_node, LetDecl)
        call = let_node.value
        assert isinstance(call, Call)
        callee = call.callee
        assert isinstance(callee, VarRef)
        assert callee.node_id in r.resolution
        assert r.resolution[callee.node_id].kind is BinderKind.function_binding
        assert r.resolution[callee.node_id].name == "ask-request"

    def test_ask_request_reserved_as_value(self) -> None:
        # ``ask-request`` is a reserved contextual keyword: a bare reference
        # (not in call position) is rejected.
        with pytest.raises(AglScopeError) as exc_info:
            parse_and_resolve("let x = ask-request\nx")
        msg = str(exc_info.value)
        assert "ask-request" in msg
        assert "value" in msg, "rejected for being used as a value, not for being unknown"

    def test_user_def_call_not_classified(self) -> None:
        """A user-defined function call does NOT appear in builtin_calls."""
        r = parse_and_resolve("def f(x: int) -> int = x\nlet y = f(1)\ny")
        let_node = r.program.body.items[1]
        assert isinstance(let_node, LetDecl)
        assert isinstance(let_node.value, Call)
        # User call: not in builtin_calls
        assert let_node.value.node_id not in r.builtin_calls

    @pytest.mark.parametrize(
        ("method_name", "builtin_call", "builtin_kind"),
        (
            ("resource", 'resource("asset")', BuiltinKind.RESOURCE),
            ("resource-dir", "resource-dir()", BuiltinKind.RESOURCE_DIR),
        ),
    )
    def test_builtin_provenance_distinguishes_qualified_user_method(
        self, method_name: str, builtin_call: str, builtin_kind: BuiltinKind
    ) -> None:
        r = parse_and_resolve(
            "record P()\n"
            f"def P::{method_name}(self) -> P = self\n"
            f"let builtin_value = {builtin_call}\n"
            f"let user_value = P::{method_name}(P())\n"
            "user_value"
        )
        method = r.program.body.items[1]
        builtin_let = r.program.body.items[2]
        user_let = r.program.body.items[3]
        assert isinstance(method, FuncDef)
        assert isinstance(builtin_let, LetDecl)
        assert isinstance(builtin_let.value, Call)
        assert isinstance(builtin_let.value.callee, VarRef)
        builtin_ref = r.resolution[builtin_let.value.callee.node_id]
        assert builtin_ref.is_builtin
        assert r.builtin_calls[builtin_let.value.node_id] is builtin_kind
        assert isinstance(user_let, LetDecl)
        assert isinstance(user_let.value, Call)
        assert isinstance(user_let.value.callee, VarRef)
        method_ref = r.resolution[user_let.value.callee.node_id]
        assert method_ref.decl_node_id == method.node_id
        assert method_ref.scope_path == ("P",)
        assert not method_ref.is_builtin
        assert user_let.value.node_id not in r.builtin_calls

    def test_lambda_call_not_classified(self) -> None:
        """Calling a lambda-bound name is not in builtin_calls."""
        r = parse_and_resolve("let f = fn(x: int) => x\nlet y = f(1)\ny")
        let_y = r.program.body.items[1]
        assert isinstance(let_y, LetDecl)
        assert isinstance(let_y.value, Call)
        assert let_y.value.node_id not in r.builtin_calls
        # The lambda binding was resolved
        call = let_y.value
        assert isinstance(call.callee, VarRef)
        assert call.callee.node_id in r.resolution
        assert r.resolution[call.callee.node_id].name == "f"

    def test_print_call_callee_resolves_to_its_builtin_declaration(self) -> None:
        """The callee VarRef of a built-in call resolves like any other
        reference — to std/prelude's own ``builtin def print`` — before the call
        is classified in ``builtin_calls``."""
        r = parse_and_resolve("let x = 1\nprint x")
        call_item = r.program.body.items[1]
        assert isinstance(call_item, Call)
        callee = call_item.callee
        assert isinstance(callee, VarRef)
        assert callee.node_id in r.resolution
        assert r.resolution[callee.node_id].kind is BinderKind.function_binding
        assert r.resolution[callee.node_id].name == "print"

    def test_print_positional_arg_resolved(self) -> None:
        """The argument to print IS resolved."""
        r = parse_and_resolve("let x = 1\nprint x")
        call_item = r.program.body.items[1]
        assert isinstance(call_item, Call)
        arg = call_item.args[0]
        assert isinstance(arg, VarRef)
        assert arg.node_id in r.resolution
        assert r.resolution[arg.node_id].name == "x"

    def test_scoped_builtin_def_bare_call_inside_its_region_classified(self) -> None:
        """A scoped ``builtin def print`` called bare, from within its own
        region, still dispatches to the host implementation."""
        r = parse_and_resolve(
            "scope Host\n  builtin def print[T](value: T) -> unit\n"
            "  def announce() -> unit = print(1)\nend Host\n\nHost::announce()"
        )
        region = r.program.body.items[0]
        assert isinstance(region, ScopeRegion)
        announce = next(
            item for item in region.items if isinstance(item, FuncDef) and item.name == "announce"
        )
        assert isinstance(announce.body, Call)
        assert r.builtin_calls[announce.body.node_id] == BuiltinKind.PRINT

    def test_scoped_builtin_def_qualified_call_classified(self) -> None:
        """A scoped ``builtin def print`` called through its full path still
        dispatches to the host implementation."""
        r = parse_and_resolve(
            "scope Host\n  builtin def print[T](value: T) -> unit\nend Host\n\nHost::print(1)"
        )
        call_item = r.program.body.items[1]
        assert isinstance(call_item, Call)
        assert r.builtin_calls[call_item.node_id] == BuiltinKind.PRINT

    def test_qualified_call_to_an_undeclared_reserved_name_is_a_scope_error(self) -> None:
        """Regression: a qualified call ending in a reserved name must resolve
        the qualifier for real, not fall through to host dispatch just
        because the tail name happens to be reserved."""
        with pytest.raises(AglScopeError):
            parse_and_resolve("scope Host\nend Host\n\nHost::print(1)")

    def test_two_scoped_builtin_defs_at_different_paths_both_classified(self) -> None:
        """Same-named builtin defs at different paths do not collide, and each
        still dispatches to the host implementation via its own path."""
        r = parse_and_resolve(
            "scope A\n  builtin def print[T](value: T) -> unit\nend A\n"
            "\n"
            "scope B\n  builtin def print[T](value: T) -> unit\nend B\n"
            "\n"
            "A::print(1)\nB::print(2)"
        )
        a_call, b_call = r.program.body.items[2], r.program.body.items[3]
        assert isinstance(a_call, Call)
        assert isinstance(b_call, Call)
        assert r.builtin_calls[a_call.node_id] == BuiltinKind.PRINT
        assert r.builtin_calls[b_call.node_id] == BuiltinKind.PRINT


# ---------------------------------------------------------------------------
# Uniform Call: named args resolved
# ---------------------------------------------------------------------------


class TestPlaceholderCallResolution:
    def test_positional_placeholder_args_resolve_without_diagnostics(self) -> None:
        r = parse_and_resolve("def f(x: int, y: int) -> int = x\nlet h = f(?1, ?2)\nh")
        assert _ref(r, "f").kind == BinderKind.function_binding
        assert _ref(r, "h").kind == BinderKind.let_binding

    def test_named_placeholder_args_resolve_without_diagnostics(self) -> None:
        r = parse_and_resolve("def f(x: int, y: int) -> int = x\nlet h = f(x = ?, y = ?)\nh")
        assert _ref(r, "f").kind == BinderKind.function_binding
        assert _ref(r, "h").kind == BinderKind.let_binding

    def test_nested_placeholder_calls_resolve_without_diagnostics(self) -> None:
        r = parse_and_resolve(
            "def f(x: int, y: int) -> int = x\ndef g(x: int) -> int = x\nlet h = f(g(?), ?)\nh"
        )
        assert _ref(r, "f").kind == BinderKind.function_binding
        assert _ref(r, "g").kind == BinderKind.function_binding
        assert _ref(r, "h").kind == BinderKind.let_binding

    def test_non_placeholder_args_still_resolve(self) -> None:
        r = parse_and_resolve("def f(x: int, y: int) -> int = x\nlet x = 1\nlet h = f(?, x)\nh")
        assert _ref(r, "f").kind == BinderKind.function_binding
        assert _ref(r, "x", occurrence=1).kind == BinderKind.let_binding


class TestCallResolution:
    def test_user_call_positional_args_resolved(self) -> None:
        r = parse_and_resolve(
            "def add(a: int, b: int) -> int = a\nlet x = 1\nlet y = 2\nlet z = add(x, y)\nz"
        )
        let_z = r.program.body.items[3]
        assert isinstance(let_z, LetDecl)
        call = let_z.value
        assert isinstance(call, Call)
        for arg in call.args:
            assert isinstance(arg, VarRef)
            assert arg.node_id in r.resolution


# ---------------------------------------------------------------------------
# Top-level def: mutual recursion + forward references
# ---------------------------------------------------------------------------


class TestFuncDefMutualRecursion:
    def test_def_at_root_accepted(self) -> None:
        r = parse_and_resolve("def f(n: int) -> int = n\nf(1)")
        assert _ref(r, "f").kind == BinderKind.function_binding
        assert "f" in r.declared_functions

    def test_def_self_recursion(self) -> None:
        r = parse_and_resolve(
            "def fact(n: int) -> int = if n <= 1 => 1 | else => n * fact(n - 1)\nfact(5)"
        )
        assert _ref(r, "fact").kind == BinderKind.function_binding

    def test_def_mutual_recursion(self) -> None:
        """Two top-level defs can reference each other."""
        r = parse_and_resolve(
            "def even(n: int) -> bool = if n == 0 => true | else => odd(n - 1)\n"
            "def odd(n: int) -> bool = if n == 0 => false | else => even(n - 1)\n"
            "even(4)"
        )
        assert _ref(r, "even").kind == BinderKind.function_binding
        assert "even" in r.declared_functions
        assert "odd" in r.declared_functions

    def test_def_forward_reference(self) -> None:
        """A def can call another def declared AFTER it (pre-pass collects all)."""
        r = parse_and_resolve(
            "def caller(n: int) -> int = callee(n)\ndef callee(n: int) -> int = n\ncaller(1)"
        )
        assert _ref(r, "caller").kind == BinderKind.function_binding

    def test_def_body_sees_param(self) -> None:
        """Param names are in scope inside the body."""
        r = parse_and_resolve("def f(x: int) -> int = x\nf(1)")
        assert _ref(r, "x").kind == BinderKind.param_binding

    def test_def_params_not_visible_outside(self) -> None:
        """Param names are not visible outside the function body."""
        err = reject_scope("def f(x: int) -> int = x\nx\nf(1)")
        _, msg = diag(err)
        assert "x" in msg
        assert "not defined" in msg

    def test_def_nested_in_block_rejected(self) -> None:
        """A def nested inside a block (e.g. if branch) is rejected."""
        err = reject_scope("if true =>\n  def f(x: int) -> int = x\n| else =>\n  ()\n")
        _, msg = diag(err)
        assert "def" in msg.lower()
        assert "root" in msg.lower()

    def test_def_directly_in_function_body_rejected(self) -> None:
        err = reject_scope("def outer() -> int =\n  def helper() -> int = 1\n  helper()\nouter()")
        _, msg = diag(err)
        assert "def" in msg.lower()
        assert "root" in msg.lower()

    def test_def_duplicate_name_rejected(self) -> None:
        err = reject_scope("def f(x: int) -> int = x\ndef f(y: int) -> int = y\nf(1)")
        _, msg = diag(err)
        assert "f" in msg

    def test_def_name_in_resolution_table(self) -> None:
        """The function name VarRef in a call is resolved to the function binding."""
        r = parse_and_resolve("def f(x: int) -> int = x\nlet y = f(1)\ny")
        let_y = r.program.body.items[1]
        assert isinstance(let_y, LetDecl)
        call = let_y.value
        assert isinstance(call, Call)
        callee = call.callee
        assert isinstance(callee, VarRef)
        assert callee.node_id in r.resolution
        ref = r.resolution[callee.node_id]
        assert ref.name == "f"
        assert ref.kind == BinderKind.function_binding

    def test_def_param_default_resolved_in_enclosing_scope(self) -> None:
        """Parameter defaults are resolved in the DEFINITION scope (outer)."""
        r = parse_and_resolve_repl("let base = 10\ndef f(x: int = base) -> int = x\nf()")
        assert _ref(r, "base").kind == BinderKind.let_binding

    def test_return_allowed_in_def_body(self) -> None:
        parse_and_resolve("def f(x: int) -> int =\n  return x\nf(1)")

    def test_return_rejected_in_def_parameter_default(self) -> None:
        err = reject_scope("def f(x: int = (return 1)) -> int = x\nf()")
        _, msg = diag(err)
        assert "return" in msg
        assert "function" in msg

    def test_def_param_duplicate_rejected(self) -> None:
        err = reject_scope("def f(x: int, x: int) -> int = x\nf(1, 2)")
        _, msg = diag(err)
        assert "x" in msg


# ---------------------------------------------------------------------------
# Lambda scoping: non-self-recursive
# ---------------------------------------------------------------------------


class TestMethodReceiverClassification:
    @pytest.mark.parametrize(
        ("declaration", "owner"),
        (
            ("record RecordOwner()", "RecordOwner"),
            ("enum EnumOwner = value", "EnumOwner"),
            ("exception ExceptionOwner()", "ExceptionOwner"),
        ),
    )
    def test_receiver_in_each_nominal_type_scope_is_a_method(
        self, declaration: str, owner: str
    ) -> None:
        resolved = parse_and_resolve(f"{declaration}\ndef {owner}::identity(self) -> int = 1\n()")

        assert resolved.method_declarations == {
            (ENTRY_ID, (owner,), "identity"): (owner,),
        }

    def test_inline_member_type_scope_allows_methods(self) -> None:
        resolved = parse_and_resolve(
            "enum Tree = Leaf | Node(value: int)\ndef Tree::Node::identity(self) -> int = 1\n()"
        )

        assert resolved.method_declarations == {
            (ENTRY_ID, ("Tree", "Node"), "identity"): ("Tree", "Node"),
        }

    def test_method_named_after_a_builtin_call_is_exempt_from_the_reserved_name_rule(
        self,
    ) -> None:
        resolved = parse_and_resolve(
            "record Point\n  x: int\ndef Point::print(self) -> int = self.x\n()"
        )

        assert resolved.method_declarations == {
            (ENTRY_ID, ("Point",), "print"): ("Point",),
        }

    def test_shorthand_and_region_methods_have_identical_classification(self) -> None:
        resolved = parse_and_resolve(
            "record Point()\n"
            "def Point::shorthand(self) -> int = 1\n"
            "\n"
            "scope Point\n"
            "  def region(self) -> int = 2\n"
            "end Point\n"
            "\n"
            "()"
        )

        assert resolved.method_declarations == {
            (ENTRY_ID, ("Point",), "shorthand"): ("Point",),
            (ENTRY_ID, ("Point",), "region"): ("Point",),
        }

    @pytest.mark.parametrize(
        "source",
        (
            "builtin def Point::builtin_host(self) -> int",
            "extern def Point::extern_host(self) -> int",
        ),
        ids=("builtin", "extern"),
    )
    def test_bodyless_method_forms_are_classified(self, source: str, tmp_path: Path) -> None:
        # `extern def` is only accepted for an entry with a real backing file that
        # has a companion `.py` sibling; `builtin def` needs no such file, so only
        # the extern case needs a real tmp_path origin.
        origin_path = tmp_path / "program.agl"
        if "extern" in source:
            origin_path.with_suffix(".py").write_text("")
        resolved = parse_and_resolve(f"record Point()\n{source}\n()", origin_path=origin_path)
        name = "builtin_host" if "builtin" in source else "extern_host"

        assert resolved.method_declarations == {
            (ENTRY_ID, ("Point",), name): ("Point",),
        }

    @pytest.mark.parametrize(
        ("source", "alias", "target"),
        (
            ("type Count = int\ndef Count::value(self) -> int = 1", "Count", "int"),
            (
                "record Target()\ntype Alias = Target\ndef Alias::value(self) -> int = 1",
                "Alias",
                "Target",
            ),
        ),
    )
    def test_alias_scope_receiver_is_rejected_with_its_target(
        self, source: str, alias: str, target: str
    ) -> None:
        err = reject_scope(source)

        _, message = diag(err)
        assert alias in message
        assert target in message

    def test_receiver_is_bound_in_method_body_and_nested_lambda(self) -> None:
        resolved = parse_and_resolve(
            "record Point()\n"
            "def Point::identity(self) -> Point =\n"
            "  let delayed = fn() => self\n"
            "  delayed()\n"
            "()"
        )

        assert _ref(resolved, "self").kind is BinderKind.param_binding

    def test_unannotated_receiver_outside_a_type_scope_has_dedicated_diagnostic(self) -> None:
        err = reject_scope("def helper(self) -> int = 1")

        _, message = diag(err)
        assert "self" in message
        assert "type scope" in message

    @pytest.mark.parametrize(
        "receiver",
        ("self", "self: Point"),
        ids=("bare", "annotated"),
    )
    def test_receiver_with_a_default_is_rejected(self, receiver: str) -> None:
        err = reject_scope(
            f"record Point\n  x: int\ndef Point::bad({receiver} = Point(x = 1)) -> Point = self"
        )

        line, message = diag(err)
        assert "receiver" in message.lower()
        assert "default" in message
        assert line == 3

    def test_annotated_self_outside_a_type_scope_is_an_ordinary_parameter(self) -> None:
        resolved = parse_and_resolve("def helper(self: int) -> int = self\nhelper(1)")

        assert resolved.method_declarations == {}
        assert _ref(resolved, "self").kind is BinderKind.param_binding

    def test_unannotated_self_in_a_lambda_is_rejected(self) -> None:
        err = reject_scope("let helper = fn(self) -> int => 1")

        _, message = diag(err)
        assert "self" in message
        assert "lambda" in message


class TestLambdaScoping:
    def test_lambda_param_visible_in_body(self) -> None:
        r = parse_and_resolve("let f = fn(x: int) => x\nf(1)")
        assert _ref(r, "x").kind == BinderKind.param_binding

    def test_lambda_non_recursive(self) -> None:
        """Lambda body does NOT see the let-binding (f is not in scope in its RHS)."""
        err = reject_scope("let f = fn(x: int) => f(x)\nf(1)")
        _, msg = diag(err)
        assert "f" in msg
        assert "not defined" in msg

    def test_lambda_param_not_visible_outside(self) -> None:
        err = reject_scope("let f = fn(x: int) => x\nx\nf(1)")
        _, msg = diag(err)
        assert "x" in msg

    def test_lambda_captures_outer(self) -> None:
        """Lambda body can reference outer-scope bindings."""
        r = parse_and_resolve("let base = 10\nlet f = fn(x: int) => x\nf(1)")
        assert _ref(r, "f").kind == BinderKind.let_binding

    def test_lambda_param_reserved_rejected(self) -> None:
        err = reject_scope("let f = fn(print: int) => print\nf(1)")
        _, msg = diag(err)
        assert "print" in msg

    def test_lambda_call_is_not_builtin(self) -> None:
        """Calling a lambda via a variable does not produce a builtin_calls entry."""
        r = parse_and_resolve("let f = fn(x: int) => x\nlet y = f(1)\ny")
        let_y = r.program.body.items[1]
        assert isinstance(let_y, LetDecl)
        call = let_y.value
        assert isinstance(call, Call)
        assert call.node_id not in r.builtin_calls

    def test_lambda_default_in_enclosing_scope(self) -> None:
        """Default expressions in a lambda are resolved in the enclosing scope."""
        r = parse_and_resolve("let base = 5\nlet f = fn(x: int = base) => x\nf()")
        assert _ref(r, "base").kind == BinderKind.let_binding

    def test_return_allowed_in_lambda_body(self) -> None:
        parse_and_resolve("let f = fn(x: int) => return x\nf(1)")

    def test_return_rejected_at_top_level(self) -> None:
        with pytest.raises(AglScopeError) as exc_info:
            parse_and_resolve_repl("return 1")
        err = exc_info.value
        _, msg = diag(err)
        assert "return" in msg
        assert "function" in msg


# ---------------------------------------------------------------------------
# Agents as value bindings
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Do body/until scoping
# ---------------------------------------------------------------------------


class TestDoScoping:
    def test_do_body_binding_visible_in_until(self) -> None:
        """A binding defined in the do body is visible in the until condition."""
        r = parse_and_resolve("var n = 0\ndo[2]\n  let probe = n\n  n := probe\nuntil n >= 1\nn")
        ref = _ref(r, "n")
        assert ref.kind == BinderKind.var_binding
        assert ref.mutable

    def test_do_body_binding_not_visible_after_loop(self) -> None:
        """A binding from the do body is not visible after the loop."""
        err = reject_scope("do[2]\n  let inner = 1\nuntil true\ninner\n")
        line, msg = diag(err)
        assert line == 4
        assert "inner" in msg

    def test_do_inline_body_resolved(self) -> None:
        """Inline (non-block) do body is also resolved."""
        r = parse_and_resolve("var n = 0\ndo[2] n := 1 until n >= 1\nn")
        ref = _ref(r, "n")
        assert ref.kind == BinderKind.var_binding
        assert ref.mutable


# ---------------------------------------------------------------------------
# If expression scoping
# ---------------------------------------------------------------------------


class TestIfScoping:
    def test_if_condition_resolved(self) -> None:
        r = parse_and_resolve("let x = true\nif x => 1 | else => 2\n")
        assert _ref(r, "x").kind == BinderKind.let_binding

    def test_if_branch_body_local(self) -> None:
        r = parse_and_resolve("let x = 1\nif true =>\n  let y = x\n  y\n| else =>\n  x\n")
        assert _ref(r, "y").kind == BinderKind.let_binding

    def test_if_inner_not_visible_outside(self) -> None:
        err = reject_scope("if true =>\n  let inner = 1\n| else =>\n  ()\ninner\n")
        line, msg = diag(err)
        assert line == 5
        assert "inner" in msg

    def test_if_no_else_accepted(self) -> None:
        r = parse_and_resolve("let x = 1\nif x == 1 => print 1\n")
        assert _ref(r, "x").kind == BinderKind.let_binding


# ---------------------------------------------------------------------------
# Case expression scoping
# ---------------------------------------------------------------------------


class TestCaseScoping:
    def test_case_var_pattern_visible_in_body(self) -> None:
        r = parse_and_resolve("let x = 1\ncase x of\n  | _ as n => n\n")
        assert _ref(r, "n").kind == BinderKind.pattern_slot

    def test_case_pattern_not_visible_outside(self) -> None:
        err = reject_scope("let x = 1\ncase x of\n  | _ as n => n\nn\n")
        line, msg = diag(err)
        assert line == 4
        assert "n" in msg

    def test_case_wildcard_pattern(self) -> None:
        r = parse_and_resolve("let x = 1\ncase x of\n  | _ => 0\n")
        assert _ref(r, "x").kind == BinderKind.let_binding


# ---------------------------------------------------------------------------
# Try/catch scoping
# ---------------------------------------------------------------------------


class TestTryScoping:
    def test_try_body_resolved(self) -> None:
        r = parse_and_resolve("try\n  let x = 1\n  x\ncatch _ =>\n  0\n")
        assert _ref(r, "x").kind == BinderKind.let_binding

    def test_catch_binder_visible_in_catch_body(self) -> None:
        r = parse_and_resolve("try\n  1\ncatch _ as err =>\n  err\n")
        assert _ref(r, "err").kind == BinderKind.catch_binder

    def test_catch_binder_not_visible_outside(self) -> None:
        err = reject_scope("try\n  1\ncatch _ as err =>\n  err\nerr\n")
        line, msg = diag(err)
        assert line == 5
        assert "err" in msg


# ---------------------------------------------------------------------------
# parent_scope seam (incremental REPL sessions)
# ---------------------------------------------------------------------------


class TestParentScopeSeam:
    def test_default_none_is_standalone(self) -> None:
        err = reject_scope("print x")
        assert "not defined" in diag(err)[1]

    def test_reference_resolves_into_parent(self) -> None:
        """A VarRef to a parent-scope binding resolves through the parent."""
        session = parse_and_resolve_repl("let x = 1\nx")
        entry = resolve_entry("print x", parent_scope=session.root_scope)
        # The print's arg VarRef resolved to the session's let binding.
        call_item = entry.program.body.items[0]
        assert isinstance(call_item, Call)
        arg = call_item.args[0]
        assert isinstance(arg, VarRef)
        ref = entry.resolution[arg.node_id]
        assert ref.name == "x"

    def test_redeclaring_parent_name_shadows_without_error(self) -> None:
        """Redeclaring a parent-visible name shadows without error."""
        session = parse_and_resolve("let x = 1\nx")
        entry = resolve_entry("let x = 2\nx", parent_scope=session.root_scope)
        let_stmt = entry.program.body.items[0]
        assert isinstance(let_stmt, LetDecl)
        assert "x" in entry.root_scope.bindings
        assert entry.root_scope.bindings["x"].decl_node_id == let_stmt.pattern.node_id

    def test_assign_to_parent_mutable_resolves(self) -> None:
        """``:=`` of a parent var binding resolves through the parent."""
        session = parse_and_resolve_repl("var n: int = 0\nn")
        entry = resolve_entry("n := 1", parent_scope=session.root_scope)
        assign_stmt = entry.program.body.items[0]
        assert isinstance(assign_stmt, AssignStmt)
        ref = entry.resolution[assign_stmt.node_id]
        assert ref.name == "n"
        assert ref.mutable is True

    def test_assign_to_parent_immutable_resolves_as_immutable(self) -> None:
        """``:=`` of a parent let binding resolves; typechecking rejects it."""
        session = parse_and_resolve_repl("let k = 1\nk")
        entry = resolve_entry("k := 2", parent_scope=session.root_scope)
        assign_stmt = entry.program.body.items[0]
        assert isinstance(assign_stmt, AssignStmt)
        ref = entry.resolution[assign_stmt.node_id]
        assert ref.name == "k"
        assert ref.mutable is False

    def test_constructor_binding_with_no_candidates_does_not_error(self) -> None:
        """A constructor_binding from a parent scope with no ambient candidates
        is resolved (scope pass succeeds) but constructor_refs is NOT populated.
        This covers the len(candidates)==0 branch in _resolve_varref."""
        prior = parse_and_resolve("enum Review\n  | Pass\n  | Fail\nPass()")
        session_scope = prior.root_scope
        # No ambient_constructor_candidates passed → candidates is empty for 'Pass'.
        entry = resolve_entry(
            "Pass()",
            parent_scope=session_scope,
        )
        # Scope resolution succeeds but does NOT populate constructor_refs.
        from agm.agl.syntax.nodes import Call as _Call

        call_node = entry.program.body.items[0]
        assert isinstance(call_node, _Call)
        assert isinstance(call_node.callee, VarRef)
        # Without ambient candidates, constructor_refs is not populated.
        assert call_node.callee.node_id not in entry.constructor_refs

    def test_ambient_nullary_variant_retains_bare_pattern_metadata(self) -> None:
        prior = parse_and_resolve("enum Flag\n  | mark\nmark()")
        entry = resolve_entry(
            "enum Packet\n"
            "  | packet(left: int, right: int)\n"
            "let item = packet(1, 2)\n"
            "case item of | packet(mark, _ as mark) => mark",
            parent_scope=prior.root_scope,
            ambient_constructor_candidates=prior.constructor_candidates,
        )

        assert entry.constructor_candidates["mark"][0].can_match_bare_pattern
        slot = next(iter(entry.pattern_slots.values()))
        assert tuple(candidate.can_match_bare_pattern for candidate in slot.candidates) == (
            True,
            False,
        )

    @pytest.mark.parametrize(
        ("candidate_source", "declaration", "name", "type_name", "value"),
        (
            pytest.param(
                "local",
                "enum Mark\n  | mark(value: int)",
                "mark",
                "Mark",
                "mark(1)",
                id="local-payload-variant",
            ),
            pytest.param(
                "local",
                "record Mark\n  value: int",
                "Mark",
                "Mark",
                "Mark(value = 1)",
                id="local-record",
            ),
            pytest.param(
                "prelude",
                "",
                "Retry",
                "ParsePolicy",
                "Retry(n = 1)",
                id="prelude-payload-variant",
            ),
            pytest.param(
                "prelude",
                "",
                "ExecResult",
                "ExecResult",
                "ExecResult",
                id="prelude-record",
            ),
            pytest.param(
                "ambient",
                "enum Mark\n  | mark(value: int)",
                "mark",
                "Mark",
                "mark(1)",
                id="ambient-payload-variant",
            ),
            pytest.param(
                "ambient",
                "record Mark\n  value: int",
                "Mark",
                "Mark",
                "Mark(value = 1)",
                id="ambient-record",
            ),
        ),
    )
    @pytest.mark.parametrize("bare_first", (True, False), ids=("bare-then-as", "as-then-bare"))
    def test_payload_and_record_candidates_reject_duplicate_pattern_binders_eagerly(
        self,
        candidate_source: str,
        declaration: str,
        name: str,
        type_name: str,
        value: str,
        bare_first: bool,
    ) -> None:
        """Only bare nullary enum variants defer this duplicate diagnosis."""
        patterns = f"{name}, _ as {name}" if bare_first else f"_ as {name}, {name}"
        entry_source = (
            f"enum Packet\n  | packet(left: {type_name}, right: {type_name})\n"
            f"let item = packet({value}, {value})\n"
            f"case item of | packet({patterns}) => {name}"
        )

        with pytest.raises(AglScopeError):
            if candidate_source == "ambient":
                prior = parse_and_resolve(f"{declaration}\n{value}")
                resolve_entry(
                    entry_source,
                    parent_scope=prior.root_scope,
                    ambient_constructor_candidates=prior.constructor_candidates,
                    ambient_type_names=prior.declared_type_names,
                )
            else:
                parse_and_resolve(f"{declaration}\n{entry_source}")

    def test_ambient_constructor_candidates_resolve_prior_entry_ctor(self) -> None:
        """Constructor from a prior REPL entry resolves via ambient_constructor_candidates."""
        from agm.agl.scope.symbols import ConstructorRef

        # Simulate a prior entry that declared enum Review | Pass | Fail.
        prior = parse_and_resolve("enum Review\n  | Pass\n  | Fail\nPass()")
        # Build ambient candidates from the prior entry's resolution.
        ambient: dict[str, tuple[ConstructorRef, ...]] = {
            name: crefs for name, crefs in prior.constructor_candidates.items()
        }
        # New entry references Pass() with a parent scope that has the constructor binding.
        session_scope = prior.root_scope
        entry = resolve_entry(
            "Pass()",
            parent_scope=session_scope,
            ambient_constructor_candidates=ambient,
        )
        # The VarRef/Call for Pass() must be in constructor_refs.
        from agm.agl.syntax.nodes import Call as _Call

        call_node = entry.program.body.items[0]
        assert isinstance(call_node, _Call)
        assert isinstance(call_node.callee, VarRef)
        assert call_node.callee.node_id in entry.constructor_refs

    def test_type_name_shadowed_by_param_resolves_as_field_access(self) -> None:
        """When a type name is shadowed by a function parameter, a field
        access on it resolves as an ordinary value field access (not qualified
        constructor access).  Covers the constructor-access branch in _resolve_field_access."""
        # 'Box' is a type name AND a parameter name inside f.
        # Inside f, Box.x is a regular field access on the parameter, not a
        # qualified constructor reference.
        source = "record Box\n  x: int\ndef f(Box: Box) -> int = Box.x\nf(Box(x = 1))"
        entry = parse_and_resolve(source)
        from agm.agl.syntax.nodes import FieldAccess as _FA
        from agm.agl.syntax.nodes import FuncDef as _FD

        fn_node = entry.program.body.items[1]
        assert isinstance(fn_node, _FD)
        fa_node = fn_node.body
        assert isinstance(fa_node, _FA)
        # Field access on a parameter creates no constructor reference.
        assert fa_node.node_id not in entry.constructor_refs

    def test_ambient_type_names_resolve_qualified_prior_entry_ctor(self) -> None:
        """Qualified constructor from a prior REPL entry resolves via ambient_type_names."""
        from agm.agl.scope.symbols import ConstructorRef

        prior = parse_and_resolve("enum Review\n  | Pass\n  | Fail\nPass()")
        ambient_candidates: dict[str, tuple[ConstructorRef, ...]] = {
            name: crefs for name, crefs in prior.constructor_candidates.items()
        }
        ambient_type_names = prior.declared_type_names
        session_scope = prior.root_scope
        entry = resolve_entry(
            "Review::Pass()",
            parent_scope=session_scope,
            ambient_constructor_candidates=ambient_candidates,
            ambient_type_names=ambient_type_names,
        )
        from agm.agl.syntax.nodes import Call as _Call
        from agm.agl.syntax.nodes import VarRef as _VarRef

        call_node = entry.program.body.items[0]
        assert isinstance(call_node, _Call)
        assert isinstance(call_node.callee, _VarRef)
        assert call_node.callee.node_id in entry.constructor_refs


# ---------------------------------------------------------------------------
# Resolution side table: VarRef and AssignStmt
# ---------------------------------------------------------------------------


class TestResolutionSideTable:
    def test_varref_resolves_to_let(self) -> None:
        r = parse_and_resolve("let x = 1\nx")
        varref_item = r.program.body.items[1]
        assert isinstance(varref_item, VarRef)
        ref = r.resolution[varref_item.node_id]
        assert ref.name == "x"
        assert not ref.mutable

    def test_assign_resolves_to_var(self) -> None:
        r = parse_and_resolve("var n = 0\nn := 1\nn")
        assign_item = r.program.body.items[1]
        assert isinstance(assign_item, AssignStmt)
        ref = r.resolution[assign_item.node_id]
        assert ref.name == "n"
        assert ref.mutable

    def test_param_binding_is_immutable(self) -> None:
        r = parse_and_resolve("param spec\nspec")
        varref = r.program.body.items[1]
        assert isinstance(varref, VarRef)
        ref = r.resolution[varref.node_id]
        assert ref.name == "spec"
        assert not ref.mutable

    def test_interp_varref_resolved(self) -> None:
        r = parse_and_resolve('let name: text = "x"\nlet q = "Hello %{name}"\nq')
        let_q = r.program.body.items[1]
        assert isinstance(let_q, LetDecl)
        tmpl = let_q.value
        assert isinstance(tmpl, Template)
        from agm.agl.syntax.nodes import InterpSegment

        interp = next(s for s in tmpl.segments if isinstance(s, InterpSegment))
        assert isinstance(interp.expr, VarRef)
        ref = r.resolution[interp.expr.node_id]
        assert ref.name == "name"

    def test_function_binding_is_immutable(self) -> None:
        r = parse_and_resolve("def f(x: int) -> int = x\nlet y = f(1)\ny")
        call_callee = r.program.body.items[1]
        assert isinstance(call_callee, LetDecl)
        call = call_callee.value
        assert isinstance(call, Call)
        callee_ref = call.callee
        assert isinstance(callee_ref, VarRef)
        ref = r.resolution[callee_ref.node_id]
        assert ref.kind == BinderKind.function_binding
        assert not ref.mutable


# ---------------------------------------------------------------------------
# Direct AST construction tests for constructs not easily parsed
# ---------------------------------------------------------------------------


class TestDirectASTConstruction:
    """Test scope resolution for constructs built directly as AST nodes."""

    # --- Do loop ---

    def test_do_body_as_block_bindings_visible_in_condition(self) -> None:
        var_n = _make_var("n", _make_intlit(0))
        let_probe = _make_let("probe", _make_varref("n"))
        assign_n = _make_assign("n", _make_varref("probe"))
        body = _make_block(let_probe, assign_n)
        cond_probe = _make_varref("probe")
        do_node = Do(
            limit=5,
            body=body,
            condition=cond_probe,
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(var_n, do_node)
        # "probe" is declared in the do body and is visible in the until condition
        assert r.resolution[cond_probe.node_id].kind == BinderKind.let_binding

    def test_do_body_single_expr_resolved(self) -> None:
        var_n = _make_var("n", _make_intlit(0))
        body_n = _make_varref("n")
        do_node = Do(
            limit=5,
            body=body_n,
            condition=_make_boollit(True),
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(var_n, do_node)
        assert r.resolution[body_n.node_id].kind == BinderKind.var_binding

    # --- If with block bodies ---

    def test_if_block_body_scope(self) -> None:
        let_x = _make_let("x", _make_intlit(1))
        let_in = _make_let("inner", _make_varref("x"))
        inner_use = _make_varref("inner")
        body_block = _make_block(let_in, inner_use)
        branch = IfBranch(
            cond=_make_boollit(True),
            body=body_block,
            span=_sp(),
            node_id=_nid(),
        )
        if_node = If(branches=(branch,), span=_sp(), node_id=_nid())
        r = resolve_program(let_x, if_node)
        assert r.resolution[inner_use.node_id].kind == BinderKind.let_binding

    def test_if_block_inner_not_visible_outside(self) -> None:
        let_in = _make_let("inner", _make_intlit(1))
        body_block = _make_block(let_in, _make_varref("inner"))
        branch = IfBranch(
            cond=_make_boollit(True),
            body=body_block,
            span=_sp(),
            node_id=_nid(),
        )
        if_node = If(branches=(branch,), span=_sp(), node_id=_nid())
        read_after = _make_varref("inner", line=3)
        err = reject_program(if_node, read_after)
        assert "inner" in err.to_diagnostic().message

    # --- Try/catch ---

    def test_try_with_catch_binder(self) -> None:
        err_use = _make_varref("err")
        clause = CatchClause(
            exc_type="SomeError",
            binding="err",
            body=err_use,
            span=_sp(),
            node_id=_nid(),
        )
        try_node = Try(
            body=_make_intlit(1),
            handlers=(clause,),
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(try_node)
        assert r.resolution[err_use.node_id].kind == BinderKind.catch_binder

    def test_try_catch_binder_not_visible_outside(self) -> None:
        clause = CatchClause(
            exc_type=None,
            binding="err",
            body=_make_varref("err"),
            span=_sp(),
            node_id=_nid(),
        )
        try_node = Try(
            body=_make_intlit(1),
            handlers=(clause,),
            span=_sp(),
            node_id=_nid(),
        )
        read_after = _make_varref("err", line=3)
        err = reject_program(try_node, read_after)
        assert "err" in err.to_diagnostic().message

    # --- Constructor + operators ---

    def test_constructor_args_resolved(self) -> None:
        # Constructors are now ordinary Call nodes; a call to a record/enum
        # constructor is a Call whose callee is a VarRef.
        # Test that named-arg values in a constructor call are resolved.
        from agm.agl.syntax.nodes import EnumDef, NamedArg, VariantDef

        sp = _sp()
        variant = VariantDef(name="point", fields=(), span=sp, node_id=_nid())
        enum_def = EnumDef(name="Shape", members=(variant,), span=sp, node_id=_nid())
        let_n = _make_let("n", _make_intlit(5))
        arg = NamedArg(name="n", value=_make_varref("n"), span=sp, node_id=_nid())
        # Constructor call: Call(callee=VarRef("point"), named_args=[n: n])
        ctor_call = Call(
            callee=_make_varref("point"),
            args=(),
            named_args=(arg,),
            span=sp,
            node_id=_nid(),
        )
        r = resolve_program(enum_def, let_n, ctor_call)
        # The named-arg value (VarRef("n")) must be resolved
        assert arg.value.node_id in r.resolution
        # The callee VarRef("point") must be resolved as constructor_binding
        callee = ctor_call.callee
        assert isinstance(callee, VarRef)
        assert callee.node_id in r.resolution
        assert r.resolution[callee.node_id].kind == BinderKind.constructor_binding

    def test_binary_op_resolved(self) -> None:
        from agm.agl.syntax.nodes import BinaryOp, BinOp

        let_x = _make_let("x", _make_intlit(1))
        let_y = _make_let("y", _make_intlit(2))
        x_ref = _make_varref("x")
        y_ref = _make_varref("y")
        binop = BinaryOp(
            op=BinOp.ADD,
            left=x_ref,
            right=y_ref,
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(let_x, let_y, binop)
        assert r.resolution[x_ref.node_id].kind == BinderKind.let_binding
        assert r.resolution[y_ref.node_id].kind == BinderKind.let_binding

    def test_unary_not_resolved(self) -> None:
        from agm.agl.syntax.nodes import UnaryNot

        let_b = _make_let("b", _make_boollit(True))
        b_ref = _make_varref("b")
        expr = UnaryNot(operand=b_ref, span=_sp(), node_id=_nid())
        r = resolve_program(let_b, expr)
        assert r.resolution[b_ref.node_id].kind == BinderKind.let_binding

    def test_unary_neg_resolved(self) -> None:
        from agm.agl.syntax.nodes import UnaryNeg

        let_n = _make_let("n", _make_intlit(1))
        n_ref = _make_varref("n")
        expr = UnaryNeg(operand=n_ref, span=_sp(), node_id=_nid())
        r = resolve_program(let_n, expr)
        assert r.resolution[n_ref.node_id].kind == BinderKind.let_binding

    def test_is_test_resolved(self) -> None:
        from agm.agl.syntax.nodes import IsTest

        let_x = _make_let("x", _make_intlit(1))
        x_ref = _make_varref("x")
        expr = IsTest(
            expr=x_ref,
            qualifier=None,
            variant="Pass",
            negated=False,
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(let_x, expr)
        assert r.resolution[x_ref.node_id].kind == BinderKind.let_binding

    def test_field_access_on_varref(self) -> None:
        let_x = _make_let("x", _make_intlit(1))
        x_ref = _make_varref("x")
        field_expr = FieldAccess(obj=x_ref, field="f", span=_sp(), node_id=_nid())
        r = resolve_program(let_x, field_expr)
        assert r.resolution[x_ref.node_id].kind == BinderKind.let_binding

    def test_array_lit_resolved(self) -> None:
        from agm.agl.syntax.nodes import ArrayLit

        let_x = _make_let("x", _make_intlit(1))
        x_ref = _make_varref("x")
        lst = ArrayLit(elements=(x_ref,), span=_sp(), node_id=_nid())
        r = resolve_program(let_x, lst)
        assert r.resolution[x_ref.node_id].kind == BinderKind.let_binding

    def test_dict_lit_resolved(self) -> None:
        from agm.agl.syntax.nodes import DictEntry, DictLit

        let_x = _make_let("x", _make_intlit(1))
        key = StringLit(value="a", span=_sp(), node_id=_nid())
        x_ref = _make_varref("x")
        entry = DictEntry(key=key, value=x_ref, span=_sp(), node_id=_nid())
        dlit = DictLit(entries=(entry,), span=_sp(), node_id=_nid())
        r = resolve_program(let_x, dlit)
        assert r.resolution[x_ref.node_id].kind == BinderKind.let_binding

    # --- Pattern variable binding ---

    def test_case_var_pattern_binds(self) -> None:
        let_x = _make_let("x", _make_intlit(1))
        pattern_var = AsPattern(
            pattern=WildcardPattern(span=_sp(), node_id=_nid()),
            name="matched",
            span=_sp(),
            node_id=_nid(),
        )
        matched_ref = _make_varref("matched")
        branch = CaseBranch(
            pattern=pattern_var,
            body=matched_ref,
            span=_sp(),
            node_id=_nid(),
        )
        from agm.agl.syntax.nodes import Case

        case_node = Case(
            subject=_make_varref("x"),
            branches=(branch,),
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(let_x, case_node)
        assert r.resolution[matched_ref.node_id].kind == BinderKind.pattern_slot

    def test_case_constructor_pattern_with_field(self) -> None:
        let_x = _make_let("x", _make_intlit(1))
        sub_pattern = VarPattern(name="issues", span=_sp(), node_id=_nid())
        pf = PatternField(name="issues", pattern=sub_pattern, span=_sp(), node_id=_nid())
        ctor_pattern = ConstructorPattern(
            qualifier=None, name="Fail", positional=(), named=(pf,), span=_sp(), node_id=_nid()
        )
        issues_ref = _make_varref("issues")
        branch = CaseBranch(
            pattern=ctor_pattern,
            body=issues_ref,
            span=_sp(),
            node_id=_nid(),
        )
        from agm.agl.syntax.nodes import Case

        case_node = Case(
            subject=_make_varref("x"),
            branches=(branch,),
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(let_x, case_node)
        assert r.resolution[issues_ref.node_id].kind == BinderKind.pattern_slot

    def test_as_pattern_binds_even_when_name_is_a_constructor(self) -> None:
        resolved = parse_and_resolve(
            "enum E\n  | A\nlet value = A()\ncase value of | A() as A => A"
        )
        program = resolved.program
        branch = program.body.items[-1].branches[0]
        assert isinstance(branch.pattern, AsPattern)
        bound_ref = _find_varref(program, "A")
        binding = resolved.resolution[bound_ref.node_id]
        assert binding.kind is BinderKind.pattern_slot
        assert binding.decl_node_id == branch.pattern.node_id

    def test_as_pattern_duplicate_with_inner_field_binder_is_rejected_in_scope(self) -> None:
        reject_scope(
            "enum E\n  | A(value: int)\nlet value = A(1)\n"
            "case value of | A(value = value as captured) as captured => captured | _ => 0"
        )

    def test_as_pattern_duplicate_chain_binder_is_rejected_in_scope(self) -> None:
        reject_scope("case 0 of | _ as captured as captured => captured")

    def test_duplicate_pattern_var_rejected(self) -> None:
        let_x = _make_let("x", _make_intlit(1))
        sub1 = VarPattern(name="dup", span=_sp(5), node_id=_nid())
        sub2 = VarPattern(name="dup", span=_sp(5), node_id=_nid())
        pf1 = PatternField(name="a", pattern=sub1, span=_sp(5), node_id=_nid())
        pf2 = PatternField(name="b", pattern=sub2, span=_sp(5), node_id=_nid())
        ctor_pat = ConstructorPattern(
            qualifier=None,
            name="Pair",
            positional=(),
            named=(pf1, pf2),
            span=_sp(5),
            node_id=_nid(),
        )
        branch = CaseBranch(pattern=ctor_pat, body=_make_unitlit(), span=_sp(5), node_id=_nid())
        from agm.agl.syntax.nodes import Case

        case_node = Case(subject=_make_varref("x"), branches=(branch,), span=_sp(5), node_id=_nid())
        err = reject_program(let_x, case_node)
        assert "dup" in err.to_diagnostic().message

    def test_pattern_var_shadows_outer_accepted(self) -> None:
        let_outer = _make_let("v", _make_intlit(99))
        let_x = _make_let("x", _make_intlit(1))
        pv = AsPattern(
            pattern=WildcardPattern(span=_sp(), node_id=_nid()),
            name="v",
            span=_sp(),
            node_id=_nid(),
        )
        v_in_branch = _make_varref("v")
        branch = CaseBranch(pattern=pv, body=v_in_branch, span=_sp(), node_id=_nid())
        from agm.agl.syntax.nodes import Case

        case_node = Case(subject=_make_varref("x"), branches=(branch,), span=_sp(), node_id=_nid())
        r = resolve_program(let_outer, let_x, case_node)
        # The inner VarRef resolves to the branch slot, not the outer let "v".
        assert r.resolution[v_in_branch.node_id].kind == BinderKind.pattern_slot
        assert r.resolution[v_in_branch.node_id].decl_node_id == pv.node_id

    # --- Type declarations outside root rejected ---

    def test_record_not_at_root_rejected(self) -> None:
        err = reject_scope("if true =>\n  record R\n    n: int\n| else =>\n  ()\n")
        line, msg = diag(err)
        assert line == 2
        assert "'record'" in msg, "the diagnostic names the misplaced declaration kind"
        assert "top level" in msg.lower()

    def test_enum_not_at_root_rejected(self) -> None:
        err = reject_scope("do[2]\n  enum E\n    | A\nuntil true\n")
        line, msg = diag(err)
        assert line == 2
        assert "'enum'" in msg, "the diagnostic names the misplaced declaration kind"
        assert "top level" in msg.lower()

    def test_type_alias_not_at_root_rejected(self) -> None:
        err = reject_scope("try\n  type T = text\ncatch _ =>\n  ()\n")
        line, msg = diag(err)
        assert line == 2
        assert "'type'" in msg, "the diagnostic names the misplaced declaration kind"
        assert "top level" in msg.lower()

    # --- param not at root ---

    def test_param_inside_if_rejected(self) -> None:
        err = reject_scope("if true =>\n  param late\n| else =>\n  ()\n")
        line, msg = diag(err)
        assert line == 2
        assert "param" in msg.lower()

    def test_param_inside_do_rejected(self) -> None:
        err = reject_scope("do[2]\n  param x\nuntil true\n")
        line, msg = diag(err)
        assert line == 2
        assert "param" in msg.lower()

    def test_param_inside_try_rejected(self) -> None:
        err = reject_scope("try\n  param x\ncatch _ =>\n  ()\n")
        line, msg = diag(err)
        assert line == 2
        assert "param" in msg.lower()

    def test_program_inside_if_rejected(self) -> None:
        err = reject_scope("if true =>\n  program def nested() = ()\n| else =>\n  ()\n")
        line, msg = diag(err)
        assert line == 2
        assert "program" in msg.lower()
        assert "root" in msg.lower()

    def test_infix_decl_inside_if_rejected(self) -> None:
        err = reject_scope("if true =>\n  infixl |> at 12\n| else =>\n  ()\n")
        line, msg = diag(err)
        assert line == 2
        assert "infix" in msg.lower()
        assert "root" in msg.lower()

    def test_import_inside_if_rejected(self) -> None:
        err = reject_scope("if true =>\n  import foo\n  ()\n()\n")
        line, msg = diag(err)
        assert line == 2
        assert "import" in msg.lower()
        assert "root" in msg.lower()

    def test_export_inside_if_rejected(self) -> None:
        err = reject_scope("if true =>\n  export foo\n  ()\n()\n")
        line, msg = diag(err)
        assert line == 2
        assert "export" in msg.lower()
        assert "root" in msg.lower()

    def test_multiple_program_definitions_are_allowed(self) -> None:
        parse_and_resolve("program def first() = ()\nprogram def second() = ()\n")

    # --- FuncDef: direct AST construction ---

    def test_funcdef_body_resolved_in_param_scope(self) -> None:
        """FuncDef body sees its own param; param is not visible outside."""
        sp = _sp()
        int_t = IntT(span=sp, node_id=_nid())
        param = Param(
            name="x",
            type_expr=int_t,
            kind=ParamKind.STANDARD,
            default=None,
            span=sp,
            node_id=_nid(),
        )
        x_in_body = _make_varref("x")
        funcdef = FuncDef(
            name="g",
            params=(param,),
            return_type=int_t,
            body=x_in_body,
            span=sp,
            node_id=_nid(),
        )
        g_ref = _make_varref("g")
        r = resolve_program(funcdef, g_ref)
        assert r.resolution[x_in_body.node_id].kind == BinderKind.param_binding
        assert r.resolution[g_ref.node_id].kind == BinderKind.function_binding

    def test_funcdef_param_not_visible_outside_body(self) -> None:
        sp = _sp()
        int_t = IntT(span=sp, node_id=_nid())
        param = Param(
            name="p",
            type_expr=int_t,
            kind=ParamKind.STANDARD,
            default=None,
            span=sp,
            node_id=_nid(),
        )
        funcdef = FuncDef(
            name="g",
            params=(param,),
            return_type=int_t,
            body=_make_varref("p"),
            span=sp,
            node_id=_nid(),
        )
        read_p_after = _make_varref("p", line=2)
        err = reject_program(funcdef, read_p_after)
        assert "p" in err.to_diagnostic().message

    # --- Lambda: direct AST construction ---

    def test_lambda_resolved(self) -> None:
        sp = _sp()
        int_t = IntT(span=sp, node_id=_nid())
        param = Param(
            name="x",
            type_expr=int_t,
            kind=ParamKind.STANDARD,
            default=None,
            span=sp,
            node_id=_nid(),
        )
        x_in_lam = _make_varref("x")
        lam = Lambda(
            params=(param,),
            return_type=None,
            body=x_in_lam,
            span=sp,
            node_id=_nid(),
        )
        let_f = _make_let("f", lam)
        f_ref = _make_varref("f")
        r = resolve_program(let_f, f_ref)
        assert r.resolution[x_in_lam.node_id].kind == BinderKind.param_binding
        assert r.resolution[f_ref.node_id].kind == BinderKind.let_binding

    def test_lambda_not_self_recursive(self) -> None:
        """A lambda body that references its own let-binding name fails."""
        sp = _sp()
        int_t = IntT(span=sp, node_id=_nid())
        param = Param(
            name="x",
            type_expr=int_t,
            kind=ParamKind.STANDARD,
            default=None,
            span=sp,
            node_id=_nid(),
        )
        lam = Lambda(
            params=(param,),
            return_type=None,
            # references "f" which is not yet in scope when lambda RHS is resolved
            body=_make_varref("f"),
            span=sp,
            node_id=_nid(),
        )
        let_f = _make_let("f", lam)
        err = reject_program(let_f)
        assert "f" in err.to_diagnostic().message


# ---------------------------------------------------------------------------
# ModuleResolution.declared_functions
# ---------------------------------------------------------------------------


class TestDeclaredFunctions:
    def test_declared_functions_populated(self) -> None:
        r = parse_and_resolve("def f(x: int) -> int = x\ndef g(x: int) -> int = x\nf(1)")
        assert "f" in r.declared_functions
        assert "g" in r.declared_functions

    def test_declared_functions_empty_when_no_defs(self) -> None:
        r = parse_and_resolve_file("let x = 1")
        assert r.declared_functions == {}


# ---------------------------------------------------------------------------
# Block as expr (covers _resolve_expr for Block nodes)
# ---------------------------------------------------------------------------


class TestBlockAsExpr:
    def test_block_as_let_value(self) -> None:
        """A Block used as the RHS of a let (via direct AST) is resolved."""
        # let result = { let x = 1; x }
        let_x = _make_let("x", _make_intlit(1))
        inner_ref = _make_varref("x")
        inner_block = _make_block(let_x, inner_ref)
        let_result = _make_let("result", inner_block)
        result_ref = _make_varref("result")
        r = resolve_program(let_result, result_ref)
        assert r.resolution[inner_ref.node_id].kind == BinderKind.let_binding
        assert r.resolution[result_ref.node_id].kind == BinderKind.let_binding

    def test_block_as_expr_inner_binding_isolated(self) -> None:
        """Bindings in a Block expr don't escape outside the block."""
        let_x = _make_let("x", _make_intlit(1))
        inner_block = _make_block(let_x, _make_varref("x"))
        let_result = _make_let("result", inner_block)
        # Reading x after the block should fail
        read_x = _make_varref("x", line=2)
        err = reject_program(let_result, read_x)
        assert "x" in err.to_diagnostic().message


# ---------------------------------------------------------------------------
# Ambient agent binding edge cases
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lambda duplicate param
# ---------------------------------------------------------------------------


class TestLambdaDuplicateParam:
    def test_lambda_duplicate_param_rejected(self) -> None:
        """A lambda with two params with the same name is rejected."""
        sp = _sp()
        from agm.agl.syntax.types import IntT as IntTNode

        int_t = IntTNode(span=sp, node_id=_nid())
        p1 = Param(
            name="x",
            type_expr=int_t,
            kind=ParamKind.STANDARD,
            default=None,
            span=sp,
            node_id=_nid(),
        )
        p2 = Param(
            name="x",
            type_expr=int_t,
            kind=ParamKind.STANDARD,
            default=None,
            span=_sp(2),
            node_id=_nid(),
        )
        lam = Lambda(
            params=(p1, p2),
            return_type=None,
            body=_make_varref("x"),
            span=sp,
            node_id=_nid(),
        )
        let_f = _make_let("f", lam)
        err = reject_program(let_f)
        assert "x" in err.to_diagnostic().message


# ---------------------------------------------------------------------------
# Constructor value bindings (generics:  / scope pass)
# ---------------------------------------------------------------------------


# Helper: build a RecordDef with optional type-parameter slots
def _make_record(name: str, *, type_param_slots: tuple[str, ...] = (), line: int = 1) -> RecordDef:
    from agm.agl.syntax.nodes import Param, ParamKind
    from agm.agl.syntax.types import IntT as IntTNode

    sp = _sp(line)
    field_t = IntTNode(span=sp, node_id=_nid())
    fd = Param(
        name="value",
        type_expr=field_t,
        kind=ParamKind.NAMED_ONLY,
        default=None,
        span=sp,
        node_id=_nid(),
    )
    return RecordDef(
        name=name, fields=(fd,), type_param_slots=type_param_slots, span=sp, node_id=_nid()
    )


# Helper: build an EnumDef with variants
def _make_enum(
    name: str,
    variant_names: tuple[str, ...],
    *,
    type_param_slots: tuple[str, ...] = (),
    line: int = 1,
) -> EnumDef:
    sp = _sp(line)
    members: list[VariantDef] = []
    for vname in variant_names:
        members.append(VariantDef(name=vname, fields=(), span=sp, node_id=_nid()))
    return EnumDef(
        name=name,
        members=tuple(members),
        type_param_slots=type_param_slots,
        span=sp,
        node_id=_nid(),
    )


class TestConstructorBindings:
    """Tests for constructor value bindings (case-neutral constructor names)."""

    # --- Record constructor resolves as value binding ---

    def test_record_constructor_resolves_as_value(self) -> None:
        """A record constructor (record name) resolves as a constructor_binding."""
        r = parse_and_resolve("record Box\n  value: int\nlet b = Box(value = 1)\nb\n")
        # The callee VarRef("Box") should be in constructor_refs
        # and the constructor candidate is for 'Box'
        assert "Box" in r.constructor_candidates
        candidates = r.constructor_candidates["Box"]
        assert len(candidates) == 1
        assert candidates[0].owner_name == "Box"
        assert candidates[0].owner_path == ()

    def test_record_constructor_lowercase_resolves(self) -> None:
        """Lowercase record names work identically (no capitalization rule)."""
        r = parse_and_resolve("record box\n  value: int\nlet b = box(value = 1)\nb\n")
        assert "box" in r.constructor_candidates
        assert r.constructor_candidates["box"][0].owner_name == "box"

    def test_enum_variant_resolves_as_value(self) -> None:
        """Enum variants resolve as constructor_bindings."""
        r = parse_and_resolve("enum Option\n  | none\n  | some\nnone\n")
        assert "none" in r.constructor_candidates
        assert "some" in r.constructor_candidates

    def test_nullary_variant_bare_ref_resolves(self) -> None:
        """A bare VarRef to a nullary enum variant resolves as a constructor."""
        r = parse_and_resolve("enum Option\n  | none\n  | some\nlet x = none\nx\n")
        # The VarRef("none") in the let should be in constructor_refs
        let_decl = r.program.body.items[1]
        assert isinstance(let_decl, LetDecl)
        vref = let_decl.value
        assert isinstance(vref, VarRef)
        assert vref.node_id in r.constructor_refs
        cref = r.constructor_refs[vref.node_id]
        assert (cref.owner_path, cref.owner_name) == (("Option",), "none")

    def test_payload_variant_callee_resolves(self) -> None:
        """A payload variant used as a call callee resolves as a constructor."""
        r = parse_and_resolve("enum Option\n  | none\n  | some\nlet x = some()\nx\n")
        let_decl = r.program.body.items[1]
        assert isinstance(let_decl, LetDecl)
        call = let_decl.value
        assert isinstance(call, Call)
        callee = call.callee
        assert isinstance(callee, VarRef)
        assert callee.node_id in r.constructor_refs
        assert r.constructor_refs[callee.node_id].owner_name == "some"

    def test_record_constructor_callee_resolves(self) -> None:
        """A record constructor used as a call callee resolves."""
        r = parse_and_resolve("record Box\n  value: int\nlet b = Box(value = 1)\nb\n")
        let_decl = r.program.body.items[1]
        assert isinstance(let_decl, LetDecl)
        call = let_decl.value
        assert isinstance(call, Call)
        callee = call.callee
        assert isinstance(callee, VarRef)
        assert callee.node_id in r.constructor_refs
        cref = r.constructor_refs[callee.node_id]
        assert (cref.owner_path, cref.owner_name) == ((), "Box")

    # --- Generic type_params on constructors ---

    def test_generic_record_constructor_has_type_params(self) -> None:
        """A generic record constructor carries its type_params in the ConstructorRef."""
        r = parse_and_resolve("record Box[T]\n  value: int\nlet b = Box(value = 1)\nb\n")
        assert r.constructor_candidates["Box"][0].type_params == ("T",)

    def test_generic_member_captures_only_referenced_type_params(self) -> None:
        r = parse_and_resolve("enum Option[T]\n  | none\n  | some(value: T)\nnone\n")
        assert r.constructor_candidates["none"][0].type_params == ()
        assert r.constructor_candidates["some"][0].type_params == ("T",)

    # --- Overload sets and ambiguity ---

    def test_unique_variant_resolves_unambiguously(self) -> None:
        """A unique variant name (one enum) resolves to a single constructor_ref."""
        r = parse_and_resolve("enum A\n  | foo\n  | bar\nenum B\n  | baz\nfoo\n")
        last_item = r.program.body.items[2]
        assert isinstance(last_item, VarRef)
        assert last_item.node_id in r.constructor_refs

    def test_overload_set_built_from_two_enums(self) -> None:
        """Two enums sharing a variant name form an overload set.

        The overload set exists even if we don't actually USE 'some'.
        """
        enum_a = _make_enum("A", ("some", "none"), line=1)
        enum_b = _make_enum("B", ("some", "other"), line=2)
        unit = _make_unitlit()
        r = resolve_program(enum_a, enum_b, unit)
        assert "some" in r.constructor_candidates
        assert len(r.constructor_candidates["some"]) == 2
        owners = {c.owner_path for c in r.constructor_candidates["some"]}
        assert owners == {("A",), ("B",)}

    def test_local_enum_member_shadows_an_automatic_prelude_constructor(self) -> None:
        resolved = parse_and_resolve(
            "enum Result\n  | Ok(value: int)\n  | Err(error: text)\nOk(value = 1)"
        )

        result = resolved.program.body.items[-1]
        assert isinstance(result, Call)
        assert resolved.constructor_refs[result.callee.node_id].owner_path == ("Result",)

    def test_ambiguous_bare_varref_raises(self) -> None:
        """Unqualified use of an ambiguous variant name raises an ambiguity error."""
        err = reject_scope("enum A\n  | some\nenum B\n  | some\nsome\n")
        msg = err.to_diagnostic().message
        assert "some" in msg
        assert "ambiguous" in msg.lower()
        # Both owners should be mentioned
        assert "A" in msg
        assert "B" in msg

    def test_ambiguous_call_raises(self) -> None:
        """Calling an ambiguous constructor name also raises ambiguity."""
        err = reject_scope("enum A\n  | some\nenum B\n  | some\nsome()\n")
        msg = err.to_diagnostic().message
        assert "ambiguous" in msg.lower()
        assert "A" in msg
        assert "B" in msg

    def test_ambiguous_mentions_qualification(self) -> None:
        """Ambiguity error tells the user to qualify the reference."""
        err = reject_scope("enum Option\n  | some\nenum Other\n  | some\nsome\n")
        msg = err.to_diagnostic().message
        names = quoted_names(msg)
        # The ambiguous reference, then both owners, then the repair spelling.
        assert names[0] == "some"
        assert {"Option::some", "Other::some"} <= set(names)
        assert names[-1] == "Option::some", "the repair is a concrete qualified spelling"

    # ---  regression: payload / type-args / context do NOT disambiguate ---

    def test_ambiguous_payload_variant_still_raises(self) -> None:
        """A payload does NOT disambiguate two enums sharing the same variant name.

         guarantees ambiguity is raised by the scope pass regardless of whether
        a matching payload is supplied.  Both candidate enums must be named in the
        error message.
        """
        err = reject_scope(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "enum Other[T]\n"
            "  | nope\n"
            "  | some(value: T)\n"
            "some(value = 1)\n"
        )
        msg = err.to_diagnostic().message
        assert "ambiguous" in msg.lower()
        assert "Option" in msg
        assert "Other" in msg

    def test_ambiguous_explicit_type_args_still_raises(self) -> None:
        """Explicit type arguments do NOT disambiguate two enums sharing a variant name.

         guarantees that ``some::[int](value = 1)`` is still ambiguous when both
        ``Option`` and ``Other`` declare a ``some`` variant.  Both candidate enums
        must be named in the error message.
        """
        err = reject_scope(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "enum Other[T]\n"
            "  | nope\n"
            "  | some(value: T)\n"
            "some::[int](value = 1)\n"
        )
        msg = err.to_diagnostic().message
        assert "ambiguous" in msg.lower()
        assert "Option" in msg
        assert "Other" in msg

    def test_ambiguous_contextual_type_still_raises(self) -> None:
        """A contextual expected type does NOT disambiguate an ambiguous variant.

         guarantees that ``let x: Option[int] = some(value = 1)`` is still
        ambiguous when both ``Option`` and ``Other`` declare a ``some`` variant.
        Both candidate enums must be named in the error message.
        """
        err = reject_scope(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "enum Other[T]\n"
            "  | nope\n"
            "  | some(value: T)\n"
            "let x: Option[int] = some(value = 1)\n"
            "x\n"
        )
        msg = err.to_diagnostic().message
        assert "ambiguous" in msg.lower()
        assert "Option" in msg
        assert "Other" in msg

    def test_qualified_payload_variant_resolves_without_error(self) -> None:
        """Qualification is the only valid disambiguation for shared variant names.

        ``Option::some(value = 1)`` resolves without error even when ``Other``
        also declares a ``some`` variant.
        """
        r = parse_and_resolve(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "enum Other[T]\n"
            "  | nope\n"
            "  | some(value: T)\n"
            "Option::some(value = 1)\n"
        )
        call_node = r.program.body.items[2]
        assert isinstance(call_node, Call)
        assert isinstance(call_node.callee, VarRef)
        ref = r.constructor_refs[call_node.callee.node_id]
        assert (ref.owner_path, ref.owner_name) == (("Option",), "some")

    # --- Qualified constructor access ---

    def test_qualified_constructor_recorded(self) -> None:
        """Qualified access Owner::member records the shared constructor reference."""
        r = parse_and_resolve("enum Option\n  | none\n  | some\nlet x = Option::some\nx\n")
        let_decl = r.program.body.items[1]
        assert isinstance(let_decl, LetDecl)
        fa = let_decl.value
        assert isinstance(fa, VarRef)
        ref = r.constructor_refs[fa.node_id]
        assert (ref.owner_path, ref.owner_name) == (("Option",), "some")

    def test_qualified_access_does_not_raise_undefined_for_owner(self) -> None:
        "Option::some does NOT raise 'Option is not defined'."
        r = parse_and_resolve("enum Option\n  | none\n  | some\nOption::some\n")
        last = r.program.body.items[1]
        assert isinstance(last, VarRef)
        ref = r.constructor_refs[last.node_id]
        assert (ref.owner_path, ref.owner_name) == (("Option",), "some")

    def test_qualified_access_none_variant(self) -> None:
        "Option::none records the shared constructor reference."
        r = parse_and_resolve("enum Option\n  | none\n  | some\nOption::none\n")
        last = r.program.body.items[1]
        assert isinstance(last, VarRef)
        ref = r.constructor_refs[last.node_id]
        assert (ref.owner_path, ref.owner_name) == (("Option",), "none")

    def test_dot_access_with_type_name_is_rejected(self) -> None:
        err = reject_scope("record Box\n  value: int\nBox.value\n")
        line, message = diag(err)
        names = quoted_names(message)
        assert names[0] == "Box", "the offending object is named first"
        assert "type name" in message.lower()
        assert "::" in message, "the repair names the qualification operator"
        assert line == 3

    # --- Collision rules (non-constructor vs constructor) ---

    def test_ordinary_bindings_can_share_constructor_spellings(self) -> None:
        """Value lookup remains ordinary while pattern lookup retains constructors."""
        resolved = parse_and_resolve(
            "enum Option\n  | some\ndef some(x: int) -> int = x\nlet value = some(1)\n"
            "case Option::some of | some => value"
        )
        assert _ref(resolved, "some", occurrence=0).kind == BinderKind.function_binding
        case = resolved.program.body.items[-1]
        assert case.branches[0].pattern.node_id in resolved.pattern_constructor_candidates

    def test_constructor_can_share_existing_def_spelling(self) -> None:
        resolved = parse_and_resolve(
            "def some(x: int) -> int = x\nenum Option\n  | some\nsome(1)\n"
        )
        assert _ref(resolved, "some").kind == BinderKind.function_binding

    def test_let_shadows_constructor_in_if_branch_no_error(self) -> None:
        """In a branch, a 'let' name can shadow a constructor from the outer scope.

        The shadowing VarRef resolves to the let binding (not the constructor),
        while a VarRef in another branch resolves to the constructor.
        """
        r = parse_and_resolve(
            "enum Option\n  | some\nif true =>\n  let some = 42\n  some\n| else =>\n  some\n"
        )
        inner = _find_varref(r.program, "some", occurrence=0)
        outer = _find_varref(r.program, "some", occurrence=1)
        # The then-branch "some" resolves to the let_binding (shadowing)
        assert r.resolution[inner.node_id].kind == BinderKind.let_binding
        # The else-branch "some" resolves to the constructor_binding (not shadowed)
        assert r.resolution[outer.node_id].kind == BinderKind.constructor_binding

    def test_root_value_binding_does_not_hide_constructor_patterns(self) -> None:
        resolved = parse_and_resolve(
            "enum Option\n  | some\nlet some = 1\nlet value: Option = Option::some\n"
            "case value of | some => 0"
        )
        case = resolved.program.body.items[-1]
        assert case.branches[0].pattern.node_id in resolved.pattern_constructor_candidates

    # --- Enum name is NOT a value (only variants are) ---

    def test_enum_name_used_as_value_is_undefined(self) -> None:
        """The enum name itself is NOT a value binding — only its variants are.

        default_stdlib=False: this program declares its own ``Option``, which
        collides with the default prelude's ``std/option::Option[T]``.
        The point of this test is purely local ("does a bare reference to a
        locally-declared enum's own name resolve as a value"), independent of
        any module graph, so nothing else needs to be in scope.
        """
        err = reject_scope(
            "enum Option\n  | none\n  | some\nlet x = Option\nx\n", default_stdlib=False
        )
        msg = err.to_diagnostic().message
        assert "Option" in msg
        assert "not defined" in msg.lower()

    def test_uppercase_enum_name_not_value(self) -> None:
        """Enum name 'Option' (uppercase) is still not a value binding.

        default_stdlib=False for the same reason as
        test_enum_name_used_as_value_is_undefined above.
        """
        err = reject_scope("enum Option\n  | None\n  | Some\nOption\n", default_stdlib=False)
        msg = err.to_diagnostic().message
        assert "Option" in msg

    # --- Case-neutral: lowercase and uppercase behave identically ---

    def test_lowercase_constructor_resolves_same_as_uppercase(self) -> None:
        "Lowercase 'option::none' and uppercase 'Option::None' behave identically."
        r_lower = parse_and_resolve("enum option\n  | none\n  | some\noption::none\n")
        r_upper = parse_and_resolve("enum Option\n  | None\n  | Some\nOption::None\n")
        # Both have one constructor-chain result.
        assert len(r_lower.constructor_refs) == 1
        assert len(r_upper.constructor_refs) == 1

    def test_no_capitalization_rule_for_constructor_lookup(self) -> None:
        """Neither lowercase nor uppercase variants require capitalization to resolve."""
        r = parse_and_resolve(
            "enum option\n  | none\n  | someVal\nlet a = none\nlet b = someVal\na\n"
        )
        assert "none" in r.constructor_candidates
        assert "someVal" in r.constructor_candidates

    # --- Type parameter duplicate validation ---

    def test_duplicate_type_param_in_def_raises(self) -> None:
        """Duplicate type parameter in a def declaration raises AglScopeError."""
        err = reject_scope("def id[T, T](x: int) -> int = x\nid(1)\n")
        msg = err.to_diagnostic().message
        assert "'T'" in msg
        assert "duplicate" in msg.lower()

    def test_duplicate_type_param_in_record_raises(self) -> None:
        """Duplicate type parameter in a record declaration raises AglScopeError."""
        err = reject_scope("record Box[T, T]\n  value: int\n()\n")
        msg = err.to_diagnostic().message
        assert "'T'" in msg
        assert "duplicate" in msg.lower()

    def test_duplicate_type_param_in_enum_raises(self) -> None:
        """Duplicate type parameter in an enum declaration raises AglScopeError."""
        err = reject_scope("enum Option[T, T]\n  | none\n()\n")
        msg = err.to_diagnostic().message
        assert "'T'" in msg
        assert "duplicate" in msg.lower()

    def test_duplicate_type_param_in_type_alias_raises(self) -> None:
        """Duplicate type parameter in a type alias raises AglScopeError."""
        err = reject_scope("type Pair[A, A] = int\n()\n")
        msg = err.to_diagnostic().message
        assert "'A'" in msg
        assert "duplicate" in msg.lower()

    def test_duplicate_record_name_raises(self) -> None:
        err = reject_scope("record P\n  n: int\nrecord P\n  t: text\n()")
        msg = err.to_diagnostic().message
        assert "P" in msg
        assert "already declared" in msg.lower()

    def test_duplicate_enum_name_raises(self) -> None:
        err = reject_scope("enum Status\n  | ok\nenum Status\n  | fail\n()")
        msg = err.to_diagnostic().message
        assert "Status" in msg
        assert "already declared" in msg.lower()

    def test_duplicate_type_alias_name_raises(self) -> None:
        err = reject_scope("type Thing = int\ntype Thing = text\n()")
        msg = err.to_diagnostic().message
        assert "Thing" in msg
        assert "already declared" in msg.lower()

    def test_type_declaration_kinds_share_type_namespace(self) -> None:
        err = reject_scope("record Id\n  value: int\ntype Id = text\n()")
        msg = err.to_diagnostic().message
        assert "Id" in msg
        assert "already declared" in msg.lower()

    def test_unique_type_params_accepted(self) -> None:
        """Unique type parameters in a def are accepted."""
        r = parse_and_resolve("def id[T](x: int) -> int = x\nid(1)\n")
        assert _ref(r, "id").kind == BinderKind.function_binding

    def test_method_receiver_wildcard_type_params_may_repeat(self) -> None:
        """A method's receiver-prefix '_' type parameters resolve, including repeats."""
        r = parse_and_resolve(
            "record Pair[A, B]\n"
            "  first: A\n"
            "  second: B\n"
            "def Pair::describe[_, _](self) -> int = 1\n"
            "()\n"
        )
        assert [key for key in r.method_declarations if key[2] == "describe"]

    def test_non_method_wildcard_type_param_rejected(self) -> None:
        """A '_' type parameter on a function without a 'self' receiver is rejected."""
        err = reject_scope("def ignored[_, _](x: int) -> int = 1\nignored(1)\n")
        msg = err.to_diagnostic().message
        assert "ignored" in msg, "the diagnostic names the declaration carrying the '_' slot"
        assert "receiver" in msg

    def test_multiple_type_params_unique_accepted(self) -> None:
        """Multiple unique type params in a record are accepted."""
        r = parse_and_resolve("record Pair[A, B]\n  value: int\n()\n")
        lookup_pair = r.root_scope.lookup("Pair")
        assert lookup_pair is not None
        assert lookup_pair.kind == BinderKind.constructor_binding

    # --- declared_type_names populated ---

    def test_declared_type_names_includes_record(self) -> None:
        r = parse_and_resolve("record Foo\n  n: int\n()")
        assert "Foo" in r.declared_type_names

    def test_declared_type_names_includes_enum(self) -> None:
        r = parse_and_resolve("enum Color\n  | red\n  | blue\n()")
        assert "Color" in r.declared_type_names

    def test_declared_type_names_includes_alias(self) -> None:
        r = parse_and_resolve("type MyInt = int\n()")
        assert "MyInt" in r.declared_type_names

    def test_alias_with_unresolvable_unqualified_target_is_presumed_constructible(
        self,
    ) -> None:
        """A standalone module has no import environment.

        An alias whose unqualified target names no local declaration cannot be
        resolved through an import either (there is none to try), so the
        alias falls back to the permissive "presumed constructible" default
        and keeps its own variant-less constructor candidate.
        """
        r = parse_and_resolve("type Local = Undeclared\n()\n")
        candidates = r.constructor_candidates["Local"]
        assert candidates[0].owner_path == ()

    def test_alias_with_unresolvable_qualified_target_is_presumed_constructible(
        self,
    ) -> None:
        """Same as above, for a module-qualified target.

        A standalone module has no import environment to resolve the
        qualifier through, so the alias is presumed constructible.
        """
        r = parse_and_resolve("type Local = pal::Something\n()\n")
        candidates = r.constructor_candidates["Local"]
        assert candidates[0].owner_path == ()

    def test_declared_type_names_excludes_variants(self) -> None:
        """Enum variant names are NOT in declared_type_names (they are values)."""
        r = parse_and_resolve("enum Color\n  | red\n  | blue\n()")
        assert "red" not in r.declared_type_names
        assert "blue" not in r.declared_type_names

    # --- Direct AST construction tests ---

    def test_record_constructor_binding_via_ast(self) -> None:
        """Direct AST: a RecordDef registers its name as a constructor binding."""
        rec = _make_record("Point")
        call = Call(
            callee=_make_varref("Point"),
            args=(),
            named_args=(),
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(rec, call)
        assert "Point" in r.constructor_candidates
        assert r.constructor_candidates["Point"][0].owner_path == ()

    def test_enum_variant_binding_via_ast(self) -> None:
        """Direct AST: enum variants register as constructor candidates."""
        enum = _make_enum("Status", ("ok", "err"))
        ref_ok = _make_varref("ok")
        r = resolve_program(enum, ref_ok)
        assert "ok" in r.constructor_candidates
        assert r.constructor_candidates["ok"][0].owner_name == "ok"
        assert r.constructor_candidates["ok"][0].owner_name == "ok"
        assert ref_ok.node_id in r.constructor_refs

    def test_constructor_binding_kind_in_scope(self) -> None:
        """The root scope binding for a constructor name has kind=constructor_binding."""
        rec = _make_record("MyRecord")
        unit = _make_unitlit()
        r = resolve_program(rec, unit)
        assert "MyRecord" in r.root_scope.bindings
        assert r.root_scope.bindings["MyRecord"].kind == BinderKind.constructor_binding

    def test_overload_set_two_enums_via_ast(self) -> None:
        """Two enums sharing a variant name form a 2-element overload set."""
        enum_a = _make_enum("EnumA", ("val",), line=1)
        enum_b = _make_enum("EnumB", ("val",), line=2)
        unit = _make_unitlit()
        r = resolve_program(enum_a, enum_b, unit)
        assert len(r.constructor_candidates["val"]) == 2

    def test_qualified_constructor_via_ast(self) -> None:
        """VarRef with a type qualifier records a constructor reference."""
        r = parse_and_resolve("enum Color\n  | red\n  | blue\nColor::red")
        ref = r.program.body.items[1]
        assert isinstance(ref, VarRef)
        constructor = r.constructor_refs[ref.node_id]
        assert constructor.owner_name == "red"
        assert constructor.owner_name == "red"

    def test_ordinary_field_access_not_qualified_ref(self) -> None:
        """FieldAccess on a regular value creates no constructor reference."""
        let_x = _make_let("x", _make_intlit(1))
        fa = FieldAccess(
            obj=_make_varref("x"),
            field="something",
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(let_x, fa)
        assert fa.node_id not in r.constructor_refs


# ---------------------------------------------------------------------------
# Cast scope tests
# ---------------------------------------------------------------------------


class TestCastScope:
    def test_cast_subexpr_resolves(self) -> None:
        """cast sub-expression resolves names normally."""
        r = parse_and_resolve("let x = 1\nlet y = x as int")
        assert r  # no exception

    def test_cast_undefined_var_is_scope_error(self) -> None:
        """undefined var inside a cast is a scope error."""
        err = reject_scope("undefinedVar as int")
        assert "undefinedVar" in err.to_diagnostic().message

    def test_copy_and_shallow_copy_resolve_as_builtins(self) -> None:
        """copy(x)/shallow-copy(x) resolve as builtins, not undefined names."""
        from agm.agl.scope.symbols import BuiltinKind

        r = parse_and_resolve("let x = [1]\nlet _ = copy(x)\nshallow-copy(x)")
        assert BuiltinKind.COPY in r.builtin_calls.values()
        assert BuiltinKind.SHALLOW_COPY in r.builtin_calls.values()

    def test_copy_as_value_is_rejected(self) -> None:
        """A bare reference to 'copy' (not a call) is rejected as a builtin name."""
        err = reject_scope("let f = copy\nf")
        assert "copy" in err.to_diagnostic().message


class TestImportDeclScope:
    """Import declarations pass through the scope resolver without errors.

    Each import here targets ``std/prelude`` — the only always-real module
    available to this file's ``resolve_entry``-backed ``parse_and_resolve``
    (its search root is the repo's real ``stdlib/`` directory; there is no
    on-disk ``foo`` module for it to find). Under the old ``resolve_module``
    wrapper, an import naming a nonexistent module never actually resolved to
    a file — the per-module pass has no loader, so a bogus module id like
    ``foo`` passed through unnoticed. Through a real module graph an
    unresolvable import fails at load time with ``ModuleNotFound``, before
    the scope pass ever runs, which is exactly the defect this file's
    conversion to a real graph is meant to expose: the old versions of these
    tests were not actually exercising import-clause scope admissibility
    against a module that resolves, and would have passed identically for a
    typo'd module name.
    """

    def test_import_decl_does_not_raise(self) -> None:
        """A bare import declaration resolves without a scope error."""
        r = parse_and_resolve("import std/prelude::*\n1")
        assert r  # no exception

    def test_import_with_alias_does_not_raise(self) -> None:
        r = parse_and_resolve("import std/prelude as core\n1")
        assert r

    def test_import_wildcard_does_not_raise(self) -> None:
        r = parse_and_resolve("import std/*\n1")
        assert r

    def test_import_selected_tail_does_not_raise(self) -> None:
        r = parse_and_resolve("import std/prelude::{print}\n1")
        assert r

    def test_import_hiding_does_not_raise(self) -> None:
        r = parse_and_resolve("import std/prelude hiding print\n1")
        assert r


class TestScopedNominalAliases:
    """A scoped nominal alias follows the same constructibility rule as a root one."""

    def test_scoped_alias_of_an_enum_publishes_no_constructor(self) -> None:
        source = (
            "scope A\n"
            "  enum Color\n"
            "    | Red\n"
            "  type Alias = Color\n"
            "end A\n"
            "\n"
            "let c: A::Alias = A::Color::Red\n"
            "c\n"
        )

        resolution = parse_and_resolve(source)

        assert not any(
            candidate.owner_name == "Alias"
            for candidates in resolution.constructor_candidates.values()
            for candidate in candidates
        )

    def test_scoped_alias_of_a_record_stays_constructible(self) -> None:
        source = (
            "scope A\n"
            "  record Point\n"
            "    x: int\n"
            "  type Alias = Point\n"
            "end A\n"
            "\n"
            "let p: A::Alias = A::Alias(x = 1)\n"
            "p.x\n"
        )

        resolved = parse_and_resolve(source)

        # The construction resolves to the alias declared inside scope ``A``.
        [constructed] = resolved.constructor_refs.values()
        assert constructed.owner_name == "Alias"
        assert constructed.owner_path == ("A",)
