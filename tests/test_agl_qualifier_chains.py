"""Behavioral coverage for expression qualifier-chain syntax."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest

from agm.agl.modules.ids import ENTRY_ID
from agm.agl.parser import parse_program
from agm.agl.scope import AglScopeError, ModuleResolution, resolve_module
from agm.agl.scope.program import resolve_program
from agm.agl.syntax import (
    AssignStmt,
    Block,
    Call,
    Case,
    FuncDef,
    IsTest,
    LetDecl,
    NameT,
    NameTarget,
    QualifierAnchor,
    ScopeRegion,
    VarRef,
)
from agm.agl.syntax.nodes import ConstructorPattern
from agm.agl.syntax.visitor import walk
from tests.agl.ir_harness import make_graph_from_files


def _ref(source: str) -> VarRef:
    program = parse_program(source)
    assert isinstance(program.body, Block)
    (expr,) = program.body.items
    assert isinstance(expr, VarRef)
    return expr


_NodeT = TypeVar("_NodeT")


def _find_nodes(program: object, node_type: type[_NodeT]) -> list[_NodeT]:
    """Collect every node of *node_type* in *program*, in tree order."""
    found: list[_NodeT] = []
    walk(program, lambda node: found.append(node) if isinstance(node, node_type) else None)
    return found


def _entry_resolution(tmp_path: Path, modules: dict[str, str]) -> ModuleResolution:
    """Resolve a multi-module program and return the entry module's resolution."""
    resolved = resolve_program(make_graph_from_files(tmp_path, modules))
    return resolved.modules[resolved.entry_id].resolved


def test_qualified_expression_keeps_segment_spans_and_type_arguments() -> None:
    ref = _ref("module::Type[int]::member")

    assert ref.qualifier is not None
    assert ref.qualifier.anchor is None
    assert ref.qualifier.member == "member"
    assert [segment.name for segment in ref.qualifier.segments] == ["module", "Type"]
    assert ref.qualifier.segments[0].type_args is None
    assert ref.qualifier.segments[1].type_args is not None
    assert ref.qualifier.segments[0].span.start_offset == 0
    assert ref.qualifier.segments[0].span.end_offset == 6
    assert ref.qualifier.segments[1].span.start_offset == 8
    assert ref.qualifier.segments[1].span.end_offset == 17


def test_qualified_expression_retains_module_and_current_module_anchors() -> None:
    module_ref = _ref("/library::value")
    current_module_ref = _ref("::Type::member")

    assert module_ref.qualifier is not None
    assert module_ref.qualifier.anchor is QualifierAnchor.MODULE
    assert module_ref.qualifier.segments[0].span.start_offset == 1
    assert current_module_ref.qualifier is not None
    assert current_module_ref.qualifier.anchor is QualifierAnchor.CURRENT_MODULE
    assert current_module_ref.qualifier.segments[0].span.start_offset == 2


def test_current_module_anchored_chain_keeps_all_qualifier_segments() -> None:
    ref = _ref("::A::B::C")

    assert ref.qualifier is not None
    assert ref.qualifier.anchor is QualifierAnchor.CURRENT_MODULE
    assert [segment.name for segment in ref.qualifier.segments] == ["A", "B"]
    assert ref.qualifier.member == "C"


def test_qualified_patterns_keep_segment_spans_and_type_arguments() -> None:
    source = "case value of | module::Option[int]::None => 1"
    program = parse_program(source)

    assert isinstance(program.body, Block)
    (case,) = program.body.items
    assert isinstance(case, Case)
    qualifier = case.branches[0].pattern.qualifier
    assert qualifier is not None
    assert qualifier.member == "None"
    assert [segment.name for segment in qualifier.segments] == ["module", "Option"]
    assert qualifier.segments[1].type_args is not None
    assert (
        source[qualifier.segments[0].span.start_offset : qualifier.segments[0].span.end_offset]
        == "module"
    )
    assert (
        source[qualifier.segments[1].span.start_offset : qualifier.segments[1].span.end_offset]
        == "Option[int]"
    )


def test_qualified_type_references_keep_segment_spans_and_type_arguments() -> None:
    source = "let value: module::Option[int]::Result = null"
    program = parse_program(source)

    assert isinstance(program.body, Block)
    (decl,) = program.body.items
    assert isinstance(decl, LetDecl)
    assert isinstance(decl.type_ann, NameT)
    qualifier = decl.type_ann.qualifier
    assert qualifier is not None
    assert qualifier.member == "Result"
    assert [segment.name for segment in qualifier.segments] == ["module", "Option"]
    assert qualifier.segments[1].type_args is not None
    assert (
        source[qualifier.segments[0].span.start_offset : qualifier.segments[0].span.end_offset]
        == "module"
    )
    assert (
        source[qualifier.segments[1].span.start_offset : qualifier.segments[1].span.end_offset]
        == "Option[int]"
    )


def test_qualified_is_tests_keep_segment_spans_and_type_arguments() -> None:
    source = "value is module::Option[int]::None"
    program = parse_program(source)

    assert isinstance(program.body, Block)
    (test,) = program.body.items
    assert isinstance(test, IsTest)
    qualifier = test.qualifier
    assert qualifier is not None
    assert qualifier.member == "None"
    assert [segment.name for segment in qualifier.segments] == ["module", "Option"]
    assert qualifier.segments[1].type_args is not None
    assert (
        source[qualifier.segments[0].span.start_offset : qualifier.segments[0].span.end_offset]
        == "module"
    )
    assert (
        source[qualifier.segments[1].span.start_offset : qualifier.segments[1].span.end_offset]
        == "Option[int]"
    )


@pytest.mark.parametrize(
    "source",
    (
        "let value: First::Second::Third::member = null",
        "let value: Option[int]::Result = null",
        "let value = 1\ncase value of | First::Second::Third::member => 1",
        "let value = 1\nvalue is First::Second::Third::member",
    ),
)
def test_long_qualifier_chains_are_not_rejected_by_legacy_shape_validation(source: str) -> None:
    resolve_module(parse_program(source))


@pytest.mark.parametrize(
    "source",
    (
        "::/foo::Baz",
        "::foo/bar::Baz",
        "foo::/bar::Baz",
        "foo::bar/baz::Baz",
    ),
)
def test_nonleading_anchor_and_route_segments_report_route_errors(source: str) -> None:
    with pytest.raises(AglScopeError, match="Only the leading qualifier"):
        resolve_module(parse_program(source))


def test_current_module_qualified_assignment_remains_an_assignment_target() -> None:
    program = parse_program("::value := 1")

    assert isinstance(program.body, Block)
    (assignment,) = program.body.items
    assert isinstance(assignment, AssignStmt)
    assert isinstance(assignment.target, NameTarget)
    assert assignment.target.qualifier is not None
    assert assignment.target.qualifier.segments == ()


def test_non_expression_chain_routes_reject_only_invalid_nonleading_routes() -> None:
    with pytest.raises(AglScopeError, match="Only the leading qualifier"):
        resolve_module(
            parse_program("let value = 1\ncase value of | module::owner/name::member => 1")
        )


def test_long_qualified_expression_retains_all_segments() -> None:
    ref = _ref("First::Second::Third::member")

    assert ref.qualifier is not None
    assert [segment.name for segment in ref.qualifier.segments] == ["First", "Second", "Third"]


def test_current_module_anchor_uses_root_binding_without_import_environment() -> None:
    program = parse_program(
        "def root() -> int = 1\nscope Nested\ndef use(root: int) -> int = ::root()\nend Nested"
    )

    resolution = resolve_module(program)
    nested = next(item for item in program.body.items if isinstance(item, ScopeRegion))
    nested_use = next(item for item in nested.items if isinstance(item, FuncDef))
    assert nested_use.name == "use"
    assert isinstance(nested_use.body, Call)
    assert isinstance(nested_use.body.callee, VarRef)
    assert resolution.resolution[nested_use.body.callee.node_id].kind.name == "function_binding"


def test_generic_constructor_ignores_unrelated_same_named_scope() -> None:
    program = parse_program(
        "scope Other\n"
        "scope A\n"
        "def ignored() -> int = 0\n"
        "end A\n"
        "end Other\n"
        "enum A[T]\n"
        "  | make(value: T)\n"
        "A[int]::make(value = 1)"
    )

    resolve_module(program)


def test_current_module_anchored_type_constructor_uses_the_chain_constructor_ref() -> None:
    program = parse_program("enum Option\n  | some\n::Option::some")
    assert isinstance(program.body, Block)
    expr = program.body.items[-1]
    assert isinstance(expr, VarRef)

    resolution = resolve_module(program)
    assert resolution.constructor_refs[expr.node_id].owner_name == "Option"
    assert resolution.constructor_refs[expr.node_id].variant == "some"


@pytest.mark.parametrize("source", ("::Unknown::On", "::Unknown[int]::On"))
def test_current_module_unknown_constructor_owner_reports_its_segment(source: str) -> None:
    program = parse_program(source)
    assert isinstance(program.body, Block)
    expr = program.body.items[-1]
    assert isinstance(expr, VarRef)
    assert expr.qualifier is not None

    with pytest.raises(AglScopeError) as exc_info:
        resolve_module(program)

    assert exc_info.value.to_diagnostic().message == "'Unknown' is not defined in this module."
    assert exc_info.value.span == expr.qualifier.segments[0].span


def test_scoped_enum_members_and_nested_type_members_run_through_the_full_pipeline() -> None:
    from tests.agl.ir_harness import evaluate_ir_output

    output = evaluate_ir_output(
        "enum Option[T] | none | some(value: T)\n"
        "scope Option\n"
        "def is_empty(value: Option[int]) -> bool = value is Option[int]::none\n"
        "end Option\n"
        "scope A\n"
        "enum T[U] | value\n"
        "end A\n"
        "let option = Option[int]::some(value = 1)\n"
        "case option of\n"
        "  | Option[int]::some(value) => print value\n"
        "  | Option[int]::none => print 0\n"
        "print(Option::is_empty(Option[int]::none))\n"
        "let nested: A::T[int] = A::T[int]::value\n"
        "print(nested is A::T[int]::value)"
    )

    assert output == "1\ntrue\ntrue\n"


def test_current_module_anchored_multi_segment_chain_reports_its_unknown_path() -> None:
    with pytest.raises(AglScopeError, match="scope path"):
        resolve_module(parse_program("::A::B::C"))


def test_module_anchored_constructor_chain_never_falls_back_to_a_local_type() -> None:
    with pytest.raises(AglScopeError, match="No module"):
        resolve_module(parse_program("enum A | value\n/A::value"))


def test_nonconstructible_scoped_type_falls_back_to_the_legacy_constructor_diagnostic() -> None:
    with pytest.raises(AglScopeError):
        resolve_module(parse_program("type A::Count = int\nA::Count"))


@pytest.mark.parametrize(
    "source",
    (
        "scope Tools\ndef f() -> int = 0\nend Tools\nTools[int]::f()",
        "scope Tools\ndef f() -> int = 0\nend Tools\nlet value: Tools[int]::T = null",
        (
            "scope Tools\ndef f() -> int = 0\nend Tools\nlet value = 1\n"
            "case value of | Tools[int]::f => 1"
        ),
        "scope Tools\ndef f() -> int = 0\nend Tools\nvalue is Tools[int]::f",
    ),
)
def test_type_arguments_on_a_plain_scope_are_rejected_in_every_chain_position(
    source: str,
) -> None:
    with pytest.raises(AglScopeError, match="Type arguments cannot be applied to scope"):
        resolve_module(parse_program(source))


@pytest.mark.parametrize("source", ("First::Second::Third::member", "Type[int]::Second::member"))
def test_long_expression_qualifier_chain_reports_the_unresolved_name(source: str) -> None:
    with pytest.raises(AglScopeError, match="not defined|No module"):
        resolve_module(parse_program(source))


def test_imported_scoped_enum_owner_retains_its_scope_path_for_is_and_case(
    tmp_path: Path,
) -> None:
    """A pattern-qualified ``lib::A::Status::Good`` keeps ``lib``'s scoped owner."""
    resolution = _entry_resolution(
        tmp_path,
        {
            "entry": (
                "import lib\n"
                "let s = lib::A::Status::Good\n"
                "print(s is lib::A::Status::Good)\n"
                "print(case s of\n"
                '  | lib::A::Status::Good => "good"\n'
                '  | lib::A::Status::Bad => "bad")\n'
            ),
            "lib": "scope A\nenum Status\n  | Good\n  | Bad\nend A\n",
        },
    )

    (is_test,) = _find_nodes(resolution.program, IsTest)
    is_cref = resolution.constructor_refs[is_test.node_id]
    assert (is_cref.owner_path, is_cref.owner_name, is_cref.variant) == (("A",), "Status", "Good")

    (case_node,) = _find_nodes(resolution.program, Case)
    good_pattern = case_node.branches[0].pattern
    assert isinstance(good_pattern, ConstructorPattern)
    case_cref = resolution.constructor_refs[good_pattern.node_id]
    assert (case_cref.owner_path, case_cref.owner_name, case_cref.variant) == (
        ("A",),
        "Status",
        "Good",
    )


def test_explicit_module_route_open_is_not_masked_by_a_same_named_local_scope(
    tmp_path: Path,
) -> None:
    """``open geo/shapes::Point`` reaches the routed scope over a same-named local one."""
    resolution = _entry_resolution(
        tmp_path,
        {
            "entry": (
                "import geo/shapes\n"
                "open geo/shapes::Point\n"
                "scope Point\n"
                "def area() -> int = 1\n"
                "end Point\n"
                "print(describe())\n"
                "print(Point::area())\n"
            ),
            "geo/shapes": 'scope Point\ndef describe() -> text = "point"\nend Point\n',
        },
    )

    var_refs = _find_nodes(resolution.program, VarRef)
    (describe_ref,) = [node for node in var_refs if node.name == "describe"]
    assert resolution.resolution[describe_ref.node_id].module_id != ENTRY_ID

    (area_ref,) = [node for node in var_refs if node.name == "area"]
    assert resolution.resolution[area_ref.node_id].module_id == ENTRY_ID


def test_unrelated_nested_scope_does_not_mask_a_root_import_route(tmp_path: Path) -> None:
    """A nested ``X::lib`` scope must not shadow an unrelated root import named ``lib``."""
    resolution = _entry_resolution(
        tmp_path,
        {
            "entry": (
                "import lib\n"
                "scope X\n"
                "scope lib\n"
                "def g() -> int = 1\n"
                "end lib\n"
                "end X\n"
                "print(lib::A::f())\n"
            ),
            "lib": "scope A\ndef f() -> int = 7\nend A\n",
        },
    )

    (call_ref,) = [node for node in _find_nodes(resolution.program, VarRef) if node.name == "f"]
    assert resolution.resolution[call_ref.node_id].module_id != ENTRY_ID


@pytest.mark.parametrize(
    "entry_source",
    (
        "import lib\nprint(lib[int]::A::f())",
        "import lib\nprint(lib::A[int]::f())",
    ),
)
def test_type_arguments_are_rejected_on_imported_route_and_scope_segments(
    tmp_path: Path, entry_source: str
) -> None:
    with pytest.raises(AglScopeError, match="Type arguments cannot be applied"):
        resolve_program(
            make_graph_from_files(
                tmp_path, {"entry": entry_source, "lib": "scope A\ndef f() -> int = 7\nend A\n"}
            )
        )


def test_type_arguments_on_an_imported_generic_type_scope_still_resolve(tmp_path: Path) -> None:
    """``lib::Box[int]::describe()`` keeps working: ``Box`` is a real generic type."""
    resolution = _entry_resolution(
        tmp_path,
        {
            "entry": "import lib\nprint(lib::Box[int]::describe())",
            "lib": 'record Box[T]\n  value: T\n\ndef Box::describe() -> text = "box"\n',
        },
    )
    (describe_call,) = [
        call
        for call in _find_nodes(resolution.program, Call)
        if isinstance(call.callee, VarRef) and call.callee.name == "describe"
    ]
    assert isinstance(describe_call.callee, VarRef)
    assert resolution.resolution[describe_call.callee.node_id].module_id != ENTRY_ID


def test_open_imported_enum_owner_qualifies_its_variant(tmp_path: Path) -> None:
    """``Color::Red`` reaches an open-imported enum owner in expression position."""
    resolution = _entry_resolution(
        tmp_path,
        {
            "entry": "open import lib\nlet c = Color::Red\nprint(c is Color::Red)\n",
            "lib": "enum Color\n  | Red\n  | Blue\n",
        },
    )

    (variant_ref,) = (
        node
        for node in _find_nodes(resolution.program, VarRef)
        if node.qualifier is not None and node.name == "Red"
    )
    cref = resolution.constructor_refs[variant_ref.node_id]
    assert (cref.owner_name, cref.variant) == ("Color", "Red")
