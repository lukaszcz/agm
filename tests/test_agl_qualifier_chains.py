"""Behavioral coverage for expression qualifier-chain syntax."""

from __future__ import annotations

import pytest

from agm.agl.parser import AglSyntaxError, parse_program
from agm.agl.scope import AglScopeError, resolve_module
from agm.agl.syntax import AssignStmt, Block, NameTarget, QualifierAnchor, VarRef


def _ref(source: str) -> VarRef:
    program = parse_program(source)
    assert isinstance(program.body, Block)
    (expr,) = program.body.items
    assert isinstance(expr, VarRef)
    return expr


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


def test_current_module_qualified_assignment_remains_an_assignment_target() -> None:
    program = parse_program("::value := 1")

    assert isinstance(program.body, Block)
    (assignment,) = program.body.items
    assert isinstance(assignment, AssignStmt)
    assert isinstance(assignment.target, NameTarget)
    assert assignment.target.module_qualifier is not None
    assert assignment.target.module_qualifier.segments == ()


def test_non_expression_type_qualifiers_retain_their_existing_validation() -> None:
    with pytest.raises(AglSyntaxError, match="single type name"):
        parse_program("case value of | module::owner/name::member => 1")


def test_long_qualified_expression_retains_all_segments() -> None:
    ref = _ref("First::Second::Third::member")

    assert ref.qualifier is not None
    assert [segment.name for segment in ref.qualifier.segments] == ["First", "Second", "Third"]


def test_current_module_anchored_type_constructor_remains_supported() -> None:
    program = parse_program("enum Option\n  | some\n::Option::some")
    assert isinstance(program.body, Block)
    expr = program.body.items[-1]
    assert isinstance(expr, VarRef)

    resolution = resolve_module(program)
    assert resolution.qualified_constructor_refs[expr.node_id] == ("Option", "some", None)


def test_current_module_anchored_multi_segment_chain_has_a_clear_scope_error() -> None:
    with pytest.raises(AglScopeError) as exc_info:
        resolve_module(parse_program("::A::B::C"))

    diagnostic = exc_info.value.to_diagnostic()
    assert "unsupported qualifier chain" in diagnostic.message.lower()
    assert diagnostic.line == 1


@pytest.mark.parametrize("source", ("First::Second::Third::member", "Type[int]::Second::member"))
def test_unsupported_expression_qualifier_chain_has_a_clear_scope_error(source: str) -> None:
    with pytest.raises(AglScopeError, match="Unsupported qualifier chain"):
        resolve_module(parse_program(source))
