"""Tests for parsing typed AgL constant expressions from host strings."""

from __future__ import annotations

import pytest

from agm.agl.constant import ConstantExpressionError, parse_constant
from agm.agl.semantics.types import (
    BUILTIN_PRELUDE_TYPES,
    OPTION_TEXT_TYPE,
    IntType,
    TextType,
    Type,
)
from agm.agl.semantics.values import EnumValue, IntValue, TextValue, Value


@pytest.mark.parametrize(
    ("source", "expected_type", "expected_value"),
    (
        ('"hello"', TextType(), TextValue("hello")),
        ("42", IntType(), IntValue(42)),
    ),
)
def test_parse_constant_returns_typed_literal_value(
    source: str, expected_type: Type, expected_value: Value
) -> None:
    assert parse_constant(source, expected_type) == expected_value


@pytest.mark.parametrize(
    "source",
    (
        '\n"hello"',
        '# a host-supplied comment\n"hello"',
        '\n# a host-supplied comment\n"hello"',
    ),
)
def test_parse_constant_accepts_standalone_expression_layout(source: str) -> None:
    assert parse_constant(source, TextType()) == TextValue("hello")


def test_parse_constant_requires_one_expression() -> None:
    source = 'let value = "not an expression"'

    with pytest.raises(ConstantExpressionError) as exc_info:
        parse_constant(source, TextType())

    assert source in str(exc_info.value)
    assert "exactly one expression" in str(exc_info.value)


def test_parse_constant_constructs_enum_with_positional_fields() -> None:
    value = parse_constant('AgentClaude("sonnet", "medium")', BUILTIN_PRELUDE_TYPES["Agent"])

    assert isinstance(value, EnumValue)
    assert value.variant == "AgentClaude"
    assert value.fields == {"model": TextValue("sonnet"), "thinking": TextValue("medium")}


def test_parse_constant_constructs_enum_with_named_fields() -> None:
    value = parse_constant(
        'AgentClaude(thinking = "medium", model = "sonnet")', BUILTIN_PRELUDE_TYPES["Agent"]
    )

    assert isinstance(value, EnumValue)
    assert value.variant == "AgentClaude"
    assert value.fields == {"model": TextValue("sonnet"), "thinking": TextValue("medium")}


def test_parse_constant_constructs_existing_stdlib_enum() -> None:
    value = parse_constant("Retry(n = 3)", BUILTIN_PRELUDE_TYPES["ParsePolicy"])

    assert isinstance(value, EnumValue)
    assert value.variant == "Retry"
    assert value.fields == {"n": IntValue(3)}


def test_parse_constant_constructs_stdlib_core_enum() -> None:
    value = parse_constant('Some("x")', OPTION_TEXT_TYPE)

    assert isinstance(value, EnumValue)
    assert value.variant == "Some"
    assert value.fields == {"value": TextValue("x")}


@pytest.mark.parametrize(
    ("source", "expected_type", "error_kind"),
    (
        ("AgentClaude(", BUILTIN_PRELUDE_TYPES["Agent"], "parse error"),
        ("true", TextType(), "type error"),
        ('(fn(value: text) -> text => value)("value")', TextType(), "non-constant expression"),
        ("unbound", TextType(), "non-constant expression"),
    ),
)
def test_parse_constant_rejections_identify_the_input(
    source: str, expected_type: Type, error_kind: str
) -> None:
    with pytest.raises(ConstantExpressionError) as exc_info:
        parse_constant(source, expected_type)

    assert source in str(exc_info.value)
    assert error_kind in str(exc_info.value)
