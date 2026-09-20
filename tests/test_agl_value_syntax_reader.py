"""Tests for the AgL value-syntax reader: `agm.agl.value_syntax.reader.read_value`."""

from __future__ import annotations

from decimal import Decimal

import pytest

from agm.agl.value_syntax.errors import ValueSyntaxError
from agm.agl.value_syntax.nodes import (
    ArrayNode,
    BoolNode,
    CtorNode,
    DecimalNode,
    DictEntry,
    DictNode,
    IntNode,
    NullNode,
    TextNode,
    ValueArg,
)
from agm.agl.value_syntax.reader import read_ctor_head, read_value

# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_reads_int() -> None:
    node = read_value("42")
    assert node == IntNode(42, 0, 2)


def test_reads_negative_int() -> None:
    node = read_value("-7")
    assert node == IntNode(-7, 0, 2)


def test_reads_negative_decimal() -> None:
    node = read_value("-1.5")
    assert node == DecimalNode(Decimal("-1.5"), 0, 4)


def test_decimal_value_is_decimal_instance() -> None:
    node = read_value("3.25")
    assert isinstance(node, DecimalNode)
    assert node.value == Decimal("3.25")


@pytest.mark.parametrize(("source", "expected"), (("true", True), ("false", False)))
def test_reads_bool(source: str, expected: bool) -> None:
    node = read_value(source)
    assert node == BoolNode(expected, 0, len(source))


def test_reads_null() -> None:
    node = read_value("null")
    assert node == NullNode(0, 4)


@pytest.mark.parametrize("quote", ('"', "'"))
def test_reads_text_in_either_quote_style(quote: str) -> None:
    source = f"{quote}hello{quote}"
    node = read_value(source)
    assert node == TextNode("hello", 0, len(source))


def test_reads_text_escapes() -> None:
    node = read_value(r'"café \n \${x}"')
    assert isinstance(node, TextNode)
    assert node.value == "café \n ${x}"


def test_reads_surrogate_pair_escape_combines_into_one_character() -> None:
    node = read_value('"\\uD83D\\uDE00"')
    assert isinstance(node, TextNode)
    assert node.value == "\U0001f600"


def test_array_with_trailing_comma_and_nesting() -> None:
    node = read_value("[1, [2, 3], 4,]")
    assert isinstance(node, ArrayNode)
    assert node.items[0] == IntNode(1, 1, 2)
    assert isinstance(node.items[1], ArrayNode)
    assert node.items[1].items == (IntNode(2, 5, 6), IntNode(3, 8, 9))
    assert node.items[2] == IntNode(4, 12, 13)


def test_empty_array() -> None:
    assert read_value("[]") == ArrayNode((), 0, 2)


def test_dict_with_name_key_text_key_and_trailing_comma() -> None:
    node = read_value('{a: 1, "b": 2,}')
    assert isinstance(node, DictNode)
    assert [entry.key for entry in node.entries] == ["a", "b"]
    assert [entry.value for entry in node.entries] == [IntNode(1, 4, 5), IntNode(2, 12, 13)]


def test_empty_dict() -> None:
    assert read_value("{}") == DictNode((), 0, 2)


def test_bare_ctor_has_no_args() -> None:
    node = read_value("Circle")
    assert node == CtorNode(None, "Circle", None, 0, 6)


def test_empty_call_has_empty_args_tuple() -> None:
    node = read_value("Circle()")
    assert isinstance(node, CtorNode)
    assert node.args == ()


def test_positional_named_and_mixed_args() -> None:
    node = read_value("Circle(1, radius = 2)")
    assert isinstance(node, CtorNode)
    assert node.args is not None
    assert len(node.args) == 2
    assert node.args[0] == ValueArg(None, IntNode(1, 7, 8), 7, 8)
    assert node.args[1].name == "radius"
    assert node.args[1].value == IntNode(2, 19, 20)


def test_qualified_ctor() -> None:
    node = read_value("Shape::Rect(1, height = 2)")
    assert isinstance(node, CtorNode)
    assert node.qualifier == "Shape"
    assert node.name == "Rect"
    assert node.args is not None
    assert len(node.args) == 2


@pytest.mark.parametrize("name", ("ask-prompt", "do-it-now?"))
def test_ctor_names_with_hyphen_and_question_mark(name: str) -> None:
    node = read_value(name)
    assert node == CtorNode(None, name, None, 0, len(name))


def test_surrounding_whitespace_and_newlines_are_allowed() -> None:
    node = read_value("\n  \t 42 \n\t ")
    assert node == IntNode(42, 5, 7)


def test_nested_ctor_arguments() -> None:
    node = read_value("Line(from = Point(0, 0), to = Point(1, 1))")
    assert isinstance(node, CtorNode)
    assert node.args is not None
    assert isinstance(node.args[0].value, CtorNode)
    assert node.args[0].value.name == "Point"


def test_dict_without_trailing_comma() -> None:
    node = read_value("{a: 1}")
    assert node == DictNode((DictEntry("a", IntNode(1, 4, 5), 1, 5),), 0, 6)


def test_dollar_brace_not_followed_by_a_name_is_literal_text() -> None:
    """`${` only opens a hole when followed by an identifier; otherwise it is literal."""
    node = read_value('"${}"')
    assert node == TextNode("${}", 0, 5)


def test_dollar_brace_hole_without_closing_brace_is_literal_text() -> None:
    """A `${NAME` run that never reaches `}` is not a hole; it stays literal."""
    node = read_value('"${HOME }"')
    assert node == TextNode("${HOME }", 0, 10)


def test_args_with_trailing_comma() -> None:
    node = read_value("Circle(1,)")
    assert isinstance(node, CtorNode)
    assert node.args == (ValueArg(None, IntNode(1, 7, 8), 7, 8),)


def test_positional_ctor_argument_without_equals() -> None:
    node = read_value("Wrap(Point)")
    assert isinstance(node, CtorNode)
    assert node.args is not None
    assert node.args[0].name is None
    assert node.args[0].value == CtorNode(None, "Point", None, 5, 10)


def test_whitespace_after_minus_is_insignificant() -> None:
    assert read_value("- 5") == IntNode(-5, 0, 3)


@pytest.mark.parametrize("source", ("Shape :: Rect", "Shape ::Rect", "Shape:: Rect"))
def test_whitespace_around_double_colon_is_insignificant(source: str) -> None:
    node = read_value(source)
    assert isinstance(node, CtorNode)
    assert node.qualifier == "Shape"
    assert node.name == "Rect"


def test_carriage_return_is_whitespace_between_tokens() -> None:
    node = read_value("[1,\r2]")
    assert isinstance(node, ArrayNode)
    assert node.items == (IntNode(1, 1, 2), IntNode(2, 4, 5))


@pytest.mark.parametrize("ch", ("\x0b", "\xa0", " "))
def test_exotic_space_like_characters_outside_text_are_errors(ch: str) -> None:
    with pytest.raises(ValueSyntaxError):
        read_value(f"{ch}5")


def test_raw_carriage_return_inside_text_is_an_error() -> None:
    with pytest.raises(ValueSyntaxError):
        read_value('"a\rb"')


def test_unterminated_dict_after_trailing_comma_has_in_range_offsets() -> None:
    source = "{a: 1,"
    with pytest.raises(ValueSyntaxError) as exc_info:
        read_value(source)
    error = exc_info.value
    assert 0 <= error.start <= error.end <= len(source)


def test_deeply_nested_array_raises_value_syntax_error_not_recursion_error() -> None:
    source = "[" * 5000
    with pytest.raises(ValueSyntaxError) as exc_info:
        read_value(source)
    error = exc_info.value
    assert 0 <= error.start <= error.end <= len(source)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    (
        '"unterminated',
        '"line one\nline two"',
        '"%{x}"',
        '"${HOME}"',
        r'"\q"',
        "if",
        "exec$",
        "-",
        "-x",
        "[1 2]",
        "{a: 1 b: 2}",
        "1 2",
        "Shape::",
        "",
        "[1, 2",
        "{a: 1",
        "Circle(1",
        '"\\',
        '"\\u12',
        r'"\uXXXX"',
        '"\\uD800x"',
        '"\\uDC00x"',
        '"\\uDC00\\uD800"',
        '"\\uD800%{1}"',
        '"\\uD800"',
        "@",
        "{1: 2}",
        "{a 1}",
        "Shape::if",
        "Circle(1 2)",
        "Circle(if = 1)",
        '"${',
        "{if: 1}",
        "{exec$: 1}",
        "{true: 1}",
    ),
    ids=(
        "unterminated-text",
        "newline-in-text",
        "expr-interpolation-in-text",
        "env-interpolation-in-text",
        "unknown-escape",
        "keyword-as-ctor",
        "raw-tail-name-as-ctor",
        "minus-without-digit",
        "minus-followed-by-name",
        "missing-comma-in-array",
        "missing-comma-in-dict",
        "trailing-garbage",
        "dcolon-without-name",
        "empty-input",
        "unclosed-array",
        "unclosed-dict",
        "unclosed-call",
        "backslash-at-eof-in-text",
        "incomplete-unicode-in-text",
        "invalid-hex-digit-in-text",
        "lone-high-surrogate-escape",
        "lone-low-surrogate-escape",
        "reversed-surrogate-pair-escape",
        "high-surrogate-before-hole",
        "high-surrogate-at-end-of-literal",
        "unexpected-character",
        "dict-key-not-a-name",
        "dict-missing-colon",
        "qualified-ctor-invalid-member",
        "args-missing-comma",
        "invalid-argument-name",
        "dollar-brace-at-eof-in-text",
        "dict-key-is-keyword",
        "dict-key-is-raw-tail-name",
        "dict-key-is-true",
    ),
)
def test_read_value_rejects_malformed_input(source: str) -> None:
    with pytest.raises(ValueSyntaxError):
        read_value(source)


def test_unknown_escape_error_offsets_span_the_escape() -> None:
    with pytest.raises(ValueSyntaxError) as exc_info:
        read_value(r'"\q"')
    error = exc_info.value
    assert (error.start, error.end) == (1, 3)


def test_lone_surrogate_escape_error_offsets_span_only_that_escape() -> None:
    source = '"\\uD800x"'
    with pytest.raises(ValueSyntaxError) as exc_info:
        read_value(source)
    error = exc_info.value
    assert source[error.start : error.end] == "\\uD800"


def test_trailing_garbage_error_offsets_point_past_the_value() -> None:
    with pytest.raises(ValueSyntaxError) as exc_info:
        read_value("1 2")
    error = exc_info.value
    assert (error.start, error.end) == (2, 3)


# ---------------------------------------------------------------------------
# read_ctor_head
# ---------------------------------------------------------------------------


def test_read_ctor_head_unqualified() -> None:
    assert read_ctor_head("Circle(1)") == (None, "Circle")


def test_read_ctor_head_qualified() -> None:
    assert read_ctor_head("Shape::Square(side = 1)") == ("Shape", "Square")


def test_read_ctor_head_allows_whitespace_around_double_colon_and_before_paren() -> None:
    assert read_ctor_head("Shape :: Square (side = 1)") == ("Shape", "Square")


def test_read_ctor_head_allows_leading_whitespace() -> None:
    assert read_ctor_head("  \n Circle(1)") == (None, "Circle")


def test_read_ctor_head_none_for_bare_name_without_call() -> None:
    assert read_ctor_head("Circle") is None


def test_read_ctor_head_none_for_text_not_opening_with_a_name() -> None:
    assert read_ctor_head("(1)") is None


def test_read_ctor_head_none_for_malformed_qualifier_chain() -> None:
    assert read_ctor_head("Shape::(x)") is None
