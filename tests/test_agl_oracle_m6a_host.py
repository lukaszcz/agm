"""M6a differential oracle — print, parse_json, and entry params.

Each test runs both the legacy AST interpreter and the new IR pipeline
and asserts they produce identical results (bindings + stdout).

Marker: ``@pytest.mark.oracle``.
"""

from __future__ import annotations

import decimal
import textwrap

import pytest

from agm.agl.eval.values import (
    DecimalValue,
    IntValue,
    JsonValue,
    TextValue,
)
from tests.agl.oracle.harness import (
    assert_oracle_agrees,
    assert_oracle_raises,
)

# ===========================================================================
# print — various value types
# ===========================================================================


@pytest.mark.oracle
def test_print_int() -> None:
    """print(int) — oracle agrees on stdout and bindings."""
    source = "let x = 1\nprint(x)\n()"
    legacy, ir = assert_oracle_agrees(source)
    assert legacy["x"] == IntValue(1)
    assert ir["x"] == IntValue(1)


@pytest.mark.oracle
def test_print_decimal() -> None:
    """print(decimal) — rendered as decimal string."""
    source = "let d: decimal = 3.14\nprint(d)\n()"
    assert_oracle_agrees(source)


@pytest.mark.oracle
def test_print_text() -> None:
    """print(text) — rendered as raw text."""
    source = 'let x = "hello world"\nprint(x)\n()'
    assert_oracle_agrees(source)


@pytest.mark.oracle
def test_print_bool_true() -> None:
    """print(bool) — true rendered as 'true'."""
    source = "let x = true\nprint(x)\n()"
    assert_oracle_agrees(source)


@pytest.mark.oracle
def test_print_bool_false() -> None:
    """print(bool) — false rendered as 'false'."""
    source = "let x = false\nprint(x)\n()"
    assert_oracle_agrees(source)


@pytest.mark.oracle
def test_print_list() -> None:
    """print(list) — oracle agrees."""
    source = "let x = [1, 2, 3]\nprint(x)\n()"
    assert_oracle_agrees(source)


@pytest.mark.oracle
def test_print_dict() -> None:
    """print(dict) — oracle agrees."""
    source = 'let x = {"a": 1, "b": 2}\nprint(x)\n()'
    assert_oracle_agrees(source)


@pytest.mark.oracle
def test_print_record() -> None:
    """print(record) — oracle agrees."""
    source = textwrap.dedent("""\
        record Point
          x: int
          y: int
        let p = Point(x: 10, y: 20)
        print(p)
        ()
    """)
    assert_oracle_agrees(source)


@pytest.mark.oracle
def test_print_enum_variant() -> None:
    """print(enum variant) — oracle agrees."""
    source = textwrap.dedent("""\
        enum Color
          | Red
          | Green
          | Blue
        let c = Green
        print(c)
        ()
    """)
    assert_oracle_agrees(source)


@pytest.mark.oracle
def test_print_multiple_calls() -> None:
    """Multiple print calls — all output matches."""
    source = textwrap.dedent("""\
        let x = 10
        print("line1")
        print(x)
        print(true)
        ()
    """)
    assert_oracle_agrees(source)


# ===========================================================================
# print inside control flow / function body
# ===========================================================================


@pytest.mark.oracle
def test_print_inside_if() -> None:
    """print inside an if branch — oracle agrees."""
    source = textwrap.dedent("""\
        let cond = true
        if cond =>
          print("yes")
        else =>
          ()
        ()
    """)
    assert_oracle_agrees(source)


@pytest.mark.oracle
def test_print_inside_function() -> None:
    """print inside a function body — oracle agrees."""
    source = textwrap.dedent("""\
        def greet(name: text) -> unit =
          print(name)
        greet("Alice")
        ()
    """)
    assert_oracle_agrees(source)


@pytest.mark.oracle
def test_print_inside_loop() -> None:
    """print inside a do…until loop — oracle agrees."""
    source = textwrap.dedent("""\
        var i = 0
        do[10]
          print(i)
          i := i + 1
        until i >= 3
        ()
    """)
    assert_oracle_agrees(source)


# ===========================================================================
# parse_json — success and failure
# ===========================================================================


@pytest.mark.oracle
def test_parse_json_success_object() -> None:
    """parse_json succeeds for a JSON object."""
    source = "let j = parse_json('{\"key\": 42}')\n()"
    legacy, ir = assert_oracle_agrees(source)
    assert isinstance(legacy["j"], JsonValue)
    assert isinstance(ir["j"], JsonValue)


@pytest.mark.oracle
def test_parse_json_success_array() -> None:
    """parse_json succeeds for a JSON array."""
    source = "let j = parse_json('[1, 2, 3]')\n()"
    legacy, ir = assert_oracle_agrees(source)
    assert isinstance(legacy["j"], JsonValue)
    assert isinstance(ir["j"], JsonValue)


@pytest.mark.oracle
def test_parse_json_success_string() -> None:
    """parse_json succeeds for a JSON string."""
    source = 'let j = parse_json(\'\"hello\"\')\n()'
    legacy, ir = assert_oracle_agrees(source)
    assert isinstance(legacy["j"], JsonValue)
    assert isinstance(ir["j"], JsonValue)


@pytest.mark.oracle
def test_parse_json_success_number() -> None:
    """parse_json succeeds for a JSON number."""
    source = "let j = parse_json('123')\n()"
    legacy, ir = assert_oracle_agrees(source)
    assert isinstance(legacy["j"], JsonValue)
    assert isinstance(ir["j"], JsonValue)


@pytest.mark.oracle
def test_parse_json_success_null() -> None:
    """parse_json('null') returns JsonValue(None)."""
    source = "let j = parse_json('null')\n()"
    legacy, ir = assert_oracle_agrees(source)
    assert legacy["j"] == JsonValue(None)
    assert ir["j"] == JsonValue(None)


@pytest.mark.oracle
def test_parse_json_failure_malformed() -> None:
    """parse_json raises JsonParseError on malformed input."""
    source = "let j = parse_json('not-json')\n()"
    assert_oracle_raises(source)


@pytest.mark.oracle
def test_parse_json_failure_empty() -> None:
    """parse_json raises JsonParseError on empty input."""
    source = "let j = parse_json('')\n()"
    assert_oracle_raises(source)


@pytest.mark.oracle
def test_parse_json_failure_trailing_garbage() -> None:
    """parse_json raises JsonParseError when trailing content follows valid JSON."""
    source = "let j = parse_json('1 2 3')\n()"
    assert_oracle_raises(source)


@pytest.mark.oracle
def test_parse_json_caught_by_try() -> None:
    """parse_json error caught in try — oracle agrees on caught exception handling."""
    source = textwrap.dedent("""\
        var result: text = "default"
        try
          let j = parse_json('bad')
          result := "ok"
        catch JsonParseError as e =>
          result := "caught"
        ()
    """)
    legacy, ir = assert_oracle_agrees(source)
    assert legacy["result"] == TextValue("caught")
    assert ir["result"] == TextValue("caught")


# ===========================================================================
# param declarations
# ===========================================================================


@pytest.mark.oracle
def test_param_provided_value() -> None:
    """param with provided value — oracle uses it."""
    source = textwrap.dedent("""\
        param name: text
        let greeting = "Hello, " + name
        ()
    """)
    legacy, ir = assert_oracle_agrees(source, param_values={"name": TextValue("World")})
    assert legacy["greeting"] == TextValue("Hello, World")
    assert ir["greeting"] == TextValue("Hello, World")


@pytest.mark.oracle
def test_param_provided_int_value() -> None:
    """param int provided — oracle uses it."""
    source = textwrap.dedent("""\
        param count: int
        let doubled = count * 2
        ()
    """)
    legacy, ir = assert_oracle_agrees(source, param_values={"count": IntValue(5)})
    assert legacy["doubled"] == IntValue(10)
    assert ir["doubled"] == IntValue(10)


@pytest.mark.oracle
def test_param_default_used_when_no_value() -> None:
    """param with default evaluated when no value provided."""
    source = textwrap.dedent("""\
        param n: int = 7
        let result = n + 1
        ()
    """)
    legacy, ir = assert_oracle_agrees(source)
    assert legacy["result"] == IntValue(8)
    assert ir["result"] == IntValue(8)


@pytest.mark.oracle
def test_param_default_int_to_decimal_coercion() -> None:
    """param default that needs int->decimal coercion."""
    source = textwrap.dedent("""\
        param d: decimal = 5
        let result = d + 1.5
        ()
    """)
    legacy, ir = assert_oracle_agrees(source)
    assert legacy["result"] == DecimalValue(decimal.Decimal("6.5"))
    assert ir["result"] == DecimalValue(decimal.Decimal("6.5"))


@pytest.mark.oracle
def test_param_provided_value_overrides_default() -> None:
    """param: provided value overrides the default."""
    source = textwrap.dedent("""\
        param n: int = 10
        let result = n + 1
        ()
    """)
    legacy, ir = assert_oracle_agrees(source, param_values={"n": IntValue(20)})
    assert legacy["result"] == IntValue(21)
    assert ir["result"] == IntValue(21)


@pytest.mark.oracle
def test_param_referenced_in_expression() -> None:
    """param referenced in an expression — oracle agrees."""
    source = textwrap.dedent("""\
        param x: int
        param y: int
        let sum = x + y
        let product = x * y
        ()
    """)
    legacy, ir = assert_oracle_agrees(
        source, param_values={"x": IntValue(3), "y": IntValue(4)}
    )
    assert legacy["sum"] == IntValue(7)
    assert ir["sum"] == IntValue(7)
    assert legacy["product"] == IntValue(12)
    assert ir["product"] == IntValue(12)


@pytest.mark.oracle
def test_param_referenced_inside_function() -> None:
    """param referenced inside a function body — oracle agrees."""
    source = textwrap.dedent("""\
        param base: int
        def double() -> int =
          base * 2
        let result = double()
        ()
    """)
    legacy, ir = assert_oracle_agrees(source, param_values={"base": IntValue(5)})
    assert legacy["result"] == IntValue(10)
    assert ir["result"] == IntValue(10)


@pytest.mark.oracle
def test_param_with_print() -> None:
    """param used in a print call — oracle agrees on stdout and bindings."""
    source = textwrap.dedent("""\
        param msg: text
        print(msg)
        let x = 1
        ()
    """)
    legacy, ir = assert_oracle_agrees(source, param_values={"msg": TextValue("hi")})
    assert legacy["x"] == IntValue(1)
    assert ir["x"] == IntValue(1)
