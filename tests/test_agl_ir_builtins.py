"""IR evaluation tests for print, parse_json, and entry params.

Each test evaluates a program through the IR pipeline and asserts
the produced bindings and stdout.

"""

from __future__ import annotations

import decimal
import textwrap

from agm.agl.semantics.values import (
    DecimalValue,
    IntValue,
    JsonValue,
    TextValue,
)
from tests.agl.ir_harness import (
    evaluate_ir,
    evaluate_ir_output,
    evaluate_ir_raises,
)

# ===========================================================================
# print — various value types
# ===========================================================================


def test_print_int() -> None:
    """print(int) — IR pipeline produces correct stdout and bindings."""
    source = "let x = 1\nprint(x)\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == IntValue(1)
    out = evaluate_ir_output(source)
    assert out == "1\n"


def test_print_decimal() -> None:
    """print(decimal) — rendered as decimal string."""
    source = "let d: decimal = 3.14\nprint(d)\n()"
    out = evaluate_ir_output(source)
    assert out == "3.14\n"


def test_print_text() -> None:
    """print(text) — rendered as raw text."""
    source = 'let x = "hello world"\nprint(x)\n()'
    out = evaluate_ir_output(source)
    assert out == "hello world\n"


def test_print_bool_true() -> None:
    """print(bool) — true rendered as 'true'."""
    source = "let x = true\nprint(x)\n()"
    out = evaluate_ir_output(source)
    assert out == "true\n"


def test_print_bool_false() -> None:
    """print(bool) — false rendered as 'false'."""
    source = "let x = false\nprint(x)\n()"
    out = evaluate_ir_output(source)
    assert out == "false\n"


def test_print_list() -> None:
    """print(list) — IR pipeline renders list correctly."""
    source = "let x = [1, 2, 3]\nprint(x)\n()"
    out = evaluate_ir_output(source)
    assert out == "[1, 2, 3]\n"


def test_print_dict() -> None:
    """print(dict) — IR pipeline renders dict correctly."""
    source = 'let x = {"a": 1, "b": 2}\nprint(x)\n()'
    out = evaluate_ir_output(source)
    assert out == '{"a": 1, "b": 2}\n'


def test_print_record() -> None:
    """print(record) — IR pipeline renders record correctly."""
    source = textwrap.dedent("""\
        record Point
          x: int
          y: int
        let p = Point(x = 10, y = 20)
        print(p)
        ()
    """)
    out = evaluate_ir_output(source)
    assert out == "Point(x = 10, y = 20)\n"


def test_print_enum_variant() -> None:
    """print(enum variant) — IR pipeline renders enum variant correctly."""
    source = textwrap.dedent("""\
        enum Color
          | Red
          | Green
          | Blue
        let c = Green
        print(c)
        ()
    """)
    out = evaluate_ir_output(source)
    assert out == 'Color::Green\n'


def test_print_multiple_calls() -> None:
    """Multiple print calls — all produce output correctly."""
    source = textwrap.dedent("""\
        let x = 10
        print("line1")
        print(x)
        print(true)
        ()
    """)
    out = evaluate_ir_output(source)
    assert out == "line1\n10\ntrue\n"


# ===========================================================================
# print inside control flow / function body
# ===========================================================================


def test_print_inside_if() -> None:
    """print inside an if branch — IR pipeline produces expected output."""
    source = textwrap.dedent("""\
        let cond = true
        if cond =>
          print("yes")
        else =>
          ()
        ()
    """)
    out = evaluate_ir_output(source)
    assert out == "yes\n"


def test_print_inside_function() -> None:
    """print inside a function body — IR pipeline produces expected output."""
    source = textwrap.dedent("""\
        def greet(name: text) -> unit =
          print(name)
        greet("Alice")
        ()
    """)
    out = evaluate_ir_output(source)
    assert out == "Alice\n"


def test_print_inside_loop() -> None:
    """print inside a do…until loop — IR pipeline produces expected output."""
    source = textwrap.dedent("""\
        var i = 0
        do[10]
          print(i)
          i := i + 1
        until i >= 3
        ()
    """)
    out = evaluate_ir_output(source)
    assert out == "0\n1\n2\n"


# ===========================================================================
# parse_json — success and failure
# ===========================================================================


def test_parse_json_success_object() -> None:
    """parse_json succeeds for a JSON object."""
    source = "let j = parse_json('{\"key\": 42}')\n()"
    ir = evaluate_ir(source)
    assert isinstance(ir["j"], JsonValue)


def test_parse_json_success_array() -> None:
    """parse_json succeeds for a JSON array."""
    source = "let j = parse_json('[1, 2, 3]')\n()"
    ir = evaluate_ir(source)
    assert isinstance(ir["j"], JsonValue)


def test_parse_json_success_string() -> None:
    """parse_json succeeds for a JSON string."""
    source = 'let j = parse_json(\'\"hello\"\')\n()'
    ir = evaluate_ir(source)
    assert isinstance(ir["j"], JsonValue)


def test_parse_json_success_number() -> None:
    """parse_json succeeds for a JSON number."""
    source = "let j = parse_json('123')\n()"
    ir = evaluate_ir(source)
    assert isinstance(ir["j"], JsonValue)


def test_parse_json_success_null() -> None:
    """parse_json('null') returns JsonValue(None)."""
    source = "let j = parse_json('null')\n()"
    ir = evaluate_ir(source)
    assert ir["j"] == JsonValue(None)


def test_parse_json_failure_malformed() -> None:
    """parse_json raises JsonParseError on malformed input."""
    source = "let j = parse_json('not-json')\n()"
    evaluate_ir_raises(source)


def test_parse_json_failure_empty() -> None:
    """parse_json raises JsonParseError on empty input."""
    source = "let j = parse_json('')\n()"
    evaluate_ir_raises(source)


def test_parse_json_failure_trailing_garbage() -> None:
    """parse_json raises JsonParseError when trailing content follows valid JSON."""
    source = "let j = parse_json('1 2 3')\n()"
    evaluate_ir_raises(source)


def test_parse_json_caught_by_try() -> None:
    """parse_json error caught in try — IR pipeline handles caught exception correctly."""
    source = textwrap.dedent("""\
        var result: text = "default"
        try
          let j = parse_json('bad')
          result := "ok"
        catch JsonParseError as e =>
          result := "caught"
        ()
    """)
    ir = evaluate_ir(source)
    assert ir["result"] == TextValue("caught")


# ===========================================================================
# param declarations
# ===========================================================================


def test_param_provided_value() -> None:
    """param with provided value — evaluated via the IR pipeline."""
    source = textwrap.dedent("""\
        param name: text
        let greeting = "Hello, " + name
        ()
    """)
    ir = evaluate_ir(source, param_values={"name": TextValue("World")})
    assert ir["greeting"] == TextValue("Hello, World")


def test_param_provided_int_value() -> None:
    """param int provided — evaluated via the IR pipeline."""
    source = textwrap.dedent("""\
        param count: int
        let doubled = count * 2
        ()
    """)
    ir = evaluate_ir(source, param_values={"count": IntValue(5)})
    assert ir["doubled"] == IntValue(10)


def test_param_default_used_when_no_value() -> None:
    """param with default evaluated when no value provided."""
    source = textwrap.dedent("""\
        param n: int = 7
        let result = n + 1
        ()
    """)
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(8)


def test_param_default_int_to_decimal_coercion() -> None:
    """param default that needs int->decimal coercion."""
    source = textwrap.dedent("""\
        param d: decimal = 5
        let result = d + 1.5
        ()
    """)
    ir = evaluate_ir(source)
    assert ir["result"] == DecimalValue(decimal.Decimal("6.5"))


def test_param_provided_value_overrides_default() -> None:
    """param: provided value overrides the default."""
    source = textwrap.dedent("""\
        param n: int = 10
        let result = n + 1
        ()
    """)
    ir = evaluate_ir(source, param_values={"n": IntValue(20)})
    assert ir["result"] == IntValue(21)


def test_param_referenced_in_expression() -> None:
    """param referenced in an expression — IR pipeline produces correct result."""
    source = textwrap.dedent("""\
        param x: int
        param y: int
        let sum = x + y
        let product = x * y
        ()
    """)
    ir = evaluate_ir(
        source, param_values={"x": IntValue(3), "y": IntValue(4)}
    )
    assert ir["sum"] == IntValue(7)
    assert ir["product"] == IntValue(12)


def test_param_referenced_inside_function() -> None:
    """param referenced inside a function body — IR pipeline produces correct result."""
    source = textwrap.dedent("""\
        param base: int
        def double() -> int =
          base * 2
        let result = double()
        ()
    """)
    ir = evaluate_ir(source, param_values={"base": IntValue(5)})
    assert ir["result"] == IntValue(10)


def test_param_with_print() -> None:
    """param used in a print call — IR pipeline produces correct stdout and bindings."""
    source = textwrap.dedent("""\
        param msg: text
        print(msg)
        let x = 1
        ()
    """)
    ir = evaluate_ir(source, param_values={"msg": TextValue("hi")})
    assert ir["x"] == IntValue(1)
    out = evaluate_ir_output(source, param_values={"msg": TextValue("hi")})
    assert out == "hi\n"
