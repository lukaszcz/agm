"""IR evaluation tests for print, parse_json, copy/shallow_copy, and entry params.

Each test evaluates a program through the IR pipeline and asserts
the produced bindings and stdout.

"""

from __future__ import annotations

import decimal
import textwrap

from agm.agl.semantics.values import (
    ArrayValue,
    BoolValue,
    DecimalValue,
    IntValue,
    JsonValue,
    TextValue,
    UnitValue,
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


def test_print_array() -> None:
    """print(array) — IR pipeline renders array correctly."""
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
    assert out == "Color::Green\n"


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
    source = "let j = parse_json('\"hello\"')\n()"
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
# copy / shallow_copy
# ===========================================================================


def test_shallow_copy_detaches_top_level_shares_nested() -> None:
    """shallow_copy detaches the top-level array; a nested array stays shared."""
    source = textwrap.dedent("""\
        var inner = [1, 2]
        var outer = [inner]
        let copied = shallow_copy(outer)
        copied[0][0] := 99
        print(outer)
        print(copied)
        outer[0] := [7]
        print(outer)
        print(copied)
        ()
    """)
    out = evaluate_ir_output(source)
    assert out == "[[99, 2]]\n[[99, 2]]\n[[7]]\n[[99, 2]]\n"


def test_copy_detaches_fully_at_every_depth() -> None:
    """copy fully detaches a nested array — no mutation crosses in either direction."""
    source = textwrap.dedent("""\
        var inner = [1, 2]
        var outer = [inner]
        let copied = copy(outer)
        copied[0][0] := 99
        print(outer)
        print(copied)
        inner[0] := -1
        print(outer)
        print(copied)
        ()
    """)
    out = evaluate_ir_output(source)
    assert out == "[[1, 2]]\n[[99, 2]]\n[[-1, 2]]\n[[99, 2]]\n"


def test_copy_of_diamond_preserves_sharing() -> None:
    """copy of a diamond-shared array preserves the sharing in the copy."""
    source = textwrap.dedent("""\
        var shared = [1]
        var outer = [shared, shared]
        let copied = copy(outer)
        copied[0][0] := 42
        print(outer)
        print(copied)
        ()
    """)
    out = evaluate_ir_output(source)
    assert out == "[[1], [1]]\n[[42], [42]]\n"


def test_copy_of_cyclic_value_does_not_raise_but_print_of_copy_does() -> None:
    """copy(cyclic) succeeds (the memo makes it terminate); print(copy(...)) still raises."""
    source = textwrap.dedent("""\
        record Node(children: array[Node])
        var xs: array[Node] = [Node(children = [])]
        let n = Node(children = xs)
        xs[0] := n
        let c = copy(xs)
        print("copy ok")
        try
          print(c)
        catch CyclicValueError =>
          print("caught")
        ()
    """)
    out = evaluate_ir_output(source)
    assert out == "copy ok\ncaught\n"


def test_copy_of_cyclic_value_produces_independent_isomorphic_cycle() -> None:
    """copy of a cyclic value terminates, and the copy is a genuinely separate cycle.

    Equality is cycle-safe and co-inductive (``==`` never raises), so it is the
    tool used here to observe the copy's shape without ever rendering it.
    """
    source = textwrap.dedent("""\
        record Node(children: array[Node], tag: array[int])
        var xs: array[Node] = [Node(children = [], tag = [0])]
        let n = Node(children = xs, tag = [0])
        xs[0] := n
        let ys = copy(xs)
        print(xs == ys)
        xs[0].tag[0] := 99
        print(xs == ys)
        ()
    """)
    out = evaluate_ir_output(source)
    assert out == "true\nfalse\n"


def test_copy_and_shallow_copy_are_identity_on_primitives() -> None:
    """copy/shallow_copy of int, decimal, bool, text, unit return an equal value."""
    source = textwrap.dedent("""\
        let a = copy(5)
        let b = shallow_copy(true)
        let c = copy("hi")
        let d: decimal = shallow_copy(3.5)
        let e = copy(())
        ()
    """)
    ir = evaluate_ir(source)
    assert ir["a"] == IntValue(5)
    assert ir["b"] == BoolValue(True)
    assert ir["c"] == TextValue("hi")
    assert ir["d"] == DecimalValue(decimal.Decimal("3.5"))
    assert ir["e"] == UnitValue()


def test_copy_through_record_detaches_field_shallow_copy_shares_it() -> None:
    """copy detaches a record's array field; shallow_copy shares it."""
    source = textwrap.dedent("""\
        record Box(items: array[int])
        var b = Box(items = [1, 2])
        let deep = copy(b)
        let shallow = shallow_copy(b)
        deep.items[0] := -1
        shallow.items[0] := -2
        print(b.items)
        print(deep.items)
        print(shallow.items)
        ()
    """)
    out = evaluate_ir_output(source)
    assert out == "[-2, 2]\n[-1, 2]\n[-2, 2]\n"


def test_copy_through_enum_field() -> None:
    """copy detaches an enum variant's array field."""
    source = textwrap.dedent("""\
        enum Bag
          | Full(items: array[int])
        var full: Bag = Full(items = [1, 2])
        let d = copy(full)
        case d of
          | Full(items) => items[0] := -1
        case full of
          | Full(items) => print(items)
    """)
    out = evaluate_ir_output(source)
    assert out == "[1, 2]\n"


def test_copy_through_exception_field() -> None:
    """copy detaches a caught exception's array field."""
    source = textwrap.dedent("""\
        exception Batch extends Exception
          items: array[int]
        try
          raise Batch(message = "boom", items = [1, 2])
        catch Batch as e =>
          let d = copy(e)
          d.items[0] := -1
          print(e.items)
          print(d.items)
        ()
    """)
    out = evaluate_ir_output(source)
    assert out == "[1, 2]\n[-1, 2]\n"


def test_copy_of_generic_container_dict_of_arrays() -> None:
    """copy detaches a dict[text, array[int]] — a generic container nesting another."""
    source = textwrap.dedent("""\
        var xs: dict[text, array[int]] = {"a": [1, 2]}
        let ys = copy(xs)
        ys["a"][0] := -1
        print(xs["a"])
        print(ys["a"])
        ()
    """)
    out = evaluate_ir_output(source)
    assert out == "[1, 2]\n[-1, 2]\n"


def test_copy_explicit_type_arg_coerces_scalar() -> None:
    """copy::[decimal](5) coerces the int argument to decimal via the checker's own type."""
    source = "let x = copy::[decimal](5)\n()"
    ir = evaluate_ir(source)
    assert isinstance(ir["x"], DecimalValue)
    assert ir["x"] == DecimalValue(decimal.Decimal(5))


def test_shallow_copy_returns_new_array_object() -> None:
    """shallow_copy(array) returns a distinct ArrayValue, not the same object."""
    source = "var xs = [1, 2]\nlet ys = shallow_copy(xs)\n()"
    ir = evaluate_ir(source)
    assert isinstance(ir["xs"], ArrayValue)
    assert isinstance(ir["ys"], ArrayValue)
    assert ir["xs"] is not ir["ys"]
    assert ir["xs"] == ir["ys"]


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
    ir = evaluate_ir(source, param_values={"x": IntValue(3), "y": IntValue(4)})
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
