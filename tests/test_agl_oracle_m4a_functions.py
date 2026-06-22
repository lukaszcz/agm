"""Oracle tests for M4a: top-level user function calls through the IR.

Covers:
- Simple positional args
- Named args (reordered)
- Default arg used
- Return type coercion (int body, decimal return type)
- Arg coercion (int arg, decimal param type)
- Self-recursion (factorial)
- Mutual recursion (even/odd)
- Call-depth guard raises RecursionError in IR
- Closure-valued binding normalized equal across both sides
"""

from __future__ import annotations

import decimal

import pytest

from agm.agl.eval.exceptions import AglRaise
from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.eval.values import BoolValue, DecimalValue, IntValue, TextValue
from agm.agl.lower import lower_program
from agm.agl.parser import parse_program
from agm.agl.scope import resolve
from agm.agl.typecheck import check
from tests.agl.oracle.harness import assert_oracle_agrees, m2_caps

pytestmark = pytest.mark.oracle


# ---------------------------------------------------------------------------
# Oracle tests — both pipelines must agree
# ---------------------------------------------------------------------------


def test_simple_positional_args() -> None:
    """Simple two-arg function called with positional args."""
    source = "def add(x: int, y: int) -> int = x + y\nlet result = add(3, 4)\n()"
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == IntValue(7)


def test_named_args_reordered() -> None:
    """Named args supplied in non-declaration order."""
    source = (
        'def greet(greeting: text, name: text) -> text = greeting + " " + name\n'
        'let result = greet(name: "World", greeting: "Hello")\n()'
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == TextValue("Hello World")


def test_default_arg_used() -> None:
    """Call where the second arg uses its default."""
    source = (
        "def inc(x: int, step: int = 1) -> int = x + step\n"
        "let a = inc(10)\n"
        "let b = inc(10, 5)\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["a"] == IntValue(11)
    assert ir["b"] == IntValue(15)


def test_return_coercion() -> None:
    """Function with int body but decimal return type — coercion applied."""
    source = "def to_dec(x: int) -> decimal = x\nlet result = to_dec(3)\n()"
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == DecimalValue(decimal.Decimal("3"))


def test_arg_coercion() -> None:
    """int argument passed to decimal parameter — coercion at call site."""
    source = "def halve(x: decimal) -> decimal = x / 2.0\nlet result = halve(10)\n()"
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == DecimalValue(decimal.Decimal("5"))


def test_self_recursion_factorial() -> None:
    """Self-recursive factorial function."""
    source = (
        "def factorial(n: int) -> int =\n"
        "  if n <= 1 => 1 else => n * factorial(n - 1)\n"
        "let result = factorial(6)\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == IntValue(720)


def test_mutual_recursion_even_odd() -> None:
    """Mutually recursive even/odd functions."""
    source = (
        "def is_even(n: int) -> bool =\n"
        "  if n = 0 => true else => is_odd(n - 1)\n"
        "def is_odd(n: int) -> bool =\n"
        "  if n = 0 => false else => is_even(n - 1)\n"
        "let r1 = is_even(4)\n"
        "let r2 = is_odd(3)\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["r1"] == BoolValue(True)
    assert ir["r2"] == BoolValue(True)


def test_closure_value_normalized() -> None:
    """Function binding normalizes to sentinel on both sides."""
    source = "def double(x: int) -> int = x * 2\nlet result = double(5)\n()"
    # assert_oracle_agrees normalizes IrClosureValue / Closure to sentinel
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == IntValue(10)


def test_call_depth_guard_ir_only() -> None:
    """IR call-depth guard raises RecursionError after exceeding max_call_depth."""
    source = "def inf(n: int) -> int = inf(n + 1)\nlet result = inf(0)\n()"
    # We only test the IR side here — legacy will hit Python's recursion limit
    # in a non-AglRaise way. The IR must raise AglRaise(RecursionError).
    program = parse_program(source)
    resolved = resolve(program)
    caps = m2_caps()
    checked = check(resolved, caps)
    executable = lower_program(
        checked, source_text=source, source_label="<test>", validate=True
    )
    interp = IrInterpreter(executable, max_call_depth=10)
    with pytest.raises(AglRaise) as exc_info:
        interp.run()
    assert exc_info.value.exc.display_name == "RecursionError"


def test_multiple_calls() -> None:
    """Multiple calls to same function."""
    source = (
        "def square(x: int) -> int = x * x\n"
        "let a = square(3)\n"
        "let b = square(4)\n"
        "let c = square(5)\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["a"] == IntValue(9)
    assert ir["b"] == IntValue(16)
    assert ir["c"] == IntValue(25)


def test_function_calling_another() -> None:
    """One function calling another (non-recursive)."""
    source = (
        "def double(x: int) -> int = x * 2\n"
        "def quad(x: int) -> int = double(double(x))\n"
        "let result = quad(3)\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == IntValue(12)


def test_simple_function_call() -> None:
    """Simple user function call works end-to-end."""
    source = "def f(x: int) -> int = x + 1\nlet result = f(1)\n()"
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == IntValue(2)


def test_function_with_let_in_body() -> None:
    """Function body with let declarations (exercises _walk_collect_locals for LetDecl)."""
    source = (
        "def sum_of_squares(a: int, b: int) -> int =\n"
        "  let sq_a = a * a\n"
        "  let sq_b = b * b\n"
        "  sq_a + sq_b\n"
        "let result = sum_of_squares(3, 4)\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == IntValue(25)


def test_function_captures_outer_variable() -> None:
    """Function captures an outer let-bound variable (exercises capture detection)."""
    source = (
        "let offset = 10\n"
        "def add_offset(x: int) -> int = x + offset\n"
        "let result = add_offset(5)\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == IntValue(15)


def test_function_with_case_in_body() -> None:
    """Function body with case expression (exercises _walk_collect_locals for Case)."""
    source = (
        "enum Color | Red | Green | Blue\n"
        "def color_code(c: Color) -> int =\n"
        "  case c of\n"
        "    | Red() => 1\n"
        "    | Green() => 2\n"
        "    | Blue() => 3\n"
        "let r = color_code(Red())\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["r"] == IntValue(1)


def test_function_with_unary_in_body() -> None:
    """Function body with unary negation (exercises _walk_for_captures for UnaryNeg)."""
    source = (
        "let scale = 2\n"
        "def neg_scaled(x: int) -> int = -(x * scale)\n"
        "let result = neg_scaled(3)\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == IntValue(-6)


def test_function_with_named_args_in_body_call() -> None:
    """Function body calling another function with named args (exercises named_args walk)."""
    source = (
        "def greet(name: text, greeting: text = \"Hi\") -> text =\n"
        "  greeting + \", \" + name + \"!\"\n"
        "def greet_world(g: text) -> text = greet(name: \"World\", greeting: g)\n"
        "let result = greet_world(\"Hello\")\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == TextValue("Hello, World!")


def test_function_with_list_in_body() -> None:
    """Function body containing a list literal (exercises _walk_for_captures for ListLit)."""
    source = (
        "let base = 1\n"
        "def make_list(x: int) -> list[int] = [base, x, x * 2]\n"
        "let result = make_list(3)\n()"
    )
    from agm.agl.eval.values import ListValue

    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == ListValue((IntValue(1), IntValue(3), IntValue(6)))


def test_function_with_field_access_in_body() -> None:
    """Function body with field access (exercises _walk_for_captures for FieldAccess)."""
    source = (
        "record Point\n"
        "  x: int\n"
        "  y: int\n"
        "def get_x(p: Point) -> int = p.x\n"
        "let p = Point(x: 3, y: 4)\n"
        "let result = get_x(p)\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == IntValue(3)


def test_function_with_cast_in_body() -> None:
    """Function body with cast (exercises _walk_for_captures for Cast)."""
    source = (
        "def cast_to_decimal(x: int) -> decimal = x as decimal\n"
        "let result = cast_to_decimal(7)\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == DecimalValue(decimal.Decimal("7"))


def test_function_with_template_in_body() -> None:
    """Function body containing a template literal (exercises _walk_for_captures for Template)."""
    source = (
        "let prefix = \"Item\"\n"
        "def label(n: int) -> text = \"${prefix} #${n}\"\n"
        "let result = label(5)\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == TextValue("Item #5")


def test_function_with_try_in_body() -> None:
    """Function body with try/catch (exercises _walk_collect_locals for Try)."""
    source = (
        "def safe_add(a: int, b: int) -> int =\n"
        "  try\n"
        "    a + b\n"
        "  catch ArithmeticError =>\n"
        "    0\n"
        "let result = safe_add(3, 4)\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == IntValue(7)


def test_function_with_do_loop_in_body() -> None:
    """Function body with do loop (exercises _walk_collect_locals for Do)."""
    source = (
        "def count_to(n: int) -> int =\n"
        "  var i = 0\n"
        "  do\n"
        "    i := i + 1\n"
        "  until i >= n\n"
        "  i\n"
        "let result = count_to(5)\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == IntValue(5)


def test_function_with_var_and_assignment_capture() -> None:
    """Function body with var + assignment capturing outer variable."""
    source = (
        "let factor = 3\n"
        "def triple_then_add(x: int, y: int) -> int =\n"
        "  var acc = factor * x\n"
        "  acc := acc + y\n"
        "  acc\n"
        "let result = triple_then_add(4, 5)\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == IntValue(17)


def test_function_with_raise_in_body() -> None:
    """Function body with raise (exercises _walk_for_captures for Raise)."""
    source = (
        "def checked_inc(n: int) -> int =\n"
        "  if n < 0 =>\n"
        "    raise Abort(message: \"negative\")\n"
        "  else =>\n"
        "    n + 1\n"
        "let result = checked_inc(5)\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == IntValue(6)


def test_function_with_index_access_and_capture() -> None:
    """Function body with index access capturing outer list."""
    source = (
        "let items = [10, 20, 30]\n"
        "def get_item(i: int) -> int = items[i]\n"
        "let result = get_item(1)\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == IntValue(20)


def test_function_with_dict_literal_and_capture() -> None:
    """Function body with dict literal capturing outer variable."""
    source = (
        "let base = 10\n"
        "def make_dict(x: int) -> dict[text, int] = {\"a\": base + x, \"b\": x}\n"
        "let result = make_dict(5)\n()"
    )
    from agm.agl.eval.values import DictValue

    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == DictValue({"a": IntValue(15), "b": IntValue(5)})


def test_function_with_is_test_and_capture() -> None:
    """Function body with is-test capturing outer enum value."""
    source = (
        "enum Color | Red | Green | Blue\n"
        "let my_color = Red()\n"
        "def check_red(c: Color) -> bool = c is Red or my_color is Red\n"
        "let result = check_red(Green())\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == BoolValue(True)


def test_function_captures_mutable_outer_var() -> None:
    """Function captures a mutable outer var (by_cell=True capture path)."""
    source = (
        "var counter = 0\n"
        "def get_counter() -> int = counter\n"
        "let result = get_counter()\n()"
    )
    legacy, ir = assert_oracle_agrees(source)
    assert ir["result"] == IntValue(0)
