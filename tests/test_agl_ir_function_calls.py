"""IR evaluation tests for top-level user function calls.

Covers:
- Simple positional args
- Named args (reordered)
- Default arg used
- Return type coercion (int body, decimal return type)
- Arg coercion (int arg, decimal param type)
- Self-recursion (factorial)
- Mutual recursion (even/odd)
- Call-depth guard raises RecursionError in IR
- Closure-valued binding normalized in the IR pipeline
"""

from __future__ import annotations

import decimal

import pytest

from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.lower import lower_program
from agm.agl.parser import parse_program
from agm.agl.scope import resolve
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import BoolValue, DecimalValue, IntValue, TextValue
from agm.agl.typecheck import check
from tests.agl.ir_harness import evaluate_ir, m2_caps

# ---------------------------------------------------------------------------
# Basic function call tests
# ---------------------------------------------------------------------------


def test_simple_positional_args() -> None:
    """Simple two-arg function called with positional args."""
    source = "def add(x: int, y: int) -> int = x + y\nlet result = add(3, 4)\n()"
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(7)


def test_named_args_reordered() -> None:
    """Named args supplied in non-declaration order."""
    source = (
        'def greet(greeting: text, name: text) -> text = greeting + " " + name\n'
        'let result = greet(name: "World", greeting: "Hello")\n()'
    )
    ir = evaluate_ir(source)
    assert ir["result"] == TextValue("Hello World")


def test_default_arg_used() -> None:
    """Call where the second arg uses its default."""
    source = (
        "def inc(x: int, step: int = 1) -> int = x + step\n"
        "let a = inc(10)\n"
        "let b = inc(10, 5)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["a"] == IntValue(11)
    assert ir["b"] == IntValue(15)


def test_return_coercion() -> None:
    """Function with int body but decimal return type — coercion applied."""
    source = "def to_dec(x: int) -> decimal = x\nlet result = to_dec(3)\n()"
    ir = evaluate_ir(source)
    assert ir["result"] == DecimalValue(decimal.Decimal("3"))


def test_arg_coercion() -> None:
    """int argument passed to decimal parameter — coercion at call site."""
    source = "def halve(x: decimal) -> decimal = x / 2.0\nlet result = halve(10)\n()"
    ir = evaluate_ir(source)
    assert ir["result"] == DecimalValue(decimal.Decimal("5"))


def test_self_recursion_factorial() -> None:
    """Self-recursive factorial function."""
    source = (
        "def factorial(n: int) -> int =\n"
        "  if n <= 1 => 1 else => n * factorial(n - 1)\n"
        "let result = factorial(6)\n()"
    )
    ir = evaluate_ir(source)
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
    ir = evaluate_ir(source)
    assert ir["r1"] == BoolValue(True)
    assert ir["r2"] == BoolValue(True)


def test_closure_value_normalized() -> None:
    """Function binding normalizes to sentinel in the IR pipeline."""
    source = "def double(x: int) -> int = x * 2\nlet result = double(5)\n()"
    # evaluate_ir normalizes IrClosureValue / Closure to sentinel
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(10)


def test_call_depth_guard_ir_only() -> None:
    """IR call-depth guard raises RecursionError at a custom low depth (IR-side only)."""
    from agm.agl.semantics.values import TextValue

    source = "def inf(n: int) -> int = inf(n + 1)\nlet result = inf(0)\n()"
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
    exc = exc_info.value.exc
    assert exc.display_name == "RecursionError"
    assert exc.fields["message"] == TextValue("Maximum call depth (10) exceeded")
    assert exc.fields["limit"] == IntValue(10)


def test_multiple_calls() -> None:
    """Multiple calls to same function."""
    source = (
        "def square(x: int) -> int = x * x\n"
        "let a = square(3)\n"
        "let b = square(4)\n"
        "let c = square(5)\n()"
    )
    ir = evaluate_ir(source)
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
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(12)


def test_simple_function_call() -> None:
    """Simple user function call works end-to-end."""
    source = "def f(x: int) -> int = x + 1\nlet result = f(1)\n()"
    ir = evaluate_ir(source)
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
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(25)


def test_function_captures_outer_variable() -> None:
    """Function captures an outer let-bound variable (exercises capture detection)."""
    source = (
        "let offset = 10\n"
        "def add_offset(x: int) -> int = x + offset\n"
        "let result = add_offset(5)\n()"
    )
    ir = evaluate_ir(source)
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
    ir = evaluate_ir(source)
    assert ir["r"] == IntValue(1)


def test_function_with_unary_in_body() -> None:
    """Function body with unary negation (exercises _walk_for_captures for UnaryNeg)."""
    source = (
        "let scale = 2\n"
        "def neg_scaled(x: int) -> int = -(x * scale)\n"
        "let result = neg_scaled(3)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(-6)


def test_function_with_named_args_in_body_call() -> None:
    """Function body calling another function with named args (exercises named_args walk)."""
    source = (
        "def greet(name: text, greeting: text = \"Hi\") -> text =\n"
        "  greeting + \", \" + name + \"!\"\n"
        "def greet_world(g: text) -> text = greet(name: \"World\", greeting: g)\n"
        "let result = greet_world(\"Hello\")\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == TextValue("Hello, World!")


def test_function_with_list_in_body() -> None:
    """Function body containing a list literal (exercises _walk_for_captures for ListLit)."""
    source = (
        "let base = 1\n"
        "def make_list(x: int) -> list[int] = [base, x, x * 2]\n"
        "let result = make_list(3)\n()"
    )
    from agm.agl.semantics.values import ListValue

    ir = evaluate_ir(source)
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
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(3)


def test_function_with_cast_in_body() -> None:
    """Function body with cast (exercises _walk_for_captures for Cast)."""
    source = (
        "def cast_to_decimal(x: int) -> decimal = x as decimal\n"
        "let result = cast_to_decimal(7)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == DecimalValue(decimal.Decimal("7"))


def test_function_with_template_in_body() -> None:
    """Function body containing a template literal (exercises _walk_for_captures for Template)."""
    source = (
        "let prefix = \"Item\"\n"
        "def label(n: int) -> text = \"${prefix} #${n}\"\n"
        "let result = label(5)\n()"
    )
    ir = evaluate_ir(source)
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
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(7)


def test_function_with_do_loop_in_body() -> None:
    """Function body with do loop (exercises _walk_collect_locals for Loop)."""
    source = (
        "def count_to(n: int) -> int =\n"
        "  var i = 0\n"
        "  do\n"
        "    i := i + 1\n"
        "  until i >= n\n"
        "  i\n"
        "let result = count_to(5)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(5)


def test_function_with_do_done_loop_in_body() -> None:
    """Function body with do-done loop (exercises _scan_captures for until_cond=None).

    `do[0] done` runs zero iterations (bound=0 ≤ 0 exits immediately).
    """
    source = (
        "def run_nothing(n: int) -> int =\n"
        "  var i = 0\n"
        "  do[n]\n"
        "    i := i + 1\n"
        "  done\n"
        "  i\n"
        "let result = run_nothing(0)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(0)


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
    ir = evaluate_ir(source)
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
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(6)


def test_function_with_index_access_and_capture() -> None:
    """Function body with index access capturing outer list."""
    source = (
        "let items = [10, 20, 30]\n"
        "def get_item(i: int) -> int = items[i]\n"
        "let result = get_item(1)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(20)


def test_function_with_dict_literal_and_capture() -> None:
    """Function body with dict literal capturing outer variable."""
    source = (
        "let base = 10\n"
        "def make_dict(x: int) -> dict[text, int] = {\"a\": base + x, \"b\": x}\n"
        "let result = make_dict(5)\n()"
    )
    from agm.agl.semantics.values import DictValue

    ir = evaluate_ir(source)
    assert ir["result"] == DictValue({"a": IntValue(15), "b": IntValue(5)})


def test_function_with_is_test_and_capture() -> None:
    """Function body with is-test capturing outer enum value."""
    source = (
        "enum Color | Red | Green | Blue\n"
        "let my_color = Red()\n"
        "def check_red(c: Color) -> bool = c is Red or my_color is Red\n"
        "let result = check_red(Green())\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == BoolValue(True)


def test_function_captures_mutable_outer_var() -> None:
    """Function captures a mutable outer var (by_cell=True capture path)."""
    source = (
        "var counter = 0\n"
        "def get_counter() -> int = counter\n"
        "let result = get_counter()\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(0)


# ---------------------------------------------------------------------------
# B1/capture fix tests (review-fixes task)
# ---------------------------------------------------------------------------


def test_index_target_capture() -> None:
    """B1 fix: function mutating outer var via index target captures the var correctly.

    `arr[k] := 99` inside a function must capture both the outer `var arr` (IndexTarget
    root) and the outer `let k` (index expression).  Before the fix the root was missed
    and lowering raised InvalidIrError.  Evaluation must yield arr = [0, 99, 0].
    """
    from agm.agl.semantics.values import ListValue

    source = (
        "var arr = [0, 0, 0]\n"
        "let k = 1\n"
        "def setit() -> unit =\n"
        "  arr[k] := 99\n"
        "setit()\n()"
    )
    ir = evaluate_ir(source)
    assert ir["arr"] == ListValue((IntValue(0), IntValue(99), IntValue(0)))


def test_assignment_as_function_result_yields_unit() -> None:
    """An assignment statement yields unit, even as a function's return value.

    Regression: the IR's IrAssign previously returned the assigned value rather
    than unit.  This was invisible while assignment results were always discarded
    (non-tail block items), but a `unit`-returning function whose body IS the
    assignment exposes it: `let z = setit()` must observe UnitValue in both
    evaluators, not the mutated container/value.
    """
    from agm.agl.semantics.values import UnitValue

    # Index-target assignment as the function body / return value.
    index_source = (
        "var arr = [0, 0, 0]\n"
        "let k = 1\n"
        "def setit() -> unit =\n"
        "  arr[k] := 99\n"
        "let z = setit()\n"
        "()"
    )
    ir = evaluate_ir(index_source)
    assert ir["z"] == UnitValue()

    # Name-target assignment as the function body / return value.
    name_source = (
        "var counter = 0\n"
        "def reset() -> unit =\n"
        "  counter := 5\n"
        "let z = reset()\n"
        "()"
    )
    ir2 = evaluate_ir(name_source)
    assert ir2["z"] == UnitValue()


def test_name_target_only_assign_capture() -> None:
    """Fix: a function whose only reference to an outer var is an assignment captures it.

    `counter := 5` must capture the outer `var counter` even though it is never
    READ inside the function.  Before the fix the NameTarget was not walked and the
    capture was missed, causing IrAssign to fail at runtime.
    """
    source = (
        "var counter = 0\n"
        "def reset() -> unit =\n"
        "  counter := 5\n"
        "reset()\n()"
    )
    ir = evaluate_ir(source)
    assert ir["counter"] == IntValue(5)


def test_capture_through_nested_positions_and_pattern_locals() -> None:
    """Fix: captures work through case/if/list/template/call-arg; pattern binders are local.

    Exercises:
    - Outer let used only inside a `case` arm (captured correctly).
    - A `case` branch that binds a pattern variable: that variable must NOT be treated
      as a capture of an outer binding (it is a local binder in the branch body).
    """
    source = (
        "enum Shape | Circle(radius: int) | Square(side: int)\n"
        "let multiplier = 3\n"
        "def describe(s: Shape) -> int =\n"
        "  case s of\n"
        "    | Circle(radius: r) => r * multiplier\n"
        "    | Square(side: sd) => sd * multiplier\n"
        "let c = describe(Shape.Circle(radius: 4))\n"
        "let sq = describe(Shape.Square(side: 5))\n()"
    )
    ir = evaluate_ir(source)
    assert ir["c"] == IntValue(12)
    assert ir["sq"] == IntValue(15)


def test_recursion_depth_limit() -> None:
    """RecursionError is raised with the expected message and limit at DEFAULT depth.

    Verifies that the IR pipeline raises RecursionError including message text and
    the ``limit`` field value.
    """
    source = "def loop(n: int) -> int = loop(n + 1)\nlet result = loop(0)\n()"
    from tests.agl.ir_harness import evaluate_ir_raises

    ir_exc = evaluate_ir_raises(source)
    # IR pipeline must raise RecursionError
    assert ir_exc.display_name == "RecursionError"
    # The limit field must match DEFAULT_MAX_CALL_DEPTH = 256
    assert ir_exc.fields["limit"] == IntValue(256)


def test_case_wildcard_and_literal_patterns_in_function_body() -> None:
    """_pattern_binding_ids covers WildcardPattern and LiteralPattern arms.

    A function body with case branches using ``_`` (wildcard) and a literal
    exercises the WildcardPattern/LiteralPattern arm that calls `pass`.
    Neither introduces a new binder, so local_ids is unaffected.
    """
    source = (
        "def classify(n: int) -> int =\n"
        "  case n of\n"
        "    | 0 => -1\n"
        "    | 1 => 1\n"
        "    | _ => 0\n"
        "let a = classify(0)\n"
        "let b = classify(1)\n"
        "let c = classify(99)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["a"] == IntValue(-1)
    assert ir["b"] == IntValue(1)
    assert ir["c"] == IntValue(0)


def test_bare_variant_pattern_in_function_body() -> None:
    """_pattern_binding_ids covers VarPattern-as-bare-constructor branch.

    A bare name in a case pattern (e.g. ``| On => ...``) is a VarPattern whose
    node_id appears in ``bare_variant_patterns``.  The bare-constructor branch is not taken
    so the node_id is NOT added to local_ids (correct: it is not a binder).
    """
    source = (
        "enum Flag | On | Off\n"
        "let flag = Flag.On()\n"
        "def check(f: Flag) -> int =\n"
        "  case f of\n"
        "    | On => 1\n"
        "    | Off => 0\n"
        "let r = check(flag)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["r"] == IntValue(1)
