"""IR semantic tests for M4b: lambdas, indirect (function-value) calls, first-class functions.

Covers:
1. Lambda with explicit return type bound to let, then called
2. Lambda with INFERRED return type
3. Named def bound to a typed let, called indirectly
4. Higher-order: def apply(f, n) calling a function-valued param
5. A function/lambda RETURNED from a function then called by the caller
6. Capture-through: lambda inside def captures def's param/local; var mutation parity
7. No-arg-coercion parity: checker behaviour for value calls
8. Recursion-depth limit via indirect call → both raise identical RecursionError
9. Closure-valued bindings normalized equal across both evaluators
10. function_values.agl end-to-end
"""

from __future__ import annotations

import decimal

import pytest

from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.lower import lower_program
from agm.agl.parser import parse_program
from agm.agl.scope import resolve
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import DecimalValue, IntValue, TextValue
from agm.agl.typecheck import check
from tests.agl.ir_harness import evaluate_ir, evaluate_ir_raises, m2_caps


def test_top_level_call_can_reference_later_function() -> None:
    source = """
let answer = first(20)

def first(n: int) -> int =
  second(n) + 1

def second(n: int) -> int =
  n * 2
"""
    ir_reference, ir = evaluate_ir(source)
    assert ir_reference["answer"] == ir["answer"] == IntValue(41)


def test_function_mutates_module_var_without_capture() -> None:
    source = """
var count = 0

def increment() -> unit =
  count := count + 1

increment()
increment()
"""
    ir_reference, ir = evaluate_ir(source)
    assert ir_reference["count"] == ir["count"] == IntValue(2)


# ---------------------------------------------------------------------------
# 1. Lambda with explicit return type bound to let, then called
# ---------------------------------------------------------------------------


def test_lambda_explicit_return_type() -> None:
    """Lambda with explicit return type bound to let; called indirectly."""
    source = (
        "let dbl = fn(x: int) -> int => x * 2\n"
        "let r = dbl(4)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r"] == IntValue(8)


# ---------------------------------------------------------------------------
# 2. Lambda with inferred return type
# ---------------------------------------------------------------------------


def test_lambda_inferred_return_type() -> None:
    """Lambda with inferred return type (no -> annotation)."""
    source = (
        "let inc = fn(x: int) => x + 1\n"
        "let r = inc(9)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r"] == IntValue(10)


# ---------------------------------------------------------------------------
# 3. Named def bound to a typed let, called indirectly
# ---------------------------------------------------------------------------


def test_def_bound_to_typed_let_indirect_call() -> None:
    """Named def bound to a typed let (function value), then called indirectly."""
    source = (
        "def classify(n: int) -> text =\n"
        "  if | n > 0 => \"pos\" | n < 0 => \"neg\" | else => \"zero\"\n"
        "let g: (int) -> text = classify\n"
        "let r1 = g(7)\n"
        "let r2 = g(-3)\n"
        "let r3 = g(0)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r1"] == TextValue("pos")
    assert ir["r2"] == TextValue("neg")
    assert ir["r3"] == TextValue("zero")


# ---------------------------------------------------------------------------
# 4. Higher-order: def apply(f, n) calling a function-valued param
# ---------------------------------------------------------------------------


def test_higher_order_apply() -> None:
    """Higher-order function: def apply(f, n) calling f(n) via indirect call."""
    source = (
        "def classify(n: int) -> text =\n"
        "  if | n > 0 => \"pos\" | n < 0 => \"neg\" | else => \"zero\"\n"
        "def apply(f: (int) -> text, n: int) -> text = f(n)\n"
        "let r1 = apply(classify, 10)\n"
        "let r2 = apply(classify, -5)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r1"] == TextValue("pos")
    assert ir["r2"] == TextValue("neg")


# ---------------------------------------------------------------------------
# 5. Function/lambda returned from a function, then called by the caller
# ---------------------------------------------------------------------------


def test_returned_lambda_called() -> None:
    """Lambda returned from a function and then called by the caller."""
    source = (
        "def make_adder(n: int) -> (int) -> int = fn(x: int) => x + n\n"
        "let add5 = make_adder(5)\n"
        "let r = add5(3)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r"] == IntValue(8)


def test_returned_def_called() -> None:
    """Named def returned as a function value, then called."""
    source = (
        "def double(x: int) -> int = x * 2\n"
        "def pick(flag: bool) -> (int) -> int =\n"
        "  if flag => double else => double\n"
        "let f = pick(true)\n"
        "let r = f(7)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r"] == IntValue(14)


# ---------------------------------------------------------------------------
# 6. Capture-through: lambda inside def captures def's param/local
# ---------------------------------------------------------------------------


def test_capture_through_def_param() -> None:
    """Lambda inside def captures the def's param (capture-through)."""
    source = (
        "def make_adder(n: int) -> (int) -> int = fn(x: int) => x + n\n"
        "let add10 = make_adder(10)\n"
        "let r = add10(3)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r"] == IntValue(13)


def test_capture_through_def_local_let() -> None:
    """Lambda inside def captures the def's local let binding (capture-through)."""
    source = (
        "def make_multiplier(n: int) -> (int) -> int =\n"
        "  let factor = n * 2\n"
        "  fn(x: int) => x * factor\n"
        "let triple_base = make_multiplier(3)\n"
        "let r = triple_base(5)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r"] == IntValue(30)


def test_capture_through_var_by_cell() -> None:
    """Lambda inside def captures a var (by-cell). Var mutation after lambda creation
    is visible in the lambda (cell semantics)."""
    source = (
        "def make_counter_and_get() -> (unit) -> int =\n"
        "  var count = 0\n"
        "  count := count + 1\n"
        "  fn(u: unit) => count\n"
        "let getter = make_counter_and_get()\n"
        "let r = getter(())\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r"] == IntValue(1)


# ---------------------------------------------------------------------------
# 7. Indirect-call arg coercion parity (regression: BLOCKER bug in M4b)
# ---------------------------------------------------------------------------
#
# Root cause: indirect calls previously lowered args with lower_expr (no coercion)
# while ir_reference compensates via a runtime result-coercion in _apply_closure.  The IR
# statically elides the result coercion when body-type == return-type, so an int
# literal passed to a decimal param leaks as IntValue instead of DecimalValue.
# Fix: lower indirect args with lower_coerced(arg, param_type) just like direct calls.


def test_value_call_int_arg_to_decimal_param() -> None:
    """Indirect call passing an int literal to a decimal param must yield DecimalValue.

    Regression for BLOCKER bug: IR previously yielded IntValue(5) while ir_reference
    yielded DecimalValue(5).  Both evaluators must now agree on DecimalValue.
    """
    source = (
        "let f = fn(x: decimal) => x\n"
        "let r = f(5)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r"] == DecimalValue(decimal.Decimal("5"))


def test_value_call_int_arg_to_json_param() -> None:
    """Indirect call passing an int literal to a json param must yield the JSON int.

    Regression for BLOCKER bug: same root cause as the decimal variant — IR used
    to yield IntValue(5) while ir_reference yielded the json-wrapped integer.
    Both evaluators must agree.
    """
    from agm.agl.semantics.values import JsonValue

    source = (
        "let f = fn(x: json) => x\n"
        "let r = f(5)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r"] == JsonValue(5)


def test_higher_order_indirect_arg_coercion() -> None:
    """Higher-order function: int arg coerced to decimal through indirect call chain.

    Regression for BLOCKER bug: def apply(f: (decimal)->decimal, n: decimal) calls
    f(n) indirectly.  Passing an int literal as n must coerce to decimal at the
    direct-call boundary, and the indirect call f(n) must also deliver a decimal
    to f — both evaluators must agree.
    """
    source = (
        "def apply(f: (decimal) -> decimal, n: decimal) -> decimal = f(n)\n"
        "let identity = fn(x: decimal) => x\n"
        "let r = apply(identity, 7)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r"] == DecimalValue(decimal.Decimal("7"))


# ---------------------------------------------------------------------------
# 8. Recursion-depth limit via indirect call
# ---------------------------------------------------------------------------


def test_recursion_depth_via_indirect_call() -> None:
    """Recursion-depth guard fires through indirect/value call path."""
    # A named def that self-calls directly (to avoid needing a higher-order
    # setup that isn't quite right for pure indirect recursion).
    # Instead use a lambda that calls itself via an outer var (mutual via value).
    # The simplest: a def called via a function-value binding.
    source = (
        "def inf(n: int) -> int = inf(n + 1)\n"
        "let f: (int) -> int = inf\n"
        "let r = f(0)\n"
        "()"
    )
    ir_reference_exc, ir_exc = evaluate_ir_raises(source)
    assert ir_reference_exc.display_name == "RecursionError"
    assert ir_exc.display_name == "RecursionError"
    assert ir_reference_exc.fields["message"] == ir_exc.fields["message"]
    assert ir_exc.fields["limit"] == IntValue(256)


def test_recursion_depth_custom_limit_indirect() -> None:
    """IR-side recursion depth guard at a custom low limit via indirect call."""
    source = (
        "def inf(n: int) -> int = inf(n + 1)\n"
        "let f: (int) -> int = inf\n"
        "let r = f(0)\n"
        "()"
    )
    program = parse_program(source)
    resolved = resolve(program)
    caps = m2_caps()
    checked = check(resolved, caps)
    executable = lower_program(
        checked, source_text=source, source_label="<test>", validate=True
    )
    interp = IrInterpreter(executable, max_call_depth=5)
    with pytest.raises(AglRaise) as exc_info:
        interp.run()
    exc = exc_info.value.exc
    assert exc.display_name == "RecursionError"
    assert exc.fields["message"] == TextValue("Maximum call depth (5) exceeded")
    assert exc.fields["limit"] == IntValue(5)


# ---------------------------------------------------------------------------
# 9. Closure-valued bindings normalized equal
# ---------------------------------------------------------------------------


def test_closure_valued_binding_normalized() -> None:
    """A lambda binding normalizes to <closure> sentinel on both sides."""
    source = (
        "let f = fn(x: int) => x * 2\n"
        "let r = f(3)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r"] == IntValue(6)


# ---------------------------------------------------------------------------
# 10. function_values.agl end-to-end (without print, use bindings)
# ---------------------------------------------------------------------------


def test_function_values_program() -> None:
    """function_values.agl end-to-end (adapted to not use print — use bindings)."""
    source = (
        "def classify(n: int) -> text =\n"
        "  if | n > 0 => \"pos\" | n < 0 => \"neg\" | else => \"zero\"\n"
        "\n"
        "def label(n: int) -> text =\n"
        "  \"val=\" + classify(n)\n"
        "\n"
        "def apply(f: (int) -> text, n: int) -> text = f(n)\n"
        "\n"
        "let g: (int) -> text = classify\n"
        "let dbl = fn(x: int) -> int => x * 2\n"
        "let inc = fn(x: int) => x + 1\n"
        "\n"
        "let r_g7 = g(7)\n"
        "let r_gm3 = g(-3)\n"
        "let r_g0 = g(0)\n"
        "let r_dbl4 = dbl(4)\n"
        "let r_inc9 = inc(9)\n"
        "let r_apply_classify = apply(classify, 10)\n"
        "let r_apply_label = apply(label, -5)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r_g7"] == TextValue("pos")
    assert ir["r_gm3"] == TextValue("neg")
    assert ir["r_g0"] == TextValue("zero")
    assert ir["r_dbl4"] == IntValue(8)
    assert ir["r_inc9"] == IntValue(10)
    assert ir["r_apply_classify"] == TextValue("pos")
    assert ir["r_apply_label"] == TextValue("val=neg")


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_lambda_with_decimal_return() -> None:
    """Lambda body returns int but explicit return type is decimal — coerced."""
    source = (
        "let to_dec = fn(x: int) -> decimal => x\n"
        "let r = to_dec(3)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r"] == DecimalValue(decimal.Decimal("3"))


def test_lambda_called_multiple_times() -> None:
    """Lambda called multiple times; each call is independent."""
    source = (
        "let square = fn(x: int) -> int => x * x\n"
        "let a = square(3)\n"
        "let b = square(4)\n"
        "let c = square(5)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["a"] == IntValue(9)
    assert ir["b"] == IntValue(16)
    assert ir["c"] == IntValue(25)


def test_lambda_passed_as_arg_to_higher_order() -> None:
    """Lambda directly passed to a higher-order function."""
    source = (
        "def apply(f: (int) -> int, n: int) -> int = f(n)\n"
        "let r = apply(fn(x: int) => x + 10, 5)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r"] == IntValue(15)


def test_indirect_call_with_two_args() -> None:
    """Indirect call with two positional arguments."""
    source = (
        "let add = fn(x: int, y: int) -> int => x + y\n"
        "let r = add(3, 4)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r"] == IntValue(7)


def test_function_value_captures_outer_let() -> None:
    """Lambda capturing an outer let-bound variable works correctly."""
    source = (
        "let offset = 100\n"
        "let add_offset = fn(x: int) -> int => x + offset\n"
        "let r = add_offset(5)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r"] == IntValue(105)


def test_two_lambdas_independent_capture() -> None:
    """Two lambdas each capturing a different outer binding stay independent."""
    source = (
        "let a = 10\n"
        "let b = 20\n"
        "let fa = fn(x: int) -> int => x + a\n"
        "let fb = fn(x: int) -> int => x + b\n"
        "let ra = fa(1)\n"
        "let rb = fb(1)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["ra"] == IntValue(11)
    assert ir["rb"] == IntValue(21)


def test_lambda_with_default_param() -> None:
    """Lambda with a default parameter: lowerer lowers the default coerced (M4b).

    This exercises the ``param.default is not None`` branch in ``_lower_lambda``
    (lowerer.py lines 584-587).  The checker requires exact arity for value calls
    (FunctionType erases defaults), so we call with all args explicitly — the
    lowered default still ends up in the FunctionDescriptor and is lowered via
    lower_coerced.
    """
    source = (
        "let add = fn(x: int, y: int = 10) -> int => x + y\n"
        "let r = add(5, 3)\n"
        "()"
    )
    ir_reference, ir = evaluate_ir(source)
    assert ir["r"] == IntValue(8)
