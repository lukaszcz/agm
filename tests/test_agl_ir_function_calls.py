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
from pathlib import Path

import pytest

from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.ir.nodes import UseDefault
from agm.agl.lower.program import lower_program
from agm.agl.matchcompile import MatchCompiledProgram, compile_program_matches
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import BoolValue, DecimalValue, IntValue, TextValue
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import (
    base_caps,
    evaluate_ir,
    lower_inline_ir,
    make_repl_graph_from_files,
    nominal_id_for,
    resolve_repl_graph,
)

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
        'def greet(greeting: text, name: text) -> text = greeting ++ " " ++ name\n'
        'let result = greet(name = "World", greeting = "Hello")\n()'
    )
    ir = evaluate_ir(source)
    assert ir["result"] == TextValue("Hello World")


def test_default_arg_used() -> None:
    """Call where the second arg uses its default."""
    source = (
        "def inc(x: int, step: int = 1) -> int = x + step\nlet a = inc(10)\nlet b = inc(10, 5)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["a"] == IntValue(11)
    assert ir["b"] == IntValue(15)


def test_default_and_named_supplied_args_evaluate_in_positional_order() -> None:
    """A single interleaved pass evaluates arguments in PARAMETER order.

    ``f``'s first parameter is defaulted and its second is supplied by name,
    so reaching ``f`` requires a named argument that leaves an earlier
    parameter defaulted. Both the default and the supplied expression have a
    visible side effect (appending to ``log``): the default for ``a`` must
    run before the caller-supplied expression for ``b``, matching parameter
    order rather than "every supplied argument, then every default."
    """
    source = (
        'var log = ""\n'
        "def note(s: text) -> int =\n"
        "  log := log ++ s\n"
        "  0\n"
        'def f(a: int = note("a"), b: int = 0) -> unit = ()\n'
        'f(b = note("b"))\n'
        "()"
    )
    ir = evaluate_ir(source)
    assert ir["log"] == TextValue("ab")


def test_return_coercion() -> None:
    """Function with int body but decimal return type — coercion applied."""
    source = "def to-dec(x: int) -> decimal = x\nlet result = to-dec(3)\n()"
    ir = evaluate_ir(source)
    assert ir["result"] == DecimalValue(decimal.Decimal("3"))


def test_explicit_return_coercion() -> None:
    """Explicit return operands are coerced to the declared result type."""
    source = "def to-dec(x: int) -> decimal =\n  return x\nlet result = to-dec(3)\n()"
    ir = evaluate_ir(source)
    assert ir["result"] == DecimalValue(decimal.Decimal("3"))


def test_arg_coercion() -> None:
    """int argument passed to decimal parameter — coercion at call site."""
    source = "def halve(x: decimal) -> decimal = x / 2.0\nlet result = halve(10)\n()"
    ir = evaluate_ir(source)
    assert ir["result"] == DecimalValue(decimal.Decimal("5"))


def test_generic_explicit_arg_coercion_uses_instantiated_param_type() -> None:
    """Generic direct calls coerce arguments against instantiated parameter types."""
    source = "def id[T](x: T) -> T = x\nlet result: decimal = id::[decimal](1)\n()"
    ir = evaluate_ir(source)
    assert ir["result"] == DecimalValue(decimal.Decimal("1"))


def test_generic_direct_method_call_coerces_against_selected_specialization() -> None:
    """A direct method call uses its concrete generic parameter type when lowered."""
    source = """\
record Box[T](value: T)

def Box::replace[T, U](self, value: U) -> U = value

let result: decimal = Box(value = 1).replace::[decimal](1)
()\n"""
    result = evaluate_ir(source)
    assert result["result"] == DecimalValue(decimal.Decimal("1"))


def test_provisional_generic_function_value_call_is_lowered_after_argument_inference() -> None:
    """A value call can solve a generic callee result from its own argument."""
    source = "def maker[T]() -> T -> T = fn(value: T) => value\nlet result = maker()(7)\n()"
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(7)


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
        "def is-even(n: int) -> bool =\n"
        "  if n == 0 => true else => is-odd(n - 1)\n"
        "def is-odd(n: int) -> bool =\n"
        "  if n == 0 => false else => is-even(n - 1)\n"
        "let r1 = is-even(4)\n"
        "let r2 = is-odd(3)\n()"
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
    executable = lower_inline_ir(source)
    interp = IrInterpreter(executable, max_call_depth=10)
    with pytest.raises(AglRaise) as exc_info:
        interp.run(program_symbol=executable.synthetic_main_symbol)
    exc = exc_info.value.exc
    assert exc.nominal == nominal_id_for(executable, "RecursionError")
    assert exc.fields["message"] == TextValue("Maximum call depth (10) exceeded")
    assert exc.fields["limit"] == IntValue(10)


def test_bound_method_can_be_called_later_and_passed_to_higher_order_function() -> None:
    source = """\
record Meter(value: int)

def Meter::add(self, amount: int) -> int = self.value + amount
def apply(value: int, f: (int) -> int) -> int = f(value)

let meter = Meter(value = 4)
let add = meter.add
let later = add(3)
let higher-order = apply(5, meter.add)
()
"""
    result = evaluate_ir(source)
    assert result["later"] == IntValue(7)
    assert result["higher-order"] == IntValue(9)


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
        "def sum-of-squares(a: int, b: int) -> int =\n"
        "  let sq-a = a * a\n"
        "  let sq-b = b * b\n"
        "  sq-a + sq-b\n"
        "let result = sum-of-squares(3, 4)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(25)


def test_function_reads_root_binding() -> None:
    """A root function reads an immutable root binding."""
    source = (
        "let offset: int = 10\n"
        "def add-offset(x: int) -> int = x + offset\n"
        "let result = add-offset(5)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(15)


def test_function_with_case_in_body() -> None:
    """Function body with case expression (exercises _walk_collect_locals for Case)."""
    source = (
        "enum Color | Red | Green | Blue\n"
        "def color-code(c: Color) -> int =\n"
        "  case c of\n"
        "    | Red() => 1\n"
        "    | Green() => 2\n"
        "    | Blue() => 3\n"
        "let r = color-code(Red())\n()"
    )
    ir = evaluate_ir(source)
    assert ir["r"] == IntValue(1)


def test_function_with_unary_in_body() -> None:
    """Function body with unary negation reads a root binding."""
    source = (
        "let scale: int = 2\n"
        "def neg-scaled(x: int) -> int = -(x * scale)\n"
        "let result = neg-scaled(3)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(-6)


def test_function_with_named_args_in_body_call() -> None:
    """Function body calling another function with named args (exercises named_args walk)."""
    source = (
        'def greet(name: text, greeting: text = "Hi") -> text =\n'
        '  greeting ++ ", " ++ name ++ "!"\n'
        'def greet-world(g: text) -> text = greet(name = "World", greeting = g)\n'
        'let result = greet-world("Hello")\n()'
    )
    ir = evaluate_ir(source)
    assert ir["result"] == TextValue("Hello, World!")


def test_function_with_array_in_body() -> None:
    """Function body containing an array literal reads a root binding."""
    source = (
        "let base: int = 1\n"
        "def make-array(x: int) -> array[int] = [base, x, x * 2]\n"
        "let result = make-array(3)\n()"
    )
    from agm.agl.semantics.values import ArrayValue

    ir = evaluate_ir(source)
    assert ir["result"] == ArrayValue([IntValue(1), IntValue(3), IntValue(6)])


def test_function_with_field_access_in_body() -> None:
    """Function body with field access (exercises _walk_for_captures for FieldAccess)."""
    source = (
        "record Point\n"
        "  x: int\n"
        "  y: int\n"
        "def get-x(p: Point) -> int = p.x\n"
        "let p = Point(x = 3, y = 4)\n"
        "let result = get-x(p)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(3)


def test_function_with_cast_in_body() -> None:
    """Function body with cast (exercises _walk_for_captures for Cast)."""
    source = (
        "def cast-to-decimal(x: int) -> decimal = x as decimal\nlet result = cast-to-decimal(7)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == DecimalValue(decimal.Decimal("7"))


def test_function_with_template_in_body() -> None:
    """Function body containing a template literal reads a root binding."""
    source = (
        'let prefix: text = "Item"\n'
        'def label(n: int) -> text = "%{prefix} #%{n}"\n'
        "let result = label(5)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == TextValue("Item #5")


def test_function_with_try_in_body() -> None:
    """Function body with try/catch (exercises _walk_collect_locals for Try)."""
    source = (
        "def safe-add(a: int, b: int) -> int =\n"
        "  try\n"
        "    a + b\n"
        "  catch ArithmeticError =>\n"
        "    0\n"
        "let result = safe-add(3, 4)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(7)


def test_function_with_do_loop_in_body() -> None:
    """Function body with do loop (exercises _walk_collect_locals for Loop)."""
    source = (
        "def count-to(n: int) -> int =\n"
        "  var i = 0\n"
        "  do\n"
        "    i := i + 1\n"
        "  until i >= n\n"
        "  i\n"
        "let result = count-to(5)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(5)


def test_function_with_do_done_loop_in_body() -> None:
    """Function body with do-done loop (exercises _scan_captures for until_cond=None).

    `do[0] done` runs zero iterations (bound=0 ≤ 0 exits immediately).
    """
    source = (
        "def run-nothing(n: int) -> int =\n"
        "  var i = 0\n"
        "  do[n]\n"
        "    i := i + 1\n"
        "  done\n"
        "  i\n"
        "let result = run-nothing(0)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(0)


def test_function_with_var_and_assignment_root_binding() -> None:
    """Function body uses a local var and an immutable root binding."""
    source = (
        "let factor: int = 3\n"
        "def triple-then-add(x: int, y: int) -> int =\n"
        "  var acc = factor * x\n"
        "  acc := acc + y\n"
        "  acc\n"
        "let result = triple-then-add(4, 5)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(17)


def test_function_with_raise_in_body() -> None:
    """Function body with raise (exercises _walk_for_captures for Raise)."""
    source = (
        "def checked-inc(n: int) -> int =\n"
        "  if n < 0 =>\n"
        '    raise Abort(message = "negative")\n'
        "  else =>\n"
        "    n + 1\n"
        "let result = checked-inc(5)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(6)


def test_function_with_index_access_and_root_binding() -> None:
    """Function body indexes an immutable root binding."""
    source = (
        "let items: array[int] = [10, 20, 30]\n"
        "def get-item(i: int) -> int = items[i]\n"
        "let result = get-item(1)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == IntValue(20)


def test_function_with_dict_literal_and_root_binding() -> None:
    """Function body with dict literal reads an immutable root binding."""
    source = (
        "let base: int = 10\n"
        'def make-dict(x: int) -> dict[text, int] = {"a": base + x, "b": x}\n'
        "let result = make-dict(5)\n()"
    )
    from agm.agl.semantics.values import DictValue

    ir = evaluate_ir(source)
    assert ir["result"] == DictValue({"a": IntValue(15), "b": IntValue(5)})


def test_function_with_is_test_and_root_binding() -> None:
    """Function body with is-test reads an immutable root binding."""
    source = (
        "enum Color | Red | Green | Blue\n"
        "let my-color: Color = Red()\n"
        "def check-red(c: Color) -> bool = c is Red or my-color is Red\n"
        "let result = check-red(Green())\n()"
    )
    ir = evaluate_ir(source)
    assert ir["result"] == BoolValue(True)


def test_function_reads_static_root_var() -> None:
    """A root function may read a static root mutable binding."""
    source = (
        "var counter = 0\n"
        "var result = 0\n"
        "def get-counter() -> int = counter\n"
        "program def main() -> unit =\n"
        "  result := get-counter()"
    )
    executable = lower_inline_ir(source)
    (main_symbol,) = executable.program_functions
    ir = IrInterpreter(executable).run(program_symbol=main_symbol)
    assert ir["result"] == IntValue(0)


def test_static_let_calls_function_assigning_prior_static_var() -> None:
    """A static let's initializer may call a function that assigns a preceding static var."""
    source = (
        "var counter = 0\n"
        "def increment() -> int =\n"
        "  counter := counter + 1\n"
        "  counter\n"
        "let result = increment()\n"
        "()"
    )

    ir = evaluate_ir(source)

    assert ir["counter"] == IntValue(1)
    assert ir["result"] == IntValue(1)


def test_static_let_reads_prior_destructured_static_let() -> None:
    """A static let's initializer may read binders from a preceding destructuring let."""
    source = (
        "record Pair(left: int, right: int)\n"
        "let Pair(left, right) = Pair(left = 20, right = 22)\n"
        "let result = left + right\n"
        "()"
    )

    ir = evaluate_ir(source)

    assert ir["left"] == IntValue(20)
    assert ir["right"] == IntValue(22)
    assert ir["result"] == IntValue(42)


# ---------------------------------------------------------------------------
# B1/capture fix tests (review-fixes task)
# ---------------------------------------------------------------------------


def test_index_target_static_root_access() -> None:
    """A root function can mutate a static root array through an index target."""
    from agm.agl.semantics.values import ArrayValue

    source = (
        "var arr = [0, 0, 0]\n"
        "let k = 1\n"
        "def setit() -> unit =\n"
        "  arr[k] := 99\n"
        "program def main() -> unit =\n"
        "  setit()"
    )
    executable = lower_inline_ir(source)
    (main_symbol,) = executable.program_functions
    ir = IrInterpreter(executable).run(program_symbol=main_symbol)
    assert ir["arr"] == ArrayValue([IntValue(0), IntValue(99), IntValue(0)])


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
        "var z: unit = ()\n"
        "def setit() -> unit =\n"
        "  arr[k] := 99\n"
        "program def main() -> unit =\n"
        "  z := setit()"
    )
    executable = lower_inline_ir(index_source)
    (main_symbol,) = executable.program_functions
    ir = IrInterpreter(executable).run(program_symbol=main_symbol)
    assert ir["z"] == UnitValue()

    # Name-target assignment as the function body / return value.
    name_source = (
        "var counter = 0\n"
        "var z: unit = ()\n"
        "def reset() -> unit =\n"
        "  counter := 5\n"
        "program def main() -> unit =\n"
        "  z := reset()"
    )
    executable2 = lower_inline_ir(name_source)
    (main_symbol2,) = executable2.program_functions
    ir2 = IrInterpreter(executable2).run(program_symbol=main_symbol2)
    assert ir2["z"] == UnitValue()


def test_name_target_only_static_root_assignment() -> None:
    """A root function can assign a static root mutable binding."""
    source = (
        "var counter = 0\n"
        "def reset() -> unit =\n"
        "  counter := 5\n"
        "program def main() -> unit =\n"
        "  reset()"
    )
    executable = lower_inline_ir(source)
    (main_symbol,) = executable.program_functions
    ir = IrInterpreter(executable).run(program_symbol=main_symbol)
    assert ir["counter"] == IntValue(5)


# ---------------------------------------------------------------------------
# Program entry: pre-evaluated arguments
# ---------------------------------------------------------------------------


def test_program_entry_runs_with_supplied_argument() -> None:
    """A ``program def`` parameter binds directly from a pre-evaluated argument."""
    source = "var result = 0\nprogram def main(x: int) -> unit =\n  result := x\n"
    executable = lower_inline_ir(source)
    (main_symbol,) = executable.program_functions
    ir = IrInterpreter(executable).run(program_symbol=main_symbol, arguments=(IntValue(7),))
    assert ir["result"] == IntValue(7)


def test_program_entry_use_default_reads_module_let(tmp_path: Path) -> None:
    """A ``UseDefault`` argument's default reads a binding a module INITIALIZER sets.

    ``base`` starts at a sentinel and is overwritten by a later top-level
    assignment — a genuine module initializer executed in sequence, unlike a
    static ``let`` value that could equally resolve at declaration time — so
    seeing the overwritten value pins that program arguments bind after
    every module initializer has run. A root assignment statement needs the
    REPL-rooted graph (``lower_inline_ir``'s inline-command entry does not
    permit one).
    """
    source = (
        "var base = 0\n"
        "base := 10\n"
        "var result = 0\n"
        "program def main(x: int = base) -> unit =\n"
        "  result := x\n"
    )
    graph = make_repl_graph_from_files(tmp_path, {"entry": source})
    checked = check_program(resolve_repl_graph(graph), base_caps())
    compiled = compile_program_matches(checked)
    assert isinstance(compiled.compiled, MatchCompiledProgram)
    executable = lower_program(compiled.compiled, _entry_source_text=source)
    (main_symbol,) = executable.program_functions
    ir = IrInterpreter(executable).run(
        program_symbol=main_symbol, arguments=(UseDefault(param_index=0),)
    )
    assert ir["result"] == IntValue(10)


def test_program_entry_raising_default_propagates_like_a_body_raise() -> None:
    """A raising program-parameter default surfaces with a span, exactly like a body raise."""
    default_source = (
        'def bad() -> int = raise Abort(message = "bad default")\n'
        "program def main(x: int = bad()) -> unit = ()\n"
    )
    body_source = 'program def main() -> unit =\n  raise Abort(message = "bad body")\n'

    default_executable = lower_inline_ir(default_source)
    (default_symbol,) = default_executable.program_functions
    with pytest.raises(AglRaise) as default_exc_info:
        IrInterpreter(default_executable).run(
            program_symbol=default_symbol, arguments=(UseDefault(param_index=0),)
        )

    body_executable = lower_inline_ir(body_source)
    (body_symbol,) = body_executable.program_functions
    with pytest.raises(AglRaise) as body_exc_info:
        IrInterpreter(body_executable).run(program_symbol=body_symbol, arguments=())

    assert default_exc_info.value.exc.nominal == nominal_id_for(default_executable, "Abort")
    assert default_exc_info.value.span is not None
    assert body_exc_info.value.span is not None
    assert type(default_exc_info.value.span) is type(body_exc_info.value.span)


def test_program_entry_with_no_parameters_takes_empty_arguments() -> None:
    """A parameterless program's entry call is unaffected by the ``arguments`` default."""
    source = "var result = 0\nprogram def main() -> unit =\n  result := 1\n"
    executable = lower_inline_ir(source)
    (main_symbol,) = executable.program_functions
    ir = IrInterpreter(executable).run(program_symbol=main_symbol, arguments=())
    assert ir["result"] == IntValue(1)


def test_program_entry_depth_limit_error_carries_entry_span() -> None:
    """The depth-limit error raised directly at the program entry carries an entry-point span."""
    source = "program def main() -> unit =\n  ()\n"
    executable = lower_inline_ir(source)
    (main_symbol,) = executable.program_functions
    with pytest.raises(AglRaise) as exc_info:
        IrInterpreter(executable, max_call_depth=0).run(program_symbol=main_symbol, arguments=())
    assert exc_info.value.exc.nominal == nominal_id_for(executable, "RecursionError")
    assert exc_info.value.span is not None


def test_root_binding_through_nested_positions_and_pattern_locals() -> None:
    """A root binding is visible through case arms; pattern binders stay local."""
    source = (
        "enum Shape | Circle(radius: int) | Square(side: int)\n"
        "let multiplier: int = 3\n"
        "def describe(s: Shape) -> int =\n"
        "  case s of\n"
        "    | Circle(radius = _ as r) => r * multiplier\n"
        "    | Square(side = _ as sd) => sd * multiplier\n"
        "let c = describe(Shape::Circle(radius = 4))\n"
        "let sq = describe(Shape::Square(side = 5))\n()"
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
    assert ir_exc.nominal == nominal_id_for(lower_inline_ir(source), "RecursionError")
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
    """_pattern_binding_ids excludes a bare constructor pattern.

    A bare name in a case pattern (for example, ``| On => ...``) is finalized
    by the checker as a constructor rather than a binder, so it is not added to
    ``local_ids``.
    """
    source = (
        "enum Flag | On | Off\n"
        "let flag = Flag::On()\n"
        "def check-module(f: Flag) -> int =\n"
        "  case f of\n"
        "    | On => 1\n"
        "    | Off => 0\n"
        "let r = check-module(flag)\n()"
    )
    ir = evaluate_ir(source)
    assert ir["r"] == IntValue(1)
