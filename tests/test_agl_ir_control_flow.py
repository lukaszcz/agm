"""IR evaluation tests for if/raise/try control flow.

Covers IrIf, IrRaise, IrTry with all cases:
- if without else → UnitValue
- if with else → taken branch value
- nested ifs
- raise propagation
- try with specific catch (with and without binding)
- try with catch-all
- first-match ordering in try handlers
- try body does not raise → body value
- no handler matches → re-raise
- defensive evaluator tests (hand-built IR)
- negative validate tests
- golden lowering tests

"""

from __future__ import annotations

import pytest

from agm.agl.ir.ids import Location, NominalId, SourceId, SymbolId
from agm.agl.ir.nodes import (
    IrBind,
    IrBlock,
    IrCatchHandler,
    IrConstBool,
    IrConstInt,
    IrConstText,
    IrConstUnit,
    IrExpr,
    IrIf,
    IrIfBranch,
    IrMakeException,
    IrRaise,
    IrReturn,
    IrTry,
)
from agm.agl.ir.program import (
    ExecutableModule,
    ExecutableProgram,
    NominalDescriptor,
    NominalKind,
    SourceFile,
    SymbolDescriptor,
)
from agm.agl.ir.validate import InvalidIrError, validate_ir
from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID
from agm.agl.semantics.values import (
    ExceptionValue,
    IntValue,
    TextValue,
    UnitValue,
)
from tests.agl.ir_harness import evaluate_ir, evaluate_ir_raises

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_SOURCE = "let x = 1\n"
_DUMMY_SOURCE_ID = SourceId(0)
_DUMMY_LOC = Location(
    source_id=_DUMMY_SOURCE_ID,
    start_offset=0,
    end_offset=1,
    start_line=1,
    start_col=0,
)


def _make_program(
    initializers: tuple[IrExpr, ...],
    *,
    source: str = _DUMMY_SOURCE,
    symbols: dict[SymbolId, SymbolDescriptor] | None = None,
    nominals: dict[NominalId, NominalDescriptor] | None = None,
) -> ExecutableProgram:
    """Build a minimal ExecutableProgram for hand-built IR tests."""
    src_id = SourceId(0)
    return ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={
            ENTRY_ID: ExecutableModule(
                module_id=ENTRY_ID,
                initializers=initializers,
            )
        },
        symbols=symbols or {},
        nominals=nominals or {},
        sources={src_id: SourceFile(display_name="<test>", normalized_text=source)},
    )


# ---------------------------------------------------------------------------
# IR evaluation tests — if without else
# ---------------------------------------------------------------------------


def test_if_without_else_true_runs_body_returns_unit() -> None:
    """if without else: taken branch still yields unit (effects run)."""
    source = "let u: unit = if true => ()\nu"
    ir = evaluate_ir(source)
    assert ir["u"] == UnitValue()


def test_if_without_else_false_returns_unit() -> None:
    """if without else: condition false, unit still returned (no side effects)."""
    source = """\
var r = 0
if false =>
  r := 99
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == IntValue(0)


def test_if_without_else_variable_effect() -> None:
    """if without else: body side-effects are applied when condition is true."""
    source = """\
var r = 0
if true =>
  r := 42
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == IntValue(42)


# ---------------------------------------------------------------------------
# IR evaluation tests — if with else (returns branch value)
# ---------------------------------------------------------------------------


def test_if_with_else_taken_true() -> None:
    """if-else: condition true → then branch value returned."""
    source = "let r = if true => 1 | else => 2\nr"
    ir = evaluate_ir(source)
    assert ir["r"] == IntValue(1)


def test_if_with_else_taken_false() -> None:
    """if-else: condition false → else branch value returned."""
    source = "let r = if false => 1 | else => 2\nr"
    ir = evaluate_ir(source)
    assert ir["r"] == IntValue(2)


def test_if_with_else_text_values() -> None:
    """if-else: returns text value from branches."""
    source = """\
let x = 5
let r = if x > 10 => "big" | else => "small"
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == TextValue("small")


def test_if_nested() -> None:
    """Nested if expressions."""
    source = """\
let x = 5
let r = if x > 0 => (if x > 3 => "big" | else => "small") | else => "neg"
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == TextValue("big")


# ---------------------------------------------------------------------------
# IR evaluation tests — return
# ---------------------------------------------------------------------------


def test_return_value_exits_function_early() -> None:
    source = """\
def choose(x: int) -> int =
  if x > 0 =>
    return x
  0
let r = choose(5)
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == IntValue(5)


def test_bare_return_yields_unit() -> None:
    source = """\
def stop(flag: bool) -> unit =
  if flag =>
    return
  print("after")
let r = stop(true)
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == UnitValue()


def test_return_unwinds_through_loop_and_try() -> None:
    source = """\
def find() -> int =
  var n = 0
  do[5]
    n := n + 1
    try
      if n == 3 =>
        return n
      ()
    catch Exception =>
      ()
  done
  0
let r = find()
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == IntValue(3)


def test_return_in_catch_handler_is_not_caught() -> None:
    source = """\
def recover() -> int =
  try
    raise Abort(message = "x")
  catch Abort =>
    return 7
  0
let r = recover()
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == IntValue(7)


# ---------------------------------------------------------------------------
# IR evaluation tests — raise
# ---------------------------------------------------------------------------


def test_raise_propagates() -> None:
    """raise propagates as AglRaise; both pipelines raise equivalent exceptions."""
    source = "raise Abort(message = \"oops\")\n"
    evaluate_ir_raises(source)


def test_raise_caught_by_try() -> None:
    """raise inside try is caught by a matching handler."""
    source = """\
let r = try
  raise Abort(message = "boom")
catch Abort =>
  99
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == IntValue(99)


# ---------------------------------------------------------------------------
# IR evaluation tests — try body does not raise
# ---------------------------------------------------------------------------


def test_try_no_raise_returns_body_value() -> None:
    """try body does not raise → returns body value."""
    source = """\
let r = try
  42
catch Exception =>
  0
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == IntValue(42)


def test_try_no_raise_expression_body() -> None:
    """try with non-raising expression body returns body value."""
    source = """\
let a = 3
let b = 4
let r = try
  a + b
catch Exception =>
  0
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == IntValue(7)


# ---------------------------------------------------------------------------
# IR evaluation tests — try with specific catch handler (no binding)
# ---------------------------------------------------------------------------


def test_try_specific_catch_no_binding() -> None:
    """try: specific exc_type match without binding variable."""
    source = """\
let r = try
  raise Abort(message = "bad")
catch Abort =>
  "caught Abort"
catch Exception =>
  "caught other"
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == TextValue("caught Abort")


# ---------------------------------------------------------------------------
# IR evaluation tests — try with specific catch handler WITH binding
# ---------------------------------------------------------------------------


def test_try_specific_catch_with_binding() -> None:
    """try: specific exc_type match WITH binding variable; bound value accessible."""
    source = """\
let r = try
  raise Abort(message = "the message")
catch Abort as e =>
  e.message
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == TextValue("the message")


def test_try_catch_with_binding_exception_value() -> None:
    """try: binding variable gives access to exception fields."""
    source = """\
let r = try
  raise Abort(message = "hello")
catch Abort as e =>
  e.message
r
"""
    ir = evaluate_ir(source)
    assert isinstance(ir["r"], TextValue)
    assert ir["r"].value == "hello"


# ---------------------------------------------------------------------------
# IR evaluation tests — catch-all handlers
# ---------------------------------------------------------------------------


def test_try_catchall_underscore() -> None:
    """try: catch-all with _ exc_type catches anything."""
    source = """\
let r = try
  raise Abort(message = "x")
catch _ =>
  "caught all"
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == TextValue("caught all")


def test_try_catchall_exception() -> None:
    """try: catch-all with 'Exception' catches anything."""
    source = """\
let r = try
  raise Abort(message = "x")
catch Exception =>
  "caught all exception"
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == TextValue("caught all exception")


def test_try_catchall_with_binding() -> None:
    """try: catch-all with binding variable."""
    source = """\
let r = try
  raise Abort(message = "catchall msg")
catch Exception as e =>
  e.message
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == TextValue("catchall msg")


# ---------------------------------------------------------------------------
# IR evaluation tests — first-match ordering
# ---------------------------------------------------------------------------


def test_try_first_match_wins() -> None:
    """try: first matching handler wins; later handlers not tried."""
    source = """\
let r = try
  raise Abort(message = "a")
catch Abort =>
  "first"
catch Exception =>
  "second"
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == TextValue("first")


def test_try_first_match_wins_catchall_last() -> None:
    """try: catch-all at end wins when specific handler doesn't match."""
    source = """\
let r = try
  raise Abort(message = "a")
catch CastError =>
  "cast"
catch _ =>
  "fallback"
r
"""
    ir = evaluate_ir(source)
    assert ir["r"] == TextValue("fallback")


# ---------------------------------------------------------------------------
# IR evaluation tests — no matching handler re-raises
# ---------------------------------------------------------------------------


def test_try_no_match_reraises() -> None:
    """try: when no handler matches, original exception re-propagates."""
    source = """\
try
  raise Abort(message = "unhandled")
catch CastError =>
  ()
"""
    evaluate_ir_raises(source)


# ---------------------------------------------------------------------------
# Golden lowering tests
# ---------------------------------------------------------------------------


def _lower(source: str) -> object:
    """Parse → check → lower; return ExecutableProgram with validate=True."""
    from agm.agl.capabilities import HostCapabilities
    from agm.agl.lower import lower_program
    from agm.agl.parser import parse_program
    from agm.agl.scope import resolve
    from agm.agl.typecheck import check

    caps = HostCapabilities(
        agent_names=frozenset(),
        has_default_agent=False,
        supports_shell_exec=False,
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset(
                {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
            ),
        },
    )
    prog = parse_program(source)
    resolved = resolve(prog)
    checked = check(resolved, caps)
    return lower_program(
        checked,
        source_text=source,
        source_label="<test>",
        validate=True,
    )


def test_lower_if_no_else_shape() -> None:
    """Golden lowering: if without else has has_else=False."""
    from agm.agl.ir.program import ExecutableProgram

    source = "if true => ()\n"
    prog = _lower(source)
    assert isinstance(prog, ExecutableProgram)
    entry = prog.modules[prog.entry_module]
    items = entry.initializers
    assert len(items) == 1
    ir_if = items[0]
    assert isinstance(ir_if, IrIf)
    assert not ir_if.has_else
    assert len(ir_if.branches) == 1
    branch = ir_if.branches[0]
    assert isinstance(branch, IrIfBranch)
    assert branch.cond is not None


def test_lower_if_with_else_shape() -> None:
    """Golden lowering: if with else has has_else=True and else branch has cond=None."""
    from agm.agl.ir.program import ExecutableProgram

    source = "let r = if true => 1 | else => 2\nr\n"
    prog = _lower(source)
    assert isinstance(prog, ExecutableProgram)
    entry = prog.modules[prog.entry_module]
    items = entry.initializers
    ir_bind = items[0]
    assert isinstance(ir_bind, IrBind)
    ir_if = ir_bind.value
    assert isinstance(ir_if, IrIf)
    assert ir_if.has_else
    assert len(ir_if.branches) == 2
    assert ir_if.branches[0].cond is not None
    assert ir_if.branches[1].cond is None


def test_lower_raise_shape() -> None:
    """Golden lowering: raise produces IrRaise with an IrMakeException for the exc."""
    from agm.agl.ir.program import ExecutableProgram

    source = "raise Abort(message = \"boom\")\n"
    prog = _lower(source)
    assert isinstance(prog, ExecutableProgram)
    entry = prog.modules[prog.entry_module]
    items = entry.initializers
    assert len(items) == 1
    ir_raise = items[0]
    assert isinstance(ir_raise, IrRaise)
    assert isinstance(ir_raise.exc, IrMakeException)
    assert ir_raise.exc.display_name == "Abort"


def test_lower_return_shape() -> None:
    """Golden lowering: return produces IrReturn with a value expression."""
    from agm.agl.ir.program import ExecutableProgram

    source = "def f() -> int =\n  return 1\n  0\nf()\n"
    prog = _lower(source)
    assert isinstance(prog, ExecutableProgram)
    desc = next(iter(prog.functions.values()))
    assert isinstance(desc.body, IrBlock)
    ir_return = desc.body.items[0]
    assert isinstance(ir_return, IrReturn)
    assert isinstance(ir_return.value, IrConstInt)


def test_lower_try_no_binding_shape() -> None:
    """Golden lowering: specific catch (no binding) → IrCatchHandler with symbol=None."""
    from agm.agl.ir.program import ExecutableProgram

    source = "let r = try\n  1\ncatch Abort =>\n  2\nr\n"
    prog = _lower(source)
    assert isinstance(prog, ExecutableProgram)
    entry = prog.modules[prog.entry_module]
    items = entry.initializers
    ir_bind = items[0]
    assert isinstance(ir_bind, IrBind)
    ir_try = ir_bind.value
    assert isinstance(ir_try, IrTry)
    assert len(ir_try.handlers) == 1
    handler = ir_try.handlers[0]
    assert isinstance(handler, IrCatchHandler)
    assert handler.nominal is not None
    assert handler.display_name == "Abort"
    assert handler.symbol is None


def test_lower_try_with_binding_shape() -> None:
    """Golden lowering: catch with binding allocates a SymbolId in program.symbols."""
    from agm.agl.ir.program import ExecutableProgram

    source = "let r = try\n  1\ncatch Abort as e =>\n  2\nr\n"
    prog = _lower(source)
    assert isinstance(prog, ExecutableProgram)
    entry = prog.modules[prog.entry_module]
    items = entry.initializers
    ir_bind = items[0]
    assert isinstance(ir_bind, IrBind)
    ir_try = ir_bind.value
    assert isinstance(ir_try, IrTry)
    handler = ir_try.handlers[0]
    assert isinstance(handler, IrCatchHandler)
    assert handler.nominal is not None
    assert handler.display_name == "Abort"
    assert handler.symbol is not None
    assert handler.symbol in prog.symbols


def test_lower_try_catchall_shape() -> None:
    """Golden lowering: catch-all (_, Exception) → nominal=None, display_name=None."""
    from agm.agl.ir.program import ExecutableProgram

    source = "let r = try\n  1\ncatch _ =>\n  2\nr\n"
    prog = _lower(source)
    assert isinstance(prog, ExecutableProgram)
    entry = prog.modules[prog.entry_module]
    items = entry.initializers
    ir_bind = items[0]
    assert isinstance(ir_bind, IrBind)
    ir_try = ir_bind.value
    assert isinstance(ir_try, IrTry)
    handler = ir_try.handlers[0]
    assert isinstance(handler, IrCatchHandler)
    assert handler.nominal is None
    assert handler.display_name is None
    assert handler.symbol is None


# ---------------------------------------------------------------------------
# Defensive evaluator tests (hand-built IR)
# ---------------------------------------------------------------------------


def test_ir_if_non_bool_cond_raises_invalid() -> None:
    """Defensive: IrIf with non-bool cond raises InvalidIrError."""
    from agm.agl.eval.ir_interpreter import IrInterpreter

    loc = _DUMMY_LOC
    ir_if = IrIf(
        location=loc,
        branches=(IrIfBranch(cond=IrConstInt(location=loc, value=1), body=IrConstUnit(loc)),),
        has_else=False,
    )
    prog = _make_program((ir_if,))
    interp = IrInterpreter(prog)
    with pytest.raises(InvalidIrError, match="BoolValue"):
        interp.run()


def test_ir_raise_non_exc_raises_invalid() -> None:
    """Defensive: IrRaise with non-ExceptionValue raises InvalidIrError."""
    from agm.agl.eval.ir_interpreter import IrInterpreter

    loc = _DUMMY_LOC
    ir_raise = IrRaise(
        location=loc,
        exc=IrConstInt(location=loc, value=42),
    )
    prog = _make_program((ir_raise,))
    interp = IrInterpreter(prog)
    with pytest.raises(InvalidIrError, match="ExceptionValue"):
        interp.run()


def test_ir_try_handler_binding_stored_in_frame() -> None:
    """Hand-built IrTry: handler with binding writes ExceptionValue to frame and body runs."""
    from agm.agl.eval.ir_interpreter import IrInterpreter

    loc = _DUMMY_LOC
    exc_sym = SymbolId(0)
    result_sym = SymbolId(1)
    exc_nominal = NominalId(PRELUDE_ID, "Abort")

    exc_node = IrMakeException(
        location=loc,
        nominal=exc_nominal,
        display_name="Abort",
        fields=(("message", IrConstText(location=loc, value="test")),),
    )
    raise_node = IrRaise(location=loc, exc=exc_node)
    handler = IrCatchHandler(
        nominal=exc_nominal,
        display_name="Abort",
        symbol=exc_sym,
        body=IrConstInt(location=loc, value=99),
    )
    ir_try = IrTry(location=loc, body=raise_node, handlers=(handler,))
    result_bind = IrBind(location=loc, symbol=result_sym, value=ir_try)

    symbols = {
        exc_sym: SymbolDescriptor(
            symbol_id=exc_sym, mutable=False, public_name=None, owner=ENTRY_ID
        ),
        result_sym: SymbolDescriptor(
            symbol_id=result_sym, mutable=False, public_name="result", owner=ENTRY_ID
        ),
    }
    nominals = {
        exc_nominal: NominalDescriptor(
            nominal=exc_nominal,
            display_name="Abort",
            kind=NominalKind.EXCEPTION,
            fields=("message", "trace_id"),
            variants=(),
        ),
    }
    prog = _make_program((result_bind,), symbols=symbols, nominals=nominals)
    interp = IrInterpreter(prog)
    results = interp.run()
    assert results["result"] == IntValue(99)
    # verify exception was bound in frame
    assert exc_sym in interp._frame
    bound = interp._frame[exc_sym]
    assert isinstance(bound, ExceptionValue)
    assert bound.display_name == "Abort"


# ---------------------------------------------------------------------------
# Negative validate tests
# ---------------------------------------------------------------------------


def test_validate_ir_try_handler_nominal_missing() -> None:
    """Negative validate: IrTry handler with nominal missing from program.nominals."""
    loc = _DUMMY_LOC
    missing_nominal = NominalId(PRELUDE_ID, "NonExistentError")
    handler = IrCatchHandler(
        nominal=missing_nominal,
        display_name="NonExistentError",
        symbol=None,
        body=IrConstUnit(loc),
    )
    ir_try = IrTry(
        location=loc,
        body=IrConstInt(loc, 1),
        handlers=(handler,),
    )
    prog = _make_program((ir_try,))
    with pytest.raises(InvalidIrError, match="NonExistentError"):
        validate_ir(prog, deep=True)


def test_validate_ir_try_handler_symbol_missing() -> None:
    """Negative validate: IrTry handler with symbol not in program.symbols."""
    loc = _DUMMY_LOC
    exc_nominal = NominalId(PRELUDE_ID, "Abort")
    orphan_sym = SymbolId(999)
    handler = IrCatchHandler(
        nominal=exc_nominal,
        display_name="Abort",
        symbol=orphan_sym,
        body=IrConstUnit(loc),
    )
    ir_try = IrTry(
        location=loc,
        body=IrConstInt(loc, 1),
        handlers=(handler,),
    )
    nominals = {
        exc_nominal: NominalDescriptor(
            nominal=exc_nominal,
            display_name="Abort",
            kind=NominalKind.EXCEPTION,
            fields=("message", "trace_id"),
            variants=(),
        ),
    }
    prog = _make_program((ir_try,), nominals=nominals)
    with pytest.raises(InvalidIrError, match="symbol"):
        validate_ir(prog, deep=True)


def test_validate_ir_if_cheap_ok() -> None:
    """Validate IrIf without deep: location checks only, should not raise."""
    loc = _DUMMY_LOC
    ir_if = IrIf(
        location=loc,
        branches=(IrIfBranch(cond=IrConstBool(loc, True), body=IrConstUnit(loc)),),
        has_else=False,
    )
    prog = _make_program((ir_if,))
    validate_ir(prog, deep=False)


def test_validate_ir_return_cheap_ok() -> None:
    """Validate IrReturn without deep: location and value are accepted."""
    loc = _DUMMY_LOC
    ir_return = IrReturn(location=loc, value=IrConstUnit(loc))
    prog = _make_program((ir_return,))
    validate_ir(prog, deep=False)


def test_validate_ir_try_cheap_ok() -> None:
    """Validate IrTry without deep: location + body only, no nominal/symbol cross-ref."""
    loc = _DUMMY_LOC
    exc_nominal = NominalId(PRELUDE_ID, "Abort")
    handler = IrCatchHandler(
        nominal=exc_nominal,
        display_name="Abort",
        symbol=None,
        body=IrConstUnit(loc),
    )
    ir_try = IrTry(location=loc, body=IrConstInt(loc, 1), handlers=(handler,))
    # No nominals registered — deep=False so nominal cross-ref check is skipped.
    prog = _make_program((ir_try,))
    validate_ir(prog, deep=False)


# ---------------------------------------------------------------------------
# Binder-leak regression tests — nested binders must not leak into results
# ---------------------------------------------------------------------------


def test_let_inside_if_branch_does_not_leak() -> None:
    """let declared inside an if branch body must not appear in top-level results.

    Regression for the binder-leak bug where nested let/var binders were allocated
    with public=True and leaked into _collect_results.
    """
    source = """\
var sink = 0
if true =>
  let inner = 7
  sink := inner
sink
"""
    ir = evaluate_ir(source)
    assert "inner" not in ir
    assert ir["sink"] == IntValue(7)


def test_var_inside_if_branch_does_not_leak() -> None:
    """var declared inside an if branch body must not appear in top-level results."""
    source = """\
var result = 0
if true =>
  var tmp = 5
  result := tmp + 1
result
"""
    ir = evaluate_ir(source)
    assert "tmp" not in ir
    assert ir["result"] == IntValue(6)


def test_let_inside_try_body_does_not_leak() -> None:
    """let declared inside a try body must not appear in top-level results."""
    source = """\
var out = 0
try
  let inner = 42
  out := inner
catch Exception =>
  ()
out
"""
    ir = evaluate_ir(source)
    assert "inner" not in ir
    assert ir["out"] == IntValue(42)


def test_let_inside_catch_handler_does_not_leak() -> None:
    """let declared inside a catch handler body must not appear in top-level results."""
    source = """\
var out = 0
try
  raise Abort(message = "boom")
catch Abort =>
  let handler_val = 99
  out := handler_val
out
"""
    ir = evaluate_ir(source)
    assert "handler_val" not in ir
    assert ir["out"] == IntValue(99)


def test_nested_block_initializer_only_outer_let_visible() -> None:
    """Block-scoped let inside if branch: outer let gets value; inner binder not in results.

    let x = (if true => body) where body contains a nested let y.
    Only x should appear in results, not y; x should have the correct value.
    """
    source = """\
let x = if true =>
  let y = 5
  y + 1
| else => 0
x
"""
    ir = evaluate_ir(source)
    assert "y" not in ir
    assert ir["x"] == IntValue(6)
