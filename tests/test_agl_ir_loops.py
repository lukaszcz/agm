"""IR evaluation tests for loops (IrLoop / IrBreak / IrContinue / for / while).

Covers:
- Source-level loop programs through the full pipeline (parse → lower → eval).
- IR-level unit tests for IrBreak and IrContinue primitives built directly.
- Control-signal bypass: IrBreak/IrContinue propagate through IrTry bodies.
- validate_ir: IrLoop, IrBreak, IrContinue, IrIterInit/HasNext/Next structural checks.
- for-loop iteration over list, dict, and text collections.
- while-clause guard with and without for-clause.
- Type error for non-iterable for-clause collection.
"""

from __future__ import annotations

import pytest

from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.ir.ids import Location, NominalId, SourceId, SymbolId
from agm.agl.ir.nodes import (
    IrArith,
    IrAssign,
    IrBind,
    IrBlock,
    IrBreak,
    IrCatchHandler,
    IrCompare,
    IrConstBool,
    IrConstInt,
    IrConstText,
    IrConstUnit,
    IrContinue,
    IrIf,
    IrIfBranch,
    IrIterHasNext,
    IrIterInit,
    IrIterNext,
    IrLoad,
    IrLoop,
    IrTry,
)
from agm.agl.ir.operations import ArithKind, ArithOp, CmpOp, CompareKind, IterKind
from agm.agl.ir.program import (
    ExecutableModule,
    ExecutableProgram,
    NominalDescriptor,
    NominalKind,
    SourceFile,
    SymbolDescriptor,
)
from agm.agl.ir.validate import validate_ir
from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import (
    VOID_VALUE,
    BoolValue,
    IntValue,
    JsonValue,
    TextValue,
    UnitValue,
)
from tests.agl.ir_harness import evaluate_ir, evaluate_ir_raises

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SRC_ID = SourceId(0)
_DUMMY_LOC = Location(
    source_id=_SRC_ID,
    start_offset=0,
    end_offset=0,
    start_line=1,
    start_col=0,
)


def _lower(source: str) -> ExecutableProgram:
    """Parse → resolve → check → lower *source*; return the ExecutableProgram."""
    from agm.agl.lower import lower_program
    from agm.agl.parser import parse_program
    from agm.agl.scope import resolve
    from agm.agl.typecheck import check
    from tests.agl.ir_harness import m2_caps

    checked = check(resolve(parse_program(source)), m2_caps())
    return lower_program(
        checked, source_text=source, source_label="<test>", validate=True
    )


def _make_minimal_program(
    initializers: tuple,
    *,
    source_text: str = "",
    symbols: dict | None = None,
) -> ExecutableProgram:
    """Build a minimal ExecutableProgram for hand-crafted IR tests."""
    from agm.agl.semantics.types import BUILTIN_EXCEPTIONS

    max_iter_nominal = NominalId(PRELUDE_ID, "MaxIterationsExceeded")
    exc_type = BUILTIN_EXCEPTIONS["MaxIterationsExceeded"]
    nominals = {
        max_iter_nominal: NominalDescriptor(
            nominal=max_iter_nominal,
            display_name="MaxIterationsExceeded",
            kind=NominalKind.EXCEPTION,
            fields=tuple(exc_type.fields.keys()),
            variants=(),
        )
    }
    return ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={
            ENTRY_ID: ExecutableModule(
                module_id=ENTRY_ID,
                initializers=tuple(initializers),
            )
        },
        symbols=symbols or {},
        nominals=nominals,
        sources={_SRC_ID: SourceFile(display_name="<test>", normalized_text=source_text)},
    )


def _make_counter_program(body_items: tuple) -> tuple[ExecutableProgram, SymbolId]:
    """Build a program with a mutable ``count`` symbol and a loop body.

    ``body_items`` are the IR nodes forming the loop body (inside IrBlock).
    The program has:
      - IrBind(count_sym, 0)  — mutable, public_name="count"
      - IrLoop(body=IrBlock(body_items))
    """
    count_sym = SymbolId(0)
    symbols = {
        count_sym: SymbolDescriptor(
            symbol_id=count_sym,
            mutable=True,
            public_name="count",
            owner=ENTRY_ID,
        )
    }
    loop = IrLoop(
        location=_DUMMY_LOC,
        body=IrBlock(location=_DUMMY_LOC, items=body_items),
    )
    prog = _make_minimal_program(
        (
            IrBind(
                location=_DUMMY_LOC,
                symbol=count_sym,
                value=IrConstInt(location=_DUMMY_LOC, value=0),
            ),
            loop,
        ),
        symbols=symbols,
    )
    return prog, count_sym


# ---------------------------------------------------------------------------
# Source-level pipeline tests: loop terminates via until (behavior unchanged)
# ---------------------------------------------------------------------------


def test_loop_terminates_explicit_limit() -> None:
    """do[10] body until cond — terminates after body mutates var; yields unit."""
    source = (
        "var counter = 0\n"
        "do[10]\n"
        "  counter := counter + 1\n"
        "until counter >= 3\n"
        "counter\n"
    )
    ir = evaluate_ir(source)
    assert ir["counter"] == IntValue(3)


def test_loop_body_bindings_visible_to_condition() -> None:
    """Condition reads a var declared before the loop; body updates it."""
    source = (
        "var x = 0\n"
        "do[5]\n"
        "  x := x + 2\n"
        "until x >= 4\n"
        "x\n"
    )
    ir = evaluate_ir(source)
    assert ir["x"] == IntValue(4)


def test_loop_no_explicit_limit_terminates() -> None:
    """do body until cond (no explicit limit) — unbounded, terminates via until."""
    source = (
        "var n = 0\n"
        "do\n"
        "  n := n + 1\n"
        "until n >= 5\n"
        "n\n"
    )
    ir = evaluate_ir(source)
    assert ir["n"] == IntValue(5)


def test_loop_exhaustion_raises() -> None:
    """do[3] body until false raises MaxIterationsExceeded; all fields match."""
    source = "var dummy = 0\ndo[3]\n  dummy := 1\nuntil false\n"
    ir_exc = evaluate_ir_raises(source)

    assert ir_exc.display_name == "MaxIterationsExceeded"

    cond_field = ir_exc.fields.get("condition")
    assert isinstance(cond_field, TextValue), f"condition field: {cond_field!r}"
    assert cond_field.value == "false", f"condition source text mismatch: {cond_field.value!r}"

    assert ir_exc.fields.get("limit") == IntValue(3)
    assert ir_exc.fields.get("last_condition_value") == BoolValue(False)
    assert ir_exc.fields.get("metadata") == JsonValue(None)


def test_condition_source_slice_complex() -> None:
    """Condition source text captures the exact condition expression text."""
    source = (
        "var i = 0\n"
        "do[2]\n"
        "  i := i + 1\n"
        "until i > 10\n"
    )
    ir_exc = evaluate_ir_raises(source)

    cond_field = ir_exc.fields.get("condition")
    assert isinstance(cond_field, TextValue)
    assert cond_field.value == "i > 10", f"got: {cond_field.value!r}"


def test_loop_succeeds_at_exact_limit() -> None:
    """do[3] until counter>=3 succeeds at exactly the limit (no MaxIterationsExceeded)."""
    source = (
        "var counter = 0\n"
        "do[3]\n"
        "  counter := counter + 1\n"
        "until counter >= 3\n"
        "counter\n"
    )
    ir = evaluate_ir(source)
    assert ir["counter"] == IntValue(3)


def test_loop_exhausts_one_short_of_condition() -> None:
    """do[2] until counter>=3 needs 3 iterations but limit is 2 → MaxIterationsExceeded."""
    exc = evaluate_ir_raises(
        "var counter = 0\n"
        "do[2]\n"
        "  counter := counter + 1\n"
        "until counter >= 3\n"
        "counter\n"
    )
    assert exc.display_name == "MaxIterationsExceeded"
    assert exc.fields.get("limit") == IntValue(2)


def test_ir_semantic_unbounded_loop_runs_to_completion() -> None:
    """A bound-less do loop is unbounded: iterates without raising MaxIterationsExceeded."""
    source = "var x = 0\ndo\n  x := x + 1\nuntil x >= 1000\nx\n"
    ir = evaluate_ir(source)
    assert ir["x"] == IntValue(1000)


def test_literal_zero_bound_runs_zero_iterations() -> None:
    """A literal do[0] runs the body zero times and completes normally (D2).

    Regression: the parser used to reject literal do[0] as a syntax error;
    it must parse and behave identically to a computed bound of 0.
    """
    source = "var r = 0\ndo[0]\n  r := r + 1\nuntil r >= 1\nr\n"
    ir = evaluate_ir(source)
    assert ir["r"] == IntValue(0)


def test_literal_negative_bound_runs_zero_iterations() -> None:
    """A literal do[-1] also runs zero iterations (consistent with do[0])."""
    source = "var r = 0\ndo[-1]\n  r := r + 1\nuntil r >= 1\nr\n"
    ir = evaluate_ir(source)
    assert ir["r"] == IntValue(0)


def test_valve_does_not_cap_for_over_finite_collection() -> None:
    """The max-iters valve must not cap a for loop over a finite collection.

    Regression: the valve applied to ALL loops, so --max-iters 3 broke
    `for x in [1,2,3,4]`.  Self-bounded loops (for/do[n]) are guarded and
    exempt from the host safety valve.
    """
    source = "var s = 0\nfor x in [1, 2, 3, 4, 5] do s := s + x done\ns\n"
    interp = IrInterpreter(_lower(source), loop_limit=3)
    result = interp.run()
    assert result["s"] == IntValue(15)


def test_valve_does_not_cap_bounded_do_n_loop() -> None:
    """The max-iters valve must not cap a do[n] loop whose own bound exceeds it."""
    source = "var i = 0\ndo[10]\n  i := i + 1\nuntil i >= 5\ni\n"
    interp = IrInterpreter(_lower(source), loop_limit=3)
    result = interp.run()
    assert result["i"] == IntValue(5)


def test_valve_caps_unbounded_do_until_loop() -> None:
    """The max-iters valve caps an unguarded (no [n], no for) do...until loop."""
    source = "var i = 0\ndo\n  i := i + 1\nuntil i >= 1000\ni\n"
    interp = IrInterpreter(_lower(source), loop_limit=3)
    try:
        interp.run()
    except AglRaise as exc:
        assert exc.exc.display_name == "MaxIterationsExceeded"
        assert exc.exc.fields.get("limit") == IntValue(3)
        return
    raise AssertionError("expected MaxIterationsExceeded")


def test_crlf_loop_exhaustion_condition_field() -> None:
    """With CRLF source, the MaxIterationsExceeded condition field is the clean source slice."""
    source = "var i = 0\r\ndo[3]\r\n  i := i + 1\r\nuntil i > 100\r\n"
    ir_exc = evaluate_ir_raises(source)
    assert ir_exc.display_name == "MaxIterationsExceeded"
    assert ir_exc.fields.get("condition") == TextValue("i > 100")


# ---------------------------------------------------------------------------
# IR-level unit tests: IrBreak exits the loop, yielding UnitValue
# ---------------------------------------------------------------------------


def test_irloop_break_exits_immediately() -> None:
    """IrLoop(body=IrBreak) exits the loop immediately and yields unit."""
    prog = _make_minimal_program(
        (IrLoop(location=_DUMMY_LOC, body=IrBreak(location=_DUMMY_LOC)),),
        source_text="",
    )
    interp = IrInterpreter(prog)
    result = interp.run()
    # The loop yields unit; no bindings → empty result
    assert result == {}
    assert interp.initializer_values == [VOID_VALUE]
    assert isinstance(interp.initializer_values[0], UnitValue)
    assert not interp.initializer_values[0].printable_in_repl


def test_irloop_break_exits_after_several_iterations() -> None:
    """IrLoop with IrBreak in a conditional: exits when count reaches target."""
    # Build: var count = 0; loop { if count >= 3 => break; count := count + 1 }
    count_sym = SymbolId(0)
    prog, _ = _make_counter_program(
        body_items=(
            IrIf(
                location=_DUMMY_LOC,
                branches=(
                    IrIfBranch(
                        cond=IrCompare(
                            location=_DUMMY_LOC,
                            op=CmpOp.GE,
                            kind=CompareKind.INT,
                            lhs=IrLoad(location=_DUMMY_LOC, symbol=count_sym),
                            rhs=IrConstInt(location=_DUMMY_LOC, value=3),
                        ),
                        body=IrBreak(location=_DUMMY_LOC),
                    ),
                ),
                has_else=False,
            ),
            IrAssign(
                location=_DUMMY_LOC,
                symbol=count_sym,
                path=(),
                value=IrArith(
                    location=_DUMMY_LOC,
                    op=ArithOp.ADD,
                    kind=ArithKind.INT,
                    lhs=IrLoad(location=_DUMMY_LOC, symbol=count_sym),
                    rhs=IrConstInt(location=_DUMMY_LOC, value=1),
                ),
            ),
        )
    )
    interp = IrInterpreter(prog)
    result = interp.run()
    assert result["count"] == IntValue(3)


# ---------------------------------------------------------------------------
# IR-level unit tests: IrContinue re-runs the loop body
# ---------------------------------------------------------------------------


def test_irloop_continue_reruns_body() -> None:
    """IrContinue at end of body re-iterates; IrBreak eventually exits."""
    # Build:
    #   var count = 0
    #   loop {
    #     if count >= 5 => break
    #     count := count + 1
    #     continue    ← explicit; semantically redundant but exercises IrContinue
    #   }
    # Expected: count = 5
    count_sym = SymbolId(0)
    prog, _ = _make_counter_program(
        body_items=(
            IrIf(
                location=_DUMMY_LOC,
                branches=(
                    IrIfBranch(
                        cond=IrCompare(
                            location=_DUMMY_LOC,
                            op=CmpOp.GE,
                            kind=CompareKind.INT,
                            lhs=IrLoad(location=_DUMMY_LOC, symbol=count_sym),
                            rhs=IrConstInt(location=_DUMMY_LOC, value=5),
                        ),
                        body=IrBreak(location=_DUMMY_LOC),
                    ),
                ),
                has_else=False,
            ),
            IrAssign(
                location=_DUMMY_LOC,
                symbol=count_sym,
                path=(),
                value=IrArith(
                    location=_DUMMY_LOC,
                    op=ArithOp.ADD,
                    kind=ArithKind.INT,
                    lhs=IrLoad(location=_DUMMY_LOC, symbol=count_sym),
                    rhs=IrConstInt(location=_DUMMY_LOC, value=1),
                ),
            ),
            IrContinue(location=_DUMMY_LOC),
        )
    )
    interp = IrInterpreter(prog)
    result = interp.run()
    assert result["count"] == IntValue(5)


# ---------------------------------------------------------------------------
# Control-signal bypass: IrBreak/IrContinue propagate through IrTry bodies
# ---------------------------------------------------------------------------


def test_break_bypasses_irtry() -> None:
    """IrBreak inside an IrTry body exits the enclosing IrLoop, not caught by handlers.

    If IrBreak were caught by the IrTry handler, the catch body would increment
    count by 100 and the loop would not exit.  Correct behavior: IrBreak
    propagates through IrTry to the enclosing IrLoop, count stays at 3.
    """
    count_sym = SymbolId(0)
    # Loop body:
    #   if count >= 3 =>
    #     try { break } catch _ => count := count + 100
    #   count := count + 1
    break_in_try = IrTry(
        location=_DUMMY_LOC,
        body=IrBreak(location=_DUMMY_LOC),
        handlers=(
            IrCatchHandler(
                nominal=None,
                display_name=None,
                symbol=None,
                body=IrAssign(
                    location=_DUMMY_LOC,
                    symbol=count_sym,
                    path=(),
                    value=IrArith(
                        location=_DUMMY_LOC,
                        op=ArithOp.ADD,
                        kind=ArithKind.INT,
                        lhs=IrLoad(location=_DUMMY_LOC, symbol=count_sym),
                        rhs=IrConstInt(location=_DUMMY_LOC, value=100),
                    ),
                ),
            ),
        ),
    )
    prog, _ = _make_counter_program(
        body_items=(
            IrIf(
                location=_DUMMY_LOC,
                branches=(
                    IrIfBranch(
                        cond=IrCompare(
                            location=_DUMMY_LOC,
                            op=CmpOp.GE,
                            kind=CompareKind.INT,
                            lhs=IrLoad(location=_DUMMY_LOC, symbol=count_sym),
                            rhs=IrConstInt(location=_DUMMY_LOC, value=3),
                        ),
                        body=break_in_try,
                    ),
                ),
                has_else=False,
            ),
            IrAssign(
                location=_DUMMY_LOC,
                symbol=count_sym,
                path=(),
                value=IrArith(
                    location=_DUMMY_LOC,
                    op=ArithOp.ADD,
                    kind=ArithKind.INT,
                    lhs=IrLoad(location=_DUMMY_LOC, symbol=count_sym),
                    rhs=IrConstInt(location=_DUMMY_LOC, value=1),
                ),
            ),
        )
    )
    interp = IrInterpreter(prog)
    result = interp.run()
    # IrBreak propagated through try; count is 3 (not 3+100).
    assert result["count"] == IntValue(3)


def test_continue_bypasses_irtry() -> None:
    """IrContinue inside an IrTry body re-iterates the enclosing IrLoop.

    If IrContinue were caught by IrTry, the handler would increment count by
    100 and the loop would never terminate properly.  Correct behavior:
    IrContinue propagates through IrTry, so only IrBreak (at count >= 3) exits.
    """
    count_sym = SymbolId(0)
    # Loop body:
    #   if count >= 3 => break
    #   try { continue } catch _ => count := count + 100
    #   count := count + 1   ← never reached (continue skips this)
    continue_in_try = IrTry(
        location=_DUMMY_LOC,
        body=IrContinue(location=_DUMMY_LOC),
        handlers=(
            IrCatchHandler(
                nominal=None,
                display_name=None,
                symbol=None,
                body=IrAssign(
                    location=_DUMMY_LOC,
                    symbol=count_sym,
                    path=(),
                    value=IrArith(
                        location=_DUMMY_LOC,
                        op=ArithOp.ADD,
                        kind=ArithKind.INT,
                        lhs=IrLoad(location=_DUMMY_LOC, symbol=count_sym),
                        rhs=IrConstInt(location=_DUMMY_LOC, value=100),
                    ),
                ),
            ),
        ),
    )
    symbols = {
        count_sym: SymbolDescriptor(
            symbol_id=count_sym,
            mutable=True,
            public_name="count",
            owner=ENTRY_ID,
        )
    }
    loop = IrLoop(
        location=_DUMMY_LOC,
        body=IrBlock(
            location=_DUMMY_LOC,
            items=(
                IrIf(
                    location=_DUMMY_LOC,
                    branches=(
                        IrIfBranch(
                            cond=IrCompare(
                                location=_DUMMY_LOC,
                                op=CmpOp.GE,
                                kind=CompareKind.INT,
                                lhs=IrLoad(location=_DUMMY_LOC, symbol=count_sym),
                                rhs=IrConstInt(location=_DUMMY_LOC, value=3),
                            ),
                            body=IrBreak(location=_DUMMY_LOC),
                        ),
                    ),
                    has_else=False,
                ),
                IrAssign(
                    location=_DUMMY_LOC,
                    symbol=count_sym,
                    path=(),
                    value=IrArith(
                        location=_DUMMY_LOC,
                        op=ArithOp.ADD,
                        kind=ArithKind.INT,
                        lhs=IrLoad(location=_DUMMY_LOC, symbol=count_sym),
                        rhs=IrConstInt(location=_DUMMY_LOC, value=1),
                    ),
                ),
                continue_in_try,
            ),
        ),
    )
    prog = _make_minimal_program(
        (
            IrBind(
                location=_DUMMY_LOC,
                symbol=count_sym,
                value=IrConstInt(location=_DUMMY_LOC, value=0),
            ),
            loop,
        ),
        symbols=symbols,
    )
    interp = IrInterpreter(prog)
    result = interp.run()
    # IrContinue propagated through try; count incremented to 3 then loop exited.
    assert result["count"] == IntValue(3)


# ---------------------------------------------------------------------------
# validate_ir: IrLoop, IrBreak, IrContinue structural validation
# ---------------------------------------------------------------------------


def test_validate_ir_irloop_simple_body() -> None:
    """validate_ir accepts a simple IrLoop(body=IrConstUnit)."""
    prog = _make_minimal_program(
        (
            IrLoop(
                location=_DUMMY_LOC,
                body=IrConstUnit(location=_DUMMY_LOC),
            ),
        ),
        source_text="",
    )
    validate_ir(prog)


def test_validate_ir_irbreak() -> None:
    """validate_ir accepts IrBreak inside an IrLoop body."""
    prog = _make_minimal_program(
        (IrLoop(location=_DUMMY_LOC, body=IrBreak(location=_DUMMY_LOC)),),
        source_text="",
    )
    validate_ir(prog)


def test_validate_ir_ircontinue() -> None:
    """validate_ir accepts IrContinue inside an IrLoop body."""
    # IrContinue alone would loop forever; embed a break to keep it valid
    count_sym = SymbolId(0)
    symbols = {
        count_sym: SymbolDescriptor(
            symbol_id=count_sym,
            mutable=True,
            public_name=None,
            owner=ENTRY_ID,
        )
    }
    prog = _make_minimal_program(
        (
            IrBind(
                location=_DUMMY_LOC,
                symbol=count_sym,
                value=IrConstInt(location=_DUMMY_LOC, value=0),
            ),
            IrLoop(
                location=_DUMMY_LOC,
                body=IrBlock(
                    location=_DUMMY_LOC,
                    items=(
                        IrIf(
                            location=_DUMMY_LOC,
                            branches=(
                                IrIfBranch(
                                    cond=IrConstBool(location=_DUMMY_LOC, value=True),
                                    body=IrBreak(location=_DUMMY_LOC),
                                ),
                            ),
                            has_else=False,
                        ),
                        IrContinue(location=_DUMMY_LOC),
                    ),
                ),
            ),
        ),
        source_text="",
        symbols=symbols,
    )
    validate_ir(prog)


# ---------------------------------------------------------------------------
# for-loop: iteration over list, dict, and text collections
# ---------------------------------------------------------------------------


def test_for_loop_list_iteration_accumulates_sum() -> None:
    """for x in list do body done — iterates over all list elements."""
    source = (
        "var items = [1, 2, 3, 4]\n"
        "var total = 0\n"
        "for x in items do\n"
        "  total := total + x\n"
        "done\n"
        "total\n"
    )
    result = evaluate_ir(source)
    assert result["total"] == IntValue(10)


def test_for_loop_list_empty_skips_body() -> None:
    """for x in empty list do body done — body never executes."""
    source = (
        "let xs: list[int] = []\n"
        "var total = 0\n"
        "for x in xs do\n"
        "  total := total + 1\n"
        "done\n"
        "total\n"
    )
    result = evaluate_ir(source)
    assert result["total"] == IntValue(0)


def test_for_loop_dict_iterates_keys() -> None:
    """for k in dict do body done — iterates over dict keys."""
    source = (
        "var count = 0\n"
        'for k in {"a": 1, "b": 2, "c": 3} do\n'
        "  count := count + 1\n"
        "done\n"
        "count\n"
    )
    result = evaluate_ir(source)
    assert result["count"] == IntValue(3)


def test_for_loop_text_iterates_characters() -> None:
    """for c in text do body done — iterates over individual characters."""
    source = (
        "var count = 0\n"
        'for c in "hello" do\n'
        "  count := count + 1\n"
        "done\n"
        "count\n"
    )
    result = evaluate_ir(source)
    assert result["count"] == IntValue(5)


def test_for_loop_var_accessible_in_body() -> None:
    """The iteration variable is in scope inside the loop body."""
    source = (
        "var last = \"\"\n"
        'for ch in "abc" do\n'
        "  last := ch\n"
        "done\n"
        "last\n"
    )
    result = evaluate_ir(source)
    assert result["last"] == TextValue("c")


def test_for_loop_list_bound_limits_iterations() -> None:
    """for x in list do[bound] body until false — bound raises MaxIterationsExceeded."""
    source = (
        "var total = 0\n"
        "for x in [10, 20, 30, 40, 50] do[3]\n"
        "  total := total + x\n"
        "until false\n"
    )
    exc = evaluate_ir_raises(source)
    assert exc.display_name == "MaxIterationsExceeded"
    assert exc.fields.get("limit") == IntValue(3)


# ---------------------------------------------------------------------------
# while-clause guard
# ---------------------------------------------------------------------------


def test_while_loop_runs_while_condition_true() -> None:
    """while cond do body done — body runs as long as condition holds."""
    source = (
        "var n = 0\n"
        "while n < 5 do\n"
        "  n := n + 1\n"
        "done\n"
        "n\n"
    )
    result = evaluate_ir(source)
    assert result["n"] == IntValue(5)


def test_while_loop_false_condition_skips_body() -> None:
    """while false do body done — condition already false; body never runs."""
    source = (
        "var n = 0\n"
        "while false do\n"
        "  n := n + 1\n"
        "done\n"
        "n\n"
    )
    result = evaluate_ir(source)
    assert result["n"] == IntValue(0)


def test_for_loop_with_while_guard() -> None:
    """for x in list while cond do body done — while guard stops early."""
    source = (
        "var total = 0\n"
        "for x in [1, 2, 3, 4, 5] while x <= 3 do\n"
        "  total := total + x\n"
        "done\n"
        "total\n"
    )
    result = evaluate_ir(source)
    assert result["total"] == IntValue(6)


# ---------------------------------------------------------------------------
# for-loop type errors
# ---------------------------------------------------------------------------


def test_for_loop_non_iterable_bool_raises_type_error() -> None:
    """for x in bool do body done — non-iterable type is a typecheck error."""
    from agm.agl.typecheck.env import AglTypeError
    from tests.agl.ir_harness import m2_caps

    source = "for x in true do\n  ()\ndone\n"
    with pytest.raises(AglTypeError):
        from agm.agl.parser import parse_program
        from agm.agl.scope import resolve
        from agm.agl.typecheck import check

        check(resolve(parse_program(source)), m2_caps())


def test_for_loop_int_collection_raises_type_error() -> None:
    """for x in int_expr do body done — int is not an iterable collection."""
    from agm.agl.typecheck.env import AglTypeError
    from tests.agl.ir_harness import m2_caps

    source = "for x in 42 do\n  ()\ndone\n"
    with pytest.raises(AglTypeError):
        from agm.agl.parser import parse_program
        from agm.agl.scope import resolve
        from agm.agl.typecheck import check

        check(resolve(parse_program(source)), m2_caps())


# ---------------------------------------------------------------------------
# validate_ir: IrIterInit, IrIterHasNext, IrIterNext structural validation
# ---------------------------------------------------------------------------


def test_validate_ir_iterinit_iternext_in_loop() -> None:
    """validate_ir accepts IrIterInit / IrIterHasNext / IrIterNext in an IrLoop."""
    it_sym = SymbolId(42)
    symbols = {
        it_sym: SymbolDescriptor(
            symbol_id=it_sym,
            mutable=True,
            public_name=None,
            owner=ENTRY_ID,
        )
    }
    loop = IrLoop(
        location=_DUMMY_LOC,
        body=IrBlock(
            location=_DUMMY_LOC,
            items=(
                IrIf(
                    location=_DUMMY_LOC,
                    branches=(
                        IrIfBranch(
                            cond=IrIterHasNext(
                                location=_DUMMY_LOC,
                                iterator=IrLoad(location=_DUMMY_LOC, symbol=it_sym),
                            ),
                            body=IrBreak(location=_DUMMY_LOC),
                        ),
                    ),
                    has_else=False,
                ),
                IrIterNext(
                    location=_DUMMY_LOC,
                    iterator=IrLoad(location=_DUMMY_LOC, symbol=it_sym),
                ),
            ),
        ),
    )
    prog = _make_minimal_program(
        (
            IrBind(
                location=_DUMMY_LOC,
                symbol=it_sym,
                value=IrIterInit(
                    location=_DUMMY_LOC,
                    kind=IterKind.LIST,
                    collection=IrConstText(location=_DUMMY_LOC, value="placeholder"),
                ),
            ),
            loop,
        ),
        source_text="",
        symbols=symbols,
    )
    validate_ir(prog)  # must not raise
