"""IR evaluation tests for do…until loops (IrLoop).

Covers:
- A do loop with an explicit small limit that terminates via until (body
  mutates a var, condition reads it); verifies the loop runs the right number
  of times and yields unit.
- A do loop with no explicit limit that terminates before the default.
- Loop exhaustion raises MaxIterationsExceeded with all fields matching (modulo
  trace_id).
- Golden lowering: IrLoop.limit (present + None) and condition_source.
- Defensive evaluator test: IrLoop with a non-bool condition → InvalidIrError.
"""

from __future__ import annotations

import pytest

from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.ir.ids import Location, NominalId, SourceId
from agm.agl.ir.nodes import IrConstBool, IrConstInt, IrConstUnit, IrLoop
from agm.agl.ir.program import (
    ExecutableModule,
    ExecutableProgram,
    NominalDescriptor,
    NominalKind,
    SourceFile,
)
from agm.agl.ir.validate import InvalidIrError, validate_ir
from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import BoolValue, IntValue, JsonValue, TextValue
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


# ---------------------------------------------------------------------------
# Loop terminates via until (explicit limit, body mutates var)
# ---------------------------------------------------------------------------


def test_loop_terminates_explicit_limit() -> None:
    """do[10] body until cond — terminates after body mutates var; yields unit.

    AgL uses `=` for equality, `:=` for assignment, `>=` for ordering.
    A top-level block must end with an expression, so we use ``counter`` as the
    final expression and read it from the snapshot.
    """
    source = (
        "var counter = 0\n"
        "do[10]\n"
        "  counter := counter + 1\n"
        "until counter >= 3\n"
        "counter\n"
    )
    ir = evaluate_ir(source)
    # The loop runs 3 iterations: counter goes 1 → 2 → 3; condition true at end of
    # iteration 3 → loop exits.
    assert ir["counter"] == IntValue(3)


# ---------------------------------------------------------------------------
# Loop terminates via until, body-bound var readable by condition
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Loop terminates without explicit limit (uses evaluator default)
# ---------------------------------------------------------------------------


def test_loop_no_explicit_limit_terminates() -> None:
    """do body until cond (no explicit limit) — terminates before the default 100."""
    source = (
        "var n = 0\n"
        "do\n"
        "  n := n + 1\n"
        "until n >= 5\n"
        "n\n"
    )
    ir = evaluate_ir(source)
    assert ir["n"] == IntValue(5)


# ---------------------------------------------------------------------------
# Loop exhaustion → MaxIterationsExceeded with matching fields
# ---------------------------------------------------------------------------


def test_loop_exhaustion_raises() -> None:
    """do[3] body until false raises MaxIterationsExceeded; all fields match."""
    # Use a small explicit limit so the test is fast.
    # Top level: must end in an expression, but the loop raises before we get there.
    source = "var dummy = 0\ndo[3]\n  dummy := 1\nuntil false\n"
    ir_exc = evaluate_ir_raises(source)

    assert ir_exc.display_name == "MaxIterationsExceeded"

    # The condition field must be the source-text slice of the condition expression.
    cond_field = ir_exc.fields.get("condition")
    assert isinstance(cond_field, TextValue), f"condition field: {cond_field!r}"
    assert cond_field.value == "false", (
        f"condition source text mismatch: {cond_field.value!r}"
    )

    assert ir_exc.fields.get("limit") == IntValue(3)
    assert ir_exc.fields.get("last_condition_value") == BoolValue(False)

    # metadata must be JsonValue(None).
    assert ir_exc.fields.get("metadata") == JsonValue(None)


# ---------------------------------------------------------------------------
# Condition source-text slice
# ---------------------------------------------------------------------------


def test_condition_source_slice_complex() -> None:
    """Condition source text captures the exact condition expression text."""
    source = (
        "var i = 0\n"
        "do[2]\n"
        "  i := i + 1\n"
        "until i > 10\n"
    )
    # This loop exhausts (i goes 1, 2 — never > 10 within 2 iterations).
    ir_exc = evaluate_ir_raises(source)

    cond_field = ir_exc.fields.get("condition")
    assert isinstance(cond_field, TextValue)
    # The condition expression text should be "i > 10".
    assert cond_field.value == "i > 10", f"got: {cond_field.value!r}"



# ---------------------------------------------------------------------------
# Loop-limit boundary — exact limit succeeds / one-short exhausts
# ---------------------------------------------------------------------------


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
    """do[2] until counter>=3 needs 3 iterations but the limit is 2 → MaxIterationsExceeded."""
    exc = evaluate_ir_raises(
        "var counter = 0\n"
        "do[2]\n"
        "  counter := counter + 1\n"
        "until counter >= 3\n"
        "counter\n"
    )
    assert exc.display_name == "MaxIterationsExceeded"
    assert exc.fields.get("limit") == IntValue(2)


# ---------------------------------------------------------------------------
# Golden lowering: IrLoop node shape
# ---------------------------------------------------------------------------


def test_golden_lowering_irloop_explicit_limit() -> None:
    """lower_program emits IrLoop with limit=<explicit int> for do[N]."""
    source = "var x = 0\ndo[7]\n  x := x + 1\nuntil x >= 3\n"
    executable = _lower(source)

    # Collect IrLoop nodes from the initializers.
    ir_loops = [
        node
        for node in executable.modules[ENTRY_ID].initializers
        if isinstance(node, IrLoop)
    ]
    assert len(ir_loops) == 1, f"expected 1 IrLoop, found {len(ir_loops)}"
    loop = ir_loops[0]
    assert loop.limit == 7, f"expected limit=7, got {loop.limit!r}"
    assert loop.condition_source.strip() == "x >= 3", (
        f"condition_source mismatch: {loop.condition_source!r}"
    )


def test_golden_lowering_irloop_no_limit() -> None:
    """lower_program emits IrLoop with limit=None for do without explicit limit."""
    source = "var y = 0\ndo\n  y := y + 1\nuntil y >= 2\n"
    executable = _lower(source)

    ir_loops = [
        node
        for node in executable.modules[ENTRY_ID].initializers
        if isinstance(node, IrLoop)
    ]
    assert len(ir_loops) == 1
    loop = ir_loops[0]
    assert loop.limit is None, f"expected limit=None, got {loop.limit!r}"


# ---------------------------------------------------------------------------
# Defensive evaluator: IrLoop with non-bool condition → InvalidIrError
# ---------------------------------------------------------------------------


def test_defensive_irloop_non_bool_condition() -> None:
    """IrLoop whose condition evaluates to non-BoolValue raises InvalidIrError."""
    # Build a hand-crafted program: do[1] () until 42  (int, not bool)
    prog = _make_minimal_program(
        (
            IrLoop(
                location=_DUMMY_LOC,
                limit=1,
                body=IrConstUnit(location=_DUMMY_LOC),
                condition=IrConstInt(location=_DUMMY_LOC, value=42),
                condition_source="42",
            ),
        ),
        source_text="",
    )
    interp = IrInterpreter(prog)
    with pytest.raises(InvalidIrError, match="IrLoop"):
        interp.run()


# ---------------------------------------------------------------------------
# validate_ir: IrLoop with negative limit raises InvalidIrError
# ---------------------------------------------------------------------------


def test_validate_ir_irloop_negative_limit() -> None:
    """validate_ir rejects IrLoop with limit < 0."""
    prog = _make_minimal_program(
        (
            IrLoop(
                location=_DUMMY_LOC,
                limit=-1,
                body=IrConstUnit(location=_DUMMY_LOC),
                condition=IrConstBool(location=_DUMMY_LOC, value=True),
                condition_source="true",
            ),
        ),
        source_text="",
    )
    with pytest.raises(InvalidIrError, match="IrLoop.*limit"):
        validate_ir(prog)


# ---------------------------------------------------------------------------
# validate_ir: IrLoop with limit=0 is allowed (exhausts immediately)
# ---------------------------------------------------------------------------


def test_validate_ir_irloop_zero_limit() -> None:
    """validate_ir accepts IrLoop with limit=0 (valid — exhausts immediately)."""
    prog = _make_minimal_program(
        (
            IrLoop(
                location=_DUMMY_LOC,
                limit=0,
                body=IrConstUnit(location=_DUMMY_LOC),
                condition=IrConstBool(location=_DUMMY_LOC, value=True),
                condition_source="true",
            ),
        ),
        source_text="",
    )
    # Should not raise.
    validate_ir(prog)


# ---------------------------------------------------------------------------
# validate_ir: IrLoop with limit=None is allowed
# ---------------------------------------------------------------------------


def test_validate_ir_irloop_none_limit() -> None:
    """validate_ir accepts IrLoop with limit=None (evaluator uses default)."""
    prog = _make_minimal_program(
        (
            IrLoop(
                location=_DUMMY_LOC,
                limit=None,
                body=IrConstUnit(location=_DUMMY_LOC),
                condition=IrConstBool(location=_DUMMY_LOC, value=True),
                condition_source="true",
            ),
        ),
        source_text="",
    )
    validate_ir(prog)


# ---------------------------------------------------------------------------
# IrInterpreter.loop_limit: explicit kwarg overrides default
# ---------------------------------------------------------------------------


def test_irinterpreter_custom_loop_limit() -> None:
    """IrInterpreter(loop_limit=2) uses 2 as default when IrLoop.limit is None."""
    source = "var x = 0\ndo\n  x := x + 1\nuntil x >= 99\n"
    executable = _lower(source)

    interp = IrInterpreter(executable, loop_limit=2)
    with pytest.raises(AglRaise) as exc_info:
        interp.run()

    raised = exc_info.value.exc
    assert raised.display_name == "MaxIterationsExceeded"
    assert raised.fields.get("limit") == IntValue(2)


# ---------------------------------------------------------------------------
# CRLF normalization: condition_source must be sliced from normalized text
# ---------------------------------------------------------------------------


def test_crlf_condition_source_is_normalized() -> None:
    """With CRLF source, the lowerer slices normalized text so condition_source is clean.

    Spans are computed by the lexer against newline-normalized source; the
    lowerer must normalize too, else the slice is offset by the stripped \\r
    bytes.  The lowerer normalizes the source text before slicing.
    """
    source = "var i = 0\r\ndo[3]\r\n  i := i + 1\r\nuntil i > 100\r\n"
    executable = _lower(source)
    ir_loops = [
        node
        for node in executable.modules[ENTRY_ID].initializers
        if isinstance(node, IrLoop)
    ]
    assert len(ir_loops) == 1
    # Clean slice — no stray '\r' and the exact condition text.
    assert ir_loops[0].condition_source == "i > 100"

    # The IR pipeline raises MaxIterationsExceeded with the condition source-text field.
    ir_exc = evaluate_ir_raises(source)
    assert ir_exc.display_name == "MaxIterationsExceeded"
    assert ir_exc.fields.get("condition") == TextValue("i > 100")
