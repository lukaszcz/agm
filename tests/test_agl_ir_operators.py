"""IR evaluation tests for operator nodes.

Tests all operator node types: IrArith, IrCompare, IrContains, IrAnd, IrOr, IrUnary.

Also includes:
- Golden lowering tests (structural IR shape assertions)
- Coverage tests for defensive branches in arith.py and validate.py
"""

from __future__ import annotations

import decimal

import pytest

from agm.agl.semantics.values import (
    BoolValue,
    DecimalValue,
    IntValue,
    TextValue,
)
from tests.agl.ir_harness import evaluate_ir, evaluate_ir_raises

# ---------------------------------------------------------------------------
# Arithmetic: int + int → int
# ---------------------------------------------------------------------------

def test_int_add() -> None:
    source = "let x: int = 2 + 3\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == IntValue(5)

def test_int_sub() -> None:
    source = "let x: int = 10 - 3\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == IntValue(7)

def test_int_mul() -> None:
    source = "let x: int = 4 * 5\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == IntValue(20)

# ---------------------------------------------------------------------------
# Arithmetic: decimal operations
# ---------------------------------------------------------------------------

def test_decimal_add() -> None:
    source = "let x: decimal = 1.5 + 2.5\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == DecimalValue(decimal.Decimal("4.0"))

def test_decimal_sub() -> None:
    source = "let x: decimal = 5.0 - 2.5\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == DecimalValue(decimal.Decimal("2.5"))

def test_decimal_mul() -> None:
    source = "let x: decimal = 2.0 * 3.0\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == DecimalValue(decimal.Decimal("6.00"))

# ---------------------------------------------------------------------------
# Arithmetic: text concatenation
# ---------------------------------------------------------------------------

def test_text_concat() -> None:
    source = 'let x: text = "hello" + " world"\n()'
    ir = evaluate_ir(source)
    assert ir["x"] == TextValue("hello world")

# ---------------------------------------------------------------------------
# Arithmetic: mixed int + decimal widening
# ---------------------------------------------------------------------------

def test_mixed_int_decimal_add() -> None:
    source = "let x: decimal = 1 + 2.5\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == DecimalValue(decimal.Decimal("3.5"))

def test_mixed_decimal_int_add() -> None:
    source = "let x: decimal = 2.5 + 1\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == DecimalValue(decimal.Decimal("3.5"))

# ---------------------------------------------------------------------------
# Division: always decimal result
# ---------------------------------------------------------------------------

def test_div_decimal_result() -> None:
    source = "let x: decimal = 10 / 4\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == DecimalValue(decimal.Decimal("2.5"))

def test_div_decimal_decimal() -> None:
    source = "let x: decimal = 9.0 / 3.0\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == DecimalValue(decimal.Decimal("3"))

# ---------------------------------------------------------------------------
# Division by zero
# ---------------------------------------------------------------------------

def test_div_by_zero_raises() -> None:
    source = "let x: decimal = 1 / 0\n()"
    ir_exc = evaluate_ir_raises(source)
    assert ir_exc.display_name == "ArithmeticError"
    assert ir_exc.fields["message"] == TextValue("Division by zero")

# ---------------------------------------------------------------------------
# Comparisons: EQ / NEQ
# ---------------------------------------------------------------------------

def test_eq_int() -> None:
    source = "let x: bool = 3 == 3\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_neq_int() -> None:
    source = "let x: bool = 3 != 4\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_eq_text() -> None:
    source = 'let x: bool = "a" == "a"\n()'
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_neq_text() -> None:
    source = 'let x: bool = "a" != "b"\n()'
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_eq_bool() -> None:
    source = "let x: bool = true == true\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_eq_int_decimal_widening() -> None:
    source = "let x: bool = 2 == 2.0\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

# ---------------------------------------------------------------------------
# Comparisons: ordering
# ---------------------------------------------------------------------------

def test_lt_int() -> None:
    source = "let x: bool = 2 < 3\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_le_int() -> None:
    source = "let x: bool = 3 <= 3\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_gt_int() -> None:
    source = "let x: bool = 5 > 3\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_ge_int() -> None:
    source = "let x: bool = 3 >= 3\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_lt_decimal() -> None:
    source = "let x: bool = 1.5 < 2.5\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_le_decimal() -> None:
    source = "let x: bool = 2.5 <= 2.5\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_gt_decimal() -> None:
    source = "let x: bool = 3.0 > 2.5\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_ge_decimal() -> None:
    source = "let x: bool = 2.5 >= 2.5\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_lt_text() -> None:
    source = 'let x: bool = "abc" < "abd"\n()'
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_le_text() -> None:
    source = 'let x: bool = "abc" <= "abc"\n()'
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_gt_text() -> None:
    source = 'let x: bool = "abd" > "abc"\n()'
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_ge_text() -> None:
    source = 'let x: bool = "abc" >= "abc"\n()'
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_ordering_mixed_int_decimal() -> None:
    source = "let x: bool = 1 < 1.5\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

# ---------------------------------------------------------------------------
# In operator
# ---------------------------------------------------------------------------

def test_in_list() -> None:
    source = "let xs: list[int] = [1, 2, 3]\nlet x: bool = 2 in xs\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_not_in_list() -> None:
    source = "let xs: list[int] = [1, 2, 3]\nlet x: bool = 5 in xs\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(False)

def test_membership_in_empty_list_is_false() -> None:
    """`x in xs` on an empty list evaluates to false."""
    ir = evaluate_ir("let xs: list[int] = []\nlet has = 5 in xs\nhas\n")
    assert ir["has"] == BoolValue(False)

def test_in_dict() -> None:
    source = 'let m: dict[text, int] = {"a": 1}\nlet x: bool = "a" in m\n()'
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_not_in_dict() -> None:
    source = 'let m: dict[text, int] = {"a": 1}\nlet x: bool = "b" in m\n()'
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(False)

def test_in_text() -> None:
    source = 'let x: bool = "ell" in "hello"\n()'
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_not_in_text() -> None:
    source = 'let x: bool = "xyz" in "hello"\n()'
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(False)

# ---------------------------------------------------------------------------
# Short-circuit and/or
# ---------------------------------------------------------------------------

def test_and_true_true() -> None:
    source = "let x: bool = true and true\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_and_true_false() -> None:
    source = "let x: bool = true and false\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(False)

def test_and_false_short_circuit() -> None:
    source = "let x: bool = false and true\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(False)

def test_or_false_false() -> None:
    source = "let x: bool = false or false\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(False)

def test_or_false_true() -> None:
    source = "let x: bool = false or true\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_or_true_short_circuit() -> None:
    source = "let x: bool = true or false\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

def test_and_short_circuit_rhs_not_evaluated() -> None:
    """false and <div-by-zero> must short-circuit: rhs must NOT be evaluated."""
    # If the rhs were evaluated, 1/0 would raise ArithmeticError.
    source = "let x: bool = false and (1 / 0 == 0.0)\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(False)

def test_or_short_circuit_rhs_not_evaluated() -> None:
    """true or <div-by-zero> must short-circuit: rhs must NOT be evaluated."""
    # If the rhs were evaluated, 1/0 would raise ArithmeticError.
    source = "let y: bool = true or (1 / 0 == 0.0)\n()"
    ir = evaluate_ir(source)
    assert ir["y"] == BoolValue(True)

# ---------------------------------------------------------------------------
# Unary NOT
# ---------------------------------------------------------------------------

def test_unary_not_true() -> None:
    source = "let x: bool = not true\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(False)

def test_unary_not_false() -> None:
    source = "let x: bool = not false\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == BoolValue(True)

# ---------------------------------------------------------------------------
# Unary NEG
# ---------------------------------------------------------------------------

def test_unary_neg_int() -> None:
    source = "let x: int = -5\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == IntValue(-5)

def test_unary_neg_decimal() -> None:
    source = "let x: decimal = -3.14\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == DecimalValue(decimal.Decimal("-3.14"))

# ---------------------------------------------------------------------------
# Defensive coverage: arith.py invalid kind branches
# ---------------------------------------------------------------------------

def test_arith_sub_text_raises() -> None:
    """sub with TEXT kind must raise AssertionError (invalid in well-formed IR)."""
    from agm.agl.eval.arith import sub
    from agm.agl.ir.operations import ArithKind
    from agm.agl.semantics.values import TextValue
    with pytest.raises(AssertionError):
        sub(ArithKind.TEXT, TextValue("a"), TextValue("b"))

def test_arith_mul_text_raises() -> None:
    """mul with TEXT kind must raise AssertionError."""
    from agm.agl.eval.arith import mul
    from agm.agl.ir.operations import ArithKind
    from agm.agl.semantics.values import TextValue
    with pytest.raises(AssertionError):
        mul(ArithKind.TEXT, TextValue("a"), TextValue("b"))

def test_arith_div_by_zero_raises_sentinel() -> None:
    """div() raises AglDivisionByZero on zero divisor."""
    from agm.agl.eval.arith import AglDivisionByZero, div
    from agm.agl.semantics.values import IntValue
    with pytest.raises(AglDivisionByZero):
        div(IntValue(5), IntValue(0))

def test_logical_not_requires_bool() -> None:
    """logical_not raises AssertionError on non-bool."""
    from agm.agl.eval.arith import logical_not
    from agm.agl.semantics.values import IntValue
    with pytest.raises(AssertionError):
        logical_not(IntValue(1))

def test_order_called_with_eq_raises() -> None:
    """order() with a non-ordering op (EQ) must raise AssertionError."""
    from agm.agl.eval.arith import order
    from agm.agl.ir.operations import CmpOp
    from agm.agl.semantics.values import IntValue
    with pytest.raises(AssertionError, match="non-ordering op"):
        order(CmpOp.EQ, IntValue(1), IntValue(2))

def test_contains_list_wrong_container() -> None:
    """contains LIST with a non-ListValue raises AssertionError."""
    from agm.agl.eval.arith import contains
    from agm.agl.ir.operations import ContainsKind
    from agm.agl.semantics.values import IntValue, TextValue
    with pytest.raises(AssertionError, match="contains LIST"):
        contains(ContainsKind.LIST, IntValue(1), TextValue("not-a-list"))

def test_contains_dict_wrong_container() -> None:
    """contains DICT with a non-DictValue raises AssertionError."""
    from agm.agl.eval.arith import contains
    from agm.agl.ir.operations import ContainsKind
    from agm.agl.semantics.values import TextValue
    with pytest.raises(AssertionError, match="contains DICT"):
        contains(ContainsKind.DICT, TextValue("a"), TextValue("not-a-dict"))

def test_contains_text_wrong_types() -> None:
    """contains TEXT with non-TextValue types raises AssertionError."""
    from agm.agl.eval.arith import contains
    from agm.agl.ir.operations import ContainsKind
    from agm.agl.semantics.values import IntValue, TextValue
    with pytest.raises(AssertionError, match="contains TEXT"):
        contains(ContainsKind.TEXT, IntValue(1), TextValue("hello"))

def test_add_int_wrong_types() -> None:
    """add INT with non-IntValues raises AssertionError."""
    from agm.agl.eval.arith import add
    from agm.agl.ir.operations import ArithKind
    from agm.agl.semantics.values import IntValue, TextValue
    with pytest.raises(AssertionError, match="add INT"):
        add(ArithKind.INT, IntValue(1), TextValue("x"))

def test_add_decimal_wrong_types() -> None:
    """add DECIMAL with non-numeric values raises AssertionError."""
    from agm.agl.eval.arith import add
    from agm.agl.ir.operations import ArithKind
    from agm.agl.semantics.values import IntValue, TextValue
    with pytest.raises(AssertionError, match="add DECIMAL"):
        add(ArithKind.DECIMAL, IntValue(1), TextValue("x"))

def test_add_text_wrong_types() -> None:
    """add TEXT with non-TextValues raises AssertionError."""
    from agm.agl.eval.arith import add
    from agm.agl.ir.operations import ArithKind
    from agm.agl.semantics.values import IntValue, TextValue
    with pytest.raises(AssertionError, match="add TEXT"):
        add(ArithKind.TEXT, IntValue(1), TextValue("x"))

def test_sub_int_wrong_types() -> None:
    """sub INT with non-IntValues raises AssertionError."""
    from agm.agl.eval.arith import sub
    from agm.agl.ir.operations import ArithKind
    from agm.agl.semantics.values import IntValue, TextValue
    with pytest.raises(AssertionError, match="sub INT"):
        sub(ArithKind.INT, IntValue(1), TextValue("x"))

def test_sub_decimal_wrong_types() -> None:
    """sub DECIMAL with non-numeric values raises AssertionError."""
    from agm.agl.eval.arith import sub
    from agm.agl.ir.operations import ArithKind
    from agm.agl.semantics.values import IntValue, TextValue
    with pytest.raises(AssertionError, match="sub DECIMAL"):
        sub(ArithKind.DECIMAL, IntValue(1), TextValue("x"))

def test_mul_int_wrong_types() -> None:
    """mul INT with non-IntValues raises AssertionError."""
    from agm.agl.eval.arith import mul
    from agm.agl.ir.operations import ArithKind
    from agm.agl.semantics.values import IntValue, TextValue
    with pytest.raises(AssertionError, match="mul INT"):
        mul(ArithKind.INT, IntValue(1), TextValue("x"))

def test_mul_decimal_wrong_types() -> None:
    """mul DECIMAL with non-numeric values raises AssertionError."""
    from agm.agl.eval.arith import mul
    from agm.agl.ir.operations import ArithKind
    from agm.agl.semantics.values import IntValue, TextValue
    with pytest.raises(AssertionError, match="mul DECIMAL"):
        mul(ArithKind.DECIMAL, IntValue(1), TextValue("x"))

def test_negate_int_wrong_type() -> None:
    """negate INT with non-IntValue raises AssertionError."""
    from agm.agl.eval.arith import negate
    from agm.agl.ir.operations import NumericKind
    from agm.agl.semantics.values import DecimalValue
    with pytest.raises(AssertionError, match="negate INT"):
        negate(NumericKind.INT, DecimalValue(decimal.Decimal("1.5")))

def test_negate_decimal_wrong_type() -> None:
    """negate DECIMAL with non-DecimalValue raises AssertionError."""
    from agm.agl.eval.arith import negate
    from agm.agl.ir.operations import NumericKind
    from agm.agl.semantics.values import IntValue
    with pytest.raises(AssertionError, match="negate DECIMAL"):
        negate(NumericKind.DECIMAL, IntValue(5))

# ---------------------------------------------------------------------------
# Defensive coverage: validate.py invalid IR
# ---------------------------------------------------------------------------

def test_validate_arith_text_sub_raises() -> None:
    """Validate raises InvalidIrError when TEXT kind used with SUB."""
    from agm.agl.ir.ids import Location, SourceId
    from agm.agl.ir.nodes import IrArith, IrConstText
    from agm.agl.ir.operations import ArithKind, ArithOp
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import InvalidIrError
    from agm.agl.modules.ids import ENTRY_ID
    loc = Location(source_id=SourceId(0), start_offset=0, end_offset=1, start_line=1, start_col=0)
    node = IrArith(
        location=loc,
        op=ArithOp.SUB,
        kind=ArithKind.TEXT,
        lhs=IrConstText(location=loc, value="a"),
        rhs=IrConstText(location=loc, value="b"),
    )
    program = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
        symbols={},
        nominals={},
        sources={SourceId(0): SourceFile(display_name="<test>", normalized_text="x")},
    )
    from agm.agl.ir.validate import validate_ir
    with pytest.raises(InvalidIrError):
        validate_ir(program, deep=False)

def test_validate_arith_div_non_decimal_raises() -> None:
    """Validate raises InvalidIrError when DIV has non-DECIMAL kind."""
    from agm.agl.ir.ids import Location, SourceId
    from agm.agl.ir.nodes import IrArith, IrConstInt
    from agm.agl.ir.operations import ArithKind, ArithOp
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import InvalidIrError, validate_ir
    from agm.agl.modules.ids import ENTRY_ID
    loc = Location(source_id=SourceId(0), start_offset=0, end_offset=1, start_line=1, start_col=0)
    node = IrArith(
        location=loc,
        op=ArithOp.DIV,
        kind=ArithKind.INT,
        lhs=IrConstInt(location=loc, value=10),
        rhs=IrConstInt(location=loc, value=2),
    )
    program = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
        symbols={},
        nominals={},
        sources={SourceId(0): SourceFile(display_name="<test>", normalized_text="x")},
    )
    with pytest.raises(InvalidIrError):
        validate_ir(program, deep=False)

def test_validate_compare_structural_with_ordering_raises() -> None:
    """Validate raises InvalidIrError when STRUCTURAL kind used with LT."""
    from agm.agl.ir.ids import Location, SourceId
    from agm.agl.ir.nodes import IrCompare, IrConstInt
    from agm.agl.ir.operations import CmpOp, CompareKind
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import InvalidIrError, validate_ir
    from agm.agl.modules.ids import ENTRY_ID
    loc = Location(source_id=SourceId(0), start_offset=0, end_offset=1, start_line=1, start_col=0)
    node = IrCompare(
        location=loc,
        op=CmpOp.LT,
        kind=CompareKind.STRUCTURAL,
        lhs=IrConstInt(location=loc, value=1),
        rhs=IrConstInt(location=loc, value=2),
    )
    program = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
        symbols={},
        nominals={},
        sources={SourceId(0): SourceFile(display_name="<test>", normalized_text="x")},
    )
    with pytest.raises(InvalidIrError):
        validate_ir(program, deep=False)

def test_validate_unary_neg_none_kind_raises() -> None:
    """Validate raises InvalidIrError when NEG has kind=None."""
    from agm.agl.ir.ids import Location, SourceId
    from agm.agl.ir.nodes import IrConstInt, IrUnary
    from agm.agl.ir.operations import UnaryOp
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import InvalidIrError, validate_ir
    from agm.agl.modules.ids import ENTRY_ID
    loc = Location(source_id=SourceId(0), start_offset=0, end_offset=1, start_line=1, start_col=0)
    node = IrUnary(location=loc, op=UnaryOp.NEG, kind=None, value=IrConstInt(location=loc, value=5))
    program = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
        symbols={},
        nominals={},
        sources={SourceId(0): SourceFile(display_name="<test>", normalized_text="x")},
    )
    with pytest.raises(InvalidIrError):
        validate_ir(program, deep=False)

def test_validate_unary_not_with_kind_raises() -> None:
    """Validate raises InvalidIrError when NOT has non-None kind."""
    from agm.agl.ir.ids import Location, SourceId
    from agm.agl.ir.nodes import IrConstBool, IrUnary
    from agm.agl.ir.operations import NumericKind, UnaryOp
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import InvalidIrError, validate_ir
    from agm.agl.modules.ids import ENTRY_ID
    loc = Location(source_id=SourceId(0), start_offset=0, end_offset=1, start_line=1, start_col=0)
    node = IrUnary(
        location=loc,
        op=UnaryOp.NOT,
        kind=NumericKind.INT,
        value=IrConstBool(location=loc, value=True),
    )
    program = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
        symbols={},
        nominals={},
        sources={SourceId(0): SourceFile(display_name="<test>", normalized_text="x")},
    )
    with pytest.raises(InvalidIrError):
        validate_ir(program, deep=False)
