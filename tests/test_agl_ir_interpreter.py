"""Tests for the IrInterpreter — hand-built ExecutableProgram programs.

These tests drive the evaluator with manually constructed IR programs; they do
NOT depend on the lowerer.  Helper factories keep
boilerplate minimal.

Coverage targets:
- Every constant type.
- List and dict construction.
- let (immutable) bind + load.
- var (mutable) bind + load + assign (cell mutation).
- IrSequence and IrBlock (value-of-last).
- All IrCoerce operations: IntToDecimal, ToJson (scalar + container), MapList,
  MapDictValues, MapRecordFields, MapEnumFields.
- Decimal context: IntToDecimal of a large int is exact (no float).
- Defensive InvalidIrError on a malformed coercion.
- Import-scan: IrInterpreter must NOT import syntax/scope/typecheck modules.
"""

from __future__ import annotations

import ast
import decimal
import importlib
import pathlib

import pytest

from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.ir import (
    ExecutableModule,
    ExecutableProgram,
    FunctionDescriptor,
    FunctionId,
    IndexKind,
    IntToDecimal,
    InvalidIrError,
    IrAssign,
    IrBind,
    IrBlock,
    IrCapture,
    IrCoerce,
    IrConstBool,
    IrConstDecimal,
    IrConstInt,
    IrConstJsonNull,
    IrConstText,
    IrConstUnit,
    IrDirectCall,
    IrExpr,
    IrField,
    IrFunctionBody,
    IrFunctionParam,
    IrIndex,
    IrIndexStep,
    IrIndirectCall,
    IrLoad,
    IrMakeClosure,
    IrMakeDict,
    IrMakeException,
    IrMakeList,
    IrMakeRecord,
    IrPrint,
    IrRaise,
    IrSequence,
    Location,
    MapDictValues,
    MapEnumFields,
    MapList,
    MapRecordFields,
    NominalDescriptor,
    NominalKind,
    SourceFile,
    SourceId,
    SymbolDescriptor,
    SymbolId,
    ToJson,
    UseDefault,
)
from agm.agl.ir.ids import NominalId
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.semantics.values import (
    VOID_VALUE,
    BoolValue,
    DecimalValue,
    DictValue,
    EnumValue,
    IntValue,
    JsonValue,
    ListValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
)

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

_SOURCE_ID = SourceId(0)
_LOC = Location(
    source_id=_SOURCE_ID,
    start_offset=0,
    end_offset=1,
    start_line=1,
    start_col=0,
)
_SOURCE_TEXT = "x"


def _make_program(
    initializers: tuple[IrExpr, ...],
    symbols: dict[SymbolId, SymbolDescriptor] | None = None,
    nominals: dict[NominalId, NominalDescriptor] | None = None,
    functions: "dict[FunctionId, FunctionDescriptor] | None" = None,
) -> ExecutableProgram:
    """Build a minimal single-module ExecutableProgram."""
    sources = {_SOURCE_ID: SourceFile(display_name="<test>", normalized_text=_SOURCE_TEXT)}
    return ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=initializers)},
        symbols=symbols or {},
        nominals=nominals or {},
        sources=sources,
        functions=functions or {},
    )


def _let_sym(n: int, name: str) -> tuple[SymbolId, SymbolDescriptor]:
    """Return a (SymbolId, SymbolDescriptor) for an immutable let binding."""
    sym = SymbolId(n)
    desc = SymbolDescriptor(symbol_id=sym, mutable=False, public_name=name, owner=ENTRY_ID)
    return sym, desc


def _var_sym(n: int, name: str) -> tuple[SymbolId, SymbolDescriptor]:
    """Return a (SymbolId, SymbolDescriptor) for a mutable var binding."""
    sym = SymbolId(n)
    desc = SymbolDescriptor(symbol_id=sym, mutable=True, public_name=name, owner=ENTRY_ID)
    return sym, desc


def _run(
    initializers: tuple[IrExpr, ...],
    symbols: dict[SymbolId, SymbolDescriptor] | None = None,
) -> dict[str, Value]:
    """Build a program and run it; return the public-name bindings."""
    prog = _make_program(initializers, symbols)
    return IrInterpreter(prog).run()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_int(self) -> None:
        sym, desc = _let_sym(0, "x")
        result = _run(
            (IrBind(_LOC, sym, IrConstInt(_LOC, 42)),),
            {sym: desc},
        )
        assert result == {"x": IntValue(42)}

    def test_decimal(self) -> None:
        sym, desc = _let_sym(0, "x")
        result = _run(
            (IrBind(_LOC, sym, IrConstDecimal(_LOC, decimal.Decimal("3.14"))),),
            {sym: desc},
        )
        assert result == {"x": DecimalValue(decimal.Decimal("3.14"))}

    def test_bool_true(self) -> None:
        sym, desc = _let_sym(0, "x")
        result = _run(
            (IrBind(_LOC, sym, IrConstBool(_LOC, True)),),
            {sym: desc},
        )
        assert result == {"x": BoolValue(True)}

    def test_bool_false(self) -> None:
        sym, desc = _let_sym(0, "f")
        result = _run(
            (IrBind(_LOC, sym, IrConstBool(_LOC, False)),),
            {sym: desc},
        )
        assert result == {"f": BoolValue(False)}

    def test_text(self) -> None:
        sym, desc = _let_sym(0, "s")
        result = _run(
            (IrBind(_LOC, sym, IrConstText(_LOC, "hello")),),
            {sym: desc},
        )
        assert result == {"s": TextValue("hello")}

    def test_unit(self) -> None:
        sym, desc = _let_sym(0, "u")
        result = _run(
            (IrBind(_LOC, sym, IrConstUnit(_LOC)),),
            {sym: desc},
        )
        assert result == {"u": UnitValue()}

    def test_json_null(self) -> None:
        sym, desc = _let_sym(0, "n")
        result = _run(
            (IrBind(_LOC, sym, IrConstJsonNull(_LOC)),),
            {sym: desc},
        )
        assert result == {"n": JsonValue(None)}


# ---------------------------------------------------------------------------
# Container construction
# ---------------------------------------------------------------------------


class TestContainers:
    def test_make_list(self) -> None:
        sym, desc = _let_sym(0, "lst")
        result = _run(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrMakeList(
                        _LOC,
                        (
                            IrConstInt(_LOC, 1),
                            IrConstInt(_LOC, 2),
                            IrConstInt(_LOC, 3),
                        ),
                    ),
                ),
            ),
            {sym: desc},
        )
        assert result == {"lst": ListValue((IntValue(1), IntValue(2), IntValue(3)))}

    def test_make_list_empty(self) -> None:
        sym, desc = _let_sym(0, "empty")
        result = _run(
            (IrBind(_LOC, sym, IrMakeList(_LOC, ())),),
            {sym: desc},
        )
        assert result == {"empty": ListValue(())}

    def test_make_dict(self) -> None:
        sym, desc = _let_sym(0, "d")
        result = _run(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrMakeDict(
                        _LOC,
                        (
                            (IrConstText(_LOC, "a"), IrConstInt(_LOC, 1)),
                            (IrConstText(_LOC, "b"), IrConstInt(_LOC, 2)),
                        ),
                    ),
                ),
            ),
            {sym: desc},
        )
        assert result == {"d": DictValue({"a": IntValue(1), "b": IntValue(2)})}

    def test_make_dict_empty(self) -> None:
        sym, desc = _let_sym(0, "d")
        result = _run(
            (IrBind(_LOC, sym, IrMakeDict(_LOC, ())),),
            {sym: desc},
        )
        assert result == {"d": DictValue({})}


# ---------------------------------------------------------------------------
# Let bind + load
# ---------------------------------------------------------------------------


class TestLetBind:
    def test_let_bind_and_load(self) -> None:
        sym, desc = _let_sym(0, "x")
        # let x = 7; load x
        result = _run(
            (
                IrBind(_LOC, sym, IrConstInt(_LOC, 7)),
                # the load is just to exercise; we inspect via run() result
                IrLoad(_LOC, sym),
            ),
            {sym: desc},
        )
        assert result == {"x": IntValue(7)}

    def test_multiple_let_bindings(self) -> None:
        sym_a, desc_a = _let_sym(0, "a")
        sym_b, desc_b = _let_sym(1, "b")
        result = _run(
            (
                IrBind(_LOC, sym_a, IrConstInt(_LOC, 10)),
                IrBind(_LOC, sym_b, IrConstInt(_LOC, 20)),
            ),
            {sym_a: desc_a, sym_b: desc_b},
        )
        assert result == {"a": IntValue(10), "b": IntValue(20)}


# ---------------------------------------------------------------------------
# Var bind + load + assign (cell mutation)
# ---------------------------------------------------------------------------


class TestVarCell:
    def test_var_load_unwraps_cell(self) -> None:
        sym, desc = _var_sym(0, "v")
        result = _run(
            (IrBind(_LOC, sym, IrConstInt(_LOC, 17)),),
            {sym: desc},
        )
        assert result == {"v": IntValue(17)}

    def test_var_assign_mutates_cell(self) -> None:
        """Assigning to a var updates the cell; subsequent load sees the new value."""
        sym, desc = _var_sym(0, "counter")
        result = _run(
            (
                IrBind(_LOC, sym, IrConstInt(_LOC, 0)),
                IrAssign(_LOC, sym, (), IrConstInt(_LOC, 42)),
                IrLoad(_LOC, sym),
            ),
            {sym: desc},
        )
        assert result == {"counter": IntValue(42)}

    def test_var_assign_returns_void(self) -> None:
        """Assignment mutates the cell and yields unprintable unit."""
        sym, desc = _var_sym(0, "counter")
        prog = _make_program(
            (
                IrBind(_LOC, sym, IrConstInt(_LOC, 0)),
                IrAssign(_LOC, sym, (), IrConstInt(_LOC, 42)),
            ),
            {sym: desc},
        )
        interp = IrInterpreter(prog)
        interp.run()

        assert interp.initializer_values[-1] == UnitValue()
        assert interp.initializer_values[-1] == VOID_VALUE
        assert isinstance(interp.initializer_values[-1], UnitValue)
        assert not interp.initializer_values[-1].printable_in_repl

    def test_var_assign_multiple_times(self) -> None:
        sym, desc = _var_sym(0, "x")
        result = _run(
            (
                IrBind(_LOC, sym, IrConstInt(_LOC, 1)),
                IrAssign(_LOC, sym, (), IrConstInt(_LOC, 2)),
                IrAssign(_LOC, sym, (), IrConstInt(_LOC, 3)),
            ),
            {sym: desc},
        )
        assert result == {"x": IntValue(3)}

    def test_assign_to_missing_symbol_raises(self) -> None:
        """IrAssign to an unbound symbol raises InvalidIrError."""
        sym = SymbolId(99)
        prog = _make_program(
            (IrAssign(_LOC, sym, (), IrConstInt(_LOC, 1)),),
            symbols={},
        )
        with pytest.raises(InvalidIrError):
            IrInterpreter(prog).run()

    def test_assign_to_let_symbol_raises(self) -> None:
        """IrAssign to a non-mutable let symbol raises InvalidIrError."""
        sym, desc = _let_sym(0, "x")
        prog = _make_program(
            (
                IrBind(_LOC, sym, IrConstInt(_LOC, 5)),
                IrAssign(_LOC, sym, (), IrConstInt(_LOC, 10)),
            ),
            {sym: desc},
        )
        with pytest.raises(InvalidIrError):
            IrInterpreter(prog).run()


# ---------------------------------------------------------------------------
# Sequence and Block (value-of-last)
# ---------------------------------------------------------------------------


class TestSequenceBlock:
    def test_sequence_value_is_last(self) -> None:
        sym, desc = _let_sym(0, "r")
        result = _run(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrSequence(
                        _LOC,
                        (
                            IrConstInt(_LOC, 1),
                            IrConstInt(_LOC, 2),
                            IrConstInt(_LOC, 99),
                        ),
                    ),
                ),
            ),
            {sym: desc},
        )
        assert result == {"r": IntValue(99)}

    def test_block_value_is_last(self) -> None:
        sym, desc = _let_sym(0, "r")
        result = _run(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrBlock(
                        _LOC,
                        (
                            IrConstBool(_LOC, False),
                            IrConstText(_LOC, "last"),
                        ),
                    ),
                ),
            ),
            {sym: desc},
        )
        assert result == {"r": TextValue("last")}

    def test_sequence_single_item(self) -> None:
        sym, desc = _let_sym(0, "x")
        result = _run(
            (IrBind(_LOC, sym, IrSequence(_LOC, (IrConstInt(_LOC, 7),))),),
            {sym: desc},
        )
        assert result == {"x": IntValue(7)}


# ---------------------------------------------------------------------------
# IrCoerce — IntToDecimal
# ---------------------------------------------------------------------------


class TestCoerceIntToDecimal:
    def test_int_to_decimal(self) -> None:
        sym, desc = _let_sym(0, "d")
        result = _run(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrCoerce(_LOC, IrConstInt(_LOC, 5), IntToDecimal()),
                ),
            ),
            {sym: desc},
        )
        assert result == {"d": DecimalValue(decimal.Decimal(5))}

    def test_int_to_decimal_large_exact(self) -> None:
        """Converting a large integer to Decimal must be exact (no float loss)."""
        big = 10**30 + 7
        sym, desc = _let_sym(0, "big")
        result = _run(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrCoerce(_LOC, IrConstInt(_LOC, big), IntToDecimal()),
                ),
            ),
            {sym: desc},
        )
        assert result == {"big": DecimalValue(decimal.Decimal(big))}
        # Exact — repr round trips without loss.
        assert result["big"] == DecimalValue(decimal.Decimal(str(big)))

    def test_int_to_decimal_wrong_type_raises(self) -> None:
        """IntToDecimal on a non-int value raises InvalidIrError."""
        sym, desc = _let_sym(0, "x")
        prog = _make_program(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrCoerce(_LOC, IrConstBool(_LOC, True), IntToDecimal()),
                ),
            ),
            {sym: desc},
        )
        with pytest.raises(InvalidIrError):
            IrInterpreter(prog).run()


# ---------------------------------------------------------------------------
# IrCoerce — ToJson
# ---------------------------------------------------------------------------


class TestCoerceToJson:
    def test_to_json_int(self) -> None:
        sym, desc = _let_sym(0, "j")
        result = _run(
            (IrBind(_LOC, sym, IrCoerce(_LOC, IrConstInt(_LOC, 3), ToJson())),),
            {sym: desc},
        )
        assert result == {"j": JsonValue(3)}

    def test_to_json_text(self) -> None:
        sym, desc = _let_sym(0, "j")
        result = _run(
            (IrBind(_LOC, sym, IrCoerce(_LOC, IrConstText(_LOC, "hi"), ToJson())),),
            {sym: desc},
        )
        assert result == {"j": JsonValue("hi")}

    def test_to_json_bool(self) -> None:
        sym, desc = _let_sym(0, "j")
        result = _run(
            (IrBind(_LOC, sym, IrCoerce(_LOC, IrConstBool(_LOC, True), ToJson())),),
            {sym: desc},
        )
        assert result == {"j": JsonValue(True)}

    def test_to_json_list(self) -> None:
        """ToJson converts a ListValue to a JsonValue wrapping a list."""
        sym, desc = _let_sym(0, "j")
        result = _run(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrCoerce(
                        _LOC,
                        IrMakeList(_LOC, (IrConstInt(_LOC, 1), IrConstInt(_LOC, 2))),
                        ToJson(),
                    ),
                ),
            ),
            {sym: desc},
        )
        assert result == {"j": JsonValue([1, 2])}

    def test_to_json_already_json_is_idempotent(self) -> None:
        """ToJson on a JsonValue returns as-is (idempotent defensively)."""
        sym, desc = _let_sym(0, "j")
        result = _run(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrCoerce(_LOC, IrConstJsonNull(_LOC), ToJson()),
                ),
            ),
            {sym: desc},
        )
        assert result == {"j": JsonValue(None)}


# ---------------------------------------------------------------------------
# IrCoerce — MapList
# ---------------------------------------------------------------------------


class TestCoerceMapList:
    def test_map_list_int_to_decimal(self) -> None:
        sym, desc = _let_sym(0, "lst")
        result = _run(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrCoerce(
                        _LOC,
                        IrMakeList(
                            _LOC,
                            (IrConstInt(_LOC, 1), IrConstInt(_LOC, 2), IrConstInt(_LOC, 3)),
                        ),
                        MapList(IntToDecimal()),
                    ),
                ),
            ),
            {sym: desc},
        )
        assert result == {
            "lst": ListValue(
                (
                    DecimalValue(decimal.Decimal(1)),
                    DecimalValue(decimal.Decimal(2)),
                    DecimalValue(decimal.Decimal(3)),
                )
            )
        }

    def test_map_list_to_json(self) -> None:
        sym, desc = _let_sym(0, "lst")
        result = _run(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrCoerce(
                        _LOC,
                        IrMakeList(_LOC, (IrConstText(_LOC, "a"), IrConstText(_LOC, "b"))),
                        MapList(ToJson()),
                    ),
                ),
            ),
            {sym: desc},
        )
        assert result == {"lst": ListValue((JsonValue("a"), JsonValue("b")))}

    def test_map_list_wrong_value_type_raises(self) -> None:
        """MapList on a non-list value raises InvalidIrError."""
        sym, desc = _let_sym(0, "x")
        prog = _make_program(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrCoerce(_LOC, IrConstInt(_LOC, 1), MapList(IntToDecimal())),
                ),
            ),
            {sym: desc},
        )
        with pytest.raises(InvalidIrError):
            IrInterpreter(prog).run()


# ---------------------------------------------------------------------------
# IrCoerce — MapDictValues
# ---------------------------------------------------------------------------


class TestCoerceMapDictValues:
    def test_map_dict_values_to_json(self) -> None:
        sym, desc = _let_sym(0, "d")
        result = _run(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrCoerce(
                        _LOC,
                        IrMakeDict(
                            _LOC,
                            (
                                (IrConstText(_LOC, "k1"), IrConstInt(_LOC, 10)),
                                (IrConstText(_LOC, "k2"), IrConstInt(_LOC, 20)),
                            ),
                        ),
                        MapDictValues(ToJson()),
                    ),
                ),
            ),
            {sym: desc},
        )
        assert result == {"d": DictValue({"k1": JsonValue(10), "k2": JsonValue(20)})}

    def test_map_dict_values_wrong_type_raises(self) -> None:
        sym, desc = _let_sym(0, "x")
        prog = _make_program(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrCoerce(_LOC, IrConstInt(_LOC, 1), MapDictValues(ToJson())),
                ),
            ),
            {sym: desc},
        )
        with pytest.raises(InvalidIrError):
            IrInterpreter(prog).run()


# ---------------------------------------------------------------------------
# IrCoerce — MapRecordFields
# ---------------------------------------------------------------------------


class TestCoerceMapRecordFields:
    def _make_record_value(self) -> RecordValue:
        return RecordValue(
            nominal=NominalId(ENTRY_ID, "Point"),
            display_name="Point",
            fields={"x": IntValue(3), "y": IntValue(4), "label": TextValue("origin")},
        )

    def test_map_record_fields(self) -> None:
        """MapRecordFields coerces only the named fields; others pass through."""
        # Exercise record coercion via the module-level _apply_coercion helper.
        from agm.agl.eval.ir_interpreter import _apply_coercion

        rec = self._make_record_value()
        coercion = MapRecordFields(
            fields=(
                ("x", IntToDecimal()),
                ("y", IntToDecimal()),
            )
        )
        result = _apply_coercion(rec, coercion)
        assert result == RecordValue(
            nominal=NominalId(ENTRY_ID, "Point"),
            display_name="Point",
            fields={
                "x": DecimalValue(decimal.Decimal(3)),
                "y": DecimalValue(decimal.Decimal(4)),
                "label": TextValue("origin"),
            },
        )

    def test_map_record_fields_wrong_type_raises(self) -> None:
        from agm.agl.eval.ir_interpreter import _apply_coercion

        with pytest.raises(InvalidIrError):
            _apply_coercion(IntValue(1), MapRecordFields(fields=(("x", IntToDecimal()),)))

    def test_map_record_fields_partial_fields(self) -> None:
        """MapRecordFields with a different field set; unlisted fields pass through."""
        from agm.agl.eval.ir_interpreter import _apply_coercion

        rec = RecordValue(
            nominal=NominalId(ENTRY_ID, "Pt"),
            display_name="Pt",
            fields={"a": IntValue(10), "b": TextValue("keep")},
        )
        coercion = MapRecordFields(fields=(("a", IntToDecimal()),))
        result = _apply_coercion(rec, coercion)
        assert result == RecordValue(
            nominal=NominalId(ENTRY_ID, "Pt"),
            display_name="Pt",
            fields={"a": DecimalValue(decimal.Decimal(10)), "b": TextValue("keep")},
        )


# ---------------------------------------------------------------------------
# IrCoerce — MapEnumFields
# ---------------------------------------------------------------------------


class TestCoerceMapEnumFields:
    def test_map_enum_fields(self) -> None:
        from agm.agl.eval.ir_interpreter import _apply_coercion

        ev = EnumValue(
            nominal=NominalId(ENTRY_ID, "Shape"),
            display_name="Shape",
            variant="Circle",
            fields={"radius": IntValue(5), "label": TextValue("c")},
        )
        coercion = MapEnumFields(
            variants=(
                ("Circle", (("radius", IntToDecimal()),)),
                ("Square", (("side", IntToDecimal()),)),
            )
        )
        result = _apply_coercion(ev, coercion)
        assert result == EnumValue(
            nominal=NominalId(ENTRY_ID, "Shape"),
            display_name="Shape",
            variant="Circle",
            fields={"radius": DecimalValue(decimal.Decimal(5)), "label": TextValue("c")},
        )

    def test_map_enum_fields_unmatched_variant_is_passthrough(self) -> None:
        """A variant not listed in MapEnumFields is left unchanged."""
        from agm.agl.eval.ir_interpreter import _apply_coercion

        ev = EnumValue(
            nominal=NominalId(ENTRY_ID, "Shape"),
            display_name="Shape",
            variant="Triangle",
            fields={"sides": IntValue(3)},
        )
        coercion = MapEnumFields(variants=(("Circle", (("radius", IntToDecimal()),)),))
        result = _apply_coercion(ev, coercion)
        # Triangle variant is not in the coercion → returned unchanged.
        assert result == ev

    def test_map_enum_fields_wrong_type_raises(self) -> None:
        from agm.agl.eval.ir_interpreter import _apply_coercion

        with pytest.raises(InvalidIrError):
            _apply_coercion(IntValue(1), MapEnumFields(variants=()))


# ---------------------------------------------------------------------------
# Decimal context
# ---------------------------------------------------------------------------


class TestDecimalContext:
    def test_decimal_context_is_pinned(self) -> None:
        """The evaluator uses a pinned 28-digit ROUND_HALF_EVEN context."""
        # Verify by checking that IntToDecimal of a number requiring > default
        # precision stays exact under the pinned context.
        big = 10**27 + 3
        sym, desc = _let_sym(0, "d")
        result = _run(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrCoerce(_LOC, IrConstInt(_LOC, big), IntToDecimal()),
                ),
            ),
            {sym: desc},
        )
        # Under the pinned 28-digit context, Decimal(10**27 + 3) is exact.
        assert result["d"] == DecimalValue(decimal.Decimal(big))

    def test_decimal_literal_preserved(self) -> None:
        """An IrConstDecimal with many significant digits is stored exactly."""
        d = decimal.Decimal("1.2345678901234567890123456789")
        sym, desc = _let_sym(0, "pi")
        result = _run(
            (IrBind(_LOC, sym, IrConstDecimal(_LOC, d)),),
            {sym: desc},
        )
        assert result["pi"] == DecimalValue(d)


# ---------------------------------------------------------------------------
# run() return-value filtering (public_name)
# ---------------------------------------------------------------------------


class TestRunReturnValues:
    def test_private_symbol_excluded(self) -> None:
        """A symbol with public_name=None should not appear in run() results."""
        sym = SymbolId(0)
        desc = SymbolDescriptor(symbol_id=sym, mutable=False, public_name=None, owner=ENTRY_ID)
        result = _run(
            (IrBind(_LOC, sym, IrConstInt(_LOC, 1)),),
            {sym: desc},
        )
        assert result == {}

    def test_foreign_module_symbol_excluded(self) -> None:
        """A symbol owned by a different module should not appear in results."""
        from agm.agl.modules.ids import ModuleId

        other_mod = ModuleId.from_path("other")
        sym = SymbolId(0)
        desc = SymbolDescriptor(symbol_id=sym, mutable=False, public_name="x", owner=other_mod)
        result = _run(
            (IrBind(_LOC, sym, IrConstInt(_LOC, 1)),),
            {sym: desc},
        )
        # Symbol is owned by 'other', not the entry module — excluded.
        assert result == {}

    def test_only_bound_symbols_in_frame_returned(self) -> None:
        """Only symbols actually bound in this run appear in results."""
        sym_bound, desc_bound = _let_sym(0, "bound")
        sym_unbound, desc_unbound = _let_sym(1, "unbound")
        result = _run(
            (IrBind(_LOC, sym_bound, IrConstInt(_LOC, 42)),),
            {sym_bound: desc_bound, sym_unbound: desc_unbound},
        )
        assert "bound" in result
        assert "unbound" not in result

    def test_empty_program(self) -> None:
        result = _run((), {})
        assert result == {}


# ---------------------------------------------------------------------------
# Defensive InvalidIrError on additional malformed inputs
# ---------------------------------------------------------------------------


class TestDefensiveErrors:
    def test_make_dict_non_text_key_raises(self) -> None:
        """IrMakeDict key that evaluates to a non-TextValue raises InvalidIrError."""
        sym, desc = _let_sym(0, "d")
        prog = _make_program(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrMakeDict(
                        _LOC,
                        ((IrConstInt(_LOC, 99), IrConstInt(_LOC, 1)),),
                    ),
                ),
            ),
            {sym: desc},
        )
        with pytest.raises(InvalidIrError):
            IrInterpreter(prog).run()

    def test_load_unbound_symbol_raises(self) -> None:
        """IrLoad of a symbol not yet bound in the frame raises InvalidIrError."""
        sym, desc = _let_sym(0, "x")
        # Register the symbol in program.symbols but never IrBind it.
        prog = _make_program(
            (IrLoad(_LOC, sym),),
            {sym: desc},
        )
        with pytest.raises(InvalidIrError):
            IrInterpreter(prog).run()

    def test_assign_with_path_list(self) -> None:
        """IrAssign with a non-empty path performs indexed assignment on a list."""
        from agm.agl.semantics.values import IntValue, ListValue

        sym, desc = _var_sym(0, "v")
        prog = _make_program(
            (
                IrBind(_LOC, sym, IrMakeList(_LOC, (IrConstInt(_LOC, 10), IrConstInt(_LOC, 20)))),
                IrAssign(
                    _LOC,
                    sym,
                    (IrIndexStep(kind=IndexKind.LIST, index=IrConstInt(_LOC, 0), location=_LOC),),
                    IrConstInt(_LOC, 99),
                ),
            ),
            {sym: desc},
        )
        result = IrInterpreter(prog).run()
        assert result["v"] == ListValue((IntValue(99), IntValue(20)))

    def test_ir_and_non_bool_lhs_raises(self) -> None:
        """IrAnd with a non-BoolValue lhs raises InvalidIrError."""
        from agm.agl.ir import IrAnd

        prog = _make_program(
            (IrAnd(_LOC, lhs=IrConstInt(_LOC, 1), rhs=IrConstBool(_LOC, True)),),
        )
        with pytest.raises(InvalidIrError, match="IrAnd: lhs"):
            IrInterpreter(prog).run()

    def test_ir_and_non_bool_rhs_raises(self) -> None:
        """IrAnd with a non-BoolValue rhs raises InvalidIrError."""
        from agm.agl.ir import IrAnd

        prog = _make_program(
            (IrAnd(_LOC, lhs=IrConstBool(_LOC, True), rhs=IrConstInt(_LOC, 1)),),
        )
        with pytest.raises(InvalidIrError, match="IrAnd: rhs"):
            IrInterpreter(prog).run()

    def test_ir_or_non_bool_lhs_raises(self) -> None:
        """IrOr with a non-BoolValue lhs raises InvalidIrError."""
        from agm.agl.ir import IrOr

        prog = _make_program(
            (IrOr(_LOC, lhs=IrConstInt(_LOC, 1), rhs=IrConstBool(_LOC, False)),),
        )
        with pytest.raises(InvalidIrError, match="IrOr: lhs"):
            IrInterpreter(prog).run()

    def test_ir_or_non_bool_rhs_raises(self) -> None:
        """IrOr with a non-BoolValue rhs raises InvalidIrError."""
        from agm.agl.ir import IrOr

        prog = _make_program(
            (IrOr(_LOC, lhs=IrConstBool(_LOC, False), rhs=IrConstInt(_LOC, 1)),),
        )
        with pytest.raises(InvalidIrError, match="IrOr: rhs"):
            IrInterpreter(prog).run()

    def test_ir_unary_not_non_bool_raises(self) -> None:
        """IrUnary NOT with a non-BoolValue raises InvalidIrError."""
        from agm.agl.ir import IrUnary, UnaryOp

        prog = _make_program(
            (IrUnary(_LOC, op=UnaryOp.NOT, kind=None, value=IrConstInt(_LOC, 1)),),
        )
        with pytest.raises(InvalidIrError, match="IrUnary NOT"):
            IrInterpreter(prog).run()

    def test_ir_unary_neg_none_kind_raises(self) -> None:
        """IrUnary NEG with kind=None raises InvalidIrError at runtime."""
        from agm.agl.ir import IrUnary, UnaryOp

        prog = _make_program(
            (IrUnary(_LOC, op=UnaryOp.NEG, kind=None, value=IrConstInt(_LOC, 5)),),
        )
        with pytest.raises(InvalidIrError, match="IrUnary NEG: kind must not be None"):
            IrInterpreter(prog).run()

    def test_ir_unary_neg_non_numeric_raises(self) -> None:
        """IrUnary NEG with non-numeric value raises InvalidIrError."""
        from agm.agl.ir import IrUnary, NumericKind, UnaryOp

        prog = _make_program(
            (
                IrUnary(
                    _LOC,
                    op=UnaryOp.NEG,
                    kind=NumericKind.INT,
                    value=IrConstText(_LOC, "not-a-number"),
                ),
            ),
        )
        with pytest.raises(InvalidIrError, match="IrUnary NEG: expected numeric"):
            IrInterpreter(prog).run()

    def test_ir_variant_is_on_non_enum_raises(self) -> None:
        """IrVariantIs on a non-enum value raises InvalidIrError (defensive)."""
        from agm.agl.ir import IrVariantIs, NominalId

        prog = _make_program(
            (
                IrVariantIs(
                    _LOC,
                    nominal=NominalId(ENTRY_ID, "Color"),
                    variant="Red",
                    value=IrConstInt(_LOC, 1),
                    negated=False,
                ),
            ),
        )
        with pytest.raises(InvalidIrError, match="IrVariantIs"):
            IrInterpreter(prog).run()


# ---------------------------------------------------------------------------
# IrField — field read on RecordValue / ExceptionValue
# ---------------------------------------------------------------------------


class TestIrField:
    """Tests for the IrField node in IrInterpreter."""

    def _run_with_record_via_make(
        self,
        field_name: str,
        *,
        x_val: int,
        y_val: int,
    ) -> Value:
        """Run an IrField read by constructing a RecordValue via IrMakeRecord.

        Constructs a Point(x, y) record inline with IrMakeRecord, binds it to
        'rec', then reads the requested field via IrField into 'out'.
        The nominal is registered in program.nominals.
        """
        rec_sym, rec_desc = _let_sym(0, "rec")
        out_sym, out_desc = _let_sym(1, "out")
        nominal = NominalId(ENTRY_ID, "Point")
        make_record = IrMakeRecord(
            location=_LOC,
            nominal=nominal,
            display_name="Point",
            fields=(
                ("x", IrConstInt(_LOC, x_val)),
                ("y", IrConstInt(_LOC, y_val)),
            ),
        )
        prog = _make_program(
            (
                IrBind(_LOC, rec_sym, make_record),
                IrBind(
                    _LOC,
                    out_sym,
                    IrField(_LOC, value=IrLoad(_LOC, rec_sym), field=field_name),
                ),
            ),
            {rec_sym: rec_desc, out_sym: out_desc},
            nominals={
                nominal: NominalDescriptor(
                    nominal=nominal,
                    display_name="Point",
                    kind=NominalKind.RECORD,
                    fields=("x", "y"),
                )
            },
        )
        return IrInterpreter(prog).run()["out"]

    def test_ir_field_reads_record_field(self) -> None:
        """IrField returns the value of a named field from a RecordValue."""
        result = self._run_with_record_via_make("x", x_val=3, y_val=4)
        assert result == IntValue(3)

    def test_ir_field_reads_second_field(self) -> None:
        """IrField returns the correct value when multiple fields are present."""
        result = self._run_with_record_via_make("y", x_val=3, y_val=7)
        assert result == IntValue(7)

    def test_ir_field_on_non_record_raises(self) -> None:
        """IrField on a non-RecordValue/non-ExceptionValue raises InvalidIrError."""
        prog = _make_program(
            (
                IrBind(_LOC, SymbolId(0), IrConstInt(_LOC, 42)),
                IrBind(
                    _LOC,
                    SymbolId(1),
                    IrField(_LOC, value=IrLoad(_LOC, SymbolId(0)), field="x"),
                ),
            ),
            {
                SymbolId(0): SymbolDescriptor(
                    symbol_id=SymbolId(0), mutable=False, public_name="n", owner=ENTRY_ID
                ),
                SymbolId(1): SymbolDescriptor(
                    symbol_id=SymbolId(1), mutable=False, public_name="out", owner=ENTRY_ID
                ),
            },
        )
        with pytest.raises(InvalidIrError, match="IrField"):
            IrInterpreter(prog).run()


# ---------------------------------------------------------------------------
# IrAssign with path — IndexError / KeyError at intermediate and final steps
# ---------------------------------------------------------------------------


class TestIrAssignPathErrors:
    """Tests for IrAssign path error handling (IndexError/KeyError in path steps).

    These tests exercise the exception-handling branches in IrInterpreter._eval
    for the IrAssign case with a non-empty path.  Each test constructs a depth-2
    assignment where the intermediate or final index is out-of-range or missing.
    """

    def _var(self, n: int, name: str) -> tuple[SymbolId, SymbolDescriptor]:
        return _var_sym(n, name)

    def test_assign_path_intermediate_list_oob_raises(self) -> None:
        """IrAssign: intermediate step with out-of-bounds list index raises AglRaise."""
        from agm.agl.semantics.exceptions import AglRaise

        # xss = [[1, 2]], then xss[5][0] := 99 — step 0 is OOB
        sym, desc = self._var(0, "xss")
        inner = IrMakeList(_LOC, (IrConstInt(_LOC, 1), IrConstInt(_LOC, 2)))
        prog = _make_program(
            (
                IrBind(_LOC, sym, IrMakeList(_LOC, (inner,))),
                IrAssign(
                    _LOC,
                    sym,
                    (
                        IrIndexStep(kind=IndexKind.LIST, index=IrConstInt(_LOC, 5), location=_LOC),
                        IrIndexStep(kind=IndexKind.LIST, index=IrConstInt(_LOC, 0), location=_LOC),
                    ),
                    IrConstInt(_LOC, 99),
                ),
            ),
            {sym: desc},
        )
        with pytest.raises(AglRaise) as exc:
            IrInterpreter(prog).run()
        assert exc.value.exc.display_name == "IndexError"

    def test_assign_path_intermediate_dict_missing_key_raises(self) -> None:
        """IrAssign: intermediate step with missing dict key raises AglRaise."""
        from agm.agl.semantics.exceptions import AglRaise

        # m = {"a": [1, 2]}, then m["z"]["a"] := 99 — step 0 key is missing
        sym, desc = self._var(0, "m")
        inner = IrMakeList(_LOC, (IrConstInt(_LOC, 1), IrConstInt(_LOC, 2)))
        prog = _make_program(
            (
                IrBind(_LOC, sym, IrMakeDict(_LOC, ((IrConstText(_LOC, "a"), inner),))),
                IrAssign(
                    _LOC,
                    sym,
                    (
                        IrIndexStep(
                            kind=IndexKind.DICT,
                            index=IrConstText(_LOC, "z"),
                            location=_LOC,
                        ),
                        IrIndexStep(kind=IndexKind.LIST, index=IrConstInt(_LOC, 0), location=_LOC),
                    ),
                    IrConstInt(_LOC, 99),
                ),
            ),
            {sym: desc},
        )
        with pytest.raises(AglRaise) as exc:
            IrInterpreter(prog).run()
        assert exc.value.exc.display_name == "KeyError"

    def test_assign_path_final_list_oob_raises(self) -> None:
        """IrAssign: final step with out-of-bounds list index raises AglRaise."""
        from agm.agl.semantics.exceptions import AglRaise

        # xs = [1, 2], then xs[5] := 99 — final step is OOB
        sym, desc = self._var(0, "xs")
        prog = _make_program(
            (
                IrBind(_LOC, sym, IrMakeList(_LOC, (IrConstInt(_LOC, 1), IrConstInt(_LOC, 2)))),
                IrAssign(
                    _LOC,
                    sym,
                    (IrIndexStep(kind=IndexKind.LIST, index=IrConstInt(_LOC, 5), location=_LOC),),
                    IrConstInt(_LOC, 99),
                ),
            ),
            {sym: desc},
        )
        with pytest.raises(AglRaise) as exc:
            IrInterpreter(prog).run()
        assert exc.value.exc.display_name == "IndexError"

    def test_assign_path_final_dict_missing_key_raises(self) -> None:
        """IrAssign: final step with missing dict key raises AglRaise."""
        from agm.agl.semantics.exceptions import AglRaise

        # m = {"a": 1}, then m["z"] := 99 — final step key is missing
        sym, desc = self._var(0, "m")
        prog = _make_program(
            (
                IrBind(
                    _LOC,
                    sym,
                    IrMakeDict(_LOC, ((IrConstText(_LOC, "a"), IrConstInt(_LOC, 1)),)),
                ),
                IrAssign(
                    _LOC,
                    sym,
                    (
                        IrIndexStep(
                            kind=IndexKind.DICT,
                            index=IrConstText(_LOC, "z"),
                            location=_LOC,
                        ),
                    ),
                    IrConstInt(_LOC, 99),
                ),
            ),
            {sym: desc},
        )
        with pytest.raises(AglRaise) as exc:
            IrInterpreter(prog).run()
        assert exc.value.exc.display_name == "KeyError"


# ---------------------------------------------------------------------------
# Import isolation: IrInterpreter must not import syntax/scope/typecheck
# ---------------------------------------------------------------------------


def _collect_import_names(source: str) -> set[str]:
    """Return all module names referenced in top-level import statements.

    Parses the source with ``ast`` and collects:
    - ``import agm.agl.foo`` → ``"agm.agl.foo"``
    - ``from agm.agl.foo import bar`` → ``"agm.agl.foo"``

    TYPE_CHECKING guards are NOT executed at runtime, but we scan ALL imports
    so the check is conservative (catches even guarded ones).
    """
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.add(node.module)
    return names


def _ir_interpreter_source() -> str:
    """Return the source text of ``agm.agl.eval.ir_interpreter``."""
    spec = importlib.util.find_spec("agm.agl.eval.ir_interpreter")
    assert spec is not None and spec.origin is not None
    return pathlib.Path(spec.origin).read_text(encoding="utf-8")


class TestImportIsolation:
    """Verify that ``ir_interpreter.py`` does not import forbidden packages.

    We parse the source file's import statements directly (AST-based) rather
    than probing ``sys.modules``, because ``agm.agl.eval.__init__`` re-exports
    unrelated frontend modules, so
    ``sys.modules`` would include those regardless of what ir_interpreter does.
    AST parsing checks what the file *itself* declares as a dependency.
    """

    def test_no_syntax_import(self) -> None:
        source = _ir_interpreter_source()
        imports = _collect_import_names(source)
        forbidden = {n for n in imports if n.startswith("agm.agl.syntax")}
        assert not forbidden, f"ir_interpreter.py imports syntax modules: {sorted(forbidden)}"

    def test_no_scope_import(self) -> None:
        source = _ir_interpreter_source()
        imports = _collect_import_names(source)
        # agm.agl.eval.scope (eval's own scope helpers) is allowed;
        # agm.agl.scope (the AST resolver) is forbidden.
        forbidden = {
            n
            for n in imports
            if n.startswith("agm.agl.scope") and not n.startswith("agm.agl.eval.scope")
        }
        assert not forbidden, f"ir_interpreter.py imports AST-scope modules: {sorted(forbidden)}"

    def test_no_typecheck_import(self) -> None:
        source = _ir_interpreter_source()
        imports = _collect_import_names(source)
        forbidden = {n for n in imports if n.startswith("agm.agl.typecheck")}
        assert not forbidden, f"ir_interpreter.py imports typecheck modules: {sorted(forbidden)}"


# ---------------------------------------------------------------------------
# IrMakeClosure and IrDirectCall evaluation + defensive error paths
# ---------------------------------------------------------------------------

_FN_SID = SymbolId(100)
_FN_ID = FunctionId(0)
_PARAM_SID = SymbolId(101)
_LOCAL_SID = SymbolId(102)


def _fn_sym_desc() -> SymbolDescriptor:
    return SymbolDescriptor(symbol_id=_FN_SID, mutable=False, public_name="f", owner=ENTRY_ID)


def _param_sym_desc() -> SymbolDescriptor:
    return SymbolDescriptor(symbol_id=_PARAM_SID, mutable=False, public_name="x", owner=_FN_ID)


def _make_fn_descriptor(
    body: IrExpr, params: "tuple[IrFunctionParam, ...]" = ()
) -> FunctionDescriptor:
    return FunctionDescriptor(
        function_id=_FN_ID,
        function_symbol=_FN_SID,
        module_id=ENTRY_ID,
        params=params,
        impl=IrFunctionBody(body=body),
    )


def _loc_at_line(line: int) -> Location:
    return Location(
        source_id=_SOURCE_ID,
        start_offset=line * 10,
        end_offset=line * 10 + 1,
        start_line=line,
        start_col=0,
    )


class TestFunctionEvaluation:
    """Tests for IrMakeClosure and IrDirectCall evaluation."""

    def test_simple_direct_call(self) -> None:
        """IrDirectCall evaluates a function returning a constant."""
        body = IrConstInt(_LOC, 42)
        fn_desc = _make_fn_descriptor(body)
        prog = _make_program(
            initializers=(
                IrBind(_LOC, _FN_SID, IrMakeClosure(_LOC, _FN_ID, ())),
                IrBind(_LOC, SymbolId(200), IrDirectCall(_LOC, _FN_ID, ())),
            ),
            symbols={
                _FN_SID: _fn_sym_desc(),
                SymbolId(200): SymbolDescriptor(
                    symbol_id=SymbolId(200), mutable=False, public_name="result", owner=ENTRY_ID
                ),
            },
            functions={_FN_ID: fn_desc},
        )
        result = IrInterpreter(prog).run()
        from agm.agl.semantics.values import IntValue

        assert result["result"] == IntValue(42)

    def test_param_default_can_call_top_level_function(self) -> None:
        """Entry param defaults can call functions whose closure initializer appears later."""
        from agm.agl.ir.program import IrParam

        param_sym = SymbolId(700)
        fn_desc = _make_fn_descriptor(IrConstInt(_LOC, 42))
        param = IrParam(
            symbol=param_sym,
            public_name="x",
            required=False,
            default=IrDirectCall(_LOC, _FN_ID, ()),
            location=_LOC,
        )
        prog = ExecutableProgram(
            entry_module=ENTRY_ID,
            modules={
                ENTRY_ID: ExecutableModule(
                    module_id=ENTRY_ID,
                    initializers=(IrBind(_LOC, _FN_SID, IrMakeClosure(_LOC, _FN_ID, ())),),
                )
            },
            symbols={
                _FN_SID: _fn_sym_desc(),
                param_sym: SymbolDescriptor(
                    symbol_id=param_sym,
                    mutable=False,
                    public_name="x",
                    owner=ENTRY_ID,
                ),
            },
            nominals={},
            sources={_SOURCE_ID: SourceFile(display_name="<test>", normalized_text="x")},
            functions={_FN_ID: fn_desc},
            params=(param,),
        )

        result = IrInterpreter(prog).run()

        assert result["x"] == IntValue(42)

    def test_agl_raise_preserves_inner_function_span(self) -> None:
        """Nested AglRaise failures keep the innermost IR node location."""
        from agm.agl.semantics.exceptions import AglRaise

        raise_loc = _loc_at_line(1)
        call_loc = _loc_at_line(3)
        fn_desc = _make_fn_descriptor(
            IrRaise(
                raise_loc,
                IrMakeException(
                    raise_loc,
                    NominalId(ENTRY_ID, "Abort"),
                    "Abort",
                    (("message", IrConstText(raise_loc, "boom")),),
                ),
            )
        )
        result_sym = SymbolId(701)
        prog = _make_program(
            initializers=(
                IrBind(raise_loc, _FN_SID, IrMakeClosure(raise_loc, _FN_ID, ())),
                IrBind(call_loc, result_sym, IrDirectCall(call_loc, _FN_ID, ())),
            ),
            symbols={
                _FN_SID: _fn_sym_desc(),
                result_sym: SymbolDescriptor(
                    symbol_id=result_sym,
                    mutable=False,
                    public_name="result",
                    owner=ENTRY_ID,
                ),
            },
            functions={_FN_ID: fn_desc},
        )

        with pytest.raises(AglRaise) as exc_info:
            IrInterpreter(prog).run()

        assert exc_info.value.span == raise_loc

    def test_uncaught_internal_error_reports_innermost_call_site(self) -> None:
        """A spanless internal error surfaces at its innermost enclosing call site.

        ``f`` calls ``g``; ``g``'s body triggers an out-of-range ``IndexError``,
        which ``_index_failure`` raises with no span of its own.  The
        ``IrDirectCall`` dispatch arm for the ``g(...)`` call inside ``f``
        backfills that span first, so it wins over the outer top-level
        ``f()`` call site — an uncaught internal error reports where it was
        invoked from, not where the top-level call happened.
        """
        from agm.agl.semantics.exceptions import AglRaise

        g_sid = SymbolId(710)
        g_id = FunctionId(1)
        g_body = IrIndex(
            _loc_at_line(20),
            IndexKind.LIST,
            IrMakeList(_LOC, (IrConstInt(_LOC, 1),)),
            IrConstInt(_LOC, 5),
        )
        g_desc = FunctionDescriptor(
            function_id=g_id,
            function_symbol=g_sid,
            module_id=ENTRY_ID,
            params=(),
            impl=IrFunctionBody(body=g_body),
        )

        g_call_loc = _loc_at_line(5)  # the g(...) call site inside f
        f_desc = _make_fn_descriptor(IrDirectCall(g_call_loc, g_id, ()))

        top_call_loc = _loc_at_line(1)
        result_sym = SymbolId(711)
        prog = _make_program(
            initializers=(
                IrBind(_LOC, g_sid, IrMakeClosure(_LOC, g_id, ())),
                IrBind(_LOC, _FN_SID, IrMakeClosure(_LOC, _FN_ID, ())),
                IrBind(top_call_loc, result_sym, IrDirectCall(top_call_loc, _FN_ID, ())),
            ),
            symbols={
                g_sid: SymbolDescriptor(
                    symbol_id=g_sid, mutable=False, public_name="g", owner=ENTRY_ID
                ),
                _FN_SID: _fn_sym_desc(),
                result_sym: SymbolDescriptor(
                    symbol_id=result_sym, mutable=False, public_name="result", owner=ENTRY_ID
                ),
            },
            functions={_FN_ID: f_desc, g_id: g_desc},
        )

        with pytest.raises(AglRaise) as exc_info:
            IrInterpreter(prog).run()

        assert exc_info.value.span == g_call_loc

    def test_direct_call_with_param(self) -> None:
        """IrDirectCall with an argument evaluates correctly."""
        param = IrFunctionParam(symbol=_PARAM_SID, default=None)
        body = IrLoad(_LOC, _PARAM_SID)  # return the param
        fn_desc = _make_fn_descriptor(body, params=(param,))
        result_sym = SymbolId(200)
        symbols = {
            _FN_SID: _fn_sym_desc(),
            _PARAM_SID: _param_sym_desc(),
            result_sym: SymbolDescriptor(
                symbol_id=result_sym, mutable=False, public_name="result", owner=ENTRY_ID
            ),
        }
        prog = _make_program(
            initializers=(
                IrBind(_LOC, _FN_SID, IrMakeClosure(_LOC, _FN_ID, ())),
                IrBind(_LOC, result_sym, IrDirectCall(_LOC, _FN_ID, (IrConstInt(_LOC, 7),))),
            ),
            symbols=symbols,
            functions={_FN_ID: fn_desc},
        )
        result = IrInterpreter(prog).run()
        from agm.agl.semantics.values import IntValue

        assert result["result"] == IntValue(7)

    def test_direct_call_with_use_default(self) -> None:
        """IrDirectCall with UseDefault uses the default expression."""
        param = IrFunctionParam(symbol=_PARAM_SID, default=IrConstInt(_LOC, 99))
        body = IrLoad(_LOC, _PARAM_SID)
        fn_desc = _make_fn_descriptor(body, params=(param,))
        result_sym = SymbolId(200)
        symbols = {
            _FN_SID: _fn_sym_desc(),
            _PARAM_SID: _param_sym_desc(),
            result_sym: SymbolDescriptor(
                symbol_id=result_sym, mutable=False, public_name="result", owner=ENTRY_ID
            ),
        }
        prog = _make_program(
            initializers=(
                IrBind(_LOC, _FN_SID, IrMakeClosure(_LOC, _FN_ID, ())),
                IrBind(_LOC, result_sym, IrDirectCall(_LOC, _FN_ID, (UseDefault(param_index=0),))),
            ),
            symbols=symbols,
            functions={_FN_ID: fn_desc},
        )
        result = IrInterpreter(prog).run()
        from agm.agl.semantics.values import IntValue

        assert result["result"] == IntValue(99)

    def test_direct_call_param_bound_by_value(self) -> None:
        """IrDirectCall binds parameters by value regardless of the symbol's mutable flag.

        Params are always bound by value; a mutable SymbolDescriptor for a param
        is unusual IR but the evaluator must still bind by value and return the arg.
        """
        mutable_param_sym = SymbolId(150)
        mutable_param_desc = SymbolDescriptor(
            symbol_id=mutable_param_sym, mutable=True, public_name="x", owner=_FN_ID
        )
        param = IrFunctionParam(symbol=mutable_param_sym, default=None)
        body = IrLoad(_LOC, mutable_param_sym)
        fn_desc = _make_fn_descriptor(body, params=(param,))
        result_sym = SymbolId(201)
        symbols = {
            _FN_SID: _fn_sym_desc(),
            mutable_param_sym: mutable_param_desc,
            result_sym: SymbolDescriptor(
                symbol_id=result_sym, mutable=False, public_name="result2", owner=ENTRY_ID
            ),
        }
        prog = _make_program(
            initializers=(
                IrBind(_LOC, _FN_SID, IrMakeClosure(_LOC, _FN_ID, ())),
                IrBind(_LOC, result_sym, IrDirectCall(_LOC, _FN_ID, (IrConstInt(_LOC, 55),))),
            ),
            symbols=symbols,
            functions={_FN_ID: fn_desc},
        )
        result = IrInterpreter(prog).run()
        from agm.agl.semantics.values import IntValue

        assert result["result2"] == IntValue(55)


class TestDirectCallInterpreterDefensivePaths:
    """Defensive error paths in IrMakeClosure and IrDirectCall evaluation."""

    def test_get_closure_for_symbol_not_in_frame_raises(self) -> None:
        """_get_closure_for raises InvalidIrError when function symbol not bound."""
        body = IrConstInt(_LOC, 0)
        fn_desc = _make_fn_descriptor(body)
        # IrDirectCall without binding the closure first — symbol not in frame
        result_sym = SymbolId(200)
        symbols = {
            _FN_SID: _fn_sym_desc(),
            result_sym: SymbolDescriptor(
                symbol_id=result_sym, mutable=False, public_name="result", owner=ENTRY_ID
            ),
        }
        prog = _make_program(
            initializers=(
                # No IrBind for _FN_SID — so slot will be None when IrDirectCall executes
                IrBind(_LOC, result_sym, IrDirectCall(_LOC, _FN_ID, ())),
            ),
            symbols=symbols,
            functions={_FN_ID: fn_desc},
        )
        with pytest.raises(InvalidIrError, match="not in base frame"):
            IrInterpreter(prog).run()

    def test_get_closure_for_slot_not_ir_closure_value_raises(self) -> None:
        """_get_closure_for raises InvalidIrError when slot holds a non-IrClosureValue."""
        body = IrConstInt(_LOC, 0)
        fn_desc = _make_fn_descriptor(body)
        result_sym = SymbolId(200)
        symbols = {
            _FN_SID: _fn_sym_desc(),
            result_sym: SymbolDescriptor(
                symbol_id=result_sym, mutable=False, public_name="result", owner=ENTRY_ID
            ),
        }
        prog = _make_program(
            initializers=(
                # Bind an IntValue (not IrClosureValue) into the function symbol slot
                IrBind(_LOC, _FN_SID, IrConstInt(_LOC, 42)),
                IrBind(_LOC, result_sym, IrDirectCall(_LOC, _FN_ID, ())),
            ),
            symbols=symbols,
            functions={_FN_ID: fn_desc},
        )
        with pytest.raises(InvalidIrError, match="not IrClosureValue"):
            IrInterpreter(prog).run()

    def test_ir_make_closure_capture_not_in_frame_raises(self) -> None:
        """IrMakeClosure raises InvalidIrError when a capture symbol is not in frame."""
        missing_sym = SymbolId(999)
        bad_cap = IrCapture(symbol=missing_sym, by_cell=False)
        body = IrConstInt(_LOC, 0)
        fn_desc = _make_fn_descriptor(body)
        symbols = {_FN_SID: _fn_sym_desc()}
        prog = _make_program(
            initializers=(
                # Try to close over missing_sym which is not bound
                IrBind(_LOC, _FN_SID, IrMakeClosure(_LOC, _FN_ID, (bad_cap,))),
            ),
            symbols=symbols,
            functions={_FN_ID: fn_desc},
        )
        with pytest.raises(InvalidIrError, match="not in frame"):
            IrInterpreter(prog).run()

    def test_ir_make_closure_by_cell_not_cell_raises(self) -> None:
        """IrMakeClosure with by_cell=True raises when slot is not a Cell."""
        # Bind an immutable (non-cell) symbol and try to capture it by_cell
        cap_sym = SymbolId(200)
        bad_cap = IrCapture(symbol=cap_sym, by_cell=True)  # by_cell but cap_sym is immutable
        body = IrConstInt(_LOC, 0)
        fn_desc = _make_fn_descriptor(body)
        symbols = {
            _FN_SID: _fn_sym_desc(),
            cap_sym: SymbolDescriptor(
                symbol_id=cap_sym, mutable=False, public_name="c", owner=ENTRY_ID
            ),
        }
        prog = _make_program(
            initializers=(
                IrBind(_LOC, cap_sym, IrConstInt(_LOC, 5)),  # immutable, not a Cell
                IrBind(_LOC, _FN_SID, IrMakeClosure(_LOC, _FN_ID, (bad_cap,))),
            ),
            symbols=symbols,
            functions={_FN_ID: fn_desc},
        )
        with pytest.raises(InvalidIrError, match="not Cell"):
            IrInterpreter(prog).run()


# ---------------------------------------------------------------------------
# IrIndirectCall evaluation + defensive error paths
# ---------------------------------------------------------------------------


class TestIndirectCallInterpreterDefensivePaths:
    """Defensive error paths in IrIndirectCall evaluation."""

    def test_indirect_call_non_closure_callee_raises(self) -> None:
        """IrIndirectCall raises InvalidIrError when callee evaluates to a non-closure."""
        # Build: let val_sym = 42; let result_sym = IrIndirectCall(IrLoad(val_sym), ())
        val_sym = SymbolId(300)
        result_sym = SymbolId(301)
        body = IrConstInt(_LOC, 0)
        fn_desc = _make_fn_descriptor(body)
        symbols = {
            _FN_SID: _fn_sym_desc(),
            val_sym: SymbolDescriptor(
                symbol_id=val_sym, mutable=False, public_name="val", owner=ENTRY_ID
            ),
            result_sym: SymbolDescriptor(
                symbol_id=result_sym, mutable=False, public_name="result", owner=ENTRY_ID
            ),
        }
        prog = _make_program(
            initializers=(
                IrBind(_LOC, val_sym, IrConstInt(_LOC, 42)),
                # Try to call an int as a function — callee is not IrClosureValue
                IrBind(_LOC, result_sym, IrIndirectCall(_LOC, IrLoad(_LOC, val_sym), ())),
            ),
            symbols=symbols,
            functions={_FN_ID: fn_desc},
        )
        with pytest.raises(InvalidIrError, match="expected IrClosureValue"):
            IrInterpreter(prog).run()

    def test_indirect_call_depth_limit_raises(self) -> None:
        """IrIndirectCall with max_call_depth=1 raises AglRaise(RecursionError) on reentry."""
        from agm.agl.semantics.exceptions import AglRaise

        # Build: def f() = f(); let result = f()
        # f() calls itself via IrIndirectCall to test the depth guard.
        fn2_id = FunctionId(1)
        fn2_sid = SymbolId(400)
        fn2_param_sym = SymbolId(401)
        result_sym = SymbolId(402)

        # Body: call fn2 indirectly — IrLoad(fn2_sid) gives the closure, then call it
        body = IrIndirectCall(_LOC, IrLoad(_LOC, fn2_sid), ())
        fn2_desc = FunctionDescriptor(
            function_id=fn2_id,
            function_symbol=fn2_sid,
            module_id=ENTRY_ID,
            params=(),
            impl=IrFunctionBody(body=body),
        )
        symbols = {
            fn2_sid: SymbolDescriptor(
                symbol_id=fn2_sid, mutable=False, public_name="f2", owner=ENTRY_ID
            ),
            fn2_param_sym: SymbolDescriptor(
                symbol_id=fn2_param_sym, mutable=False, public_name="p2", owner=fn2_id
            ),
            result_sym: SymbolDescriptor(
                symbol_id=result_sym, mutable=False, public_name="result2", owner=ENTRY_ID
            ),
        }
        prog = _make_program(
            initializers=(
                IrBind(_LOC, fn2_sid, IrMakeClosure(_LOC, fn2_id, ())),
                IrBind(_LOC, result_sym, IrIndirectCall(_LOC, IrLoad(_LOC, fn2_sid), ())),
            ),
            symbols=symbols,
            functions={fn2_id: fn2_desc},
        )
        with pytest.raises(AglRaise) as exc:
            IrInterpreter(prog, max_call_depth=1).run()
        assert exc.value.exc.display_name == "RecursionError"

    def test_indirect_call_uses_param_default_when_arg_omitted(self) -> None:
        """IrIndirectCall falls back to param.default when fewer args than params."""
        # Build a function with a param that has a default; call it with zero args.
        param = IrFunctionParam(symbol=_PARAM_SID, default=IrConstInt(_LOC, 77))
        body = IrLoad(_LOC, _PARAM_SID)
        fn_desc = _make_fn_descriptor(body, params=(param,))
        result_sym = SymbolId(500)
        fn_closure_sym = SymbolId(501)
        symbols = {
            _FN_SID: _fn_sym_desc(),
            _PARAM_SID: _param_sym_desc(),
            fn_closure_sym: SymbolDescriptor(
                symbol_id=fn_closure_sym, mutable=False, public_name="fn_ref", owner=ENTRY_ID
            ),
            result_sym: SymbolDescriptor(
                symbol_id=result_sym, mutable=False, public_name="r", owner=ENTRY_ID
            ),
        }
        prog = _make_program(
            initializers=(
                IrBind(_LOC, _FN_SID, IrMakeClosure(_LOC, _FN_ID, ())),
                # fn_closure_sym holds the closure value; call it via indirect call with 0 args
                IrBind(_LOC, fn_closure_sym, IrLoad(_LOC, _FN_SID)),
                IrBind(_LOC, result_sym, IrIndirectCall(_LOC, IrLoad(_LOC, fn_closure_sym), ())),
            ),
            symbols=symbols,
            functions={_FN_ID: fn_desc},
        )
        result = IrInterpreter(prog).run()
        assert result["r"] == IntValue(77)

    def test_indirect_call_missing_arg_no_default_raises(self) -> None:
        """IrIndirectCall raises InvalidIrError when arg is missing and no default exists."""
        # Function has a required param (no default); call it with zero args.
        param = IrFunctionParam(symbol=_PARAM_SID, default=None)
        body = IrLoad(_LOC, _PARAM_SID)
        fn_desc = _make_fn_descriptor(body, params=(param,))
        result_sym = SymbolId(600)
        fn_closure_sym = SymbolId(601)
        symbols = {
            _FN_SID: _fn_sym_desc(),
            _PARAM_SID: _param_sym_desc(),
            fn_closure_sym: SymbolDescriptor(
                symbol_id=fn_closure_sym, mutable=False, public_name="fn_ref2", owner=ENTRY_ID
            ),
            result_sym: SymbolDescriptor(
                symbol_id=result_sym, mutable=False, public_name="r2", owner=ENTRY_ID
            ),
        }
        prog = _make_program(
            initializers=(
                IrBind(_LOC, _FN_SID, IrMakeClosure(_LOC, _FN_ID, ())),
                IrBind(_LOC, fn_closure_sym, IrLoad(_LOC, _FN_SID)),
                # Call with 0 args but fn expects 1 required param — should raise
                IrBind(_LOC, result_sym, IrIndirectCall(_LOC, IrLoad(_LOC, fn_closure_sym), ())),
            ),
            symbols=symbols,
            functions={_FN_ID: fn_desc},
        )
        with pytest.raises(InvalidIrError, match="missing argument"):
            IrInterpreter(prog).run()


# ===========================================================================
# IrPrint / IrParseJson / IrParam evaluator tests
# ===========================================================================


class TestPrintParseJsonParam:
    """Unit tests for host operations in the IrInterpreter."""

    def test_required_param_without_value_raises_invalid_ir_error(self) -> None:
        """run() raises InvalidIrError when a required param has no value supplied."""
        from agm.agl.ir.program import IrParam

        sym, desc = _let_sym(0, "n")
        p = IrParam(
            symbol=sym,
            public_name="n",
            required=True,
            default=None,
            location=_LOC,
        )
        prog = ExecutableProgram(
            entry_module=ENTRY_ID,
            modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=())},
            symbols={sym: desc},
            nominals={},
            sources={_SOURCE_ID: SourceFile(display_name="<test>", normalized_text="n")},
            functions={},
            params=(p,),
        )
        # No param_values provided — the required param has no value
        with pytest.raises(InvalidIrError, match="n"):
            IrInterpreter(prog).run()

    def test_ir_parse_json_non_text_value_raises_invalid_ir_error(self) -> None:
        """IrParseJson with a non-TextValue argument raises InvalidIrError (bad IR)."""
        from agm.agl.ir.nodes import IrParseJson

        # Construct a program where parse_json is called on a bool (bad IR)
        sym, desc = _let_sym(0, "r")
        node = IrBind(
            _LOC,
            sym,
            IrParseJson(_LOC, IrConstBool(_LOC, True)),  # bool is not TextValue
        )
        prog = _make_program(initializers=(node,), symbols={sym: desc})
        with pytest.raises(InvalidIrError, match="IrParseJson"):
            IrInterpreter(prog).run()

    def test_ir_print_returns_void(self, capsys: pytest.CaptureFixture[str]) -> None:
        """IrPrint writes its argument and yields unprintable unit."""
        node = IrPrint(_LOC, IrConstUnit(_LOC))
        prog = _make_program(initializers=(node,))

        interp = IrInterpreter(prog)
        interp.run()

        assert capsys.readouterr().out == "()\n"
        assert interp.initializer_values == [VOID_VALUE]

    def test_ir_render_non_bool_option_raises_invalid_ir_error(self) -> None:
        """IrRenderValue with a non-bool option raises InvalidIrError (bad IR)."""
        from agm.agl.ir.nodes import IrRenderValue

        sym, desc = _let_sym(0, "r")
        node = IrBind(
            _LOC,
            sym,
            IrRenderValue(
                _LOC,
                IrConstText(_LOC, "x"),
                pretty=IrConstText(_LOC, "yes"),
            ),
        )
        prog = _make_program(initializers=(node,), symbols={sym: desc})
        with pytest.raises(InvalidIrError, match="pretty"):
            IrInterpreter(prog).run()


# ===========================================================================
# IrExec evaluator unit tests
# ===========================================================================


class TestIrExec:
    """Unit tests for _eval_ir_exec in IrInterpreter."""

    def _make_exec_program(
        self,
        command: "IrExpr",
        *,
        codec_name: str = "text",
        structured_exec: bool = False,
        max_attempts: int = 1,
        is_unit: bool = False,
        location: Location = _LOC,
    ) -> ExecutableProgram:
        """Build a minimal program with a single IrExec initializer."""
        from agm.agl.ir.contracts import ContractRequest
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.nodes import IrExec

        cid = ContractId(value=0)
        contract = ContractRequest(
            codec_name=codec_name,
            strict_json=None,
            json_schema=None,
            decode=None,
            target_type_label="text",
            structured_exec=structured_exec,
            format_instructions="",
            is_unit=is_unit,
        )
        node = IrExec(
            location=location,
            command=command,
            contract_id=cid,
            max_attempts=max_attempts,
        )
        return ExecutableProgram(
            entry_module=ENTRY_ID,
            modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
            symbols={},
            nominals={},
            sources={_SOURCE_ID: SourceFile(display_name="<test>", normalized_text="x")},
            functions={},
            contracts={cid: contract},
        )

    def test_ir_exec_non_text_command_renders_via_render_value(self) -> None:
        """IrExec with a non-TextValue command renders via render_value."""
        import unittest.mock

        from agm.core.process import ProcessCaptureResult

        fake_result = ProcessCaptureResult(
            returncode=0,
            stdout="hello\n",
            stderr="",
            elapsed=0.01,
            timed_out=False,
            spawn_error=None,
            spawn_errno=None,
        )
        # Use a bool command (non-text) — render_value("True") is "True"
        prog = self._make_exec_program(IrConstBool(_LOC, True))
        with unittest.mock.patch(
            "agm.core.process.run_capture_result",
            return_value=fake_result,
        ) as mock_rcr:
            result = IrInterpreter(prog).run()
        # Verify that the command passed to shell was "True"
        call_args = mock_rcr.call_args
        assert call_args[0][0] == ["sh", "-c", "true"]
        assert result == {}  # no named bindings

    def test_ir_exec_retry_spawn_error_raises_exec_error(self) -> None:
        """On retry, spawn_error in subsequent shell call raises ExecError."""
        import unittest.mock

        from agm.agl.semantics.exceptions import AglRaise
        from agm.core.process import ProcessCaptureResult

        call_count = [0]

        def fake_rcr(
            args: list[str],
            *,
            idle_timeout: float | None = None,
            isolate_process_group: bool = False,
        ) -> ProcessCaptureResult:
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: succeeds but returns invalid JSON (triggers retry)
                return ProcessCaptureResult(
                    returncode=0,
                    stdout="not_valid_json\n",
                    stderr="",
                    elapsed=0.01,
                    timed_out=False,
                    spawn_error=None,
                    spawn_errno=None,
                )
            # Second call: spawn error
            return ProcessCaptureResult(
                returncode=None,
                stdout="",
                stderr="",
                elapsed=0.0,
                timed_out=False,
                spawn_error="No such file or directory",
                spawn_errno=2,
            )

        # Use int codec so parse fails and retry triggers
        import json

        from agm.agl.ir.contracts import ContractRequest
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.nodes import IrExec
        from agm.agl.semantics.type_table import create_seeded_type_table
        from agm.agl.semantics.types import IntType
        from agm.agl.type_schema import build_decode_schema

        cid = ContractId(value=0)
        decode_schema = build_decode_schema(IntType(), create_seeded_type_table()).root
        contract = ContractRequest(
            codec_name="json",
            strict_json=False,
            json_schema=json.dumps({"type": "integer"}),
            decode=decode_schema,
            target_type_label="int",
            structured_exec=False,
            format_instructions="",
            is_unit=False,
        )
        node = IrExec(
            location=_LOC,
            command=IrConstText(_LOC, "dummy"),
            contract_id=cid,
            max_attempts=2,
        )
        prog = ExecutableProgram(
            entry_module=ENTRY_ID,
            modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
            symbols={},
            nominals={},
            sources={_SOURCE_ID: SourceFile(display_name="<test>", normalized_text="x")},
            functions={},
            contracts={cid: contract},
        )
        with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=fake_rcr):
            with pytest.raises(AglRaise) as exc_info:
                IrInterpreter(prog).run()
        assert exc_info.value.exc.display_name == "ExecError"

    def test_ir_exec_spawn_error_records_exec_trace(self, tmp_path: pathlib.Path) -> None:
        """A shell spawn failure still records the attempted exec command."""
        import json
        import unittest.mock

        from agm.agl.runtime.trace import TraceStore
        from agm.agl.semantics.exceptions import AglRaise
        from agm.core.process import ProcessCaptureResult

        fake_result = ProcessCaptureResult(
            returncode=None,
            stdout="",
            stderr="",
            elapsed=0.0,
            timed_out=False,
            spawn_error="No such file or directory",
            spawn_errno=2,
        )
        prog = self._make_exec_program(IrConstText(_LOC, "dummy"))
        trace_path = tmp_path / "trace.jsonl"
        trace = TraceStore(path=trace_path)
        with unittest.mock.patch(
            "agm.core.process.run_capture_result",
            return_value=fake_result,
        ):
            with pytest.raises(AglRaise) as exc_info:
                IrInterpreter(prog, trace=trace).run()

        records = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
        exec_records = [rec for rec in records if rec["kind"] == "exec_command"]
        assert len(exec_records) == 1
        record = exec_records[0]
        assert record["command"] == "dummy"
        assert record["exit_code"] == -1
        assert record["stderr"] == "No such file or directory"
        assert exc_info.value.exc.fields["trace_id"].value == record["trace_id"]

    def test_ir_exec_preserves_command_expression_raise_span(self) -> None:
        """IrExec does not overwrite a span from its command expression."""
        from agm.agl.semantics.exceptions import AglRaise

        command_loc = _loc_at_line(4)
        exec_loc = _loc_at_line(8)
        command = IrRaise(
            command_loc,
            IrMakeException(
                command_loc,
                NominalId(ENTRY_ID, "Abort"),
                "Abort",
                (("message", IrConstText(command_loc, "boom")),),
            ),
        )
        prog = self._make_exec_program(command, location=exec_loc)

        with pytest.raises(AglRaise) as exc_info:
            IrInterpreter(prog).run()

        assert exc_info.value.span == command_loc

    def test_ir_exec_retry_timeout_raises_exec_error(self) -> None:
        """On retry, timeout in subsequent shell call raises ExecError."""
        import json
        import unittest.mock

        from agm.agl.ir.contracts import ContractRequest
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.nodes import IrExec
        from agm.agl.semantics.exceptions import AglRaise
        from agm.agl.semantics.type_table import create_seeded_type_table
        from agm.agl.semantics.types import IntType
        from agm.agl.type_schema import build_decode_schema
        from agm.core.process import ProcessCaptureResult

        call_count = [0]

        def fake_rcr(
            args: list[str],
            *,
            idle_timeout: float | None = None,
            isolate_process_group: bool = False,
        ) -> ProcessCaptureResult:
            call_count[0] += 1
            if call_count[0] == 1:
                return ProcessCaptureResult(
                    returncode=0,
                    stdout="not_valid_json\n",
                    stderr="",
                    elapsed=0.01,
                    timed_out=False,
                    spawn_error=None,
                    spawn_errno=None,
                )
            # Second call: timeout
            return ProcessCaptureResult(
                returncode=-1,
                stdout="",
                stderr="",
                elapsed=5.0,
                timed_out=True,
                spawn_error=None,
                spawn_errno=None,
            )

        cid = ContractId(value=0)
        decode_schema = build_decode_schema(IntType(), create_seeded_type_table()).root
        contract = ContractRequest(
            codec_name="json",
            strict_json=False,
            json_schema=json.dumps({"type": "integer"}),
            decode=decode_schema,
            target_type_label="int",
            structured_exec=False,
            format_instructions="",
            is_unit=False,
        )
        node = IrExec(
            location=_LOC,
            command=IrConstText(_LOC, "dummy"),
            contract_id=cid,
            max_attempts=2,
        )
        prog = ExecutableProgram(
            entry_module=ENTRY_ID,
            modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
            symbols={},
            nominals={},
            sources={_SOURCE_ID: SourceFile(display_name="<test>", normalized_text="x")},
            functions={},
            contracts={cid: contract},
        )
        with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=fake_rcr):
            with pytest.raises(AglRaise) as exc_info:
                IrInterpreter(prog).run()
        from agm.agl.semantics.values import BoolValue

        assert exc_info.value.exc.display_name == "ExecError"
        assert exc_info.value.exc.fields["timed_out"] == BoolValue(True)

    def test_ir_exec_retry_nonzero_exit_raises_exec_error(self) -> None:
        """On retry, non-zero exit in subsequent shell call raises ExecError."""
        import json
        import unittest.mock

        from agm.agl.ir.contracts import ContractRequest
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.nodes import IrExec
        from agm.agl.semantics.exceptions import AglRaise
        from agm.agl.semantics.type_table import create_seeded_type_table
        from agm.agl.semantics.types import IntType
        from agm.agl.type_schema import build_decode_schema
        from agm.core.process import ProcessCaptureResult

        call_count = [0]

        def fake_rcr(
            args: list[str],
            *,
            idle_timeout: float | None = None,
            isolate_process_group: bool = False,
        ) -> ProcessCaptureResult:
            call_count[0] += 1
            if call_count[0] == 1:
                return ProcessCaptureResult(
                    returncode=0,
                    stdout="not_valid_json\n",
                    stderr="",
                    elapsed=0.01,
                    timed_out=False,
                    spawn_error=None,
                    spawn_errno=None,
                )
            # Second call: non-zero exit
            return ProcessCaptureResult(
                returncode=2,
                stdout="",
                stderr="error",
                elapsed=0.01,
                timed_out=False,
                spawn_error=None,
                spawn_errno=None,
            )

        cid = ContractId(value=0)
        decode_schema = build_decode_schema(IntType(), create_seeded_type_table()).root
        contract = ContractRequest(
            codec_name="json",
            strict_json=False,
            json_schema=json.dumps({"type": "integer"}),
            decode=decode_schema,
            target_type_label="int",
            structured_exec=False,
            format_instructions="",
            is_unit=False,
        )
        node = IrExec(
            location=_LOC,
            command=IrConstText(_LOC, "dummy"),
            contract_id=cid,
            max_attempts=2,
        )
        prog = ExecutableProgram(
            entry_module=ENTRY_ID,
            modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
            symbols={},
            nominals={},
            sources={_SOURCE_ID: SourceFile(display_name="<test>", normalized_text="x")},
            functions={},
            contracts={cid: contract},
        )
        with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=fake_rcr):
            with pytest.raises(AglRaise) as exc_info:
                IrInterpreter(prog).run()
        assert exc_info.value.exc.display_name == "ExecError"

    def test_ir_exec_structured_parse_errors_path(self) -> None:
        """IrExec JSON parse with structured errors populates last_errors."""
        import json
        import unittest.mock

        from agm.agl.ir.contracts import ContractRequest
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.nodes import IrExec
        from agm.agl.semantics.exceptions import AglRaise
        from agm.agl.semantics.type_table import create_seeded_type_table
        from agm.agl.semantics.types import IntType
        from agm.agl.type_schema import build_decode_schema
        from agm.core.process import ProcessCaptureResult

        # JSON parse of a string where int is expected yields structured errors
        def fake_rcr(
            args: list[str],
            *,
            idle_timeout: float | None = None,
            isolate_process_group: bool = False,
        ) -> ProcessCaptureResult:
            # Returns a valid JSON string (not int), so schema validation fails with errors
            return ProcessCaptureResult(
                returncode=0,
                stdout='"not_a_number"\n',
                stderr="",
                elapsed=0.01,
                timed_out=False,
                spawn_error=None,
                spawn_errno=None,
            )

        cid = ContractId(value=0)
        decode_schema = build_decode_schema(IntType(), create_seeded_type_table()).root
        contract = ContractRequest(
            codec_name="json",
            strict_json=True,
            json_schema=json.dumps({"type": "integer"}),
            decode=decode_schema,
            target_type_label="int",
            structured_exec=False,
            format_instructions="",
            is_unit=False,
        )
        node = IrExec(
            location=_LOC,
            command=IrConstText(_LOC, "dummy"),
            contract_id=cid,
            max_attempts=1,
        )
        prog = ExecutableProgram(
            entry_module=ENTRY_ID,
            modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
            symbols={},
            nominals={},
            sources={_SOURCE_ID: SourceFile(display_name="<test>", normalized_text="x")},
            functions={},
            contracts={cid: contract},
        )
        with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=fake_rcr):
            with pytest.raises(AglRaise) as exc_info:
                IrInterpreter(prog).run()
        assert exc_info.value.exc.display_name == "AgentParseError"
