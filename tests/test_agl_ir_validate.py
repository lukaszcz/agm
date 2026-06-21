"""Tests for agm.agl.ir.validate — structural IR validator (M1-C).

TDD: these tests were written before the implementation.  Each invariant from
the M1-C task spec has at least one failing case to exercise the error path.

Sections:
- Helpers / factories
- Valid program — passes both cheap and deep
- Cheap tier invariants (no tables needed)
- Deep tier invariants (cross-reference checks)
- deep=False skips cross-reference checks
- assert_never dispatch (closed IrExpr union)
"""

from __future__ import annotations

import decimal

import pytest

from agm.agl.ir import (
    CmpOp,
    CompareKind,
    ExecutableModule,
    ExecutableProgram,
    FunctionId,
    IntToDecimal,
    IrAssign,
    IrBind,
    IrBlock,
    IrCoerce,
    IrCompare,
    IrConstBool,
    IrConstDecimal,
    IrConstInt,
    IrConstJsonNull,
    IrConstText,
    IrConstUnit,
    IrIndexStep,
    IrLoad,
    IrMakeDict,
    IrMakeList,
    IrSequence,
    Location,
    NominalDescriptor,
    NominalId,
    NominalKind,
    SourceFile,
    SourceId,
    SymbolDescriptor,
    SymbolId,
)
from agm.agl.ir.validate import InvalidIrError, validate_ir
from agm.agl.modules.ids import ModuleId

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOD_A = ModuleId.from_dotted("mod_a")
MOD_B = ModuleId.from_dotted("mod_b")

SID0 = SourceId(value=0)
SID1 = SourceId(value=1)
SYM0 = SymbolId(value=0)
SYM1 = SymbolId(value=1)
SYM_MUT = SymbolId(value=2)
FN0 = FunctionId(value=0)
NOM0 = NominalId(module_id=MOD_A, declared_name="Foo")

SOURCE_TEXT = "let x = 1"  # 9 characters

# ---------------------------------------------------------------------------
# Location helpers
# ---------------------------------------------------------------------------


def loc(
    source_id: SourceId = SID0,
    start_offset: int = 0,
    end_offset: int = 5,
    start_line: int = 1,
    start_col: int = 0,
) -> Location:
    return Location(
        source_id=source_id,
        start_offset=start_offset,
        end_offset=end_offset,
        start_line=start_line,
        start_col=start_col,
    )


LOC = loc()

# ---------------------------------------------------------------------------
# Default descriptor helpers
# ---------------------------------------------------------------------------


def _sym_desc_imm() -> SymbolDescriptor:
    return SymbolDescriptor(symbol_id=SYM0, mutable=False, public_name="x", owner=MOD_A)


def _sym_desc_mut() -> SymbolDescriptor:
    return SymbolDescriptor(symbol_id=SYM_MUT, mutable=True, public_name="y", owner=MOD_A)


def _default_symbols() -> dict[SymbolId, SymbolDescriptor]:
    return {SYM0: _sym_desc_imm(), SYM_MUT: _sym_desc_mut()}


def _source_file(text: str = SOURCE_TEXT) -> SourceFile:
    return SourceFile(display_name="main.agl", normalized_text=text)


# ---------------------------------------------------------------------------
# Program factories
# ---------------------------------------------------------------------------


def _make_program(
    *,
    initializers: tuple = (),
    entry_module: ModuleId = MOD_A,
    modules: dict[ModuleId, ExecutableModule] | None = None,
    symbols: dict[SymbolId, SymbolDescriptor] | None = None,
    nominals: dict[NominalId, NominalDescriptor] | None = None,
    sources: dict[SourceId, SourceFile] | None = None,
) -> ExecutableProgram:
    """Build a valid base program; callers override individual tables."""
    nom_desc = NominalDescriptor(nominal=NOM0, display_name="Foo", kind=NominalKind.RECORD)
    sf = _source_file()
    em = ExecutableModule(module_id=MOD_A, initializers=initializers)
    return ExecutableProgram(
        entry_module=entry_module,
        modules={MOD_A: em} if modules is None else modules,
        symbols=_default_symbols() if symbols is None else symbols,
        nominals={NOM0: nom_desc} if nominals is None else nominals,
        sources={SID0: sf} if sources is None else sources,
    )


# ---------------------------------------------------------------------------
# Helper nodes that reference known-good IDs
# ---------------------------------------------------------------------------


def _load_sym0() -> IrLoad:
    return IrLoad(location=LOC, symbol=SYM0)


def _bind_sym0(value: IrConstInt | None = None) -> IrBind:
    return IrBind(
        location=LOC,
        symbol=SYM0,
        value=value if value is not None else IrConstInt(location=LOC, value=1),
    )


def _assign_sym_mut(value: IrConstInt | None = None) -> IrAssign:
    return IrAssign(
        location=LOC,
        symbol=SYM_MUT,
        path=(),
        value=value if value is not None else IrConstInt(location=LOC, value=42),
    )


def _int(v: int = 0) -> IrConstInt:
    return IrConstInt(location=LOC, value=v)


# ===========================================================================
# Valid program
# ===========================================================================


class TestValidProgram:
    """A well-formed program passes validate_ir in both tiers."""

    def test_empty_initializers(self) -> None:
        prog = _make_program()
        validate_ir(prog)  # no exception

    def test_with_various_nodes(self) -> None:
        nodes = (
            IrConstInt(location=LOC, value=1),
            IrConstDecimal(location=LOC, value=decimal.Decimal("3.14")),
            IrConstBool(location=LOC, value=True),
            IrConstText(location=LOC, value="hi"),
            IrConstUnit(location=LOC),
            IrConstJsonNull(location=LOC),
            IrMakeList(location=LOC, items=(IrConstInt(location=LOC, value=0),)),
            IrMakeDict(
                location=LOC,
                entries=((IrConstText(location=LOC, value="k"), _int(1)),),
            ),
            _load_sym0(),
            _bind_sym0(),
            _assign_sym_mut(),
            IrCoerce(location=LOC, value=_int(2), operation=IntToDecimal()),
            IrSequence(location=LOC, items=(IrConstUnit(location=LOC),)),
            IrBlock(location=LOC, items=(IrConstUnit(location=LOC),)),
        )
        prog = _make_program(initializers=nodes)
        validate_ir(prog)

    def test_deep_false_also_passes_valid(self) -> None:
        prog = _make_program(initializers=(_load_sym0(),))
        validate_ir(prog, deep=False)  # no exception

    def test_assign_with_index_path(self) -> None:
        idx_step = IrIndexStep(index=_int(0), location=LOC)
        assign = IrAssign(location=LOC, symbol=SYM_MUT, path=(idx_step,), value=_int(99))
        prog = _make_program(initializers=(assign,))
        validate_ir(prog)

    def test_nested_sequence_in_block(self) -> None:
        seq = IrSequence(
            location=LOC,
            items=(IrConstInt(location=LOC, value=1), IrConstBool(location=LOC, value=False)),
        )
        block = IrBlock(location=LOC, items=(seq,))
        prog = _make_program(initializers=(block,))
        validate_ir(prog)

    def test_coerce_with_nested_load(self) -> None:
        coerce = IrCoerce(location=LOC, value=_load_sym0(), operation=IntToDecimal())
        prog = _make_program(initializers=(coerce,))
        validate_ir(prog)


# ===========================================================================
# Cheap tier — location invariants
# ===========================================================================


class TestCheapTierLocation:
    """Location structural constraints, no table lookups."""

    def test_negative_start_offset(self) -> None:
        bad_loc = loc(start_offset=-1, end_offset=0)
        node = IrConstInt(location=bad_loc, value=1)
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(prog, deep=False)

    def test_start_gt_end_offset(self) -> None:
        bad_loc = loc(start_offset=5, end_offset=2)
        node = IrConstBool(location=bad_loc, value=True)
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(prog, deep=False)

    def test_start_line_less_than_one(self) -> None:
        bad_loc = loc(start_line=0)
        node = IrConstText(location=bad_loc, value="x")
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="start_line"):
            validate_ir(prog, deep=False)

    def test_negative_start_col(self) -> None:
        bad_loc = loc(start_col=-1)
        node = IrConstUnit(location=bad_loc)
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="start_col"):
            validate_ir(prog, deep=False)

    def test_index_step_bad_location(self) -> None:
        bad_loc = loc(start_offset=-5, end_offset=0)
        step = IrIndexStep(index=_int(0), location=bad_loc)
        assign = IrAssign(location=LOC, symbol=SYM_MUT, path=(step,), value=_int())
        prog = _make_program(initializers=(assign,))
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(prog, deep=False)

    def test_index_step_index_bad_location(self) -> None:
        bad_loc = loc(start_offset=10, end_offset=5)
        step = IrIndexStep(index=IrConstInt(location=bad_loc, value=0), location=LOC)
        assign = IrAssign(location=LOC, symbol=SYM_MUT, path=(step,), value=_int())
        prog = _make_program(initializers=(assign,))
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(prog, deep=False)


# ===========================================================================
# Cheap tier — non-empty sequence/block
# ===========================================================================


class TestCheapTierNonEmpty:
    def test_empty_sequence(self) -> None:
        node = IrSequence(location=LOC, items=())
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="IrSequence"):
            validate_ir(prog, deep=False)

    def test_empty_block(self) -> None:
        node = IrBlock(location=LOC, items=())
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="IrBlock"):
            validate_ir(prog, deep=False)

    def test_nonempty_sequence_passes(self) -> None:
        node = IrSequence(location=LOC, items=(IrConstUnit(location=LOC),))
        prog = _make_program(initializers=(node,))
        validate_ir(prog, deep=False)  # no exception

    def test_nonempty_block_passes(self) -> None:
        node = IrBlock(location=LOC, items=(IrConstUnit(location=LOC),))
        prog = _make_program(initializers=(node,))
        validate_ir(prog, deep=False)  # no exception


# ===========================================================================
# Deep tier — entry_module
# ===========================================================================


class TestDeepTierEntryModule:
    def test_entry_module_missing(self) -> None:
        # MOD_B not in modules (only MOD_A is)
        prog = _make_program(entry_module=MOD_B)
        with pytest.raises(InvalidIrError, match="entry_module"):
            validate_ir(prog)

    def test_entry_module_missing_skipped_when_shallow(self) -> None:
        prog = _make_program(entry_module=MOD_B)
        # cheap tier does NOT check entry_module — passes
        validate_ir(prog, deep=False)


# ===========================================================================
# Deep tier — module key/id consistency
# ===========================================================================


class TestDeepTierModuleConsistency:
    def test_module_id_key_mismatch(self) -> None:
        """ExecutableModule.module_id must equal its dict key."""
        em_bad = ExecutableModule(module_id=MOD_B, initializers=())
        prog = _make_program(modules={MOD_A: em_bad})
        with pytest.raises(InvalidIrError, match="module_id"):
            validate_ir(prog)

    def test_module_id_key_mismatch_skipped_when_shallow(self) -> None:
        em_bad = ExecutableModule(module_id=MOD_B, initializers=())
        prog = _make_program(modules={MOD_A: em_bad})
        validate_ir(prog, deep=False)


# ===========================================================================
# Deep tier — symbol descriptor consistency
# ===========================================================================


def _symbols_with_sym0(desc: SymbolDescriptor) -> dict[SymbolId, SymbolDescriptor]:
    """Return the default symbols table with SYM0 overridden by desc."""
    return {SYM0: desc, SYM_MUT: _sym_desc_mut()}


class TestDeepTierSymbolDescriptor:
    def test_symbol_id_key_mismatch(self) -> None:
        """SymbolDescriptor.symbol_id must equal its dict key."""
        # SYM1 stored under key SYM0 — mismatch
        bad = SymbolDescriptor(symbol_id=SYM1, mutable=False, public_name="z", owner=MOD_A)
        prog = _make_program(symbols=_symbols_with_sym0(bad))
        with pytest.raises(InvalidIrError, match="symbol_id"):
            validate_ir(prog)

    def test_symbol_owner_module_missing(self) -> None:
        """SymbolDescriptor.owner (when ModuleId) must exist in program.modules."""
        bad = SymbolDescriptor(
            symbol_id=SYM0, mutable=False, public_name="z", owner=MOD_B
        )
        prog = _make_program(symbols=_symbols_with_sym0(bad))
        with pytest.raises(InvalidIrError, match="owner"):
            validate_ir(prog)

    def test_symbol_owner_function_id_is_violation(self) -> None:
        """In M1 there is no functions table — FunctionId owner is a violation."""
        bad = SymbolDescriptor(
            symbol_id=SYM0, mutable=False, public_name="z", owner=FN0
        )
        prog = _make_program(symbols=_symbols_with_sym0(bad))
        with pytest.raises(InvalidIrError, match="FunctionId"):
            validate_ir(prog)

    def test_symbol_consistency_skipped_when_shallow(self) -> None:
        bad = SymbolDescriptor(symbol_id=SYM1, mutable=False, public_name="z", owner=MOD_A)
        prog = _make_program(symbols=_symbols_with_sym0(bad))
        validate_ir(prog, deep=False)


# ===========================================================================
# Deep tier — nominal descriptor consistency
# ===========================================================================


class TestDeepTierNominalDescriptor:
    def test_nominal_key_mismatch(self) -> None:
        """NominalDescriptor.nominal must equal its dict key."""
        nom_wrong = NominalId(module_id=MOD_A, declared_name="Bar")
        nom_desc = NominalDescriptor(
            nominal=nom_wrong,  # key will be NOM0 — mismatch
            display_name="Foo",
            kind=NominalKind.RECORD,
        )
        prog = _make_program(nominals={NOM0: nom_desc})
        with pytest.raises(InvalidIrError, match="nominal"):
            validate_ir(prog)

    def test_nominal_key_mismatch_skipped_when_shallow(self) -> None:
        nom_wrong = NominalId(module_id=MOD_A, declared_name="Bar")
        nom_desc = NominalDescriptor(
            nominal=nom_wrong, display_name="Foo", kind=NominalKind.RECORD
        )
        prog = _make_program(nominals={NOM0: nom_desc})
        validate_ir(prog, deep=False)


# ===========================================================================
# Deep tier — source key consistency
# ===========================================================================


class TestDeepTierSourceConsistency:
    def test_sources_are_accessible(self) -> None:
        """Basic: program with a source file passes validate_ir."""
        prog = _make_program()
        validate_ir(prog)  # no exception


# ===========================================================================
# Deep tier — IrLoad/IrBind/IrAssign symbol resolution
# ===========================================================================


class TestDeepTierSymbolResolution:
    def test_ir_load_dangling_symbol(self) -> None:
        """IrLoad referencing a SymbolId not in program.symbols raises InvalidIrError."""
        dangling = SymbolId(value=999)
        node = IrLoad(location=LOC, symbol=dangling)
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="999"):
            validate_ir(prog)

    def test_ir_load_dangling_skipped_cheap(self) -> None:
        """deep=False does not cross-check symbol references."""
        dangling = SymbolId(value=999)
        node = IrLoad(location=LOC, symbol=dangling)
        prog = _make_program(initializers=(node,))
        validate_ir(prog, deep=False)  # no exception

    def test_ir_bind_dangling_symbol(self) -> None:
        dangling = SymbolId(value=888)
        node = IrBind(location=LOC, symbol=dangling, value=_int(1))
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="888"):
            validate_ir(prog)

    def test_ir_bind_child_value_also_checked(self) -> None:
        """IrBind.value is recursively validated."""
        dangling = SymbolId(value=777)
        child = IrLoad(location=LOC, symbol=dangling)
        node = IrBind(location=LOC, symbol=SYM0, value=child)
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="777"):
            validate_ir(prog)

    def test_ir_assign_dangling_symbol(self) -> None:
        dangling = SymbolId(value=666)
        node = IrAssign(location=LOC, symbol=dangling, path=(), value=_int())
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="666"):
            validate_ir(prog)

    def test_ir_bind_deep_false_skips_symbol_check(self) -> None:
        """deep=False skips symbol resolution for IrBind but still recurses into value."""
        dangling = SymbolId(value=888)
        node = IrBind(location=LOC, symbol=dangling, value=IrConstUnit(location=LOC))
        prog = _make_program(initializers=(node,))
        validate_ir(prog, deep=False)  # dangling symbol not checked; no exception

    def test_ir_assign_dangling_skipped_cheap(self) -> None:
        dangling = SymbolId(value=666)
        node = IrAssign(location=LOC, symbol=dangling, path=(), value=_int())
        prog = _make_program(initializers=(node,))
        validate_ir(prog, deep=False)  # no exception


# ===========================================================================
# Deep tier — IrAssign mutability
# ===========================================================================


class TestDeepTierAssignMutability:
    def test_assign_to_immutable_raises(self) -> None:
        """Root symbol of IrAssign must be mutable (var)."""
        node = IrAssign(
            location=LOC,
            symbol=SYM0,  # SYM0 is immutable (let)
            path=(),
            value=_int(),
        )
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="mutable"):
            validate_ir(prog)

    def test_assign_to_mutable_passes(self) -> None:
        node = _assign_sym_mut()
        prog = _make_program(initializers=(node,))
        validate_ir(prog)

    def test_assign_mutability_skipped_cheap(self) -> None:
        node = IrAssign(
            location=LOC,
            symbol=SYM0,  # immutable, but deep=False won't catch it
            path=(),
            value=_int(),
        )
        prog = _make_program(initializers=(node,))
        validate_ir(prog, deep=False)  # no exception


# ===========================================================================
# Deep tier — Location source_id and offset range
# ===========================================================================


class TestDeepTierLocationSourceId:
    def test_location_source_id_missing(self) -> None:
        """Location.source_id must exist in program.sources."""
        bad_loc = loc(source_id=SID1)  # SID1 not in sources
        node = IrConstInt(location=bad_loc, value=0)
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="source_id"):
            validate_ir(prog)

    def test_location_source_id_missing_skipped_cheap(self) -> None:
        bad_loc = loc(source_id=SID1)
        node = IrConstInt(location=bad_loc, value=0)
        prog = _make_program(initializers=(node,))
        validate_ir(prog, deep=False)  # no exception

    def test_location_end_offset_beyond_source(self) -> None:
        """end_offset must be <= len(source.normalized_text)."""
        text = "abc"  # length 3
        sf = SourceFile(display_name="f.agl", normalized_text=text)
        bad_loc = loc(source_id=SID0, start_offset=0, end_offset=100)
        node = IrConstText(location=bad_loc, value="x")
        prog = _make_program(initializers=(node,), sources={SID0: sf})
        with pytest.raises(InvalidIrError, match="end_offset"):
            validate_ir(prog)

    def test_location_end_offset_at_boundary_passes(self) -> None:
        text = "abc"
        sf = SourceFile(display_name="f.agl", normalized_text=text)
        ok_loc = loc(source_id=SID0, start_offset=0, end_offset=3)
        node = IrConstText(location=ok_loc, value="x")
        prog = _make_program(initializers=(node,), sources={SID0: sf})
        validate_ir(prog)  # no exception

    def test_index_step_location_source_id_missing(self) -> None:
        """IrIndexStep.location source_id is also validated."""
        bad_loc = loc(source_id=SID1)
        step = IrIndexStep(index=_int(), location=bad_loc)
        assign = IrAssign(location=LOC, symbol=SYM_MUT, path=(step,), value=_int())
        prog = _make_program(initializers=(assign,))
        with pytest.raises(InvalidIrError, match="source_id"):
            validate_ir(prog)


# ===========================================================================
# Deep tier — multi-module programs
# ===========================================================================


class TestDeepTierMultiModule:
    def test_two_modules_valid(self) -> None:
        sf = _source_file()
        em_a = ExecutableModule(module_id=MOD_A, initializers=())
        em_b = ExecutableModule(module_id=MOD_B, initializers=())
        sym_desc = SymbolDescriptor(symbol_id=SYM0, mutable=False, public_name="x", owner=MOD_A)
        sym_mut = SymbolDescriptor(symbol_id=SYM_MUT, mutable=True, public_name="y", owner=MOD_B)
        prog = ExecutableProgram(
            entry_module=MOD_A,
            modules={MOD_A: em_a, MOD_B: em_b},
            symbols={SYM0: sym_desc, SYM_MUT: sym_mut},
            nominals={},
            sources={SID0: sf},
        )
        validate_ir(prog)

    def test_second_module_id_mismatch(self) -> None:
        em_a = ExecutableModule(module_id=MOD_A, initializers=())
        # Wrong module_id stored under MOD_B key:
        em_b_bad = ExecutableModule(module_id=MOD_A, initializers=())
        sf = _source_file()
        prog = ExecutableProgram(
            entry_module=MOD_A,
            modules={MOD_A: em_a, MOD_B: em_b_bad},
            symbols=_default_symbols(),
            nominals={},
            sources={SID0: sf},
        )
        with pytest.raises(InvalidIrError, match="module_id"):
            validate_ir(prog)


# ===========================================================================
# Recursive child traversal
# ===========================================================================


class TestChildTraversal:
    """Validator must recurse into child expressions to find violations."""

    def test_make_list_item_bad_location(self) -> None:
        bad_loc = loc(start_offset=-1, end_offset=0)
        item = IrConstInt(location=bad_loc, value=1)
        node = IrMakeList(location=LOC, items=(item,))
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError):
            validate_ir(prog, deep=False)

    def test_make_dict_key_bad_location(self) -> None:
        bad_loc = loc(start_line=0)
        key = IrConstText(location=bad_loc, value="k")
        val = _int(1)
        node = IrMakeDict(location=LOC, entries=((key, val),))
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError):
            validate_ir(prog, deep=False)

    def test_make_dict_value_bad_location(self) -> None:
        bad_loc = loc(start_col=-1)
        key = IrConstText(location=LOC, value="k")
        val = IrConstInt(location=bad_loc, value=1)
        node = IrMakeDict(location=LOC, entries=((key, val),))
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError):
            validate_ir(prog, deep=False)

    def test_sequence_item_bad_location(self) -> None:
        bad_loc = loc(start_offset=5, end_offset=2)
        child = IrConstBool(location=bad_loc, value=False)
        node = IrSequence(location=LOC, items=(child,))
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError):
            validate_ir(prog, deep=False)

    def test_block_item_bad_location(self) -> None:
        bad_loc = loc(start_line=0)
        child = IrConstText(location=bad_loc, value="x")
        node = IrBlock(location=LOC, items=(child,))
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError):
            validate_ir(prog, deep=False)

    def test_coerce_child_dangling_symbol(self) -> None:
        dangling = SymbolId(value=555)
        child = IrLoad(location=LOC, symbol=dangling)
        node = IrCoerce(location=LOC, value=child, operation=IntToDecimal())
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="555"):
            validate_ir(prog)

    def test_assign_value_dangling_symbol(self) -> None:
        dangling = SymbolId(value=444)
        child = IrLoad(location=LOC, symbol=dangling)
        node = IrAssign(location=LOC, symbol=SYM_MUT, path=(), value=child)
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="444"):
            validate_ir(prog)


# ===========================================================================
# InvalidIrError is exported from agm.agl.ir
# ===========================================================================


class TestInvalidIrErrorExport:
    def test_importable_from_package(self) -> None:
        from agm.agl.ir import InvalidIrError as _IIE

        assert issubclass(_IIE, Exception)

    def test_error_has_message(self) -> None:
        node = IrSequence(location=LOC, items=())
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError) as exc_info:
            validate_ir(prog)
        assert str(exc_info.value)  # message is non-empty


# ===========================================================================
# IrCompare: tightened EQ/NEQ ↔ STRUCTURAL constraints
# ===========================================================================


def _make_compare_program(op: CmpOp, kind: CompareKind) -> ExecutableProgram:
    """Build a minimal program containing a single IrCompare node."""
    node = IrCompare(
        location=LOC,
        op=op,
        kind=kind,
        lhs=IrConstInt(location=LOC, value=1),
        rhs=IrConstInt(location=LOC, value=2),
    )
    return _make_program(initializers=(node,))


class TestIrCompareTightenedConstraints:
    """EQ/NEQ ⇒ STRUCTURAL and ordering ops ⇒ non-STRUCTURAL."""

    # --- EQ with non-STRUCTURAL kind must be rejected ---

    def test_eq_with_int_kind_raises(self) -> None:
        """EQ requires STRUCTURAL kind; INT kind must raise."""
        prog = _make_compare_program(CmpOp.EQ, CompareKind.INT)
        with pytest.raises(InvalidIrError, match="EQ/NEQ requires STRUCTURAL"):
            validate_ir(prog, deep=False)

    def test_eq_with_decimal_kind_raises(self) -> None:
        """EQ requires STRUCTURAL kind; DECIMAL kind must raise."""
        prog = _make_compare_program(CmpOp.EQ, CompareKind.DECIMAL)
        with pytest.raises(InvalidIrError, match="EQ/NEQ requires STRUCTURAL"):
            validate_ir(prog, deep=False)

    def test_eq_with_text_kind_raises(self) -> None:
        """EQ requires STRUCTURAL kind; TEXT kind must raise."""
        prog = _make_compare_program(CmpOp.EQ, CompareKind.TEXT)
        with pytest.raises(InvalidIrError, match="EQ/NEQ requires STRUCTURAL"):
            validate_ir(prog, deep=False)

    def test_neq_with_int_kind_raises(self) -> None:
        """NEQ requires STRUCTURAL kind; INT kind must raise."""
        prog = _make_compare_program(CmpOp.NEQ, CompareKind.INT)
        with pytest.raises(InvalidIrError, match="EQ/NEQ requires STRUCTURAL"):
            validate_ir(prog, deep=False)

    def test_eq_with_structural_kind_passes(self) -> None:
        """EQ + STRUCTURAL is valid."""
        prog = _make_compare_program(CmpOp.EQ, CompareKind.STRUCTURAL)
        validate_ir(prog, deep=False)  # no exception

    def test_neq_with_structural_kind_passes(self) -> None:
        """NEQ + STRUCTURAL is valid."""
        prog = _make_compare_program(CmpOp.NEQ, CompareKind.STRUCTURAL)
        validate_ir(prog, deep=False)  # no exception

    # --- Ordering ops with STRUCTURAL kind must be rejected ---

    def test_lt_with_structural_kind_raises(self) -> None:
        """LT requires non-STRUCTURAL kind."""
        prog = _make_compare_program(CmpOp.LT, CompareKind.STRUCTURAL)
        with pytest.raises(InvalidIrError, match="ordering op.*STRUCTURAL"):
            validate_ir(prog, deep=False)

    def test_le_with_structural_kind_raises(self) -> None:
        """LE requires non-STRUCTURAL kind."""
        prog = _make_compare_program(CmpOp.LE, CompareKind.STRUCTURAL)
        with pytest.raises(InvalidIrError, match="ordering op.*STRUCTURAL"):
            validate_ir(prog, deep=False)

    def test_gt_with_structural_kind_raises(self) -> None:
        """GT requires non-STRUCTURAL kind."""
        prog = _make_compare_program(CmpOp.GT, CompareKind.STRUCTURAL)
        with pytest.raises(InvalidIrError, match="ordering op.*STRUCTURAL"):
            validate_ir(prog, deep=False)

    def test_ge_with_structural_kind_raises(self) -> None:
        """GE requires non-STRUCTURAL kind."""
        prog = _make_compare_program(CmpOp.GE, CompareKind.STRUCTURAL)
        with pytest.raises(InvalidIrError, match="ordering op.*STRUCTURAL"):
            validate_ir(prog, deep=False)

    def test_lt_with_int_kind_passes(self) -> None:
        """LT + INT is valid."""
        prog = _make_compare_program(CmpOp.LT, CompareKind.INT)
        validate_ir(prog, deep=False)  # no exception

    def test_gt_with_decimal_kind_passes(self) -> None:
        """GT + DECIMAL is valid."""
        prog = _make_compare_program(CmpOp.GT, CompareKind.DECIMAL)
        validate_ir(prog, deep=False)  # no exception

    def test_le_with_text_kind_passes(self) -> None:
        """LE + TEXT is valid."""
        prog = _make_compare_program(CmpOp.LE, CompareKind.TEXT)
        validate_ir(prog, deep=False)  # no exception
