"""Tests for agm.agl.ir.validate — structural IR validator.

TDD: these tests were written before the implementation.  Each invariant from
the C task spec has at least one failing case to exercise the error path.

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
    FunctionDescriptor,
    FunctionId,
    IndexKind,
    IntToDecimal,
    IrAssign,
    IrBind,
    IrBlock,
    IrCapture,
    IrCase,
    IrCaseArm,
    IrCoerce,
    IrCompare,
    IrConfigBind,
    IrConstBool,
    IrConstDecimal,
    IrConstInt,
    IrConstJsonNull,
    IrConstText,
    IrConstUnit,
    IrDirectCall,
    IrEnumCaseKey,
    IrField,
    IrFunctionBody,
    IrFunctionParam,
    IrIndex,
    IrIndexStep,
    IrIndirectCall,
    IrLiteralCaseKey,
    IrLiteralKind,
    IrLoad,
    IrMakeClosure,
    IrMakeDict,
    IrMakeList,
    IrRenderTemplate,
    IrSequence,
    IrTemplateText,
    IrTemplateValue,
    Location,
    NominalDescriptor,
    NominalId,
    NominalKind,
    SourceFile,
    SourceId,
    SymbolDescriptor,
    SymbolId,
    UseDefault,
    VariantDescriptor,
)
from agm.agl.ir.validate import InvalidIrError, validate_ir
from agm.agl.modules.ids import ModuleId
from tests.agl.ir_harness import _compiled_checked

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
    functions: "dict[FunctionId, FunctionDescriptor] | None" = None,
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
        functions=functions or {},
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
# IrCase invariants
# ===========================================================================


def test_case_arm_cannot_bind_multiple_fields_to_one_symbol() -> None:
    enum_nominal = NominalId(module_id=MOD_A, declared_name="Pair")
    program = _make_program(
        initializers=(
            IrCase(
                location=LOC,
                subject=IrConstInt(location=LOC, value=1),
                arms=(
                    IrCaseArm(
                        key=IrEnumCaseKey(nominal=enum_nominal, variant="Both"),
                        field_bindings=(("left", SYM1), ("right", SYM1)),
                        body=IrConstUnit(location=LOC),
                    ),
                ),
                default=IrConstUnit(location=LOC),
            ),
        ),
        symbols={SYM1: SymbolDescriptor(SYM1, mutable=False, public_name=None, owner=MOD_A)},
        nominals={
            enum_nominal: NominalDescriptor(
                nominal=enum_nominal,
                display_name="Pair",
                kind=NominalKind.ENUM,
                variants=(VariantDescriptor("Both", ("left", "right")),),
            )
        },
    )

    with pytest.raises(InvalidIrError, match="symbol"):
        validate_ir(program)


def test_case_without_default_requires_complete_boolean_domain() -> None:
    program = _make_program(
        initializers=(
            IrCase(
                location=LOC,
                subject=IrConstBool(location=LOC, value=True),
                arms=(
                    IrCaseArm(
                        key=IrLiteralCaseKey(IrLiteralKind.BOOL, True),
                        field_bindings=(),
                        body=IrConstUnit(location=LOC),
                    ),
                ),
                default=None,
            ),
        )
    )

    with pytest.raises(InvalidIrError, match="default"):
        validate_ir(program)


def test_case_without_default_requires_complete_enum_domain() -> None:
    enum_nominal = NominalId(module_id=MOD_A, declared_name="Result")
    program = _make_program(
        initializers=(
            IrCase(
                location=LOC,
                subject=IrConstInt(location=LOC, value=1),
                arms=(
                    IrCaseArm(
                        key=IrEnumCaseKey(nominal=enum_nominal, variant="Ok"),
                        field_bindings=(),
                        body=IrConstUnit(location=LOC),
                    ),
                ),
                default=None,
            ),
        ),
        nominals={
            enum_nominal: NominalDescriptor(
                nominal=enum_nominal,
                display_name="Result",
                kind=NominalKind.ENUM,
                variants=(VariantDescriptor("Ok", ()), VariantDescriptor("Error", ())),
            )
        },
    )

    with pytest.raises(InvalidIrError, match="default"):
        validate_ir(program)


def test_case_without_default_allows_complete_enum_domain() -> None:
    enum_nominal = NominalId(module_id=MOD_A, declared_name="Result")
    program = _make_program(
        initializers=(
            IrCase(
                location=LOC,
                subject=IrConstInt(location=LOC, value=1),
                arms=(
                    IrCaseArm(
                        key=IrEnumCaseKey(nominal=enum_nominal, variant="Ok"),
                        field_bindings=(),
                        body=IrConstUnit(location=LOC),
                    ),
                    IrCaseArm(
                        key=IrEnumCaseKey(nominal=enum_nominal, variant="Error"),
                        field_bindings=(),
                        body=IrConstUnit(location=LOC),
                    ),
                ),
                default=None,
            ),
        ),
        nominals={
            enum_nominal: NominalDescriptor(
                nominal=enum_nominal,
                display_name="Result",
                kind=NominalKind.ENUM,
                variants=(VariantDescriptor("Ok", ()), VariantDescriptor("Error", ())),
            )
        },
    )

    validate_ir(program)


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

    def test_lowered_partial_application_program_passes_deep_validation(self) -> None:
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.lower import lower_program
        from agm.agl.parser import parse_program
        from agm.agl.scope import resolve
        from agm.agl.typecheck import check

        source = "def add(x: int, y: int) -> int = x + y\nlet add1 = add(1, ?)\nadd1(2)"
        capabilities = HostCapabilities(
            agent_names=frozenset(),
            has_default_agent=True,
            supports_shell_exec=True,
            codec_kinds={},
        )
        checked = check(resolve(parse_program(source)), capabilities)
        program = lower_program(
            _compiled_checked(checked),
            source_text=source,
            source_label="<test>",
            validate=False,
        )

        validate_ir(program, deep=True)

    def test_assign_with_index_path(self) -> None:
        idx_step = IrIndexStep(kind=IndexKind.LIST, index=_int(0), location=LOC)
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
        step = IrIndexStep(kind=IndexKind.LIST, index=_int(0), location=bad_loc)
        assign = IrAssign(location=LOC, symbol=SYM_MUT, path=(step,), value=_int())
        prog = _make_program(initializers=(assign,))
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(prog, deep=False)

    def test_index_step_index_bad_location(self) -> None:
        bad_loc = loc(start_offset=10, end_offset=5)
        step = IrIndexStep(
            kind=IndexKind.LIST, index=IrConstInt(location=bad_loc, value=0), location=LOC
        )
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

    def test_symbol_owner_function_id_is_violation_when_not_in_functions(self) -> None:
        """FunctionId owner is a violation when the function is not in program.functions."""
        bad = SymbolDescriptor(
            symbol_id=SYM0, mutable=False, public_name="z", owner=FN0
        )
        prog = _make_program(symbols=_symbols_with_sym0(bad))
        # FN0 is not in program.functions, so it's still a violation
        with pytest.raises(InvalidIrError, match="not in program.functions"):
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

    def test_ir_config_bind_dangling_symbol(self) -> None:
        """IrConfigBind referencing a SymbolId not in program.symbols raises."""
        dangling = SymbolId(value=555)
        node = IrConfigBind(
            location=LOC, symbol=dangling, public_name="max-iters", value=None
        )
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="555"):
            validate_ir(prog)

    def test_ir_config_bind_deep_false_skips_symbol_check(self) -> None:
        """deep=False skips symbol resolution for IrConfigBind."""
        dangling = SymbolId(value=555)
        node = IrConfigBind(
            location=LOC, symbol=dangling, public_name="max-iters", value=None
        )
        prog = _make_program(initializers=(node,))
        validate_ir(prog, deep=False)  # dangling symbol not checked; no exception


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
        step = IrIndexStep(kind=IndexKind.LIST, index=_int(), location=bad_loc)
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


# ---------------------------------------------------------------------------
# IrField / IrIndex / IrRenderTemplate validation
# ---------------------------------------------------------------------------


class TestIrFieldValidation:
    """Structural validation for IrField nodes."""

    def test_ir_field_valid_passes(self) -> None:
        """IrField with valid location and sub-expr passes validation."""
        prog = _make_program(
            initializers=(IrBind(LOC, SYM0, IrField(LOC, IrConstInt(LOC, 1), "x")),)
        )
        validate_ir(prog, deep=False)  # no exception

    def test_ir_field_bad_location_raises(self) -> None:
        """IrField with an invalid location raises InvalidIrError."""
        bad_loc = Location(
            source_id=SourceId(999), start_offset=0, end_offset=1, start_line=1, start_col=0
        )
        prog = _make_program(
            initializers=(IrBind(LOC, SYM0, IrField(bad_loc, IrConstInt(LOC, 1), "x")),)
        )
        with pytest.raises(InvalidIrError, match="source_id"):
            validate_ir(prog, deep=True)


class TestIrIndexValidation:
    """Structural validation for IrIndex nodes."""

    def test_ir_index_valid_passes(self) -> None:
        """IrIndex with valid sub-expressions passes validation."""
        prog = _make_program(
            initializers=(
                IrBind(
                    LOC,
                    SYM0,
                    IrIndex(LOC, IndexKind.LIST, IrMakeList(LOC, ()), IrConstInt(LOC, 0)),
                ),
            )
        )
        validate_ir(prog, deep=False)  # no exception

    def test_ir_index_bad_location_raises(self) -> None:
        """IrIndex with an invalid location raises InvalidIrError."""
        bad_loc = Location(
            source_id=SourceId(999), start_offset=0, end_offset=1, start_line=1, start_col=0
        )
        prog = _make_program(
            initializers=(
                IrBind(
                    LOC,
                    SYM0,
                    IrIndex(
                        bad_loc, IndexKind.LIST, IrMakeList(LOC, ()), IrConstInt(LOC, 0)
                    ),
                ),
            )
        )
        with pytest.raises(InvalidIrError, match="source_id"):
            validate_ir(prog, deep=True)


class TestIrRenderTemplateValidation:
    """Structural validation for IrRenderTemplate nodes."""

    def test_ir_render_template_text_only_passes(self) -> None:
        """IrRenderTemplate with only text segments passes validation."""
        prog = _make_program(
            initializers=(
                IrBind(
                    LOC,
                    SYM0,
                    IrRenderTemplate(LOC, (IrTemplateText("hello"),)),
                ),
            )
        )
        validate_ir(prog, deep=False)  # no exception

    def test_ir_render_template_with_value_passes(self) -> None:
        """IrRenderTemplate with an IrTemplateValue segment passes validation."""
        prog = _make_program(
            initializers=(
                IrBind(
                    LOC,
                    SYM0,
                    IrRenderTemplate(
                        LOC,
                        (IrTemplateText("n="), IrTemplateValue(IrConstInt(LOC, 7))),
                    ),
                ),
            )
        )
        validate_ir(prog, deep=False)  # no exception

    def test_ir_render_template_bad_location_raises(self) -> None:
        """IrRenderTemplate with an invalid location raises InvalidIrError."""
        bad_loc = Location(
            source_id=SourceId(999), start_offset=0, end_offset=1, start_line=1, start_col=0
        )
        prog = _make_program(
            initializers=(
                IrBind(
                    LOC,
                    SYM0,
                    IrRenderTemplate(bad_loc, (IrTemplateText("hi"),)),
                ),
            )
        )
        with pytest.raises(InvalidIrError, match="source_id"):
            validate_ir(prog, deep=True)

# ===========================================================================
# FunctionDescriptor table, IrMakeClosure, IrDirectCall
# ===========================================================================


def _fn_sym_desc() -> SymbolDescriptor:
    """A symbol descriptor with FunctionId owner."""
    return SymbolDescriptor(symbol_id=SYM1, mutable=False, public_name="f", owner=FN0)


def _make_fn_param(sym: SymbolId = SYM1) -> IrFunctionParam:
    return IrFunctionParam(symbol=sym, default=None)


def _make_fn_desc(
    fn_id: FunctionId = FN0,
    fn_sym: SymbolId = SYM0,
    mod_id: ModuleId = MOD_A,
    params: "tuple[IrFunctionParam, ...]" = (),
) -> FunctionDescriptor:
    return FunctionDescriptor(
        function_id=fn_id,
        function_symbol=fn_sym,
        module_id=mod_id,
        params=params,
        impl=IrFunctionBody(body=IrConstInt(location=LOC, value=42)),
    )


def _make_closure(fn_id: FunctionId = FN0, captures: "tuple[IrCapture, ...]" = ()) -> IrMakeClosure:
    return IrMakeClosure(location=LOC, function_id=fn_id, captures=captures)


def _make_direct_call(fn_id: FunctionId = FN0, args: "tuple" = ()) -> IrDirectCall:
    return IrDirectCall(location=LOC, function_id=fn_id, arguments=args)


class TestFunctionDescriptorTable:
    """FunctionDescriptor table consistency invariants."""

    def test_valid_function_with_symbol_in_table(self) -> None:
        """A valid FunctionDescriptor with symbol in program.symbols passes."""
        # fn_sym is SYM0 which is in the default symbols table
        fn_desc = _make_fn_desc(fn_sym=SYM0)
        prog = _make_program(functions={FN0: fn_desc})
        validate_ir(prog)  # no exception

    def test_function_id_key_mismatch_raises(self) -> None:
        """function_id inside FunctionDescriptor must match dict key."""
        fn_id_wrong = FunctionId(value=99)
        fn_desc = _make_fn_desc(fn_id=fn_id_wrong, fn_sym=SYM0)
        prog = _make_program(functions={FN0: fn_desc})  # key=FN0 but desc.fn_id=99
        with pytest.raises(InvalidIrError, match="mismatch"):
            validate_ir(prog)

    def test_function_symbol_not_in_symbols_raises(self) -> None:
        """FunctionDescriptor.function_symbol must be in program.symbols."""
        fn_desc = _make_fn_desc(fn_sym=SymbolId(value=999))  # 999 not in default symbols
        prog = _make_program(functions={FN0: fn_desc})
        with pytest.raises(InvalidIrError, match="function_symbol"):
            validate_ir(prog)

    def test_function_module_not_in_modules_raises(self) -> None:
        """FunctionDescriptor.module_id must be in program.modules."""
        fn_desc = _make_fn_desc(mod_id=MOD_B, fn_sym=SYM0)  # MOD_B not in default modules
        prog = _make_program(functions={FN0: fn_desc})
        with pytest.raises(InvalidIrError, match="module_id"):
            validate_ir(prog)

    def test_function_param_symbol_not_in_symbols_raises(self) -> None:
        """FunctionDescriptor params must reference symbols in program.symbols."""
        bad_param = _make_fn_param(sym=SymbolId(value=999))
        fn_desc = _make_fn_desc(fn_sym=SYM0, params=(bad_param,))
        prog = _make_program(functions={FN0: fn_desc})
        with pytest.raises(InvalidIrError, match="param symbol"):
            validate_ir(prog)

    def test_symbol_owner_function_id_valid_when_in_functions(self) -> None:
        """FunctionId owner is valid when fn_id is in program.functions."""
        fn_sym = SymbolDescriptor(symbol_id=SYM0, mutable=False, public_name="f", owner=FN0)
        fn_desc = _make_fn_desc(fn_id=FN0, fn_sym=SYM0)
        mut_sym = SymbolDescriptor(symbol_id=SYM_MUT, mutable=True, public_name="y", owner=MOD_A)
        prog = _make_program(
            symbols={SYM0: fn_sym, SYM_MUT: mut_sym},
            functions={FN0: fn_desc},
        )
        validate_ir(prog)  # no exception


class TestIrMakeClosure:
    """IrMakeClosure validation invariants."""

    def test_valid_make_closure_passes(self) -> None:
        """IrMakeClosure with valid function_id in program.functions passes."""
        fn_desc = _make_fn_desc(fn_sym=SYM0)
        prog = _make_program(
            initializers=(IrBind(LOC, SYM0, _make_closure(FN0)),),
            functions={FN0: fn_desc},
        )
        validate_ir(prog)  # no exception

    def test_make_closure_unknown_function_id_raises(self) -> None:
        """IrMakeClosure with unknown function_id raises InvalidIrError (deep=True)."""
        # FN0 is not in program.functions
        closure = _make_closure(FN0)
        prog = _make_program(initializers=(IrBind(LOC, SYM0, closure),))
        with pytest.raises(InvalidIrError, match="function_id"):
            validate_ir(prog, deep=True)

    def test_make_closure_bad_capture_symbol_raises(self) -> None:
        """IrMakeClosure with unknown capture symbol raises InvalidIrError (deep=True)."""
        fn_desc = _make_fn_desc(fn_sym=SYM0)
        bad_cap = IrCapture(symbol=SymbolId(value=999), by_cell=False)
        closure = _make_closure(FN0, captures=(bad_cap,))
        prog = _make_program(
            initializers=(IrBind(LOC, SYM0, closure),),
            functions={FN0: fn_desc},
        )
        with pytest.raises(InvalidIrError, match="capture"):
            validate_ir(prog, deep=True)

    def test_make_closure_deep_false_skips_checks(self) -> None:
        """IrMakeClosure with deep=False skips cross-reference checks."""
        # FN0 not in functions, but deep=False should skip check
        closure = _make_closure(FN0)
        prog = _make_program(initializers=(IrBind(LOC, SYM0, closure),))
        validate_ir(prog, deep=False)  # no exception


class TestIrDirectCall:
    """IrDirectCall validation invariants."""

    def test_valid_direct_call_passes(self) -> None:
        """IrDirectCall with valid function_id and UseDefault arg passes."""
        # Param must have a default for UseDefault to be valid
        fn_param = IrFunctionParam(symbol=SYM0, default=IrConstInt(location=LOC, value=42))
        fn_desc = _make_fn_desc(fn_sym=SYM0, params=(fn_param,))
        call = _make_direct_call(FN0, args=(UseDefault(param_index=0),))
        prog = _make_program(
            initializers=(IrBind(LOC, SYM0, call),),
            functions={FN0: fn_desc},
        )
        validate_ir(prog)  # no exception

    def test_direct_call_unknown_function_id_raises(self) -> None:
        """IrDirectCall with unknown function_id raises InvalidIrError (deep=True)."""
        call = _make_direct_call(FN0)
        prog = _make_program(initializers=(IrBind(LOC, SYM0, call),))
        with pytest.raises(InvalidIrError, match="function_id"):
            validate_ir(prog, deep=True)

    def test_direct_call_deep_false_skips_check(self) -> None:
        """IrDirectCall with deep=False skips function_id cross-reference check."""
        call = _make_direct_call(FN0)
        prog = _make_program(initializers=(IrBind(LOC, SYM0, call),))
        validate_ir(prog, deep=False)  # no exception

    def test_direct_call_wrong_arg_count_raises(self) -> None:
        """IrDirectCall with wrong argument count raises InvalidIrError."""
        # Function has 1 param but call passes 0 args
        fn_param = IrFunctionParam(symbol=SYM0, default=None)
        fn_desc = _make_fn_desc(fn_sym=SYM0, params=(fn_param,))
        # No args — should fail arity check
        call = _make_direct_call(FN0, args=())
        prog = _make_program(
            initializers=(IrBind(LOC, SYM0, call),),
            functions={FN0: fn_desc},
        )
        with pytest.raises(InvalidIrError, match="arguments"):
            validate_ir(prog)

    def test_use_default_out_of_position_raises(self) -> None:
        """UseDefault at wrong position (param_index != slot index) raises InvalidIrError."""
        fn_param = IrFunctionParam(symbol=SYM0, default=IrConstInt(location=LOC, value=1))
        fn_desc = _make_fn_desc(fn_sym=SYM0, params=(fn_param,))
        # UseDefault with param_index=99 but it is at position 0
        call = _make_direct_call(FN0, args=(UseDefault(param_index=99),))
        prog = _make_program(
            initializers=(IrBind(LOC, SYM0, call),),
            functions={FN0: fn_desc},
        )
        with pytest.raises(InvalidIrError, match="param_index"):
            validate_ir(prog)

    def test_use_default_for_non_defaulted_param_raises(self) -> None:
        """UseDefault for a param that has no default raises InvalidIrError."""
        fn_param = IrFunctionParam(symbol=SYM0, default=None)  # no default!
        fn_desc = _make_fn_desc(fn_sym=SYM0, params=(fn_param,))
        call = _make_direct_call(FN0, args=(UseDefault(param_index=0),))
        prog = _make_program(
            initializers=(IrBind(LOC, SYM0, call),),
            functions={FN0: fn_desc},
        )
        with pytest.raises(InvalidIrError, match="no default"):
            validate_ir(prog)

    def test_function_param_default_deep_validated(self) -> None:
        """FunctionDescriptor param default is deep-validated (structurally bad node raises)."""
        # A param default that references a dangling symbol — must be caught by validator
        bad_default = IrLoad(location=LOC, symbol=SymbolId(value=9999))
        fn_param = IrFunctionParam(symbol=SYM0, default=bad_default)
        fn_desc = _make_fn_desc(fn_sym=SYM0, params=(fn_param,))
        prog = _make_program(functions={FN0: fn_desc})
        with pytest.raises(InvalidIrError, match="9999"):
            validate_ir(prog)


# ===========================================================================
# IrIndirectCall validation invariants
# ===========================================================================


def _make_indirect_call(
    callee: "IrConstInt | IrLoad | IrMakeClosure | None" = None,
    args: "tuple" = (),
) -> IrIndirectCall:
    """Build an IrIndirectCall with a default callee (IrConstInt as placeholder)."""
    _callee = callee if callee is not None else IrConstInt(location=LOC, value=42)
    return IrIndirectCall(location=LOC, callee=_callee, arguments=args)


class TestIrIndirectCall:
    """IrIndirectCall validation invariants."""

    def test_valid_indirect_call_passes(self) -> None:
        """IrIndirectCall with valid callee and positional args passes (cheap tier)."""
        callee = IrLoad(location=LOC, symbol=SYM0)
        arg = IrConstInt(location=LOC, value=5)
        call = IrIndirectCall(location=LOC, callee=callee, arguments=(arg,))
        prog = _make_program(initializers=(call,))
        validate_ir(prog, deep=False)  # no exception

    def test_indirect_call_validates_callee(self) -> None:
        """IrIndirectCall validator recurses into the callee expression."""
        # A callee with an invalid location (start_offset > end_offset) must raise.
        bad_loc = Location(
            source_id=SID0,
            start_offset=10,
            end_offset=5,  # bad: start > end
            start_line=1,
            start_col=0,
        )
        bad_callee = IrConstInt(location=bad_loc, value=1)
        call = IrIndirectCall(location=LOC, callee=bad_callee, arguments=())
        prog = _make_program(initializers=(call,))
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(prog, deep=False)

    def test_indirect_call_validates_args(self) -> None:
        """IrIndirectCall validator recurses into each argument expression."""
        bad_loc = Location(
            source_id=SID0,
            start_offset=10,
            end_offset=5,  # bad: start > end
            start_line=1,
            start_col=0,
        )
        bad_arg = IrConstInt(location=bad_loc, value=7)
        callee = IrConstInt(location=LOC, value=1)
        call = IrIndirectCall(location=LOC, callee=callee, arguments=(bad_arg,))
        prog = _make_program(initializers=(call,))
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(prog, deep=False)

    def test_indirect_call_invalid_location_raises(self) -> None:
        """IrIndirectCall with an invalid own location raises InvalidIrError."""
        bad_loc = Location(
            source_id=SID0,
            start_offset=10,
            end_offset=3,  # bad
            start_line=1,
            start_col=0,
        )
        call = IrIndirectCall(
            location=bad_loc,
            callee=IrConstInt(location=LOC, value=1),
            arguments=(),
        )
        prog = _make_program(initializers=(call,))
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(prog, deep=False)

    def test_indirect_call_deep_mode_validates_args_symbols(self) -> None:
        """IrIndirectCall with deep=True validates arg symbol cross-references."""
        # Arg loads a symbol not in program.symbols
        bad_load = IrLoad(location=LOC, symbol=SymbolId(value=9999))
        call = IrIndirectCall(
            location=LOC, callee=IrConstInt(location=LOC, value=1), arguments=(bad_load,)
        )
        prog = _make_program(initializers=(call,))
        with pytest.raises(InvalidIrError, match="9999"):
            validate_ir(prog, deep=True)

    def test_indirect_call_no_args_valid(self) -> None:
        """IrIndirectCall with zero arguments passes validation."""
        call = _make_indirect_call(args=())
        prog = _make_program(initializers=(call,))
        validate_ir(prog, deep=False)  # no exception


# ===========================================================================
# IrPrint / IrParseJson validation
# ===========================================================================


class TestPrintParseJsonValidation:
    """Negative validate tests for IrPrint and IrParseJson nodes."""

    def test_ir_print_valid(self) -> None:
        """IrPrint with valid location and inner expr passes validation."""
        from agm.agl.ir import IrPrint

        node = IrPrint(location=LOC, value=IrConstInt(location=LOC, value=42))
        prog = _make_program(initializers=(node,))
        validate_ir(prog, deep=False)  # no exception

    def test_ir_print_bad_location_raises(self) -> None:
        """IrPrint with invalid own location raises InvalidIrError."""
        from agm.agl.ir import IrPrint

        bad_loc = Location(
            source_id=SID0,
            start_offset=10,
            end_offset=3,  # bad: start > end
            start_line=1,
            start_col=0,
        )
        node = IrPrint(location=bad_loc, value=IrConstInt(location=LOC, value=1))
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(prog, deep=False)

    def test_ir_print_bad_inner_location_raises(self) -> None:
        """IrPrint validator recurses into the inner value expression."""
        from agm.agl.ir import IrPrint

        bad_loc = Location(
            source_id=SID0,
            start_offset=10,
            end_offset=3,  # bad
            start_line=1,
            start_col=0,
        )
        inner = IrConstInt(location=bad_loc, value=1)
        node = IrPrint(location=LOC, value=inner)
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(prog, deep=False)

    def test_ir_parse_json_valid(self) -> None:
        """IrParseJson with valid location and inner expr passes validation."""
        from agm.agl.ir import IrParseJson

        node = IrParseJson(
            location=LOC, value=IrConstText(location=LOC, value="null")
        )
        prog = _make_program(initializers=(node,))
        validate_ir(prog, deep=False)  # no exception

    def test_ir_parse_json_bad_location_raises(self) -> None:
        """IrParseJson with invalid own location raises InvalidIrError."""
        from agm.agl.ir import IrParseJson

        bad_loc = Location(
            source_id=SID0,
            start_offset=10,
            end_offset=3,  # bad: start > end
            start_line=1,
            start_col=0,
        )
        node = IrParseJson(
            location=bad_loc, value=IrConstText(location=LOC, value="null")
        )
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(prog, deep=False)

    def test_ir_parse_json_bad_inner_location_raises(self) -> None:
        """IrParseJson validator recurses into the inner value expression."""
        from agm.agl.ir import IrParseJson

        bad_loc = Location(
            source_id=SID0,
            start_offset=10,
            end_offset=3,  # bad
            start_line=1,
            start_col=0,
        )
        inner = IrConstText(location=bad_loc, value="null")
        node = IrParseJson(location=LOC, value=inner)
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(prog, deep=False)


# ===========================================================================
# IrParam validation
# ===========================================================================


class TestIrParamValidation:
    """Negative validate tests for IrParam in program.params."""

    def _make_program_with_params(
        self, params: "tuple", symbols: "dict | None" = None
    ) -> ExecutableProgram:
        """Build a valid base program augmented with the given params tuple."""
        from agm.agl.ir.program import ExecutableProgram

        base = _make_program(symbols=symbols)
        return ExecutableProgram(
            entry_module=base.entry_module,
            modules=base.modules,
            symbols=base.symbols,
            nominals=base.nominals,
            sources=base.sources,
            functions=base.functions,
            params=params,
        )

    def test_valid_required_param_passes(self) -> None:
        """A required IrParam with a known symbol and no default passes validation."""
        from agm.agl.ir.program import IrParam

        p = IrParam(
            symbol=SYM0,
            public_name="n",
            required=True,
            default=None,
            location=LOC,
        )
        prog = self._make_program_with_params((p,))
        validate_ir(prog, deep=True)  # no exception

    def test_valid_optional_param_with_default_passes(self) -> None:
        """An optional IrParam with a default expr and a known symbol passes validation."""
        from agm.agl.ir.program import IrParam

        p = IrParam(
            symbol=SYM0,
            public_name="n",
            required=False,
            default=IrConstInt(location=LOC, value=7),
            location=LOC,
        )
        prog = self._make_program_with_params((p,))
        validate_ir(prog, deep=True)  # no exception

    def test_ir_param_unknown_symbol_raises_deep(self) -> None:
        """IrParam referencing an unknown SymbolId raises InvalidIrError in deep mode."""
        from agm.agl.ir.program import IrParam

        bad_sym = SymbolId(value=8888)
        p = IrParam(
            symbol=bad_sym,
            public_name="n",
            required=True,
            default=None,
            location=LOC,
        )
        prog = self._make_program_with_params((p,))
        with pytest.raises(InvalidIrError, match="8888"):
            validate_ir(prog, deep=True)

    def test_ir_param_bad_location_raises_cheap(self) -> None:
        """IrParam with an invalid location raises InvalidIrError in cheap mode."""
        from agm.agl.ir.program import IrParam

        bad_loc = Location(
            source_id=SID0,
            start_offset=10,
            end_offset=3,  # bad
            start_line=1,
            start_col=0,
        )
        p = IrParam(
            symbol=SYM0,
            public_name="n",
            required=True,
            default=None,
            location=bad_loc,
        )
        prog = self._make_program_with_params((p,))
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(prog, deep=False)

    def test_ir_param_bad_default_expr_raises(self) -> None:
        """IrParam with a default expr containing a bad location raises InvalidIrError."""
        from agm.agl.ir.program import IrParam

        bad_loc = Location(
            source_id=SID0,
            start_offset=10,
            end_offset=3,  # bad
            start_line=1,
            start_col=0,
        )
        bad_default = IrConstInt(location=bad_loc, value=0)
        p = IrParam(
            symbol=SYM0,
            public_name="n",
            required=False,
            default=bad_default,
            location=LOC,
        )
        prog = self._make_program_with_params((p,))
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(prog, deep=False)

    def test_ir_param_external_decoder_refs_are_validated_deep(self) -> None:
        """IrParam external decoders must not carry dangling recursive refs."""
        from agm.agl.ir.contracts import ParamDecoder, RefDecode
        from agm.agl.ir.program import IrParam

        p = IrParam(
            symbol=SYM0,
            public_name="tree",
            required=True,
            default=None,
            location=LOC,
            external_decoder=ParamDecoder(
                target_type_label="Tree",
                json_schema="{}",
                decode=RefDecode("Tree"),
            ),
        )
        prog = self._make_program_with_params((p,))
        with pytest.raises(InvalidIrError, match="RefDecode.*Tree"):
            validate_ir(prog, deep=True)


# ===========================================================================
# IrExec validation
# ===========================================================================


class TestIrExecValidation:
    """Validation tests for IrExec nodes."""

    def _make_prog_with_contract(
        self,
        node: "object",
        contract_id: "object",
        contracts: "object",
    ) -> ExecutableProgram:
        """Build a program with a custom initializer and contracts table."""

        em = ExecutableModule(module_id=MOD_A, initializers=(node,))  # type: ignore[arg-type]
        sf = SourceFile(display_name="main.agl", normalized_text="exec(\"x\")\n()")
        nom_desc = NominalDescriptor(nominal=NOM0, display_name="Foo", kind=NominalKind.RECORD)
        return ExecutableProgram(
            entry_module=MOD_A,
            modules={MOD_A: em},
            symbols=_default_symbols(),
            nominals={NOM0: nom_desc},
            sources={SID0: sf},
            functions={},
            contracts=contracts,  # type: ignore[arg-type]
        )

    def test_ir_exec_valid_cheap(self) -> None:
        """IrExec with valid location and command expr passes cheap validation."""
        from agm.agl.ir.contracts import ContractRequest
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.nodes import IrExec

        cid = ContractId(value=0)
        contract = ContractRequest(
            codec_name="text",
            strict_json=None,
            json_schema=None,
            decode=None,
            target_type_label="text",
            structured_exec=False,
            format_instructions="",
            is_unit=False,
        )
        node = IrExec(
            location=LOC,
            command=IrConstText(location=LOC, value="echo hi"),
            contract_id=cid,
            max_attempts=1,
        )
        prog = self._make_prog_with_contract(node, cid, {cid: contract})
        validate_ir(prog, deep=False)  # no exception

    def test_ir_exec_bad_contract_id_raises_deep(self) -> None:
        """IrExec referencing a missing contract_id raises InvalidIrError in deep mode."""
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.nodes import IrExec

        cid = ContractId(value=9999)
        node = IrExec(
            location=LOC,
            command=IrConstText(location=LOC, value="echo hi"),
            contract_id=cid,
            max_attempts=1,
        )
        prog = self._make_prog_with_contract(node, cid, {})  # empty contracts
        with pytest.raises(InvalidIrError, match="9999"):
            validate_ir(prog, deep=True)

    def test_contract_refdecode_cycle_raises_deep(self) -> None:
        """Contract decoders must not contain ref-only cycles in their defs."""
        from agm.agl.ir.contracts import ContractRequest, RefDecode
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.nodes import IrExec

        cid = ContractId(value=0)
        contract = ContractRequest(
            codec_name="json",
            strict_json=None,
            json_schema="{}",
            decode=RefDecode("A"),
            target_type_label="A",
            structured_exec=False,
            format_instructions="",
            is_unit=False,
            defs=(("A", RefDecode("A")),),
        )
        node = IrExec(
            location=LOC,
            command=IrConstText(location=LOC, value="echo hi"),
            contract_id=cid,
            max_attempts=1,
        )
        prog = self._make_prog_with_contract(node, cid, {cid: contract})
        with pytest.raises(InvalidIrError, match="RefDecode.*cycle.*A"):
            validate_ir(prog, deep=True)

    def test_contract_duplicate_decode_defs_key_raises_deep(self) -> None:
        """Duplicate decode defs keys are rejected before dict coercion."""
        from agm.agl.ir.contracts import ContractRequest, RefDecode, ScalarDecode, ScalarKind
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.nodes import IrExec

        cid = ContractId(value=0)
        contract = ContractRequest(
            codec_name="json",
            strict_json=None,
            json_schema="{}",
            decode=RefDecode("A"),
            target_type_label="A",
            structured_exec=False,
            format_instructions="",
            is_unit=False,
            defs=(
                ("A", ScalarDecode(ScalarKind.INT)),
                ("A", ScalarDecode(ScalarKind.TEXT)),
            ),
        )
        node = IrExec(
            location=LOC,
            command=IrConstText(location=LOC, value="echo hi"),
            contract_id=cid,
            max_attempts=1,
        )
        prog = self._make_prog_with_contract(node, cid, {cid: contract})
        with pytest.raises(InvalidIrError, match="duplicate.*A"):
            validate_ir(prog, deep=True)

    def test_custom_contract_rejects_defs_without_decode(self) -> None:
        """Custom contracts may carry decode metadata, but defs require a decode root."""
        from agm.agl.ir.contracts import ContractRequest, ScalarDecode, ScalarKind
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.nodes import IrExec

        cid = ContractId(value=0)
        contract = ContractRequest(
            codec_name="custom-json",
            strict_json=None,
            json_schema="{}",
            decode=None,
            target_type_label="text",
            structured_exec=False,
            format_instructions="",
            is_unit=False,
            defs=(("A", ScalarDecode(ScalarKind.INT)),),
        )
        node = IrExec(
            location=LOC,
            command=IrConstText(location=LOC, value="echo hi"),
            contract_id=cid,
            max_attempts=1,
        )
        prog = self._make_prog_with_contract(node, cid, {cid: contract})
        with pytest.raises(InvalidIrError, match="defs but decode is None"):
            validate_ir(prog, deep=True)

    def test_text_contract_rejects_stale_json_decode_fields(self) -> None:
        """Text contracts must not carry stale JSON-only decode fields."""
        from agm.agl.ir.contracts import ContractRequest, ScalarDecode, ScalarKind
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.nodes import IrExec

        cid = ContractId(value=0)
        contract = ContractRequest(
            codec_name="text",
            strict_json=None,
            json_schema="{}",
            decode=ScalarDecode(ScalarKind.INT),
            target_type_label="text",
            structured_exec=False,
            format_instructions="",
            is_unit=False,
        )
        node = IrExec(
            location=LOC,
            command=IrConstText(location=LOC, value="echo hi"),
            contract_id=cid,
            max_attempts=1,
        )
        prog = self._make_prog_with_contract(node, cid, {cid: contract})
        with pytest.raises(InvalidIrError, match="must not carry json_schema/decode/defs"):
            validate_ir(prog, deep=True)

    def test_unit_contract_rejects_stale_json_decode_fields(self) -> None:
        """Unit contracts skip parsing and must not carry JSON decode fields."""
        from agm.agl.ir.contracts import ContractRequest, ScalarDecode, ScalarKind
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.nodes import IrExec

        cid = ContractId(value=0)
        contract = ContractRequest(
            codec_name="json",
            strict_json=None,
            json_schema="{}",
            decode=ScalarDecode(ScalarKind.INT),
            target_type_label="unit",
            structured_exec=False,
            format_instructions="",
            is_unit=True,
        )
        node = IrExec(
            location=LOC,
            command=IrConstText(location=LOC, value="echo hi"),
            contract_id=cid,
            max_attempts=1,
        )
        prog = self._make_prog_with_contract(node, cid, {cid: contract})
        with pytest.raises(InvalidIrError, match="must not carry json_schema/decode/defs"):
            validate_ir(prog, deep=True)

    def test_ir_exec_bad_max_attempts_raises_deep(self) -> None:
        """IrExec with max_attempts=0 raises InvalidIrError in deep mode."""
        from agm.agl.ir.contracts import ContractRequest
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.nodes import IrExec

        cid = ContractId(value=0)
        contract = ContractRequest(
            codec_name="text",
            strict_json=None,
            json_schema=None,
            decode=None,
            target_type_label="text",
            structured_exec=False,
            format_instructions="",
            is_unit=False,
        )
        node = IrExec(
            location=LOC,
            command=IrConstText(location=LOC, value="echo hi"),
            contract_id=cid,
            max_attempts=0,  # invalid
        )
        prog = self._make_prog_with_contract(node, cid, {cid: contract})
        with pytest.raises(InvalidIrError, match="max_attempts"):
            validate_ir(prog, deep=True)
