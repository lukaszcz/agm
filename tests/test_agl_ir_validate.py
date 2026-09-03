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
    ExternFunctionBody,
    FunctionDescriptor,
    FunctionId,
    IndexKind,
    IntToDecimal,
    IrAssign,
    IrBind,
    IrBlock,
    IrBuiltinLoad,
    IrBuiltinStore,
    IrCapture,
    IrCase,
    IrCaseArm,
    IrCoerce,
    IrCompare,
    IrConstBool,
    IrConstDecimal,
    IrConstInt,
    IrConstJsonNull,
    IrConstText,
    IrConstUnit,
    IrDirectCall,
    IrField,
    IrFieldMode,
    IrFieldSet,
    IrFunctionBody,
    IrFunctionParam,
    IrIndex,
    IrIndexSet,
    IrIndirectCall,
    IrLiteralCaseKey,
    IrLiteralKind,
    IrLoad,
    IrMakeArray,
    IrMakeClosure,
    IrMakeDict,
    IrMakeJsonArray,
    IrMakeJsonObject,
    IrNominalCaseKey,
    IrProgramParam,
    IrRenderTemplate,
    IrSequence,
    IrTemplateText,
    IrTemplateValue,
    Location,
    NominalDescriptor,
    NominalId,
    NominalKind,
    ParamZone,
    SourceFile,
    SourceId,
    SymbolDescriptor,
    SymbolId,
    UseDefault,
    VariantDescriptor,
)
from agm.agl.ir.validate import InvalidIrError, validate_ir
from agm.agl.modules.ids import STD_CONFIG_ID, ModuleId

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOD_A = ModuleId.from_path("mod_a")
MOD_B = ModuleId.from_path("mod_b")

SID0 = SourceId(value=0)
SID1 = SourceId(value=1)
SYM0 = SymbolId(value=0)
SYM1 = SymbolId(value=1)
SYM_MUT = SymbolId(value=2)
FN0 = FunctionId(value=0)
NOM0 = NominalId(0)

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
    program_symbols: dict[int, SymbolId] | None = None,
    program_functions: dict[SymbolId, FunctionId] | None = None,
    program_signatures: "dict[SymbolId, tuple[IrProgramParam, ...]] | None" = None,
    synthetic_main_symbol: SymbolId | None = None,
    builtin_var_declarations: frozenset[tuple[ModuleId, tuple[str, ...], str]] = frozenset(),
) -> ExecutableProgram:
    """Build a valid base program; callers override individual tables.

    ``program_signatures`` defaults to an empty-tuple entry for every
    ``program_functions`` key (a parameterless program), matching what
    lowering always produces, so callers exercising unrelated invariants
    need not supply one explicitly.
    """
    nom_desc = NominalDescriptor(
        nominal=NOM0,
        module_id=MOD_A,
        scope_path=(),
        declared_name="Foo",
        kind=NominalKind.RECORD,
    )
    sf = _source_file()
    em = ExecutableModule(module_id=MOD_A, initializers=initializers)
    resolved_program_functions = {} if program_functions is None else program_functions
    return ExecutableProgram(
        entry_module=entry_module,
        modules={MOD_A: em} if modules is None else modules,
        symbols=_default_symbols() if symbols is None else symbols,
        nominals={NOM0: nom_desc} if nominals is None else nominals,
        sources={SID0: sf} if sources is None else sources,
        functions=functions or {},
        program_symbols={} if program_symbols is None else program_symbols,
        program_functions=resolved_program_functions,
        program_signatures=(
            dict.fromkeys(resolved_program_functions, ())
            if program_signatures is None
            else program_signatures
        ),
        synthetic_main_symbol=synthetic_main_symbol,
        builtin_var_declarations=builtin_var_declarations,
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
        value=value if value is not None else IrConstInt(location=LOC, value=42),
    )


def _int(v: int = 0) -> IrConstInt:
    return IrConstInt(location=LOC, value=v)


# ===========================================================================
# IrCase invariants
# ===========================================================================


def test_case_arm_cannot_bind_multiple_fields_to_one_symbol() -> None:
    enum_nominal = NominalId(11)
    program = _make_program(
        initializers=(
            IrCase(
                location=LOC,
                subject=IrConstInt(location=LOC, value=1),
                arms=(
                    IrCaseArm(
                        key=IrNominalCaseKey(nominal=enum_nominal),
                        field_bindings=(("left", SYM1), ("right", SYM1)),
                        body=IrConstUnit(location=LOC),
                    ),
                ),
                default=IrConstUnit(location=LOC),
            ),
        ),
        symbols={
            SYM1: SymbolDescriptor(
                SYM1, mutable=False, public_name=None, owner=MOD_A, synthetic=True
            )
        },
        nominals={
            enum_nominal: NominalDescriptor(
                nominal=enum_nominal,
                module_id=MOD_A,
                scope_path=(),
                declared_name="Pair",
                kind=NominalKind.RECORD,
                fields=("left", "right"),
            )
        },
    )

    with pytest.raises(InvalidIrError, match="destination symbol"):
        validate_ir(program)


def test_case_rejects_closure_capture_outside_payload_dominance() -> None:
    enum_nominal = NominalId(12)
    closure = _make_closure(FN0, captures=(IrCapture(symbol=SYM1, by_cell=False),))
    program = _make_program(
        initializers=(
            IrCase(
                location=LOC,
                subject=IrConstInt(location=LOC, value=1),
                arms=(
                    IrCaseArm(
                        key=IrNominalCaseKey(nominal=enum_nominal),
                        field_bindings=(),
                        body=closure,
                    ),
                    IrCaseArm(
                        key=IrNominalCaseKey(nominal=NominalId(13)),
                        field_bindings=(("value", SYM1),),
                        body=closure,
                    ),
                ),
                default=None,
            ),
        ),
        symbols={
            SYM0: _sym_desc_imm(),
            SYM1: SymbolDescriptor(
                SYM1, mutable=False, public_name=None, owner=MOD_A, synthetic=True
            ),
        },
        nominals={
            enum_nominal: NominalDescriptor(
                nominal=enum_nominal,
                module_id=MOD_A,
                scope_path=(),
                declared_name="Payload",
                kind=NominalKind.RECORD,
            ),
            NominalId(13): NominalDescriptor(
                nominal=NominalId(13),
                module_id=MOD_A,
                scope_path=(),
                declared_name="PayloadMember",
                kind=NominalKind.RECORD,
                fields=("value",),
            ),
        },
        functions={FN0: _make_fn_desc(fn_sym=SYM0)},
    )

    with pytest.raises(InvalidIrError, match="payload symbol"):
        validate_ir(program)


def test_private_synthetic_temporary_may_be_loaded_outside_any_case() -> None:
    """Only symbols an arm actually binds are payloads.

    Lowering temporaries look exactly like payload bindings in the symbol table
    (private, immutable, synthetic), so a load of one outside a case must stay
    valid.
    """
    program = _make_program(
        initializers=(IrLoad(location=LOC, symbol=SYM1),),
        symbols={
            SYM1: SymbolDescriptor(
                SYM1, mutable=False, public_name=None, owner=MOD_A, synthetic=True
            )
        },
    )

    validate_ir(program)


def test_case_arm_cannot_bind_a_private_source_symbol() -> None:
    enum_nominal = NominalId(13)
    program = _make_program(
        initializers=(
            IrCase(
                location=LOC,
                subject=IrConstInt(location=LOC, value=1),
                arms=(
                    IrCaseArm(
                        key=IrNominalCaseKey(nominal=enum_nominal),
                        field_bindings=(("value", SYM1),),
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
                module_id=MOD_A,
                scope_path=(),
                declared_name="Box",
                kind=NominalKind.RECORD,
                fields=("value",),
            )
        },
    )

    with pytest.raises(InvalidIrError, match="synthetic"):
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


def test_case_without_default_requires_default_for_open_literal_domain() -> None:
    program = _make_program(
        initializers=(
            IrCase(
                location=LOC,
                subject=IrConstInt(location=LOC, value=1),
                arms=(
                    IrCaseArm(
                        key=IrLiteralCaseKey(IrLiteralKind.NUMERIC, 1),
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


def test_case_without_default_allows_nominal_member_keys() -> None:
    enum_nominal = NominalId(14)
    program = _make_program(
        initializers=(
            IrCase(
                location=LOC,
                subject=IrConstInt(location=LOC, value=1),
                arms=(
                    IrCaseArm(
                        key=IrNominalCaseKey(nominal=enum_nominal),
                        field_bindings=(),
                        body=IrConstUnit(location=LOC),
                    ),
                ),
                default=None,
            ),
        ),
        nominals={
            enum_nominal: NominalDescriptor(enum_nominal, MOD_A, (), "Result", NominalKind.RECORD)
        },
    )

    validate_ir(program)


def test_case_without_default_allows_nominal_member_domain() -> None:
    enum_nominal = NominalId(15)
    program = _make_program(
        initializers=(
            IrCase(
                location=LOC,
                subject=IrConstInt(location=LOC, value=1),
                arms=(
                    IrCaseArm(
                        key=IrNominalCaseKey(nominal=enum_nominal),
                        field_bindings=(),
                        body=IrConstUnit(location=LOC),
                    ),
                ),
                default=None,
            ),
        ),
        nominals={
            enum_nominal: NominalDescriptor(enum_nominal, MOD_A, (), "Result", NominalKind.RECORD)
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
            IrMakeArray(location=LOC, items=(IrConstInt(location=LOC, value=0),)),
            IrMakeDict(
                location=LOC,
                entries=((IrConstText(location=LOC, value="k"), _int(1)),),
            ),
            IrMakeJsonArray(location=LOC, items=(IrConstInt(location=LOC, value=0),)),
            IrMakeJsonObject(
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
        from tests.agl.ir_harness import lower_inline_ir

        source = "def add(x: int, y: int) -> int = x + y\nlet add1 = add(1, ?)\nadd1(2)"
        capabilities = HostCapabilities(
            supports_shell_exec=True,
            codec_kinds={},
        )
        program = lower_inline_ir(source, caps=capabilities)

        validate_ir(program, deep=True)

    def test_index_set_container_and_value(self) -> None:
        index_set = IrIndexSet(
            location=LOC,
            container=_load_sym0(),
            kind=IndexKind.ARRAY,
            index=_int(0),
            value=_int(99),
        )
        prog = _make_program(initializers=(index_set,))
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

    def test_index_set_bad_location(self) -> None:
        bad_loc = loc(start_offset=-5, end_offset=0)
        index_set = IrIndexSet(
            location=bad_loc,
            container=_load_sym0(),
            kind=IndexKind.ARRAY,
            index=_int(0),
            value=_int(),
        )
        prog = _make_program(initializers=(index_set,))
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(prog, deep=False)

    def test_index_set_index_bad_location(self) -> None:
        bad_loc = loc(start_offset=10, end_offset=5)
        index_set = IrIndexSet(
            location=LOC,
            container=_load_sym0(),
            kind=IndexKind.ARRAY,
            index=IrConstInt(location=bad_loc, value=0),
            value=_int(),
        )
        prog = _make_program(initializers=(index_set,))
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
# Deep tier — builtin-var keys
# ===========================================================================


def test_deep_rejects_unknown_builtin_load_key() -> None:
    program = _make_program(initializers=(IrBuiltinLoad(location=LOC, key="bogus"),))

    with pytest.raises(InvalidIrError, match="builtin.*key"):
        validate_ir(program)


def test_deep_rejects_unknown_builtin_store_key() -> None:
    program = _make_program(
        initializers=(IrBuiltinStore(location=LOC, key="bogus", value=_int(1)),)
    )

    with pytest.raises(InvalidIrError, match="builtin.*key"):
        validate_ir(program)


def test_cheap_validation_does_not_require_known_builtin_key() -> None:
    program = _make_program(initializers=(IrBuiltinLoad(location=LOC, key="bogus"),))

    validate_ir(program, deep=False)


def test_deep_rejects_module_builtin_key_for_unloaded_module() -> None:
    program = _make_program(initializers=(IrBuiltinLoad(location=LOC, key=(MOD_B, (), "value")),))

    with pytest.raises(InvalidIrError, match="unloaded module"):
        validate_ir(program)

    validate_ir(program, deep=False)


def test_deep_rejects_undeclared_module_builtin_key() -> None:
    program = _make_program(initializers=(IrBuiltinLoad(location=LOC, key=(MOD_A, (), "value")),))

    with pytest.raises(InvalidIrError, match="declaration"):
        validate_ir(program)


def test_deep_rejects_module_builtin_key_at_undeclared_scope() -> None:
    modules = {
        MOD_A: ExecutableModule(
            module_id=MOD_A,
            initializers=(IrBuiltinLoad(location=LOC, key=(STD_CONFIG_ID, ("Region",), "value")),),
        ),
        STD_CONFIG_ID: ExecutableModule(module_id=STD_CONFIG_ID, initializers=()),
    }
    program = _make_program(
        modules=modules,
        builtin_var_declarations=frozenset(((STD_CONFIG_ID, ("Other",), "value"),)),
    )

    with pytest.raises(InvalidIrError, match="declaration"):
        validate_ir(program)


def test_deep_accepts_declared_scoped_module_builtin_key() -> None:
    key = (MOD_A, ("Region",), "value")
    program = _make_program(
        initializers=(IrBuiltinLoad(location=LOC, key=key),),
        builtin_var_declarations=frozenset((key,)),
    )

    validate_ir(program)


def test_deep_validates_module_qualified_engine_keys() -> None:
    key = (STD_CONFIG_ID, (), "max-iters")
    modules = {
        MOD_A: ExecutableModule(
            module_id=MOD_A,
            initializers=(IrBuiltinLoad(location=LOC, key=key),),
        ),
        STD_CONFIG_ID: ExecutableModule(module_id=STD_CONFIG_ID, initializers=()),
    }
    known = _make_program(modules=modules, builtin_var_declarations=frozenset((key,)))
    validate_ir(known)

    modules[MOD_A] = ExecutableModule(
        module_id=MOD_A,
        initializers=(IrBuiltinLoad(location=LOC, key=(STD_CONFIG_ID, (), "unknown")),),
    )
    unknown = _make_program(modules=modules)
    with pytest.raises(InvalidIrError, match="unknown engine key"):
        validate_ir(unknown)


def test_deep_accepts_legacy_string_key_builtin_default() -> None:
    program = _make_program()
    program.builtin_setting_defaults["max-iters"] = _int(1)

    validate_ir(program)


def test_deep_rejects_undeclared_module_builtin_default_key() -> None:
    program = _make_program()
    program.builtin_setting_defaults[(MOD_A, ("Region",), "value")] = _int(1)

    with pytest.raises(InvalidIrError, match="declaration"):
        validate_ir(program)


def test_deep_accepts_known_builtin_keys() -> None:
    program = _make_program(
        initializers=(
            IrBuiltinLoad(location=LOC, key="max-iters"),
            IrBuiltinStore(
                location=LOC,
                key="log",
                value=IrConstBool(location=LOC, value=True),
            ),
        )
    )

    validate_ir(program)


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
        bad = SymbolDescriptor(symbol_id=SYM0, mutable=False, public_name="z", owner=MOD_B)
        prog = _make_program(symbols=_symbols_with_sym0(bad))
        with pytest.raises(InvalidIrError, match="owner"):
            validate_ir(prog)

    def test_symbol_owner_function_id_is_violation_when_not_in_functions(self) -> None:
        """FunctionId owner is a violation when the function is not in program.functions."""
        bad = SymbolDescriptor(symbol_id=SYM0, mutable=False, public_name="z", owner=FN0)
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
        nom_wrong = NominalId(16)
        nom_desc = NominalDescriptor(
            nominal=nom_wrong,  # key will be NOM0 — mismatch
            module_id=MOD_A,
            scope_path=(),
            declared_name="Foo",
            kind=NominalKind.RECORD,
        )
        prog = _make_program(nominals={NOM0: nom_desc})
        with pytest.raises(InvalidIrError, match="nominal"):
            validate_ir(prog)

    def test_nominal_key_mismatch_skipped_when_shallow(self) -> None:
        nom_wrong = NominalId(17)
        nom_desc = NominalDescriptor(
            nominal=nom_wrong,
            module_id=MOD_A,
            scope_path=(),
            declared_name="Foo",
            kind=NominalKind.RECORD,
        )
        prog = _make_program(nominals={NOM0: nom_desc})
        validate_ir(prog, deep=False)

    @pytest.mark.parametrize("mutable_fields", (frozenset({"z"}), frozenset({"x", "z"})))
    def test_record_mutable_fields_must_be_declared_fields(
        self, mutable_fields: frozenset[str]
    ) -> None:
        descriptor = NominalDescriptor(
            nominal=NOM0,
            module_id=MOD_A,
            scope_path=(),
            declared_name="Foo",
            kind=NominalKind.RECORD,
            fields=("x", "y"),
            mutable_fields=mutable_fields,
        )
        with pytest.raises(InvalidIrError, match="mutable fields"):
            validate_ir(_make_program(nominals={NOM0: descriptor}))

    @pytest.mark.parametrize("kind", (NominalKind.ENUM, NominalKind.EXCEPTION))
    def test_non_record_descriptor_cannot_declare_mutable_fields(self, kind: NominalKind) -> None:
        descriptor = NominalDescriptor(
            nominal=NOM0,
            module_id=MOD_A,
            scope_path=(),
            declared_name="Foo",
            kind=kind,
            mutable_fields=frozenset({"x"}),
        )
        with pytest.raises(InvalidIrError, match="mutable_fields"):
            validate_ir(_make_program(nominals={NOM0: descriptor}))


# ===========================================================================
# Deep tier — source key consistency
# ===========================================================================


class TestDeepTierSourceConsistency:
    def test_sources_are_accessible(self) -> None:
        """Every source key a node's location names is resolved, not just the first."""
        second = SourceFile(display_name="other.agl", normalized_text=SOURCE_TEXT)
        nodes = (
            IrConstInt(location=loc(source_id=SID0), value=1),
            IrConstInt(location=loc(source_id=SID1), value=2),
        )
        prog = _make_program(
            initializers=nodes,
            sources={SID0: _source_file(), SID1: second},
        )
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
        node = IrAssign(location=LOC, symbol=dangling, value=_int())
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
        node = IrAssign(location=LOC, symbol=dangling, value=_int())
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

    def test_index_set_location_source_id_missing(self) -> None:
        """IrIndexSet.location source_id is also validated."""
        bad_loc = loc(source_id=SID1)
        index_set = IrIndexSet(
            location=bad_loc,
            container=_load_sym0(),
            kind=IndexKind.ARRAY,
            index=_int(),
            value=_int(),
        )
        prog = _make_program(initializers=(index_set,))
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

    def test_make_array_item_bad_location(self) -> None:
        bad_loc = loc(start_offset=-1, end_offset=0)
        item = IrConstInt(location=bad_loc, value=1)
        node = IrMakeArray(location=LOC, items=(item,))
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

    def test_make_json_array_item_bad_location(self) -> None:
        bad_loc = loc(start_offset=-1, end_offset=0)
        item = IrConstInt(location=bad_loc, value=1)
        node = IrMakeJsonArray(location=LOC, items=(item,))
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError):
            validate_ir(prog, deep=False)

    def test_make_json_object_key_bad_location(self) -> None:
        bad_loc = loc(start_line=0)
        key = IrConstText(location=bad_loc, value="k")
        val = _int(1)
        node = IrMakeJsonObject(location=LOC, entries=((key, val),))
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError):
            validate_ir(prog, deep=False)

    def test_make_json_object_value_bad_location(self) -> None:
        bad_loc = loc(start_col=-1)
        key = IrConstText(location=LOC, value="k")
        val = IrConstInt(location=bad_loc, value=1)
        node = IrMakeJsonObject(location=LOC, entries=((key, val),))
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
        node = IrAssign(location=LOC, symbol=SYM_MUT, value=child)
        prog = _make_program(initializers=(node,))
        with pytest.raises(InvalidIrError, match="444"):
            validate_ir(prog)


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
            initializers=(IrBind(LOC, SYM0, IrField(LOC, IrConstInt(LOC, 1), NOM0, "x")),)
        )
        validate_ir(prog, deep=False)  # no exception

    def test_ir_field_bad_location_raises(self) -> None:
        """IrField with an invalid location raises InvalidIrError."""
        bad_loc = Location(
            source_id=SourceId(999), start_offset=0, end_offset=1, start_line=1, start_col=0
        )
        prog = _make_program(
            initializers=(IrBind(LOC, SYM0, IrField(bad_loc, IrConstInt(LOC, 1), NOM0, "x")),)
        )
        with pytest.raises(InvalidIrError, match="source_id"):
            validate_ir(prog, deep=True)

    @pytest.mark.parametrize(
        "descriptor",
        (
            NominalDescriptor(
                nominal=NOM0,
                module_id=MOD_A,
                scope_path=(),
                declared_name="Foo",
                kind=NominalKind.RECORD,
                fields=("x",),
            ),
            NominalDescriptor(
                nominal=NOM0,
                module_id=MOD_A,
                scope_path=(),
                declared_name="Foo",
                kind=NominalKind.ENUM,
                variants=(VariantDescriptor("some", ("x",), NominalId(100)),),
            ),
        ),
    )
    def test_ir_field_registered_record_or_enum_payload_passes_deep_validation(
        self, descriptor: NominalDescriptor
    ) -> None:
        prog = _make_program(
            initializers=(IrBind(LOC, SYM0, IrField(LOC, IrConstInt(LOC, 1), NOM0, "x")),),
            nominals={
                NOM0: descriptor,
                NominalId(100): NominalDescriptor(
                    NominalId(100),
                    MOD_A,
                    (),
                    "some",
                    NominalKind.RECORD,
                    ("x",),
                ),
            },
        )
        validate_ir(prog, deep=True)

    @pytest.mark.parametrize(
        "descriptor",
        (
            NominalDescriptor(
                nominal=NOM0,
                module_id=MOD_A,
                scope_path=(),
                declared_name="Foo",
                kind=NominalKind.RECORD,
                fields=("x",),
            ),
            NominalDescriptor(
                nominal=NOM0,
                module_id=MOD_A,
                scope_path=(),
                declared_name="Foo",
                kind=NominalKind.ENUM,
                variants=(VariantDescriptor("some", ("x",), NominalId(100)),),
            ),
        ),
    )
    @pytest.mark.parametrize("mode", (IrFieldMode.EXACT, IrFieldMode.UPPER_BOUND))
    def test_ir_field_unknown_nominal_field_fails_deep_validation(
        self, descriptor: NominalDescriptor, mode: IrFieldMode
    ) -> None:
        prog = _make_program(
            initializers=(
                IrBind(
                    LOC,
                    SYM0,
                    IrField(
                        LOC,
                        IrConstInt(LOC, 1),
                        NOM0,
                        "missing",
                        mode=mode,
                    ),
                ),
            ),
            nominals={
                NOM0: descriptor,
                NominalId(100): NominalDescriptor(
                    NominalId(100),
                    MOD_A,
                    (),
                    "some",
                    NominalKind.RECORD,
                    ("x",),
                ),
            },
        )
        with pytest.raises(InvalidIrError, match="unknown field"):
            validate_ir(prog, deep=True)

    def test_ir_field_upper_bound_uses_declaring_nominal_fields(self) -> None:
        """Upper-bound mode validates against the bound nominal descriptor."""
        prog = _make_program(
            initializers=(
                IrBind(
                    LOC,
                    SYM0,
                    IrField(
                        LOC,
                        IrConstInt(LOC, 1),
                        NOM0,
                        "x",
                        mode=IrFieldMode.UPPER_BOUND,
                    ),
                ),
            ),
            nominals={
                NOM0: NominalDescriptor(
                    nominal=NOM0,
                    module_id=MOD_A,
                    scope_path=(),
                    declared_name="Foo",
                    kind=NominalKind.RECORD,
                    fields=("x",),
                )
            },
        )
        validate_ir(prog, deep=True)

    def test_ir_field_empty_name_fails_shallow_validation(self) -> None:
        prog = _make_program(
            initializers=(IrBind(LOC, SYM0, IrField(LOC, IrConstInt(LOC, 1), NOM0, "")),)
        )
        with pytest.raises(InvalidIrError, match="non-empty"):
            validate_ir(prog, deep=False)


class TestIrFieldSetValidation:
    """Structural validation for mutable record-field stores."""

    def _program_with_store(
        self,
        *,
        mutable: bool,
        nominal_kind: NominalKind = NominalKind.RECORD,
    ) -> ExecutableProgram:
        store = IrFieldSet(LOC, IrConstInt(LOC, 1), NOM0, "x", IrConstInt(LOC, 2))
        descriptor = NominalDescriptor(
            nominal=NOM0,
            module_id=MOD_A,
            scope_path=(),
            declared_name="Foo",
            kind=nominal_kind,
            fields=("x",),
            mutable_fields=frozenset({"x"})
            if mutable and nominal_kind is NominalKind.RECORD
            else frozenset(),
        )
        return _make_program(initializers=(store,), nominals={NOM0: descriptor})

    def test_mutable_record_field_store_passes_deep_validation(self) -> None:
        validate_ir(self._program_with_store(mutable=True), deep=True)

    def test_immutable_record_field_store_fails_deep_validation(self) -> None:
        with pytest.raises(InvalidIrError, match="mutable"):
            validate_ir(self._program_with_store(mutable=False), deep=True)

    def test_non_record_field_store_fails_deep_validation(self) -> None:
        with pytest.raises(InvalidIrError, match="record"):
            validate_ir(
                self._program_with_store(mutable=True, nominal_kind=NominalKind.EXCEPTION),
                deep=True,
            )

    def test_unknown_field_store_fails_deep_validation(self) -> None:
        descriptor = NominalDescriptor(
            nominal=NOM0,
            module_id=MOD_A,
            scope_path=(),
            declared_name="Foo",
            kind=NominalKind.RECORD,
            fields=("x",),
            mutable_fields=frozenset({"x"}),
        )
        prog = _make_program(
            initializers=(
                IrFieldSet(LOC, IrConstInt(LOC, 1), NOM0, "missing", IrConstInt(LOC, 2)),
            ),
            nominals={NOM0: descriptor},
        )
        with pytest.raises(InvalidIrError, match="unknown field"):
            validate_ir(prog, deep=True)

    def test_mutable_field_store_passes_shallow_validation(self) -> None:
        validate_ir(self._program_with_store(mutable=True), deep=False)

    def test_empty_field_name_fails_shallow_validation(self) -> None:
        prog = _make_program(
            initializers=(IrFieldSet(LOC, IrConstInt(LOC, 1), NOM0, "", IrConstInt(LOC, 2)),)
        )
        with pytest.raises(InvalidIrError, match="non-empty"):
            validate_ir(prog, deep=False)


class TestIrIndexValidation:
    """Structural validation for IrIndex nodes."""

    def test_ir_index_valid_passes(self) -> None:
        """IrIndex with valid sub-expressions passes validation."""
        prog = _make_program(
            initializers=(
                IrBind(
                    LOC,
                    SYM0,
                    IrIndex(LOC, IndexKind.ARRAY, IrMakeArray(LOC, ()), IrConstInt(LOC, 0)),
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
                    IrIndex(bad_loc, IndexKind.ARRAY, IrMakeArray(LOC, ()), IrConstInt(LOC, 0)),
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


def _make_program_param(*, name: str = "n", required: bool = True) -> IrProgramParam:
    from agm.agl.ir.contracts import ParamDecoder, ScalarDecode, ScalarKind

    return IrProgramParam(
        name=name,
        kind=ParamZone.NAMED_ONLY,
        required=required,
        external_decoder=ParamDecoder(
            target_type_label="int",
            json_schema="{}",
            decode=ScalarDecode(kind=ScalarKind.INT),
        ),
    )


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


class TestProgramEntryMaps:
    """Linked ``program def`` symbol/function maps must agree with each other."""

    def test_valid_program_entry_maps_pass(self) -> None:
        fn_desc = _make_fn_desc(fn_sym=SYM0)
        prog = _make_program(
            functions={FN0: fn_desc},
            program_symbols={10: SYM0},
            program_functions={SYM0: FN0},
        )

        validate_ir(prog)

    def test_synthetic_main_must_resolve_to_the_marked_descriptor(self) -> None:
        marked = FunctionDescriptor(
            function_id=FN0,
            function_symbol=SYM0,
            module_id=MOD_A,
            params=(),
            impl=IrFunctionBody(body=IrConstInt(location=LOC, value=42)),
            is_synthetic_main=True,
        )
        alias = _make_fn_desc(fn_id=FunctionId(1), fn_sym=SYM0)
        prog = _make_program(
            functions={FN0: marked, FunctionId(1): alias},
            program_symbols={10: SYM0},
            program_functions={SYM0: FunctionId(1)},
            synthetic_main_symbol=SYM0,
        )

        with pytest.raises(InvalidIrError, match="synthetic_main_symbol"):
            validate_ir(prog)

    def test_synthetic_main_must_be_a_linked_program_entry(self) -> None:
        marked = FunctionDescriptor(
            function_id=FN0,
            function_symbol=SYM0,
            module_id=MOD_A,
            params=(),
            impl=IrFunctionBody(body=IrConstInt(location=LOC, value=42)),
            is_synthetic_main=True,
        )
        other_function_id = FunctionId(1)
        other = _make_fn_desc(fn_id=other_function_id, fn_sym=SYM_MUT)
        prog = _make_program(
            functions={FN0: marked, other_function_id: other},
            program_symbols={10: SYM_MUT},
            program_functions={SYM_MUT: other_function_id},
            synthetic_main_symbol=SYM0,
        )

        with pytest.raises(InvalidIrError, match="linked program entry"):
            validate_ir(prog)

    def test_synthetic_main_requires_exactly_one_marked_descriptor(self) -> None:
        marked = FunctionDescriptor(
            function_id=FN0,
            function_symbol=SYM0,
            module_id=MOD_A,
            params=(),
            impl=IrFunctionBody(body=IrConstInt(location=LOC, value=42)),
            is_synthetic_main=True,
        )
        duplicate_marked = FunctionDescriptor(
            function_id=FunctionId(1),
            function_symbol=SYM0,
            module_id=MOD_A,
            params=(),
            impl=IrFunctionBody(body=IrConstInt(location=LOC, value=42)),
            is_synthetic_main=True,
        )
        prog = _make_program(
            functions={FN0: marked, FunctionId(1): duplicate_marked},
            program_symbols={10: SYM0},
            program_functions={SYM0: FN0},
            synthetic_main_symbol=SYM0,
        )

        with pytest.raises(InvalidIrError, match="sole marked"):
            validate_ir(prog)

    def test_program_entry_cannot_target_extern_function(self) -> None:
        extern = FunctionDescriptor(
            function_id=FN0,
            function_symbol=SYM0,
            module_id=MOD_A,
            params=(),
            impl=ExternFunctionBody(name="main"),
        )
        prog = _make_program(
            functions={FN0: extern},
            program_symbols={10: SYM0},
            program_functions={SYM0: FN0},
        )

        with pytest.raises(InvalidIrError, match="IrFunctionBody"):
            validate_ir(prog)

    def test_program_entry_can_target_a_parameterized_function_when_signature_agrees(
        self,
    ) -> None:
        """A program def with declared parameters validates when its signature agrees."""
        fn_desc = _make_fn_desc(fn_sym=SYM0, params=(_make_fn_param(),))
        prog = _make_program(
            symbols={SYM0: _sym_desc_imm(), SYM1: _fn_sym_desc()},
            functions={FN0: fn_desc},
            program_symbols={10: SYM0},
            program_functions={SYM0: FN0},
            program_signatures={SYM0: (_make_program_param(),)},
        )

        validate_ir(prog)  # no exception

    def test_program_symbol_must_be_registered(self) -> None:
        prog = _make_program(program_symbols={10: SYM1})

        with pytest.raises(InvalidIrError, match="program_symbols"):
            validate_ir(prog)

    def test_program_symbol_must_have_a_function_entry(self) -> None:
        prog = _make_program(program_symbols={10: SYM0})

        with pytest.raises(InvalidIrError, match="program_functions"):
            validate_ir(prog)

    def test_program_symbols_reject_duplicate_declarations_for_one_symbol(self) -> None:
        fn_desc = _make_fn_desc(fn_sym=SYM0)
        prog = _make_program(
            functions={FN0: fn_desc},
            program_symbols={10: SYM0, 11: SYM0},
            program_functions={SYM0: FN0},
        )

        with pytest.raises(InvalidIrError, match="program_symbols"):
            validate_ir(prog)

    def test_program_function_symbol_must_be_registered(self) -> None:
        prog = _make_program(program_functions={SYM1: FN0})

        with pytest.raises(InvalidIrError, match="program_functions"):
            validate_ir(prog)

    def test_program_function_must_be_registered(self) -> None:
        prog = _make_program(
            program_symbols={10: SYM0},
            program_functions={SYM0: FunctionId(99)},
        )

        with pytest.raises(InvalidIrError, match="program_functions"):
            validate_ir(prog)

    def test_program_function_must_match_its_function_descriptor(self) -> None:
        fn_desc = _make_fn_desc(fn_sym=SYM0)
        prog = _make_program(
            symbols={SYM0: _sym_desc_imm(), SYM1: _fn_sym_desc()},
            functions={FN0: fn_desc},
            program_symbols={10: SYM1},
            program_functions={SYM1: FN0},
        )

        with pytest.raises(InvalidIrError, match="function_symbol"):
            validate_ir(prog)

    def test_program_function_must_have_a_source_declaration(self) -> None:
        fn_desc = _make_fn_desc(fn_sym=SYM0)
        prog = _make_program(functions={FN0: fn_desc}, program_functions={SYM0: FN0})

        with pytest.raises(InvalidIrError, match="program_symbols"):
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


class TestProgramSignatureValidation:
    """``program_signatures`` must index exactly the linked programs and agree
    with their function descriptors."""

    def test_program_function_without_a_signature_raises(self) -> None:
        """A linked program symbol missing a program_signatures entry is rejected."""
        fn_desc = _make_fn_desc(fn_sym=SYM0)
        prog = _make_program(
            functions={FN0: fn_desc},
            program_symbols={10: SYM0},
            program_functions={SYM0: FN0},
            program_signatures={},
        )

        with pytest.raises(InvalidIrError, match="program_signatures"):
            validate_ir(prog)

    def test_signature_for_a_non_program_symbol_raises(self) -> None:
        """A program_signatures entry not indexed by a linked program is rejected."""
        prog = _make_program(program_signatures={SYM1: ()})

        with pytest.raises(InvalidIrError, match="program_signatures"):
            validate_ir(prog)

    def test_signature_disagreeing_with_the_descriptor_raises(self) -> None:
        """A signature whose parameter count disagrees with its function is rejected."""
        fn_desc = _make_fn_desc(fn_sym=SYM0)  # zero declared IrFunctionParams
        prog = _make_program(
            functions={FN0: fn_desc},
            program_symbols={10: SYM0},
            program_functions={SYM0: FN0},
            program_signatures={SYM0: (_make_program_param(),)},
        )

        with pytest.raises(InvalidIrError, match="program_signatures"):
            validate_ir(prog)

    def test_signature_required_flag_disagreeing_with_default_raises(self) -> None:
        """A signature parameter's required flag must agree with its default presence."""
        fn_desc = _make_fn_desc(fn_sym=SYM0, params=(_make_fn_param(),))  # no default: required
        prog = _make_program(
            symbols={SYM0: _sym_desc_imm(), SYM1: _fn_sym_desc()},
            functions={FN0: fn_desc},
            program_symbols={10: SYM0},
            program_functions={SYM0: FN0},
            program_signatures={SYM0: (_make_program_param(required=False),)},
        )

        with pytest.raises(InvalidIrError, match="disagrees"):
            validate_ir(prog)


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
# IrPrint validation
# ===========================================================================


class TestPrintParseJsonValidation:
    """Negative validation tests for IrPrint nodes."""

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


# ===========================================================================
# IrCopyValue validation
# ===========================================================================


class TestCopyValidation:
    """Negative validate tests for IrCopyValue nodes."""

    def test_ir_copy_value_valid_for_both_kinds(self) -> None:
        """Both copy kinds with valid locations and inner expressions pass validation."""
        from agm.agl.ir import CopyKind, IrCopyValue

        for kind in CopyKind:
            node = IrCopyValue(location=LOC, kind=kind, value=IrConstInt(location=LOC, value=42))
            validate_ir(_make_program(initializers=(node,)), deep=False)

    def test_ir_copy_value_bad_location_raises(self) -> None:
        """An IrCopyValue with an invalid own location raises InvalidIrError."""
        from agm.agl.ir import CopyKind, IrCopyValue

        bad_loc = Location(
            source_id=SID0,
            start_offset=10,
            end_offset=3,  # bad: start > end
            start_line=1,
            start_col=0,
        )
        node = IrCopyValue(
            location=bad_loc,
            kind=CopyKind.SHALLOW,
            value=IrConstInt(location=LOC, value=1),
        )
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(_make_program(initializers=(node,)), deep=False)

    def test_ir_copy_value_bad_inner_location_raises(self) -> None:
        """An IrCopyValue validator recurses into the inner value expression."""
        from agm.agl.ir import CopyKind, IrCopyValue

        bad_loc = Location(
            source_id=SID0,
            start_offset=10,
            end_offset=3,  # bad
            start_line=1,
            start_col=0,
        )
        inner = IrConstInt(location=bad_loc, value=1)
        node = IrCopyValue(location=LOC, kind=CopyKind.DEEP, value=inner)
        with pytest.raises(InvalidIrError, match="start_offset"):
            validate_ir(_make_program(initializers=(node,)), deep=False)


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
        sf = SourceFile(display_name="main.agl", normalized_text='exec("x")\n()')
        nom_desc = NominalDescriptor(
            nominal=NOM0,
            module_id=MOD_A,
            scope_path=(),
            declared_name="Foo",
            kind=NominalKind.RECORD,
        )
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
            env=IrConstText(location=LOC, value="env"),
            cwd=IrConstText(location=LOC, value="cwd"),
            timeout=IrConstText(location=LOC, value="timeout"),
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
            env=IrConstText(location=LOC, value="env"),
            cwd=IrConstText(location=LOC, value="cwd"),
            timeout=IrConstText(location=LOC, value="timeout"),
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
            env=IrConstText(location=LOC, value="env"),
            cwd=IrConstText(location=LOC, value="cwd"),
            timeout=IrConstText(location=LOC, value="timeout"),
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
            env=IrConstText(location=LOC, value="env"),
            cwd=IrConstText(location=LOC, value="cwd"),
            timeout=IrConstText(location=LOC, value="timeout"),
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
            env=IrConstText(location=LOC, value="env"),
            cwd=IrConstText(location=LOC, value="cwd"),
            timeout=IrConstText(location=LOC, value="timeout"),
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
            env=IrConstText(location=LOC, value="env"),
            cwd=IrConstText(location=LOC, value="cwd"),
            timeout=IrConstText(location=LOC, value="timeout"),
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
            env=IrConstText(location=LOC, value="env"),
            cwd=IrConstText(location=LOC, value="cwd"),
            timeout=IrConstText(location=LOC, value="timeout"),
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
            env=IrConstText(location=LOC, value="env"),
            cwd=IrConstText(location=LOC, value="cwd"),
            timeout=IrConstText(location=LOC, value="timeout"),
            contract_id=cid,
            max_attempts=0,  # invalid
        )
        prog = self._make_prog_with_contract(node, cid, {cid: contract})
        with pytest.raises(InvalidIrError, match="max_attempts"):
            validate_ir(prog, deep=True)
