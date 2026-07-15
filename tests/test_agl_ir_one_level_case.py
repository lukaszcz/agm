"""Construction, validation, and direct evaluation of one-level ``IrCase``."""

from __future__ import annotations

import decimal

import pytest

from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.ir import (
    ExecutableModule,
    ExecutableProgram,
    InvalidIrError,
    IrBind,
    IrCase,
    IrCaseArm,
    IrConstBool,
    IrConstDecimal,
    IrConstInt,
    IrConstText,
    IrEnumCaseKey,
    IrLiteralCaseKey,
    IrLiteralKind,
    IrLoad,
    IrMakeEnum,
    Location,
    NominalDescriptor,
    NominalId,
    NominalKind,
    SourceFile,
    SourceId,
    SymbolDescriptor,
    SymbolId,
    VariantDescriptor,
    validate_ir,
)
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.semantics.values import IntValue

_SOURCE = "case"
_LOC = Location(SourceId(0), 0, 1, 1, 0)
_COLOR = NominalId(ENTRY_ID, "Color")
_OTHER = NominalId(ENTRY_ID, "Other")
_PAYLOAD = SymbolId(0)
_RESULT = SymbolId(1)


def _program(
    expr: IrCase | IrBind,
    *,
    symbols: dict[SymbolId, SymbolDescriptor] | None = None,
    nominals: dict[NominalId, NominalDescriptor] | None = None,
) -> ExecutableProgram:
    return ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(ENTRY_ID, (expr,))},
        symbols=symbols or {},
        nominals=nominals or {},
        sources={SourceId(0): SourceFile("<test>", _SOURCE)},
    )


def _color_descriptor() -> NominalDescriptor:
    return NominalDescriptor(
        nominal=_COLOR,
        display_name="Color",
        kind=NominalKind.ENUM,
        variants=(
            VariantDescriptor("Plain", ()),
            VariantDescriptor("With", ("value", "unused")),
        ),
    )


def _private_symbol(symbol: SymbolId = _PAYLOAD) -> SymbolDescriptor:
    return SymbolDescriptor(
        symbol, mutable=False, public_name=None, owner=ENTRY_ID, synthetic=True
    )


def _literal_case(*arms: IrCaseArm, default: IrConstInt | None = None) -> IrCase:
    return IrCase(_LOC, IrConstInt(_LOC, 1), arms, default)


def test_numeric_keys_are_runtime_canonical_and_duplicate_semantics_are_rejected() -> None:
    integer = IrLiteralCaseKey(IrLiteralKind.NUMERIC, 1)
    widened = IrLiteralCaseKey(IrLiteralKind.NUMERIC, decimal.Decimal("1.0"))
    assert integer.scalar_value == decimal.Decimal(1)
    assert integer == widened
    case = _literal_case(
        IrCaseArm(integer, (), IrConstInt(_LOC, 1)),
        IrCaseArm(widened, (), IrConstInt(_LOC, 2)),
    )
    with pytest.raises(InvalidIrError):
        validate_ir(_program(case))


def test_validation_accepts_enum_field_binding_and_default() -> None:
    case = IrCase(
        _LOC,
        IrConstInt(_LOC, 0),
        (
            IrCaseArm(
                IrEnumCaseKey(_COLOR, "With"),
                (("value", _PAYLOAD),),
                IrLoad(_LOC, _PAYLOAD),
            ),
        ),
        IrConstInt(_LOC, 0),
    )
    validate_ir(
        _program(
            case,
            symbols={_PAYLOAD: _private_symbol()},
            nominals={_COLOR: _color_descriptor()},
        )
    )


@pytest.mark.parametrize(
    "arms",
    [
        (
            IrCaseArm(IrLiteralCaseKey(IrLiteralKind.BOOL, True), (), IrConstInt(_LOC, 1)),
            IrCaseArm(IrLiteralCaseKey(IrLiteralKind.TEXT, "x"), (), IrConstInt(_LOC, 2)),
        ),
        (
            IrCaseArm(IrEnumCaseKey(_COLOR, "Plain"), (), IrConstInt(_LOC, 1)),
            IrCaseArm(IrEnumCaseKey(_OTHER, "Plain"), (), IrConstInt(_LOC, 2)),
        ),
    ],
)
def test_validation_rejects_incompatible_case_families(arms: tuple[IrCaseArm, ...]) -> None:
    with pytest.raises(InvalidIrError):
        validate_ir(_program(IrCase(_LOC, IrConstInt(_LOC, 0), arms, None)), deep=False)


@pytest.mark.parametrize(
    ("field_bindings", "symbols"),
    [
        ((("missing", _PAYLOAD),), {_PAYLOAD: _private_symbol()}),
        (
            (("value", _PAYLOAD), ("value", SymbolId(2))),
            {_PAYLOAD: _private_symbol(), SymbolId(2): _private_symbol(SymbolId(2))},
        ),
        ((("value", SymbolId(99)),), {}),
        (
            (("value", _PAYLOAD),),
            {_PAYLOAD: SymbolDescriptor(_PAYLOAD, True, None, ENTRY_ID)},
        ),
        (
            (("value", _PAYLOAD),),
            {_PAYLOAD: SymbolDescriptor(_PAYLOAD, False, "visible", ENTRY_ID)},
        ),
    ],
)
def test_validation_rejects_bad_enum_field_bindings(
    field_bindings: tuple[tuple[str, SymbolId], ...],
    symbols: dict[SymbolId, SymbolDescriptor],
) -> None:
    case = IrCase(
        _LOC,
        IrConstInt(_LOC, 0),
        (IrCaseArm(IrEnumCaseKey(_COLOR, "With"), field_bindings, IrConstInt(_LOC, 1)),),
        None,
    )
    with pytest.raises(InvalidIrError):
        validate_ir(_program(case, symbols=symbols, nominals={_COLOR: _color_descriptor()}))


def test_validation_rejects_unknown_nominal_and_variant() -> None:
    unknown_nominal = IrCase(
        _LOC,
        IrConstInt(_LOC, 0),
        (IrCaseArm(IrEnumCaseKey(_OTHER, "Plain"), (), IrConstInt(_LOC, 1)),),
        None,
    )
    with pytest.raises(InvalidIrError):
        validate_ir(_program(unknown_nominal, nominals={_COLOR: _color_descriptor()}))
    unknown_variant = IrCase(
        _LOC,
        IrConstInt(_LOC, 0),
        (IrCaseArm(IrEnumCaseKey(_COLOR, "Missing"), (), IrConstInt(_LOC, 1)),),
        None,
    )
    with pytest.raises(InvalidIrError):
        validate_ir(_program(unknown_variant, nominals={_COLOR: _color_descriptor()}))


def test_validation_rejects_non_enum_nominal_and_corrupted_literal_key() -> None:
    record_descriptor = NominalDescriptor(
        nominal=_COLOR,
        display_name="Color",
        kind=NominalKind.RECORD,
    )
    enum_case = IrCase(
        _LOC,
        IrConstInt(_LOC, 0),
        (IrCaseArm(IrEnumCaseKey(_COLOR, "Plain"), (), IrConstInt(_LOC, 1)),),
        None,
    )
    with pytest.raises(InvalidIrError):
        validate_ir(_program(enum_case, nominals={_COLOR: record_descriptor}))

    key = IrLiteralCaseKey(IrLiteralKind.TEXT, "valid")
    object.__setattr__(key, "scalar_value", 1)
    corrupted = IrCase(
        _LOC,
        IrConstText(_LOC, "valid"),
        (IrCaseArm(key, (), IrConstInt(_LOC, 1)),),
        None,
    )
    with pytest.raises(InvalidIrError):
        validate_ir(_program(corrupted), deep=False)


def test_validation_rejects_literal_fields_and_shallow_enum_fields_skip_tables() -> None:
    literal = IrCase(
        _LOC,
        IrConstText(_LOC, "x"),
        (
            IrCaseArm(
                IrLiteralCaseKey(IrLiteralKind.TEXT, "x"),
                (("value", _PAYLOAD),),
                IrConstInt(_LOC, 1),
            ),
        ),
        None,
    )
    with pytest.raises(InvalidIrError):
        validate_ir(_program(literal), deep=False)

    shallow = IrCase(
        _LOC,
        IrConstInt(_LOC, 0),
        (
            IrCaseArm(
                IrEnumCaseKey(_COLOR, "With"),
                (("value", SymbolId(99)),),
                IrConstInt(_LOC, 1),
            ),
        ),
        None,
    )
    validate_ir(_program(shallow), deep=False)


def test_validation_accepts_shared_dag_and_rejects_cycle() -> None:
    shared = IrConstInt(_LOC, 7)
    case = IrCase(
        _LOC,
        IrConstBool(_LOC, True),
        (
            IrCaseArm(IrLiteralCaseKey(IrLiteralKind.BOOL, False), (), shared),
            IrCaseArm(IrLiteralCaseKey(IrLiteralKind.BOOL, True), (), shared),
        ),
        None,
    )
    validate_ir(_program(case))
    object.__setattr__(case, "default", case)
    with pytest.raises(InvalidIrError):
        validate_ir(_program(case))


def test_direct_literal_dispatch_uses_numeric_value_equality_and_default() -> None:
    case = IrCase(
        _LOC,
        IrConstInt(_LOC, 1),
        (
            IrCaseArm(
                IrLiteralCaseKey(IrLiteralKind.NUMERIC, decimal.Decimal("1.0")),
                (),
                IrConstInt(_LOC, 9),
            ),
        ),
        IrConstInt(_LOC, 0),
    )
    result_bind = IrBind(_LOC, _RESULT, case)
    result_desc = SymbolDescriptor(_RESULT, False, "result", ENTRY_ID)
    assert IrInterpreter(_program(result_bind, symbols={_RESULT: result_desc})).run() == {
        "result": IntValue(9)
    }


def test_direct_malformed_no_match_raises_invalid_ir_not_match_error() -> None:
    case = IrCase(
        _LOC,
        IrConstText(_LOC, "no"),
        (
            IrCaseArm(
                IrLiteralCaseKey(IrLiteralKind.TEXT, "yes"), (), IrConstInt(_LOC, 1)
            ),
        ),
        None,
    )
    with pytest.raises(InvalidIrError):
        IrInterpreter(_program(case)).run()


def test_direct_enum_dispatch_copies_fields_and_rejects_missing_payload() -> None:
    subject = IrMakeEnum(
        _LOC,
        _COLOR,
        "Color",
        "With",
        (("value", IrConstInt(_LOC, 42)), ("unused", IrConstInt(_LOC, 0))),
    )
    enum_case = IrCase(
        _LOC,
        subject,
        (
            IrCaseArm(
                IrEnumCaseKey(_COLOR, "With"),
                (("value", _PAYLOAD),),
                IrLoad(_LOC, _PAYLOAD),
            ),
        ),
        None,
    )
    result = IrBind(_LOC, _RESULT, enum_case)
    symbols = {
        _PAYLOAD: _private_symbol(),
        _RESULT: SymbolDescriptor(_RESULT, False, "result", ENTRY_ID),
    }
    assert IrInterpreter(_program(result, symbols=symbols)).run() == {
        "result": IntValue(42)
    }

    missing_field_subject = IrMakeEnum(_LOC, _COLOR, "Color", "With", ())
    malformed = IrCase(
        _LOC,
        missing_field_subject,
        (
            IrCaseArm(
                IrEnumCaseKey(_COLOR, "With"),
                (("value", _PAYLOAD),),
                IrConstInt(_LOC, 1),
            ),
        ),
        None,
    )
    with pytest.raises(InvalidIrError):
        IrInterpreter(_program(malformed)).run()


def test_direct_malformed_literal_payload_binding_rejects_non_enum_subject() -> None:
    malformed = IrCase(
        _LOC,
        IrConstText(_LOC, "x"),
        (
            IrCaseArm(
                IrLiteralCaseKey(IrLiteralKind.TEXT, "x"),
                (("value", _PAYLOAD),),
                IrConstInt(_LOC, 1),
            ),
        ),
        None,
    )
    with pytest.raises(InvalidIrError):
        IrInterpreter(_program(malformed)).run()


def test_direct_literal_key_rejects_invalid_scalar_kind_pair() -> None:
    with pytest.raises(ValueError):
        IrLiteralCaseKey(IrLiteralKind.BOOL, 1)
    with pytest.raises(ValueError):
        IrLiteralCaseKey(IrLiteralKind.NUMERIC, True)
    with pytest.raises(ValueError):
        IrLiteralCaseKey(IrLiteralKind.NULL, "null")
    assert IrLiteralCaseKey(IrLiteralKind.TEXT, "x").scalar_value == "x"
    assert IrLiteralCaseKey(IrLiteralKind.NULL, None).scalar_value is None
    assert IrLiteralCaseKey(IrLiteralKind.BOOL, False).scalar_value is False
    assert IrConstDecimal(_LOC, decimal.Decimal(1)).value == decimal.Decimal(1)
