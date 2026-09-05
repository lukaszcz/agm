"""IR evaluation tests for `is` / `is not` enum-variant membership (IrNominalIs)."""

from __future__ import annotations

import pytest

from agm.agl.ir.ids import Location, NominalId, SourceId
from agm.agl.ir.nodes import IrBind, IrConstInt, IrNominalCast, IrNominalIs, IrSequence
from agm.agl.ir.program import (
    ExecutableModule,
    ExecutableProgram,
    NominalDescriptor,
    NominalKind,
    SourceFile,
)
from agm.agl.ir.validate import InvalidIrError, validate_ir
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.semantics.values import BoolValue
from tests._agl_helpers import let_root_capture
from tests.agl.ir_harness import evaluate_ir, inline_main_items, lower_inline_ir


def _lower(source: str) -> ExecutableProgram:
    """Parse → check → lower the source; return ExecutableProgram."""
    from agm.agl.capabilities import HostCapabilities

    caps = HostCapabilities(
        supports_shell_exec=False,
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset(
                {"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}
            ),
        },
    )
    return lower_inline_ir(source, caps=caps)


# ---------------------------------------------------------------------------
# IR evaluation tests — is / is not
# ---------------------------------------------------------------------------


def test_is_matching_variant() -> None:
    """`c is Red` is True when the value is that variant."""
    source = """\
enum Color | Red | Blue
let c: Color = Color::Red()
let r = c is Red
()
"""
    ir = evaluate_ir(source)
    assert ir["r"] == BoolValue(True)


def test_is_non_matching_variant() -> None:
    """`c is Blue` is False when the value is a different variant."""
    source = """\
enum Color | Red | Blue
let c: Color = Color::Red()
let r = c is Blue
()
"""
    ir = evaluate_ir(source)
    assert ir["r"] == BoolValue(False)


def test_is_not_matching_variant() -> None:
    """`c is not Red` negates the membership test."""
    source = """\
enum Color | Red | Blue
let c: Color = Color::Red()
let r = c is not Red
let s = c is not Blue
()
"""
    ir = evaluate_ir(source)
    assert ir["r"] == BoolValue(False)
    assert ir["s"] == BoolValue(True)


def test_is_field_carrying_variant() -> None:
    """`is` works on a value of a variant that carries fields."""
    source = """\
enum Shape | Circle(radius: decimal) | Rectangle(w: decimal, h: decimal)
let s: Shape = Shape::Circle(radius = 2.5)
let is-circle = s is Circle
let is-rect = s is Rectangle
()
"""
    ir = evaluate_ir(source)
    assert ir["is-circle"] == BoolValue(True)
    assert ir["is-rect"] == BoolValue(False)


def test_is_qualified_variant() -> None:
    """`is` accepts a qualified variant name (Color::Red)."""
    source = """\
enum Color | Red | Blue
let c: Color = Color::Blue()
let r = c is Color::Blue
()
"""
    ir = evaluate_ir(source)
    assert ir["r"] == BoolValue(True)


# ---------------------------------------------------------------------------
# Golden lowering test
# ---------------------------------------------------------------------------


def test_golden_is_test_lowers_to_ir_nominal_member() -> None:
    """An `is` test lowers to a nominal member identity and negation flag."""
    source = """\
enum Color | Red | Blue
let c: Color = Color::Red()
let r = c is not Blue
()
"""
    prog = _lower(source)
    prog.modules[prog.entry_module]
    found = False
    for node in inline_main_items(prog):
        if isinstance(node, (IrSequence, IrBind)) and isinstance(
            let_root_capture(node).value, IrNominalIs
        ):
            vi = let_root_capture(node).value
            assert vi.nominal == next(
                nominal
                for nominal, descriptor in prog.nominals.items()
                if descriptor.scope_path == ("Color",) and descriptor.declared_name == "Blue"
            )
            assert vi.negated is True
            found = True
    assert found, "Expected IrBind(value=IrNominalIs) in initializers"


# ---------------------------------------------------------------------------
# Negative validate tests
# ---------------------------------------------------------------------------


def _variant_is_program(
    node: IrNominalIs | IrNominalCast, nominals: dict[NominalId, NominalDescriptor]
) -> ExecutableProgram:
    sid = SourceId(0)
    return ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
        symbols={},
        nominals=nominals,
        sources={sid: SourceFile(display_name="<test>", normalized_text=" ")},
    )


def test_validate_cheap_tier_skips_nominal_checks_for_ir_variant_is() -> None:
    """deep=False validation of IrNominalIs skips the nominal/variant table checks."""
    loc = Location(source_id=SourceId(0), start_offset=0, end_offset=1, start_line=1, start_col=0)
    node = IrNominalIs(
        location=loc,
        nominal=NominalId(1),  # not registered — ignored when deep=False
        value=IrConstInt(loc, 1),
        negated=False,
    )
    validate_ir(_variant_is_program(node, {}), deep=False)  # no exception


def test_validate_cheap_tier_skips_nominal_checks_for_ir_nominal_cast() -> None:
    loc = Location(source_id=SourceId(0), start_offset=0, end_offset=1, start_line=1, start_col=0)
    node = IrNominalCast(
        location=loc,
        nominal=NominalId(1),
        value=IrConstInt(loc, 1),
        test_only=False,
        source_label="int",
        target_label="Record",
    )
    validate_ir(_variant_is_program(node, {}), deep=False)


def test_validate_rejects_ir_variant_is_with_unknown_nominal() -> None:
    """Validator rejects IrNominalIs whose nominal is absent from program.nominals."""
    loc = Location(source_id=SourceId(0), start_offset=0, end_offset=1, start_line=1, start_col=0)
    node = IrNominalIs(
        location=loc,
        nominal=NominalId(2),
        value=IrConstInt(loc, 1),
        negated=False,
    )
    with pytest.raises(InvalidIrError, match="nominal"):
        validate_ir(_variant_is_program(node, {}), deep=True)


def test_validate_ir_nominal_is_requires_member_record() -> None:
    """Nominal tests accept member-record identities and reject enum identities."""
    loc = Location(source_id=SourceId(0), start_offset=0, end_offset=1, start_line=1, start_col=0)
    member_id = NominalId(3)
    node = IrNominalIs(
        location=loc,
        nominal=member_id,
        value=IrConstInt(loc, 1),
        negated=False,
    )
    member = NominalDescriptor(member_id, ENTRY_ID, ("Color",), "Red", NominalKind.RECORD)
    validate_ir(_variant_is_program(node, {member_id: member}), deep=True)

    enum = NominalDescriptor(member_id, ENTRY_ID, (), "Color", NominalKind.ENUM, variants=())
    with pytest.raises(InvalidIrError, match="non-record"):
        validate_ir(_variant_is_program(node, {member_id: enum}), deep=True)
