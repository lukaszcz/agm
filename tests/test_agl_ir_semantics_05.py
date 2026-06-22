"""M3e-1 differential ir_semantic — `is` / `is not` enum-variant membership (IrVariantIs).

"""

from __future__ import annotations

import pytest

from agm.agl.eval.values import BoolValue
from agm.agl.ir.ids import Location, NominalId, SourceId
from agm.agl.ir.nodes import IrBind, IrConstInt, IrVariantIs
from agm.agl.ir.program import (
    ExecutableModule,
    ExecutableProgram,
    NominalDescriptor,
    NominalKind,
    SourceFile,
    VariantDescriptor,
)
from agm.agl.ir.validate import InvalidIrError, validate_ir
from agm.agl.modules.ids import ENTRY_ID
from tests.agl.ir_harness import evaluate_ir


def _lower(source: str) -> ExecutableProgram:
    """Parse → check → lower the source; return ExecutableProgram with validate=True."""
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


# ---------------------------------------------------------------------------
# IR semantic tests — is / is not
# ---------------------------------------------------------------------------


def test_is_matching_variant() -> None:
    """`c is Red` is True when the value is that variant."""
    source = """\
enum Color | Red | Blue
let c = Color.Red()
let r = c is Red
()
"""
    ir_reference, ir = evaluate_ir(source)
    assert ir_reference["r"] == BoolValue(True)
    assert ir["r"] == BoolValue(True)


def test_is_non_matching_variant() -> None:
    """`c is Blue` is False when the value is a different variant."""
    source = """\
enum Color | Red | Blue
let c = Color.Red()
let r = c is Blue
()
"""
    ir_reference, ir = evaluate_ir(source)
    assert ir_reference["r"] == BoolValue(False)
    assert ir["r"] == BoolValue(False)


def test_is_not_matching_variant() -> None:
    """`c is not Red` negates the membership test."""
    source = """\
enum Color | Red | Blue
let c = Color.Red()
let r = c is not Red
let s = c is not Blue
()
"""
    ir_reference, ir = evaluate_ir(source)
    assert ir_reference["r"] == BoolValue(False)
    assert ir_reference["s"] == BoolValue(True)
    assert ir["r"] == BoolValue(False)
    assert ir["s"] == BoolValue(True)


def test_is_field_carrying_variant() -> None:
    """`is` works on a value of a variant that carries fields."""
    source = """\
enum Shape | Circle(radius: decimal) | Rectangle(w: decimal, h: decimal)
let s = Shape.Circle(radius: 2.5)
let is_circle = s is Circle
let is_rect = s is Rectangle
()
"""
    ir_reference, ir = evaluate_ir(source)
    assert ir_reference["is_circle"] == BoolValue(True)
    assert ir_reference["is_rect"] == BoolValue(False)
    assert ir["is_circle"] == BoolValue(True)
    assert ir["is_rect"] == BoolValue(False)


def test_is_qualified_variant() -> None:
    """`is` accepts a qualified variant name (Color.Red)."""
    source = """\
enum Color | Red | Blue
let c = Color.Blue()
let r = c is Color.Blue
()
"""
    ir_reference, ir = evaluate_ir(source)
    assert ir_reference["r"] == BoolValue(True)
    assert ir["r"] == BoolValue(True)


# ---------------------------------------------------------------------------
# Golden lowering test
# ---------------------------------------------------------------------------


def test_golden_is_test_lowers_to_ir_variant_is() -> None:
    """An `is` test lowers to IrVariantIs with the resolved nominal/variant/negated."""
    source = """\
enum Color | Red | Blue
let c = Color.Red()
let r = c is not Blue
()
"""
    prog = _lower(source)
    entry = prog.modules[prog.entry_module]
    found = False
    for node in entry.initializers:
        if isinstance(node, IrBind) and isinstance(node.value, IrVariantIs):
            vi = node.value
            assert vi.nominal == NominalId(ENTRY_ID, "Color")
            assert vi.variant == "Blue"
            assert vi.negated is True
            found = True
    assert found, "Expected IrBind(value=IrVariantIs) in initializers"


# ---------------------------------------------------------------------------
# Negative validate tests
# ---------------------------------------------------------------------------


def _variant_is_program(
    node: IrVariantIs, nominals: dict[NominalId, NominalDescriptor]
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
    """deep=False validation of IrVariantIs skips the nominal/variant table checks."""
    loc = Location(source_id=SourceId(0), start_offset=0, end_offset=1, start_line=1, start_col=0)
    node = IrVariantIs(
        location=loc,
        nominal=NominalId(ENTRY_ID, "Ghost"),  # not registered — ignored when deep=False
        variant="Red",
        value=IrConstInt(loc, 1),
        negated=False,
    )
    validate_ir(_variant_is_program(node, {}), deep=False)  # no exception


def test_validate_rejects_ir_variant_is_with_unknown_nominal() -> None:
    """Validator rejects IrVariantIs whose nominal is absent from program.nominals."""
    loc = Location(source_id=SourceId(0), start_offset=0, end_offset=1, start_line=1, start_col=0)
    node = IrVariantIs(
        location=loc,
        nominal=NominalId(ENTRY_ID, "Ghost"),
        variant="Red",
        value=IrConstInt(loc, 1),
        negated=False,
    )
    with pytest.raises(InvalidIrError, match="nominal"):
        validate_ir(_variant_is_program(node, {}), deep=True)


def test_validate_rejects_ir_variant_is_with_unknown_variant() -> None:
    """Validator rejects IrVariantIs whose variant is absent from the descriptor."""
    loc = Location(source_id=SourceId(0), start_offset=0, end_offset=1, start_line=1, start_col=0)
    nominal_id = NominalId(ENTRY_ID, "Color")
    desc = NominalDescriptor(
        nominal=nominal_id,
        display_name="Color",
        kind=NominalKind.ENUM,
        fields=(),
        variants=(VariantDescriptor(name="Red", fields=()),),
    )
    node = IrVariantIs(
        location=loc,
        nominal=nominal_id,
        variant="Purple",
        value=IrConstInt(loc, 1),
        negated=False,
    )
    with pytest.raises(InvalidIrError, match="variant"):
        validate_ir(_variant_is_program(node, {nominal_id: desc}), deep=True)
