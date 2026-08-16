"""IR evaluation tests for record/enum/exception construction and constructor refs.

Tests all node types: IrMakeRecord, IrMakeEnum, IrMakeException, IrMakeConstructor.
"""

from __future__ import annotations

import dataclasses
import decimal

import pytest

from agm.agl.ir.ids import NominalId, SourceId
from agm.agl.ir.nodes import (
    IrBind,
    IrMakeConstructor,
    IrMakeEnum,
    IrMakeException,
    IrMakeRecord,
    IrSequence,
)
from agm.agl.ir.program import (
    ExecutableProgram,
    NominalDescriptor,
    NominalKind,
    VariantDescriptor,
)
from agm.agl.ir.validate import InvalidIrError
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.semantics.values import (
    BoolValue,
    ConstructorValue,
    DecimalValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    RecordValue,
    TextValue,
)
from tests._agl_helpers import let_root_capture
from tests.agl.ir_harness import evaluate_ir, inline_main_items, lower_inline_ir, nominal_id_for

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# IR evaluation tests — record construction
# ---------------------------------------------------------------------------


def test_record_construction_basic() -> None:
    """Record construction: Point(x: 3, y: 4) produces the correct RecordValue."""
    source = """\
record Point
  x: int
  y: int
let p = Point(x = 3, y = 4)
()
"""
    ir = evaluate_ir(source)
    p = ir["p"]
    assert isinstance(p, RecordValue)
    assert p.fields["x"] == IntValue(3)
    assert p.fields["y"] == IntValue(4)
    assert p.display_name == "Point"
    prog = lower_inline_ir(source)
    assert p.nominal == nominal_id_for(prog, "Point")


def test_record_field_access_now_unblocked() -> None:
    """Record construction + field access: p.x works end-to-end."""
    source = """\
record Point
  x: int
  y: int
let p = Point(x = 3, y = 4)
let px = p.x
()
"""
    ir = evaluate_ir(source)
    assert ir["px"] == IntValue(3)


def test_record_field_coercion_int_to_decimal() -> None:
    """Record construction with int→decimal field coercion.

    The score field is declared as decimal; passing an int literal triggers
    lower_coerced to insert IrCoerce(IntToDecimal) at lowering time.
    """
    source = """\
record Score
  name: text
  value: decimal
let s = Score(name = "Alice", value = 42)
()
"""
    ir = evaluate_ir(source)
    s = ir["s"]
    assert isinstance(s, RecordValue)
    assert s.fields["name"] == TextValue("Alice")
    assert s.fields["value"] == DecimalValue(decimal.Decimal(42))


def test_record_equality_by_nominal_and_fields() -> None:
    """Two records with the same nominal and fields compare equal."""
    source = """\
record Pair
  a: int
  b: int
let p1 = Pair(a = 1, b = 2)
let p2 = Pair(a = 1, b = 2)
let eq = p1 == p2
()
"""
    ir = evaluate_ir(source)
    assert ir["eq"] == BoolValue(True)


def test_record_inequality_different_fields() -> None:
    """Two records with same nominal but different fields compare not-equal."""
    source = """\
record Pair
  a: int
  b: int
let p1 = Pair(a = 1, b = 2)
let p2 = Pair(a = 1, b = 9)
let ne = p1 != p2
()
"""
    ir = evaluate_ir(source)
    assert ir["ne"] == BoolValue(True)


def test_template_with_record_interpolation_now_unblocked() -> None:
    """Template interpolation with a record value."""
    source = """\
record Point
  x: int
  y: int
let p = Point(x = 1, y = 2)
let s: text = "point: %{p}"
()
"""
    ir = evaluate_ir(source)
    assert isinstance(ir["s"], TextValue)


# ---------------------------------------------------------------------------
# IR evaluation tests — enum construction
# ---------------------------------------------------------------------------


def test_enum_nullary_variant() -> None:
    """Enum nullary variant: Color::Red() constructs correctly."""
    source = """\
enum Color | Red | Blue
let c = Color::Red()
()
"""
    ir = evaluate_ir(source)
    c = ir["c"]
    assert isinstance(c, EnumValue)
    assert c.variant == "Red"
    assert c.fields == {}


def test_enum_variant_with_fields() -> None:
    """Enum variant with fields constructs correctly."""
    source = """\
enum Shape | Circle(radius: decimal) | Rectangle(w: decimal, h: decimal)
let s = Shape::Circle(radius = 3.0)
()
"""
    ir = evaluate_ir(source)
    s = ir["s"]
    assert isinstance(s, EnumValue)
    assert s.variant == "Circle"
    assert s.fields["radius"] == DecimalValue(decimal.Decimal("3.0"))


def test_enum_equality_same_variant() -> None:
    """Two enum values with same variant and fields compare equal."""
    source = """\
enum Color | Red | Blue
let c1 = Color::Red()
let c2 = Color::Red()
let eq = c1 == c2
()
"""
    ir = evaluate_ir(source)
    assert ir["eq"] == BoolValue(True)


def test_enum_inequality_different_variants() -> None:
    """Two enum values with different variants compare not-equal."""
    source = """\
enum Color | Red | Blue
let c1 = Color::Red()
let c2 = Color::Blue()
let ne = c1 != c2
()
"""
    ir = evaluate_ir(source)
    assert ir["ne"] == BoolValue(True)


def test_enum_inequality_different_nominals() -> None:
    """Two enum values from different nominals produce different NominalIds.

    We cannot compare them with != in AgL (the checker requires same type for ==).
    Instead we verify that evaluation produces EnumValues with different nominals.

    NominalId is an opaque per-declaration handle unique within a single
    ``ExecutableProgram`` (see ``agm.agl.ir.ids.NominalId``), not a globally
    comparable name+module key, so both enums are declared in the same
    program to keep this identity check meaningful.
    """
    source = """\
enum ColorA | Red
enum ColorB | Red
let ca = ColorA::Red()
let cb = ColorB::Red()
()
"""
    ir = evaluate_ir(source)
    ca = ir["ca"]
    cb = ir["cb"]
    assert isinstance(ca, EnumValue)
    assert isinstance(cb, EnumValue)
    assert ca.nominal != cb.nominal, "Different enum types must have different NominalIds"


def test_enum_variant_field_coercion() -> None:
    """Enum variant field: int→decimal coercion applied at lowering time."""
    source = """\
enum Size | Big(amount: decimal) | Small
let s = Size::Big(amount = 7)
()
"""
    ir = evaluate_ir(source)
    sv = ir["s"]
    assert isinstance(sv, EnumValue)
    assert sv.fields["amount"] == DecimalValue(decimal.Decimal(7))


# ---------------------------------------------------------------------------
# IR evaluation tests — NominalId hashing
# ---------------------------------------------------------------------------


def test_nominal_id_equality_record() -> None:
    """NominalId-based equality: constructed records compare by nominal + fields.

    Records hold a mutable ``fields`` dict (it may embed an array or dict), so
    a ``RecordValue`` is unhashable — it cannot be a set member or dict key.
    Instead we construct two *equal* records (p1, p2) and one *different*
    record (p3) via AgL and assert on ``==``, exercising the full pipeline —
    lowering → evaluation → NominalId equality.
    """
    source = """\
record Point
  x: int
  y: int
let p1 = Point(x = 1, y = 2)
let p2 = Point(x = 1, y = 2)
let p3 = Point(x = 9, y = 9)
()
"""
    ir = evaluate_ir(source)

    p1 = ir["p1"]
    p2 = ir["p2"]
    p3 = ir["p3"]
    assert isinstance(p1, RecordValue)
    assert isinstance(p2, RecordValue)
    assert isinstance(p3, RecordValue)

    assert p1 == p2
    assert p1 != p3
    with pytest.raises(TypeError):
        hash(p1)


def test_nominal_id_equality_enum() -> None:
    """NominalId-based equality: constructed enum values compare by nominal + variant + fields.

    Two equal enum values (same variant, same nominal) compare equal; a value
    from a different variant does not. An ``EnumValue`` is unhashable for the
    same reason a ``RecordValue`` is — its ``fields`` may hold an array or dict.
    """
    source = """\
enum Color | Red | Blue
let c1 = Color::Red()
let c2 = Color::Red()
let c3 = Color::Blue()
()
"""
    ir = evaluate_ir(source)

    c1 = ir["c1"]
    c2 = ir["c2"]
    c3 = ir["c3"]
    assert isinstance(c1, EnumValue)
    assert isinstance(c2, EnumValue)
    assert isinstance(c3, EnumValue)

    assert c1 == c2
    assert c1 != c3
    with pytest.raises(TypeError):
        hash(c1)


# ---------------------------------------------------------------------------
# IR evaluation tests — exception construction
# ---------------------------------------------------------------------------


def test_exception_construction_builtin_explicit_fields() -> None:
    """Exception construction preserves all caller-supplied built-in fields."""
    source = """\
let e = ArithmeticError(message = "div/0", operation = "/")
()
"""
    ir = evaluate_ir(source)
    e = ir["e"]
    assert isinstance(e, ExceptionValue)
    assert e.fields["message"] == TextValue("div/0")
    assert e.fields["operation"] == TextValue("/")


# ---------------------------------------------------------------------------
# IR evaluation tests — first-class constructor references
# ---------------------------------------------------------------------------


def test_first_class_record_constructor_ref() -> None:
    """A record type name used as a first-class value gives a ConstructorValue.

    A record with at least one field is not nullary, so referencing it by name
    without calling it produces a ConstructorValue that can later be applied.
    """
    source = """\
record Pt
  x: int
let mk = Pt
()
"""
    ir = evaluate_ir(source)
    mk = ir["mk"]
    assert isinstance(mk, ConstructorValue), f"ir: {mk!r}"
    assert mk.display_name == "Pt"
    assert mk.variant is None


def test_first_class_enum_constructor_ref_nullary_gives_enum_value() -> None:
    """A nullary enum variant accessed without calling it produces an EnumValue directly.

    Nullary variants (no fields) are always eagerly evaluated — no ConstructorValue
    wrapper is created.
    """
    source = """\
enum Color | Red | Blue
let mk = Color::Red
()
"""
    ir = evaluate_ir(source)
    mk = ir["mk"]
    assert isinstance(mk, EnumValue), f"ir: {mk!r}"
    assert mk.variant == "Red"


def test_first_class_enum_constructor_ref_with_fields_gives_constructor_value() -> None:
    """An enum variant WITH fields used as a value (not called) gives a ConstructorValue.

    Only non-nullary variants (those with at least one field) produce a ConstructorValue
    when accessed in value position via qualified form (Enum::Variant).
    """
    source = """\
enum Shape
  | Circle(radius: int)
  | Square(side: int)
let mk = Shape::Circle
()
"""
    ir = evaluate_ir(source)
    mk = ir["mk"]
    assert isinstance(mk, ConstructorValue), f"ir: {mk!r}"
    assert mk.display_name == "Shape"
    assert mk.variant == "Circle"


def test_bare_payload_constructor_type_apply_is_callable_value() -> None:
    """``some::[int]`` is a constructor function value that can be applied."""
    source = """\
enum Option[T]
  | none
  | some(value: T)
let mk = some::[int]
let v = mk(7)
()
"""
    ir = evaluate_ir(source)
    mk = ir["mk"]
    assert isinstance(mk, ConstructorValue)
    assert mk.display_name == "Option" and mk.variant == "some"
    v = ir["v"]
    assert isinstance(v, EnumValue)
    assert v.variant == "some" and v.fields["value"] == IntValue(7)


def test_bare_nullary_constructor_type_apply_constructs_directly() -> None:
    """``none::[int]`` constructs the nullary value without parentheses."""
    source = """\
enum Option[T]
  | none
  | some(value: T)
let z = none::[int]
()
"""
    ir = evaluate_ir(source)
    z = ir["z"]
    assert isinstance(z, EnumValue)
    assert z.variant == "none" and z.fields == {}


def test_qualified_constructor_type_apply_is_callable_value() -> None:
    """``Option[int]::some`` and ``Option[int]::none`` work as values."""
    source = """\
enum Option[T]
  | none
  | some(value: T)
let mk = Option[int]::some
let v = mk(7)
let z = Option[int]::none
()
"""
    ir = evaluate_ir(source)
    mk = ir["mk"]
    assert isinstance(mk, ConstructorValue)
    assert mk.variant == "some"
    assert isinstance(ir["v"], EnumValue) and ir["v"].variant == "some"
    assert isinstance(ir["z"], EnumValue) and ir["z"].variant == "none"


# ---------------------------------------------------------------------------
# Golden lowering tests — node shapes
# ---------------------------------------------------------------------------


def test_golden_record_lowers_to_ir_make_record() -> None:
    """Record constructor call lowers to IrMakeRecord with correct fields."""
    source = """\
record Point
  x: int
  y: int
let p = Point(x = 3, y = 4)
()
"""
    prog = _lower(source)
    prog.modules[prog.entry_module]
    found = False
    for node in inline_main_items(prog):
        if isinstance(node, (IrSequence, IrBind)) and isinstance(
            let_root_capture(node).value, IrMakeRecord
        ):
            mr = let_root_capture(node).value
            assert mr.display_name == "Point"
            assert prog.nominals[mr.nominal].declared_name == "Point"
            assert len(mr.fields) == 2
            assert mr.fields[0][0] == "x"
            assert mr.fields[1][0] == "y"
            found = True
    assert found, "Expected IrBind(value=IrMakeRecord) in initializers"


def test_golden_enum_lowers_to_ir_make_enum() -> None:
    """Enum variant call lowers to IrMakeEnum with correct variant."""
    source = """\
enum Color | Red | Blue
let c = Color::Red()
()
"""
    prog = _lower(source)
    prog.modules[prog.entry_module]
    found = False
    for node in inline_main_items(prog):
        if isinstance(node, (IrSequence, IrBind)) and isinstance(
            let_root_capture(node).value, IrMakeEnum
        ):
            me = let_root_capture(node).value
            assert me.display_name == "Color"
            assert me.variant == "Red"
            assert me.fields == ()
            found = True
    assert found, "Expected IrBind(value=IrMakeEnum) in initializers"


def test_golden_exception_lowers_to_ir_make_exception() -> None:
    """Exception construction lowers each caller-supplied field to an expression."""
    source = """\
let e = ArithmeticError(message = "oops", operation = "/")
()
"""
    prog = _lower(source)
    prog.modules[prog.entry_module]
    found = False
    for node in inline_main_items(prog):
        if isinstance(node, (IrSequence, IrBind)) and isinstance(
            let_root_capture(node).value, IrMakeException
        ):
            me = let_root_capture(node).value
            assert me.display_name == "ArithmeticError"
            assert [name for name, _ in me.fields] == ["message", "operation"]
            found = True
    assert found, "Expected IrBind(value=IrMakeException) in initializers"


def test_golden_record_field_coercion_lowered() -> None:
    """Record field with int→decimal coercion: the field expr is wrapped in IrCoerce."""
    from agm.agl.ir.nodes import IrCoerce
    from agm.agl.ir.operations import IntToDecimal

    source = """\
record Score
  name: text
  value: decimal
let s = Score(name = "Bob", value = 5)
()
"""
    prog = _lower(source)
    prog.modules[prog.entry_module]
    found = False
    for node in inline_main_items(prog):
        if isinstance(node, (IrSequence, IrBind)) and isinstance(
            let_root_capture(node).value, IrMakeRecord
        ):
            for fname, fexpr in let_root_capture(node).value.fields:
                if fname == "value":
                    assert isinstance(fexpr, IrCoerce), (
                        "value field expr should be IrCoerce(IntToDecimal)"
                    )
                    assert isinstance(fexpr.operation, IntToDecimal)
                    found = True
    assert found, "Expected coerced decimal field in IrMakeRecord"


def test_golden_constructor_ref_lowers_to_ir_make_constructor() -> None:
    """First-class constructor ref lowers to IrMakeConstructor."""
    source = """\
record Pt
  x: int
let mk = Pt
()
"""
    prog = _lower(source)
    prog.modules[prog.entry_module]
    found = False
    for node in inline_main_items(prog):
        if isinstance(node, (IrSequence, IrBind)) and isinstance(
            let_root_capture(node).value, IrMakeConstructor
        ):
            mc = let_root_capture(node).value
            assert mc.display_name == "Pt"
            assert mc.variant is None
            assert prog.nominals[mc.nominal].declared_name == "Pt"
            found = True
    assert found, "Expected IrBind(value=IrMakeConstructor) in initializers"


# ---------------------------------------------------------------------------
# Golden lowering tests — program.nominals table
# ---------------------------------------------------------------------------


def test_nominals_table_contains_user_record() -> None:
    """program.nominals contains a descriptor for each user-declared record."""
    source = """\
record Point
  x: int
  y: int
let p = Point(x = 1, y = 2)
()
"""
    prog = _lower(source)
    nominal_id = nominal_id_for(prog, "Point")
    assert nominal_id in prog.nominals, "Expected NominalId for Point in program.nominals"
    desc = prog.nominals[nominal_id]
    assert desc.kind == NominalKind.RECORD
    assert desc.display_name == "Point"
    assert desc.fields == ("x", "y")


def test_nominals_table_contains_user_enum() -> None:
    """program.nominals contains a descriptor for each user-declared enum."""
    source = """\
enum Color | Red | Blue(shade: int)
()
"""
    prog = _lower(source)
    nominal_id = nominal_id_for(prog, "Color")
    assert nominal_id in prog.nominals
    desc = prog.nominals[nominal_id]
    assert desc.kind == NominalKind.ENUM
    # variants in declaration order
    variant_names = [v.name for v in desc.variants]
    assert variant_names == ["Red", "Blue"]
    blue = next(v for v in desc.variants if v.name == "Blue")
    assert blue.fields == ("shade",)


def test_nominals_table_contains_builtin_exception_fields() -> None:
    """program.nominals includes ArithmeticError with its declared fields in order."""
    source = "()"
    prog = _lower(source)
    nominal_id = nominal_id_for(prog, "ArithmeticError")
    assert nominal_id in prog.nominals
    desc = prog.nominals[nominal_id]
    assert desc.kind == NominalKind.EXCEPTION
    assert desc.fields == ("message", "operation")


def test_nominals_table_contains_builtin_exceptions() -> None:
    """program.nominals includes all built-in exception descriptors."""
    source = "()"
    prog = _lower(source)
    builtin_names = [
        "IndexError",
        "KeyError",
        "ArithmeticError",
        "RecursionError",
        "Exception",
        "AgentCallError",
        "AgentParseError",
    ]
    for name in builtin_names:
        nominal_id = nominal_id_for(prog, name)
        assert nominal_id in prog.nominals, f"Expected built-in {name!r} in program.nominals"
        desc = prog.nominals[nominal_id]
        assert desc.kind == NominalKind.EXCEPTION
        assert desc.nominal == nominal_id


def test_nominals_table_key_value_consistency() -> None:
    """NominalDescriptor.nominal matches the dict key for every entry."""
    source = """\
record Foo
  x: int
enum Bar | A | B(n: int)
()
"""
    prog = _lower(source)
    for key, desc in prog.nominals.items():
        assert desc.nominal == key, (
            f"program.nominals key {key!r} disagrees with descriptor.nominal {desc.nominal!r}"
        )


# ---------------------------------------------------------------------------
# Validate tests — deep completeness checks
# ---------------------------------------------------------------------------


def test_validate_rejects_ir_make_record_with_unknown_nominal() -> None:
    """Validator rejects IrMakeRecord whose nominal is absent from program.nominals."""
    from agm.agl.ir.ids import Location, SourceId
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import validate_ir

    sid = SourceId(0)
    loc = Location(source_id=sid, start_offset=0, end_offset=1, start_line=1, start_col=0)
    unknown_nominal = NominalId(1)
    node = IrMakeRecord(
        location=loc,
        nominal=unknown_nominal,
        display_name="Ghost",
        fields=(),
    )
    from agm.agl.modules.ids import ENTRY_ID as EID

    prog = ExecutableProgram(
        entry_module=EID,
        modules={EID: ExecutableModule(module_id=EID, initializers=(node,))},
        symbols={},
        nominals={},  # Ghost not registered
        sources={sid: SourceFile(display_name="<test>", normalized_text=" ")},
    )
    with pytest.raises(InvalidIrError, match="nominal"):
        validate_ir(prog, deep=True)


def test_validate_rejects_ir_make_enum_with_unknown_variant() -> None:
    """Validator rejects IrMakeEnum whose variant is absent from the descriptor."""
    from agm.agl.ir.ids import Location, SourceId
    from agm.agl.ir.program import (
        ExecutableModule,
        ExecutableProgram,
        NominalDescriptor,
        SourceFile,
    )
    from agm.agl.ir.validate import validate_ir

    sid = SourceId(0)
    loc = Location(source_id=sid, start_offset=0, end_offset=1, start_line=1, start_col=0)
    nominal_id = NominalId(1)
    desc = NominalDescriptor(
        nominal=nominal_id,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Color",
        kind=NominalKind.ENUM,
        fields=(),
        variants=(VariantDescriptor(name="Red", fields=(), member=NominalId(2)),),
    )
    node = IrMakeEnum(
        location=loc,
        nominal=nominal_id,
        display_name="Color",
        variant="Purple",  # not in descriptor
        fields=(),
    )
    from agm.agl.modules.ids import ENTRY_ID as EID

    prog = ExecutableProgram(
        entry_module=EID,
        modules={EID: ExecutableModule(module_id=EID, initializers=(node,))},
        symbols={},
        nominals={
            nominal_id: desc,
            NominalId(2): NominalDescriptor(
                NominalId(2), EID, ("Color",), "Red", NominalKind.RECORD
            ),
        },
        sources={sid: SourceFile(display_name="<test>", normalized_text=" ")},
    )
    with pytest.raises(InvalidIrError, match="variant"):
        validate_ir(prog, deep=True)


@pytest.mark.parametrize(
    "variants",
    (
        (
            VariantDescriptor(name="same", fields=(), member=NominalId(2)),
            VariantDescriptor(name="same", fields=(), member=NominalId(3)),
        ),
        (
            VariantDescriptor(name="first", fields=(), member=NominalId(2)),
            VariantDescriptor(name="second", fields=(), member=NominalId(2)),
        ),
    ),
    ids=("duplicate-variant-name", "duplicate-member-nominal"),
)
def test_validate_rejects_duplicate_enum_descriptor_members(
    variants: tuple[VariantDescriptor, VariantDescriptor],
) -> None:
    """Deep validation requires each enum descriptor to identify each member once."""
    from agm.agl.ir.program import ExecutableModule, SourceFile
    from agm.agl.ir.validate import validate_ir

    enum = NominalId(1)
    member_one = NominalId(2)
    member_two = NominalId(3)
    program = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=())},
        symbols={},
        nominals={
            enum: NominalDescriptor(
                enum, ENTRY_ID, (), "Choice", NominalKind.ENUM, variants=variants
            ),
            member_one: NominalDescriptor(
                member_one, ENTRY_ID, ("Choice",), "first", NominalKind.RECORD
            ),
            member_two: NominalDescriptor(
                member_two, ENTRY_ID, ("Choice",), "second", NominalKind.RECORD
            ),
        },
        sources={SourceId(0): SourceFile(display_name="<test>", normalized_text="")},
    )

    with pytest.raises(InvalidIrError):
        validate_ir(program, deep=True)


def test_validate_accepts_valid_ir_make_record() -> None:
    """Validator accepts IrMakeRecord with a known nominal."""
    from agm.agl.ir.ids import Location, SourceId
    from agm.agl.ir.program import (
        ExecutableModule,
        ExecutableProgram,
        NominalDescriptor,
        SourceFile,
    )
    from agm.agl.ir.validate import validate_ir

    sid = SourceId(0)
    loc = Location(source_id=sid, start_offset=0, end_offset=1, start_line=1, start_col=0)
    nominal_id = NominalId(1)
    desc = NominalDescriptor(
        nominal=nominal_id,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Pt",
        kind=NominalKind.RECORD,
        fields=("x",),
        variants=(),
    )
    node = IrMakeRecord(
        location=loc,
        nominal=nominal_id,
        display_name="Pt",
        fields=(),
    )
    from agm.agl.modules.ids import ENTRY_ID as EID

    prog = ExecutableProgram(
        entry_module=EID,
        modules={EID: ExecutableModule(module_id=EID, initializers=(node,))},
        symbols={},
        nominals={nominal_id: desc},
        sources={sid: SourceFile(display_name="<test>", normalized_text=" ")},
    )
    validate_ir(prog, deep=True)  # must not raise


# ---------------------------------------------------------------------------
# NominalDescriptor / VariantDescriptor unit tests
# ---------------------------------------------------------------------------


def test_nominal_descriptor_record_defaults() -> None:
    """NominalDescriptor for a record has variants=() by default."""
    nom = NominalId(1)
    desc = NominalDescriptor(
        nominal=nom,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Foo",
        kind=NominalKind.RECORD,
        fields=("x", "y"),
    )
    assert desc.variants == ()
    assert desc.fields == ("x", "y")


def test_nominal_descriptor_enum_with_variants() -> None:
    """NominalDescriptor for an enum carries VariantDescriptor objects."""
    nom = NominalId(2)
    variants = (
        VariantDescriptor(name="Circle", fields=("radius",), member=NominalId(3)),
        VariantDescriptor(name="Square", fields=("side",), member=NominalId(4)),
    )
    desc = NominalDescriptor(
        nominal=nom,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Shape",
        kind=NominalKind.ENUM,
        fields=(),
        variants=variants,
    )
    assert len(desc.variants) == 2
    assert desc.variants[0].name == "Circle"
    assert desc.variants[0].fields == ("radius",)


def test_ir_make_record_node_frozen() -> None:
    """IrMakeRecord is a frozen dataclass."""
    from agm.agl.ir.ids import Location, SourceId

    sid = SourceId(0)
    loc = Location(source_id=sid, start_offset=0, end_offset=1, start_line=1, start_col=0)
    nom = NominalId(1)
    node = IrMakeRecord(location=loc, nominal=nom, display_name="Pt", fields=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(node, "display_name", "Other")


# ---------------------------------------------------------------------------
# Validate tests — non-deep mode (shallow structural checks only)
# ---------------------------------------------------------------------------


def test_validate_non_deep_accepts_unknown_nominal_in_ir_make_record() -> None:
    """Non-deep validation does not check program.nominals (deep=False).

    A program with an IrMakeRecord referencing an unknown nominal must pass
    shallow validation (location and sub-expr checks only).
    """
    from agm.agl.ir.ids import Location, SourceId
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import validate_ir

    sid = SourceId(0)
    loc = Location(source_id=sid, start_offset=0, end_offset=1, start_line=1, start_col=0)
    node = IrMakeRecord(
        location=loc,
        nominal=NominalId(1),
        display_name="Ghost",
        fields=(),
    )
    from agm.agl.modules.ids import ENTRY_ID as EID

    prog = ExecutableProgram(
        entry_module=EID,
        modules={EID: ExecutableModule(module_id=EID, initializers=(node,))},
        symbols={},
        nominals={},
        sources={sid: SourceFile(display_name="<test>", normalized_text=" ")},
    )
    validate_ir(prog, deep=False)  # must not raise


def test_validate_non_deep_accepts_unknown_nominal_in_ir_make_enum() -> None:
    """Non-deep validation skips nominal and variant checks for IrMakeEnum."""
    from agm.agl.ir.ids import Location, SourceId
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import validate_ir

    sid = SourceId(0)
    loc = Location(source_id=sid, start_offset=0, end_offset=1, start_line=1, start_col=0)
    node = IrMakeEnum(
        location=loc,
        nominal=NominalId(1),
        display_name="Ghost",
        variant="Purple",
        fields=(),
    )
    from agm.agl.modules.ids import ENTRY_ID as EID

    prog = ExecutableProgram(
        entry_module=EID,
        modules={EID: ExecutableModule(module_id=EID, initializers=(node,))},
        symbols={},
        nominals={},
        sources={sid: SourceFile(display_name="<test>", normalized_text=" ")},
    )
    validate_ir(prog, deep=False)  # must not raise


def test_validate_non_deep_accepts_unknown_nominal_in_ir_make_exception() -> None:
    """Non-deep validation skips nominal checks for IrMakeException."""
    from agm.agl.ir.ids import Location, SourceId
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import validate_ir

    sid = SourceId(0)
    loc = Location(source_id=sid, start_offset=0, end_offset=1, start_line=1, start_col=0)
    node = IrMakeException(
        location=loc,
        nominal=NominalId(1),
        display_name="Ghost",
        fields=(),
    )
    from agm.agl.modules.ids import ENTRY_ID as EID

    prog = ExecutableProgram(
        entry_module=EID,
        modules={EID: ExecutableModule(module_id=EID, initializers=(node,))},
        symbols={},
        nominals={},
        sources={sid: SourceFile(display_name="<test>", normalized_text=" ")},
    )
    validate_ir(prog, deep=False)  # must not raise


def test_validate_non_deep_accepts_ir_make_constructor_with_unknown_nominal() -> None:
    """Non-deep validation skips nominal/variant checks for IrMakeConstructor."""
    from agm.agl.ir.ids import Location, SourceId
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import validate_ir

    sid = SourceId(0)
    loc = Location(source_id=sid, start_offset=0, end_offset=1, start_line=1, start_col=0)
    node = IrMakeConstructor(
        location=loc,
        nominal=NominalId(1),
        display_name="Ghost",
        variant="Missing",
    )
    from agm.agl.modules.ids import ENTRY_ID as EID

    prog = ExecutableProgram(
        entry_module=EID,
        modules={EID: ExecutableModule(module_id=EID, initializers=(node,))},
        symbols={},
        nominals={},
        sources={sid: SourceFile(display_name="<test>", normalized_text=" ")},
    )
    validate_ir(prog, deep=False)  # must not raise


def test_validate_rejects_ir_make_constructor_variant_on_a_record() -> None:
    """A first-class variant constructor must name a linked enum variant."""
    from agm.agl.ir.ids import Location
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import validate_ir

    sid = SourceId(0)
    loc = Location(source_id=sid, start_offset=0, end_offset=1, start_line=1, start_col=0)
    nominal = NominalId(1)
    program = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={
            ENTRY_ID: ExecutableModule(
                module_id=ENTRY_ID,
                initializers=(IrMakeConstructor(loc, nominal, "Point", "not-a-variant"),),
            )
        },
        symbols={},
        nominals={
            nominal: NominalDescriptor(
                nominal, ENTRY_ID, (), "Point", NominalKind.RECORD, fields=("x",)
            )
        },
        sources={sid: SourceFile(display_name="<test>", normalized_text=" ")},
    )

    with pytest.raises(InvalidIrError):
        validate_ir(program, deep=True)
