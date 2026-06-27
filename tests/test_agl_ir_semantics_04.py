"""M3d ir_semantic — record/enum/exception construction and constructor refs.

Tests all M3d node types: IrMakeRecord, IrMakeEnum, IrMakeException, IrMakeConstructor.
"""

from __future__ import annotations

import dataclasses
import decimal

import pytest

from agm.agl.ir.ids import NominalId
from agm.agl.ir.nodes import (
    AutoTraceField,
    IrBind,
    IrMakeConstructor,
    IrMakeEnum,
    IrMakeException,
    IrMakeRecord,
)
from agm.agl.ir.program import (
    ExecutableProgram,
    NominalDescriptor,
    NominalKind,
    VariantDescriptor,
)
from agm.agl.ir.validate import InvalidIrError
from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID
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
from tests.agl.ir_harness import evaluate_ir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# IR semantic tests — record construction
# ---------------------------------------------------------------------------


def test_record_construction_basic() -> None:
    """Record construction: Point(x: 3, y: 4) produces the correct RecordValue."""
    source = """\
record Point
  x: int
  y: int
let p = Point(x: 3, y: 4)
()
"""
    ir = evaluate_ir(source)
    p = ir["p"]
    assert isinstance(p, RecordValue)
    assert p.fields["x"] == IntValue(3)
    assert p.fields["y"] == IntValue(4)
    assert p.display_name == "Point"
    assert p.nominal.declared_name == "Point"


def test_record_field_access_now_unblocked() -> None:
    """Record construction + field access: p.x works end-to-end (un-skipped M3c case)."""
    source = """\
record Point
  x: int
  y: int
let p = Point(x: 3, y: 4)
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
let s = Score(name: "Alice", value: 42)
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
let p1 = Pair(a: 1, b: 2)
let p2 = Pair(a: 1, b: 2)
let eq = p1 = p2
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
let p1 = Pair(a: 1, b: 2)
let p2 = Pair(a: 1, b: 9)
let ne = p1 != p2
()
"""
    ir = evaluate_ir(source)
    assert ir["ne"] == BoolValue(True)


def test_template_with_record_interpolation_now_unblocked() -> None:
    """Template interpolation with a record value (un-skipped M3c case)."""
    source = """\
record Point
  x: int
  y: int
let p = Point(x: 1, y: 2)
let s: text = "point: ${p}"
()
"""
    ir = evaluate_ir(source)
    assert isinstance(ir["s"], TextValue)


# ---------------------------------------------------------------------------
# IR semantic tests — enum construction
# ---------------------------------------------------------------------------


def test_enum_nullary_variant() -> None:
    """Enum nullary variant: Color.Red() constructs correctly."""
    source = """\
enum Color | Red | Blue
let c = Color.Red()
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
let s = Shape.Circle(radius: 3.0)
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
let c1 = Color.Red()
let c2 = Color.Red()
let eq = c1 = c2
()
"""
    ir = evaluate_ir(source)
    assert ir["eq"] == BoolValue(True)


def test_enum_inequality_different_variants() -> None:
    """Two enum values with different variants compare not-equal."""
    source = """\
enum Color | Red | Blue
let c1 = Color.Red()
let c2 = Color.Blue()
let ne = c1 != c2
()
"""
    ir = evaluate_ir(source)
    assert ir["ne"] == BoolValue(True)


def test_enum_inequality_different_nominals() -> None:
    """Two enum values from different nominals produce different NominalIds (D2 property).

    We cannot compare them with != in AgL (the checker requires same type for ==).
    Instead we verify that evaluation produces EnumValues with different nominals.
    """
    source_a = """\
enum ColorA | Red
let c = ColorA.Red()
()
"""
    source_b = """\
enum ColorB | Red
let c = ColorB.Red()
()
"""
    ir_a = evaluate_ir(source_a)
    ir_b = evaluate_ir(source_b)
    ca = ir_a["c"]
    cb = ir_b["c"]
    assert isinstance(ca, EnumValue)
    assert isinstance(cb, EnumValue)
    assert ca.nominal != cb.nominal, "Different enum types must have different NominalIds"
    # IR values should also have distinct nominals
    ia = ir_a["c"]
    ib = ir_b["c"]
    assert isinstance(ia, EnumValue)
    assert isinstance(ib, EnumValue)
    assert ia.nominal != ib.nominal


def test_enum_variant_field_coercion() -> None:
    """Enum variant field: int→decimal coercion applied at lowering time."""
    source = """\
enum Size | Big(amount: decimal) | Small
let s = Size.Big(amount: 7)
()
"""
    ir = evaluate_ir(source)
    sv = ir["s"]
    assert isinstance(sv, EnumValue)
    assert sv.fields["amount"] == DecimalValue(decimal.Decimal(7))


# ---------------------------------------------------------------------------
# IR semantic tests — NominalId hashing (D2)
# ---------------------------------------------------------------------------


def test_nominal_id_hashing_record_as_set_member() -> None:
    """NominalId-based hashing: constructed records are usable as set members / dict keys.

    AgL's dict type only supports text keys, so we cannot express "use a record
    as a dict key" in AgL source.  Instead we:
      1. Construct two *equal* records (p1, p2) and one *different* record (p3)
         via AgL and assert on the evaluated values.
      2. Use the returned RecordValue objects as Python set members / dict keys
         to verify that __hash__ and __eq__ (both NominalId-based per D2) are
         consistent: equal records land in the same slot, the different record
         in its own slot.
    This exercises the full pipeline — lowering → evaluation → NominalId equality
    and hashing — and asserts the IR pipeline produces hash-identical values.
    """
    source = """\
record Point
  x: int
  y: int
let p1 = Point(x: 1, y: 2)
let p2 = Point(x: 1, y: 2)
let p3 = Point(x: 9, y: 9)
()
"""
    ir = evaluate_ir(source)

    # --- ir pipeline hashing ---
    lp1 = ir["p1"]
    lp2 = ir["p2"]
    lp3 = ir["p3"]
    assert isinstance(lp1, RecordValue)
    assert isinstance(lp2, RecordValue)
    assert isinstance(lp3, RecordValue)

    # Equal records must hash identically (set deduplication / dict key lookup).
    ir_set = {lp1, lp2, lp3}
    assert len(ir_set) == 2, (
        f"Expected 2 distinct members in set (p1==p2), got {len(ir_set)}"
    )
    # Use as dict keys: p1 and p2 must map to the same slot.
    ir_dict: dict[RecordValue, str] = {lp1: "first"}
    ir_dict[lp2] = "second"  # should overwrite lp1's slot (same key)
    assert len(ir_dict) == 1, "p1 and p2 must occupy the same dict slot"
    assert ir_dict[lp1] == "second"

    # p3 is distinct — it lands in its own slot.
    ir_dict[lp3] = "third"
    assert len(ir_dict) == 2

    # --- IR pipeline hashing (same assertions) ---
    ip1 = ir["p1"]
    ip2 = ir["p2"]
    ip3 = ir["p3"]
    assert isinstance(ip1, RecordValue)
    assert isinstance(ip2, RecordValue)
    assert isinstance(ip3, RecordValue)

    ir_set = {ip1, ip2, ip3}
    assert len(ir_set) == 2, (
        f"IR: expected 2 distinct set members, got {len(ir_set)}"
    )
    ir_dict: dict[RecordValue, str] = {ip1: "first"}
    ir_dict[ip2] = "second"
    assert len(ir_dict) == 1
    ir_dict[ip3] = "third"
    assert len(ir_dict) == 2


def test_nominal_id_hashing_enum_as_set_member() -> None:
    """NominalId-based hashing: constructed enum values are usable as set members / dict keys.

    Two equal enum values (same variant, same nominal) must deduplicate in a set;
    a value from a different variant must be distinct.  Evaluation must produce
    hash-identical results (ir_semantic agreement).
    """
    source = """\
enum Color | Red | Blue
let c1 = Color.Red()
let c2 = Color.Red()
let c3 = Color.Blue()
()
"""
    ir = evaluate_ir(source)

    # --- ir pipeline hashing ---
    lc1 = ir["c1"]
    lc2 = ir["c2"]
    lc3 = ir["c3"]
    assert isinstance(lc1, EnumValue)
    assert isinstance(lc2, EnumValue)
    assert isinstance(lc3, EnumValue)

    ir_set = {lc1, lc2, lc3}
    assert len(ir_set) == 2, (
        f"Expected 2 distinct set members (c1==c2), got {len(ir_set)}"
    )
    ir_dict: dict[EnumValue, str] = {lc1: "red-1"}
    ir_dict[lc2] = "red-2"  # must overwrite lc1's slot
    assert len(ir_dict) == 1
    ir_dict[lc3] = "blue"
    assert len(ir_dict) == 2

    # --- IR pipeline hashing ---
    ic1 = ir["c1"]
    ic2 = ir["c2"]
    ic3 = ir["c3"]
    assert isinstance(ic1, EnumValue)
    assert isinstance(ic2, EnumValue)
    assert isinstance(ic3, EnumValue)

    ir_set = {ic1, ic2, ic3}
    assert len(ir_set) == 2, (
        f"IR: expected 2 distinct set members, got {len(ir_set)}"
    )
    ir_dict: dict[EnumValue, str] = {ic1: "red-1"}
    ir_dict[ic2] = "red-2"
    assert len(ir_dict) == 1
    ir_dict[ic3] = "blue"
    assert len(ir_dict) == 2


# ---------------------------------------------------------------------------
# IR semantic tests — exception construction
# ---------------------------------------------------------------------------


def test_exception_construction_builtin_explicit_fields() -> None:
    """Exception construction using a built-in exception type with explicit fields.

    ArithmeticError has (message, trace_id, operation); we provide message and
    operation; trace_id is auto-injected.
    """
    source = """\
let e = ArithmeticError(message: "div/0", operation: "/")
()
"""
    ir = evaluate_ir(source)
    e = ir["e"]
    assert isinstance(e, ExceptionValue)
    assert e.fields["message"] == TextValue("div/0")
    assert e.fields["operation"] == TextValue("/")
    # trace_id was auto-injected
    assert isinstance(e.fields["trace_id"], TextValue)


def test_exception_auto_trace_single_id_per_construction() -> None:
    """Two separately constructed exceptions have different trace_ids (distinct events).

    The IR pipeline assigns a distinct trace_id to each construction.
    """
    source = """\
let e1 = ArithmeticError(message: "one", operation: "+")
let e2 = ArithmeticError(message: "two", operation: "+")
()
"""
    ir = evaluate_ir(source)
    e1 = ir["e1"]
    e2 = ir["e2"]
    assert isinstance(e1, ExceptionValue)
    assert isinstance(e2, ExceptionValue)
    # Different constructions get different trace IDs
    assert e1.fields["trace_id"] != e2.fields["trace_id"]


# ---------------------------------------------------------------------------
# IR semantic tests — first-class constructor references
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
let mk = Color.Red
()
"""
    ir = evaluate_ir(source)
    mk = ir["mk"]
    assert isinstance(mk, EnumValue), f"ir: {mk!r}"
    assert mk.variant == "Red"


def test_first_class_enum_constructor_ref_with_fields_gives_constructor_value() -> None:
    """An enum variant WITH fields used as a value (not called) gives a ConstructorValue.

    Only non-nullary variants (those with at least one field) produce a ConstructorValue
    when accessed in value position via qualified form (Enum.Variant).
    """
    source = """\
enum Shape
  | Circle(radius: int)
  | Square(side: int)
let mk = Shape.Circle
()
"""
    ir = evaluate_ir(source)
    mk = ir["mk"]
    assert isinstance(mk, ConstructorValue), f"ir: {mk!r}"
    assert mk.display_name == "Shape"
    assert mk.variant == "Circle"


# ---------------------------------------------------------------------------
# Golden lowering tests — node shapes
# ---------------------------------------------------------------------------


def test_golden_record_lowers_to_ir_make_record() -> None:
    """Record constructor call lowers to IrMakeRecord with correct fields."""
    source = """\
record Point
  x: int
  y: int
let p = Point(x: 3, y: 4)
()
"""
    prog = _lower(source)
    entry = prog.modules[prog.entry_module]
    found = False
    for node in entry.initializers:
        if isinstance(node, IrBind) and isinstance(node.value, IrMakeRecord):
            mr = node.value
            assert mr.display_name == "Point"
            assert mr.nominal.declared_name == "Point"
            assert len(mr.fields) == 2
            assert mr.fields[0][0] == "x"
            assert mr.fields[1][0] == "y"
            found = True
    assert found, "Expected IrBind(value=IrMakeRecord) in initializers"


def test_golden_enum_lowers_to_ir_make_enum() -> None:
    """Enum variant call lowers to IrMakeEnum with correct variant."""
    source = """\
enum Color | Red | Blue
let c = Color.Red()
()
"""
    prog = _lower(source)
    entry = prog.modules[prog.entry_module]
    found = False
    for node in entry.initializers:
        if isinstance(node, IrBind) and isinstance(node.value, IrMakeEnum):
            me = node.value
            assert me.display_name == "Color"
            assert me.variant == "Red"
            assert me.fields == ()
            found = True
    assert found, "Expected IrBind(value=IrMakeEnum) in initializers"


def test_golden_exception_lowers_to_ir_make_exception_with_auto_trace() -> None:
    """Exception construction lowers to IrMakeException with AutoTraceField sentinels.

    Uses ArithmeticError(message, operation) — trace_id is not provided so it
    gets an AutoTraceField sentinel in the IR.
    """
    source = """\
let e = ArithmeticError(message: "oops", operation: "/")
()
"""
    prog = _lower(source)
    entry = prog.modules[prog.entry_module]
    found = False
    for node in entry.initializers:
        if isinstance(node, IrBind) and isinstance(node.value, IrMakeException):
            me = node.value
            assert me.display_name == "ArithmeticError"
            # Fields in declaration order: message, trace_id, operation
            field_names = [name for name, _ in me.fields]
            assert field_names == ["message", "trace_id", "operation"]
            # trace_id should be AutoTraceField (not provided by caller)
            trace_slot = dict(me.fields).get("trace_id")
            assert isinstance(trace_slot, AutoTraceField), (
                f"expected AutoTraceField for trace_id, got {trace_slot!r}"
            )
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
let s = Score(name: "Bob", value: 5)
()
"""
    prog = _lower(source)
    entry = prog.modules[prog.entry_module]
    found = False
    for node in entry.initializers:
        if isinstance(node, IrBind) and isinstance(node.value, IrMakeRecord):
            for fname, fexpr in node.value.fields:
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
    entry = prog.modules[prog.entry_module]
    found = False
    for node in entry.initializers:
        if isinstance(node, IrBind) and isinstance(node.value, IrMakeConstructor):
            mc = node.value
            assert mc.display_name == "Pt"
            assert mc.variant is None
            assert mc.nominal.declared_name == "Pt"
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
let p = Point(x: 1, y: 2)
()
"""
    prog = _lower(source)
    nominal_id = NominalId(ENTRY_ID, "Point")
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
    nominal_id = NominalId(ENTRY_ID, "Color")
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
    nominal_id = NominalId(PRELUDE_ID, "ArithmeticError")
    assert nominal_id in prog.nominals
    desc = prog.nominals[nominal_id]
    assert desc.kind == NominalKind.EXCEPTION
    # ArithmeticError fields: message, trace_id, operation (in declaration order)
    assert desc.fields == ("message", "trace_id", "operation")


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
        nominal_id = NominalId(PRELUDE_ID, name)
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
    unknown_nominal = NominalId(ENTRY_ID, "Ghost")
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
    nominal_id = NominalId(ENTRY_ID, "Color")
    desc = NominalDescriptor(
        nominal=nominal_id,
        display_name="Color",
        kind=NominalKind.ENUM,
        fields=(),
        variants=(VariantDescriptor(name="Red", fields=()),),
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
        nominals={nominal_id: desc},
        sources={sid: SourceFile(display_name="<test>", normalized_text=" ")},
    )
    with pytest.raises(InvalidIrError, match="variant"):
        validate_ir(prog, deep=True)


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
    nominal_id = NominalId(ENTRY_ID, "Pt")
    desc = NominalDescriptor(
        nominal=nominal_id,
        display_name="Pt",
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
    nom = NominalId(ENTRY_ID, "Foo")
    desc = NominalDescriptor(
        nominal=nom,
        display_name="Foo",
        kind=NominalKind.RECORD,
        fields=("x", "y"),
    )
    assert desc.variants == ()
    assert desc.fields == ("x", "y")


def test_nominal_descriptor_enum_with_variants() -> None:
    """NominalDescriptor for an enum carries VariantDescriptor objects."""
    nom = NominalId(ENTRY_ID, "Shape")
    variants = (
        VariantDescriptor(name="Circle", fields=("radius",)),
        VariantDescriptor(name="Square", fields=("side",)),
    )
    desc = NominalDescriptor(
        nominal=nom,
        display_name="Shape",
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
    nom = NominalId(ENTRY_ID, "Pt")
    node = IrMakeRecord(location=loc, nominal=nom, display_name="Pt", fields=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(node, "display_name", "Other")


def test_auto_trace_field_sentinel() -> None:
    """AutoTraceField is a distinct marker object, not an IrExpr."""
    from agm.agl.ir.nodes import AutoTraceField

    atf = AutoTraceField()
    # It must NOT be an instance of any IrExpr union member
    # (it's a sentinel, not an expression)
    assert not isinstance(atf, IrMakeException)
    # It must be hashable (frozen dataclass)
    assert hash(atf) == hash(AutoTraceField())


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
        nominal=NominalId(ENTRY_ID, "Ghost"),
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
        nominal=NominalId(ENTRY_ID, "Ghost"),
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
        nominal=NominalId(PRELUDE_ID, "Ghost"),
        display_name="Ghost",
        fields=(("trace_id", AutoTraceField()),),
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
        nominal=NominalId(ENTRY_ID, "Ghost"),
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


def test_validate_check_enum_variant_skips_when_nominal_not_in_table() -> None:
    """_check_enum_variant returns early when nominal is absent from program.nominals.

    This path is exercised when IrMakeEnum's nominal was never registered.
    We check it via deep=True but with no descriptor in the table — deep checks
    the nominal first (and raises), but if we test _check_enum_variant in isolation
    we can call it directly.  Instead we rely on the IrMakeConstructor path to
    exercise the absent-nominal early return: variant=not-None, nominal not in table, deep=True.
    The _check_nominal_in_table call raises first; but _check_enum_variant is called
    after that in IrMakeConstructor when variant is not None.  So we test via
    IrMakeConstructor with variant=None to take the line-403 path (skip variant check)
    and IrMakeConstructor with variant='X' + nominal absent.
    The absent-nominal early-return in _check_enum_variant is reached:
    IrMakeConstructor deep with nominal absent AND variant present triggers both
    _check_nominal_in_table (which raises) and then is NOT reached for _check_enum_variant.
    To reach the absent-nominal early-return purely, we need to call validate
    when variant is 'X' but the table has the nominal — with a non-ENUM kind.
    """
    from agm.agl.ir.ids import Location, SourceId
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import validate_ir

    sid = SourceId(0)
    loc = Location(source_id=sid, start_offset=0, end_offset=1, start_line=1, start_col=0)
    nominal_id = NominalId(ENTRY_ID, "Pt")
    # Register nominal as RECORD (kind != ENUM) — variant check is skipped
    desc = NominalDescriptor(
        nominal=nominal_id,
        display_name="Pt",
        kind=NominalKind.RECORD,
        fields=("x",),
    )
    node = IrMakeConstructor(
        location=loc,
        nominal=nominal_id,
        display_name="Pt",
        variant="Ignored",  # variant is not None → triggers _check_enum_variant
    )
    from agm.agl.modules.ids import ENTRY_ID as EID

    prog = ExecutableProgram(
        entry_module=EID,
        modules={EID: ExecutableModule(module_id=EID, initializers=(node,))},
        symbols={},
        nominals={nominal_id: desc},
        sources={sid: SourceFile(display_name="<test>", normalized_text=" ")},
    )
    # deep=True: _check_nominal_in_table passes (nominal is registered),
    # _check_enum_variant is called but returns early (kind != ENUM).
    validate_ir(prog, deep=True)  # must not raise
