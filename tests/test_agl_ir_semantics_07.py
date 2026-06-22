"""M3f-B differential ir_semantic — case expressions and match plans.

Covers IrCase with all pattern kinds:
- literal patterns (int/decimal/bool/text/null)
- binder pattern (VarPattern as binder)
- wildcard pattern
- nullary bare-variant patterns (VarPattern as nullary constructor)
- constructor patterns with field destructuring
- nested patterns (constructor containing literal/binder/nested-constructor)
- non-exhaustive no-match raises MatchError with correct scrutinee_type/scrutinee
- first-match ordering (earlier arm shadows later)
- case binders do NOT leak into top-level results name-set
- golden lowering: each plan kind (wildcard/bind→SymbolId/literal→IrConst/variant/constructor)
- defensive evaluator tests (hand-built IR): variant/constructor plan on non-enum → InvalidIrError
- negative validate test: IrBindPlan symbol missing from program.symbols


AgL syntax notes used in test programs:
- Enum definition: ``enum Name | Variant1 | Variant2`` (pipe-separated, no braces)
- Nullary constructor (value): ``Name.Variant()`` with empty parens
- Nullary constructor (pattern, ConstructorPattern): ``| Variant() => body``
- Bare-variant pattern (VarPattern, bare_variant_patterns): ``| Variant => body``
  (bare name that resolves to a constructor in scope)
- Constructor with fields (pattern): ``| Variant(field: binder) => body``
- Program must end with an expression (not a let/var decl).
"""

from __future__ import annotations

import pytest

from agm.agl.eval.exceptions import AglRaise
from agm.agl.eval.values import (
    IntValue,
    JsonValue,
    TextValue,
)
from agm.agl.ir.ids import Location, NominalId, SourceId, SymbolId
from agm.agl.ir.nodes import (
    IrBind,
    IrBindPlan,
    IrCase,
    IrCaseArm,
    IrConstBool,
    IrConstInt,
    IrConstructorPlan,
    IrConstText,
    IrExpr,
    IrLiteralPlan,
    IrLoad,
    IrVariantPlan,
    IrWildcardPlan,
)
from agm.agl.ir.program import (
    ExecutableModule,
    ExecutableProgram,
    NominalDescriptor,
    NominalKind,
    SourceFile,
    SymbolDescriptor,
    VariantDescriptor,
)
from agm.agl.ir.validate import InvalidIrError, validate_ir
from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID
from tests.agl.ir_harness import evaluate_ir, evaluate_ir_raises

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_SOURCE = "let x = 1\nx"
_DUMMY_SOURCE_ID = SourceId(0)
_DUMMY_LOC = Location(
    source_id=_DUMMY_SOURCE_ID,
    start_offset=0,
    end_offset=1,
    start_line=1,
    start_col=0,
)

_PRELUDE_NOM = NominalId(PRELUDE_ID, "MatchError")


def _make_program(
    initializers: tuple[IrExpr, ...],
    *,
    source: str = _DUMMY_SOURCE,
    symbols: dict[SymbolId, SymbolDescriptor] | None = None,
    nominals: dict[NominalId, NominalDescriptor] | None = None,
) -> ExecutableProgram:
    """Build a minimal ExecutableProgram for hand-built IR tests."""
    src_id = SourceId(0)
    return ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={
            ENTRY_ID: ExecutableModule(
                module_id=ENTRY_ID,
                initializers=initializers,
            )
        },
        symbols=symbols or {},
        nominals=nominals or {},
        sources={src_id: SourceFile(display_name="<test>", normalized_text=source)},
    )


def _make_enum_nom() -> tuple[NominalId, NominalDescriptor]:
    """Return a minimal Color enum nominal (Red/Blue) for hand-built tests."""
    nom = NominalId(ENTRY_ID, "Color")
    desc = NominalDescriptor(
        nominal=nom,
        display_name="Color",
        kind=NominalKind.ENUM,
        variants=(
            VariantDescriptor(name="Red", fields=()),
            VariantDescriptor(name="Blue", fields=()),
        ),
    )
    return nom, desc


# ---------------------------------------------------------------------------
# IR semantic tests — literal patterns
# ---------------------------------------------------------------------------


def test_ir_semantic_case_literal_int_match() -> None:
    """int literal pattern: first matching arm is taken."""
    src = """\
let x: int = 2
let r = case x of
  | 1 => "one"
  | 2 => "two"
  | _ => "other"
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == TextValue("two")


def test_ir_semantic_case_literal_int_fallthrough_wildcard() -> None:
    """Wildcard default arm catches when no literal matches."""
    src = """\
let x: int = 99
let r = case x of
  | 1 => "one"
  | _ => "other"
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == TextValue("other")


def test_ir_semantic_case_literal_bool() -> None:
    """bool literal pattern."""
    src = """\
let x = true
let r = case x of
  | false => "no"
  | true => "yes"
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == TextValue("yes")


def test_ir_semantic_case_literal_text() -> None:
    """text (string) literal pattern."""
    src = """\
let x = "hello"
let r = case x of
  | "world" => 0
  | "hello" => 1
  | _ => 2
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == IntValue(1)


def test_ir_semantic_case_literal_null() -> None:
    """null literal pattern."""
    src = """\
let x: json = null
let r = case x of
  | null => "got null"
  | _ => "other"
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == TextValue("got null")


def test_ir_semantic_case_first_match_ordering() -> None:
    """Earlier arm shadows a later arm that would also match."""
    src = """\
let x: int = 1
let r = case x of
  | 1 => "first"
  | 1 => "second"
  | _ => "other"
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == TextValue("first")


# ---------------------------------------------------------------------------
# IR semantic tests — binder and wildcard patterns
# ---------------------------------------------------------------------------


def test_ir_semantic_case_binder_pattern() -> None:
    """VarPattern binder captures value and body uses it."""
    src = """\
let x: int = 42
let r = case x of
  | n => n
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == IntValue(42)


def test_ir_semantic_case_wildcard_pattern() -> None:
    """Wildcard pattern matches without binding."""
    src = """\
let x: int = 7
let r = case x of
  | _ => "matched"
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == TextValue("matched")


def test_ir_semantic_case_binder_does_not_leak() -> None:
    """Case binder symbol does not appear in top-level result names."""
    src = """\
let x: int = 5
let r = case x of
  | bound_var => bound_var
r"""
    ir_reference, ir = evaluate_ir(src)
    # 'bound_var' must not appear as a top-level name
    assert "bound_var" not in ir, (
        f"Case binder 'bound_var' leaked into IR results: {sorted(ir.keys())}"
    )
    assert "bound_var" not in ir_reference, (
        f"Case binder 'bound_var' leaked into ir_reference results: {sorted(ir_reference.keys())}"
    )
    assert ir["r"] == IntValue(5)


# ---------------------------------------------------------------------------
# IR semantic tests — nullary bare-variant patterns
# ---------------------------------------------------------------------------


def test_ir_semantic_case_nullary_variant_match() -> None:
    """VarPattern as bare-variant: bare name that resolves to a constructor."""
    # Using bare names (VarPattern classified as bare_variant_patterns by scope resolver)
    src = """\
enum Flag | On | Off
let f = Flag.On()
let r = case f of
  | Off => 0
  | On => 1
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == IntValue(1)


def test_ir_semantic_case_nullary_variant_no_binding() -> None:
    """Nullary bare-variant match does not bind anything."""
    src = """\
enum Flag | On | Off
let f = Flag.Off()
let r = case f of
  | On => "on"
  | Off => "off"
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == TextValue("off")


def test_ir_semantic_case_nullary_constructor_pattern() -> None:
    """ConstructorPattern with no fields (Red()) matches the variant."""
    src = """\
enum Color | Red | Blue
let c = Color.Red()
let r = case c of
  | Blue() => "blue"
  | Red() => "red"
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == TextValue("red")


# ---------------------------------------------------------------------------
# IR semantic tests — constructor patterns (with fields)
# ---------------------------------------------------------------------------


def test_ir_semantic_case_constructor_field_destructure() -> None:
    """ConstructorPattern destructures enum variant fields."""
    src = """\
enum Shape | Circle(radius: int) | Square(side: int)
let s = Shape.Circle(radius: 5)
let r = case s of
  | Circle(radius: n) => n
  | Square(side: m) => m
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == IntValue(5)


def test_ir_semantic_case_constructor_field_no_match_fallback() -> None:
    """Constructor pattern on wrong variant falls through to next arm."""
    src = """\
enum Shape | Circle(radius: int) | Square(side: int)
let s = Shape.Square(side: 10)
let r = case s of
  | Circle(radius: n) => n
  | Square(side: m) => m
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == IntValue(10)


def test_ir_semantic_case_constructor_nested_literal() -> None:
    """Constructor pattern with nested literal sub-pattern."""
    src = """\
enum Shape | Circle(radius: int) | Square(side: int)
let s = Shape.Circle(radius: 3)
let r = case s of
  | Circle(radius: 3) => "three"
  | Circle(radius: n) => "other"
  | _ => "not circle"
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == TextValue("three")


def test_ir_semantic_case_constructor_nested_binder() -> None:
    """Constructor pattern with nested binder sub-pattern captures field."""
    src = """\
enum Shape | Circle(radius: int) | Square(side: int)
let s = Shape.Circle(radius: 7)
let r = case s of
  | Square(side: x) => x
  | Circle(radius: n) => n
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == IntValue(7)


def test_ir_semantic_case_constructor_nested_wildcard() -> None:
    """Constructor pattern with nested wildcard sub-pattern."""
    src = """\
enum Shape | Circle(radius: int) | Square(side: int)
let s = Shape.Square(side: 99)
let r = case s of
  | Circle(radius: _) => "circle"
  | Square(side: _) => "square"
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == TextValue("square")


def test_ir_semantic_case_constructor_nested_constructor() -> None:
    """Nested: constructor pattern with nested bare-variant sub-pattern."""
    src = """\
enum Color | Red | Blue
enum Shape | Colored(size: int)
let s = Shape.Colored(size: 10)
let r = case s of
  | Colored(size: n) => n
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == IntValue(10)


def test_ir_semantic_case_constructor_multi_field() -> None:
    """Constructor pattern matching multiple fields, first field returned."""
    src = """\
enum Point | Pt(x: int, y: int)
let p = Point.Pt(x: 3, y: 4)
let r = case p of
  | Pt(x: a, y: b) => a
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == IntValue(3)


# ---------------------------------------------------------------------------
# IR semantic tests — no-match raises MatchError
# ---------------------------------------------------------------------------


def test_ir_semantic_case_no_match_raises_match_error() -> None:
    """Non-exhaustive case raises MatchError with scrutinee_type and scrutinee."""
    src = """\
let x: int = 5
let r = case x of
  | 1 => "one"
  | 2 => "two"
r"""
    ir_reference_exc, ir_exc = evaluate_ir_raises(src)
    # Both sides must produce MatchError
    assert ir_reference_exc.display_name == "MatchError"
    assert ir_exc.display_name == "MatchError"
    # scrutinee_type must match
    assert ir_reference_exc.fields["scrutinee_type"] == ir_exc.fields["scrutinee_type"]
    assert ir_reference_exc.fields["scrutinee_type"] == TextValue("int")
    # scrutinee JSON must match (trace_id normalized by evaluate_ir_raises)
    assert ir_reference_exc.fields["scrutinee"] == ir_exc.fields["scrutinee"]
    assert ir_reference_exc.fields["scrutinee"] == JsonValue(5)


def test_ir_semantic_case_no_match_enum_scrutinee_type() -> None:
    """MatchError scrutinee_type for an enum value uses the enum display_name."""
    src = """\
enum Color | Red | Blue
let c = Color.Red()
let r = case c of
  | Blue() => "blue"
r"""
    ir_reference_exc, ir_exc = evaluate_ir_raises(src)
    assert ir_exc.display_name == "MatchError"
    assert ir_reference_exc.fields["scrutinee_type"] == ir_exc.fields["scrutinee_type"]
    assert ir_exc.fields["scrutinee_type"] == TextValue("Color")


# ---------------------------------------------------------------------------
# IR semantic tests — mixed arms, first-match ordering with constructors
# ---------------------------------------------------------------------------


def test_ir_semantic_case_first_match_constructor_then_wildcard() -> None:
    """Constructor arm first, then wildcard catches all others."""
    src = """\
enum Shape | Circle(radius: int) | Square(side: int)
let s = Shape.Square(side: 3)
let r = case s of
  | Circle(radius: _) => "circle"
  | _ => "other"
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == TextValue("other")


# ---------------------------------------------------------------------------
# Golden lowering helpers
# ---------------------------------------------------------------------------


def _lower(source: str) -> ExecutableProgram:
    """Parse → resolve → check → lower; return ExecutableProgram with validate=True."""
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
    return lower_program(checked, source_text=source, source_label="<test>", validate=True)


def _find_r_bind(executable: ExecutableProgram) -> IrBind:
    """Return the IrBind for the public binding named 'r'."""
    return next(
        n for n in executable.modules[executable.entry_module].initializers
        if isinstance(n, IrBind) and (
            executable.symbols.get(n.symbol) is not None
            and executable.symbols[n.symbol].public_name == "r"
        )
    )


# ---------------------------------------------------------------------------
# Golden lowering tests (verify IR plan structure)
# ---------------------------------------------------------------------------


def test_golden_lowering_wildcard_plan() -> None:
    """IrWildcardPlan is emitted for a wildcard pattern."""
    src = """\
let x: int = 1
let r = case x of
  | _ => 0
r"""
    executable = _lower(src)
    r_bind = _find_r_bind(executable)
    case_node = r_bind.value
    assert isinstance(case_node, IrCase)
    assert len(case_node.arms) == 1
    assert isinstance(case_node.arms[0].plan, IrWildcardPlan)


def test_golden_lowering_bind_plan() -> None:
    """IrBindPlan carries a SymbolId for a VarPattern binder."""
    src = """\
let x: int = 1
let r = case x of
  | n => n
r"""
    executable = _lower(src)
    r_bind = _find_r_bind(executable)
    case_node = r_bind.value
    assert isinstance(case_node, IrCase)
    plan = case_node.arms[0].plan
    assert isinstance(plan, IrBindPlan)
    # The SymbolId must be registered in program.symbols
    assert plan.symbol in executable.symbols
    # And it must be private (public_name is None)
    assert executable.symbols[plan.symbol].public_name is None


def test_golden_lowering_literal_plan() -> None:
    """IrLiteralPlan carries the lowered IrConst* for a LiteralPattern."""
    src = """\
let x: int = 1
let r = case x of
  | 42 => "yes"
  | _ => "no"
r"""
    executable = _lower(src)
    r_bind = _find_r_bind(executable)
    case_node = r_bind.value
    assert isinstance(case_node, IrCase)
    plan = case_node.arms[0].plan
    assert isinstance(plan, IrLiteralPlan)
    assert isinstance(plan.value, IrConstInt)
    assert plan.value.value == 42


def test_golden_lowering_variant_plan() -> None:
    """IrVariantPlan is emitted for a nullary bare-variant VarPattern."""
    src = """\
enum Flag | On | Off
let f = Flag.On()
let r = case f of
  | On => 1
  | Off => 0
r"""
    executable = _lower(src)
    r_bind = _find_r_bind(executable)
    case_node = r_bind.value
    assert isinstance(case_node, IrCase)
    plan = case_node.arms[0].plan
    assert isinstance(plan, IrVariantPlan)
    assert plan.variant == "On"


def test_golden_lowering_constructor_plan() -> None:
    """IrConstructorPlan is emitted for a ConstructorPattern."""
    src = """\
enum Shape | Circle(radius: int)
let s = Shape.Circle(radius: 5)
let r = case s of
  | Circle(radius: n) => n
r"""
    executable = _lower(src)
    r_bind = _find_r_bind(executable)
    case_node = r_bind.value
    assert isinstance(case_node, IrCase)
    plan = case_node.arms[0].plan
    assert isinstance(plan, IrConstructorPlan)
    assert plan.variant == "Circle"
    assert len(plan.fields) == 1
    fname, subplan = plan.fields[0]
    assert fname == "radius"
    assert isinstance(subplan, IrBindPlan)


# ---------------------------------------------------------------------------
# Defensive evaluator tests (hand-built IR)
# ---------------------------------------------------------------------------


def test_defensive_variant_plan_on_non_enum_raises_invalid_ir() -> None:
    """IrVariantPlan applied to a non-EnumValue raises InvalidIrError."""
    from agm.agl.eval.ir_interpreter import IrInterpreter

    # Build: IrCase subject=IrConstInt(42), arms=[IrVariantPlan("Red") => ...]
    prog = _make_program(
        (
            IrCase(
                location=_DUMMY_LOC,
                subject=IrConstInt(location=_DUMMY_LOC, value=42),
                arms=(
                    IrCaseArm(
                        plan=IrVariantPlan(variant="Red"),
                        body=IrConstInt(location=_DUMMY_LOC, value=1),
                    ),
                ),
            ),
        )
    )
    interp = IrInterpreter(prog)
    with pytest.raises(InvalidIrError):
        interp.run()


def test_defensive_constructor_plan_on_non_enum_raises_invalid_ir() -> None:
    """IrConstructorPlan applied to a non-EnumValue raises InvalidIrError."""
    from agm.agl.eval.ir_interpreter import IrInterpreter

    prog = _make_program(
        (
            IrCase(
                location=_DUMMY_LOC,
                subject=IrConstText(location=_DUMMY_LOC, value="hello"),
                arms=(
                    IrCaseArm(
                        plan=IrConstructorPlan(variant="Foo", fields=()),
                        body=IrConstInt(location=_DUMMY_LOC, value=1),
                    ),
                ),
            ),
        )
    )
    interp = IrInterpreter(prog)
    with pytest.raises(InvalidIrError):
        interp.run()


def test_defensive_no_arm_matches_raises_match_error() -> None:
    """Hand-built IrCase with no matching arm raises AglRaise(MatchError)."""
    from agm.agl.eval.ir_interpreter import IrInterpreter

    # IrCase subject=true, arms=[IrLiteralPlan(false) => 0]: no match → MatchError
    prog = _make_program(
        (
            IrCase(
                location=_DUMMY_LOC,
                subject=IrConstBool(location=_DUMMY_LOC, value=True),
                arms=(
                    IrCaseArm(
                        plan=IrLiteralPlan(
                            value=IrConstBool(location=_DUMMY_LOC, value=False)
                        ),
                        body=IrConstInt(location=_DUMMY_LOC, value=0),
                    ),
                ),
            ),
        )
    )
    interp = IrInterpreter(prog)
    with pytest.raises(AglRaise) as exc_info:
        interp.run()
    exc = exc_info.value
    assert exc.exc.display_name == "MatchError"
    assert exc.exc.fields["scrutinee_type"] == TextValue("bool")


def test_defensive_wildcard_arm_matches_anything() -> None:
    """Hand-built IrCase with IrWildcardPlan always matches."""
    from agm.agl.eval.ir_interpreter import IrInterpreter

    prog = _make_program(
        (
            IrCase(
                location=_DUMMY_LOC,
                subject=IrConstText(location=_DUMMY_LOC, value="anything"),
                arms=(
                    IrCaseArm(
                        plan=IrWildcardPlan(),
                        body=IrConstInt(location=_DUMMY_LOC, value=99),
                    ),
                ),
            ),
        )
    )
    interp = IrInterpreter(prog)
    result = interp.run()
    # The case arm body yields 99 but is not exported (no IrBind wrapping the IrCase)
    assert result == {}


def test_defensive_bind_plan_writes_to_frame() -> None:
    """IrBindPlan writes the value into the frame and body can read it."""
    from agm.agl.eval.ir_interpreter import IrInterpreter

    sym = SymbolId(0)
    binder_sym = SymbolId(1)
    binder_desc = SymbolDescriptor(
        symbol_id=binder_sym,
        mutable=False,
        public_name=None,  # private
        owner=ENTRY_ID,
    )
    result_desc = SymbolDescriptor(
        symbol_id=sym,
        mutable=False,
        public_name="r",
        owner=ENTRY_ID,
    )
    prog = _make_program(
        (
            IrBind(
                location=_DUMMY_LOC,
                symbol=sym,
                value=IrCase(
                    location=_DUMMY_LOC,
                    subject=IrConstInt(location=_DUMMY_LOC, value=42),
                    arms=(
                        IrCaseArm(
                            plan=IrBindPlan(symbol=binder_sym),
                            body=IrLoad(location=_DUMMY_LOC, symbol=binder_sym),
                        ),
                    ),
                ),
            ),
        ),
        symbols={
            sym: result_desc,
            binder_sym: binder_desc,
        },
    )
    interp = IrInterpreter(prog)
    result = interp.run()
    assert result == {"r": IntValue(42)}


# ---------------------------------------------------------------------------
# Negative validate tests
# ---------------------------------------------------------------------------


def test_validate_ircase_bind_plan_symbol_missing() -> None:
    """validate_ir raises InvalidIrError when IrBindPlan references unknown SymbolId."""
    missing_sym = SymbolId(999)
    prog = _make_program(
        (
            IrCase(
                location=_DUMMY_LOC,
                subject=IrConstInt(location=_DUMMY_LOC, value=1),
                arms=(
                    IrCaseArm(
                        plan=IrBindPlan(symbol=missing_sym),
                        body=IrConstInt(location=_DUMMY_LOC, value=0),
                    ),
                ),
            ),
        )
    )
    with pytest.raises(InvalidIrError):
        validate_ir(prog, deep=True)


def test_validate_ircase_body_recurses() -> None:
    """validate_ir recurses into IrCase arm bodies."""
    bad_sym = SymbolId(999)
    prog = _make_program(
        (
            IrCase(
                location=_DUMMY_LOC,
                subject=IrConstInt(location=_DUMMY_LOC, value=1),
                arms=(
                    IrCaseArm(
                        plan=IrWildcardPlan(),
                        body=IrLoad(location=_DUMMY_LOC, symbol=bad_sym),
                    ),
                ),
            ),
        )
    )
    with pytest.raises(InvalidIrError):
        validate_ir(prog, deep=True)


def test_validate_ircase_subject_recurses() -> None:
    """validate_ir recurses into IrCase subject expression."""
    bad_sym = SymbolId(999)
    prog = _make_program(
        (
            IrCase(
                location=_DUMMY_LOC,
                subject=IrLoad(location=_DUMMY_LOC, symbol=bad_sym),
                arms=(
                    IrCaseArm(
                        plan=IrWildcardPlan(),
                        body=IrConstInt(location=_DUMMY_LOC, value=0),
                    ),
                ),
            ),
        )
    )
    with pytest.raises(InvalidIrError):
        validate_ir(prog, deep=True)


def test_validate_ircase_literal_plan_recurses() -> None:
    """validate_ir recurses into IrLiteralPlan's value expr."""
    bad_sym = SymbolId(999)
    prog = _make_program(
        (
            IrCase(
                location=_DUMMY_LOC,
                subject=IrConstInt(location=_DUMMY_LOC, value=1),
                arms=(
                    IrCaseArm(
                        plan=IrLiteralPlan(value=IrLoad(location=_DUMMY_LOC, symbol=bad_sym)),
                        body=IrConstInt(location=_DUMMY_LOC, value=0),
                    ),
                ),
            ),
        )
    )
    with pytest.raises(InvalidIrError):
        validate_ir(prog, deep=True)


def test_validate_ircase_constructor_plan_recurses() -> None:
    """validate_ir recurses into IrConstructorPlan sub-plans."""
    missing_sym = SymbolId(999)
    nom, nom_desc = _make_enum_nom()
    prog = _make_program(
        (
            IrCase(
                location=_DUMMY_LOC,
                subject=IrConstInt(location=_DUMMY_LOC, value=1),
                arms=(
                    IrCaseArm(
                        plan=IrConstructorPlan(
                            variant="Red",
                            fields=(("x", IrBindPlan(symbol=missing_sym)),),
                        ),
                        body=IrConstInt(location=_DUMMY_LOC, value=0),
                    ),
                ),
            ),
        ),
        nominals={nom: nom_desc},
    )
    with pytest.raises(InvalidIrError):
        validate_ir(prog, deep=True)


def test_validate_ircase_bind_plan_shallow_ok() -> None:
    """validate_ir with deep=False does not check IrBindPlan symbol existence."""
    missing_sym = SymbolId(999)
    prog = _make_program(
        (
            IrCase(
                location=_DUMMY_LOC,
                subject=IrConstInt(location=_DUMMY_LOC, value=1),
                arms=(
                    IrCaseArm(
                        plan=IrBindPlan(symbol=missing_sym),
                        body=IrConstInt(location=_DUMMY_LOC, value=0),
                    ),
                ),
            ),
        )
    )
    # shallow validate should not raise even though symbol is missing
    validate_ir(prog, deep=False)


def test_ir_semantic_case_constructor_nested_literal_no_match_fallback() -> None:
    """Constructor arm matched but nested literal sub-plan fails; falls to next arm."""
    # s = Circle(radius: 7); arm 0: Circle(radius: 3) — variant matches, literal fails
    # arm 1: Circle(radius: n) — catches
    src = """\
enum Shape | Circle(radius: int) | Square(side: int)
let s = Shape.Circle(radius: 7)
let r = case s of
  | Circle(radius: 3) => "three"
  | Circle(radius: n) => "other"
  | _ => "not circle"
r"""
    ir_reference, ir = evaluate_ir(src)
    assert ir["r"] == TextValue("other")
