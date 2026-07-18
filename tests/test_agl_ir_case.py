"""Source-level behavior for cases lowered through one-level decision IR."""

from __future__ import annotations

from agm.agl.semantics.values import IntValue, TextValue
from tests.agl.ir_harness import evaluate_ir

# ---------------------------------------------------------------------------
# IR evaluation tests — literal patterns
# ---------------------------------------------------------------------------


def test_case_literal_int_match() -> None:
    """int literal pattern: first matching arm is taken."""
    src = """\
let x: int = 2
let r = case x of
  | 1 => "one"
  | 2 => "two"
  | _ => "other"
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == TextValue("two")


def test_case_literal_int_fallthrough_wildcard() -> None:
    """Wildcard default arm catches when no literal matches."""
    src = """\
let x: int = 99
let r = case x of
  | 1 => "one"
  | _ => "other"
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == TextValue("other")


def test_case_literal_bool() -> None:
    """bool literal pattern."""
    src = """\
let x = true
let r = case x of
  | false => "no"
  | true => "yes"
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == TextValue("yes")


def test_case_literal_text() -> None:
    """text (string) literal pattern."""
    src = """\
let x = "hello"
let r = case x of
  | "world" => 0
  | "hello" => 1
  | _ => 2
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == IntValue(1)


def test_case_literal_null() -> None:
    """null literal pattern."""
    src = """\
let x: json = null
let r = case x of
  | null => "got null"
  | _ => "other"
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == TextValue("got null")


def test_case_redundant_later_arm_is_rejected() -> None:
    """An earlier identical arm makes the later arm a static error."""
    from agm.agl import PipelineDriver

    src = """\
let x: int = 1
let r = case x of
  | 1 => "first"
  | 1 => "second"
  | _ => "other"
r"""
    result = PipelineDriver().run(src)
    assert not result.ok
    assert any("Redundant" in diagnostic.message for diagnostic in result.diagnostics)


# ---------------------------------------------------------------------------
# IR evaluation tests — binder and wildcard patterns
# ---------------------------------------------------------------------------


def test_case_binder_pattern() -> None:
    """VarPattern binder captures value and body uses it."""
    src = """\
let x: int = 42
let r = case x of
  | n => n
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == IntValue(42)


def test_case_wildcard_pattern() -> None:
    """Wildcard pattern matches without binding."""
    src = """\
let x: int = 7
let r = case x of
  | _ => "matched"
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == TextValue("matched")


def test_case_as_patterns_bind_scalar_record_and_named_enum_field() -> None:
    """As-patterns bind their matched occurrence at every pattern depth."""
    src = """\
record Point
  x: int
  y: int
enum Box
  | value(item: int)
let point = Point(2, 3)
let scalar = 7
let boxed = value(item = 4)
let record_result = case point of | _ as whole => whole.x + whole.y
let scalar_result = case scalar of | 7 as matched => matched | _ => 0
let enum_result = case boxed of
  | value(item = 4 as item) as first as second => item + (if first == second => 1 | else => 0)
  | value(item = _) => 0
record_result + scalar_result + enum_result
"""
    ir = evaluate_ir(src)
    assert ir["record_result"] == IntValue(5)
    assert ir["scalar_result"] == IntValue(7)
    assert ir["enum_result"] == IntValue(5)


def test_case_binder_does_not_leak() -> None:
    """Case binder symbol does not appear in top-level result names."""
    src = """\
let x: int = 5
let r = case x of
  | bound_var => bound_var
r"""
    ir = evaluate_ir(src)
    # 'bound_var' must not appear as a top-level name
    assert "bound_var" not in ir, (
        f"Case binder 'bound_var' leaked into IR results: {sorted(ir.keys())}"
    )
    assert "bound_var" not in ir, (
        f"Case binder 'bound_var' leaked into ir results: {sorted(ir.keys())}"
    )
    assert ir["r"] == IntValue(5)


# ---------------------------------------------------------------------------
# IR evaluation tests — nullary bare-variant patterns
# ---------------------------------------------------------------------------


def test_case_nullary_variant_match() -> None:
    """VarPattern as bare-variant: bare name that resolves to a constructor."""
    # Using bare names (VarPattern classified as bare_variant_patterns by scope resolver)
    src = """\
enum Flag | On | Off
let f = Flag::On()
let r = case f of
  | Off => 0
  | On => 1
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == IntValue(1)


def test_case_nullary_variant_no_binding() -> None:
    """Nullary bare-variant match does not bind anything."""
    src = """\
enum Flag | On | Off
let f = Flag::Off()
let r = case f of
  | On => "on"
  | Off => "off"
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == TextValue("off")


def test_case_nullary_constructor_pattern() -> None:
    """ConstructorPattern with no fields (Red()) matches the variant."""
    src = """\
enum Color | Red | Blue
let c = Color::Red()
let r = case c of
  | Blue() => "blue"
  | Red() => "red"
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == TextValue("red")


# ---------------------------------------------------------------------------
# IR evaluation tests — constructor patterns (with fields)
# ---------------------------------------------------------------------------


def test_case_constructor_field_destructure() -> None:
    """ConstructorPattern destructures enum variant fields."""
    src = """\
enum Shape | Circle(radius: int) | Square(side: int)
let s = Shape::Circle(radius = 5)
let r = case s of
  | Circle(radius = n) => n
  | Square(side = m) => m
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == IntValue(5)


def test_case_constructor_field_no_match_fallback() -> None:
    """Constructor pattern on wrong variant falls through to next arm."""
    src = """\
enum Shape | Circle(radius: int) | Square(side: int)
let s = Shape::Square(side = 10)
let r = case s of
  | Circle(radius = n) => n
  | Square(side = m) => m
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == IntValue(10)


def test_case_constructor_nested_literal() -> None:
    """Constructor pattern with nested literal sub-pattern."""
    src = """\
enum Shape | Circle(radius: int) | Square(side: int)
let s = Shape::Circle(radius = 3)
let r = case s of
  | Circle(radius = 3) => "three"
  | Circle(radius = n) => "other"
  | _ => "not circle"
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == TextValue("three")


def test_case_constructor_nested_binder() -> None:
    """Constructor pattern with nested binder sub-pattern captures field."""
    src = """\
enum Shape | Circle(radius: int) | Square(side: int)
let s = Shape::Circle(radius = 7)
let r = case s of
  | Square(side = x) => x
  | Circle(radius = n) => n
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == IntValue(7)


def test_case_constructor_nested_wildcard() -> None:
    """Constructor pattern with nested wildcard sub-pattern."""
    src = """\
enum Shape | Circle(radius: int) | Square(side: int)
let s = Shape::Square(side = 99)
let r = case s of
  | Circle(radius = _) => "circle"
  | Square(side = _) => "square"
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == TextValue("square")


def test_case_constructor_nested_constructor() -> None:
    """Nested: constructor pattern with nested bare-variant sub-pattern."""
    src = """\
enum Color | Red | Blue
enum Shape | Colored(size: int)
let s = Shape::Colored(size = 10)
let r = case s of
  | Colored(size = n) => n
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == IntValue(10)


def test_case_constructor_multi_field() -> None:
    """Constructor pattern matching multiple fields, first field returned."""
    src = """\
enum Point | Pt(x: int, y: int)
let p = Point::Pt(x = 3, y = 4)
let r = case p of
  | Pt(x = a, y = b) => a
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == IntValue(3)


# ---------------------------------------------------------------------------
# Static match-compilation failures
# ---------------------------------------------------------------------------


def test_case_no_match_is_rejected_before_lowering() -> None:
    """An open-domain case without a catch-all is a static error."""
    from agm.agl import PipelineDriver

    src = """\
let x: int = 5
let r = case x of
  | 1 => "one"
  | 2 => "two"
r"""
    result = PipelineDriver().run(src)
    assert not result.ok
    assert any("Non-exhaustive" in diagnostic.message for diagnostic in result.diagnostics)


def test_case_no_match_enum_reports_missing_constructor() -> None:
    """An enum witness names the missing constructor."""
    from agm.agl import PipelineDriver

    src = """\
enum Color | Red | Blue
let c = Color::Red()
let r = case c of
  | Blue() => "blue"
r"""
    result = PipelineDriver().run(src)
    assert not result.ok
    assert any("Red" in diagnostic.message for diagnostic in result.diagnostics)


# ---------------------------------------------------------------------------
# IR evaluation tests — mixed arms, first-match ordering with constructors
# ---------------------------------------------------------------------------


def test_case_first_match_constructor_then_wildcard() -> None:
    """Constructor arm first, then wildcard catches all others."""
    src = """\
enum Shape | Circle(radius: int) | Square(side: int)
let s = Shape::Square(side = 3)
let r = case s of
  | Circle(radius = _) => "circle"
  | _ => "other"
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == TextValue("other")


# ---------------------------------------------------------------------------


def test_case_constructor_nested_literal_no_match_fallback() -> None:
    """Constructor arm matched but nested literal sub-plan fails; falls to next arm."""
    # s = Circle(radius = 7); arm 0: Circle(radius = 3) — variant matches, literal fails
    # arm 1: Circle(radius = n) — catches
    src = """\
enum Shape | Circle(radius: int) | Square(side: int)
let s = Shape::Circle(radius = 7)
let r = case s of
  | Circle(radius = 3) => "three"
  | Circle(radius = n) => "other"
  | _ => "not circle"
r"""
    ir = evaluate_ir(src)
    assert ir["r"] == TextValue("other")


def test_case_subject_effect_runs_exactly_once() -> None:
    source = """\
var calls = 0
def subject() -> int =
  calls := calls + 1
  2
let result = case subject() of
  | 1 => "one"
  | _ => "other"
()
"""
    values = evaluate_ir(source)
    assert values["calls"] == IntValue(1)
    assert values["result"] == TextValue("other")


def test_explicit_exhaustive_match_error_raise_remains_catchable() -> None:
    source = """\
let result = try
  case 2 of
    | 1 => 1
    | _ =>
      raise MatchError(
        message = "explicit",
        scrutinee_type = "int",
        scrutinee = null,
      )
catch MatchError =>
  7
result
"""
    assert evaluate_ir(source)["result"] == IntValue(7)
