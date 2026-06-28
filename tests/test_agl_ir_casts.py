"""IR evaluation tests for casts (`as` / `as?`) via IrConvert and ConversionRecipe.

Exercises the IR pipeline across the full conversion matrix.  Golden lowering tests pin
the resolved recipe/strategy shapes.  Unit tests exercise the typeless decode walk directly
(its error branches are shadowed by JSON-Schema validation on the real cast path).
"""

from __future__ import annotations

import re
from decimal import Decimal

import pytest

from agm.agl.eval.conversions import AglCastConversion, run_recipe
from agm.agl.eval.conversions import decode_value as _decode
from agm.agl.ir.contracts import (
    ConversionFailureMode,
    ConversionRecipe,
    ConversionStrategy,
    DictDecode,
    EnumDecode,
    ListDecode,
    RecordDecode,
    ScalarDecode,
    ScalarKind,
    VariantDecode,
)
from agm.agl.ir.ids import NominalId
from agm.agl.ir.nodes import IrBind, IrConvert, IrSequence
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.semantics.values import (
    BoolValue,
    DecimalValue,
    EnumValue,
    IntValue,
    JsonValue,
    ListValue,
    RecordValue,
    TextValue,
)
from tests.agl.ir_harness import evaluate_ir, evaluate_ir_raises


def _lower(source: str):
    from agm.agl.lower import lower_program
    from agm.agl.parser import parse_program
    from agm.agl.scope import resolve
    from agm.agl.typecheck import check
    from tests.agl.ir_harness import m2_caps

    prog = parse_program(source)
    checked = check(resolve(prog), m2_caps())
    return lower_program(checked, source_text=source, source_label="<test>", validate=True)


# ---------------------------------------------------------------------------
# IR evaluation tests — total casts (`as`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "let x = 3 as decimal\n()\n",  # int -> decimal widen
        "let x = 42 as int\n()\n",  # identity noop
        "let x = 42 as text\n()\n",  # render scalar
        "let x = 4.5 as text\n()\n",
        "let x = true as text\n()\n",
        "let x = [1, 2, 3] as text\n()\n",
        'let x = {"k": 1} as text\n()\n',
        "let x = 42 as json\n()\n",
        'let x = "hi" as json\n()\n',  # text -> json wraps as JSON string
        "let x = [1, 2] as json\n()\n",
    ],
)
def test_total_cast_agrees(source: str) -> None:
    evaluate_ir(source)


def test_record_and_enum_render_and_json() -> None:
    source = """\
record Foo
  a: int
enum Color | Red | Blue
let r_text = Foo(a: 1) as text
let r_json = Foo(a: 1) as json
let c_text = Color.Red() as text
let c_json = Color.Red() as json
()
"""
    evaluate_ir(source)


# ---------------------------------------------------------------------------
# IR evaluation tests — fallible casts (`as`), success paths (covers every scalar decode leaf)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        'let x = "42" as int\n()\n',
        'let x = "4.5" as decimal\n()\n',
        'let x = "true" as bool\n()\n',
        'let x = "4.0" as int\n()\n',  # integral decimal narrows to int
        'let x = "[1, 2, 3]" as list[int]\n()\n',
        'let x = "[\\"a\\", \\"b\\"]" as list[text]\n()\n',
        'let x = "[1, 2]" as list[json]\n()\n',  # json leaf decode
        'let x = "{\\"k\\": [1, 2]}" as dict[text, list[int]]\n()\n',
    ],
)
def test_fallible_text_cast_success_agrees(source: str) -> None:
    evaluate_ir(source)


def test_decimal_to_int_integral_agrees() -> None:
    evaluate_ir("let x = 4.0 as int\n()\n")


def test_text_to_record_and_nested_agrees() -> None:
    source = """\
record Foo
  a: int
let one = "{\\"a\\": 1}" as Foo
let many = "[{\\"a\\": 1}, {\\"a\\": 2}]" as list[Foo]
()
"""
    evaluate_ir(source)


def test_text_to_enum_agrees() -> None:
    source = """\
enum Color | Red | Blue
let x = "{\\"$case\\": \\"Red\\"}" as Color
()
"""
    evaluate_ir(source)


def test_text_to_enum_with_fields_agrees() -> None:
    source = """\
enum Shape | Circle(radius: decimal) | Square(side: decimal)
let x = "{\\"$case\\": \\"Circle\\", \\"radius\\": 2.5}" as Shape
()
"""
    evaluate_ir(source)


def test_json_to_typed_agrees() -> None:
    source = """\
let j = 42 as json
let x = j as int
()
"""
    evaluate_ir(source)


# ---------------------------------------------------------------------------
# IR evaluation tests — fallible casts that raise CastError (`as`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "let x = 4.5 as int\n()\n",  # non-integral narrowing
        'let x = "nope" as int\n()\n',  # strict parse failure
        'let x = "true" as int\n()\n',  # schema mismatch
    ],
)
def test_cast_raises_cast_error(source: str) -> None:
    ir_exc = evaluate_ir_raises(source)
    assert ir_exc.display_name == "CastError"


def test_cast_missing_field_and_unknown_variant_raise() -> None:
    missing = """\
record Foo
  a: int
let x = "{}" as Foo
()
"""
    unknown = """\
enum Color | Red | Blue
let x = "{\\"$case\\": \\"Purple\\"}" as Color
()
"""
    for src in (missing, unknown):
        ir_exc = evaluate_ir_raises(src)
        assert ir_exc.display_name == "CastError"


# ---------------------------------------------------------------------------
# IR evaluation tests — `as?` booleans
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ("let r = 42 as? text\n()\n", True),  # total as?
        ("let r = 42 as? json\n()\n", True),
        ("let r = 3 as? decimal\n()\n", True),  # total noop/widen as?
        ('let r = "42" as? int\n()\n', True),  # fallible success
        ('let r = "nope" as? int\n()\n', False),  # fallible failure
        ("let r = 4.5 as? int\n()\n", False),
        ("let r = 4.0 as? int\n()\n", True),
    ],
)
def test_as_optional_booleans_agree(source: str, expected: bool) -> None:
    ir = evaluate_ir(source)
    assert ir["r"] == BoolValue(expected)


def test_total_as_optional_evaluates_source_then_true() -> None:
    """A total `as?` evaluates its (bound) source and yields True (IrSequence path)."""
    source = """\
let x = 5
let r = x as? text
()
"""
    ir = evaluate_ir(source)
    assert ir["r"] == BoolValue(True)
    assert ir["x"] == IntValue(5)


# ---------------------------------------------------------------------------
# Golden lowering tests
# ---------------------------------------------------------------------------


def _bound_value(source: str, name: str):
    prog = _lower(source)
    entry = prog.modules[prog.entry_module]
    for node in entry.initializers:
        if isinstance(node, IrBind):
            desc = prog.symbols.get(node.symbol)
            if desc is not None and desc.public_name == name:
                return node.value
    raise AssertionError(f"no IrBind for {name!r}")


def test_golden_as_lowers_to_ir_convert_raise() -> None:
    value = _bound_value('let x = "42" as int\n()\n', "x")
    assert isinstance(value, IrConvert)
    assert value.failure_mode is ConversionFailureMode.RAISE_CAST_ERROR
    assert value.recipe.strategy is ConversionStrategy.PARSE_TEXT_THEN_DECODE
    assert value.recipe.json_schema is not None
    assert value.recipe.decode == ScalarDecode(ScalarKind.INT)


def test_golden_fallible_as_optional_lowers_to_ir_convert_return_bool() -> None:
    value = _bound_value('let r = "42" as? int\n()\n', "r")
    assert isinstance(value, IrConvert)
    assert value.failure_mode is ConversionFailureMode.RETURN_BOOL


def test_golden_total_as_optional_lowers_to_sequence() -> None:
    value = _bound_value("let r = 42 as? text\n()\n", "r")
    assert isinstance(value, IrSequence)
    # last item is the constant True; the source is preserved as the first item.
    assert len(value.items) == 2


def _strategy_of(source: str, name: str) -> ConversionStrategy:
    value = _bound_value(source, name)
    assert isinstance(value, IrConvert)
    return value.recipe.strategy


def test_golden_widen_and_render_and_tojson_strategies() -> None:
    assert _strategy_of("let x = 3 as decimal\n()\n", "x") is (
        ConversionStrategy.WIDEN_INT_TO_DECIMAL
    )
    assert _strategy_of("let x = 42 as int\n()\n", "x") is ConversionStrategy.NOOP
    assert _strategy_of("let x = 42 as text\n()\n", "x") is ConversionStrategy.RENDER_TO_TEXT
    assert _strategy_of("let x = 42 as json\n()\n", "x") is ConversionStrategy.TO_JSON


def test_golden_nested_decode_schema_shape() -> None:
    value = _bound_value(
        'let x = "{\\"k\\": [1, 2]}" as dict[text, list[int]]\n()\n', "x"
    )
    assert isinstance(value, IrConvert)
    assert value.recipe.decode == DictDecode(ListDecode(ScalarDecode(ScalarKind.INT)))


def test_golden_decimal_to_int_strategy() -> None:
    value = _bound_value("let x = 4.0 as int\n()\n", "x")
    assert isinstance(value, IrConvert)
    assert value.recipe.strategy is ConversionStrategy.NARROW_DECIMAL_TO_INT


# ---------------------------------------------------------------------------
# Unit tests — typeless decode walk (_decode)
#
# These error branches in runtime.convert.decode_value are shadowed by
# JSON-Schema validation on the real cast path, so they are exercised directly.
# ---------------------------------------------------------------------------

_RED = NominalId(ENTRY_ID, "Color")
_FOO = NominalId(ENTRY_ID, "Foo")


def test_decode_scalar_success_branches() -> None:
    assert _decode(ScalarDecode(ScalarKind.TEXT), "a") == TextValue("a")
    assert _decode(ScalarDecode(ScalarKind.INT), 7) == IntValue(7)
    assert _decode(ScalarDecode(ScalarKind.DECIMAL), Decimal("1.5")) == DecimalValue(Decimal("1.5"))
    assert _decode(ScalarDecode(ScalarKind.DECIMAL), 5) == DecimalValue(Decimal(5))
    assert _decode(ScalarDecode(ScalarKind.BOOL), True) == BoolValue(True)
    assert _decode(ScalarDecode(ScalarKind.JSON), {"k": 1}) == JsonValue({"k": 1})


@pytest.mark.parametrize(
    "schema,obj,message",
    [
        (ScalarDecode(ScalarKind.TEXT), 5, "Expected string, got int"),
        (ScalarDecode(ScalarKind.INT), True, "Expected integer, got bool"),
        (ScalarDecode(ScalarKind.INT), "x", "Expected integer, got str 'x'"),
        (ScalarDecode(ScalarKind.DECIMAL), True, "Expected decimal, got bool"),
        (ScalarDecode(ScalarKind.DECIMAL), "x", "Expected decimal, got str 'x'"),
        (ScalarDecode(ScalarKind.BOOL), 1, "Expected bool, got int"),
        (ListDecode(ScalarDecode(ScalarKind.INT)), 5, "Expected array, got int"),
        (DictDecode(ScalarDecode(ScalarKind.INT)), 5, "Expected object, got int"),
        (DictDecode(ScalarDecode(ScalarKind.INT)), {1: 2}, "Dict key must be string, got int"),
        (
            RecordDecode(_FOO, "Foo", (("a", ScalarDecode(ScalarKind.INT)),)),
            5,
            "Expected object for record, got int",
        ),
        (
            RecordDecode(_FOO, "Foo", (("a", ScalarDecode(ScalarKind.INT)),)),
            {},
            "Missing field 'a'",
        ),
        (
            EnumDecode(_RED, "Color", (VariantDecode("Red", ()),)),
            5,
            "Expected object for enum, got int",
        ),
        (
            EnumDecode(_RED, "Color", (VariantDecode("Red", ()),)),
            {},
            "Enum object must have a string '$case' field",
        ),
        (
            EnumDecode(_RED, "Color", (VariantDecode("Red", ()),)),
            {"$case": "Purple"},
            "Unknown enum variant 'Purple' for 'Color'. Valid variants: ['Red']",
        ),
        (
            EnumDecode(
                _FOO,
                "Shape",
                (VariantDecode("Circle", (("r", ScalarDecode(ScalarKind.INT)),)),),
            ),
            {"$case": "Circle"},
            "Enum variant 'Circle' is missing field 'r'",
        ),
    ],
)
def test_decode_error_branches(schema, obj, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        _decode(schema, obj)


def test_decode_nested_record_and_enum_success() -> None:
    rec = _decode(
        RecordDecode(_FOO, "Foo", (("a", ScalarDecode(ScalarKind.INT)),)), {"a": 3}
    )
    assert rec == RecordValue(nominal=_FOO, display_name="Foo", fields={"a": IntValue(3)})
    enum_val = _decode(
        EnumDecode(_RED, "Color", (VariantDecode("Red", ()),)), {"$case": "Red"}
    )
    assert enum_val == EnumValue(nominal=_RED, display_name="Color", variant="Red", fields={})
    lst = _decode(ListDecode(ScalarDecode(ScalarKind.INT)), [1, 2])
    assert lst == ListValue((IntValue(1), IntValue(2)))
    dct = _decode(DictDecode(ScalarDecode(ScalarKind.INT)), {"k": 1})
    assert dct.entries == {"k": IntValue(1)}
    variant_with_field = _decode(
        EnumDecode(
            _FOO, "Shape", (VariantDecode("Circle", (("r", ScalarDecode(ScalarKind.INT)),)),)
        ),
        {"$case": "Circle", "r": 5},
    )
    assert variant_with_field == EnumValue(
        nominal=_FOO, display_name="Shape", variant="Circle", fields={"r": IntValue(5)}
    )


def test_run_recipe_value_conversion_failed_when_schema_permits() -> None:
    """A permissive schema that the decode walk still rejects → 'Value conversion failed'."""
    recipe = ConversionRecipe(
        strategy=ConversionStrategy.DECODE_JSON,
        source_label="json",
        target_label="int",
        json_schema="{}",  # accepts anything
        decode=ScalarDecode(ScalarKind.INT),
    )
    with pytest.raises(AglCastConversion, match="Value conversion failed"):
        run_recipe(recipe, JsonValue("not-an-int"))


def test_run_recipe_return_bool_on_failure() -> None:
    """A RETURN_BOOL conversion that fails surfaces via AglCastConversion (caught by caller)."""
    recipe = ConversionRecipe(
        strategy=ConversionStrategy.PARSE_TEXT_THEN_DECODE,
        source_label="text",
        target_label="int",
        json_schema='{"type": "integer"}',
        decode=ScalarDecode(ScalarKind.INT),
    )
    with pytest.raises(AglCastConversion):
        run_recipe(recipe, TextValue("not json"))


# ---------------------------------------------------------------------------
# Validate — recipe consistency
# ---------------------------------------------------------------------------


def _convert_program(recipe: ConversionRecipe):
    from agm.agl.ir.ids import Location, SourceId
    from agm.agl.ir.nodes import IrConstText
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile

    sid = SourceId(0)
    loc = Location(source_id=sid, start_offset=0, end_offset=1, start_line=1, start_col=0)
    node = IrConvert(
        location=loc,
        value=IrConstText(loc, "x"),
        recipe=recipe,
        failure_mode=ConversionFailureMode.RAISE_CAST_ERROR,
    )
    return ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
        symbols={},
        nominals={},
        sources={sid: SourceFile(display_name="<test>", normalized_text=" ")},
    )


def test_validate_rejects_decode_strategy_without_schema() -> None:
    from agm.agl.ir.validate import InvalidIrError, validate_ir

    recipe = ConversionRecipe(
        strategy=ConversionStrategy.DECODE_JSON,
        source_label="json",
        target_label="int",
        json_schema=None,  # missing
        decode=None,
    )
    with pytest.raises(InvalidIrError, match="requires json_schema and decode"):
        validate_ir(_convert_program(recipe), deep=True)


def test_validate_rejects_total_strategy_with_schema() -> None:
    from agm.agl.ir.validate import InvalidIrError, validate_ir

    recipe = ConversionRecipe(
        strategy=ConversionStrategy.RENDER_TO_TEXT,
        source_label="int",
        target_label="text",
        json_schema='{"type": "string"}',  # must not be present
        decode=ScalarDecode(ScalarKind.TEXT),
    )
    with pytest.raises(InvalidIrError, match="must not carry"):
        validate_ir(_convert_program(recipe), deep=True)


def test_validate_accepts_well_formed_convert() -> None:
    from agm.agl.ir.validate import validate_ir

    recipe = ConversionRecipe(
        strategy=ConversionStrategy.RENDER_TO_TEXT, source_label="int", target_label="text"
    )
    validate_ir(_convert_program(recipe), deep=True)  # no exception


def test_validate_rejects_decode_with_unregistered_nominal() -> None:
    """Deep validate rejects a decode schema referencing a nominal not in program.nominals."""
    from agm.agl.ir.validate import InvalidIrError, validate_ir

    recipe = ConversionRecipe(
        strategy=ConversionStrategy.DECODE_JSON,
        source_label="json",
        target_label="Ghost",
        json_schema="{}",
        decode=ListDecode(
            RecordDecode(NominalId(ENTRY_ID, "Ghost"), "Ghost", ()),
        ),
    )
    with pytest.raises(InvalidIrError, match="not in program.nominals"):
        validate_ir(_convert_program(recipe), deep=True)


def test_recipe_is_hashable() -> None:
    """A decode-strategy recipe (and the IrConvert holding it) must be hashable."""
    recipe = ConversionRecipe(
        strategy=ConversionStrategy.DECODE_JSON,
        source_label="json",
        target_label="int",
        json_schema='{"type": "integer"}',
        decode=ScalarDecode(ScalarKind.INT),
    )
    assert len({recipe, recipe}) == 1
