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
    ArrayDecode,
    ArrayEncode,
    ConversionFailureMode,
    ConversionRecipe,
    ConversionStrategy,
    DictDecode,
    EncodeDefinition,
    EnumDecode,
    EnumEncode,
    ExceptionEncode,
    FieldDecode,
    FieldEncode,
    RecordDecode,
    RecordEncode,
    RefDecode,
    RefEncode,
    ScalarDecode,
    ScalarEncode,
    ScalarKind,
    TypeParameterEncode,
    VariantDecode,
    VariantEncode,
)
from agm.agl.ir.ids import NominalId
from agm.agl.ir.nodes import IrBind, IrConvert, IrNominalCast, IrSequence
from agm.agl.ir.program import NominalDescriptor, NominalKind, VariantDescriptor
from agm.agl.ir.validate import validate_ir
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.semantics.values import (
    ArrayValue,
    BoolValue,
    DecimalValue,
    IntValue,
    JsonValue,
    RecordValue,
    TextValue,
)
from tests.agl.ir_harness import evaluate_ir, evaluate_ir_raises, inline_main_items, lower_inline_ir


def _lower(source: str):
    return lower_inline_ir(source)


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


def test_array_int_ref_to_json() -> None:
    """let a: array[int] = [1, 2]; let j: json = a as json  — explicit cast, whole-array."""
    source = "let a: array[int] = [1, 2]\nlet j: json = a as json\n()\n"
    ir = evaluate_ir(source)
    assert ir["j"] == JsonValue([1, 2])


def test_dict_int_ref_to_json() -> None:
    """let d: dict[text, int] = {"k": 5}; let j: json = d as json  — explicit cast, whole-dict."""
    source = 'let d: dict[text, int] = {"k": 5}\nlet j: json = d as json\n()\n'
    ir = evaluate_ir(source)
    assert ir["j"] == JsonValue({"k": 5})


def test_record_and_enum_render_and_json() -> None:
    source = """\
record Foo
  a: int
enum Color | Red | Blue
let r-text = Foo(a = 1) as text
let r-json = Foo(a = 1) as json
let c-text = Color::Red() as text
let c-json = Color::Red() as json
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
        'let x = "[1, 2, 3]" as array[int]\n()\n',
        'let x = "[\\"a\\", \\"b\\"]" as array[text]\n()\n',
        'let x = "[1, 2]" as array[json]\n()\n',  # json leaf decode
        'let x = "{\\"k\\": [1, 2]}" as dict[text, array[int]]\n()\n',
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
let many = "[{\\"a\\": 1}, {\\"a\\": 2}]" as array[Foo]
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


_MISSING_MEMBER_SOURCES = (
    """\
record Foo
  a: int
let x = "{}" as Foo
()
""",
    """\
enum Color | Red | Blue
let x = "{\\"$case\\": \\"Purple\\"}" as Color
()
""",
)


@pytest.mark.parametrize("source", _MISSING_MEMBER_SOURCES)
def test_cast_missing_field_and_unknown_variant_raise(source: str) -> None:
    ir_exc = evaluate_ir_raises(source)
    assert ir_exc.display_name == "CastError"


# ---------------------------------------------------------------------------
# IR evaluation tests — JSON-Schema validation rejects undeclared properties
#
# Record and enum-variant schemas are closed (``additionalProperties: false``).
# The decode walk itself only reads the fields it knows about, so a payload
# carrying an undeclared property is caught by schema validation alone.
# ---------------------------------------------------------------------------


_UNDECLARED_PROPERTY_SOURCES = {
    "record": """\
record Foo
  a: int
let x = "{\\"a\\": 1, \\"extra\\": 9}" as Foo
()
""",
    "nested_record": """\
record Inner
  n: int
record Outer
  inner: Inner
let x = "{\\"inner\\": {\\"n\\": 1, \\"extra\\": 9}}" as Outer
()
""",
    "record_in_array": """\
record Foo
  a: int
let x = "[{\\"a\\": 1, \\"extra\\": 9}]" as array[Foo]
()
""",
    "enum_variant": """\
enum Shape
  | Circle(radius: int)
  | Square(side: int)
let x = "{\\"$case\\": \\"Circle\\", \\"radius\\": 1, \\"extra\\": 9}" as Shape
()
""",
}


@pytest.mark.parametrize("shape", sorted(_UNDECLARED_PROPERTY_SOURCES))
def test_cast_rejects_undeclared_property(shape: str) -> None:
    """An undeclared JSON property fails the cast rather than being silently dropped."""
    ir_exc = evaluate_ir_raises(_UNDECLARED_PROPERTY_SOURCES[shape])
    assert ir_exc.display_name == "CastError"


@pytest.mark.parametrize("shape", sorted(_UNDECLARED_PROPERTY_SOURCES))
def test_as_question_is_false_for_undeclared_property(shape: str) -> None:
    """The total form of the same cast reports failure instead of raising."""
    source = _UNDECLARED_PROPERTY_SOURCES[shape].replace(" as ", " as? ", 1)
    assert evaluate_ir(source)["x"] == BoolValue(False)


# ---------------------------------------------------------------------------
# IR evaluation tests — boolean `as?`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ("let r = 42 as? text\n()\n", True),
        ("let r = 42 as? json\n()\n", True),
        ("let r = 3 as? decimal\n()\n", True),
        ('let r = "42" as? int\n()\n', True),
        ('let r = "nope" as? int\n()\n', False),
        ("let r = 4.5 as? int\n()\n", False),
        ("let r = 4.0 as? int\n()\n", True),
    ],
)
def test_as_question_returns_bool(source: str, expected: bool) -> None:
    ir = evaluate_ir(source)
    assert ir["r"] == BoolValue(expected)


def test_total_as_question_evaluates_source() -> None:
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


def _program_bound_value(prog, name: str):
    """Return the lowered value bound to *name* in an already lowered program.

    Taking the program rather than the source lets one test read several
    bindings out of a single lowering instead of re-running the whole
    frontend once per name.
    """
    prog.modules[prog.entry_module]
    for node in inline_main_items(prog):
        if isinstance(node, IrBind):
            desc = prog.symbols.get(node.symbol)
            if desc is not None and desc.public_name == name:
                return node.value
            continue
        if not isinstance(node, IrSequence):
            continue
        root_capture, leaf = node.items
        if not isinstance(root_capture, IrBind) or not isinstance(leaf, IrSequence):
            continue
        binder = leaf.items[0]
        if isinstance(binder, IrBind):
            desc = prog.symbols.get(binder.symbol)
            if desc is not None and desc.public_name == name:
                return root_capture.value
    raise AssertionError(f"no let root capture for {name!r}")


def _bound_value(source: str, name: str):
    return _program_bound_value(_lower(source), name)


def test_golden_as_lowers_to_ir_convert_raise() -> None:
    value = _bound_value('let x = "42" as int\n()\n', "x")
    assert isinstance(value, IrConvert)
    assert value.failure_mode is ConversionFailureMode.RAISE_CAST_ERROR
    assert value.recipe.strategy is ConversionStrategy.PARSE_TEXT_THEN_DECODE
    assert value.recipe.json_schema is not None
    assert value.recipe.decode == ScalarDecode(ScalarKind.INT)


def test_golden_fallible_as_question_lowers_to_ir_convert_return_bool() -> None:
    value = _bound_value('let r = "42" as? int\n()\n', "r")
    assert isinstance(value, IrConvert)
    assert value.failure_mode is ConversionFailureMode.RETURN_BOOL


def test_nominal_downcasts_lower_to_identity_checks() -> None:
    source = """\
enum Shape | Circle(radius: int) | Square
let shape: Shape = Circle(radius = 2)
let circle = shape as Shape::Circle
let is-circle = shape as? Shape::Circle
let is-square = shape as? Shape::Square
let upcast = Circle(radius = 3) as? Shape
()
"""
    program = _lower(source)
    circle = _program_bound_value(program, "circle")
    is_circle = _program_bound_value(program, "is-circle")
    is_square = _program_bound_value(program, "is-square")
    upcast = _program_bound_value(program, "upcast")
    assert isinstance(circle, IrNominalCast) and circle.test_only is False
    assert isinstance(is_circle, IrNominalCast) and is_circle.test_only is True
    assert isinstance(is_square, IrNominalCast) and is_square.test_only is True
    assert isinstance(upcast, IrConvert)
    validate_ir(program, deep=True)
    values = evaluate_ir(source)
    assert values["circle"] == RecordValue(circle.nominal, "Shape::Circle", {"radius": IntValue(2)})
    assert values["is-circle"] == BoolValue(True)
    assert values["is-square"] == BoolValue(False)
    assert values["upcast"] == BoolValue(True)


def test_golden_total_as_question_lowers_to_ir_convert() -> None:
    value = _bound_value("let r = 3 as? decimal\n()\n", "r")
    assert isinstance(value, IrConvert)
    assert value.failure_mode is ConversionFailureMode.RETURN_BOOL
    assert value.recipe.strategy is ConversionStrategy.WIDEN_INT_TO_DECIMAL


def _strategy_of(program, name: str) -> ConversionStrategy:
    value = _program_bound_value(program, name)
    assert isinstance(value, IrConvert)
    return value.recipe.strategy


def test_golden_widen_and_render_and_tojson_strategies() -> None:
    program = _lower(
        "let widened = 3 as decimal\n"
        "let identity = 42 as int\n"
        "let rendered = 42 as text\n"
        "let encoded = 42 as json\n"
        "()\n"
    )

    assert _strategy_of(program, "widened") is ConversionStrategy.WIDEN_INT_TO_DECIMAL
    assert _strategy_of(program, "identity") is ConversionStrategy.NOOP
    assert _strategy_of(program, "rendered") is ConversionStrategy.RENDER_TO_TEXT
    assert _strategy_of(program, "encoded") is ConversionStrategy.TO_JSON


def test_golden_finite_scalar_json_cast_uses_a_static_plan() -> None:
    value = _bound_value("let x = 42 as json\n()\n", "x")
    assert isinstance(value, IrConvert)
    assert value.recipe.strategy is ConversionStrategy.TO_JSON
    assert value.recipe.encode == ScalarEncode()
    assert value.recipe.encode_definitions == ()


def test_golden_finite_recursive_json_cast_uses_a_static_plan() -> None:
    value = _bound_value(
        "enum Tree | Leaf | Node(children: array[Tree])\n"
        "let tree: Tree = Tree::Leaf\nlet x = tree as json\n()\n",
        "x",
    )
    assert isinstance(value, IrConvert)
    assert value.recipe.strategy is ConversionStrategy.TO_JSON
    assert isinstance(value.recipe.encode, RefEncode)
    assert value.recipe.encode_definitions


def test_golden_growing_recursive_json_cast_uses_a_generic_template_plan() -> None:
    value = _bound_value(
        "record Pair[A, B]\n"
        "  first: A\n"
        "  second: B\n"
        "enum Perfect[T]\n"
        "  | Single(value: T)\n"
        "  | Succ(next: Perfect[Pair[T, T]])\n"
        "let p: Perfect[int] = Single(value = 1)\n"
        "let encoded = p as json\n"
        "()\n",
        "encoded",
    )
    assert isinstance(value, IrConvert)
    assert value.recipe.strategy is ConversionStrategy.TO_JSON
    # A growing source cannot name a concrete instantiation, so its plan is a
    # reference into parameterized declaration templates rather than a walk.
    assert isinstance(value.recipe.encode, RefEncode)
    assert any(definition.parameter_count > 0 for definition in value.recipe.encode_definitions)


_BOTTOM_JSON_CAST_SOURCES = (
    'let result: json = (raise Abort(message = "stop")) as json\n()\n',
    'let result = (raise Abort(message = "stop")) as? json\n()\n',
)


@pytest.mark.parametrize("source", _BOTTOM_JSON_CAST_SOURCES)
def test_golden_bottom_json_cast_lowers_to_noop(source: str) -> None:
    value = _bound_value(source, "result")
    assert isinstance(value, IrConvert)
    assert value.recipe.strategy is ConversionStrategy.NOOP
    assert value.recipe.encode is None
    assert value.recipe.encode_definitions == ()


@pytest.mark.parametrize("source", _BOTTOM_JSON_CAST_SOURCES)
def test_bottom_json_cast_preserves_the_raised_source(source: str) -> None:
    raised = evaluate_ir_raises(source)
    assert raised.display_name == "Abort"
    assert raised.fields["message"] == TextValue("stop")


def test_golden_nested_decode_schema_shape() -> None:
    value = _bound_value('let x = "{\\"k\\": [1, 2]}" as dict[text, array[int]]\n()\n', "x")
    assert isinstance(value, IrConvert)
    assert value.recipe.decode == DictDecode(ArrayDecode(ScalarDecode(ScalarKind.INT)))


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

_RED = NominalId(1)
_FOO = NominalId(2)
_TREE = NominalId(3)


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
        (ArrayDecode(ScalarDecode(ScalarKind.INT)), 5, "Expected array, got int"),
        (DictDecode(ScalarDecode(ScalarKind.INT)), 5, "Expected object, got int"),
        (DictDecode(ScalarDecode(ScalarKind.INT)), {1: 2}, "Dict key must be string, got int"),
        (
            RecordDecode(_FOO, "Foo", (FieldDecode("a", "a", ScalarDecode(ScalarKind.INT)),)),
            5,
            "Expected object for record, got int",
        ),
        (
            RecordDecode(_FOO, "Foo", (FieldDecode("a", "a", ScalarDecode(ScalarKind.INT)),)),
            {},
            "Missing field 'a'",
        ),
        (
            EnumDecode(_RED, "Color", (VariantDecode("Red", "Red", NominalId(999), "Red", ()),)),
            5,
            "Expected object for enum, got int",
        ),
        (
            EnumDecode(_RED, "Color", (VariantDecode("Red", "Red", NominalId(999), "Red", ()),)),
            {},
            "Enum object must have a string '$case' field",
        ),
        (
            EnumDecode(_RED, "Color", (VariantDecode("Red", "Red", NominalId(999), "Red", ()),)),
            {"$case": "Purple"},
            "Unknown enum variant 'Purple' for 'Color'. Valid variants: ['Red']",
        ),
        (
            EnumDecode(
                _FOO,
                "Shape",
                (
                    VariantDecode(
                        "Circle",
                        "Circle",
                        NominalId(999),
                        "Circle",
                        (FieldDecode("r", "r", ScalarDecode(ScalarKind.INT)),),
                    ),
                ),
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
        RecordDecode(_FOO, "Foo", (FieldDecode("a", "a", ScalarDecode(ScalarKind.INT)),)), {"a": 3}
    )
    assert rec == RecordValue(nominal=_FOO, display_name="Foo", fields={"a": IntValue(3)})
    enum_val = _decode(
        EnumDecode(_RED, "Color", (VariantDecode("Red", "Red", NominalId(999), "Color::Red", ()),)),
        {"$case": "Red"},
    )
    assert enum_val == RecordValue(nominal=NominalId(999), display_name="Color::Red", fields={})
    lst = _decode(ArrayDecode(ScalarDecode(ScalarKind.INT)), [1, 2])
    assert lst == ArrayValue([IntValue(1), IntValue(2)])
    dct = _decode(DictDecode(ScalarDecode(ScalarKind.INT)), {"k": 1})
    assert dct.entries == {"k": IntValue(1)}
    variant_with_field = _decode(
        EnumDecode(
            _FOO,
            "Shape",
            (
                VariantDecode(
                    "Circle",
                    "Circle",
                    NominalId(999),
                    "Shape::Circle",
                    (FieldDecode("r", "r", ScalarDecode(ScalarKind.INT)),),
                ),
            ),
        ),
        {"$case": "Circle", "r": 5},
    )
    assert variant_with_field == RecordValue(
        nominal=NominalId(999), display_name="Shape::Circle", fields={"r": IntValue(5)}
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
    """A failing conversion surfaces via AglCastConversion for its caller."""
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


def test_validate_rejects_to_json_without_encode_plan() -> None:
    from agm.agl.ir.validate import InvalidIrError, validate_ir

    recipe = ConversionRecipe(
        strategy=ConversionStrategy.TO_JSON,
        source_label="int",
        target_label="json",
    )
    with pytest.raises(InvalidIrError, match="requires encode"):
        validate_ir(_convert_program(recipe), deep=True)


def test_validate_rejects_to_json_with_decode_or_malformed_encode_plan() -> None:
    from agm.agl.ir.validate import InvalidIrError, validate_ir

    bad_recipes = (
        ConversionRecipe(
            strategy=ConversionStrategy.TO_JSON,
            source_label="int",
            target_label="json",
            encode=ScalarEncode(),
            decode=ScalarDecode(ScalarKind.INT),
        ),
        ConversionRecipe(
            strategy=ConversionStrategy.TO_JSON,
            source_label="Tree",
            target_label="json",
            encode=RefEncode("missing"),
        ),
        ConversionRecipe(
            strategy=ConversionStrategy.TO_JSON,
            source_label="Tree",
            target_label="json",
            encode=RefEncode("loop"),
            encode_definitions=(EncodeDefinition("loop", 0, RefEncode("loop")),),
        ),
        ConversionRecipe(
            strategy=ConversionStrategy.TO_JSON,
            source_label="Tree",
            target_label="json",
            encode=RefEncode("same"),
            encode_definitions=(
                EncodeDefinition("same", 0, ScalarEncode()),
                EncodeDefinition("same", 0, ScalarEncode()),
            ),
        ),
        ConversionRecipe(
            strategy=ConversionStrategy.NOOP,
            source_label="int",
            target_label="int",
            encode=ScalarEncode(),
        ),
        # A parameter at the root, which binds none.
        ConversionRecipe(
            strategy=ConversionStrategy.TO_JSON,
            source_label="Perfect[int]",
            target_label="json",
            encode=TypeParameterEncode(0),
        ),
        # A parameter index past its own definition's arity.
        ConversionRecipe(
            strategy=ConversionStrategy.TO_JSON,
            source_label="Perfect[int]",
            target_label="json",
            encode=RefEncode("Box", (ScalarEncode(),)),
            encode_definitions=(EncodeDefinition("Box", 1, TypeParameterEncode(1)),),
        ),
        # A reference whose arguments disagree with the definition's arity.
        ConversionRecipe(
            strategy=ConversionStrategy.TO_JSON,
            source_label="Perfect[int]",
            target_label="json",
            encode=RefEncode("Box"),
            encode_definitions=(EncodeDefinition("Box", 1, TypeParameterEncode(0)),),
        ),
        ConversionRecipe(
            strategy=ConversionStrategy.TO_JSON,
            source_label="Perfect[int]",
            target_label="json",
            encode=RefEncode("Box", (ScalarEncode(),)),
            encode_definitions=(EncodeDefinition("Box", -1, ScalarEncode()),),
        ),
        # A definition no reference reaches.
        ConversionRecipe(
            strategy=ConversionStrategy.TO_JSON,
            source_label="Tree",
            target_label="json",
            encode=ScalarEncode(),
            encode_definitions=(EncodeDefinition("Orphan", 0, ScalarEncode()),),
        ),
    )
    for recipe in bad_recipes:
        with pytest.raises(InvalidIrError):
            validate_ir(_convert_program(recipe), deep=True)


def test_validate_rejects_to_json_encode_with_unregistered_nominal() -> None:
    from agm.agl.ir.validate import InvalidIrError, validate_ir

    recipe = ConversionRecipe(
        strategy=ConversionStrategy.TO_JSON,
        source_label="Ghost",
        target_label="json",
        encode=ArrayEncode(RecordEncode(NominalId(4), ())),
    )
    with pytest.raises(InvalidIrError, match="not in program.nominals"):
        validate_ir(_convert_program(recipe), deep=True)


def test_validate_rejects_malformed_encode_nominal_shapes() -> None:
    from agm.agl.ir.validate import InvalidIrError, validate_ir

    record = NominalId(4)
    exception = NominalId(5)
    enum = NominalId(6)
    program_nominals = {
        record: NominalDescriptor(
            record,
            ENTRY_ID,
            (),
            "Record",
            NominalKind.RECORD,
            ("field",),
        ),
        exception: NominalDescriptor(
            exception, ENTRY_ID, (), "Exception", NominalKind.EXCEPTION, ("field",)
        ),
        enum: NominalDescriptor(
            enum,
            ENTRY_ID,
            (),
            "Enum",
            NominalKind.ENUM,
            variants=(VariantDescriptor("Good", ("field",), record),),
        ),
    }
    bad_encodes = (
        RecordEncode(exception, (FieldEncode("field", "field", ScalarEncode()),)),
        ExceptionEncode(record, (FieldEncode("field", "field", ScalarEncode()),)),
        EnumEncode(record, ()),
        EnumEncode(enum, ()),
        EnumEncode(
            enum,
            (
                VariantEncode(
                    "Bad", "Bad", record, (FieldEncode("field", "field", ScalarEncode()),)
                ),
            ),
        ),
        RecordEncode(record, ()),
    )
    for encode in bad_encodes:
        recipe = ConversionRecipe(
            strategy=ConversionStrategy.TO_JSON,
            source_label="Bad",
            target_label="json",
            encode=encode,
        )
        program = _convert_program(recipe)
        program.nominals.update(program_nominals)
        with pytest.raises(InvalidIrError):
            validate_ir(program, deep=True)


def test_validate_accepts_recursive_to_json_encode_plan() -> None:
    from agm.agl.ir.validate import validate_ir

    tree = NominalId(4)
    recipe = ConversionRecipe(
        strategy=ConversionStrategy.TO_JSON,
        source_label="Tree",
        target_label="json",
        encode=RefEncode("Tree"),
        encode_definitions=(
            EncodeDefinition(
                "Tree",
                0,
                EnumEncode(
                    tree,
                    (
                        VariantEncode("Leaf", "Leaf", NominalId(5), ()),
                        VariantEncode(
                            "Node",
                            "Node",
                            NominalId(6),
                            (FieldEncode("child", "child", RefEncode("Tree")),),
                        ),
                    ),
                ),
            ),
        ),
    )
    program = _convert_program(recipe)
    program.nominals.update(
        {
            tree: NominalDescriptor(
                tree,
                ENTRY_ID,
                (),
                "Tree",
                NominalKind.ENUM,
                variants=(
                    VariantDescriptor("Leaf", (), NominalId(5)),
                    VariantDescriptor("Node", ("child",), NominalId(6)),
                ),
            ),
            NominalId(5): NominalDescriptor(
                NominalId(5), ENTRY_ID, ("Tree",), "Leaf", NominalKind.RECORD
            ),
            NominalId(6): NominalDescriptor(
                NominalId(6),
                ENTRY_ID,
                ("Tree",),
                "Node",
                NominalKind.RECORD,
                ("child",),
            ),
        }
    )
    validate_ir(program, deep=True)


def test_validate_accepts_a_parameterized_to_json_encode_plan() -> None:
    """A generic-template definition validates under its own arity, transitively."""
    from agm.agl.ir.validate import validate_ir

    box = NominalId(4)
    inner = NominalId(5)
    recipe = ConversionRecipe(
        strategy=ConversionStrategy.TO_JSON,
        source_label="Box[int]",
        target_label="json",
        encode=RefEncode("Box", (ScalarEncode(),)),
        encode_definitions=(
            EncodeDefinition(
                "Box",
                1,
                RecordEncode(
                    box,
                    (FieldEncode("item", "item", RefEncode("Inner", (TypeParameterEncode(0),))),),
                ),
            ),
            EncodeDefinition(
                "Inner",
                1,
                RecordEncode(inner, (FieldEncode("value", "value", TypeParameterEncode(0)),)),
            ),
        ),
    )
    program = _convert_program(recipe)
    program.nominals.update(
        {
            box: NominalDescriptor(
                box,
                ENTRY_ID,
                (),
                "Box",
                NominalKind.RECORD,
                ("item",),
            ),
            inner: NominalDescriptor(
                inner,
                ENTRY_ID,
                (),
                "Inner",
                NominalKind.RECORD,
                ("value",),
            ),
        }
    )
    validate_ir(program, deep=True)


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


def test_run_error_encodes_attached_nominal_plan_without_display_name_inspection() -> None:
    """Host-error fields use their attached static plan, not value display metadata."""
    from agm.agl.ir.contracts import EncodePlan, ExceptionFieldEncode
    from agm.agl.pipeline import exception_value_to_run_error
    from agm.agl.semantics.values import ExceptionValue

    agent = NominalId(4)
    command = NominalId(5)
    plan = EncodePlan(
        EnumEncode(
            agent,
            (
                VariantEncode(
                    "AgentCommand",
                    "AgentCommand",
                    command,
                    (FieldEncode("command", "command", ScalarEncode()),),
                ),
            ),
        )
    )
    error = exception_value_to_run_error(
        ExceptionValue(
            nominal=NominalId(3),
            display_name="AgentCallError",
            fields={
                "message": TextValue("failed"),
                "agent": RecordValue(command, "not-a-tag", {"command": TextValue("worker")}),
            },
        ),
        exception_field_encodes={
            NominalId(3): (ExceptionFieldEncode("agent", "agent-payload", plan),)
        },
    )

    assert error.fields["agent-payload"] == {"$case": "AgentCommand", "command": "worker"}
    assert "agent" not in error.fields


@pytest.mark.parametrize(
    "descriptor",
    (None, NominalDescriptor(NominalId(1), ENTRY_ID, (), "NotAnException", NominalKind.RECORD)),
)
def test_validate_rejects_exception_field_encodes_for_non_exception_nominal(
    descriptor: NominalDescriptor | None,
) -> None:
    """Exception encoding provenance must name a linked exception nominal."""
    from agm.agl.ir.validate import InvalidIrError, validate_ir

    program = _convert_program(
        ConversionRecipe(ConversionStrategy.NOOP, source_label="int", target_label="int")
    )
    nominal = NominalId(1)
    program.exception_field_encodes[nominal] = ()
    if descriptor is not None:
        program.nominals[nominal] = descriptor

    with pytest.raises(InvalidIrError):
        validate_ir(program, deep=True)


def test_validate_rejects_invalid_exception_field_encode_metadata() -> None:
    """Exception provenance names real, unique fields and well-formed plans."""
    from agm.agl.ir.contracts import EncodePlan, ExceptionFieldEncode
    from agm.agl.ir.validate import InvalidIrError, validate_ir

    nominal = NominalId(1)
    program = _convert_program(
        ConversionRecipe(ConversionStrategy.NOOP, source_label="int", target_label="int")
    )
    program.nominals[nominal] = NominalDescriptor(
        nominal, ENTRY_ID, (), "Problem", NominalKind.EXCEPTION, ("message", "choice")
    )
    bad_field_encodes = (
        (ExceptionFieldEncode("missing", "missing", EncodePlan(ScalarEncode())),),
        (
            ExceptionFieldEncode("choice", "choice", EncodePlan(ScalarEncode())),
            ExceptionFieldEncode("choice", "choice", EncodePlan(ScalarEncode())),
        ),
        (ExceptionFieldEncode("choice", "choice", EncodePlan(RefEncode("missing"))),),
        # Covers "message" but omits the descriptor's other field "choice" entirely.
        (ExceptionFieldEncode("message", "message", EncodePlan(ScalarEncode())),),
        # Distinct fields collapsed onto the same JSON name.
        (
            ExceptionFieldEncode("message", "same", EncodePlan(ScalarEncode())),
            ExceptionFieldEncode("choice", "same", EncodePlan(ScalarEncode())),
        ),
    )
    for field_encodes in bad_field_encodes:
        program.exception_field_encodes[nominal] = field_encodes
        with pytest.raises(InvalidIrError):
            validate_ir(program, deep=True)


def test_validate_rejects_decode_with_unregistered_nominal() -> None:
    """Deep validate rejects a decode schema referencing a nominal not in program.nominals."""
    from agm.agl.ir.validate import InvalidIrError, validate_ir

    recipe = ConversionRecipe(
        strategy=ConversionStrategy.DECODE_JSON,
        source_label="json",
        target_label="Ghost",
        json_schema="{}",
        decode=ArrayDecode(
            RecordDecode(NominalId(4), "Ghost", ()),
        ),
    )
    with pytest.raises(InvalidIrError, match="not in program.nominals"):
        validate_ir(_convert_program(recipe), deep=True)


@pytest.mark.parametrize(
    ("decode", "descriptor", "error"),
    (
        (
            RecordDecode(NominalId(10), "Tree", ()),
            NominalDescriptor(NominalId(10), ENTRY_ID, (), "Tree", NominalKind.ENUM),
            "non-record nominal",
        ),
        (
            RecordDecode(
                NominalId(10),
                "Record",
                (FieldDecode("wrong", "wrong", ScalarDecode(ScalarKind.INT)),),
            ),
            NominalDescriptor(
                NominalId(10),
                ENTRY_ID,
                (),
                "Record",
                NominalKind.RECORD,
                ("value",),
            ),
            "fields disagree",
        ),
        (
            RecordDecode(
                NominalId(10),
                "Wrong",
                (FieldDecode("value", "value", ScalarDecode(ScalarKind.INT)),),
            ),
            NominalDescriptor(
                NominalId(10),
                ENTRY_ID,
                (),
                "Record",
                NominalKind.RECORD,
                ("value",),
            ),
            "display name disagrees",
        ),
    ),
)
def test_validate_rejects_record_decode_that_disagrees_with_linked_record(
    decode: RecordDecode, descriptor: NominalDescriptor, error: str
) -> None:
    """A record decode must exactly describe its linked record nominal."""
    from agm.agl.ir.validate import InvalidIrError, validate_ir

    recipe = ConversionRecipe(
        strategy=ConversionStrategy.DECODE_JSON,
        source_label="json",
        target_label="Record",
        json_schema="{}",
        decode=decode,
    )
    program = _convert_program(recipe)
    program.nominals[descriptor.nominal] = descriptor

    with pytest.raises(InvalidIrError, match=error):
        validate_ir(program, deep=True)


@pytest.mark.parametrize(
    "variant",
    (
        VariantDecode("Leaf", "Leaf", NominalId(12), "Tree::Leaf", ()),
        VariantDecode("Branch", "Branch", NominalId(11), "Tree::Leaf", ()),
        VariantDecode("Leaf", "Leaf", NominalId(11), "Wrong::Leaf", ()),
        VariantDecode(
            "Leaf",
            "Leaf",
            NominalId(11),
            "Tree::Leaf",
            (FieldDecode("value", "value", ScalarDecode(ScalarKind.INT)),),
        ),
    ),
)
def test_validate_rejects_decode_variant_that_disagrees_with_linked_member(
    variant: VariantDecode,
) -> None:
    """A decode variant must exactly describe its linked enum member record."""
    from agm.agl.ir.validate import InvalidIrError, validate_ir

    enum = NominalId(10)
    member = NominalId(11)
    recipe = ConversionRecipe(
        strategy=ConversionStrategy.DECODE_JSON,
        source_label="json",
        target_label="Tree",
        json_schema="{}",
        decode=EnumDecode(enum, "Tree", (variant,)),
    )
    program = _convert_program(recipe)
    program.nominals.update(
        {
            enum: NominalDescriptor(
                enum,
                ENTRY_ID,
                (),
                "Tree",
                NominalKind.ENUM,
                variants=(VariantDescriptor("Leaf", (), member),),
            ),
            member: NominalDescriptor(member, ENTRY_ID, ("Tree",), "Leaf", NominalKind.RECORD),
        }
    )

    with pytest.raises(InvalidIrError):
        validate_ir(program, deep=True)


@pytest.mark.parametrize(
    ("enum_descriptor", "member_descriptor", "display_name"),
    (
        (
            NominalDescriptor(NominalId(10), ENTRY_ID, (), "Tree", NominalKind.RECORD),
            None,
            "Tree",
        ),
        (
            NominalDescriptor(
                NominalId(10),
                ENTRY_ID,
                (),
                "OtherTree",
                NominalKind.ENUM,
                variants=(VariantDescriptor("Leaf", (), NominalId(11)),),
            ),
            NominalDescriptor(NominalId(11), ENTRY_ID, ("Tree",), "Leaf", NominalKind.RECORD),
            "Tree",
        ),
    ),
)
def test_validate_rejects_decode_with_invalid_linked_enum_metadata(
    enum_descriptor: NominalDescriptor,
    member_descriptor: NominalDescriptor | None,
    display_name: str,
) -> None:
    """Deep validation requires the linked enum and member-record identities."""
    from agm.agl.ir.validate import InvalidIrError, validate_ir

    enum = NominalId(10)
    member = NominalId(11)
    recipe = ConversionRecipe(
        strategy=ConversionStrategy.DECODE_JSON,
        source_label="json",
        target_label="Tree",
        json_schema="{}",
        decode=EnumDecode(
            enum, display_name, (VariantDecode("Leaf", "Leaf", member, "Tree::Leaf", ()),)
        ),
    )
    program = _convert_program(recipe)
    program.nominals[enum] = enum_descriptor
    if member_descriptor is not None:
        program.nominals[member] = member_descriptor

    with pytest.raises(InvalidIrError):
        validate_ir(program, deep=True)


def _tree_decode_defs() -> tuple[tuple[str, EnumDecode], ...]:
    """A self-recursive `Tree` decode $defs table (matches the recursive e2e programs)."""
    tree_body = EnumDecode(
        nominal=_TREE,
        display_name="Tree",
        variants=(
            VariantDecode("Leaf", "Leaf", NominalId(1), "Tree::Leaf", ()),
            VariantDecode(
                "Node",
                "Node",
                NominalId(2),
                "Tree::Node",
                (
                    FieldDecode("value", "value", ScalarDecode(ScalarKind.INT)),
                    FieldDecode("left", "left", RefDecode("Tree")),
                    FieldDecode("right", "right", RefDecode("Tree")),
                ),
            ),
        ),
    )
    return (("Tree", tree_body),)


def test_validate_accepts_recursive_recipe_with_matching_defs() -> None:
    """A recursive ConversionRecipe (RefDecode root + defs) validates when the nominal exists."""
    from agm.agl.ir.ids import Location, SourceId
    from agm.agl.ir.nodes import IrConstText
    from agm.agl.ir.program import (
        ExecutableModule,
        ExecutableProgram,
        NominalDescriptor,
        NominalKind,
        SourceFile,
        VariantDescriptor,
    )
    from agm.agl.ir.validate import validate_ir

    tree_nominal = _TREE
    recipe = ConversionRecipe(
        strategy=ConversionStrategy.DECODE_JSON,
        source_label="json",
        target_label="Tree",
        json_schema="{}",
        decode=RefDecode("Tree"),
        defs=_tree_decode_defs(),
    )
    sid = SourceId(0)
    loc = Location(source_id=sid, start_offset=0, end_offset=1, start_line=1, start_col=0)
    node = IrConvert(
        location=loc,
        value=IrConstText(loc, "x"),
        recipe=recipe,
        failure_mode=ConversionFailureMode.RAISE_CAST_ERROR,
    )
    prog = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
        symbols={},
        nominals={
            tree_nominal: NominalDescriptor(
                nominal=tree_nominal,
                module_id=ENTRY_ID,
                scope_path=(),
                declared_name="Tree",
                kind=NominalKind.ENUM,
                variants=(
                    VariantDescriptor("Leaf", (), NominalId(1)),
                    VariantDescriptor("Node", ("value", "left", "right"), NominalId(2)),
                ),
            ),
            NominalId(1): NominalDescriptor(
                NominalId(1), ENTRY_ID, ("Tree",), "Leaf", NominalKind.RECORD
            ),
            NominalId(2): NominalDescriptor(
                NominalId(2),
                ENTRY_ID,
                ("Tree",),
                "Node",
                NominalKind.RECORD,
                ("value", "left", "right"),
            ),
        },
        sources={sid: SourceFile(display_name="<test>", normalized_text=" ")},
    )
    validate_ir(prog, deep=True)  # no exception


def test_validate_rejects_refdecode_with_unknown_defs_key() -> None:
    """A RefDecode whose key has no matching defs entry → InvalidIrError."""
    from agm.agl.ir.validate import InvalidIrError, validate_ir

    recipe = ConversionRecipe(
        strategy=ConversionStrategy.DECODE_JSON,
        source_label="json",
        target_label="Tree",
        json_schema="{}",
        decode=RefDecode("Tree"),
        defs=(),  # missing the "Tree" entry the root RefDecode names
    )
    with pytest.raises(InvalidIrError, match=r"unknown \$defs key"):
        validate_ir(_convert_program(recipe), deep=True)


def test_validate_rejects_total_strategy_with_defs() -> None:
    """A total-strategy recipe must not carry a non-empty defs table either."""
    from agm.agl.ir.validate import InvalidIrError, validate_ir

    recipe = ConversionRecipe(
        strategy=ConversionStrategy.RENDER_TO_TEXT,
        source_label="int",
        target_label="text",
        defs=_tree_decode_defs(),  # must not be present for a total strategy
    )
    with pytest.raises(InvalidIrError, match="must not carry"):
        validate_ir(_convert_program(recipe), deep=True)


def test_run_recipe_decode_json_resolves_recursive_defs() -> None:
    """run_recipe (the IR evaluator's cast executor) resolves RefDecode via recipe.defs."""
    recipe = ConversionRecipe(
        strategy=ConversionStrategy.DECODE_JSON,
        source_label="json",
        target_label="Tree",
        json_schema="{}",  # permissive: exercises the decode walk, not schema validation
        decode=RefDecode("Tree"),
        defs=_tree_decode_defs(),
    )
    payload = {
        "$case": "Node",
        "value": 1,
        "left": {"$case": "Leaf"},
        "right": {"$case": "Leaf"},
    }
    result = run_recipe(recipe, JsonValue(payload))
    assert isinstance(result, RecordValue)
    assert result.display_name.rsplit("::", maxsplit=1)[-1] == "Node"
    assert result.fields["value"] == IntValue(1)


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
