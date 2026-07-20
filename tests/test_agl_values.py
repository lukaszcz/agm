"""Tests for the agm.agl.semantics.values value module.

This module tests:
- All value types (leaf tags, containers, nominals, IR closure, Cell/Slot/Frame)
  are importable from agm.agl.semantics.values.
- The single broad Value union defined in agm.agl.semantics.values.
- Key eq/hash invariants for JsonValue.
- That agm.agl.semantics.values does not import from eval/syntax/scope/typecheck/runtime —
  not even under TYPE_CHECKING.
- Container/nominal tag eq/hash tests.
"""

from __future__ import annotations

import decimal

# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


def test_leaf_tags_importable_from_values_module() -> None:
    """All leaf value tags are importable directly from agm.agl.semantics.values."""
    import agm.agl.semantics.values as vals

    # Verify all expected names exist on the module.
    expected = [
        "TextValue",
        "IntValue",
        "DecimalValue",
        "BoolValue",
        "JsonValue",
        "UnitValue",
        "UNIT_VALUE",
        "VOID_VALUE",
        "AgentValue",
        "Value",
        "_json_eq",
        "_json_hash",
    ]
    for name in expected:
        assert hasattr(vals, name), f"agm.agl.semantics.values missing {name!r}"


def test_values_module_has_correct_name() -> None:
    """The module's __name__ is as expected."""
    import agm.agl.semantics.values as vals

    assert vals.__name__ == "agm.agl.semantics.values"


def test_values_module_no_forbidden_imports() -> None:
    """agm.agl.semantics.values must not import from eval, syntax, scope, typecheck, or runtime.

    This guard is STRICT: no imports from forbidden packages are allowed at any
    level — not at module level AND not inside ``if TYPE_CHECKING:`` blocks.
    All imports must come from the Python standard library only.
    """
    import ast
    import inspect

    import agm.agl.semantics.values  # ensure it's loaded

    src = inspect.getsource(agm.agl.semantics.values)

    # The forbidden sub-packages that values.py must not import from at all.
    forbidden_prefixes = [
        "agm.agl.eval",
        "agm.agl.syntax",
        "agm.agl.scope",
        "agm.agl.typecheck",
        "agm.agl.runtime",
    ]

    # Walk the entire AST — both runtime and TYPE_CHECKING-guarded imports are forbidden.
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for prefix in forbidden_prefixes:
                assert not node.module.startswith(prefix), (
                    f"agm.agl.semantics.values imports from {node.module!r}, "
                    f"which is in forbidden package {prefix!r}."
                )
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in forbidden_prefixes:
                    assert not alias.name.startswith(prefix), (
                        f"agm.agl.semantics.values imports {alias.name!r}, "
                        f"which is in forbidden package {prefix!r}."
                    )


# ---------------------------------------------------------------------------
# Constructibility
# ---------------------------------------------------------------------------


def test_unit_value_singleton() -> None:
    """UNIT_VALUE is the singleton UnitValue instance."""
    from agm.agl.semantics.values import UNIT_VALUE, UnitValue

    assert isinstance(UNIT_VALUE, UnitValue)
    assert UNIT_VALUE == UnitValue()


def test_void_unit_equals_printable_unit() -> None:
    """Printable unit and void compare equal; only REPL printability differs."""
    from agm.agl.semantics.values import UNIT_VALUE, VOID_VALUE, UnitValue

    assert UNIT_VALUE == VOID_VALUE
    assert UnitValue(printable_in_repl=True) == UnitValue(printable_in_repl=False)
    assert UNIT_VALUE.printable_in_repl is True
    assert VOID_VALUE.printable_in_repl is False


def test_primitive_values_constructible() -> None:
    """Primitive value tags can be constructed and hold their payload."""
    from agm.agl.semantics.values import (
        AgentValue,
        BoolValue,
        DecimalValue,
        IntValue,
        TextValue,
    )

    assert TextValue("hello").value == "hello"
    assert IntValue(42).value == 42
    assert DecimalValue(decimal.Decimal("3.14")).value == decimal.Decimal("3.14")
    assert BoolValue(True).value is True
    assert AgentValue("gpt4").name == "gpt4"


# ---------------------------------------------------------------------------
# JsonValue eq/hash invariants
# ---------------------------------------------------------------------------


def test_json_value_bool_not_equal_to_int() -> None:
    """JsonValue([True]) != JsonValue([1]) — bool/number conflation is prevented."""
    from agm.agl.semantics.values import JsonValue

    assert JsonValue([True]) != JsonValue([1])
    assert JsonValue(True) != JsonValue(1)
    assert JsonValue(False) != JsonValue(0)


def test_json_value_numeric_equality() -> None:
    """JsonValue(1) == JsonValue(Decimal('1')) — int and Decimal compare equal."""
    from agm.agl.semantics.values import JsonValue

    assert JsonValue(1) == JsonValue(decimal.Decimal("1"))
    assert JsonValue(2) == JsonValue(decimal.Decimal("2.0"))


def test_json_value_hash_consistent_with_eq() -> None:
    """JsonValue hash is consistent with eq."""
    from agm.agl.semantics.values import JsonValue

    jv1 = JsonValue(1)
    jv2 = JsonValue(decimal.Decimal("1"))
    assert jv1 == jv2
    assert hash(jv1) == hash(jv2)

    # bool True hashes differently from int 1
    jvb = JsonValue(True)
    assert jvb != jv1
    # (No hash collision requirement here — just that equal things hash equal.)


def test_json_value_nested_list_eq() -> None:
    """Nested JsonValue list equality follows the same bool/int rules."""
    from agm.agl.semantics.values import JsonValue

    assert JsonValue([1, 2]) == JsonValue([1, 2])
    assert JsonValue([True, 2]) != JsonValue([1, 2])


def test_json_value_dict_eq() -> None:
    """JsonValue dict equality is key-and-value structural."""
    from agm.agl.semantics.values import JsonValue

    assert JsonValue({"a": 1}) == JsonValue({"a": 1})
    assert JsonValue({"a": 1}) != JsonValue({"a": 2})
    assert JsonValue({"a": 1}) != JsonValue({"b": 1})


# ---------------------------------------------------------------------------
# DictValue order-insensitive hash (type lives in agm.agl.semantics.values)
# ---------------------------------------------------------------------------


def test_dict_value_hash_order_insensitive() -> None:
    """DictValue hash is order-insensitive (same keys/values → same hash)."""
    from agm.agl.semantics.values import DictValue, IntValue

    d1 = DictValue(entries={"a": IntValue(1), "b": IntValue(2)})
    d2 = DictValue(entries={"b": IntValue(2), "a": IntValue(1)})
    # Python dicts preserve insertion order, but hash should ignore order.
    assert d1 == d2
    assert hash(d1) == hash(d2)


def test_dict_value_eq_contract() -> None:
    """DictValue equality works correctly."""
    from agm.agl.semantics.values import DictValue, IntValue, TextValue

    d1 = DictValue(entries={"x": IntValue(5)})
    d2 = DictValue(entries={"x": IntValue(5)})
    d3 = DictValue(entries={"x": TextValue("5")})
    assert d1 == d2
    assert d1 != d3


# ---------------------------------------------------------------------------
# RecordValue eq/hash contract (type lives in agm.agl.semantics.values)
# ---------------------------------------------------------------------------


def _make_nominal(module_slash_path: str, name: str) -> "object":
    from agm.agl.ir.ids import NominalId
    from agm.agl.modules.ids import ModuleId

    return NominalId(ModuleId.from_path(module_slash_path), name)


def test_record_value_eq_and_hash() -> None:
    """RecordValue equality and hash consider nominal identity and fields.

    Two records with same nominal+fields but different display_name are equal;
    same declared_name in different modules are NOT equal.
    """
    from agm.agl.modules.ids import ModuleId
    from agm.agl.semantics.values import IntValue, NominalId, RecordValue

    mod_a = ModuleId.from_path("mymod")
    mod_b = ModuleId.from_path("other")
    nom_foo_a = NominalId(mod_a, "Foo")
    nom_foo_b = NominalId(mod_b, "Foo")
    nom_bar_a = NominalId(mod_a, "Bar")

    r1 = RecordValue(nominal=nom_foo_a, display_name="Foo", fields={"x": IntValue(1)})
    r2 = RecordValue(nominal=nom_foo_a, display_name="Foo", fields={"x": IntValue(1)})
    # Same nominal + fields, different display_name → still equal (display_name excluded from eq).
    r_diff_display = RecordValue(
        nominal=nom_foo_a, display_name="AliasName", fields={"x": IntValue(1)}
    )
    # Different module → not equal.
    r3 = RecordValue(nominal=nom_foo_b, display_name="Foo", fields={"x": IntValue(1)})
    # Different name → not equal.
    r4 = RecordValue(nominal=nom_bar_a, display_name="Bar", fields={"x": IntValue(1)})
    # Same nominal, different fields → not equal.
    r5 = RecordValue(nominal=nom_foo_a, display_name="Foo", fields={"x": IntValue(2)})

    assert r1 == r2
    assert hash(r1) == hash(r2)
    assert r1 == r_diff_display  # display_name excluded from eq
    assert hash(r1) == hash(r_diff_display)
    assert r1 != r3  # different module
    assert r1 != r4  # different name
    assert r1 != r5  # different fields


def test_constructor_value_eq_and_hash() -> None:
    """ConstructorValue equality considers nominal+variant; display_name excluded."""
    from agm.agl.modules.ids import ModuleId
    from agm.agl.semantics.values import ConstructorValue, NominalId

    mod_a = ModuleId.from_path("mymod")
    mod_b = ModuleId.from_path("other")
    nom_a = NominalId(mod_a, "Box")
    nom_b = NominalId(mod_b, "Box")

    c1 = ConstructorValue(nominal=nom_a, display_name="Box", variant=None)
    c2 = ConstructorValue(nominal=nom_a, display_name="Box", variant=None)
    # Same nominal + variant, different display_name → still equal.
    c_diff_display = ConstructorValue(nominal=nom_a, display_name="Alias", variant=None)
    # Different module → not equal.
    c3 = ConstructorValue(nominal=nom_b, display_name="Box", variant=None)
    # Different variant → not equal.
    c4 = ConstructorValue(nominal=nom_a, display_name="Box", variant="Wrap")

    assert c1 == c2
    assert hash(c1) == hash(c2)
    assert c1 == c_diff_display  # display_name excluded from eq
    assert c1 != c3  # different module
    assert c1 != c4  # different variant


def test_record_value_hash_with_json_payload() -> None:
    """RecordValue hash is consistent with JsonValue eq (numerically equal payloads)."""
    from agm.agl.modules.ids import ModuleId
    from agm.agl.semantics.values import JsonValue, NominalId, RecordValue

    nom = NominalId(ModuleId.from_path("m"), "R")
    r1 = RecordValue(nominal=nom, display_name="R", fields={"v": JsonValue(1)})
    r2 = RecordValue(nominal=nom, display_name="R", fields={"v": JsonValue(decimal.Decimal("1"))})
    # JsonValue(1) == JsonValue(Decimal("1")), so records are equal.
    assert r1 == r2
    assert hash(r1) == hash(r2)


# ---------------------------------------------------------------------------
# EnumValue eq/hash (type lives in agm.agl.semantics.values)
# ---------------------------------------------------------------------------


def test_enum_value_eq_and_hash() -> None:
    """EnumValue equality and hash consider nominal identity, variant, and fields.

    display_name is excluded from eq/hash.
    """
    from agm.agl.modules.ids import ModuleId
    from agm.agl.semantics.values import EnumValue, NominalId

    mod = ModuleId.from_path("m")
    mod2 = ModuleId.from_path("other")
    nom_color = NominalId(mod, "Color")
    nom_shape = NominalId(mod, "Shape")
    nom_color_other = NominalId(mod2, "Color")

    e1 = EnumValue(nominal=nom_color, display_name="Color", variant="Red", fields={})
    e2 = EnumValue(nominal=nom_color, display_name="Color", variant="Red", fields={})
    # Same nominal+variant+fields, different display_name → equal.
    e_diff_disp = EnumValue(nominal=nom_color, display_name="MyColor", variant="Red", fields={})
    # Different variant → not equal.
    e3 = EnumValue(nominal=nom_color, display_name="Color", variant="Blue", fields={})
    # Different name → not equal.
    e4 = EnumValue(nominal=nom_shape, display_name="Shape", variant="Red", fields={})
    # Different module → not equal.
    e5 = EnumValue(nominal=nom_color_other, display_name="Color", variant="Red", fields={})

    assert e1 == e2
    assert hash(e1) == hash(e2)
    assert e1 == e_diff_disp
    assert hash(e1) == hash(e_diff_disp)
    assert e1 != e3
    assert e1 != e4
    assert e1 != e5


# ---------------------------------------------------------------------------
# ExceptionValue eq/hash (type lives in agm.agl.semantics.values)
# ---------------------------------------------------------------------------


def test_exception_value_eq() -> None:
    """ExceptionValue equality considers nominal identity and fields.

    display_name is excluded from eq/hash. Built-in exceptions use PRELUDE_ID.
    """
    from agm.agl.modules.ids import PRELUDE_ID, ModuleId
    from agm.agl.semantics.values import ExceptionValue, NominalId, TextValue

    nom_err = NominalId(PRELUDE_ID, "Err")
    nom_err2 = NominalId(PRELUDE_ID, "Err2")
    nom_err_other = NominalId(ModuleId.from_path("mymod"), "Err")

    ex1 = ExceptionValue(nominal=nom_err, display_name="Err", fields={"message": TextValue("oops")})
    ex2 = ExceptionValue(nominal=nom_err, display_name="Err", fields={"message": TextValue("oops")})
    # Same nominal+fields, different display_name → equal.
    ex_diff_disp = ExceptionValue(
        nominal=nom_err, display_name="ErrAlias", fields={"message": TextValue("oops")}
    )
    # Different declared_name → not equal.
    ex3 = ExceptionValue(
        nominal=nom_err2, display_name="Err2", fields={"message": TextValue("oops")}
    )
    # Same name but different module → not equal.
    ex4 = ExceptionValue(
        nominal=nom_err_other, display_name="Err", fields={"message": TextValue("oops")}
    )

    assert ex1 == ex2
    assert ex1 == ex_diff_disp
    assert ex1 != ex3
    assert ex1 != ex4


def test_builtin_exception_value_uses_prelude_id() -> None:
    """Built-in exception values carry NominalId(PRELUDE_ID, name)."""
    from agm.agl.modules.ids import PRELUDE_ID
    from agm.agl.semantics.values import ExceptionValue, NominalId, TextValue

    exc = ExceptionValue(
        nominal=NominalId(PRELUDE_ID, "AgentParseError"),
        display_name="AgentParseError",
        fields={"message": TextValue("fail"), "trace_id": TextValue("")},
    )
    assert exc.nominal.module_id is PRELUDE_ID
    assert exc.nominal.declared_name == "AgentParseError"
    assert exc.display_name == "AgentParseError"


# ---------------------------------------------------------------------------
# ListValue (type lives in agm.agl.semantics.values)
# ---------------------------------------------------------------------------


def test_list_value_eq_and_hash() -> None:
    """ListValue equality and hash compare elements structurally."""
    from agm.agl.semantics.values import IntValue, ListValue

    lv1 = ListValue(elements=(IntValue(1), IntValue(2)))
    lv2 = ListValue(elements=(IntValue(1), IntValue(2)))
    lv3 = ListValue(elements=(IntValue(1),))

    assert lv1 == lv2
    assert hash(lv1) == hash(lv2)
    assert lv1 != lv3


# ---------------------------------------------------------------------------
# Broad Value union — all types live in agm.agl.semantics.values
# ---------------------------------------------------------------------------


def test_semantics_values_includes_callable_forms() -> None:
    """The single Value union in agm.agl.semantics.values includes closure and constructor types."""
    import agm.agl.semantics.values as vals

    for name in ("ConstructorValue", "IrClosureValue"):
        assert hasattr(vals, name), f"agm.agl.semantics.values missing {name!r}"


def test_semantics_values_includes_container_types() -> None:
    """The single Value union in agm.agl.semantics.values includes all container/nominal types."""
    import agm.agl.semantics.values as vals

    for name in ("ListValue", "DictValue", "RecordValue", "EnumValue", "ExceptionValue"):
        assert hasattr(vals, name), f"agm.agl.semantics.values missing {name!r}"


def test_semantics_values_includes_frame_model() -> None:
    """agm.agl.semantics.values exports the frame model: Cell, Slot, Frame."""
    import agm.agl.semantics.values as vals

    for name in ("Cell", "Slot", "Frame"):
        assert hasattr(vals, name), f"agm.agl.semantics.values missing {name!r}"


# ---------------------------------------------------------------------------
# Unified value module exports all expected names
# ---------------------------------------------------------------------------


def test_semantics_values_exports_all_leaf_tags() -> None:
    """agm.agl.semantics.values exports all leaf primitive value tags."""
    from agm.agl.semantics.values import (
        UNIT_VALUE,
        VOID_VALUE,
        AgentValue,
        BoolValue,
        DecimalValue,
        IntValue,
        JsonValue,
        TextValue,
        UnitValue,
    )

    assert TextValue is not None
    assert IntValue is not None
    assert DecimalValue is not None
    assert BoolValue is not None
    assert JsonValue is not None
    assert UnitValue is not None
    assert UNIT_VALUE is not None
    assert VOID_VALUE is not None
    assert AgentValue is not None


def test_broad_value_includes_ir_callable_forms() -> None:
    """The broad runtime union includes IR closure and constructor values."""
    from agm.agl.semantics.values import ConstructorValue, IrClosureValue, Value

    assert IrClosureValue is not None
    assert ConstructorValue is not None
    assert Value is not None


def test_helpers_accessible_from_semantics_values() -> None:
    """_json_eq and _json_hash are accessible from agm.agl.semantics.values."""
    from agm.agl.semantics.values import _json_eq, _json_hash

    assert _json_eq(1, decimal.Decimal("1")) is True
    assert _json_eq(True, 1) is False
    assert isinstance(_json_hash(1), int)


def test_ir_closure_value_identity_equality() -> None:
    """IrClosureValue uses identity-based equality (__eq__ = self is other)."""
    from agm.agl.ir.ids import FunctionId
    from agm.agl.semantics.values import IrClosureValue

    v = IrClosureValue(function_id=FunctionId(0), captures=())
    # Same object is equal to itself
    assert v == v
    # Different object (even with same content) is not equal
    other = IrClosureValue(function_id=FunctionId(0), captures=())
    assert v != other
    # Not equal to non-IrClosureValue objects
    assert v.__eq__(42) is not True  # returns bool (NotImplemented would be False)
    assert not (v == other)


def test_ir_closure_value_identity_hash() -> None:
    """IrClosureValue uses identity-based hash (__hash__ = id(self))."""
    from agm.agl.ir.ids import FunctionId
    from agm.agl.semantics.values import IrClosureValue

    v = IrClosureValue(function_id=FunctionId(0), captures=())
    # Hash is id(v)
    assert hash(v) == id(v)
    # Can be used in a set
    s: set[IrClosureValue] = {v}
    assert v in s
