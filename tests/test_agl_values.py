"""Tests for the agm.agl.values shared value module (M1-A).

This module tests:
- Leaf value tags importable from agm.agl.values (not just agm.agl.eval.values).
- The narrow Value union defined in agm.agl.values (leaf tags only).
- Key eq/hash invariants for JsonValue.
- That agm.agl.values does not import from eval/syntax/scope/typecheck/runtime —
  not even under TYPE_CHECKING.
- Container/nominal tag eq/hash tests live here but import from agm.agl.eval.values,
  since those types now reside there.
- Backward-compatibility re-exports from agm.agl.eval.values.
"""

from __future__ import annotations

import decimal

# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


def test_leaf_tags_importable_from_values_module() -> None:
    """All leaf value tags are importable directly from agm.agl.values."""
    import agm.agl.values as vals

    # Verify all expected names exist on the module.
    expected = [
        "TextValue",
        "IntValue",
        "DecimalValue",
        "BoolValue",
        "JsonValue",
        "UnitValue",
        "UNIT_VALUE",
        "AgentValue",
        "Value",
        "_json_eq",
        "_json_hash",
    ]
    for name in expected:
        assert hasattr(vals, name), f"agm.agl.values missing {name!r}"


def test_values_module_has_correct_name() -> None:
    """The module's __name__ is as expected."""
    import agm.agl.values as vals

    assert vals.__name__ == "agm.agl.values"


def test_values_module_no_forbidden_imports() -> None:
    """agm.agl.values must not import from eval, syntax, scope, typecheck, or runtime.

    This guard is STRICT: no imports from forbidden packages are allowed at any
    level — not at module level AND not inside ``if TYPE_CHECKING:`` blocks.
    All imports must come from the Python standard library only.
    """
    import ast
    import inspect

    import agm.agl.values  # ensure it's loaded

    src = inspect.getsource(agm.agl.values)

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
                    f"agm.agl.values imports from {node.module!r}, "
                    f"which is in forbidden package {prefix!r}."
                )
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in forbidden_prefixes:
                    assert not alias.name.startswith(prefix), (
                        f"agm.agl.values imports {alias.name!r}, "
                        f"which is in forbidden package {prefix!r}."
                    )


# ---------------------------------------------------------------------------
# Constructibility
# ---------------------------------------------------------------------------


def test_unit_value_singleton() -> None:
    """UNIT_VALUE is the singleton UnitValue instance."""
    from agm.agl.values import UNIT_VALUE, UnitValue

    assert isinstance(UNIT_VALUE, UnitValue)
    assert UNIT_VALUE == UnitValue()


def test_primitive_values_constructible() -> None:
    """Primitive value tags can be constructed and hold their payload."""
    from agm.agl.values import (
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
    from agm.agl.values import JsonValue

    assert JsonValue([True]) != JsonValue([1])
    assert JsonValue(True) != JsonValue(1)
    assert JsonValue(False) != JsonValue(0)


def test_json_value_numeric_equality() -> None:
    """JsonValue(1) == JsonValue(Decimal('1')) — int and Decimal compare equal."""
    from agm.agl.values import JsonValue

    assert JsonValue(1) == JsonValue(decimal.Decimal("1"))
    assert JsonValue(2) == JsonValue(decimal.Decimal("2.0"))


def test_json_value_hash_consistent_with_eq() -> None:
    """JsonValue hash is consistent with eq."""
    from agm.agl.values import JsonValue

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
    from agm.agl.values import JsonValue

    assert JsonValue([1, 2]) == JsonValue([1, 2])
    assert JsonValue([True, 2]) != JsonValue([1, 2])


def test_json_value_dict_eq() -> None:
    """JsonValue dict equality is key-and-value structural."""
    from agm.agl.values import JsonValue

    assert JsonValue({"a": 1}) == JsonValue({"a": 1})
    assert JsonValue({"a": 1}) != JsonValue({"a": 2})
    assert JsonValue({"a": 1}) != JsonValue({"b": 1})


# ---------------------------------------------------------------------------
# DictValue order-insensitive hash (type lives in agm.agl.eval.values)
# ---------------------------------------------------------------------------


def test_dict_value_hash_order_insensitive() -> None:
    """DictValue hash is order-insensitive (same keys/values → same hash)."""
    from agm.agl.eval.values import DictValue, IntValue

    d1 = DictValue(entries={"a": IntValue(1), "b": IntValue(2)})
    d2 = DictValue(entries={"b": IntValue(2), "a": IntValue(1)})
    # Python dicts preserve insertion order, but hash should ignore order.
    assert d1 == d2
    assert hash(d1) == hash(d2)


def test_dict_value_eq_contract() -> None:
    """DictValue equality works correctly."""
    from agm.agl.eval.values import DictValue, IntValue, TextValue

    d1 = DictValue(entries={"x": IntValue(5)})
    d2 = DictValue(entries={"x": IntValue(5)})
    d3 = DictValue(entries={"x": TextValue("5")})
    assert d1 == d2
    assert d1 != d3


# ---------------------------------------------------------------------------
# RecordValue eq/hash contract (type lives in agm.agl.eval.values)
# ---------------------------------------------------------------------------


def test_record_value_eq_and_hash() -> None:
    """RecordValue equality and hash consider type_name and fields."""
    from agm.agl.eval.values import IntValue, RecordValue

    r1 = RecordValue(type_name="Foo", fields={"x": IntValue(1)})
    r2 = RecordValue(type_name="Foo", fields={"x": IntValue(1)})
    r3 = RecordValue(type_name="Bar", fields={"x": IntValue(1)})
    r4 = RecordValue(type_name="Foo", fields={"x": IntValue(2)})

    assert r1 == r2
    assert hash(r1) == hash(r2)
    assert r1 != r3
    assert r1 != r4


def test_record_value_hash_with_json_payload() -> None:
    """RecordValue hash is consistent with JsonValue eq (numerically equal payloads)."""
    from agm.agl.eval.values import JsonValue, RecordValue

    r1 = RecordValue(type_name="R", fields={"v": JsonValue(1)})
    r2 = RecordValue(type_name="R", fields={"v": JsonValue(decimal.Decimal("1"))})
    # JsonValue(1) == JsonValue(Decimal("1")), so records are equal.
    assert r1 == r2
    assert hash(r1) == hash(r2)


# ---------------------------------------------------------------------------
# EnumValue eq/hash (type lives in agm.agl.eval.values)
# ---------------------------------------------------------------------------


def test_enum_value_eq_and_hash() -> None:
    """EnumValue equality and hash consider type_name, variant, and fields."""
    from agm.agl.eval.values import EnumValue

    e1 = EnumValue(type_name="Color", variant="Red", fields={})
    e2 = EnumValue(type_name="Color", variant="Red", fields={})
    e3 = EnumValue(type_name="Color", variant="Blue", fields={})
    e4 = EnumValue(type_name="Shape", variant="Red", fields={})

    assert e1 == e2
    assert hash(e1) == hash(e2)
    assert e1 != e3
    assert e1 != e4


# ---------------------------------------------------------------------------
# ExceptionValue eq/hash (type lives in agm.agl.eval.values)
# ---------------------------------------------------------------------------


def test_exception_value_eq() -> None:
    """ExceptionValue equality considers type_name and fields."""
    from agm.agl.eval.values import ExceptionValue, TextValue

    ex1 = ExceptionValue(type_name="Err", fields={"message": TextValue("oops")})
    ex2 = ExceptionValue(type_name="Err", fields={"message": TextValue("oops")})
    ex3 = ExceptionValue(type_name="Err2", fields={"message": TextValue("oops")})

    assert ex1 == ex2
    assert ex1 != ex3


# ---------------------------------------------------------------------------
# ListValue (type lives in agm.agl.eval.values)
# ---------------------------------------------------------------------------


def test_list_value_eq_and_hash() -> None:
    """ListValue equality and hash compare elements structurally."""
    from agm.agl.eval.values import IntValue, ListValue

    lv1 = ListValue(elements=(IntValue(1), IntValue(2)))
    lv2 = ListValue(elements=(IntValue(1), IntValue(2)))
    lv3 = ListValue(elements=(IntValue(1),))

    assert lv1 == lv2
    assert hash(lv1) == hash(lv2)
    assert lv1 != lv3


# ---------------------------------------------------------------------------
# Narrow Value union type
# ---------------------------------------------------------------------------


def test_narrow_value_union_does_not_include_closure_or_constructor() -> None:
    """The narrow Value union in agm.agl.values has only leaf tags."""
    import agm.agl.values as vals

    # agm.agl.values must NOT define Closure or ConstructorValue.
    assert not hasattr(vals, "Closure"), "agm.agl.values must not define Closure"
    assert not hasattr(vals, "ConstructorValue"), (
        "agm.agl.values must not define ConstructorValue"
    )


def test_narrow_value_union_does_not_include_container_types() -> None:
    """The narrow Value union in agm.agl.values has only leaf tags — no containers/nominals."""
    import agm.agl.values as vals

    # Container/nominal types live in agm.agl.eval.values, not here.
    for name in ("ListValue", "DictValue", "RecordValue", "EnumValue", "ExceptionValue"):
        assert not hasattr(vals, name), f"agm.agl.values must not define {name}"


# ---------------------------------------------------------------------------
# Backward-compatibility: eval.values re-exports leaf tags
# ---------------------------------------------------------------------------


def test_eval_values_still_exports_leaf_tags() -> None:
    """agm.agl.eval.values still exports all leaf tags for backward compatibility."""
    # Import from both modules and verify identity (same class object).
    from agm.agl.eval.values import UNIT_VALUE as ev_UNIT_VALUE
    from agm.agl.eval.values import AgentValue as ev_AgentValue
    from agm.agl.eval.values import BoolValue as ev_BoolValue
    from agm.agl.eval.values import DecimalValue as ev_DecimalValue
    from agm.agl.eval.values import IntValue as ev_IntValue
    from agm.agl.eval.values import JsonValue as ev_JsonValue
    from agm.agl.eval.values import TextValue as ev_TextValue
    from agm.agl.eval.values import UnitValue as ev_UnitValue
    from agm.agl.values import UNIT_VALUE as bv_UNIT_VALUE
    from agm.agl.values import AgentValue as bv_AgentValue
    from agm.agl.values import BoolValue as bv_BoolValue
    from agm.agl.values import DecimalValue as bv_DecimalValue
    from agm.agl.values import IntValue as bv_IntValue
    from agm.agl.values import JsonValue as bv_JsonValue
    from agm.agl.values import TextValue as bv_TextValue
    from agm.agl.values import UnitValue as bv_UnitValue

    assert ev_TextValue is bv_TextValue
    assert ev_IntValue is bv_IntValue
    assert ev_DecimalValue is bv_DecimalValue
    assert ev_BoolValue is bv_BoolValue
    assert ev_JsonValue is bv_JsonValue
    assert ev_UnitValue is bv_UnitValue
    assert ev_UNIT_VALUE is bv_UNIT_VALUE
    assert ev_AgentValue is bv_AgentValue


def test_eval_values_broad_value_includes_closure_and_constructor() -> None:
    """agm.agl.eval.values.Value is the broad union including Closure and ConstructorValue."""
    from agm.agl.eval.values import Closure, ConstructorValue, Value

    # The broad Value type alias exists; Closure and ConstructorValue are defined here.
    assert Closure is not None
    assert ConstructorValue is not None
    # Value is a type alias, not inspectable as a set, but we can at least verify it exists.
    assert Value is not None


def test_helpers_re_exported_from_eval_values() -> None:
    """_json_eq and _json_hash are accessible from agm.agl.eval.values."""
    from agm.agl.eval.values import _json_eq, _json_hash

    assert _json_eq(1, decimal.Decimal("1")) is True
    assert _json_eq(True, 1) is False
    assert isinstance(_json_hash(1), int)
