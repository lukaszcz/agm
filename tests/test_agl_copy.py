"""Tests for the agm.agl.semantics.copying value-level copy walks.

This module tests ``deep_copy_value`` and ``shallow_copy_value`` directly at
the value level (no source program involved):
- Identity on primitives, unit, and opaque values (agent, constructor, closure).
- ``shallow_copy_value`` detaches exactly one container level and shares
  everything nested.
- ``deep_copy_value`` detaches fully, preserves diamond sharing via its
  identity memo, and terminates (producing an isomorphic, independent cycle)
  on a self-referential array or dict.
- A ``json`` leaf is duplicated via ``copy.deepcopy`` under ``deep_copy_value``.
"""

from __future__ import annotations

import decimal

from agm.agl.ir.ids import FunctionId, NominalId
from agm.agl.modules.ids import PRELUDE_ID
from agm.agl.semantics.copying import deep_copy_value, shallow_copy_value
from agm.agl.semantics.values import (
    AgentValue,
    ArrayValue,
    BoolValue,
    ConstructorValue,
    DecimalValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    IrClosureValue,
    JsonValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
)

_NOMINAL = NominalId(PRELUDE_ID, "Box")


def _record(fields: dict[str, Value]) -> RecordValue:
    return RecordValue(nominal=_NOMINAL, display_name="Box", fields=fields)


# ---------------------------------------------------------------------------
# Identity on primitives and opaque values (both copy kinds)
# ---------------------------------------------------------------------------


class TestIdentityOnPrimitivesAndOpaqueValues:
    """Primitives, unit, and opaque values are returned as-is by both copies."""

    def test_deep_copy_primitives_are_identity(self) -> None:
        for value in (
            IntValue(5),
            DecimalValue(decimal.Decimal("1.5")),
            BoolValue(True),
            TextValue("hi"),
            UnitValue(),
        ):
            assert deep_copy_value(value) is value

    def test_shallow_copy_primitives_are_identity(self) -> None:
        for value in (
            IntValue(5),
            DecimalValue(decimal.Decimal("1.5")),
            BoolValue(True),
            TextValue("hi"),
            UnitValue(),
        ):
            assert shallow_copy_value(value) is value

    def test_deep_copy_agent_value_is_identity(self) -> None:
        agent = AgentValue(name="reviewer")
        assert deep_copy_value(agent) is agent

    def test_shallow_copy_agent_value_is_identity(self) -> None:
        agent = AgentValue(name="reviewer")
        assert shallow_copy_value(agent) is agent

    def test_deep_copy_constructor_value_is_identity(self) -> None:
        ctor = ConstructorValue(nominal=_NOMINAL, display_name="Box", variant=None)
        assert deep_copy_value(ctor) is ctor

    def test_shallow_copy_constructor_value_is_identity(self) -> None:
        ctor = ConstructorValue(nominal=_NOMINAL, display_name="Box", variant=None)
        assert shallow_copy_value(ctor) is ctor

    def test_shallow_copy_json_value_is_identity(self) -> None:
        """Unlike deep_copy_value, shallow_copy_value never detaches a json leaf."""
        j = JsonValue([1, {"a": 2}])
        assert shallow_copy_value(j) is j

    def test_deep_copy_closure_value_is_identity(self) -> None:
        closure = IrClosureValue(function_id=FunctionId(value=0), captures=())
        assert deep_copy_value(closure) is closure

    def test_shallow_copy_closure_value_is_identity(self) -> None:
        closure = IrClosureValue(function_id=FunctionId(value=0), captures=())
        assert shallow_copy_value(closure) is closure


# ---------------------------------------------------------------------------
# shallow_copy_value — one level only
# ---------------------------------------------------------------------------


class TestShallowCopyOneLevel:
    """shallow_copy_value rebuilds one level and shares everything nested."""

    def test_array_top_level_detached(self) -> None:
        original = ArrayValue(elements=[IntValue(1), IntValue(2)])
        copied = shallow_copy_value(original)
        assert isinstance(copied, ArrayValue)
        assert copied is not original
        assert copied.elements is not original.elements
        assert copied.elements == original.elements

    def test_array_nested_container_shared(self) -> None:
        inner = ArrayValue(elements=[IntValue(1)])
        outer = ArrayValue(elements=[inner])
        copied = shallow_copy_value(outer)
        assert isinstance(copied, ArrayValue)
        assert copied.elements[0] is inner

    def test_dict_top_level_detached_nested_shared(self) -> None:
        inner = ArrayValue(elements=[IntValue(1)])
        original = DictValue(entries={"a": inner})
        copied = shallow_copy_value(original)
        assert isinstance(copied, DictValue)
        assert copied is not original
        assert copied.entries is not original.entries
        assert copied.entries["a"] is inner

    def test_record_fields_shared(self) -> None:
        inner = ArrayValue(elements=[IntValue(1)])
        original = _record({"items": inner})
        copied = shallow_copy_value(original)
        assert isinstance(copied, RecordValue)
        assert copied is not original
        assert copied.fields is not original.fields
        assert copied.fields["items"] is inner
        assert copied.nominal == original.nominal
        assert copied.display_name == original.display_name

    def test_enum_fields_shared(self) -> None:
        inner = ArrayValue(elements=[IntValue(1)])
        original = EnumValue(
            nominal=_NOMINAL, display_name="Choice", variant="Some", fields={"value": inner}
        )
        copied = shallow_copy_value(original)
        assert isinstance(copied, EnumValue)
        assert copied is not original
        assert copied.fields is not original.fields
        assert copied.fields["value"] is inner
        assert copied.variant == original.variant

    def test_exception_fields_shared(self) -> None:
        inner = ArrayValue(elements=[IntValue(1)])
        original = ExceptionValue(
            nominal=_NOMINAL,
            display_name="Oops",
            fields={"message": TextValue("m"), "trace_id": TextValue(""), "data": inner},
        )
        copied = shallow_copy_value(original)
        assert isinstance(copied, ExceptionValue)
        assert copied is not original
        assert copied.fields is not original.fields
        assert copied.fields["data"] is inner

    def test_never_recurses_on_a_cyclic_array(self) -> None:
        """shallow_copy on a self-referential array does not loop (one level only)."""
        cyclic = ArrayValue(elements=[])
        cyclic.elements.append(cyclic)
        copied = shallow_copy_value(cyclic)
        assert isinstance(copied, ArrayValue)
        assert copied is not cyclic
        # The single element still points at the ORIGINAL cyclic array — shallow_copy
        # never recurses, so it does not (and cannot) rewrite nested references.
        assert copied.elements[0] is cyclic


# ---------------------------------------------------------------------------
# deep_copy_value — full detachment
# ---------------------------------------------------------------------------


class TestDeepCopyFullDetachment:
    """deep_copy_value detaches every reachable array/dict/record/enum/exception."""

    def test_array_of_arrays_fully_detached(self) -> None:
        inner = ArrayValue(elements=[IntValue(1)])
        outer = ArrayValue(elements=[inner])
        copied = deep_copy_value(outer)
        assert isinstance(copied, ArrayValue)
        assert copied is not outer
        copied_inner = copied.elements[0]
        assert isinstance(copied_inner, ArrayValue)
        assert copied_inner is not inner
        assert copied_inner == inner
        # Mutating the original's nested array is not observed through the copy.
        inner.elements.append(IntValue(2))
        assert copied_inner.elements == [IntValue(1)]

    def test_dict_of_dicts_fully_detached(self) -> None:
        inner = DictValue(entries={"x": IntValue(1)})
        outer = DictValue(entries={"a": inner})
        copied = deep_copy_value(outer)
        assert isinstance(copied, DictValue)
        copied_inner = copied.entries["a"]
        assert isinstance(copied_inner, DictValue)
        assert copied_inner is not inner
        inner.entries["x"] = IntValue(99)
        assert copied_inner.entries["x"] == IntValue(1)

    def test_record_with_array_field_detached(self) -> None:
        inner = ArrayValue(elements=[IntValue(1)])
        original = _record({"items": inner})
        copied = deep_copy_value(original)
        assert isinstance(copied, RecordValue)
        copied_items = copied.fields["items"]
        assert isinstance(copied_items, ArrayValue)
        assert copied_items is not inner
        inner.elements.append(IntValue(2))
        assert copied_items.elements == [IntValue(1)]

    def test_json_leaf_is_deep_copied(self) -> None:
        original = JsonValue([1, {"a": 2}])
        copied = deep_copy_value(original)
        assert isinstance(copied, JsonValue)
        assert copied == original
        assert copied.raw is not original.raw
        assert isinstance(original.raw, list)
        assert isinstance(copied.raw, list)
        assert copied.raw[1] is not original.raw[1]


# ---------------------------------------------------------------------------
# deep_copy_value — sharing preservation (diamonds) and cycles
# ---------------------------------------------------------------------------


class TestDeepCopySharingAndCycles:
    """The identity memo preserves diamond sharing and terminates on a cycle."""

    def test_diamond_sharing_preserved(self) -> None:
        """Two sibling positions holding the SAME array copy to the SAME new array."""
        shared = ArrayValue(elements=[IntValue(1)])
        outer = ArrayValue(elements=[shared, shared])
        copied = deep_copy_value(outer)
        assert isinstance(copied, ArrayValue)
        assert copied.elements[0] is copied.elements[1]
        assert copied.elements[0] is not shared

    def test_diamond_through_record_fields_preserved(self) -> None:
        """Two records sharing one array copy to records sharing ONE new array."""
        shared = ArrayValue(elements=[IntValue(1)])
        r1 = _record({"items": shared})
        r2 = _record({"items": shared})
        outer = ArrayValue(elements=[r1, r2])
        copied = deep_copy_value(outer)
        assert isinstance(copied, ArrayValue)
        copied_r1, copied_r2 = copied.elements
        assert isinstance(copied_r1, RecordValue)
        assert isinstance(copied_r2, RecordValue)
        assert copied_r1.fields["items"] is copied_r2.fields["items"]

    def test_json_leaf_sharing_preserved(self) -> None:
        """Two slots holding the SAME json leaf copy to the SAME new json leaf."""
        shared = JsonValue([1, 2])
        outer = ArrayValue(elements=[shared, shared])
        copied = deep_copy_value(outer)
        assert isinstance(copied, ArrayValue)
        assert copied.elements[0] is copied.elements[1]
        assert copied.elements[0] is not shared

    def test_shared_record_itself_is_deduplicated(self) -> None:
        """Two array slots holding the SAME record copy to the SAME new record."""
        shared_record = _record({"n": IntValue(1)})
        outer = ArrayValue(elements=[shared_record, shared_record])
        copied = deep_copy_value(outer)
        assert isinstance(copied, ArrayValue)
        assert copied.elements[0] is copied.elements[1]
        assert copied.elements[0] is not shared_record

    def test_shared_enum_itself_is_deduplicated(self) -> None:
        """Two array slots holding the SAME enum value copy to the SAME new enum."""
        shared_enum = EnumValue(
            nominal=_NOMINAL, display_name="Choice", variant="Some", fields={"value": IntValue(1)}
        )
        outer = ArrayValue(elements=[shared_enum, shared_enum])
        copied = deep_copy_value(outer)
        assert isinstance(copied, ArrayValue)
        assert copied.elements[0] is copied.elements[1]
        assert copied.elements[0] is not shared_enum

    def test_shared_exception_itself_is_deduplicated(self) -> None:
        """Two array slots holding the SAME exception value copy to the SAME new exception."""
        shared_exc = ExceptionValue(
            nominal=_NOMINAL,
            display_name="Oops",
            fields={"message": TextValue("m"), "trace_id": TextValue("")},
        )
        outer = ArrayValue(elements=[shared_exc, shared_exc])
        copied = deep_copy_value(outer)
        assert isinstance(copied, ArrayValue)
        assert copied.elements[0] is copied.elements[1]
        assert copied.elements[0] is not shared_exc

    def test_self_referential_array_terminates_and_is_independent(self) -> None:
        """copy of a cyclic array terminates, producing an isomorphic, distinct cycle."""
        cyclic = ArrayValue(elements=[IntValue(1)])
        cyclic.elements.append(cyclic)
        copied = deep_copy_value(cyclic)
        assert isinstance(copied, ArrayValue)
        assert copied is not cyclic
        # The copy is itself cyclic, closing back onto the NEW array.
        assert copied.elements[1] is copied
        assert copied.elements[0] == IntValue(1)
        # Mutating the original's cycle does not affect the copy's cycle.
        cyclic.elements[0] = IntValue(99)
        assert copied.elements[0] == IntValue(1)

    def test_self_referential_dict_terminates_and_is_independent(self) -> None:
        cyclic = DictValue(entries={})
        cyclic.entries["self"] = cyclic
        copied = deep_copy_value(cyclic)
        assert isinstance(copied, DictValue)
        assert copied is not cyclic
        assert copied.entries["self"] is copied

    def test_record_array_cycle_terminates(self) -> None:
        """A cycle closed through BOTH a record and an array copies without looping."""
        arr = ArrayValue(elements=[])
        rec = _record({"children": arr})
        arr.elements.append(rec)
        copied_arr = deep_copy_value(arr)
        assert isinstance(copied_arr, ArrayValue)
        copied_rec = copied_arr.elements[0]
        assert isinstance(copied_rec, RecordValue)
        assert copied_rec.fields["children"] is copied_arr
        assert copied_arr is not arr
        assert copied_rec is not rec
