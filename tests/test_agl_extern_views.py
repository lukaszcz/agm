"""Live AgL views exposed to extern companions."""

from __future__ import annotations

import itertools
from typing import Protocol, cast

import pytest

from agm.agl.ir.ids import NominalId
from agm.agl.ir.program import NominalDescriptor, NominalKind, VariantDescriptor
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.runtime.boundary import (
    AglArrayView,
    AglDictView,
    BoundaryTypeError,
    decode_boundary_value,
    encode_boundary_value,
    synthesize_nominal_classes,
)
from agm.agl.semantics.values import ArrayValue, DictValue, IntValue, RecordValue


class _RecordCompanion(Protocol):
    value: object
    fixed: object

    def __init__(self, **fields: object) -> None: ...


class _EventCompanion(Protocol):
    Changed: type[_RecordCompanion]


_next_nominal = itertools.count(80_000_000)


def _next_id() -> NominalId:
    return NominalId(next(_next_nominal))


def _record_class(*, mutable: bool) -> tuple[NominalId, type[_RecordCompanion]]:
    nominal = _next_id()
    descriptor = NominalDescriptor(
        nominal=nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Mutable" if mutable else "Snapshot",
        kind=NominalKind.RECORD,
        fields=("value", "fixed"),
        field_mutability=(mutable, False),
    )
    return nominal, cast(type[_RecordCompanion], synthesize_nominal_classes((descriptor,))[nominal])


def test_array_views_over_the_same_container_compare_and_hash_alike() -> None:
    value = ArrayValue([IntValue(1)])
    first = AglArrayView(value)
    second = AglArrayView(value)

    assert first == second
    assert hash(first) == hash(second)


def test_dict_views_remain_live_without_a_call_scope() -> None:
    value = DictValue({"one": IntValue(1)})
    view = AglDictView(value)
    view["two"] = 2

    assert value.entries == {"one": IntValue(1), "two": IntValue(2)}


def test_mutable_record_view_reads_current_values_and_writes_through() -> None:
    nominal, record_cls = _record_class(mutable=True)
    value = RecordValue(nominal, "Mutable", {"value": IntValue(1), "fixed": IntValue(2)})
    view = cast(_RecordCompanion, encode_boundary_value(value))

    initial = 1
    assert view.value == initial
    value.fields["value"] = IntValue(3)
    current = 3
    assert view.value == current
    assert repr(view) == "Mutable(value = 3, fixed = 2)"

    view.value = 4

    assert value.fields["value"] == IntValue(4)
    with pytest.raises(AttributeError):
        view.fixed = 5
    with pytest.raises(BoundaryTypeError):
        view.value = object()
    with pytest.raises(AttributeError):
        view.missing
    with pytest.raises(AttributeError):
        view.missing = 5
    assert isinstance(view, record_cls)


def test_mutable_record_view_round_trips_its_underlying_value_and_constructs_fresh_values() -> None:
    nominal, record_cls = _record_class(mutable=True)
    original = RecordValue(nominal, "Mutable", {"value": IntValue(1), "fixed": IntValue(2)})
    view = cast(_RecordCompanion, encode_boundary_value(original))

    assert decode_boundary_value(view) is original

    with pytest.raises(TypeError):
        record_cls(value=3)
    constructed = record_cls(value=3, fixed=4)
    decoded = decode_boundary_value(constructed)
    assert decoded == RecordValue(nominal, "Mutable", {"value": IntValue(3), "fixed": IntValue(4)})
    assert decoded is not original


def test_mutable_record_views_compare_structurally_and_are_unhashable() -> None:
    nominal, _ = _record_class(mutable=True)
    first = cast(
        _RecordCompanion,
        encode_boundary_value(
            RecordValue(nominal, "Mutable", {"value": IntValue(1), "fixed": IntValue(2)})
        ),
    )
    second = cast(
        _RecordCompanion,
        encode_boundary_value(
            RecordValue(nominal, "Mutable", {"value": IntValue(1), "fixed": IntValue(2)})
        ),
    )

    assert first == second
    assert first != object()
    with pytest.raises(TypeError):
        hash(first)


def test_immutable_record_snapshots_keep_their_equality_hashing_and_write_rejection() -> None:
    nominal, record_cls = _record_class(mutable=False)
    first = cast(
        _RecordCompanion,
        encode_boundary_value(
            RecordValue(nominal, "Snapshot", {"value": IntValue(1), "fixed": IntValue(2)})
        ),
    )
    second = record_cls(value=1, fixed=2)

    assert first == second
    assert hash(first) == hash(second)
    with pytest.raises(AttributeError):
        first.value = 3


def test_mutable_enum_member_crosses_as_a_live_record_view() -> None:
    enum_nominal = _next_id()
    member_nominal = _next_id()
    descriptors = (
        NominalDescriptor(
            nominal=enum_nominal,
            module_id=ENTRY_ID,
            scope_path=(),
            declared_name="Event",
            kind=NominalKind.ENUM,
            variants=(VariantDescriptor("Changed", ("value", "fixed"), member_nominal),),
        ),
        NominalDescriptor(
            nominal=member_nominal,
            module_id=ENTRY_ID,
            scope_path=("Event",),
            declared_name="Changed",
            kind=NominalKind.RECORD,
            fields=("value", "fixed"),
            field_mutability=(True, False),
        ),
    )
    event_cls = cast(_EventCompanion, synthesize_nominal_classes(descriptors)[enum_nominal])
    value = RecordValue(
        member_nominal, "Event::Changed", {"value": IntValue(1), "fixed": IntValue(2)}
    )
    view = cast(_RecordCompanion, encode_boundary_value(value))

    assert isinstance(view, event_cls.Changed)
    view.value = 3

    assert value.fields["value"] == IntValue(3)
    assert decode_boundary_value(view) is value
