"""Live AgL container views exposed to extern companions."""

from __future__ import annotations

from agm.agl.runtime.boundary import AglArrayView, AglDictView
from agm.agl.semantics.values import ArrayValue, DictValue, IntValue


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
