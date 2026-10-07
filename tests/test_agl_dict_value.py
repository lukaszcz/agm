"""Tests for the runtime ``DictValue`` API: its two representations, representation
fixing, equal-but-distinct key retention, bulk operations, and how ``values_equal``
and ``semantics.copying`` use the public API only.
"""

from __future__ import annotations

import decimal
from collections.abc import Iterable

from agm.agl.ir.ids import NominalId
from agm.agl.semantics.copying import deep_copy_value, shallow_copy_value
from agm.agl.semantics.values import (
    ArrayValue,
    BoolValue,
    DecimalValue,
    DictValue,
    ExceptionValue,
    IntValue,
    JsonValue,
    RecordValue,
    TextValue,
    Value,
    values_equal,
)
from tests.agl.ir_harness import evaluate_ir

_POINT = NominalId(101)
_COLOR_RED = NominalId(102)
_COLOR_BLUE = NominalId(103)
_SOME_ERROR = NominalId(104)


def _dict_from_pairs(pairs: Iterable[tuple[Value, Value]]) -> DictValue:
    """Build a ``DictValue`` from key/value pairs via ``insert``, for test setup only."""
    result = DictValue()
    for key, value in pairs:
        result.insert(key, value)
    return result


# ---------------------------------------------------------------------------
# Representation fixing
# ---------------------------------------------------------------------------


def test_empty_dict_is_undetermined_and_reports_no_entries() -> None:
    d = DictValue()
    assert len(d) == 0
    assert list(d.items()) == []


def test_first_insert_of_a_text_key_keeps_text_representation() -> None:
    d = DictValue()
    d.insert(TextValue("a"), IntValue(1))
    assert d.lookup(TextValue("a")) == IntValue(1)


def test_first_insert_of_a_non_text_key_fixes_token_representation() -> None:
    d = DictValue()
    d.insert(IntValue(1), TextValue("one"))
    assert d.lookup(IntValue(1)) == TextValue("one")
    assert d.lookup(IntValue(2)) is None


def test_representation_stays_fixed_after_every_entry_is_removed() -> None:
    d = DictValue()
    d.insert(IntValue(1), TextValue("one"))
    d.remove(IntValue(1))
    assert len(d) == 0
    # Representation stays token-keyed: a later text key does not collide with
    # an unrelated text-keyed dict's storage.
    d.insert(IntValue(2), TextValue("two"))
    assert d.lookup(IntValue(2)) == TextValue("two")


def test_constructor_entries_argument_builds_a_text_keyed_dict() -> None:
    d = DictValue({"a": IntValue(1), "b": IntValue(2)})
    assert d.lookup(TextValue("a")) == IntValue(1)
    assert d.lookup(TextValue("b")) == IntValue(2)
    assert len(d) == 2


# ---------------------------------------------------------------------------
# insert / update_existing / remove / lookup
# ---------------------------------------------------------------------------


def test_insert_reports_whether_the_key_was_new() -> None:
    d = DictValue()
    assert d.insert(IntValue(1), TextValue("one")) is True
    assert d.insert(IntValue(1), TextValue("uno")) is False
    assert d.lookup(IntValue(1)) == TextValue("uno")


def test_update_existing_reports_presence_and_only_updates_present_keys() -> None:
    d = DictValue()
    assert d.update_existing(TextValue("a"), IntValue(1)) is False
    d.insert(TextValue("a"), IntValue(1))
    assert d.update_existing(TextValue("a"), IntValue(2)) is True
    assert d.lookup(TextValue("a")) == IntValue(2)


def test_update_existing_on_a_token_keyed_dict_reports_absence() -> None:
    d = DictValue()
    d.insert(IntValue(1), TextValue("one"))
    assert d.update_existing(IntValue(2), TextValue("two")) is False
    assert d.lookup(IntValue(2)) is None


def test_update_existing_on_a_token_keyed_dict_replaces_the_value() -> None:
    d = DictValue()
    d.insert(IntValue(1), TextValue("one"))
    assert d.update_existing(IntValue(1), TextValue("uno")) is True
    assert d.lookup(IntValue(1)) == TextValue("uno")


def test_remove_returns_the_removed_value_or_none() -> None:
    d = DictValue()
    d.insert(TextValue("a"), IntValue(1))
    assert d.remove(TextValue("a")) == IntValue(1)
    assert d.remove(TextValue("a")) is None
    assert len(d) == 0


def test_lookup_of_a_mismatched_key_kind_returns_none_rather_than_raising() -> None:
    d = DictValue({"a": IntValue(1)})
    assert d.lookup(IntValue(1)) is None


def test_remove_of_a_mismatched_key_kind_returns_none_rather_than_raising() -> None:
    d = DictValue({"a": IntValue(1)})
    assert d.remove(IntValue(1)) is None
    assert len(d) == 1


# ---------------------------------------------------------------------------
# Equal-but-distinct keys keep the original key value
# ---------------------------------------------------------------------------


def test_decimal_1_5_and_1_50_are_the_same_key_and_keep_the_original() -> None:
    d = DictValue()
    d.insert(DecimalValue(decimal.Decimal("1.5")), TextValue("first"))
    is_new = d.insert(DecimalValue(decimal.Decimal("1.50")), TextValue("second"))

    assert is_new is False
    assert len(d) == 1
    [(stored_key, stored_value)] = list(d.items())
    assert isinstance(stored_key, DecimalValue)
    assert str(stored_key.value) == "1.5"
    assert stored_value == TextValue("second")


def test_update_existing_over_an_equal_decimal_key_keeps_the_original_key() -> None:
    d = DictValue()
    original_key = DecimalValue(decimal.Decimal("2"))
    d.insert(original_key, TextValue("a"))
    assert d.update_existing(DecimalValue(decimal.Decimal("2.00")), TextValue("b")) is True
    [(stored_key, stored_value)] = list(d.items())
    assert stored_key is original_key
    assert stored_value == TextValue("b")


def test_deep_copy_preserves_decimal_key_identity() -> None:
    original_key = DecimalValue(decimal.Decimal("1.50"))
    original = _dict_from_pairs([(original_key, IntValue(1))])
    copied = deep_copy_value(original)

    assert isinstance(copied, DictValue)
    [(key, value)] = list(copied.items())
    assert key is original_key
    assert value == IntValue(1)


# ---------------------------------------------------------------------------
# bool vs int, and json bool vs number
# ---------------------------------------------------------------------------


def test_bool_true_and_int_1_are_distinct_keys() -> None:
    d = DictValue()
    d.insert(BoolValue(True), TextValue("bool"))
    d.insert(IntValue(1), TextValue("int"))

    assert len(d) == 2
    assert d.lookup(BoolValue(True)) == TextValue("bool")
    assert d.lookup(IntValue(1)) == TextValue("int")


def test_json_true_and_json_1_are_distinct_keys() -> None:
    d = DictValue()
    d.insert(JsonValue(True), TextValue("bool"))
    d.insert(JsonValue(1), TextValue("number"))

    assert len(d) == 2
    assert d.lookup(JsonValue(True)) == TextValue("bool")
    assert d.lookup(JsonValue(1)) == TextValue("number")


def test_json_1e1_and_json_10_are_the_same_key() -> None:
    d = DictValue()
    d.insert(JsonValue(decimal.Decimal("1E+1")), TextValue("first"))
    assert d.insert(JsonValue(10), TextValue("second")) is False
    assert len(d) == 1
    assert d.lookup(JsonValue(10)) == TextValue("second")


def test_json_scalar_keys_text_and_null_are_distinct() -> None:
    d = DictValue()
    d.insert(JsonValue("a"), TextValue("text"))
    d.insert(JsonValue(None), TextValue("null"))

    assert len(d) == 2
    assert d.lookup(JsonValue("a")) == TextValue("text")
    assert d.lookup(JsonValue(None)) == TextValue("null")


def test_json_container_keys_are_hashable_and_order_independent() -> None:
    d = DictValue()
    d.insert(JsonValue({"a": 1, "b": True}), TextValue("first"))
    is_new = d.insert(JsonValue({"b": True, "a": 1}), TextValue("second"))

    assert is_new is False
    assert len(d) == 1
    assert d.lookup(JsonValue([1, 2, True])) is None
    d.insert(JsonValue([1, 2, True]), TextValue("array"))
    assert d.lookup(JsonValue([1, 2, True])) == TextValue("array")


# ---------------------------------------------------------------------------
# Record, exception, and enum-member keys
# ---------------------------------------------------------------------------


def test_records_with_equal_fields_are_the_same_key() -> None:
    point_a = RecordValue(nominal=_POINT, fields={"x": IntValue(1), "y": IntValue(2)})
    point_b = RecordValue(nominal=_POINT, fields={"x": IntValue(1), "y": IntValue(2)})
    d = DictValue()
    d.insert(point_a, TextValue("first"))
    assert d.insert(point_b, TextValue("second")) is False
    assert len(d) == 1
    assert d.lookup(point_b) == TextValue("second")


def test_records_with_different_field_values_are_distinct_keys() -> None:
    point_a = RecordValue(nominal=_POINT, fields={"x": IntValue(1), "y": IntValue(2)})
    point_c = RecordValue(nominal=_POINT, fields={"x": IntValue(9), "y": IntValue(2)})
    d = DictValue()
    d.insert(point_a, TextValue("a"))
    d.insert(point_c, TextValue("c"))
    assert len(d) == 2


def test_a_record_and_an_exception_with_the_same_fields_are_distinct_keys() -> None:
    point = RecordValue(nominal=_POINT, fields={"x": IntValue(1)})
    error = ExceptionValue(nominal=_SOME_ERROR, fields={"x": IntValue(1)})
    d = DictValue()
    d.insert(point, TextValue("record"))
    d.insert(error, TextValue("exception"))
    assert len(d) == 2
    assert d.lookup(point) == TextValue("record")
    assert d.lookup(error) == TextValue("exception")


def test_two_nullary_enum_members_of_the_same_kind_are_the_same_key() -> None:
    red = RecordValue(nominal=_COLOR_RED, fields={})
    other_red = RecordValue(nominal=_COLOR_RED, fields={})
    d = DictValue()
    d.insert(red, IntValue(1))
    assert d.insert(other_red, IntValue(2)) is False
    assert d.lookup(other_red) == IntValue(2)
    assert len(d) == 1


def test_two_nullary_enum_members_of_different_kinds_are_distinct_keys() -> None:
    red = RecordValue(nominal=_COLOR_RED, fields={})
    blue = RecordValue(nominal=_COLOR_BLUE, fields={})
    d = DictValue()
    d.insert(red, IntValue(1))
    d.insert(blue, IntValue(2))
    assert len(d) == 2
    assert d.lookup(red) == IntValue(1)
    assert d.lookup(blue) == IntValue(2)


def test_records_built_via_constructor_and_with_update_are_the_same_key() -> None:
    """``fields`` iterates in declaration order regardless of construction path.

    A record built directly and a differently-valued one updated via ``with``
    to the same fields must token identically as dict keys — key tokens walk
    ``fields`` in whatever order the record was built, so this only holds if
    every construction path fills fields in declaration order.
    """
    source = """\
record Point
  x: int
  y: int
let a = Point(x = 1, y = 2)
let b = Point(x = 0, y = 2) with x = 1
"""
    bindings = evaluate_ir(source)
    a, b = bindings["a"], bindings["b"]
    assert isinstance(a, RecordValue)
    assert isinstance(b, RecordValue)
    assert a == b

    d = DictValue()
    d.insert(a, TextValue("first"))
    assert d.insert(b, TextValue("second")) is False
    assert len(d) == 1


# ---------------------------------------------------------------------------
# Iteration order, keys/values, text_items, pop_last
# ---------------------------------------------------------------------------


def test_iteration_order_is_insertion_order_with_original_key_values() -> None:
    d = DictValue()
    d.insert(IntValue(3), TextValue("three"))
    d.insert(IntValue(1), TextValue("one"))
    d.insert(IntValue(2), TextValue("two"))

    assert list(d.keys()) == [IntValue(3), IntValue(1), IntValue(2)]
    assert list(d.values()) == [TextValue("three"), TextValue("one"), TextValue("two")]
    assert list(d.items()) == [
        (IntValue(3), TextValue("three")),
        (IntValue(1), TextValue("one")),
        (IntValue(2), TextValue("two")),
    ]


def test_text_keyed_iteration_order_is_insertion_order() -> None:
    d = DictValue()
    d.insert(TextValue("z"), IntValue(1))
    d.insert(TextValue("a"), IntValue(2))

    assert list(d.keys()) == [TextValue("z"), TextValue("a")]
    assert list(d.values()) == [IntValue(1), IntValue(2)]


def test_text_items_yields_str_value_pairs_in_insertion_order() -> None:
    d = DictValue()
    d.insert(TextValue("z"), IntValue(1))
    d.insert(TextValue("a"), IntValue(2))

    assert list(d.text_items()) == [("z", IntValue(1)), ("a", IntValue(2))]


def test_text_items_on_an_undetermined_empty_dict_is_empty() -> None:
    assert list(DictValue().text_items()) == []


def test_pop_last_removes_and_returns_the_most_recently_inserted_pair() -> None:
    d = DictValue()
    d.insert(TextValue("a"), IntValue(1))
    d.insert(TextValue("b"), IntValue(2))

    assert d.pop_last() == (TextValue("b"), IntValue(2))
    assert len(d) == 1
    assert d.pop_last() == (TextValue("a"), IntValue(1))
    assert d.pop_last() is None


def test_pop_last_on_a_token_keyed_dict() -> None:
    d = DictValue()
    d.insert(IntValue(1), TextValue("one"))
    d.insert(IntValue(2), TextValue("two"))

    assert d.pop_last() == (IntValue(2), TextValue("two"))
    assert d.pop_last() == (IntValue(1), TextValue("one"))
    assert d.pop_last() is None


# ---------------------------------------------------------------------------
# values_equal
# ---------------------------------------------------------------------------


def test_values_equal_across_non_text_representations() -> None:
    a = _dict_from_pairs([(IntValue(1), TextValue("one")), (IntValue(2), TextValue("two"))])
    b = _dict_from_pairs([(IntValue(2), TextValue("two")), (IntValue(1), TextValue("one"))])
    c = _dict_from_pairs([(IntValue(1), TextValue("one"))])

    assert a == b
    assert values_equal(a, b)
    assert a != c
    assert not values_equal(a, c)


def test_values_equal_dict_compared_to_a_non_dict_is_unequal() -> None:
    d = DictValue({"a": IntValue(1)})
    assert not values_equal(d, IntValue(1))


def test_values_equal_same_length_but_different_key_sets_is_unequal() -> None:
    a = _dict_from_pairs([(IntValue(1), TextValue("one"))])
    b = _dict_from_pairs([(IntValue(2), TextValue("two"))])

    assert a != b


def test_values_equal_text_keyed_same_length_but_different_key_sets_is_unequal() -> None:
    a = DictValue({"a": IntValue(1)})
    b = DictValue({"b": IntValue(1)})

    assert a != b


def test_values_equal_recurses_into_dict_values() -> None:
    a = _dict_from_pairs([(IntValue(1), ArrayValue([IntValue(1), IntValue(2)]))])
    b = _dict_from_pairs([(IntValue(1), ArrayValue([IntValue(1), IntValue(2)]))])
    c = _dict_from_pairs([(IntValue(1), ArrayValue([IntValue(9)]))])

    assert a == b
    assert a != c


def test_values_equal_across_text_and_token_representations() -> None:
    """A text-keyed dict and a token-keyed dict holding the same text key still compare equal."""
    text_keyed = DictValue({"a": IntValue(1)})
    token_keyed = DictValue()
    token_keyed.insert(IntValue(9), IntValue(0))
    token_keyed.remove(IntValue(9))
    token_keyed.insert(TextValue("a"), IntValue(1))

    assert text_keyed == token_keyed
    assert values_equal(text_keyed, token_keyed)


def test_values_equal_across_representations_with_a_key_only_other_has() -> None:
    """Same length, mismatched representations, disjoint keys: the per-key
    fallback in ``aligned_values`` must reject rather than mis-align."""
    text_keyed = DictValue({"a": IntValue(1)})
    token_keyed = DictValue()
    token_keyed.insert(IntValue(9), IntValue(0))
    token_keyed.remove(IntValue(9))
    token_keyed.insert(TextValue("b"), IntValue(1))

    assert text_keyed != token_keyed
    assert not values_equal(text_keyed, token_keyed)


def test_two_undetermined_empty_dicts_compare_equal() -> None:
    assert DictValue() == DictValue()
    assert DictValue() == DictValue({})


def test_determined_and_undetermined_empty_dicts_compare_equal() -> None:
    determined = DictValue()
    determined.insert(IntValue(1), TextValue("one"))
    determined.remove(IntValue(1))
    assert determined == DictValue()


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_shows_key_and_value_reprs_in_insertion_order() -> None:
    d = DictValue()
    d.insert(TextValue("a"), IntValue(1))
    d.insert(TextValue("b"), IntValue(2))

    expected = (
        f"DictValue({{{TextValue('a')!r}: {IntValue(1)!r}, {TextValue('b')!r}: {IntValue(2)!r}}})"
    )
    assert repr(d) == expected


def test_repr_of_a_token_keyed_dict_shows_original_key_reprs() -> None:
    d = DictValue()
    d.insert(IntValue(1), TextValue("one"))

    assert repr(d) == f"DictValue({{{IntValue(1)!r}: {TextValue('one')!r}}})"


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_empties_a_token_keyed_dict() -> None:
    d = DictValue()
    d.insert(IntValue(1), TextValue("one"))
    d.clear()
    assert len(d) == 0
    assert d.lookup(IntValue(1)) is None


# ---------------------------------------------------------------------------
# copy / shallow_copy through semantics.copying
# ---------------------------------------------------------------------------


def test_shallow_copy_of_a_token_keyed_dict_shares_values_but_not_storage() -> None:
    inner = ArrayValue([IntValue(1)])
    original = _dict_from_pairs([(IntValue(1), inner)])

    copied = shallow_copy_value(original)

    assert isinstance(copied, DictValue)
    assert copied is not original
    assert copied.lookup(IntValue(1)) is inner
    original.insert(IntValue(2), TextValue("two"))
    assert copied.lookup(IntValue(2)) is None


def test_deep_copy_of_a_token_keyed_dict_fully_detaches_nested_containers() -> None:
    inner = ArrayValue([IntValue(1)])
    original = _dict_from_pairs([(IntValue(1), inner)])

    copied = deep_copy_value(original)

    assert isinstance(copied, DictValue)
    copied_inner = copied.lookup(IntValue(1))
    assert isinstance(copied_inner, ArrayValue)
    assert copied_inner is not inner
    inner.elements.append(IntValue(99))
    assert copied_inner.elements == [IntValue(1)]


def test_shallow_copy_of_a_text_keyed_dict_preserves_insertion_order() -> None:
    original = DictValue({"z": IntValue(1), "a": IntValue(2)})
    copied = shallow_copy_value(original)
    assert isinstance(copied, DictValue)
    assert list(copied.keys()) == [TextValue("z"), TextValue("a")]


def test_deep_copy_of_a_token_keyed_dict_preserves_insertion_order() -> None:
    original = _dict_from_pairs([(IntValue(3), IntValue(1)), (IntValue(1), IntValue(2))])
    copied = deep_copy_value(original)
    assert isinstance(copied, DictValue)
    assert list(copied.keys()) == [IntValue(3), IntValue(1)]


def test_copy_of_an_undetermined_empty_dict_stays_undetermined() -> None:
    """An empty copy source with no fixed representation copies to another
    empty, still-text-keyed shell: a later text insert lands in text storage."""
    original = DictValue()
    copied = shallow_copy_value(original)
    assert isinstance(copied, DictValue)
    assert len(copied) == 0
    copied.insert(TextValue("a"), IntValue(1))
    assert list(copied.text_items()) == [("a", IntValue(1))]


def test_copy_of_an_emptied_token_determined_dict_preserves_token_representation() -> None:
    """An emptied-but-token-determined source copies to a shell that stays
    token-keyed: a later text insert does not land in text storage."""
    original = DictValue()
    original.insert(IntValue(1), TextValue("one"))
    original.remove(IntValue(1))
    assert len(original) == 0

    copied = shallow_copy_value(original)
    assert isinstance(copied, DictValue)
    assert len(copied) == 0
    copied.insert(TextValue("a"), IntValue(1))
    assert list(copied.text_items()) == []
    assert copied.lookup(TextValue("a")) == IntValue(1)
