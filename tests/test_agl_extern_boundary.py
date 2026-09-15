"""Value-directed extern boundary behavior."""

from __future__ import annotations

import itertools
from decimal import Decimal
from pathlib import Path

import pytest

from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS
from agm.agl.ir.ids import NominalId
from agm.agl.ir.program import NominalDescriptor, NominalKind, ValueDescriptors, VariantDescriptor
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.runtime.boundary import (
    AglArrayView,
    AglDictView,
    AglJson,
    BoundaryTypeError,
    BoundaryViolation,
    decode_boundary_value,
    encode_boundary_value,
    synthesize_nominal_classes,
)
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.runtime.render import render_value
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import (
    UNIT_VALUE,
    ArrayValue,
    BoolValue,
    ConstructorValue,
    DecimalValue,
    DictValue,
    ExceptionValue,
    IntValue,
    JsonValue,
    RecordValue,
    TextValue,
    Value,
)
from tests._timeouts import fail_if_slow
from tests.agl.ir_harness import (
    _prepare_extern_program,
    evaluate_ir_raises_with_externs,
    evaluate_ir_with_externs,
    nominal_id_for,
)

#: An empty descriptor view for tests that build or encode a boundary value
#: without a real compiled program behind it -- rendering shows no display
#: spellings, which none of these tests check.
_NO_DESCRIPTORS = ValueDescriptors(nominals={}, functions={})


class TestValueDirectedBoundary:
    def test_json_scalars_remain_distinct_from_native_scalars(self, tmp_path: Path) -> None:
        source = (
            "extern def json_null() -> json\n"
            "extern def json_number() -> json\n"
            "let a = json_null()\n"
            "let b = json_number()\n"
            "a\n"
        )
        companion = (
            "from agl import json\n"
            "def json_null(): return json(None)\n"
            "def json_number(): return json(3)\n"
        )
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["a"] == JsonValue(None)
        assert result["b"] == JsonValue(3)

    def test_nominals_cross_as_program_specific_classes(self, tmp_path: Path) -> None:
        source = (
            "record Box\n  value: int\n"
            "extern def bump(box: Box) -> Box\n"
            "let result = bump(Box(value = 1))\n"
            "result\n"
        )
        companion = "from agl import Box\ndef bump(box): return Box(value=box.value + 1)\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert isinstance(result["result"], RecordValue)
        assert result["result"].fields == {"value": IntValue(2)}

    def test_builtin_agent_enum_crosses_as_ordinary_enum_data(self, tmp_path: Path) -> None:
        source = (
            "extern def relay(a: Agent) -> Agent\n"
            'let result = relay(AgentCommand(command = "runner"))\n'
            "result\n"
        )
        companion = "def relay(a): return a\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        agent = result["result"]
        assert isinstance(agent, RecordValue)
        executable, _ = _prepare_extern_program(source, companion, tmp_path)
        assert agent.nominal == nominal_id_for(executable, "Agent::AgentCommand")
        assert agent.fields == {"command": TextValue("runner")}

    def test_bare_python_container_is_not_an_agl_value(self, tmp_path: Path) -> None:
        exc = evaluate_ir_raises_with_externs(
            "extern def f() -> array[int]\nlet _ = f()\n()\n",
            "def f(): return [1]\n",
            tmp_path,
        )
        assert exc.fields["python-type"].value == ""

    def test_agl_dict_rejects_non_text_keys(self, tmp_path: Path) -> None:
        exc = evaluate_ir_raises_with_externs(
            "extern def f() -> dict[text, int]\nlet _ = f()\n()\n",
            "from agl import dict as agl_dict\ndef f(): return agl_dict({1: 1})\n",
            tmp_path,
        )

        assert exc.fields["python-type"].value == "TypeError"

    def test_generic_aliases_need_no_schema_reconciliation(self, tmp_path: Path) -> None:
        source = (
            "extern def add[T, U](a: array[T], b: array[U]) -> unit\n"
            "var xs = [1]\n"
            "let _: unit = add(xs, xs)\n"
            "xs\n"
        )
        companion = "def add(a, b):\n    assert a == b\n    a.append(2)\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["xs"] == ArrayValue([IntValue(1), IntValue(2)])

    def test_companion_can_construct_nested_nominals_at_import_time(self, tmp_path: Path) -> None:
        source = (
            "record Inner\n  x: int\n"
            "record Outer\n  inner: Inner\n"
            "extern def default_outer() -> Outer\n"
            "let result = default_outer()\n"
            "result\n"
        )
        companion = (
            "from agl import Inner, Outer\n"
            "DEFAULT = Outer(inner=Inner(x=1))\n"
            "def default_outer(): return DEFAULT\n"
        )
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        outer = result["result"]
        assert isinstance(outer, RecordValue)
        inner = outer.fields["inner"]
        assert isinstance(inner, RecordValue)
        executable, _ = _prepare_extern_program(source, companion, tmp_path)
        assert inner.nominal == nominal_id_for(executable, "Inner")
        assert inner.fields == {"x": IntValue(1)}

    def test_worker_thread_can_read_and_write_nominal_elements_through_a_view(
        self, tmp_path: Path
    ) -> None:
        source = (
            "record Inner\n  x: int\n"
            "extern def read_and_write_via_thread(xs: array[Inner]) -> int\n"
            "var xs = [Inner(x = 5)]\n"
            "let result = read_and_write_via_thread(xs)\n"
            "result\n"
        )
        companion = (
            "import threading\n"
            "from agl import Inner\n"
            "def read_and_write_via_thread(xs):\n"
            "    box = {}\n"
            "    def worker():\n"
            "        box['x'] = xs[0].x\n"
            "        xs.append(Inner(x=box['x'] + 1))\n"
            "    thread = threading.Thread(target=worker)\n"
            "    thread.start()\n"
            "    thread.join()\n"
            "    return box['x']\n"
        )
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["result"] == IntValue(5)
        executable, _ = _prepare_extern_program(source, companion, tmp_path)
        inner = nominal_id_for(executable, "Inner")
        assert result["xs"] == ArrayValue(
            [
                RecordValue(inner, {"x": IntValue(5)}),
                RecordValue(inner, {"x": IntValue(6)}),
            ]
        )

    def test_nominal_elements_support_equality_membership_and_index(self, tmp_path: Path) -> None:
        source = (
            "record Box\n  value: int\n"
            "extern def checks(xs: array[Box]) -> bool\n"
            "let result = checks([Box(value = 1), Box(value = 2)])\n"
            "result\n"
        )
        companion = (
            "def checks(xs):\n"
            "    a = xs[0]\n"
            "    b = xs[0]\n"
            "    same_eq = a == b\n"
            "    same_in = a in xs\n"
            "    same_index = xs.index(a) == 0\n"
            "    dedup_ok = len({xs[0], xs[1], xs[0]}) == 2\n"
            "    return same_eq and same_in and same_index and dedup_ok\n"
        )
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["result"] == BoolValue(True)

    def test_scope_and_nominal_sharing_a_name_resolves_both(self, tmp_path: Path) -> None:
        source = (
            "record Box(value: text)\n"
            "\n"
            "scope Box\n"
            "  record Inner(label: text)\n"
            "end Box\n"
            "\n"
            "extern def make_box() -> Box\n"
            "extern def make_inner() -> Box::Inner\n"
            "let b = make_box()\n"
            "let i = make_inner()\n"
            "b\n"
        )
        companion = (
            "from agl import nominals\n"
            "def make_box(): return nominals.entry.Box(value='outer')\n"
            "def make_inner(): return nominals.entry.Box.Inner(label='scoped')\n"
        )
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["b"].fields == {"value": TextValue("outer")}
        assert result["i"].fields == {"label": TextValue("scoped")}

    def test_function_argument_round_trips_through_an_extern(self, tmp_path: Path) -> None:
        source = (
            "def f(x: int) = x\n"
            "extern def id[T](x: T) -> T\n"
            "let callback = id(f)\n"
            "let result = callback(2)\n"
            "result\n"
        )
        companion = "def id(x): x(2); return x\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["result"] == IntValue(2)

    def test_function_field_in_mutable_record_crosses_as_callable(self, tmp_path: Path) -> None:
        source = (
            "record Box\n  var callback: (int) -> int\n"
            "extern def apply(box: Box) -> int\n"
            "let result = apply(Box(callback = fn(value: int) -> int => value + 1))\n"
            "result\n"
        )
        companion = "def apply(box): return box.callback(2)\n"

        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)

        assert result["result"] == IntValue(3)

    def test_companion_constructed_and_updated_function_fields_remain_callable(
        self, tmp_path: Path
    ) -> None:
        source = (
            "record Box\n"
            "  var callback: (int) -> int\n"
            "extern def build_and_apply(callback: (int) -> int) -> int\n"
            "extern def build_update_and_apply(first: (int) -> int, second: (int) -> int) -> int\n"
            "extern def update_and_apply(box: Box, callback: (int) -> int) -> int\n"
            "let increment = fn(value: int) -> int => value + 1\n"
            "let doubled = fn(value: int) -> int => value * 2\n"
            "let built = build_and_apply(increment)\n"
            "let updated = build_update_and_apply(increment, doubled)\n"
            "let incoming-updated = update_and_apply(Box(callback = increment), doubled)\n"
        )
        companion = (
            "from agl import Box\n"
            "def build_and_apply(callback): return Box(callback=callback).callback(2)\n"
            "def build_update_and_apply(first, second):\n"
            "    box = Box(callback=first)\n"
            "    box.callback = second\n"
            "    return box.callback(2)\n"
            "def update_and_apply(box, callback):\n"
            "    box.callback = callback\n"
            "    return box.callback(2)\n"
        )

        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)

        assert result["built"] == IntValue(3)
        assert result["updated"] == IntValue(4)
        assert result["incoming-updated"] == IntValue(4)

    def test_companion_exception_message_over_a_cyclic_argument_raises_cyclic_value_error(
        self, tmp_path: Path
    ) -> None:
        source = (
            "record Node(children: array[Node])\n"
            "extern def boom(xs: array[Node]) -> unit\n"
            "let leaf = Node(children = [])\n"
            "var xs: array[Node] = [leaf]\n"
            "let n = Node(children = xs)\n"
            "xs[0] := n\n"
            "try\n"
            "  let _ = boom(xs)\n"
            '  print "unreached"\n'
            "catch CyclicValueError as e =>\n"
            '  print "caught: %{e.message}"\n'
        )
        companion = "def boom(xs): raise ValueError(xs)\n"
        _, stdout = evaluate_ir_with_externs(source, companion, tmp_path)
        assert stdout == "caught: value contains a reference cycle\n"

    def test_repr_of_a_cyclic_mutable_record_view_raises_cyclic_value_error(
        self, tmp_path: Path
    ) -> None:
        source = (
            "record Node(var children: array[Node])\n"
            "extern def show(node: Node) -> unit\n"
            "var node = Node(children = [])\n"
            "node.children := [node]\n"
            "try\n"
            "  let _ = show(node)\n"
            '  print "unreached"\n'
            "catch CyclicValueError as e =>\n"
            '  print "caught: %{e.message}"\n'
        )
        companion = "def show(node): repr(node)\n"

        _, stdout = evaluate_ir_with_externs(source, companion, tmp_path)

        assert stdout == "caught: value contains a reference cycle\n"


def test_array_view_slice_getitem_returns_encoded_elements() -> None:
    view = AglArrayView(ArrayValue([IntValue(3), IntValue(1), IntValue(2)]), _NO_DESCRIPTORS)

    assert view[1:] == [1, 2]


def test_array_view_index_getitem_returns_the_encoded_element() -> None:
    view = AglArrayView(ArrayValue([IntValue(3), IntValue(1), IntValue(2)]), _NO_DESCRIPTORS)

    assert view[0] == 3


def test_array_view_slice_setitem_replaces_a_range_of_elements() -> None:
    array_value = ArrayValue([IntValue(1), IntValue(2), IntValue(3)])
    view = AglArrayView(array_value, _NO_DESCRIPTORS)

    view[1:] = [4, 5]

    assert array_value.elements == [IntValue(1), IntValue(4), IntValue(5)]


def test_array_view_index_setitem_with_an_undecodable_value_raises() -> None:
    view = AglArrayView(ArrayValue([IntValue(1)]), _NO_DESCRIPTORS)

    with pytest.raises(BoundaryTypeError):
        view[0] = object()


def test_array_view_slice_setitem_with_a_non_iterable_value_raises_type_error() -> None:
    view = AglArrayView(ArrayValue([IntValue(1)]), _NO_DESCRIPTORS)

    with pytest.raises(TypeError):
        view[0:1] = object()


def test_array_view_slice_delitem_removes_a_range_of_elements() -> None:
    array_value = ArrayValue([IntValue(1), IntValue(2), IntValue(3)])
    view = AglArrayView(array_value, _NO_DESCRIPTORS)

    del view[1:2]

    assert array_value.elements == [IntValue(1), IntValue(3)]


def test_array_view_insert_adds_an_element_at_a_position() -> None:
    array_value = ArrayValue([IntValue(1), IntValue(3)])
    view = AglArrayView(array_value, _NO_DESCRIPTORS)

    view.insert(1, 2)

    assert array_value.elements == [IntValue(1), IntValue(2), IntValue(3)]


def test_array_view_insert_with_an_undecodable_value_raises() -> None:
    view = AglArrayView(ArrayValue([]), _NO_DESCRIPTORS)

    with pytest.raises(BoundaryTypeError):
        view.insert(0, object())


def test_array_view_append_adds_an_element_at_the_end() -> None:
    array_value = ArrayValue([IntValue(1)])
    view = AglArrayView(array_value, _NO_DESCRIPTORS)

    view.append(2)

    assert array_value.elements == [IntValue(1), IntValue(2)]


def test_array_view_extend_appends_the_given_values() -> None:
    array_value = ArrayValue([IntValue(1)])
    view = AglArrayView(array_value, _NO_DESCRIPTORS)

    view.extend([2, 3])

    assert array_value.elements == [IntValue(1), IntValue(2), IntValue(3)]


def test_array_view_reverse_reverses_the_element_order() -> None:
    array_value = ArrayValue([IntValue(1), IntValue(2), IntValue(3)])
    view = AglArrayView(array_value, _NO_DESCRIPTORS)

    view.reverse()

    assert list(view) == [3, 2, 1]


def test_array_view_sort_orders_elements_ascending_by_default() -> None:
    array_value = ArrayValue([IntValue(3), IntValue(1), IntValue(2)])
    view = AglArrayView(array_value, _NO_DESCRIPTORS)

    view.sort()

    assert list(view) == [1, 2, 3]


def test_array_view_iteration_yields_encoded_elements() -> None:
    view = AglArrayView(ArrayValue([IntValue(1), IntValue(2)]), _NO_DESCRIPTORS)

    assert list(view) == [1, 2]


def test_array_view_contains_returns_true_for_a_present_value() -> None:
    view = AglArrayView(ArrayValue([IntValue(1), IntValue(2)]), _NO_DESCRIPTORS)

    assert 2 in view


def test_array_view_index_uses_agl_equality_for_nested_arrays_and_json() -> None:
    nested = AglArrayView(ArrayValue([ArrayValue([IntValue(1)])]), _NO_DESCRIPTORS)
    json_values = AglArrayView(ArrayValue([JsonValue([True]), JsonValue([1])]), _NO_DESCRIPTORS)

    assert nested.index(AglArrayView(ArrayValue([IntValue(1)]), _NO_DESCRIPTORS)) == 0
    assert json_values.index(AglJson([1])) == 1
    assert json_values.index(AglJson([1]), -1) == 1
    assert json_values.index(AglJson([True]), 0, -1) == 0


def test_array_view_clear_empties_the_array() -> None:
    view = AglArrayView(ArrayValue([IntValue(1)]), _NO_DESCRIPTORS)

    view.clear()

    assert not view


def test_array_view_is_not_equal_to_a_non_view_object() -> None:
    view = AglArrayView(ArrayValue([]), _NO_DESCRIPTORS)

    assert view != object()


def test_array_views_over_the_same_container_compare_equal_and_hash_alike() -> None:
    array_value = ArrayValue([IntValue(1)])

    assert AglArrayView(array_value, _NO_DESCRIPTORS) == AglArrayView(array_value, _NO_DESCRIPTORS)
    assert hash(AglArrayView(array_value, _NO_DESCRIPTORS)) == hash(
        AglArrayView(array_value, _NO_DESCRIPTORS)
    )


def test_array_view_repr_matches_the_rendered_agl_value() -> None:
    array_value = ArrayValue([IntValue(1), IntValue(2)])
    view = AglArrayView(array_value, _NO_DESCRIPTORS)

    assert repr(view) == render_value(array_value, _NO_DESCRIPTORS)


def test_dict_view_getitem_returns_the_encoded_value() -> None:
    view = AglDictView(DictValue({"one": IntValue(1)}), _NO_DESCRIPTORS)

    assert view["one"] == 1


def test_dict_view_contains_checks_key_membership() -> None:
    view = AglDictView(DictValue({"two": IntValue(2)}), _NO_DESCRIPTORS)

    assert "two" in view


def test_dict_view_setitem_with_a_non_string_key_raises_type_error() -> None:
    view = AglDictView(DictValue(), _NO_DESCRIPTORS)

    with pytest.raises(TypeError):
        view[1] = 1


def test_dict_view_setitem_with_an_undecodable_value_raises() -> None:
    view = AglDictView(DictValue(), _NO_DESCRIPTORS)

    with pytest.raises(BoundaryTypeError):
        view["bad"] = object()


def test_dict_view_popitem_removes_and_returns_the_last_entry() -> None:
    view = AglDictView(DictValue({"one": IntValue(1), "two": IntValue(2)}), _NO_DESCRIPTORS)

    assert view.popitem() == ("two", 2)


def test_dict_view_delitem_removes_an_entry() -> None:
    dict_value = DictValue({"one": IntValue(1), "two": IntValue(2)})
    view = AglDictView(dict_value, _NO_DESCRIPTORS)

    del view["one"]

    assert dict_value.entries == {"two": IntValue(2)}


def test_dict_view_clear_empties_the_mapping() -> None:
    view = AglDictView(DictValue({"one": IntValue(1)}), _NO_DESCRIPTORS)

    view.clear()

    assert not view
    assert list(view) == []


def test_dict_view_is_not_equal_to_a_non_view_object() -> None:
    view = AglDictView(DictValue(), _NO_DESCRIPTORS)

    assert view != object()


def test_dict_views_over_the_same_container_compare_equal_and_hash_alike() -> None:
    dict_value = DictValue({"one": IntValue(1)})

    assert AglDictView(dict_value, _NO_DESCRIPTORS) == AglDictView(dict_value, _NO_DESCRIPTORS)
    assert hash(AglDictView(dict_value, _NO_DESCRIPTORS)) == hash(
        AglDictView(dict_value, _NO_DESCRIPTORS)
    )


def test_dict_view_repr_matches_the_rendered_agl_value() -> None:
    dict_value = DictValue({"one": IntValue(1)})
    view = AglDictView(dict_value, _NO_DESCRIPTORS)

    assert repr(view) == render_value(dict_value, _NO_DESCRIPTORS)


def test_array_extend_growing_from_itself_terminates() -> None:
    array_value = ArrayValue([IntValue(1), IntValue(2)])
    view = AglArrayView(array_value, _NO_DESCRIPTORS)

    view.extend(view)

    assert array_value.elements == [IntValue(1), IntValue(2), IntValue(1), IntValue(2)]


def test_array_extend_does_not_partially_mutate_on_a_bad_element() -> None:
    array_value = ArrayValue([IntValue(1)])
    view = AglArrayView(array_value, _NO_DESCRIPTORS)

    with pytest.raises(BoundaryTypeError):
        view.extend([2, 3.5, 4])

    assert array_value.elements == [IntValue(1)]


def test_array_sort_accepts_a_key_function() -> None:
    array_value = ArrayValue([TextValue("ccc"), TextValue("a"), TextValue("bb")])
    view = AglArrayView(array_value, _NO_DESCRIPTORS)

    view.sort(key=len)

    assert list(view) == ["a", "bb", "ccc"]


def test_array_sort_without_a_key_still_sorts_by_encoded_value() -> None:
    array_value = ArrayValue([IntValue(3), IntValue(1), IntValue(2)])
    view = AglArrayView(array_value, _NO_DESCRIPTORS)

    view.sort(reverse=True)

    assert list(view) == [3, 2, 1]


def test_array_sort_in_reverse_preserves_equal_key_order() -> None:
    array_value = ArrayValue([TextValue("first"), TextValue("second"), TextValue("third")])
    view = AglArrayView(array_value, _NO_DESCRIPTORS)

    view.sort(key=lambda _: 0, reverse=True)

    assert list(view) == ["first", "second", "third"]


def test_array_contains_returns_false_for_an_undecodable_probe() -> None:
    array_value = ArrayValue([IntValue(1)])
    view = AglArrayView(array_value, _NO_DESCRIPTORS)

    assert object() not in view


#: Identity generator for this module's registry-level tests. The synthesized
#: class registry (``boundary._NOMINAL_CLASSES``) is process-global, keyed by
#: ``NominalId``, and never drops an entry -- and a ``NominalId`` is opaque, so
#: it carries no module or name to keep two nominals apart. Every identity
#: minted here is therefore distinct from every other one AND far above the
#: declaration node ids a really-lowered program allocates, so no test can
#: silently inherit (or clobber) another's synthesized class.
_next_test_nominal = itertools.count(9_000_000)

#: An identity nothing ever registers a class for, at the top of this module's
#: range. Kept out of ``_next_test_nominal``'s reach so "unregistered" stays
#: true however many nominals the tests below synthesize, in any order.
_UNREGISTERED_NOMINAL = NominalId(9_999_999)


def _fresh_nominal() -> NominalId:
    """Return an identity no other test in this module uses."""
    return NominalId(next(_next_test_nominal))


def _synthesize_box_class() -> tuple[NominalId, type[object]]:
    """Build a fresh synthesized ``Box`` record class for one test's isolated use."""
    nominal = _fresh_nominal()
    descriptor = NominalDescriptor(
        nominal=nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Box",
        kind=NominalKind.RECORD,
        fields=("value",),
    )
    return nominal, synthesize_nominal_classes((descriptor,))[nominal]


def _synthesize_choice_classes() -> tuple[NominalId, type[object]]:
    """Build a fresh synthesized ``Choice`` enum class for one test's isolated use."""
    nominal = _fresh_nominal()
    some = _fresh_nominal()
    none = _fresh_nominal()
    descriptor = NominalDescriptor(
        nominal=nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Choice",
        kind=NominalKind.ENUM,
        variants=(VariantDescriptor("Some", ("value",), some), VariantDescriptor("None", (), none)),
    )
    return nominal, synthesize_nominal_classes(
        (
            descriptor,
            NominalDescriptor(some, ENTRY_ID, ("Choice",), "Some", NominalKind.RECORD, ("value",)),
            NominalDescriptor(none, ENTRY_ID, ("Choice",), "None", NominalKind.RECORD),
        )
    )[nominal]


def _synthesize_problem_class() -> tuple[NominalId, type[object]]:
    """Build a fresh synthesized ``Problem`` exception class for one test's isolated use."""
    nominal = _fresh_nominal()
    descriptor = NominalDescriptor(
        nominal=nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Problem",
        kind=NominalKind.EXCEPTION,
        fields=("detail",),
    )
    return nominal, synthesize_nominal_classes((descriptor,))[nominal]


def test_unit_value_round_trips_through_the_boundary() -> None:
    assert encode_boundary_value(UNIT_VALUE, _NO_DESCRIPTORS) is None
    assert decode_boundary_value(None) == UNIT_VALUE


def test_bool_value_round_trips_through_the_boundary() -> None:
    assert encode_boundary_value(BoolValue(True), _NO_DESCRIPTORS) is True
    assert decode_boundary_value(True) == BoolValue(True)


def test_int_value_round_trips_through_the_boundary() -> None:
    assert encode_boundary_value(IntValue(1), _NO_DESCRIPTORS) == 1
    assert decode_boundary_value(1) == IntValue(1)


def test_decimal_value_round_trips_through_the_boundary() -> None:
    assert encode_boundary_value(DecimalValue(Decimal("1.5")), _NO_DESCRIPTORS) == Decimal("1.5")
    assert decode_boundary_value(Decimal("1.5")) == DecimalValue(Decimal("1.5"))


def test_text_value_round_trips_through_the_boundary() -> None:
    assert encode_boundary_value(TextValue("x"), _NO_DESCRIPTORS) == "x"
    assert decode_boundary_value("x") == TextValue("x")


def test_encode_boundary_value_wraps_json_in_an_agl_json_object() -> None:
    assert encode_boundary_value(JsonValue(None), _NO_DESCRIPTORS) == AglJson(None)


def test_encode_boundary_value_wraps_array_in_a_live_array_view() -> None:
    assert isinstance(
        encode_boundary_value(ArrayValue([IntValue(1)]), _NO_DESCRIPTORS), AglArrayView
    )


def test_encode_boundary_value_wraps_dict_in_a_live_dict_view() -> None:
    assert isinstance(
        encode_boundary_value(DictValue({"x": IntValue(1)}), _NO_DESCRIPTORS), AglDictView
    )


def test_decode_boundary_value_returns_the_array_views_wrapped_value() -> None:
    array_value = ArrayValue([])
    view = AglArrayView(array_value, _NO_DESCRIPTORS)

    assert decode_boundary_value(view) is array_value


def test_decode_boundary_value_returns_the_dict_views_wrapped_value() -> None:
    dict_value = DictValue()
    view = AglDictView(dict_value, _NO_DESCRIPTORS)

    assert decode_boundary_value(view) is dict_value


def test_encode_boundary_value_rejects_an_unregistered_nominal() -> None:
    """An identity with no synthesized class cannot cross, whatever else was synthesized.

    Regression test: the synthesized class registry is process-global and
    keyed by the now-opaque ``NominalId``, so a nominal that reuses another
    test's identity silently inherits its class and encodes instead of being
    rejected. Synthesizing the module's other shapes first pins that this
    identity stays unregistered regardless of test order.
    """
    _synthesize_box_class()
    _synthesize_choice_classes()
    _synthesize_problem_class()
    with pytest.raises(BoundaryViolation):
        encode_boundary_value(RecordValue(_UNREGISTERED_NOMINAL, {}), _NO_DESCRIPTORS)


def test_encode_boundary_value_names_a_known_but_unsynthesized_nominal_in_its_message() -> None:
    """A descriptor view can name the nominal even when no class was ever synthesized for it."""
    known_descriptors = ValueDescriptors(
        nominals={
            _UNREGISTERED_NOMINAL: NominalDescriptor(
                nominal=_UNREGISTERED_NOMINAL,
                module_id=ENTRY_ID,
                scope_path=(),
                declared_name="Ghost",
                kind=NominalKind.RECORD,
                fields=(),
            )
        },
        functions={},
    )
    with pytest.raises(BoundaryViolation, match="Ghost"):
        encode_boundary_value(RecordValue(_UNREGISTERED_NOMINAL, {}), known_descriptors)


def test_encode_boundary_value_rejects_a_constructor_value() -> None:
    with pytest.raises(BoundaryViolation):
        encode_boundary_value(ConstructorValue(_fresh_nominal()), _NO_DESCRIPTORS)


def test_decode_boundary_value_rejects_an_unsupported_python_object() -> None:
    with pytest.raises(BoundaryViolation):
        decode_boundary_value(object())


def test_synthesized_record_round_trips_through_decode() -> None:
    nominal, box_cls = _synthesize_box_class()

    box = box_cls(value=1)

    assert decode_boundary_value(box) == RecordValue(nominal, {"value": IntValue(1)})


def test_synthesized_record_instances_are_immutable() -> None:
    _, box_cls = _synthesize_box_class()
    box = box_cls(value=1)

    with pytest.raises(AttributeError):
        box.value = 2


def test_synthesized_record_constructor_requires_all_fields() -> None:
    _, box_cls = _synthesize_box_class()

    with pytest.raises(TypeError):
        box_cls()


def test_synthesized_record_instances_with_equal_fields_are_equal_and_hash_alike() -> None:
    _, box_cls = _synthesize_box_class()

    assert box_cls(value=1) == box_cls(value=1)
    assert hash(box_cls(value=1)) == hash(box_cls(value=1))


def test_synthesized_record_instances_with_different_fields_are_not_equal() -> None:
    _, box_cls = _synthesize_box_class()

    assert box_cls(value=1) != box_cls(value=2)


def test_synthesized_record_instance_is_not_equal_to_an_unrelated_object() -> None:
    _, box_cls = _synthesize_box_class()

    assert box_cls(value=1) != object()


def test_synthesized_record_instance_raises_on_unknown_field_access() -> None:
    _, box_cls = _synthesize_box_class()
    box = box_cls(value=1)

    with pytest.raises(AttributeError):
        box.missing


def test_synthesized_record_repr_identifies_the_type_without_the_field_values() -> None:
    _, box_cls = _synthesize_box_class()

    assert repr(box_cls(value=1)) == "Box(...)"


def test_synthesized_enum_variant_class_is_not_a_subclass_of_the_enum_class() -> None:
    """The enum class is a pure namespace over its members' own record classes:

    a member's runtime shape does not depend on how it was declared, so a
    variant is never a subclass of its enum -- matching a bare record
    referenced by a qualified name, which also never subclasses the enums
    that list it as a member.
    """
    _, choice_cls = _synthesize_choice_classes()

    some = choice_cls.Some(value=2)

    assert not issubclass(choice_cls.Some, choice_cls)
    assert not isinstance(some, choice_cls)


def test_synthesized_enum_variant_round_trips_through_decode() -> None:
    _, choice_cls = _synthesize_choice_classes()
    some = choice_cls.Some(value=2)

    assert decode_boundary_value(some) == RecordValue(
        nominal=getattr(choice_cls.Some, "_agl_nominal"),
        fields={"value": IntValue(2)},
    )


def test_encoding_an_enum_value_produces_the_matching_variant_class() -> None:
    _, choice_cls = _synthesize_choice_classes()

    encoded = encode_boundary_value(
        RecordValue(
            nominal=getattr(getattr(choice_cls, "None"), "_agl_nominal"),
            fields={},
        ),
        _NO_DESCRIPTORS,
    )

    assert isinstance(encoded, getattr(choice_cls, "None"))


def test_synthesized_exception_round_trips_through_decode() -> None:
    nominal, problem_cls = _synthesize_problem_class()

    problem = problem_cls(detail="bad")

    assert decode_boundary_value(problem) == ExceptionValue(nominal, {"detail": TextValue("bad")})


def test_json_payload_accepts_every_scalar_json_shape() -> None:
    assert decode_boundary_value(AglJson(None)) == JsonValue(None)
    assert decode_boundary_value(AglJson(True)) == JsonValue(True)
    assert decode_boundary_value(AglJson("x")) == JsonValue("x")
    assert decode_boundary_value(AglJson(3)) == JsonValue(3)
    assert decode_boundary_value(AglJson(Decimal("2.5"))) == JsonValue(Decimal("2.5"))


def test_json_payload_accepts_nested_containers() -> None:
    assert decode_boundary_value(AglJson({"x": []})) == JsonValue({"x": []})


def test_json_payload_crosses_the_boundary_without_copying() -> None:
    payload = [1, 2]

    value = decode_boundary_value(AglJson(payload))

    assert isinstance(value, JsonValue)
    assert value.raw is payload


def test_json_value_encodes_to_its_own_payload_without_copying() -> None:
    value = JsonValue([1, 2])

    encoded = encode_boundary_value(value, _NO_DESCRIPTORS)

    assert isinstance(encoded, AglJson)
    assert encoded.value is value.raw


def test_synthesized_nominals_support_non_python_field_names() -> None:
    nominal = _fresh_nominal()
    descriptor = NominalDescriptor(
        nominal=nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Prompt",
        kind=NominalKind.RECORD,
        fields=("ask-prompt", "count"),
    )
    classes = synthesize_nominal_classes((descriptor,))
    prompt = classes[nominal](**{"ask-prompt": "continue", "count": 3})

    assert getattr(prompt, "ask-prompt") == "continue"
    assert prompt.count == 3
    assert decode_boundary_value(prompt) == RecordValue(
        nominal,
        {"ask-prompt": TextValue("continue"), "count": IntValue(3)},
    )


def test_deep_recursive_nominal_construction_terminates() -> None:
    """Decoding a deeply nested boundary value must not degrade into nontermination.

    The failure mode this guards is an unbounded or superlinear walk of the
    nesting, so the deadline only has to tell finite from infinite; a tight
    wall-clock budget would instead report a loaded machine as a regression.
    """
    nominal = _fresh_nominal()
    descriptor = NominalDescriptor(
        nominal=nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Box",
        kind=NominalKind.RECORD,
        fields=("value", "inner"),
    )
    classes = synthesize_nominal_classes((descriptor,))
    box_cls = classes[nominal]

    node: object = None
    for value in range(300):
        node = box_cls(value=value, inner=node)
    with fail_if_slow("decoding a deeply nested boundary value did not terminate"):
        decoded = decode_boundary_value(node)

    depth = 0
    current: Value = decoded
    while isinstance(current, RecordValue) and isinstance(current.fields["inner"], RecordValue):
        depth += 1
        current = current.fields["inner"]
    assert depth == 299


def test_shared_immutable_record_graph_stays_shared_across_the_boundary() -> None:
    leaf = _fresh_nominal()
    branch = _fresh_nominal()
    descriptors = (
        NominalDescriptor(leaf, ENTRY_ID, (), "Leaf", NominalKind.RECORD),
        NominalDescriptor(
            branch,
            ENTRY_ID,
            (),
            "Branch",
            NominalKind.RECORD,
            ("left", "right"),
        ),
    )
    synthesize_nominal_classes(descriptors)
    value: RecordValue = RecordValue(leaf, {})
    for _ in range(12):
        value = RecordValue(branch, {"left": value, "right": value})

    encoded = encode_boundary_value(value, _NO_DESCRIPTORS)

    encoded_nodes = [encoded]
    for _ in range(12):
        current = encoded_nodes[-1]
        left = getattr(current, "left")
        assert left is getattr(current, "right")
        encoded_nodes.append(left)
    assert len({id(node) for node in encoded_nodes}) == 13

    decoded = decode_boundary_value(encoded)
    assert isinstance(decoded, RecordValue)
    decoded_nodes = [decoded]
    for _ in range(12):
        current = decoded_nodes[-1]
        left = current.fields["left"]
        assert left is current.fields["right"]
        assert isinstance(left, RecordValue)
        decoded_nodes.append(left)
    assert len({id(node) for node in decoded_nodes}) == 13

    separately_encoded = encode_boundary_value(value, _NO_DESCRIPTORS)
    separately_decoded = decode_boundary_value(encoded)
    assert separately_encoded is not encoded
    assert separately_decoded is not decoded


def test_synthesizing_an_already_present_identity_reuses_its_class_unchanged() -> None:
    """A ``NominalId``'s layout is fixed at its declaration: a class, once
    synthesized for an identity, is reused verbatim on every later call that
    carries the same identity again -- never re-shaped in place. A real
    lowering mints a fresh identity for a redeclaration rather than repeating
    one (the test below covers that shape), but the insert-only contract
    itself holds regardless of what a second descriptor for the same identity
    claims.
    """
    nominal = _fresh_nominal()
    some = _fresh_nominal()
    gone = _fresh_nominal()
    first = NominalDescriptor(
        nominal=nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Choice",
        kind=NominalKind.ENUM,
        variants=(VariantDescriptor("Some", ("value",), some), VariantDescriptor("Gone", (), gone)),
    )
    classes = synthesize_nominal_classes((first,))
    enum_cls = classes[nominal]
    some_cls = enum_cls.Some

    second = NominalDescriptor(
        nominal=nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Choice",
        kind=NominalKind.ENUM,
        variants=(VariantDescriptor("Some", ("value", "extra"), some),),
    )
    reused = synthesize_nominal_classes((second,), classes)

    assert reused[nominal] is enum_cls
    assert enum_cls.Some is some_cls
    assert some_cls._agl_fields == ("value",)
    assert hasattr(enum_cls, "Gone")


def test_referenced_record_keeps_one_class_across_multiple_enums() -> None:
    record = _fresh_nominal()
    left = _fresh_nominal()
    right = _fresh_nominal()
    descriptors = (
        NominalDescriptor(record, ENTRY_ID, (), "Shared", NominalKind.RECORD, ("value",)),
        NominalDescriptor(
            left,
            ENTRY_ID,
            (),
            "Left",
            NominalKind.ENUM,
            variants=(VariantDescriptor("Shared", ("value",), record),),
        ),
        NominalDescriptor(
            right,
            ENTRY_ID,
            (),
            "Right",
            NominalKind.ENUM,
            variants=(VariantDescriptor("Shared", ("value",), record),),
        ),
    )

    classes = synthesize_nominal_classes(descriptors)

    assert classes[record] is classes[left].Shared
    assert classes[record] is classes[right].Shared


def test_referenced_member_decodes_with_its_own_scope_and_display_name() -> None:
    """A referenced member decodes through its own descriptor, not the enum's:

    its ``NominalId`` and display name come from where it was declared, even
    though an enum also lists it as a member.
    """
    record = _fresh_nominal()
    enum = _fresh_nominal()
    descriptors = (
        NominalDescriptor(record, ENTRY_ID, ("M",), "Go", NominalKind.RECORD, ("amount",)),
        NominalDescriptor(
            enum,
            ENTRY_ID,
            (),
            "Step",
            NominalKind.ENUM,
            variants=(VariantDescriptor("Go", ("amount",), record),),
        ),
    )
    classes = synthesize_nominal_classes(descriptors)
    instance = classes[enum].Go(amount=5)

    assert classes[record] is classes[enum].Go
    assert not issubclass(classes[enum].Go, classes[enum])
    assert decode_boundary_value(instance) == RecordValue(record, {"amount": IntValue(5)})
    assert getattr(classes[enum].Go, "_agl_descriptor").display_name == "M::Go"


def test_enum_variant_built_without_its_own_descriptor_gets_a_scoped_display_name() -> None:
    """The fallback path :func:`synthesize_nominal_classes` takes when a
    member's own descriptor is reachable in neither the current synthesis
    batch nor the existing class table (an inline member whose descriptor was
    never separately supplied) derives one that names the member below its
    enum's scope, matching what a real lowering would have produced for it.
    """
    nominal = _fresh_nominal()
    some = _fresh_nominal()
    descriptor = NominalDescriptor(
        nominal=nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Choice",
        kind=NominalKind.ENUM,
        variants=(VariantDescriptor("Some", ("value",), some),),
    )
    classes = synthesize_nominal_classes((descriptor,))
    instance = classes[nominal].Some(value=1)

    assert decode_boundary_value(instance) == RecordValue(some, {"value": IntValue(1)})
    assert getattr(classes[nominal].Some, "_agl_descriptor").display_name == "Choice::Some"


def test_companion_namespace_keeps_same_named_nominals_distinct() -> None:
    left = _fresh_nominal()
    right = _fresh_nominal()
    registry = ExternRegistry()
    registry.set_nominals(
        {
            left: NominalDescriptor(
                nominal=left,
                module_id=ModuleId.from_path("left"),
                scope_path=(),
                declared_name="Box",
                kind=NominalKind.RECORD,
                fields=("left",),
            ),
            right: NominalDescriptor(
                nominal=right,
                module_id=ModuleId.from_path("right"),
                scope_path=(),
                declared_name="Box",
                kind=NominalKind.RECORD,
                fields=("right",),
            ),
        },
    )

    agl = registry._agl_module()

    assert not hasattr(agl, "Box")
    assert agl.nominals.left.Box is registry._nominal_classes[left]
    assert agl.nominals.right.Box is registry._nominal_classes[right]


def test_companion_namespace_resolves_a_shared_name_path_to_the_current_bearer() -> None:
    """Which identity a shared name path resolves to is decided by
    ``NominalDescriptor.bears_name_path``, never by the order the classes were
    synthesized in. The superseded identity is registered LAST here, so a
    namespace built in insertion order alone would hand a companion importing
    now the stale class under both spellings.
    """
    current = _fresh_nominal()
    superseded = _fresh_nominal()
    bearer = NominalDescriptor(
        nominal=current,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Box",
        kind=NominalKind.RECORD,
        fields=("new",),
    )
    registry = ExternRegistry()
    registry.set_nominals({current: bearer})
    registry.set_nominals(
        {
            current: bearer,
            superseded: NominalDescriptor(
                nominal=superseded,
                module_id=ENTRY_ID,
                scope_path=(),
                declared_name="Box",
                kind=NominalKind.RECORD,
                fields=("old",),
                bears_name_path=False,
            ),
        },
    )

    agl = registry._agl_module()

    assert agl.Box is registry._nominal_classes[current]
    assert agl.nominals.entry.Box is registry._nominal_classes[current]


def test_re_registering_the_same_identity_reuses_its_synthesized_class() -> None:
    nominal = _fresh_nominal()
    descriptor = NominalDescriptor(
        nominal=nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Box",
        kind=NominalKind.RECORD,
        fields=("value",),
    )
    registry = ExternRegistry()
    registry.set_nominals({nominal: descriptor})
    before = registry._nominal_classes[nominal]

    registry.set_nominals({nominal: descriptor})

    assert registry._nominal_classes[nominal] is before


def test_redeclaring_a_nominal_keeps_default_argument_captured_classes_on_the_old_shape(
    tmp_path: Path,
) -> None:
    """A redeclaration lowers to two DISTINCT identities sharing one name
    path, never to the same ``NominalId`` twice. The class a companion
    captured by default argument before the redeclaration keeps constructing
    and recognizing the OLD identity's shape; it is never migrated to the new
    one. A companion importing afterward instead sees the new declaration
    under both its bare name and its ``nominals`` path.
    """
    old_nominal = _fresh_nominal()
    new_nominal = _fresh_nominal()
    old = NominalDescriptor(
        nominal=old_nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Box",
        kind=NominalKind.RECORD,
        fields=("old",),
    )
    new = NominalDescriptor(
        nominal=new_nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Box",
        kind=NominalKind.RECORD,
        fields=("new",),
    )
    companion = tmp_path / "companion.py"
    companion.write_text("from agl import Box\ndef make(box_cls=Box):\n    return box_cls(old=2)\n")
    registry = ExternRegistry()
    registry.set_nominals({old_nominal: old})
    registry.load_companion(ENTRY_ID, companion)
    before = registry._nominal_classes[old_nominal]

    # A real lowering re-derives every descriptor on every call, so the old
    # identity's own snapshot is refreshed too -- no longer bearing the name
    # path the new identity now claims.
    old_superseded = NominalDescriptor(
        nominal=old_nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Box",
        kind=NominalKind.RECORD,
        fields=("old",),
        bears_name_path=False,
    )
    registry.set_nominals({old_nominal: old_superseded, new_nominal: new})

    assert registry._nominal_classes[old_nominal] is before
    assert registry.invoke(
        "make",
        registry.resolve(ENTRY_ID, "make"),
        (),
        nominals=NO_BUILTIN_DECLARATIONS,
        descriptors=_NO_DESCRIPTORS,
    ) == RecordValue(old_nominal, {"old": IntValue(2)})

    agl = registry._agl_module()
    assert agl.Box is registry._nominal_classes[new_nominal]
    assert agl.nominals.entry.Box is registry._nominal_classes[new_nominal]


def test_stashed_view_with_nominal_elements_decodes_outside_any_call(tmp_path: Path) -> None:
    nominal = _fresh_nominal()
    descriptor = NominalDescriptor(
        nominal=nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Inner",
        kind=NominalKind.RECORD,
        fields=("x",),
    )
    registry = ExternRegistry()
    registry.set_nominals({nominal: descriptor})
    companion = tmp_path / "companion.py"
    companion.write_text(
        "STASHED = []\n"
        "def stash(xs):\n"
        "    STASHED.append(xs)\n"
        "def read_stashed():\n"
        "    return STASHED[0][0]\n"
    )
    module = registry.load_companion(ENTRY_ID, companion)
    inner = registry._nominal_classes[nominal](x=1)
    array_value = ArrayValue([decode_boundary_value(inner)])
    registry.invoke(
        "stash",
        registry.resolve(ENTRY_ID, "stash"),
        [array_value],
        nominals=NO_BUILTIN_DECLARATIONS,
        descriptors=_NO_DESCRIPTORS,
    )

    # Read the stashed view's nominal element completely outside any `invoke`
    # call, exactly as a companion callback running later would.
    read_directly = module.read_stashed
    assert decode_boundary_value(read_directly()) == RecordValue(nominal, {"x": IntValue(1)})


def test_registry_wraps_unexpected_decode_errors_as_extern_errors() -> None:
    nominal = _fresh_nominal()
    descriptor = NominalDescriptor(
        nominal=nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Box",
        kind=NominalKind.RECORD,
        fields=("value",),
    )
    registry = ExternRegistry()
    registry.set_nominals({nominal: descriptor})
    broken = registry._nominal_classes[nominal](value=1)
    del broken._agl_values["value"]

    with pytest.raises(AglRaise) as excinfo:
        registry.invoke(
            "broken",
            lambda: broken,
            (),
            nominals=NO_BUILTIN_DECLARATIONS,
            descriptors=_NO_DESCRIPTORS,
        )

    assert excinfo.value.exc.fields["python-type"] == TextValue("KeyError")
