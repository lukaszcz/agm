"""Live AgL views exposed to extern companions."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Protocol, cast

import pytest

from agm.agl.ir.ids import FunctionId, NominalId, SymbolId
from agm.agl.ir.program import (
    ExternFunctionBody,
    FunctionDescriptor,
    NominalDescriptor,
    NominalKind,
    ValueDescriptors,
    VariantDescriptor,
)
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.runtime.boundary import (
    AglArrayView,
    AglDictView,
    BoundaryTypeError,
    BoundaryViolation,
    active_function_encoder,
    decode_boundary_value,
    encode_boundary_value,
    synthesize_nominal_classes,
)
from agm.agl.runtime.externs import AglCallableProxy, ExternCallWindow
from agm.agl.semantics.values import (
    ArrayValue,
    DictValue,
    IntValue,
    IrClosureValue,
    RecordValue,
    TextValue,
)
from tests.agl.ir_harness import evaluate_ir_with_externs


class _RecordCompanion(Protocol):
    value: object
    fixed: object

    def __init__(self, **fields: object) -> None: ...


class _EventCompanion(Protocol):
    Changed: type[_RecordCompanion]


_next_nominal = itertools.count(80_000_000)

#: An empty descriptor view for tests that never check a rendered spelling.
_NO_DESCRIPTORS = ValueDescriptors(nominals={}, functions={})


def _next_id() -> NominalId:
    return NominalId(next(_next_nominal))


def _record_class(
    *, mutable: bool, fields: tuple[str, ...] = ("value", "fixed")
) -> tuple[NominalId, type[_RecordCompanion], ValueDescriptors]:
    nominal = _next_id()
    descriptor = NominalDescriptor(
        nominal=nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Mutable" if mutable else "Snapshot",
        kind=NominalKind.RECORD,
        fields=fields,
        mutable_fields=frozenset({fields[0]}) if mutable else frozenset(),
    )
    record_cls = cast(type[_RecordCompanion], synthesize_nominal_classes((descriptor,))[nominal])
    descriptors = ValueDescriptors(nominals={nominal: descriptor}, functions={})
    return nominal, record_cls, descriptors


def test_array_views_over_the_same_container_compare_and_hash_alike() -> None:
    value = ArrayValue([IntValue(1)])
    first = AglArrayView(value, _NO_DESCRIPTORS)
    second = AglArrayView(value, _NO_DESCRIPTORS)

    assert first == second
    assert hash(first) == hash(second)


def test_dict_views_remain_live_without_a_call_scope() -> None:
    value = DictValue({"one": IntValue(1)})
    view = AglDictView(value, _NO_DESCRIPTORS)
    view["two"] = 2

    assert value.entries == {"one": IntValue(1), "two": IntValue(2)}


def test_mutable_record_view_reads_current_values_and_writes_through() -> None:
    nominal, record_cls, descriptors = _record_class(mutable=True)
    value = RecordValue(nominal, {"value": IntValue(1), "fixed": IntValue(2)})
    view = cast(_RecordCompanion, encode_boundary_value(value, descriptors))

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
    nominal, record_cls, descriptors = _record_class(mutable=True)
    original = RecordValue(nominal, {"value": IntValue(1), "fixed": IntValue(2)})
    view = cast(_RecordCompanion, encode_boundary_value(original, descriptors))

    assert decode_boundary_value(view) is original

    with pytest.raises(TypeError):
        record_cls(value=3)
    constructed = record_cls(value=3, fixed=4)
    decoded = decode_boundary_value(constructed)
    assert decoded == RecordValue(nominal, {"value": IntValue(3), "fixed": IntValue(4)})
    assert decoded is not original


def test_mutable_record_view_storage_shadows_a_field_that_collides_with_it() -> None:
    """Ordinary attribute lookup wins in both directions, as for a snapshot."""
    nominal, _, descriptors = _record_class(mutable=True, fields=("_agl_value",))
    value = RecordValue(nominal, {"_agl_value": IntValue(1)})
    view = encode_boundary_value(value, descriptors)

    assert getattr(view, "_agl_value") is value
    with pytest.raises(AttributeError):
        setattr(view, "_agl_value", 3)

    # The field itself stays reachable, just not by dot access.
    assert decode_boundary_value(view).fields["_agl_value"] == IntValue(1)


def _callback_box_class() -> tuple[NominalId, type[_RecordCompanion]]:
    nominal = _next_id()
    descriptor = NominalDescriptor(
        nominal=nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Box",
        kind=NominalKind.RECORD,
        fields=("value", "callback"),
        mutable_fields=frozenset({"callback"}),
    )
    return nominal, cast(type[_RecordCompanion], synthesize_nominal_classes((descriptor,))[nominal])


def test_companion_constructed_view_encodes_any_closure_through_the_active_encoder() -> None:
    """A view a companion built itself is a window onto every AgL function.

    It has no encoder of its own, so it reads the one the active extern call
    published -- which is what lets it encode a closure the companion never
    handed it (here, one AgL wrote into the record afterwards).
    """
    _, record_cls = _callback_box_class()
    window = ExternCallWindow()

    results = {FunctionId(1): IntValue(10), FunctionId(2): IntValue(20)}

    def encode(closure: IrClosureValue) -> object:
        return AglCallableProxy(
            arity=1,
            closure=closure,
            require_active_window=window.require_active,
            invoke=lambda _args: results[closure.function_id],
        )

    with active_function_encoder(encode):
        view = record_cls(value=0, callback=encode(IrClosureValue(FunctionId(1), ())))
        with window.active():
            assert getattr(view, "callback")(2) == 10

        # A closure the companion never saw, written straight into the record.
        decode_boundary_value(view).fields["callback"] = IrClosureValue(FunctionId(2), ())
        with window.active():
            assert getattr(view, "callback")(2) == 20


def test_companion_constructed_view_reports_a_closure_with_no_active_encoder() -> None:
    _, record_cls = _callback_box_class()
    view = record_cls(value=0, callback=1)
    decode_boundary_value(view).fields["callback"] = IrClosureValue(FunctionId(1), ())

    with pytest.raises(BoundaryViolation):
        getattr(view, "callback")


def test_mutable_record_views_compare_structurally_and_are_unhashable() -> None:
    nominal, _, descriptors = _record_class(mutable=True)
    first = cast(
        _RecordCompanion,
        encode_boundary_value(
            RecordValue(nominal, {"value": IntValue(1), "fixed": IntValue(2)}), descriptors
        ),
    )
    second = cast(
        _RecordCompanion,
        encode_boundary_value(
            RecordValue(nominal, {"value": IntValue(1), "fixed": IntValue(2)}), descriptors
        ),
    )

    assert first == second
    assert first != object()
    with pytest.raises(TypeError):
        hash(first)


def test_immutable_record_snapshots_keep_their_equality_hashing_and_write_rejection() -> None:
    nominal, record_cls, descriptors = _record_class(mutable=False)
    first = cast(
        _RecordCompanion,
        encode_boundary_value(
            RecordValue(nominal, {"value": IntValue(1), "fixed": IntValue(2)}), descriptors
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
            mutable_fields=frozenset({"value"}),
        ),
    )
    event_cls = cast(_EventCompanion, synthesize_nominal_classes(descriptors)[enum_nominal])
    value = RecordValue(member_nominal, {"value": IntValue(1), "fixed": IntValue(2)})
    view = cast(_RecordCompanion, encode_boundary_value(value, _NO_DESCRIPTORS))

    assert isinstance(view, event_cls.Changed)
    view.value = 3

    assert value.fields["value"] == IntValue(3)
    assert decode_boundary_value(view) is value


def test_view_retained_past_a_call_reads_plain_fields_but_not_a_function_field() -> None:
    """A retained view stays live; the functions it holds stay call-scoped.

    Encoding a closure needs the interpreter the active extern call
    publishes, so a function field is readable during a call (a later one
    included) and reports outside every call, while every other field keeps
    reading through.
    """
    nominal, record_cls = _callback_box_class()
    window = ExternCallWindow()

    def encode(closure: IrClosureValue) -> object:
        return AglCallableProxy(
            arity=0,
            closure=closure,
            require_active_window=window.require_active,
            invoke=lambda _args: IntValue(7),
        )

    view = cast(
        _RecordCompanion,
        encode_boundary_value(
            RecordValue(
                nominal,
                {"value": IntValue(1), "callback": IrClosureValue(FunctionId(1), ())},
            ),
            _NO_DESCRIPTORS,
        ),
    )

    with active_function_encoder(encode):
        assert getattr(view, "callback") is not None

    assert view.value == 1
    with pytest.raises(BoundaryViolation):
        getattr(view, "callback")


def test_companion_construction_reports_an_unsupported_field_as_a_type_error() -> None:
    """A companion's own bad value is a type error wherever it is written."""
    _, record_cls, _ = _record_class(mutable=True)

    with pytest.raises(BoundaryTypeError):
        record_cls(value=object(), fixed=1)


def test_mutable_record_view_separates_an_unknown_name_from_an_immutable_field() -> None:
    nominal, _, descriptors = _record_class(mutable=True)
    view = cast(
        _RecordCompanion,
        encode_boundary_value(
            RecordValue(nominal, {"value": IntValue(1), "fixed": IntValue(2)}), descriptors
        ),
    )

    with pytest.raises(AttributeError, match=r"^valu$"):
        setattr(view, "valu", 3)
    with pytest.raises(AttributeError, match="immutable"):
        view.fixed = 3


def test_companion_repr_of_a_mutable_view_renders_a_nested_record_closure_and_array(
    tmp_path: Path,
) -> None:
    """A view's ``repr`` walks the raw AgL value, resolving every nested spelling.

    ``Outer`` crosses as a live view because it has a ``var`` field; that
    view's ``repr`` renders its nested ``Inner`` record, its function field's
    signature, and its array field, all from the one program's descriptors.
    """
    source = (
        "record Inner\n"
        "  a: int\n"
        "record Outer\n"
        "  var inner: Inner\n"
        "  var callback: (int) -> int\n"
        "  var items: array[int]\n"
        "extern def show(outer: Outer) -> text\n"
        "let increment = fn(value: int) -> int => value + 1\n"
        "let result = show(Outer(inner = Inner(a = 1), callback = increment, items = [1, 2]))\n"
        "result\n"
    )
    companion = "def show(outer): return repr(outer)\n"

    result, _ = evaluate_ir_with_externs(source, companion, tmp_path)

    assert result["result"] == TextValue(
        "Outer(inner = Inner(a = 1), callback = <function: int -> int>, items = [1, 2])"
    )


def test_companion_view_built_at_import_renders_after_the_import_returns(tmp_path: Path) -> None:
    """A view a companion builds at module import time still renders correctly later.

    ``agl.array`` runs during ``load_companion``'s top-level exec, before any
    call reaches this companion; the view it returns must still resolve
    ``Item``'s display name when a later call ``repr``'s it -- regression
    coverage for the import running outside the program's descriptor view.
    """
    source = "record Item\n  a: int\nextern def show() -> text\nlet result = show()\nresult\n"
    companion = (
        "import agl\n"
        "\n"
        "_cached = agl.array([agl.Item(a=1)])\n"
        "\n"
        "\n"
        "def show():\n"
        "    return repr(_cached)\n"
    )

    result, _ = evaluate_ir_with_externs(source, companion, tmp_path)

    assert result["result"] == TextValue("[Item(a = 1)]")


def test_two_programs_reusing_a_function_id_render_each_views_own_spelling() -> None:
    """A retained view's descriptors slot, not any shared table, resolves its closure.

    ``FunctionId(1)`` denotes a different function in each of two descriptor
    views built here, standing in for two independently lowered programs that
    each start numbering from the same per-program counter. Encoding a
    closure with that id under each view must render that view's own
    signature, and a later encode under the other view must not retroactively
    change what the first view renders -- the regression the removed
    module-level function-descriptor registry was prone to.
    """
    nominal = _next_id()
    descriptor = NominalDescriptor(
        nominal=nominal,
        module_id=ENTRY_ID,
        scope_path=(),
        declared_name="Box",
        kind=NominalKind.RECORD,
        fields=("fn",),
        mutable_fields=frozenset({"fn"}),
    )
    synthesize_nominal_classes((descriptor,))
    function_id = FunctionId(1)

    def function_descriptor(param_labels: tuple[str, ...], result_label: str) -> FunctionDescriptor:
        return FunctionDescriptor(
            function_id=function_id,
            function_symbol=SymbolId(1),
            module_id=ENTRY_ID,
            params=(),
            impl=ExternFunctionBody(name="fn", companion_name="fn"),
            param_labels=param_labels,
            result_label=result_label,
        )

    first_descriptors = ValueDescriptors(
        nominals={nominal: descriptor},
        functions={function_id: function_descriptor(("int",), "int")},
    )
    second_descriptors = ValueDescriptors(
        nominals={nominal: descriptor},
        functions={function_id: function_descriptor(("text",), "bool")},
    )
    value = RecordValue(nominal, {"fn": IrClosureValue(function_id, ())})

    first_view = encode_boundary_value(value, first_descriptors)
    second_view = encode_boundary_value(value, second_descriptors)

    assert repr(first_view) == "Box(fn = <function: int -> int>)"
    assert repr(second_view) == "Box(fn = <function: text -> bool>)"
    # The first view still reads through its own stored descriptors.
    assert repr(first_view) == "Box(fn = <function: int -> int>)"
