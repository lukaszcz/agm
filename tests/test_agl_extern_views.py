"""Unit specifications for live array and dict values at the extern boundary."""

from __future__ import annotations

from collections.abc import MutableMapping, MutableSequence
from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.contracts import (
    BoundaryArray,
    BoundaryDict,
    BoundaryEnum,
    BoundaryException,
    BoundaryRecord,
    BoundaryRef,
    BoundaryScalar,
    BoundarySchema,
    BoundarySealVar,
    BoundaryVariantShape,
    ExternContract,
    ScalarKind,
)
from agm.agl.ir.ids import NominalId
from agm.agl.parser import parse_program
from agm.agl.runtime.boundary import (
    AglArrayView,
    AglDictView,
    BoundaryScope,
    BoundaryTypeError,
    BoundaryViewRevoked,
    BoundaryViolation,
    SealedHandle,
    decode_boundary_value,
    encode_boundary_value,
)
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.runtime.render import render_value
from agm.agl.semantics.cycles import AglCyclicValue
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import (
    ArrayValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    JsonValue,
    RecordValue,
    TextValue,
)
from agm.agl.type_schema import build_extern_contract
from tests.agl.module_graph import resolve_and_check_program_ast

_PATH = Path("/virtual/extern_views.agl")
_CAPS = HostCapabilities(
    agent_names=frozenset(),
    has_default_agent=True,
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}),
    },
)


def build_contract(source: str, fn_name: str = "f") -> ExternContract:
    """Compile an extern contract from a real checked AgL signature."""
    checked = resolve_and_check_program_ast(parse_program(source), _CAPS, origin_path=_PATH)
    return build_extern_contract(checked.function_signatures[fn_name], checked.type_env.type_table)


def _nominal(schema: BoundarySchema) -> NominalId:
    assert isinstance(schema, (BoundaryRecord, BoundaryEnum, BoundaryException))
    return schema.nominal


def _array_schema(element: str = "int", type_params: str = "") -> BoundarySchema:
    contract = build_contract(
        f"extern def f{type_params}(xs: array[{element}]) -> array[{element}]\n0"
    )
    return contract.params[0].schema


def _dict_schema(value: str = "int", type_params: str = "") -> BoundarySchema:
    contract = build_contract(
        f"extern def f{type_params}(xs: dict[text, {value}]) -> dict[text, {value}]\n0"
    )
    return contract.params[0].schema


def _array_view(schema: BoundarySchema, value: ArrayValue, scope: BoundaryScope) -> AglArrayView:
    encoded = encode_boundary_value(schema, value, scope)
    assert isinstance(encoded, AglArrayView)
    return encoded


def _dict_view(schema: BoundarySchema, value: DictValue, scope: BoundaryScope) -> AglDictView:
    encoded = encode_boundary_value(schema, value, scope)
    assert isinstance(encoded, AglDictView)
    return encoded


class _CustomIndex:
    """An indexable object that is deliberately not an int."""

    def __init__(self, value: int) -> None:
        self._value = value

    def __index__(self) -> int:
        return self._value


class TestViewIdentityAndLiveness:
    @pytest.mark.parametrize("generic_first", [True, False])
    def test_mixed_generic_and_concrete_array_aliases_use_the_concrete_view_schema(
        self, generic_first: bool
    ) -> None:
        generic_schema = _array_schema("T", "[T]")
        concrete_schema = _array_schema()
        value = ArrayValue([IntValue(1)])
        scope = BoundaryScope(seals={"T": object()})
        schemas = (
            (generic_schema, concrete_schema)
            if generic_first
            else (concrete_schema, generic_schema)
        )

        first = encode_boundary_value(schemas[0], value, scope)
        second = encode_boundary_value(schemas[1], value, scope)

        assert isinstance(first, AglArrayView)
        assert first is second
        assert list(first) == [1]
        assert list(second) == [1]

    @pytest.mark.parametrize("generic_first", [True, False])
    def test_mixed_generic_and_concrete_dict_aliases_use_the_concrete_view_schema(
        self, generic_first: bool
    ) -> None:
        generic_schema = _dict_schema("T", "[T]")
        concrete_schema = _dict_schema()
        value = DictValue({"item": IntValue(1)})
        scope = BoundaryScope(seals={"T": object()})
        schemas = (
            (generic_schema, concrete_schema)
            if generic_first
            else (concrete_schema, generic_schema)
        )

        first = encode_boundary_value(schemas[0], value, scope)
        second = encode_boundary_value(schemas[1], value, scope)

        assert isinstance(first, AglDictView)
        assert first is second
        assert dict(first) == {"item": 1}
        assert dict(second) == {"item": 1}

    @pytest.mark.parametrize("generic_first", [True, False])
    def test_nested_mixed_generic_and_concrete_dict_aliases_use_concrete_value_views(
        self, generic_first: bool
    ) -> None:
        generic_schema = _dict_schema("array[T]", "[T]")
        concrete_schema = _dict_schema("array[int]")
        value = DictValue({"items": ArrayValue([IntValue(1)])})
        scope = BoundaryScope(seals={"T": object()})
        schemas = (
            (generic_schema, concrete_schema)
            if generic_first
            else (concrete_schema, generic_schema)
        )

        first = _dict_view(schemas[0], value, scope)
        second = _dict_view(schemas[1], value, scope)

        assert first is second
        nested = first["items"]
        assert isinstance(nested, AglArrayView)
        assert list(nested) == [1]

    @pytest.mark.parametrize("generic_first", [True, False])
    def test_nested_dict_generic_and_concrete_aliases_use_concrete_value_views(
        self, generic_first: bool
    ) -> None:
        generic_schema = _dict_schema("dict[text, T]", "[T]")
        concrete_schema = _dict_schema("dict[text, int]")
        schemas = (
            (generic_schema, concrete_schema)
            if generic_first
            else (concrete_schema, generic_schema)
        )
        scope = BoundaryScope(seals={"T": object()})
        value = DictValue({"items": DictValue({"item": IntValue(1)})})

        first = encode_boundary_value(schemas[0], value, scope)
        second = encode_boundary_value(schemas[1], value, scope)

        assert isinstance(first, AglDictView)
        assert first is second
        nested = first["items"]
        assert isinstance(nested, AglDictView)
        assert dict(nested) == {"item": 1}

    @pytest.mark.parametrize("int_first", [True, False])
    def test_incompatible_nested_dict_value_schemas_are_rejected_regardless_of_order(
        self, int_first: bool
    ) -> None:
        int_schema = _dict_schema("dict[text, int]")
        text_schema = _dict_schema("dict[text, text]")
        schemas = (int_schema, text_schema) if int_first else (text_schema, int_schema)
        scope = BoundaryScope()
        value = DictValue({"items": DictValue({"item": IntValue(1)})})

        encode_boundary_value(schemas[0], value, scope)
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(schemas[1], value, scope)

    @pytest.mark.parametrize("int_first", [True, False])
    def test_incompatible_nested_dict_alias_schemas_are_rejected_regardless_of_order(
        self, int_first: bool
    ) -> None:
        int_schema = _dict_schema("array[int]")
        text_schema = _dict_schema("array[text]")
        schemas = (int_schema, text_schema) if int_first else (text_schema, int_schema)
        scope = BoundaryScope()
        value = DictValue({"items": ArrayValue([IntValue(1)])})

        _dict_view(schemas[0], value, scope)
        with pytest.raises(BoundaryViolation):
            _dict_view(schemas[1], value, scope)

    @pytest.mark.parametrize("value_types", [("int", "text"), ("T", "U")])
    def test_incompatible_dict_alias_schemas_are_rejected_regardless_of_order(
        self, value_types: tuple[str, str]
    ) -> None:
        type_params = "[T, U]" if value_types == ("T", "U") else ""
        first_schema = _dict_schema(value_types[0], type_params)
        second_schema = _dict_schema(value_types[1], type_params)

        for left, right in ((first_schema, second_schema), (second_schema, first_schema)):
            scope = BoundaryScope(seals={"T": object(), "U": object()})
            value = DictValue({"item": IntValue(1)})
            encode_boundary_value(left, value, scope)
            with pytest.raises(BoundaryViolation):
                encode_boundary_value(right, value, scope)

    @pytest.mark.parametrize("generic_first", [True, False])
    def test_nested_mixed_generic_and_concrete_array_aliases_use_concrete_element_views(
        self, generic_first: bool
    ) -> None:
        generic_schema = _array_schema("array[T]", "[T]")
        concrete_schema = _array_schema("array[int]")
        value = ArrayValue([ArrayValue([IntValue(1)])])
        scope = BoundaryScope(seals={"T": object()})
        schemas = (
            (generic_schema, concrete_schema)
            if generic_first
            else (concrete_schema, generic_schema)
        )

        first = _array_view(schemas[0], value, scope)
        second = _array_view(schemas[1], value, scope)

        assert first is second
        nested = first[0]
        assert isinstance(nested, AglArrayView)
        assert list(nested) == [1]

    @pytest.mark.parametrize("int_first", [True, False])
    def test_incompatible_nested_array_alias_schemas_are_rejected_regardless_of_order(
        self, int_first: bool
    ) -> None:
        int_schema = _array_schema("array[int]")
        text_schema = _array_schema("array[text]")
        schemas = (int_schema, text_schema) if int_first else (text_schema, int_schema)
        scope = BoundaryScope()
        value = ArrayValue([ArrayValue([IntValue(1)])])

        _array_view(schemas[0], value, scope)
        with pytest.raises(BoundaryViolation):
            _array_view(schemas[1], value, scope)

    def test_equal_generic_array_schemas_keep_their_sealed_element_representation(self) -> None:
        schema = _array_schema("T", "[T]")
        value = ArrayValue([IntValue(1)])
        scope = BoundaryScope(seals={"T": object()})

        first = _array_view(schema, value, scope)
        second = _array_view(schema, value, scope)

        assert first is second
        assert isinstance(first[0], SealedHandle)

    @pytest.mark.parametrize("element_types", [("int", "text"), ("T", "U")])
    def test_incompatible_array_alias_schemas_are_rejected_regardless_of_order(
        self, element_types: tuple[str, str]
    ) -> None:
        type_params = "[T, U]" if element_types == ("T", "U") else ""
        first_schema = _array_schema(element_types[0], type_params)
        second_schema = _array_schema(element_types[1], type_params)

        for left, right in ((first_schema, second_schema), (second_schema, first_schema)):
            scope = BoundaryScope(seals={"T": object(), "U": object()})
            value = ArrayValue([IntValue(1)])
            encode_boundary_value(left, value, scope)
            with pytest.raises(BoundaryViolation):
                encode_boundary_value(right, value, scope)

    def test_array_views_are_memoized_per_scope_and_use_identity_equality(self) -> None:
        schema = _array_schema()
        value = ArrayValue([IntValue(1)])
        first_scope = BoundaryScope()
        first = _array_view(schema, value, first_scope)

        assert isinstance(first, MutableSequence)
        assert not isinstance(first, list)
        assert _array_view(schema, value, first_scope) is first
        equivalent_container = _array_view(schema, ArrayValue([IntValue(1)]), first_scope)
        assert equivalent_container is not first
        other = _array_view(schema, value, BoundaryScope())
        assert other is not first
        assert first != [1]
        assert first != other
        assert hash(first) == object.__hash__(first)
        assert len({first, other}) == 2
        assert {first: "first"}[first] == "first"

    def test_dict_views_are_memoized_per_scope_and_use_identity_equality(self) -> None:
        schema = _dict_schema()
        value = DictValue({"a": IntValue(1)})
        first_scope = BoundaryScope()
        first = _dict_view(schema, value, first_scope)

        assert isinstance(first, MutableMapping)
        assert not isinstance(first, dict)
        assert _dict_view(schema, value, first_scope) is first
        equivalent_container = _dict_view(schema, DictValue({"a": IntValue(1)}), first_scope)
        assert equivalent_container is not first
        other = _dict_view(schema, value, BoundaryScope())
        assert other is not first
        assert first != {"a": 1}
        assert first != other
        assert hash(first) == object.__hash__(first)
        assert len({first, other}) == 2
        assert {first: "first"}[first] == "first"

    def test_array_view_and_value_observe_each_others_mutations(self) -> None:
        value = ArrayValue([IntValue(1)])
        view = _array_view(_array_schema(), value, BoundaryScope())

        view.append(2)
        assert value.elements == [IntValue(1), IntValue(2)]
        value.elements[0] = IntValue(3)
        assert view[0] == 3

    def test_dict_view_and_value_observe_each_others_mutations(self) -> None:
        value = DictValue({"a": IntValue(1)})
        view = _dict_view(_dict_schema(), value, BoundaryScope())

        view["b"] = 2
        assert value.entries == {"a": IntValue(1), "b": IntValue(2)}
        value.entries["a"] = IntValue(3)
        assert view["a"] == 3

    @pytest.mark.parametrize("container", ["array", "dict"])
    @pytest.mark.parametrize("generic_first", [True, False])
    @pytest.mark.parametrize(
        ("declaration", "name"),
        [
            ("record Box[T]\n  value: T", "Box"),
            ("enum Choice[T]\n  | some(value: T)", "Choice"),
        ],
    )
    def test_nominal_generic_and_concrete_aliases_reconcile_and_preserve_provenance(
        self, declaration: str, name: str, container: str, generic_first: bool
    ) -> None:
        generic_shape = f"array[{name}[T]]" if container == "array" else f"dict[text, {name}[T]]"
        concrete_shape = (
            f"array[{name}[int]]" if container == "array" else f"dict[text, {name}[int]]"
        )
        contract = build_contract(
            f"{declaration}\n"
            f"extern def f[T](generic: {generic_shape}, concrete: {concrete_shape}) -> int\n0"
        )
        generic_schema = contract.params[0].schema
        concrete_schema = contract.params[1].schema
        nominal_schema = (
            generic_schema.element
            if isinstance(generic_schema, BoundaryArray)
            else generic_schema.value
        )
        if isinstance(nominal_schema, BoundaryRecord):
            item = RecordValue(nominal_schema.nominal, name, {"value": IntValue(1)})
        else:
            assert isinstance(nominal_schema, BoundaryEnum)
            item = EnumValue(nominal_schema.nominal, name, "some", {"value": IntValue(1)})
        value = ArrayValue([item]) if container == "array" else DictValue({"item": item})
        scope = BoundaryScope(seals={"T": object()})
        schemas = (
            (generic_schema, concrete_schema)
            if generic_first
            else (concrete_schema, generic_schema)
        )

        first = encode_boundary_value(schemas[0], value, scope)
        second = encode_boundary_value(schemas[1], value, scope)

        assert first is second
        received = first[0] if isinstance(first, AglArrayView) else first["item"]
        assert received == (
            {"value": 1} if isinstance(item, RecordValue) else {"$case": "some", "value": 1}
        )
        assert decode_boundary_value(generic_schema, first, scope) is value

    @pytest.mark.parametrize("container", ["array", "dict"])
    @pytest.mark.parametrize("generic_first", [True, False])
    def test_exception_generic_and_concrete_aliases_reconcile_and_preserve_provenance(
        self, container: str, generic_first: bool
    ) -> None:
        shape = "array[Oops]" if container == "array" else "dict[text, Oops]"
        contract = build_contract(
            f"exception Oops extends Exception\n  value: int\nextern def f(xs: {shape}) -> int\n0"
        )
        concrete_schema = contract.params[0].schema
        exception_schema = (
            concrete_schema.element
            if isinstance(concrete_schema, BoundaryArray)
            else concrete_schema.value
        )
        assert isinstance(exception_schema, BoundaryException)
        generic_exception_schema = BoundaryException(
            exception_schema.nominal,
            exception_schema.display_name,
            tuple(
                (field_name, BoundarySealVar("T") if field_name == "value" else field_schema)
                for field_name, field_schema in exception_schema.fields
            ),
        )
        generic_schema = (
            BoundaryArray(generic_exception_schema)
            if container == "array"
            else BoundaryDict(generic_exception_schema)
        )
        item = ExceptionValue(
            exception_schema.nominal,
            "Oops",
            {"message": TextValue("old"), "trace_id": TextValue(""), "value": IntValue(1)},
        )
        value = ArrayValue([item]) if container == "array" else DictValue({"item": item})
        scope = BoundaryScope(seals={"T": object()})
        schemas = (
            (generic_schema, concrete_schema)
            if generic_first
            else (concrete_schema, generic_schema)
        )

        first = encode_boundary_value(schemas[0], value, scope)
        second = encode_boundary_value(schemas[1], value, scope)

        assert first is second
        received = first[0] if isinstance(first, AglArrayView) else first["item"]
        assert received == {"message": "old", "trace_id": "", "value": 1}
        assert decode_boundary_value(generic_schema, first, scope) is value

    def test_decode_rejects_unobserved_nominal_alias_schema(self) -> None:
        contract = build_contract(
            "record Box[T]\n  value: T\n"
            "extern def f[T](generic: array[Box[T]], concrete: array[Box[int]]) -> int\n0"
        )
        generic_schema, concrete_schema = (param.schema for param in contract.params)
        assert isinstance(generic_schema, BoundaryArray)
        item = RecordValue(_nominal(generic_schema.element), "Box", {"value": IntValue(1)})

        generic_scope = BoundaryScope(seals={"T": object()})
        generic_view = _array_view(generic_schema, ArrayValue([item]), generic_scope)
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(concrete_schema, generic_view, generic_scope)

        concrete_scope = BoundaryScope(seals={"T": object()})
        concrete_view = _array_view(concrete_schema, ArrayValue([item]), concrete_scope)
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(generic_schema, concrete_view, concrete_scope)

    @pytest.mark.parametrize("concrete_first", [True, False])
    def test_conflicting_nominal_alias_fields_are_rejected_regardless_of_order(
        self, concrete_first: bool
    ) -> None:
        contract = build_contract(
            "record Box[T]\n  value: T\n"
            "extern def f(left: array[Box[int]], right: array[Box[text]]) -> int\n0"
        )
        int_schema, text_schema = (param.schema for param in contract.params)
        schemas = (int_schema, text_schema) if concrete_first else (text_schema, int_schema)
        assert isinstance(int_schema, BoundaryArray)
        item = RecordValue(_nominal(int_schema.element), "Box", {"value": IntValue(1)})
        scope = BoundaryScope()
        value = ArrayValue([item])

        encode_boundary_value(schemas[0], value, scope)
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(schemas[1], value, scope)

    @pytest.mark.parametrize("generic_first", [True, False])
    def test_registry_reconciles_recursive_generic_and_concrete_aliases(
        self, generic_first: bool
    ) -> None:
        parameters = (
            "generic: array[Node[T]], concrete: array[Node[int]]"
            if generic_first
            else "concrete: array[Node[int]], generic: array[Node[T]]"
        )
        contract = build_contract(
            "record Node[T]\n"
            "  value: T\n"
            "  children: array[Node[T]]\n"
            f"extern def f[T]({parameters}) -> array[Node[T]]\n0"
        )
        generic_schema = contract.params[0 if generic_first else 1].schema
        assert isinstance(generic_schema, BoundaryArray)
        assert isinstance(generic_schema.element, BoundaryRef)
        node_schema = dict(contract.defs)[generic_schema.element.key]
        assert isinstance(node_schema, BoundaryRecord)
        child = RecordValue(
            node_schema.nominal,
            "Node",
            {"value": IntValue(2), "children": ArrayValue([])},
        )
        node = RecordValue(
            node_schema.nominal,
            "Node",
            {"value": IntValue(1), "children": ArrayValue([child])},
        )
        value = ArrayValue([node])

        def fn(*args: object) -> object:
            assert len(args) == 2
            assert isinstance(args[0], AglArrayView)
            assert args[0] is args[1]
            received = args[0][0]
            assert isinstance(received, dict)
            assert received["value"] == 1
            children = received["children"]
            assert isinstance(children, AglArrayView)
            child = children[0]
            assert isinstance(child, dict)
            assert child["value"] == 2
            assert isinstance(child["children"], AglArrayView)
            assert list(child["children"]) == []
            return args[0]

        assert ExternRegistry().invoke("f", contract, fn, [value, value], "trace") is value

    @pytest.mark.parametrize("int_first", [True, False])
    def test_registry_rejects_incompatible_recursive_aliases(self, int_first: bool) -> None:
        parameters = (
            "left: array[Node[int]], right: array[Node[text]]"
            if int_first
            else "right: array[Node[text]], left: array[Node[int]]"
        )
        contract = build_contract(
            "record Node[T]\n"
            "  children: array[Node[T]]\n"
            "  value: T\n"
            f"extern def f({parameters}) -> int\n0"
        )
        node_schema = next(iter(dict(contract.defs).values()))
        assert isinstance(node_schema, BoundaryRecord)
        value = ArrayValue(
            [
                RecordValue(
                    node_schema.nominal,
                    "Node",
                    {"value": IntValue(1), "children": ArrayValue([])},
                )
            ]
        )

        def fn(*_args: object) -> object:
            return 0

        with pytest.raises(AglRaise):
            ExternRegistry().invoke("f", contract, fn, [value, value], "trace")

    def test_hybrid_nominal_aliases_reconcile_field_by_field(self) -> None:
        record_contract = build_contract(
            "record Pair\n  first: int\n  second: int\nextern def f(xs: array[Pair]) -> int\n0"
        )
        enum_contract = build_contract(
            "enum Choice\n  | item(first: int, second: int)\n"
            "extern def f(xs: array[Choice]) -> int\n0"
        )
        exception_contract = build_contract(
            "exception Oops extends Exception\n  first: int\n  second: int\n"
            "extern def f(xs: array[Oops]) -> int\n0"
        )
        record_schema = record_contract.params[0].schema
        enum_schema = enum_contract.params[0].schema
        exception_schema = exception_contract.params[0].schema
        assert isinstance(record_schema, BoundaryArray)
        assert isinstance(enum_schema, BoundaryArray)
        assert isinstance(exception_schema, BoundaryArray)
        assert isinstance(record_schema.element, BoundaryRecord)
        assert isinstance(enum_schema.element, BoundaryEnum)
        assert isinstance(exception_schema.element, BoundaryException)

        def hybrid_fields(
            fields: tuple[tuple[str, BoundarySchema], ...], generic: bool
        ) -> tuple[tuple[str, BoundarySchema], ...]:
            return tuple(
                (
                    name,
                    BoundarySealVar("T")
                    if generic and name == "first"
                    else BoundarySealVar("U")
                    if not generic and name == "second"
                    else BoundaryScalar(ScalarKind.TEXT)
                    if not generic and name == "first"
                    else BoundaryScalar(ScalarKind.INT),
                )
                if name in {"first", "second"}
                else (name, schema)
                for name, schema in fields
            )

        record_aliases = (
            BoundaryArray(
                BoundaryRecord(
                    record_schema.element.nominal,
                    record_schema.element.display_name,
                    hybrid_fields(record_schema.element.fields, True),
                )
            ),
            BoundaryArray(
                BoundaryRecord(
                    record_schema.element.nominal,
                    record_schema.element.display_name,
                    hybrid_fields(record_schema.element.fields, False),
                )
            ),
        )
        enum_aliases = (
            BoundaryArray(
                BoundaryEnum(
                    enum_schema.element.nominal,
                    enum_schema.element.display_name,
                    tuple(
                        BoundaryVariantShape(variant.name, hybrid_fields(variant.fields, True))
                        for variant in enum_schema.element.variants
                    ),
                )
            ),
            BoundaryArray(
                BoundaryEnum(
                    enum_schema.element.nominal,
                    enum_schema.element.display_name,
                    tuple(
                        BoundaryVariantShape(variant.name, hybrid_fields(variant.fields, False))
                        for variant in enum_schema.element.variants
                    ),
                )
            ),
        )
        exception_aliases = (
            BoundaryArray(
                BoundaryException(
                    exception_schema.element.nominal,
                    exception_schema.element.display_name,
                    hybrid_fields(exception_schema.element.fields, True),
                )
            ),
            BoundaryArray(
                BoundaryException(
                    exception_schema.element.nominal,
                    exception_schema.element.display_name,
                    hybrid_fields(exception_schema.element.fields, False),
                )
            ),
        )

        for first_schema, second_schema in (record_aliases, enum_aliases, exception_aliases):
            scope = BoundaryScope(seals={"T": object(), "U": object()})
            value = ArrayValue([])
            first = encode_boundary_value(first_schema, value, scope)
            assert first is encode_boundary_value(second_schema, value, scope)

    def test_view_schema_ref_resolution_rejects_missing_and_cyclic_chains(self) -> None:
        schema = BoundaryArray(BoundaryRef("loop"))

        with pytest.raises(BoundaryViolation):
            encode_boundary_value(schema, ArrayValue([]), BoundaryScope())
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(
                schema,
                ArrayValue([]),
                BoundaryScope(defs={"loop": BoundaryRef("loop")}),
            )

    def test_structurally_conflicting_nominal_shapes_are_rejected(self) -> None:
        record_contract = build_contract(
            "record Box\n  value: int\nextern def f(xs: array[Box]) -> int\n0"
        )
        enum_contract = build_contract(
            "enum Choice\n  | some(value: int)\nextern def f(xs: array[Choice]) -> int\n0"
        )
        exception_contract = build_contract(
            "exception Oops extends Exception\n  value: int\n"
            "extern def f(xs: array[Oops]) -> int\n0"
        )
        assert isinstance(record_contract.params[0].schema, BoundaryArray)
        assert isinstance(enum_contract.params[0].schema, BoundaryArray)
        assert isinstance(exception_contract.params[0].schema, BoundaryArray)
        record = record_contract.params[0].schema.element
        enum = enum_contract.params[0].schema.element
        exception = exception_contract.params[0].schema.element
        assert isinstance(record, BoundaryRecord)
        assert isinstance(enum, BoundaryEnum)
        assert isinstance(exception, BoundaryException)
        variant = enum.variants[0]
        text = BoundaryScalar(ScalarKind.TEXT)
        conflicts = (
            (
                BoundaryArray(record),
                BoundaryArray(BoundaryRecord(record.nominal, record.display_name, ())),
            ),
            (
                BoundaryArray(record),
                BoundaryArray(
                    BoundaryRecord(
                        record.nominal, record.display_name, (("other", record.fields[0][1]),)
                    )
                ),
            ),
            (
                BoundaryArray(enum),
                BoundaryArray(BoundaryEnum(enum.nominal, enum.display_name, ())),
            ),
            (
                BoundaryArray(enum),
                BoundaryArray(
                    BoundaryEnum(
                        enum.nominal,
                        enum.display_name,
                        (BoundaryVariantShape("other", variant.fields),),
                    )
                ),
            ),
            (
                BoundaryArray(enum),
                BoundaryArray(
                    BoundaryEnum(
                        enum.nominal,
                        enum.display_name,
                        (BoundaryVariantShape(variant.name, ()),),
                    )
                ),
            ),
            (
                BoundaryArray(enum),
                BoundaryArray(
                    BoundaryEnum(
                        enum.nominal,
                        enum.display_name,
                        (BoundaryVariantShape(variant.name, (("value", text),)),),
                    )
                ),
            ),
            (
                BoundaryArray(exception),
                BoundaryArray(
                    BoundaryException(
                        exception.nominal,
                        exception.display_name,
                        tuple(
                            (field_name, text if field_name == "value" else field_schema)
                            for field_name, field_schema in exception.fields
                        ),
                    )
                ),
            ),
        )

        for left, right in conflicts:
            for first, second in ((left, right), (right, left)):
                scope = BoundaryScope()
                value = ArrayValue([])
                encode_boundary_value(first, value, scope)
                with pytest.raises(BoundaryViolation):
                    encode_boundary_value(second, value, scope)


class TestArrayViewSurface:
    def test_index_and_slice_reads_and_writes_match_list(self) -> None:
        value = ArrayValue([IntValue(1), IntValue(2), IntValue(3), IntValue(4)])
        view = _array_view(_array_schema(), value, BoundaryScope())

        assert len(view) == 4
        assert view[0] == 1
        assert view[-1] == 4
        sliced = view[1:3]
        assert type(sliced) is list
        assert sliced == [2, 3]
        sliced.append(99)
        assert list(view) == [1, 2, 3, 4]

        view[0] = 10
        view[1:3] = [20, 30, 40]
        assert list(view) == [10, 20, 30, 40, 4]

    def test_deletion_and_insertion_operations_match_list(self) -> None:
        view = _array_view(
            _array_schema(),
            ArrayValue([IntValue(1), IntValue(2), IntValue(3), IntValue(4)]),
            BoundaryScope(),
        )

        del view[0]
        del view[1:]
        view.insert(1, 9)
        view.append(10)
        view.extend([11, 12])
        view += [13, 14]
        assert list(view) == [2, 9, 10, 11, 12, 13, 14]

        view.clear()
        assert list(view) == []

    def test_pop_remove_search_and_iteration_operations_match_list(self) -> None:
        view = _array_view(
            _array_schema(),
            ArrayValue([IntValue(1), IntValue(2), IntValue(2), IntValue(3)]),
            BoundaryScope(),
        )

        assert view.pop() == 3
        assert view.pop(0) == 1
        view.remove(2)
        assert list(view) == [2]
        assert view.index(2) == 0
        assert view.count(2) == 1
        assert 2 in view
        assert 4 not in view
        assert list(iter(view)) == [2]
        with pytest.raises(ValueError):
            view.remove(4)
        with pytest.raises(ValueError):
            view.index(4)
        assert view.pop() == 2
        with pytest.raises(IndexError):
            view.pop()

    def test_sort_supports_plain_key_reverse_and_stable_ordering(self) -> None:
        value = ArrayValue([IntValue(3), IntValue(1), IntValue(2)])
        view = _array_view(_array_schema(), value, BoundaryScope())

        view.sort()
        assert list(view) == [1, 2, 3]
        view.sort(key=lambda item: -item)
        assert list(view) == [3, 2, 1]
        view.sort(reverse=True)
        assert list(view) == [3, 2, 1]

        value.elements[:] = [IntValue(21), IntValue(11), IntValue(22), IntValue(12)]
        view.sort(key=lambda item: item % 10)
        assert list(view) == [21, 11, 22, 12]

    def test_out_of_range_index_operations_raise_index_error(self) -> None:
        view = _array_view(_array_schema(), ArrayValue([IntValue(1)]), BoundaryScope())

        with pytest.raises(IndexError):
            _ = view[4]
        with pytest.raises(IndexError):
            view[4] = 2
        with pytest.raises(IndexError):
            del view[4]

    def test_custom_index_objects_work_for_reads_writes_and_deletes(self) -> None:
        view = _array_view(
            _array_schema(), ArrayValue([IntValue(1), IntValue(2), IntValue(3)]), BoundaryScope()
        )

        assert view[_CustomIndex(-1)] == 3
        view[_CustomIndex(1)] = 20
        del view[_CustomIndex(0)]
        assert list(view) == [20, 3]


class TestArrayViewElementIdentityAndIteration:
    def test_reordering_and_moving_sealed_elements_preserves_value_objects(self) -> None:
        schema = _array_schema("T", "[T]")
        scope = BoundaryScope(seals={"T": object()})
        first, second, third, fourth = IntValue(1), IntValue(2), IntValue(3), IntValue(4)
        value = ArrayValue([first, second, third, fourth])
        view = _array_view(schema, value, scope)

        view.reverse()
        assert value.elements == [fourth, third, second, first]
        assert all(
            actual is expected
            for actual, expected in zip(value.elements, [fourth, third, second, first])
        )

        removed = view.pop()
        assert isinstance(removed, SealedHandle)
        view.append(removed)
        assert value.elements[-1] is first
        view.remove(view[1])
        del view[1:]
        view += [view[0]]
        assert len(value.elements) == 2
        assert value.elements[0] is fourth
        assert value.elements[1] is fourth

    def test_sort_reorders_existing_value_objects_without_redecoding_them(self) -> None:
        schema = _array_schema()
        three, one, two = IntValue(3), IntValue(1), IntValue(2)
        value = ArrayValue([three, one, two])
        view = _array_view(schema, value, BoundaryScope())

        view.sort()
        assert value.elements == [one, two, three]
        assert all(
            actual is expected for actual, expected in zip(value.elements, [one, two, three])
        )

    def test_extending_a_view_with_itself_preserves_value_identities(self) -> None:
        first, second = IntValue(1), IntValue(2)
        value = ArrayValue([first, second])
        view = _array_view(_array_schema(), value, BoundaryScope())

        view.extend(view)

        assert list(view) == [1, 2, 1, 2]
        assert all(
            actual is expected
            for actual, expected in zip(value.elements, [first, second, first, second])
        )

    def test_iadd_a_view_to_itself_preserves_value_identities(self) -> None:
        first, second = IntValue(1), IntValue(2)
        value = ArrayValue([first, second])
        view = _array_view(_array_schema(), value, BoundaryScope())

        view += view

        assert list(view) == [1, 2, 1, 2]
        assert all(
            actual is expected
            for actual, expected in zip(value.elements, [first, second, first, second])
        )

    def test_sorting_sealed_elements_raises_type_error(self) -> None:
        schema = _array_schema("T", "[T]")
        scope = BoundaryScope(seals={"T": object()})
        value = ArrayValue([IntValue(2), IntValue(1)])
        view = _array_view(schema, value, scope)
        before = tuple(value.elements)

        with pytest.raises(TypeError):
            view.sort()
        assert tuple(value.elements) == before
        assert all(actual is expected for actual, expected in zip(value.elements, before))

    def test_iteration_reads_not_yet_reached_positions_by_index(self) -> None:
        view = _array_view(_array_schema(), ArrayValue([IntValue(1), IntValue(2)]), BoundaryScope())
        iterator = iter(view)

        assert next(iterator) == 1
        view[1] = 3
        view.append(4)
        assert list(iterator) == [3, 4]

    def test_array_iteration_checks_liveness_on_creation_and_each_step(self) -> None:
        scope = BoundaryScope()
        view = _array_view(_array_schema(), ArrayValue([IntValue(1), IntValue(2)]), scope)
        iterator = iter(view)

        assert next(iterator) == 1
        scope.revoke()
        with pytest.raises(BoundaryViewRevoked):
            next(iterator)
        with pytest.raises(BoundaryViewRevoked):
            iter(view)

    def test_reversed_array_iteration_checks_liveness_on_creation_and_each_step(self) -> None:
        scope = BoundaryScope()
        view = _array_view(_array_schema(), ArrayValue([IntValue(1), IntValue(2)]), scope)
        assert list(reversed(view)) == [2, 1]
        iterator = reversed(view)

        assert next(iterator) == 2
        scope.revoke()
        with pytest.raises(BoundaryViewRevoked):
            next(iterator)
        with pytest.raises(BoundaryViewRevoked):
            reversed(view)


class TestDictViewSurface:
    def test_mapping_operations_match_dict(self) -> None:
        view = _dict_view(
            _dict_schema(), DictValue({"a": IntValue(1), "b": IntValue(2)}), BoundaryScope()
        )

        assert len(view) == 2
        assert view["a"] == 1
        assert "a" in view
        assert "missing" not in view
        assert view.get("a") == 1
        assert view.get("missing") is None
        assert view.get("missing", 9) == 9
        assert list(view.keys()) == ["a", "b"]
        assert list(view.values()) == [1, 2]
        assert list(view.items()) == [("a", 1), ("b", 2)]

        snapshot = dict(view)
        view["a"] = 10
        assert snapshot == {"a": 1, "b": 2}

        assert view.pop("a") == 10
        assert view.pop("missing", 9) == 9
        with pytest.raises(KeyError):
            view.pop("missing")
        view.update({"b": 20, "c": 30})
        view.update([("d", 40)])
        view.update(e=50)
        assert view.setdefault("b", 99) == 20
        assert view.setdefault("f", 60) == 60
        assert view.popitem() == ("f", 60)
        del view["c"]
        del view["d"]
        del view["e"]
        assert dict(view) == {"b": 20}
        view.clear()
        assert dict(view) == {}
        with pytest.raises(KeyError):
            view.popitem()

    def test_pop_existing_key_with_default_returns_the_value_and_removes_the_key(self) -> None:
        default = object()
        view = _dict_view(_dict_schema(), DictValue({"a": IntValue(1)}), BoundaryScope())

        result = view.pop("a", default)

        assert result == 1
        assert result is not default
        assert dict(view) == {}

    def test_iteration_uses_a_key_snapshot(self) -> None:
        view = _dict_view(
            _dict_schema(), DictValue({"a": IntValue(1), "b": IntValue(2)}), BoundaryScope()
        )
        iterator = iter(view)

        del view["b"]
        view["c"] = 3
        assert list(iterator) == ["a", "b"]

    def test_reversed_iteration_uses_a_key_snapshot_and_checks_liveness(self) -> None:
        scope = BoundaryScope()
        view = _dict_view(_dict_schema(), DictValue({"a": IntValue(1), "b": IntValue(2)}), scope)
        iterator = reversed(view)

        del view["b"]
        view["c"] = 3
        assert list(iterator) == ["b", "a"]
        assert list(reversed(view)) == ["c", "a"]
        revoked_iterator = reversed(view)
        assert next(revoked_iterator) == "c"

        scope.revoke()
        with pytest.raises(BoundaryViewRevoked):
            next(revoked_iterator)
        with pytest.raises(BoundaryViewRevoked):
            reversed(view)

    def test_dict_iteration_views_check_liveness_on_creation_and_each_step(self) -> None:
        scope = BoundaryScope()
        view = _dict_view(_dict_schema(), DictValue({"a": IntValue(1), "b": IntValue(2)}), scope)
        iterators = [iter(view), iter(view.keys()), iter(view.items()), iter(view.values())]

        assert [next(iterator) for iterator in iterators] == ["a", "a", ("a", 1), 1]
        scope.revoke()
        for iterator in iterators:
            with pytest.raises(BoundaryViewRevoked):
                next(iterator)
        with pytest.raises(BoundaryViewRevoked):
            iter(view)
        with pytest.raises(BoundaryViewRevoked):
            view.keys()
        with pytest.raises(BoundaryViewRevoked):
            view.items()
        with pytest.raises(BoundaryViewRevoked):
            view.values()

    def test_empty_dict_update_checks_liveness(self) -> None:
        scope = BoundaryScope()
        view = _dict_view(_dict_schema(), DictValue(), scope)
        scope.revoke()

        with pytest.raises(BoundaryViewRevoked):
            view.update({})

    def test_missing_and_non_string_keys_raise_the_required_errors(self) -> None:
        view = _dict_view(_dict_schema(), DictValue({"a": IntValue(1)}), BoundaryScope())

        with pytest.raises(KeyError):
            _ = view["missing"]
        with pytest.raises(KeyError):
            del view["missing"]
        with pytest.raises(TypeError):
            _ = view[1]
        with pytest.raises(TypeError):
            view[1] = 2
        with pytest.raises(TypeError):
            del view[1]


class TestViewWriteValidation:
    def test_all_value_accepting_array_mutators_reject_invalid_values(self) -> None:
        view = _array_view(_array_schema(), ArrayValue([IntValue(1)]), BoundaryScope())

        assert issubclass(BoundaryTypeError, TypeError)
        for invalid in ("s", True, 1.5):
            with pytest.raises(BoundaryTypeError):
                view[0] = invalid
        with pytest.raises(BoundaryTypeError):
            view[:] = ["s"]
        with pytest.raises(BoundaryTypeError):
            view.insert(0, "s")
        with pytest.raises(BoundaryTypeError):
            view.append("s")
        with pytest.raises(BoundaryTypeError):
            view.extend(["s"])
        with pytest.raises(BoundaryTypeError):
            view += ["s"]

    def test_all_value_accepting_dict_mutators_reject_invalid_values(self) -> None:
        view = _dict_view(_dict_schema(), DictValue({"a": IntValue(1)}), BoundaryScope())

        for invalid in ("s", True, 1.5):
            with pytest.raises(BoundaryTypeError):
                view["a"] = invalid
        with pytest.raises(BoundaryTypeError):
            view.update({"b": "s"})
        with pytest.raises(BoundaryTypeError):
            view.setdefault("b", "s")

    def test_nominal_writes_require_the_declared_shape_exactly(self) -> None:
        contract = build_contract(
            "record Box\n  value: int\n  label: text\nextern def f(xs: array[Box]) -> array[Box]\n0"
        )
        schema = contract.params[0].schema
        assert isinstance(schema, BoundaryArray)
        value = ArrayValue([])
        view = _array_view(schema, value, BoundaryScope())

        view.append({"value": 1, "label": "ok"})
        assert value.elements == [
            RecordValue(
                _nominal(schema.element), "Box", {"value": IntValue(1), "label": TextValue("ok")}
            )
        ]
        for invalid in (
            {"value": 1},
            {"value": 1, "label": "ok", "extra": 2},
            {"value": 1, "name": "ok"},
        ):
            with pytest.raises(BoundaryTypeError):
                view.append(invalid)

    def test_sealed_writes_require_this_scope_and_variable_handle(self) -> None:
        schema = _array_schema("T", "[T]")
        scope = BoundaryScope(seals={"T": object(), "U": object()})
        value = ArrayValue([IntValue(1)])
        donor_value = IntValue(2)
        view = _array_view(schema, value, scope)
        donor = _array_view(schema, ArrayValue([donor_value]), scope)
        same_scope_handle = donor[0]

        view[0] = same_scope_handle
        view[:] = [same_scope_handle]
        view.insert(0, same_scope_handle)
        view.append(same_scope_handle)
        view.extend([same_scope_handle])
        view += [same_scope_handle]
        assert all(element is donor_value for element in value.elements)

        with pytest.raises(BoundaryTypeError):
            view[0] = 3
        with pytest.raises(BoundaryTypeError):
            view[:] = [3]
        with pytest.raises(BoundaryTypeError):
            view.insert(0, 3)
        with pytest.raises(BoundaryTypeError):
            view.append(3)
        with pytest.raises(BoundaryTypeError):
            view.extend([3])
        with pytest.raises(BoundaryTypeError):
            view += [3]

        other_scope = BoundaryScope(seals={"T": object()})
        cross_scope_handle = _array_view(schema, ArrayValue([IntValue(4)]), other_scope)[0]
        with pytest.raises(BoundaryTypeError):
            view[0] = cross_scope_handle

        other_variable_schema = _array_schema("U", "[U]")
        other_variable_handle = _array_view(
            other_variable_schema, ArrayValue([IntValue(5)]), scope
        )[0]
        with pytest.raises(BoundaryTypeError):
            view[0] = other_variable_handle


class TestNestedViewValues:
    def test_nested_arrays_and_dicts_are_memoized_live_views(self) -> None:
        arrays = ArrayValue([ArrayValue([IntValue(1)])])
        array_view = _array_view(_array_schema("array[int]"), arrays, BoundaryScope())
        nested_array = array_view[0]
        assert isinstance(nested_array, AglArrayView)
        assert array_view[0] is nested_array
        nested_array.append(2)
        assert arrays.elements[0].elements == [IntValue(1), IntValue(2)]

        dictionaries = ArrayValue([DictValue({"a": IntValue(1)})])
        dict_view = _array_view(_array_schema("dict[text, int]"), dictionaries, BoundaryScope())
        nested_dict = dict_view[0]
        assert isinstance(nested_dict, AglDictView)
        assert dict_view[0] is nested_dict
        nested_dict["b"] = 2
        assert dictionaries.elements[0].entries == {"a": IntValue(1), "b": IntValue(2)}

    def test_nested_arrays_and_dicts_retrieved_from_dict_views_are_memoized_live_views(
        self,
    ) -> None:
        arrays = DictValue({"items": ArrayValue([IntValue(1)])})
        array_dict_view = _dict_view(_dict_schema("array[int]"), arrays, BoundaryScope())
        nested_array = array_dict_view["items"]
        assert isinstance(nested_array, AglArrayView)
        assert array_dict_view["items"] is nested_array
        nested_array.append(2)
        assert arrays.entries["items"] == ArrayValue([IntValue(1), IntValue(2)])

        dictionaries = DictValue({"settings": DictValue({"a": IntValue(1)})})
        dict_dict_view = _dict_view(_dict_schema("dict[text, int]"), dictionaries, BoundaryScope())
        nested_dict = dict_dict_view["settings"]
        assert isinstance(nested_dict, AglDictView)
        assert dict_dict_view["settings"] is nested_dict
        nested_dict["b"] = 2
        assert dictionaries.entries["settings"] == DictValue({"a": IntValue(1), "b": IntValue(2)})

    def test_record_fields_are_copies_but_nested_arrays_remain_live(self) -> None:
        contract = build_contract(
            "record Box\n  label: text\n  items: array[int]\n"
            "extern def f(xs: array[Box]) -> array[Box]\n0"
        )
        schema = contract.params[0].schema
        assert isinstance(schema, BoundaryArray)
        box = RecordValue(
            _nominal(schema.element),
            "Box",
            {"label": TextValue("old"), "items": ArrayValue([IntValue(1)])},
        )
        view = _array_view(schema, ArrayValue([box]), BoundaryScope())

        received = view[0]
        assert isinstance(received, dict)
        another_received = view[0]
        assert isinstance(another_received, dict)
        assert another_received is not received
        received["label"] = "changed"
        items = received["items"]
        assert isinstance(items, AglArrayView)
        repeated_received = view[0]
        assert isinstance(repeated_received, dict)
        assert repeated_received["items"] is items
        items.append(2)
        assert box.fields["label"] == TextValue("old")
        assert box.fields["items"] == ArrayValue([IntValue(1), IntValue(2)])

    def test_enum_and_exception_fields_are_copies_but_nested_arrays_remain_live(self) -> None:
        enum_contract = build_contract(
            "enum Choice\n  | some(items: array[int])\n"
            "extern def f(xs: array[Choice]) -> array[Choice]\n0"
        )
        enum_schema = enum_contract.params[0].schema
        assert isinstance(enum_schema, BoundaryArray)
        choice = EnumValue(
            _nominal(enum_schema.element), "Choice", "some", {"items": ArrayValue([IntValue(1)])}
        )
        enum_view = _array_view(enum_schema, ArrayValue([choice]), BoundaryScope())
        received_enum = enum_view[0]
        assert isinstance(received_enum, dict)
        another_enum = enum_view[0]
        assert isinstance(another_enum, dict)
        assert another_enum is not received_enum
        received_enum["$case"] = "changed"
        enum_items = received_enum["items"]
        assert isinstance(enum_items, AglArrayView)
        enum_items.append(2)
        assert choice.variant == "some"
        assert choice.fields["items"] == ArrayValue([IntValue(1), IntValue(2)])

        exception_contract = build_contract(
            "exception Oops extends Exception\n  items: array[int]\n"
            "extern def f(xs: array[Oops]) -> array[Oops]\n0"
        )
        exception_schema = exception_contract.params[0].schema
        assert isinstance(exception_schema, BoundaryArray)
        oops = ExceptionValue(
            _nominal(exception_schema.element),
            "Oops",
            {
                "message": TextValue("old"),
                "trace_id": TextValue(""),
                "items": ArrayValue([IntValue(1)]),
            },
        )
        exception_view = _array_view(exception_schema, ArrayValue([oops]), BoundaryScope())
        received_exception = exception_view[0]
        assert isinstance(received_exception, dict)
        another_exception = exception_view[0]
        assert isinstance(another_exception, dict)
        assert another_exception is not received_exception
        received_exception["message"] = "changed"
        exception_items = received_exception["items"]
        assert isinstance(exception_items, AglArrayView)
        exception_items.append(2)
        assert oops.fields["message"] == TextValue("old")
        assert oops.fields["items"] == ArrayValue([IntValue(1), IntValue(2)])

    def test_json_elements_are_copied(self) -> None:
        raw: dict[str, object] = {"items": [1]}
        value = ArrayValue([JsonValue(raw)])
        view = _array_view(_array_schema("json"), value, BoundaryScope())

        received = view[0]
        assert isinstance(received, dict)
        assert isinstance(received["items"], list)
        received["items"].append(2)
        assert raw == {"items": [1]}


class TestViewRevocationAndRepresentation:
    def test_revoked_array_view_rejects_read_iteration_and_mutation_operations(self) -> None:
        scope = BoundaryScope()
        view = _array_view(_array_schema(), ArrayValue([IntValue(1)]), scope)
        scope.revoke()

        assert scope.live is False
        with pytest.raises(BoundaryViewRevoked):
            _ = len(view)
        with pytest.raises(BoundaryViewRevoked):
            next(iter(view))
        with pytest.raises(BoundaryViewRevoked):
            view.append(2)

    def test_revoked_dict_view_rejects_read_iteration_and_mutation_operations(self) -> None:
        scope = BoundaryScope()
        view = _dict_view(_dict_schema(), DictValue({"a": IntValue(1)}), scope)
        scope.revoke()

        assert scope.live is False
        with pytest.raises(BoundaryViewRevoked):
            _ = view["a"]
        with pytest.raises(BoundaryViewRevoked):
            next(iter(view))
        with pytest.raises(BoundaryViewRevoked):
            view.update({"a": 2})

    def test_boundary_view_revoked_is_a_runtime_error(self) -> None:
        assert issubclass(BoundaryViewRevoked, RuntimeError)

    def test_repr_uses_the_value_renderer_and_surfaces_cycles(self) -> None:
        array = ArrayValue([IntValue(1)])
        assert repr(_array_view(_array_schema(), array, BoundaryScope())) == render_value(array)
        cyclic_array = ArrayValue([])
        cyclic_array.elements.append(cyclic_array)
        cyclic_array_view = _array_view(_array_schema(), cyclic_array, BoundaryScope())
        with pytest.raises(AglCyclicValue):
            repr(cyclic_array_view)

        dictionary = DictValue({"a": IntValue(1)})
        assert repr(_dict_view(_dict_schema(), dictionary, BoundaryScope())) == render_value(
            dictionary
        )
        cyclic_dict = DictValue()
        cyclic_dict.entries["self"] = cyclic_dict
        cyclic_dict_view = _dict_view(_dict_schema(), cyclic_dict, BoundaryScope())
        with pytest.raises(AglCyclicValue):
            repr(cyclic_dict_view)


class TestViewDecodeDirection:
    def test_matching_views_decode_to_the_same_underlying_containers(self) -> None:
        array_schema = _array_schema()
        dictionary_schema = _dict_schema()
        scope = BoundaryScope()
        array = ArrayValue([IntValue(1)])
        dictionary = DictValue({"a": IntValue(1)})

        assert (
            decode_boundary_value(array_schema, _array_view(array_schema, array, scope), scope)
            is array
        )
        assert (
            decode_boundary_value(
                dictionary_schema, _dict_view(dictionary_schema, dictionary, scope), scope
            )
            is dictionary
        )

    def test_plain_containers_decode_to_new_underlying_containers(self) -> None:
        scope = BoundaryScope()
        raw_array = [1]
        raw_dict = {"a": 1}
        array = decode_boundary_value(_array_schema(), raw_array, scope)
        dictionary = decode_boundary_value(_dict_schema(), raw_dict, scope)

        assert isinstance(array, ArrayValue)
        assert array is not raw_array
        assert array.elements == [IntValue(1)]
        assert isinstance(dictionary, DictValue)
        assert dictionary is not raw_dict
        assert dictionary.entries == {"a": IntValue(1)}

    def test_cross_scope_wrong_schema_and_cross_kind_views_are_boundary_violations(self) -> None:
        array_schema = _array_schema()
        dictionary_schema = _dict_schema()
        scope = BoundaryScope()
        array_view = _array_view(array_schema, ArrayValue([IntValue(1)]), scope)
        dictionary_view = _dict_view(dictionary_schema, DictValue({"a": IntValue(1)}), scope)

        with pytest.raises(BoundaryViolation):
            decode_boundary_value(array_schema, array_view, BoundaryScope())
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(_array_schema("text"), array_view, scope)
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(dictionary_schema, dictionary_view, BoundaryScope())
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(_dict_schema("text"), dictionary_view, scope)
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(array_schema, dictionary_view, scope)
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(dictionary_schema, array_view, scope)

    @pytest.mark.parametrize("generic_first", [True, False])
    def test_decode_accepts_mixed_direct_alias_views_at_their_observed_generic_schemas(
        self, generic_first: bool
    ) -> None:
        generic_array_schema = _array_schema("T", "[T]")
        concrete_array_schema = _array_schema()
        generic_dict_schema = _dict_schema("T", "[T]")
        concrete_dict_schema = _dict_schema()
        scope = BoundaryScope(seals={"T": object()})
        array = ArrayValue([IntValue(1)])
        dictionary = DictValue({"item": IntValue(1)})

        array_schemas = (
            (generic_array_schema, concrete_array_schema)
            if generic_first
            else (concrete_array_schema, generic_array_schema)
        )
        dict_schemas = (
            (generic_dict_schema, concrete_dict_schema)
            if generic_first
            else (concrete_dict_schema, generic_dict_schema)
        )
        array_view = _array_view(array_schemas[0], array, scope)
        _array_view(array_schemas[1], array, scope)
        dictionary_view = _dict_view(dict_schemas[0], dictionary, scope)
        _dict_view(dict_schemas[1], dictionary, scope)

        assert decode_boundary_value(generic_array_schema, array_view, scope) is array
        assert decode_boundary_value(generic_dict_schema, dictionary_view, scope) is dictionary
        assert list(array_view) == [1]
        assert dict(dictionary_view) == {"item": 1}

    def test_decode_rejects_unobserved_direct_generic_and_concrete_aliases(self) -> None:
        generic_array_schema = _array_schema("T", "[T]")
        concrete_array_schema = _array_schema()
        generic_dict_schema = _dict_schema("T", "[T]")
        concrete_dict_schema = _dict_schema()

        generic_scope = BoundaryScope(seals={"T": object()})
        generic_array_view = _array_view(
            generic_array_schema, ArrayValue([IntValue(1)]), generic_scope
        )
        generic_dict_view = _dict_view(
            generic_dict_schema, DictValue({"item": IntValue(1)}), generic_scope
        )
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(concrete_array_schema, generic_array_view, generic_scope)
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(concrete_dict_schema, generic_dict_view, generic_scope)

        concrete_scope = BoundaryScope(seals={"T": object()})
        concrete_array_view = _array_view(
            concrete_array_schema, ArrayValue([IntValue(1)]), concrete_scope
        )
        concrete_dict_view = _dict_view(
            concrete_dict_schema, DictValue({"item": IntValue(1)}), concrete_scope
        )
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(generic_array_schema, concrete_array_view, concrete_scope)
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(generic_dict_schema, concrete_dict_view, concrete_scope)

    @pytest.mark.parametrize("generic_first", [True, False])
    def test_decode_accepts_mixed_nested_alias_views_at_their_observed_generic_schemas(
        self, generic_first: bool
    ) -> None:
        generic_array_schema = _array_schema("array[T]", "[T]")
        concrete_array_schema = _array_schema("array[int]")
        generic_dict_schema = _dict_schema("dict[text, T]", "[T]")
        concrete_dict_schema = _dict_schema("dict[text, int]")
        scope = BoundaryScope(seals={"T": object()})
        arrays = ArrayValue([ArrayValue([IntValue(1)])])
        dictionaries = DictValue({"items": DictValue({"item": IntValue(1)})})

        array_schemas = (
            (generic_array_schema, concrete_array_schema)
            if generic_first
            else (concrete_array_schema, generic_array_schema)
        )
        dict_schemas = (
            (generic_dict_schema, concrete_dict_schema)
            if generic_first
            else (concrete_dict_schema, generic_dict_schema)
        )
        array_view = _array_view(array_schemas[0], arrays, scope)
        _array_view(array_schemas[1], arrays, scope)
        dictionary_view = _dict_view(dict_schemas[0], dictionaries, scope)
        _dict_view(dict_schemas[1], dictionaries, scope)

        assert decode_boundary_value(generic_array_schema, array_view, scope) is arrays
        assert decode_boundary_value(generic_dict_schema, dictionary_view, scope) is dictionaries
        nested_array = array_view[0]
        nested_dict = dictionary_view["items"]
        assert isinstance(nested_array, AglArrayView)
        assert isinstance(nested_dict, AglDictView)
        assert list(nested_array) == [1]
        assert dict(nested_dict) == {"item": 1}

    def test_decode_rejects_unobserved_nested_generic_and_concrete_aliases(self) -> None:
        generic_array_schema = _array_schema("array[T]", "[T]")
        concrete_array_schema = _array_schema("array[int]")
        generic_dict_schema = _dict_schema("dict[text, T]", "[T]")
        concrete_dict_schema = _dict_schema("dict[text, int]")

        generic_scope = BoundaryScope(seals={"T": object()})
        generic_array_view = _array_view(
            generic_array_schema, ArrayValue([ArrayValue([IntValue(1)])]), generic_scope
        )
        generic_dict_view = _dict_view(
            generic_dict_schema,
            DictValue({"items": DictValue({"item": IntValue(1)})}),
            generic_scope,
        )
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(concrete_array_schema, generic_array_view, generic_scope)
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(concrete_dict_schema, generic_dict_view, generic_scope)

        concrete_scope = BoundaryScope(seals={"T": object()})
        concrete_array_view = _array_view(
            concrete_array_schema, ArrayValue([ArrayValue([IntValue(1)])]), concrete_scope
        )
        concrete_dict_view = _dict_view(
            concrete_dict_schema,
            DictValue({"items": DictValue({"item": IntValue(1)})}),
            concrete_scope,
        )
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(generic_array_schema, concrete_array_view, concrete_scope)
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(generic_dict_schema, concrete_dict_view, concrete_scope)

    def test_implicit_scopes_are_fresh_boundary_calls(self) -> None:
        schema = _array_schema()
        encoded = encode_boundary_value(schema, ArrayValue([IntValue(1)]))
        assert isinstance(encoded, AglArrayView)

        with pytest.raises(BoundaryViolation):
            decode_boundary_value(schema, encoded)


class TestRegistryViewRevocation:
    @pytest.mark.parametrize("raises", [False, True])
    def test_invoke_revokes_captured_views_after_success_and_failure(self, raises: bool) -> None:
        contract = build_contract("extern def f(xs: array[int]) -> int\n0")
        captured: list[AglArrayView] = []

        def fn(*args: object) -> object:
            assert len(args) == 1
            assert isinstance(args[0], AglArrayView)
            captured.append(args[0])
            if raises:
                raise RuntimeError("boom")
            return 1

        registry = ExternRegistry()
        arguments = [ArrayValue([IntValue(1)])]
        if raises:
            with pytest.raises(AglRaise):
                registry.invoke("f", contract, fn, arguments, "trace")
        else:
            assert registry.invoke("f", contract, fn, arguments, "trace") == IntValue(1)

        assert len(captured) == 1
        with pytest.raises(BoundaryViewRevoked):
            len(captured[0])

    def test_invoke_revokes_captured_views_after_return_decode_failure(self) -> None:
        contract = build_contract("extern def f(xs: array[int]) -> int\n0")
        captured: list[AglArrayView] = []

        def fn(*args: object) -> object:
            assert len(args) == 1
            assert isinstance(args[0], AglArrayView)
            captured.append(args[0])
            return "not an int"

        with pytest.raises(AglRaise):
            ExternRegistry().invoke("f", contract, fn, [ArrayValue([IntValue(1)])], "trace")

        assert len(captured) == 1
        with pytest.raises(BoundaryViewRevoked):
            len(captured[0])
