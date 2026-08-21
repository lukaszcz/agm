"""Shared Value → JSON serialization for the AgL runtime.

This is the single source of truth for converting AgL ``Value`` objects to
JSON-shaped Python objects and for emitting them as JSON text.

Design constraint: a
``DecimalValue`` carries an exact :class:`decimal.Decimal`.  It is **never**
routed through :class:`float`.  Instead :func:`value_to_json_obj` preserves the
``Decimal`` in the JSON-shaped object, and :func:`dumps_exact` emits it as
unquoted numeric text using the ``Decimal``'s own exact string form.

Two entry points:

- :func:`value_to_json_obj` — ``Value`` → JSON-shaped object (``dict``/``list``/
  ``str``/``int``/``Decimal``/``bool``/``None``).  ``Decimal`` is preserved.
- :func:`dumps_exact` — render such an object as JSON text, emitting ``Decimal``
  as exact unquoted numeric text.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import assert_never

from agm.agl.ir.contracts import (
    ArrayEncode,
    DictEncode,
    DynamicApplyEncode,
    DynamicArrayEncode,
    DynamicDictEncode,
    DynamicEncodeDefinition,
    DynamicEncodePlan,
    DynamicEncodeSchema,
    DynamicEnumEncode,
    DynamicExceptionEncode,
    DynamicRecordEncode,
    DynamicTypeParameterEncode,
    DynamicVariantEncode,
    EncodeDefinition,
    EncodePlan,
    EncodeSchema,
    EnumEncode,
    ExceptionEncode,
    RecordEncode,
    RefEncode,
    ScalarEncode,
    TypeParameterEncode,
    VariantEncode,
    forwarded_encode_key,
    resolve_schema_ref,
)
from agm.agl.ir.ids import NominalId
from agm.agl.semantics.cycles import CYCLIC_VALUE_MARKER, AglCyclicValue, enter_container
from agm.agl.semantics.values import (
    ArrayValue,
    BoolValue,
    ConstructorValue,
    DecimalValue,
    DictValue,
    ExceptionValue,
    IntValue,
    IrClosureValue,
    IteratorValue,
    JsonValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
)


class AglNonDataValue(Exception):
    """Sentinel: a value kind with no JSON representation reached the walk.

    Raised by :func:`value_to_json_obj` for ``unit``, ``constructor``,
    ``function``, and ``iterator`` values — none of these
    kinds has a JSON-shaped representation. ``kind`` is the user-facing kind
    name (not the Python class name), used to build the substituted marker
    text in :func:`degraded_marker`.

    No evaluator path can produce a catchable AgL exception from this
    sentinel: every reachable conversion to ``json`` is statically gated
    (``is_json_convertible``, see ``semantics/type_table.py``) to types that
    have a JSON representation, so a non-data value can only ever reach this
    walk through a caller that degrades it rather than propagating it — see
    ``pipeline.py``'s ``exception_value_to_run_error``.
    """

    def __init__(self, kind: str) -> None:
        super().__init__(f"{kind} has no JSON representation")
        self.kind = kind


def degraded_marker(exc: "AglCyclicValue | AglNonDataValue") -> str:
    """Return the placeholder text substituted for a value that cannot convert.

    Maps either walk sentinel to its marker, so a caller that must not fail on
    a value it had no say in — error reporting, already unwinding a real
    error — can degrade it rather than crashing. Every other caller wants the
    sentinel and does not use this.
    """
    if isinstance(exc, AglNonDataValue):
        return f"<{exc.kind} has no JSON representation>"
    return CYCLIC_VALUE_MARKER


def encode_value(plan: EncodePlan, value: Value) -> object:
    """Encode *value* through its lowering-derived JSON plan.

    Plans select enum ``$case`` tags from the slot type rather than the runtime
    value. A finite source's definitions take no parameters; a growing
    polymorphic-recursive source's definitions are generic templates whose
    parameters each reference binds (see :class:`RefEncode`).
    """
    return _encode(plan.root, value, {d.key: d for d in plan.definitions}, (), None)


def _encode(
    schema: EncodeSchema,
    value: Value,
    definitions: dict[str, EncodeDefinition],
    arguments: tuple[EncodeSchema, ...],
    active: "set[int] | None",
) -> object:
    if isinstance(schema, TypeParameterEncode):
        return _encode(
            _bound_argument(schema.index, arguments), value, definitions, arguments, active
        )
    if isinstance(schema, RefEncode):
        definition = _resolve_encode_ref(schema.key, definitions)
        if len(schema.arguments) != definition.parameter_count:
            raise AssertionError(
                f"encode plan reference to {schema.key!r} supplies {len(schema.arguments)}"
                f" arguments for {definition.parameter_count} parameters"
            )
        # A reference's arguments are written in the CALLER's parameter space,
        # so they are substituted before they become the callee's bindings.
        bound = tuple(_substitute_arguments(argument, arguments) for argument in schema.arguments)
        return _encode(definition.body, value, definitions, bound, active)
    if isinstance(schema, ScalarEncode):
        if isinstance(value, TextValue):
            return value.value
        if isinstance(value, IntValue):
            return value.value
        if isinstance(value, DecimalValue):
            return value.value
        if isinstance(value, BoolValue):
            return value.value
        if isinstance(value, JsonValue):
            return value.raw
        raise AssertionError(f"scalar encode plan received {type(value).__name__}")
    if isinstance(schema, ArrayEncode):
        if not isinstance(value, ArrayValue):
            raise AssertionError(f"array encode plan received {type(value).__name__}")
        active = enter_container(id(value), active)
        try:
            return [
                _encode(schema.elem, item, definitions, arguments, active)
                for item in value.elements
            ]
        finally:
            active.discard(id(value))
    if isinstance(schema, DictEncode):
        if not isinstance(value, DictValue):
            raise AssertionError(f"dict encode plan received {type(value).__name__}")
        active = enter_container(id(value), active)
        try:
            return {
                name: _encode(schema.value, item, definitions, arguments, active)
                for name, item in value.entries.items()
            }
        finally:
            active.discard(id(value))
    if isinstance(schema, RecordEncode):
        if not isinstance(value, RecordValue):
            raise AssertionError(f"record encode plan received {type(value).__name__}")
        if value.nominal != schema.nominal:
            raise AssertionError(
                f"record encode plan received {value.nominal!r}, expected {schema.nominal!r}"
            )
        return {
            name: _encode(field, value.fields[name], definitions, arguments, active)
            for name, field in schema.fields
        }
    if isinstance(schema, ExceptionEncode):
        if not isinstance(value, ExceptionValue):
            raise AssertionError(f"exception encode plan received {type(value).__name__}")
        if value.nominal != schema.nominal:
            raise AssertionError(
                f"exception encode plan received {value.nominal!r}, expected {schema.nominal!r}"
            )
        return {
            name: _encode(field, value.fields[name], definitions, arguments, active)
            for name, field in schema.fields
        }
    if isinstance(schema, EnumEncode):
        variant, fields = _variant_for_encode(schema, value)
        result: dict[str, object] = {"$case": variant.name}
        result.update(
            {
                name: _encode(field, fields[name], definitions, arguments, active)
                for name, field in variant.fields
            }
        )
        return result
    raise AssertionError(f"unknown encode schema {schema!r}")  # pragma: no cover


def _variant_for_encode(schema: EnumEncode, value: Value) -> tuple[VariantEncode, dict[str, Value]]:
    """Select the member record carried by an enum-typed static slot."""
    if not isinstance(value, RecordValue):
        raise AssertionError(f"enum encode plan received {type(value).__name__}")
    for variant in schema.variants:
        if variant.nominal == value.nominal:
            return variant, value.fields
    raise AssertionError(f"enum encode plan has no member {value.nominal!r}")


def encode_dynamic_value(plan: DynamicEncodePlan, value: Value) -> object:
    """Encode through a generic-template plan without runtime tag inference."""
    definitions = {definition.nominal: definition for definition in plan.definitions}
    return _encode_dynamic(plan.root, value, definitions, (), None)


def _encode_dynamic(
    schema: DynamicEncodeSchema,
    value: Value,
    definitions: dict[NominalId, DynamicEncodeDefinition],
    arguments: tuple[DynamicEncodeSchema, ...],
    active: set[int] | None,
) -> object:
    if isinstance(schema, DynamicTypeParameterEncode):
        index = schema.index
        try:
            resolved = arguments[index]
        except IndexError as exc:
            raise AssertionError(f"dynamic encode parameter {index} is unbound") from exc
        return _encode_dynamic(resolved, value, definitions, arguments, active)
    if isinstance(schema, DynamicApplyEncode):
        definition = definitions.get(schema.nominal)
        if definition is None:
            raise AssertionError(f"dynamic encode plan has no definition for {schema.nominal!r}")
        if len(schema.arguments) != definition.parameter_count:
            raise AssertionError(
                f"dynamic encode application has wrong arity for {schema.nominal!r}"
            )
        bound_arguments = tuple(
            _instantiate_dynamic(argument, arguments) for argument in schema.arguments
        )
        return _encode_dynamic(definition.body, value, definitions, bound_arguments, active)
    if isinstance(schema, ScalarEncode):
        return _encode(schema, value, {}, (), active)
    if isinstance(schema, DynamicArrayEncode):
        if not isinstance(value, ArrayValue):
            raise AssertionError(f"array encode plan received {type(value).__name__}")
        active = enter_container(id(value), active)
        try:
            return [
                _encode_dynamic(schema.elem, item, definitions, arguments, active)
                for item in value.elements
            ]
        finally:
            active.discard(id(value))
    if isinstance(schema, DynamicDictEncode):
        if not isinstance(value, DictValue):
            raise AssertionError(f"dict encode plan received {type(value).__name__}")
        active = enter_container(id(value), active)
        try:
            return {
                name: _encode_dynamic(schema.value, item, definitions, arguments, active)
                for name, item in value.entries.items()
            }
        finally:
            active.discard(id(value))
    if isinstance(schema, DynamicRecordEncode):
        if not isinstance(value, RecordValue) or value.nominal != schema.nominal:
            raise AssertionError("dynamic record encode plan received the wrong nominal value")
        return {
            name: _encode_dynamic(field, value.fields[name], definitions, arguments, active)
            for name, field in schema.fields
        }
    if isinstance(schema, DynamicExceptionEncode):
        if not isinstance(value, ExceptionValue) or value.nominal != schema.nominal:
            raise AssertionError("dynamic exception encode plan received the wrong nominal value")
        return {
            name: _encode_dynamic(field, value.fields[name], definitions, arguments, active)
            for name, field in schema.fields
        }
    if isinstance(schema, DynamicEnumEncode):
        variant, fields = _dynamic_variant_for_encode(schema, value)
        result: dict[str, object] = {"$case": variant.name}
        result.update(
            {
                name: _encode_dynamic(field, fields[name], definitions, arguments, active)
                for name, field in variant.fields
            }
        )
        return result
    raise AssertionError(f"unknown dynamic encode schema {schema!r}")  # pragma: no cover


def _dynamic_variant_for_encode(
    schema: DynamicEnumEncode, value: Value
) -> tuple[DynamicVariantEncode, dict[str, Value]]:
    if not isinstance(value, RecordValue):
        raise AssertionError(f"enum encode plan received {type(value).__name__}")
    for variant in schema.variants:
        if variant.nominal == value.nominal:
            return variant, value.fields
    raise AssertionError(f"enum encode plan has no member {value.nominal!r}")


def _instantiate_dynamic(
    schema: DynamicEncodeSchema, arguments: tuple[DynamicEncodeSchema, ...]
) -> DynamicEncodeSchema:
    """Substitute a caller's type parameters before entering an application."""
    if isinstance(schema, DynamicTypeParameterEncode):
        try:
            return arguments[schema.index]
        except IndexError as exc:
            raise AssertionError(f"dynamic encode parameter {schema.index} is unbound") from exc
    if isinstance(schema, DynamicApplyEncode):
        return DynamicApplyEncode(
            schema.nominal,
            tuple(_instantiate_dynamic(argument, arguments) for argument in schema.arguments),
        )
    if isinstance(schema, DynamicArrayEncode):
        return DynamicArrayEncode(_instantiate_dynamic(schema.elem, arguments))
    if isinstance(schema, DynamicDictEncode):
        return DynamicDictEncode(_instantiate_dynamic(schema.value, arguments))
    if isinstance(schema, DynamicRecordEncode):
        return DynamicRecordEncode(
            schema.nominal,
            tuple((name, _instantiate_dynamic(field, arguments)) for name, field in schema.fields),
        )
    if isinstance(schema, DynamicExceptionEncode):
        return DynamicExceptionEncode(
            schema.nominal,
            tuple((name, _instantiate_dynamic(field, arguments)) for name, field in schema.fields),
        )
    if isinstance(schema, DynamicEnumEncode):
        return DynamicEnumEncode(
            schema.nominal,
            tuple(
                DynamicVariantEncode(
                    variant.name,
                    variant.nominal,
                    tuple(
                        (name, _instantiate_dynamic(field, arguments))
                        for name, field in variant.fields
                    ),
                )
                for variant in schema.variants
            ),
        )
    return schema


def _bound_argument(index: int, arguments: tuple[EncodeSchema, ...]) -> EncodeSchema:
    """Read one enclosing definition parameter out of the current bindings."""
    try:
        return arguments[index]
    except IndexError as exc:
        raise AssertionError(f"encode plan parameter {index} is unbound") from exc


def _substitute_arguments(
    schema: EncodeSchema, arguments: tuple[EncodeSchema, ...]
) -> EncodeSchema:
    """Replace every parameter in *schema* with its binding from *arguments*.

    Applied to a reference's arguments on the way into a definition: they are
    written in terms of the enclosing definition's parameters, which must be
    resolved before they can bind the callee's own.
    """
    match schema:
        case TypeParameterEncode(index=index):
            return _bound_argument(index, arguments)
        case ScalarEncode():
            return schema
        case RefEncode(key=key, arguments=reference_arguments):
            return RefEncode(
                key,
                tuple(
                    _substitute_arguments(argument, arguments) for argument in reference_arguments
                ),
            )
        case ArrayEncode(elem=elem):
            return ArrayEncode(_substitute_arguments(elem, arguments))
        case DictEncode(value=value_schema):
            return DictEncode(_substitute_arguments(value_schema, arguments))
        case RecordEncode(nominal=nominal, fields=fields):
            return RecordEncode(nominal, _substitute_fields(fields, arguments))
        case ExceptionEncode(nominal=nominal, fields=fields):
            return ExceptionEncode(nominal, _substitute_fields(fields, arguments))
        case EnumEncode(nominal=nominal, variants=variants):
            return EnumEncode(
                nominal,
                tuple(
                    VariantEncode(
                        variant.name,
                        variant.nominal,
                        _substitute_fields(variant.fields, arguments),
                    )
                    for variant in variants
                ),
            )
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _substitute_fields(
    fields: "tuple[tuple[str, EncodeSchema], ...]", arguments: tuple[EncodeSchema, ...]
) -> "tuple[tuple[str, EncodeSchema], ...]":
    """Substitute *arguments* through one ordered field or variant-field list."""
    return tuple((name, _substitute_arguments(field, arguments)) for name, field in fields)


def _resolve_encode_ref(key: str, definitions: dict[str, EncodeDefinition]) -> EncodeDefinition:
    """Resolve a plan reference to the definition whose body is not itself a reference."""
    return resolve_schema_ref(
        key,
        definitions,
        forwarded_encode_key,
        subject="encode plan",
    )


def value_to_json_obj(value: Value, active: "set[int] | None" = None) -> object:
    """Convert a ``Value`` to a JSON-shaped Python object.

    The result is drawn from the closed JSON-shape domain
    ``dict | list | str | int | Decimal | bool | None``.  ``DecimalValue`` is
    preserved as :class:`decimal.Decimal` (never converted to ``float``).

    Reference semantics makes a cyclic array/dict constructible; ``active``
    (an active-container-id set, allocated lazily on first use) detects a
    cycle and raises :class:`~agm.agl.semantics.cycles.AglCyclicValue` rather
    than recursing forever. Callers pass no *active* argument — it exists
    only to thread the walk's own recursive calls.

    A ``unit``, ``constructor``, ``function``, or ``iterator``
    value has no JSON representation at all; such a value raises
    :class:`AglNonDataValue` rather than a bare :class:`TypeError`, so a
    caller that can legitimately receive one (e.g. because it carries the
    field of an in-flight exception) can degrade it to a marker instead of
    crashing. This untyped walk deliberately emits no enum tags. Lowered casts
    use :func:`encode_value` or :func:`encode_dynamic_value`, both of which
    retain source-slot context; this fallback serves reporting values that have
    no associated static slot plan.
    """
    if isinstance(value, TextValue):
        return value.value
    if isinstance(value, IntValue):
        return value.value
    if isinstance(value, DecimalValue):
        return value.value
    if isinstance(value, BoolValue):
        return value.value
    if isinstance(value, JsonValue):
        return value.raw
    if isinstance(value, ArrayValue):
        active = enter_container(id(value), active)
        try:
            return [value_to_json_obj(e, active) for e in value.elements]
        finally:
            active.discard(id(value))
    if isinstance(value, DictValue):
        active = enter_container(id(value), active)
        try:
            return {k: value_to_json_obj(v, active) for k, v in value.entries.items()}
        finally:
            active.discard(id(value))
    if isinstance(value, RecordValue):
        return {k: value_to_json_obj(v, active) for k, v in value.fields.items()}
    if isinstance(value, ExceptionValue):
        return {k: value_to_json_obj(v, active) for k, v in value.fields.items()}
    if isinstance(value, UnitValue):
        raise AglNonDataValue("unit")
    if isinstance(value, ConstructorValue):
        raise AglNonDataValue("constructor")
    if isinstance(value, IrClosureValue):
        raise AglNonDataValue("function")
    if isinstance(value, IteratorValue):
        raise AglNonDataValue("iterator")
    assert_never(value)  # pragma: no cover


def dumps_exact(obj: object, *, indent: int | None = 2) -> str:
    """Serialize a JSON-shaped object to text, emitting decimals exactly.

    Operates over the closed JSON-shape domain produced by
    :func:`value_to_json_obj` (``dict``/``list``/``str``/``int``/``Decimal``/
    ``bool``/``None``).  A :class:`decimal.Decimal` is emitted as unquoted
    numeric text using its exact string form — it is never routed through
    :class:`float`.

    A small recursive emitter is used (rather than ``json.dumps``) because the
    stdlib encoder cannot serialize ``Decimal`` without a binary-float round
    trip.  ``str``/``bool``/``int``/``None`` leaves are still delegated to
    ``json.dumps`` so that escaping and formatting match the stdlib exactly.
    """
    return _emit(obj, indent=indent, level=0)


def _emit(obj: object, *, indent: int | None, level: int) -> str:
    # ``bool`` must be checked before ``int`` (bool is a subclass of int).
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, Decimal):
        return _decimal_text(obj)
    if isinstance(obj, (str, int)) or obj is None:
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, list):
        return _emit_array(obj, indent=indent, level=level)
    if isinstance(obj, dict):
        return _emit_dict(obj, indent=indent, level=level)
    # Defensive: anything outside the closed domain is rendered via json.dumps,
    # which raises a clear TypeError for genuinely unsupported objects.
    return json.dumps(obj, ensure_ascii=False)  # pragma: no cover


def _decimal_text(d: Decimal) -> str:
    """Exact unquoted numeric text for a ``Decimal`` (no float round trip)."""
    # ``str`` preserves the Decimal's exact value but can use scientific
    # notation (e.g. ``1E+2``); ``format(d, "f")`` forces plain fixed-point
    # while remaining exact.
    return format(d, "f")


def _emit_array(obj: list[object], *, indent: int | None, level: int) -> str:
    if not obj:
        return "[]"
    if indent is None:
        items = [_emit(e, indent=None, level=level) for e in obj]
        return "[" + ", ".join(items) + "]"
    pad = " " * (indent * (level + 1))
    close_pad = " " * (indent * level)
    items = [pad + _emit(e, indent=indent, level=level + 1) for e in obj]
    return "[\n" + ",\n".join(items) + "\n" + close_pad + "]"


def _emit_dict(obj: dict[object, object], *, indent: int | None, level: int) -> str:
    if not obj:
        return "{}"
    keys = [json.dumps(str(k), ensure_ascii=False) for k in obj]
    values = list(obj.values())
    if indent is None:
        items = [
            f"{k}: {_emit(v, indent=None, level=level)}" for k, v in zip(keys, values, strict=True)
        ]
        return "{" + ", ".join(items) + "}"
    pad = " " * (indent * (level + 1))
    close_pad = " " * (indent * level)
    items = [
        f"{pad}{k}: {_emit(v, indent=indent, level=level + 1)}"
        for k, v in zip(keys, values, strict=True)
    ]
    return "{\n" + ",\n".join(items) + "\n" + close_pad + "}"
