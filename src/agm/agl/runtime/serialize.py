"""Shared Value → JSON serialization for the AgL runtime.

This is the single source of truth for converting AgL ``Value`` objects to
JSON-shaped Python objects and for emitting them as JSON text.

Design constraint: a
``DecimalValue`` carries an exact :class:`decimal.Decimal`.  It is **never**
routed through :class:`float`.  Instead :func:`value_to_json_obj` preserves the
``Decimal`` in the JSON-shaped object, and :func:`dumps_exact` emits it as
unquoted numeric text using the ``Decimal``'s own exact string form.

Entry points:

- :func:`value_to_json_obj` — ``Value`` → JSON-shaped object (``dict``/``list``/
  ``str``/``int``/``Decimal``/``bool``/``None``).  ``Decimal`` is preserved.
- :func:`encode_value` — ``Value`` → JSON-shaped object through a lowering-derived
  :class:`~agm.agl.ir.contracts.EncodePlan` (an ``as json`` cast).
- :func:`encode_scalar` — one scalar or ``json`` value (an implicit ``json`` coercion).
- :func:`report_exception_fields` — an uncaught exception's own fields, selected
  by its runtime nominal like ``encode_value``'s, degrading unconvertible fields
  to markers for the error report.
- :func:`value_to_trace_json_obj` — best-effort trace data, retaining enum tags
  and degrading cycles and non-data values.
- :func:`dumps_exact` — render such an object as JSON text, emitting ``Decimal``
  as exact unquoted numeric text.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from math import isfinite
from typing import TYPE_CHECKING, assert_never, cast

from agm.agl.ir.contracts import (
    ArrayEncode,
    DictEncode,
    DictKeyForm,
    EncodeDefinition,
    EncodePlan,
    EncodeSchema,
    EnumEncode,
    ExceptionEncode,
    ExceptionFieldEncode,
    FieldEncode,
    RecordEncode,
    RefEncode,
    ScalarEncode,
    TypeParameterEncode,
    VariantEncode,
    dict_key_form,
    forwarded_encode_key,
    is_plain_enum,
    resolve_schema_ref,
)
from agm.agl.ir.ids import NominalId
from agm.agl.ir.program import NominalDescriptor
from agm.agl.semantics.cycles import (
    CYCLIC_VALUE_MARKER,
    AglCyclicValue,
    enter_value,
    non_data_marker,
)
from agm.agl.semantics.values import (
    ArrayValue,
    BoolValue,
    ConstructorValue,
    ContractValue,
    DecimalValue,
    DictValue,
    ExceptionValue,
    IntValue,
    IrClosureValue,
    IteratorValue,
    JsonValue,
    ObservableValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
)
from agm.util.decimal import strip_trailing_zeros
from agm.util.unicode import surrogate_index

if TYPE_CHECKING:
    from agm.agl.ir.builtin_nominals import BuiltinNominals
    from agm.agl.ir.program import ValueDescriptors

#: The closed JSON-shape domain ``dumps_exact``/``_emit`` serialize: every
#: scalar a JSON document may hold, plus recursively JSON-shaped
#: ``list``/``dict``.
type JsonShaped = (
    None | bool | int | float | Decimal | str | list[JsonShaped] | dict[str, JsonShaped]
)


class AglNonDataValue(Exception):
    """Sentinel: a value with no JSON representation reached a JSON walk.

    Raised by :func:`value_to_json_obj` for ``unit``, ``constructor``,
    ``function``, and ``contract`` values, and by
    :func:`encode_value` for an exception field with no JSON form. ``kind``
    is the user-facing name of what failed (not a Python class name), used to
    build the marker text in :func:`degraded_marker`.

    A cast reports it as a failed conversion (``eval.ir_interpreter``); only
    uncaught-error reporting (:func:`report_exception_fields`) degrades it to
    a marker.
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
        return non_data_marker(exc.kind)
    return CYCLIC_VALUE_MARKER


def encode_value(
    plan: EncodePlan,
    value: Value,
    exception_field_encodes: "Mapping[NominalId, tuple[ExceptionFieldEncode, ...]]",
) -> object:
    """Encode *value* through its lowering-derived JSON plan.

    Plans select an enum's shape (a plain enum's tag string, else a ``$case``
    object) and its tags from the slot type rather than the runtime value. A
    finite source's definitions take no parameters; a growing
    polymorphic-recursive source's definitions are generic templates whose
    parameters each reference binds (see :class:`RefEncode`).

    *exception_field_encodes* is the program-wide, nominal-keyed table of
    every exception's own field-encode plans (``ExecutableProgram
    .exception_field_encodes``). An exception reached anywhere in *plan*
    encodes through this table by its value's own runtime nominal (see
    :class:`~agm.agl.ir.contracts.ExceptionEncode`).

    Raises :class:`~agm.agl.semantics.cycles.AglCyclicValue` on a cyclic
    value and :class:`AglNonDataValue` on an exception field with no JSON
    form (a runtime subtype's field no static cast site examined).
    """
    return _encode(plan.root, value, plan.definitions_by_key, (), None, exception_field_encodes)


def encode_scalar(value: Value) -> JsonShaped:
    """Encode one scalar or ``json`` value (a :class:`ScalarEncode` slot)."""
    if isinstance(value, JsonValue):
        return cast(JsonShaped, value.raw)
    return cast("TextValue | IntValue | DecimalValue | BoolValue", value).value


#: An :data:`EncodeSchema` resolved past ``TypeParameterEncode``/``RefEncode``
#: indirection — see :func:`_resolve`.
type _ResolvedEncodeSchema = (
    ScalarEncode | ArrayEncode | DictEncode | RecordEncode | ExceptionEncode | EnumEncode
)


def _resolve(
    schema: EncodeSchema,
    definitions: dict[str, EncodeDefinition],
    arguments: tuple[EncodeSchema, ...],
) -> tuple[_ResolvedEncodeSchema, tuple[EncodeSchema, ...]]:
    """Follow ``TypeParameterEncode``/``RefEncode`` indirection to a schema's own concrete shape.

    Returns the resolved schema alongside the argument bindings in effect at
    that point — the context a ``TypeParameterEncode`` nested inside its own
    children resolves against. Shared by :func:`_encode`'s own dispatch and
    by dict-key resolution (:func:`_encode_dict`): a key's own encode schema
    may itself be a ``TypeParameterEncode`` (a growing generic template's
    dict field keyed by its own type parameter) or a ``RefEncode`` (a
    recursive or shared instantiation); its :class:`DictKeyForm` depends only
    on the resolved concrete shape.
    """
    while True:
        match schema:
            case TypeParameterEncode(index=index):
                schema = _bound_argument(index, arguments)
            case RefEncode(key=key, arguments=ref_args):
                definition = _resolve_encode_ref(key, definitions)
                # A reference's arguments are written in the CALLER's parameter
                # space, so they are substituted before they become the callee's
                # bindings.
                arguments = tuple(
                    _substitute_arguments(argument, arguments) for argument in ref_args
                )
                schema = definition.body
            case _:
                return schema, arguments


def _encode(
    schema: EncodeSchema,
    value: Value,
    definitions: dict[str, EncodeDefinition],
    arguments: tuple[EncodeSchema, ...],
    active: "set[int] | None",
    exception_field_encodes: "Mapping[NominalId, tuple[ExceptionFieldEncode, ...]]",
) -> object:
    schema, arguments = _resolve(schema, definitions, arguments)
    return _encode_resolved(schema, value, definitions, arguments, active, exception_field_encodes)


def _encode_resolved(
    schema: "_ResolvedEncodeSchema",
    value: Value,
    definitions: dict[str, EncodeDefinition],
    arguments: tuple[EncodeSchema, ...],
    active: "set[int] | None",
    exception_field_encodes: "Mapping[NominalId, tuple[ExceptionFieldEncode, ...]]",
) -> object:
    """Encode *value* through an already-resolved schema head (see :func:`_resolve`).

    Split out of :func:`_encode` so a caller that resolved a schema once for
    several values — :func:`_encode_dict`, once per dict for its key schema —
    reuses that resolution instead of re-resolving it per value.
    """
    match schema:
        case ScalarEncode():
            return encode_scalar(value)
        case ArrayEncode(elem=elem):
            value = cast(ArrayValue, value)
            active = enter_value(id(value), active)
            try:
                return [
                    _encode(elem, item, definitions, arguments, active, exception_field_encodes)
                    for item in value.elements
                ]
            finally:
                active.discard(id(value))
        case DictEncode(key=key_schema, value=value_schema):
            value = cast(DictValue, value)
            active = enter_value(id(value), active)
            try:
                return _encode_dict(
                    key_schema,
                    value_schema,
                    value,
                    definitions,
                    arguments,
                    active,
                    exception_field_encodes,
                )
            finally:
                active.discard(id(value))
        case RecordEncode(fields=fields):
            value = cast(RecordValue, value)
            active = enter_value(id(value), active)
            try:
                return {
                    fenc.json_name: _encode(
                        fenc.schema,
                        value.fields[fenc.name],
                        definitions,
                        arguments,
                        active,
                        exception_field_encodes,
                    )
                    for fenc in fields
                }
            finally:
                active.discard(id(value))
        case ExceptionEncode():
            value = cast(ExceptionValue, value)
            active = enter_value(id(value), active)
            try:
                return {
                    field_encode.json_name: _encode_exception_field(
                        field_encode,
                        value.fields[field_encode.field_name],
                        active,
                        exception_field_encodes,
                    )
                    for field_encode in exception_field_encodes[value.nominal]
                }
            finally:
                active.discard(id(value))
        case EnumEncode():
            value = cast(RecordValue, value)
            variant = _variant_for_encode(schema, value)
            if is_plain_enum(schema):
                return variant.json_name
            active = enter_value(id(value), active)
            try:
                result: dict[str, object] = {"$case": variant.json_name}
                result.update(
                    {
                        fenc.json_name: _encode(
                            fenc.schema,
                            value.fields[fenc.name],
                            definitions,
                            arguments,
                            active,
                            exception_field_encodes,
                        )
                        for fenc in variant.fields
                    }
                )
                return result
            finally:
                active.discard(id(value))
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _encode_dict(
    key_schema: EncodeSchema,
    value_schema: EncodeSchema,
    value: DictValue,
    definitions: dict[str, EncodeDefinition],
    arguments: tuple[EncodeSchema, ...],
    active: "set[int] | None",
    exception_field_encodes: "Mapping[NominalId, tuple[ExceptionFieldEncode, ...]]",
) -> object:
    """Encode a dict per its resolved key's ``DictKeyForm`` (see ``ir.contracts.DictKeyForm``).

    The key schema is resolved, and classified, once per dict.
    """
    resolved_key_schema, resolved_key_arguments = _resolve(key_schema, definitions, arguments)
    form = dict_key_form(resolved_key_schema)
    if form is DictKeyForm.OBJECT_TEXT:
        return {
            key: _encode(
                value_schema, item, definitions, arguments, active, exception_field_encodes
            )
            for key, item in value.text_items()
        }
    if form is DictKeyForm.OBJECT_STRINGIFIED:
        return {
            stringified_key_text(
                _encode_resolved(
                    resolved_key_schema,
                    key_value,
                    definitions,
                    resolved_key_arguments,
                    active,
                    exception_field_encodes,
                )
            ): _encode(value_schema, item, definitions, arguments, active, exception_field_encodes)
            for key_value, item in value.items()
        }
    return [
        {
            "key": _encode_resolved(
                resolved_key_schema,
                key_value,
                definitions,
                resolved_key_arguments,
                active,
                exception_field_encodes,
            ),
            "value": _encode(
                value_schema, item, definitions, arguments, active, exception_field_encodes
            ),
        }
        for key_value, item in value.items()
    ]


def _encode_exception_field(
    field_encode: ExceptionFieldEncode,
    value: Value,
    active: "set[int] | None",
    exception_field_encodes: "Mapping[NominalId, tuple[ExceptionFieldEncode, ...]]",
) -> object:
    """Encode one exception field through its own self-contained plan.

    The field's plan was compiled independently from its declared type, so it
    is unrelated to any enclosing plan's ``definitions``/``arguments``. A
    field with no plan has no JSON form and raises :class:`AglNonDataValue`.
    """
    if field_encode.plan is None:
        raise AglNonDataValue(f"field '{field_encode.field_name}'")
    return _encode(
        field_encode.plan.root,
        value,
        field_encode.plan.definitions_by_key,
        (),
        active,
        exception_field_encodes,
    )


@dataclass(frozen=True, slots=True)
class WalkTags:
    """Per-nominal JSON metadata for :func:`value_to_json_obj`'s untyped walk.

    Built once (by :func:`_walk_tags`) from the program's whole nominal
    descriptor table, so a bare ``Value`` walk — which, unlike
    :func:`encode_value`, carries no encode-schema head to read either from —
    names an enum member's ``$case`` and a nested record/exception field's
    JSON key the SAME way a typed encoding would, without re-deriving either
    per occurrence.

    ``member_tags`` maps an enum member's own ``NominalId`` to its variant's
    effective JSON tag (``VariantDescriptor.json_name``). ``field_names``
    maps a record/exception nominal to its declared-name -> effective
    JSON-name field map (``NominalDescriptor.fields``/``.field_json_names``).
    """

    member_tags: "Mapping[NominalId, str]"
    field_names: "Mapping[NominalId, Mapping[str, str]]"


def _walk_tags(nominals: "Mapping[NominalId, NominalDescriptor]") -> WalkTags:
    """Build :class:`WalkTags` once from the program's whole nominal descriptor table."""
    return WalkTags(
        member_tags={
            variant.member: variant.json_name
            for descriptor in nominals.values()
            for variant in descriptor.variants
        },
        field_names={
            descriptor.nominal: dict(
                zip(descriptor.fields, descriptor.field_json_names, strict=True)
            )
            for descriptor in nominals.values()
        },
    )


def report_exception_fields(
    value: ExceptionValue,
    exception_field_encodes: "Mapping[NominalId, tuple[ExceptionFieldEncode, ...]]",
    nominals: "Mapping[NominalId, NominalDescriptor]",
) -> dict[str, object]:
    """Encode an uncaught exception's fields for its error report, by its runtime nominal.

    Reporting-only: this runs while an error is already in flight, so a field
    that cannot convert degrades to a :func:`degraded_marker` rather than
    raising. A field with no JSON form is reported through the untyped
    :func:`value_to_json_obj` walk, which names an enum member's ``$case``
    and a nested record/exception field's JSON key from *nominals* (via
    :func:`_walk_tags`) since it has no compiled encode schema to read either
    from. Casts go through :func:`encode_value`, which raises instead.
    """
    tags: WalkTags | None = None
    fields: dict[str, object] = {}
    for field_encode in exception_field_encodes[value.nominal]:
        field_value = value.fields[field_encode.field_name]
        try:
            if field_encode.plan is None:
                tags = tags or _walk_tags(nominals)
                fields[field_encode.json_name] = value_to_json_obj(field_value, tags=tags)
            else:
                fields[field_encode.json_name] = _encode_exception_field(
                    field_encode, field_value, None, exception_field_encodes
                )
        except (AglCyclicValue, AglNonDataValue) as field_exc:
            fields[field_encode.json_name] = degraded_marker(field_exc)
    return fields


def _variant_for_encode(schema: EnumEncode, value: RecordValue) -> VariantEncode:
    """Select the member record carried by an enum-typed static slot."""
    return next(v for v in schema.variants if v.nominal == value.nominal)


def _bound_argument(index: int, arguments: tuple[EncodeSchema, ...]) -> EncodeSchema:
    """Read one enclosing definition parameter out of the current bindings."""
    return arguments[index]


def stringified_key_text(key: object) -> str:
    """Return the object-key text of an encoded stringified key.

    An enum key is already its tag; a scalar key is its JSON scalar text,
    exactly as ``as json`` writes it.
    """
    if isinstance(key, str):
        return key
    if isinstance(key, bool):
        return "true" if key else "false"
    if isinstance(key, int):
        return str(key)
    return dumps_exact(cast(JsonShaped, key), indent=None)


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
        case DictEncode(key=key_schema, value=value_schema):
            return DictEncode(
                _substitute_arguments(key_schema, arguments),
                _substitute_arguments(value_schema, arguments),
            )
        case RecordEncode(nominal=nominal, fields=fields):
            return RecordEncode(nominal, _substitute_fields(fields, arguments))
        case ExceptionEncode():
            return schema
        case EnumEncode(nominal=nominal, variants=variants):
            return EnumEncode(
                nominal,
                tuple(
                    VariantEncode(
                        variant.name,
                        variant.json_name,
                        variant.nominal,
                        _substitute_fields(variant.fields, arguments),
                    )
                    for variant in variants
                ),
            )
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _substitute_fields(
    fields: "tuple[FieldEncode, ...]", arguments: tuple[EncodeSchema, ...]
) -> "tuple[FieldEncode, ...]":
    """Substitute *arguments* through one ordered field or variant-field list."""
    return tuple(
        FieldEncode(fenc.name, fenc.json_name, _substitute_arguments(fenc.schema, arguments))
        for fenc in fields
    )


def _resolve_encode_ref(key: str, definitions: dict[str, EncodeDefinition]) -> EncodeDefinition:
    """Resolve a plan reference to the definition whose body is not itself a reference."""
    return resolve_schema_ref(key, definitions, forwarded_encode_key)


def _walk_dict[T](
    value: DictValue, walk: Callable[[Value], T]
) -> dict[str, T] | list[dict[str, T]]:
    """Walk a dict untyped: an object when text-keyed or empty, else key/value entries."""
    if value.is_text_keyed() or len(value) == 0:
        return {key: walk(item) for key, item in value.text_items()}
    return [{"key": walk(key), "value": walk(item)} for key, item in value.items()]


def value_to_json_obj(
    value: Value,
    active: "set[int] | None" = None,
    *,
    tags: "WalkTags",
) -> JsonShaped:
    """Convert a ``Value`` to a JSON-shaped Python object.

    The result is drawn from :data:`JsonShaped`.  ``DecimalValue`` is
    preserved as :class:`decimal.Decimal` (never converted to ``float``); a
    ``json``-typed value's own raw payload may already carry a ``float``,
    exactly as ``json.loads`` produced it.

    Reference semantics makes cyclic arrays, dicts, and records constructible;
    ``active`` (an active-value-id set, allocated lazily on first use) detects
    a cycle and raises :class:`~agm.agl.semantics.cycles.AglCyclicValue` rather
    than recursing forever. Callers pass no *active* argument — it exists
    only to thread the walk's own recursive calls.

    A ``unit``, ``constructor``, ``function``, or ``contract``
    value has no JSON representation at all; such a value raises
    :class:`AglNonDataValue` rather than a bare :class:`TypeError`, so a
    caller that can legitimately receive one (e.g. because it carries the
    field of an in-flight exception) can degrade it to a marker instead of
    crashing. Lowered casts use :func:`encode_value`, which retains source-slot
    context; this fallback serves reporting values that have no associated
    static slot plan. ``tags`` (built once by :func:`_walk_tags`) is how such
    a caller still names an enum member's ``$case`` and a nested
    record/exception field's effective JSON key.

    A dict has no static key type here (unlike :func:`encode_value`, which
    picks a key's wire form from its plan's resolved encode head via
    ``DictKeyForm``): a ``text``-keyed dict (:meth:`DictValue.is_text_keyed`),
    and an empty dict of any representation, walks as a JSON object; any
    other (non-empty, token-keyed) dict walks as the entries-array form
    (``[{"key": ..., "value": ...}, ...]``) instead of stringifying its keys
    with ``str`` — which would collapse distinct keys together (an enum's
    Python repr, ``json`` `1` and `"1"`, ...).
    """
    value = cast(ObservableValue, value)
    if isinstance(value, TextValue):
        return value.value
    if isinstance(value, IntValue):
        return value.value
    if isinstance(value, DecimalValue):
        return value.value
    if isinstance(value, BoolValue):
        return value.value
    if isinstance(value, JsonValue):
        return cast(JsonShaped, value.raw)
    if isinstance(value, ArrayValue):
        active = enter_value(id(value), active)
        try:
            return [value_to_json_obj(e, active, tags=tags) for e in value.elements]
        finally:
            active.discard(id(value))
    if isinstance(value, DictValue):
        active = enter_value(id(value), active)
        try:
            return cast(
                JsonShaped,
                _walk_dict(value, lambda item: value_to_json_obj(item, active, tags=tags)),
            )
        finally:
            active.discard(id(value))
    if isinstance(value, (RecordValue, ExceptionValue)):
        active = enter_value(id(value), active)
        try:
            field_names = tags.field_names[value.nominal]
            fields = {
                field_names[k]: value_to_json_obj(v, active, tags=tags)
                for k, v in value.fields.items()
            }
            tag = tags.member_tags.get(value.nominal) if isinstance(value, RecordValue) else None
            return {"$case": tag, **fields} if tag is not None else fields
        finally:
            active.discard(id(value))
    if isinstance(value, UnitValue):
        raise AglNonDataValue("unit")
    if isinstance(value, ConstructorValue):
        raise AglNonDataValue("constructor")
    if isinstance(value, IrClosureValue):
        raise AglNonDataValue("function")
    if isinstance(value, ContractValue):
        raise AglNonDataValue("contract")
    assert_never(value)  # pragma: no cover


def value_to_trace_json_obj(
    value: Value,
    descriptors: "ValueDescriptors",
    builtin_nominals: "BuiltinNominals",
) -> object:
    """Convert a runtime value to a best-effort JSON shape for a trace record.

    Unlike :func:`value_to_json_obj`, this walk degrades cycles and non-data
    values instead of raising. Enum members retain their ``$case`` tag, while
    decimal values use their exact text form because JSONL trace records use
    the standard JSON encoder.
    """
    enum_members = {
        variant.member: variant.name
        for descriptor in descriptors.nominals.values()
        for variant in descriptor.variants
    }
    return _trace_value(value, enum_members, builtin_nominals, None)


def _trace_value(
    value: Value,
    enum_members: "Mapping[NominalId, str]",
    builtin_nominals: "BuiltinNominals",
    active: set[int] | None,
) -> object:
    if isinstance(value, TextValue):
        return value.value if surrogate_index(value.value) is None else non_data_marker("text")
    if isinstance(value, IntValue):
        return value.value
    if isinstance(value, DecimalValue):
        return dumps_exact(value.value, indent=None)
    if isinstance(value, BoolValue):
        return value.value
    if isinstance(value, JsonValue):
        return _trace_json_data(value.raw, active)
    if isinstance(value, ArrayValue):
        active = _trace_enter(value, active)
        if active is None:
            return CYCLIC_VALUE_MARKER
        try:
            return [
                _trace_value(item, enum_members, builtin_nominals, active)
                for item in value.elements
            ]
        finally:
            active.discard(id(value))
    if isinstance(value, DictValue):
        active = _trace_enter(value, active)
        if active is None:
            return CYCLIC_VALUE_MARKER
        try:
            return _walk_dict(
                value, lambda item: _trace_value(item, enum_members, builtin_nominals, active)
            )
        finally:
            active.discard(id(value))
    if isinstance(value, (RecordValue, ExceptionValue)):
        active = _trace_enter(value, active)
        if active is None:
            return CYCLIC_VALUE_MARKER
        try:
            fields = {
                key: _trace_value(item, enum_members, builtin_nominals, active)
                for key, item in value.fields.items()
            }
            variant = enum_members.get(value.nominal)
            if variant is None:
                builtin_member = builtin_nominals.reverse(value.nominal)
                variant = None if builtin_member is None else builtin_member[1]
            return {"$case": variant, **fields} if variant is not None else fields
        finally:
            active.discard(id(value))
    if isinstance(value, UnitValue):
        return non_data_marker("unit")
    if isinstance(value, ConstructorValue):
        return non_data_marker("constructor")
    if isinstance(value, IrClosureValue):
        return non_data_marker("function")
    if isinstance(value, IteratorValue):
        return non_data_marker("iterator")
    if isinstance(value, ContractValue):
        return non_data_marker("contract")
    assert_never(value)  # pragma: no cover


def _trace_enter(value: object, active: set[int] | None) -> set[int] | None:
    try:
        return enter_value(id(value), active)
    except AglCyclicValue:
        return None


def _trace_json_data(value: object, active: set[int] | None) -> object:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return value if surrogate_index(value) is None else non_data_marker("text")
    if isinstance(value, float):
        return value if isfinite(value) else non_data_marker("float")
    if isinstance(value, Decimal):
        return dumps_exact(value, indent=None)
    if isinstance(value, Mapping):
        nested = _trace_enter(value, active)
        if nested is None:
            return CYCLIC_VALUE_MARKER
        try:
            return {str(key): _trace_json_data(item, nested) for key, item in value.items()}
        finally:
            nested.discard(id(value))
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        nested = _trace_enter(value, active)
        if nested is None:
            return CYCLIC_VALUE_MARKER
        try:
            return [_trace_json_data(item, nested) for item in value]
        finally:
            nested.discard(id(value))
    return non_data_marker(type(value).__name__)


def dumps_exact(obj: JsonShaped, *, indent: int | None = 2) -> str:
    """Serialize a JSON-shaped object to text, emitting decimals exactly.

    Operates over the closed :data:`JsonShaped` domain.  A
    :class:`decimal.Decimal` is emitted as unquoted numeric text using its
    exact string form — it is never routed through :class:`float`; a
    ``float`` itself (never produced by :func:`value_to_json_obj`, but valid
    for a host-native value crossing the argument-decoding boundary) is
    delegated to ``json.dumps``.

    A small recursive emitter is used (rather than ``json.dumps`` throughout)
    because the stdlib encoder cannot serialize ``Decimal`` without a
    binary-float round trip.  ``str``/``bool``/``int``/``float``/``None``
    leaves are still delegated to ``json.dumps`` so that escaping and
    formatting match the stdlib exactly.
    """
    return _emit(obj, indent=indent, level=0)


def _emit(obj: JsonShaped, *, indent: int | None, level: int) -> str:
    # ``bool`` must be checked before ``int`` (bool is a subclass of int).
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, Decimal):
        return _decimal_text(obj)
    if isinstance(obj, (str, int, float)) or obj is None:
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, list):
        return _emit_array(obj, indent=indent, level=level)
    return _emit_dict(obj, indent=indent, level=level)


#: Above this many characters, a decimal's exact fixed-point JSON-number text
#: risks exhausting memory. ``str`` alone is not a bound: it only switches to
#: scientific notation for a positive exponent or an adjusted exponent below
#: -6, so a mid-range value -- positive exponent, but not so negative that
#: ``str`` would already switch -- still renders fixed-point at whatever
#: length its digits and exponent demand. A ``json``-typed value (exempt from
#: the pinned range, and so of any magnitude) or a decimal near the pinned
#: context's own Emax/Etiny can force such an expansion of well over a
#: million characters.
_MAX_FIXED_POINT_DIGITS = 10_000


def _decimal_text(d: Decimal) -> str:
    """Exact unquoted JSON-number text for a finite ``Decimal`` (no float round trip).

    Every decimal creation site, ``json`` values included, rejects a
    non-finite number. Renders as plain fixed-point (``format(d, "f")``)
    whenever its length stays bounded; above :data:`_MAX_FIXED_POINT_DIGITS`,
    the minimal-coefficient scientific form (``str`` on the trailing-zero-
    stripped value, e.g. ``"1E+40"``) is emitted instead -- still exact, and
    still a valid JSON number (``int exp``).
    """
    _, digits, exponent = d.as_tuple()
    # Finite: only NaN and infinity carry a letter exponent.
    exponent = cast(int, exponent)
    if len(digits) + abs(exponent) <= _MAX_FIXED_POINT_DIGITS:
        return format(d, "f")
    return str(strip_trailing_zeros(d))


def _emit_array(obj: list[JsonShaped], *, indent: int | None, level: int) -> str:
    if not obj:
        return "[]"
    if indent is None:
        items = [_emit(e, indent=None, level=level) for e in obj]
        return "[" + ", ".join(items) + "]"
    pad = " " * (indent * (level + 1))
    close_pad = " " * (indent * level)
    items = [pad + _emit(e, indent=indent, level=level + 1) for e in obj]
    return "[\n" + ",\n".join(items) + "\n" + close_pad + "]"


def _emit_dict(obj: dict[str, JsonShaped], *, indent: int | None, level: int) -> str:
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
