"""Sealed handles and walkers for the extern boundary.

This eval-free runtime module converts AgL values to Python arguments and
strictly converts Python return values back to AgL values. Arrays cross as
live views; nominal values are rebuilt and ``json`` values are deep-copied.
Each :class:`BoundaryScope` carries one call's seals, recursive definitions,
sealed-handle vault, and array-view memo.
"""

from __future__ import annotations

import copy
import decimal
import operator
import weakref
from collections.abc import Iterable, Iterator, Mapping, MutableSequence
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Callable, Protocol, Self, SupportsIndex, assert_never, cast, overload

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
    BoundaryUnit,
    ScalarKind,
    reconcile_array_view_element_schema,
)
from agm.agl.runtime.render import render_value
from agm.agl.runtime.serialize import value_to_json_obj
from agm.agl.semantics.cycles import enter_container
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
    IteratorValue,
    JsonValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
)

# ---------------------------------------------------------------------------
# Internal exceptions
# ---------------------------------------------------------------------------


class BoundaryViolation(Exception):
    """Internal signal for a contract violation while crossing the boundary.

    Raised by :func:`encode_boundary_value` / :func:`decode_boundary_value` on
    any structural mismatch (wrong type, wrong variant, missing/extra/
    misnamed fields, a missing or mismatched seal, ...).  Caught by
    :meth:`ExternRegistry.invoke`, which reports it as ``ExternError`` with an
    empty ``python_type``.
    """


class BoundaryTypeError(TypeError):
    """A value written through a boundary view violates its element schema."""


class BoundaryViewRevoked(RuntimeError):
    """A boundary view was used after its scope ended."""


class _SupportsLessThan(Protocol):
    """A value that can be ordered by Python's less-than operator."""

    def __lt__(self, other: object, /) -> bool: ...


# ---------------------------------------------------------------------------
# SealedHandle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SealedPayload:
    """Boundary-private payload for a minted sealed handle."""

    value: Value
    seal: object


_HANDLE_FACTORY_KEY = object()


class SealedHandle:
    """Opaque wrapper for an AgL value at a sealed type-variable position.

    A Python companion may rearrange, count, and compare handles it receives,
    but cannot inspect or forge them: handles expose no value/seal attributes,
    and the public constructor rejects companion-created instances. Only the
    boundary encoder can mint an instance whose private id resolves in its
    call's boundary scope.

    ``__eq__``/``__hash__`` mirror the wrapped value's own equality and hash
    (never equal to a non-handle), so handles compose correctly in Python sets
    and dicts without exposing the wrapped value on the handle — except for a
    wrapped array or dict, where the key is the container's **identity**, not
    its structure (see ``_sealed_value_eq_key``): two handles wrapping equal
    but distinct arrays are not equal.  ``__repr__`` shows the rendered AgL
    value as a debugging aid; repr'ing a wrapped value that contains a
    reference cycle surfaces as the catchable ``CyclicValueError`` at the
    enclosing extern call, mirroring ``render``/``print`` on a cyclic value.
    """

    __slots__ = ("__weakref__",)

    def __init__(self, *_args: object, _factory_key: object | None = None) -> None:
        if _factory_key is not _HANDLE_FACTORY_KEY:
            raise TypeError("SealedHandle instances can only be minted by the extern boundary")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SealedHandle):
            return False
        self_state = _HANDLE_STATE.get(id(self))
        other_state = _HANDLE_STATE.get(id(other))
        if self_state is None or other_state is None:
            return False
        return self_state.eq_key() == other_state.eq_key()

    def __hash__(self) -> int:
        state = _HANDLE_STATE.get(id(self))
        if state is None:
            raise TypeError("uninitialized sealed handle is not hashable")
        return hash(state.eq_key())

    def __repr__(self) -> str:
        state = _HANDLE_STATE.get(id(self))
        if state is None:
            return "<sealed handle>"
        return state.rendered()


class _HandleState:
    """Deferred equality key and repr for a minted sealed handle.

    The wrapped value is already retained by the minting vault, so this holder
    keeps only a reference to it and derives the structural equality key
    lazily, on first ``__hash__``/``__eq__``.  A handle that merely crosses
    the boundary and is passed back — never hashed or compared — therefore
    pays none of that O(size) work.  The repr, by contrast, is never memoized:
    ``ArrayValue``/``DictValue`` are mutable in place, so a cached repr could
    report pre-mutation contents for a handle a companion retains across
    calls; :meth:`rendered` re-renders the wrapped value on every access.
    """

    __slots__ = ("_value", "_eq_key", "_has_eq_key")

    def __init__(self, value: Value) -> None:
        self._value = value
        self._eq_key: object = None
        self._has_eq_key = False

    def eq_key(self) -> object:
        if not self._has_eq_key:
            self._eq_key = _sealed_value_eq_key(self._value)
            self._has_eq_key = True
        return self._eq_key

    def rendered(self) -> str:
        return render_value(self._value)


class _HandleVault:
    """Per-boundary storage for sealed payloads, kept out of module globals."""

    def __init__(self) -> None:
        self._payloads: dict[int, _SealedPayload] = {}

    def make(self, value: Value, seal: object) -> SealedHandle:
        """Mint a sealed handle and retain its payload in this vault."""
        handle = SealedHandle(_factory_key=_HANDLE_FACTORY_KEY)
        handle_id = id(handle)
        self._payloads[handle_id] = _SealedPayload(value=value, seal=seal)
        _HANDLE_STATE[handle_id] = _HandleState(value)
        weakref.finalize(handle, _drop_handle_state, self._payloads, handle_id)
        return handle

    def open(self, handle: SealedHandle) -> _SealedPayload:
        """Return a minted handle's payload, rejecting handles outside this vault."""
        payload = self._payloads.get(id(handle))
        if payload is None:
            raise BoundaryViolation("sealed handle was not minted by this boundary")
        return payload


_HANDLE_STATE: dict[int, _HandleState] = {}


def _drop_handle_state(payloads: dict[int, _SealedPayload], handle_id: int) -> None:
    payloads.pop(handle_id, None)
    _HANDLE_STATE.pop(handle_id, None)


def _sealed_value_eq_key(value: Value) -> object:
    """Return an immutable equality key for a sealed value.

    ``ArrayValue``/``DictValue`` use **object identity**, not a structural
    walk: reference semantics makes a mutable array/dict cyclic and its
    payload never stable, so a structural key could both loop forever and go
    stale the moment the container is mutated. Identity sidesteps both
    problems, and is sound precisely because ``_HandleState`` retains the
    wrapped value for as long as the handle — and therefore this key — can be
    observed, so the ``id()`` can never be reused by an unrelated object while
    it is still in play. The nominal arms below still recurse into their
    fields structurally, which is safe because a cycle can only ever be
    closed through an array or dict.
    """
    if isinstance(value, TextValue):
        return ("text", value.value)
    if isinstance(value, IntValue):
        return ("int", value.value)
    if isinstance(value, DecimalValue):
        return ("decimal", value.value)
    if isinstance(value, BoolValue):
        return ("bool", value.value)
    if isinstance(value, JsonValue):
        return ("json", _json_eq_key(value.raw))
    if isinstance(value, ArrayValue):
        return ("array", id(value))
    if isinstance(value, DictValue):
        return ("dict", id(value))
    if isinstance(value, RecordValue):
        return (
            "record",
            value.nominal,
            tuple(sorted((key, _sealed_value_eq_key(item)) for key, item in value.fields.items())),
        )
    if isinstance(value, EnumValue):
        return (
            "enum",
            value.nominal,
            value.variant,
            tuple(sorted((key, _sealed_value_eq_key(item)) for key, item in value.fields.items())),
        )
    if isinstance(value, ExceptionValue):
        return (
            "exception",
            value.nominal,
            tuple(sorted((key, _sealed_value_eq_key(item)) for key, item in value.fields.items())),
        )
    if isinstance(value, UnitValue):
        return ("unit",)
    if isinstance(value, AgentValue):
        return ("agent", value.name)
    if isinstance(value, ConstructorValue):
        return ("constructor", value.nominal, value.variant)
    if isinstance(value, (IrClosureValue, IteratorValue)):
        return (type(value).__name__, id(value))
    assert_never(value)  # pragma: no cover


def _json_eq_key(obj: object) -> object:
    """Return an immutable key matching ``JsonValue`` equality/hash semantics."""
    if isinstance(obj, bool):
        return ("bool", obj)
    if isinstance(obj, (int, decimal.Decimal)):
        return ("number", decimal.Decimal(obj))
    if isinstance(obj, list):
        return ("list", tuple(_json_eq_key(item) for item in obj))
    if isinstance(obj, dict):
        return ("dict", tuple(sorted((_json_eq_key(k), _json_eq_key(v)) for k, v in obj.items())))
    return ("scalar", obj)


def _typename(obj: object) -> str:
    """Human-readable type name (AgL value-kind or Python type) for a boundary message."""
    return type(obj).__name__


#: Shared read-only empty ``defs`` table for boundary walks over a self-contained
#: schema (one with no ``BoundaryRef`` leaf, i.e. no recursive type).
_NO_DEFS: Mapping[str, BoundarySchema] = MappingProxyType({})


class BoundaryScope:
    """One extern call's boundary state: seals, defs, handle vault, view memo, live flag."""

    def __init__(
        self,
        seals: Mapping[str, object] | None = None,
        defs: Mapping[str, BoundarySchema] | None = None,
    ) -> None:
        self.seals = seals if seals is not None else MappingProxyType({})
        self.defs = defs if defs else _NO_DEFS
        self._vault = _HandleVault()
        self._array_views: dict[int, AglArrayView] = {}
        self.live = True

    def array_view(self, value: ArrayValue, element_schema: BoundarySchema) -> AglArrayView:
        """Return this scope's live view for *value*, reconciling its element schema."""
        element_schema = _resolve_boundary_ref(element_schema, self)
        marker = id(value)
        cached = self._array_views.get(marker)
        if cached is not None:
            cached.reconcile_element_schema(element_schema)
            return cached
        view = AglArrayView(self, value, element_schema)
        self._array_views[marker] = view
        return view

    def revoke(self) -> None:
        """Mark the scope inactive for future boundary views."""
        self.live = False


class AglArrayView(MutableSequence[object]):
    """A live Python sequence view of an AgL array value."""

    __slots__ = ("_scope", "_value", "_element_schema")

    def __init__(
        self, scope: BoundaryScope, value: ArrayValue, element_schema: BoundarySchema
    ) -> None:
        self._scope = scope
        self._value = value
        self._element_schema = _resolve_boundary_ref(element_schema, scope)

    def __len__(self) -> int:
        return len(self._value.elements)

    @overload
    def __getitem__(self, index: SupportsIndex) -> object: ...

    @overload
    def __getitem__(self, index: slice[int | None, int | None, int | None]) -> list[object]: ...

    def __getitem__(
        self, index: SupportsIndex | slice[int | None, int | None, int | None]
    ) -> object | list[object]:
        if isinstance(index, SupportsIndex):
            return self._encode(self._value.elements[operator.index(index)])
        return [self._encode(item) for item in self._value.elements[index]]

    @overload
    def __setitem__(self, index: SupportsIndex, value: object) -> None: ...

    @overload
    def __setitem__(
        self,
        index: slice[int | None, int | None, int | None],
        value: Iterable[object],
    ) -> None: ...

    def __setitem__(
        self,
        index: SupportsIndex | slice[int | None, int | None, int | None],
        value: object,
    ) -> None:
        if isinstance(index, SupportsIndex):
            normalized_index = operator.index(index)
            self._value.elements[normalized_index]
            self._value.elements[normalized_index] = self._decode(value)
            return
        values = cast(Iterable[object], value)
        self._value.elements[index] = [self._decode(item) for item in values]

    @overload
    def __delitem__(self, index: SupportsIndex) -> None: ...

    @overload
    def __delitem__(self, index: slice[int | None, int | None, int | None]) -> None: ...

    def __delitem__(self, index: SupportsIndex | slice[int | None, int | None, int | None]) -> None:
        if isinstance(index, SupportsIndex):
            del self._value.elements[operator.index(index)]
            return
        del self._value.elements[index]

    def insert(self, index: int, value: object) -> None:
        self._value.elements.insert(index, self._decode(value))

    def append(self, value: object) -> None:
        self._value.elements.append(self._decode(value))

    def clear(self) -> None:
        self._value.elements.clear()

    def extend(self, values: Iterable[object]) -> None:
        if values is self:
            self._value.elements.extend(self._value.elements.copy())
            return
        self._value.elements.extend(self._decode(item) for item in values)

    def pop(self, index: int = -1) -> object:
        return self._encode(self._value.elements.pop(index))

    def remove(self, value: object) -> None:
        for index, item in enumerate(self._value.elements):
            if self._encode(item) == value:
                del self._value.elements[index]
                return
        raise ValueError(f"{value!r} is not in list")

    def reverse(self) -> None:
        self._value.elements.reverse()

    def __iadd__(self, values: Iterable[object]) -> Self:
        self.extend(values)
        return self

    def sort(self, *, key: Callable[[object], object] | None = None, reverse: bool = False) -> None:
        pairs = [(self._encode(value), value) for value in self._value.elements]
        keys = [key(encoded) if key is not None else encoded for encoded, _ in pairs]
        for index in range(1, len(pairs)):
            pair = pairs[index]
            sort_key = keys[index]
            position = index
            while position > 0 and _sorts_before(sort_key, keys[position - 1], reverse):
                pairs[position] = pairs[position - 1]
                keys[position] = keys[position - 1]
                position -= 1
            pairs[position] = pair
            keys[position] = sort_key
        self._value.elements[:] = [value for _, value in pairs]

    def __iter__(self) -> Iterator[object]:
        index = 0
        while index < len(self._value.elements):
            yield self[index]
            index += 1

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return object.__hash__(self)

    def __repr__(self) -> str:
        return render_value(self._value)

    def reconcile_element_schema(self, element_schema: BoundarySchema) -> None:
        """Adopt a compatible alias schema without changing this view's identity."""
        reconciled = _reconcile_array_view_element_schema(self._element_schema, element_schema)
        if reconciled is None:
            raise BoundaryViolation("aliased array has incompatible element schemas")
        self._element_schema = reconciled

    def _encode(self, value: Value) -> object:
        return _encode_boundary_value(self._element_schema, value, self._scope, None)

    def _decode(self, value: object) -> Value:
        return _decode_boundary_write(self._element_schema, value, self._scope)


class AglDictView:
    pass


def _resolve_boundary_ref(schema: BoundarySchema, scope: BoundaryScope) -> BoundarySchema:
    """Resolve the reference chain at a view element position."""
    while isinstance(schema, BoundaryRef):
        schema = scope.defs[schema.key]
    return schema


def _reconcile_array_view_element_schema(
    existing: BoundarySchema, incoming: BoundarySchema
) -> BoundarySchema | None:
    """Choose a shared array-view element schema, if the aliases are compatible.

    A type-variable seal is less specific than a concrete schema. Nested
    arrays reconcile their element schemas recursively so aliases choose the
    same concrete representation at every array-view boundary.
    """
    if existing == incoming:
        return existing
    if isinstance(existing, BoundaryArray) and isinstance(incoming, BoundaryArray):
        element = _reconcile_array_view_element_schema(existing.element, incoming.element)
        if element is None:
            return None
        return BoundaryArray(element)
    return reconcile_array_view_element_schema(existing, incoming)


def _sorts_before(left: object, right: object, reverse: bool) -> bool:
    """Compare two encoded sort keys, preserving Python comparison behavior."""
    if reverse:
        return cast(_SupportsLessThan, right) < left
    return cast(_SupportsLessThan, left) < right


def _decode_boundary_write(schema: BoundarySchema, value: object, scope: BoundaryScope) -> Value:
    """Decode a view write, exposing schema violations as ``TypeError``."""
    try:
        return _decode_boundary_value(schema, value, scope, set())
    except BoundaryViolation as exc:
        raise BoundaryTypeError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Encode: AgL Value -> Python argument
# ---------------------------------------------------------------------------


def encode_boundary_value(
    schema: BoundarySchema, value: Value, scope: BoundaryScope | None = None
) -> object:
    """Encode one AgL value crossing an extern boundary as a Python argument."""
    return _encode_boundary_value(
        schema, value, scope if scope is not None else BoundaryScope(), None
    )


def _encode_boundary_value(
    schema: BoundarySchema,
    value: Value,
    scope: BoundaryScope,
    active: set[int] | None,
) -> object:
    """Recursively encode a value using one call's boundary scope."""
    match schema:
        case BoundaryScalar(kind=kind):
            return _encode_scalar(kind, value)
        case BoundaryUnit():
            if not isinstance(value, UnitValue):
                raise BoundaryViolation(f"expected unit, got {_typename(value)}")
            return None
        case BoundaryArray(element=elem_schema):
            if not isinstance(value, ArrayValue):
                raise BoundaryViolation(f"expected an array value, got {_typename(value)}")
            return scope.array_view(value, elem_schema)
        case BoundaryDict(value=val_schema):
            if not isinstance(value, DictValue):
                raise BoundaryViolation(f"expected a dict value, got {_typename(value)}")
            active = enter_container(id(value), active)
            try:
                return {
                    key: _encode_boundary_value(val_schema, item, scope, active)
                    for key, item in value.entries.items()
                }
            finally:
                active.discard(id(value))
        case BoundaryRecord(display_name=display_name, fields=fields):
            if not isinstance(value, RecordValue):
                raise BoundaryViolation(f"expected record {display_name!r}, got {_typename(value)}")
            return _encode_boundary_fields(fields, value.fields, scope, active)
        case BoundaryEnum(display_name=display_name, variants=variants):
            if not isinstance(value, EnumValue):
                raise BoundaryViolation(f"expected enum {display_name!r}, got {_typename(value)}")
            variant = next((item for item in variants if item.name == value.variant), None)
            if variant is None:
                raise BoundaryViolation(f"enum {display_name!r}: unknown variant {value.variant!r}")
            result: dict[str, object] = {"$case": value.variant}
            result.update(_encode_boundary_fields(variant.fields, value.fields, scope, active))
            return result
        case BoundaryException(display_name=display_name, fields=fields):
            if not isinstance(value, ExceptionValue):
                raise BoundaryViolation(
                    f"expected exception {display_name!r}, got {_typename(value)}"
                )
            return _encode_boundary_fields(fields, value.fields, scope, active)
        case BoundarySealVar(var=var):
            return scope._vault.make(value, scope.seals[var])
        case BoundaryRef(key=key):
            return _encode_boundary_value(scope.defs[key], value, scope, active)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _encode_boundary_fields(
    fields: tuple[tuple[str, BoundarySchema], ...],
    values: Mapping[str, Value],
    scope: BoundaryScope,
    active: set[int] | None,
) -> dict[str, object]:
    """Encode a nominal payload's ordered fields through the boundary schema."""
    return {
        field_name: _encode_boundary_value(schema, values[field_name], scope, active)
        for field_name, schema in fields
    }


def _encode_scalar(kind: ScalarKind, value: Value) -> object:
    """Encode one scalar (or opaque json) leaf as its Python argument."""
    match kind:
        case ScalarKind.TEXT:
            if not isinstance(value, TextValue):
                raise BoundaryViolation(f"expected text, got {_typename(value)}")
            return value.value
        case ScalarKind.INT:
            if not isinstance(value, IntValue):
                raise BoundaryViolation(f"expected int, got {_typename(value)}")
            return value.value
        case ScalarKind.DECIMAL:
            if not isinstance(value, DecimalValue):
                raise BoundaryViolation(f"expected decimal, got {_typename(value)}")
            return value.value
        case ScalarKind.BOOL:
            if not isinstance(value, BoolValue):
                raise BoundaryViolation(f"expected bool, got {_typename(value)}")
            return value.value
        case ScalarKind.JSON:
            if not isinstance(value, JsonValue):
                raise BoundaryViolation(f"expected json, got {_typename(value)}")
            return copy.deepcopy(value_to_json_obj(value))
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


# ---------------------------------------------------------------------------
# Decode: Python return value -> AgL Value (strict)
# ---------------------------------------------------------------------------


def decode_boundary_value(
    schema: BoundarySchema, obj: object, scope: BoundaryScope | None = None
) -> Value:
    """Strictly decode one Python return value against a boundary schema."""
    return _decode_boundary_value(
        schema, obj, scope if scope is not None else BoundaryScope(), set()
    )


def _decode_boundary_value(
    schema: BoundarySchema,
    obj: object,
    scope: BoundaryScope,
    active_containers: set[int],
) -> Value:
    """Recursively decode a Python value using one call's boundary scope."""
    match schema:
        case BoundaryScalar(kind=kind):
            return _decode_scalar(kind, obj)
        case BoundaryUnit():
            if obj is not None:
                raise BoundaryViolation(f"expected unit (None), got {_typename(obj)}")
            return UnitValue()
        case BoundaryArray(element=elem_schema):
            if type(obj) is not list:
                raise BoundaryViolation(f"expected a list, got {_typename(obj)}")
            items = cast(list[object], obj)
            marker = _enter_python_container(items, active_containers, "array")
            try:
                return ArrayValue(
                    [
                        _decode_boundary_value(elem_schema, item, scope, active_containers)
                        for item in items
                    ]
                )
            finally:
                active_containers.remove(marker)
        case BoundaryDict(value=val_schema):
            if type(obj) is not dict:
                raise BoundaryViolation(f"expected a dict, got {_typename(obj)}")
            mapping = cast(dict[object, object], obj)
            marker = _enter_python_container(mapping, active_containers, "dict")
            try:
                entries: dict[str, Value] = {}
                for k, v in mapping.items():
                    if not isinstance(k, str):
                        raise BoundaryViolation(f"dict key must be str, got {_typename(k)}")
                    entries[k] = _decode_boundary_value(val_schema, v, scope, active_containers)
                return DictValue(entries=entries)
            finally:
                active_containers.remove(marker)
        case BoundaryRecord(nominal=nominal, display_name=display_name, fields=fields):
            marker = _enter_python_container(obj, active_containers, display_name)
            try:
                obj_fields = _expect_object(obj, display_name)
                _check_exact_fields(display_name, {fname for fname, _ in fields}, obj_fields)
                record_fields = _decode_boundary_fields(
                    fields, obj_fields, scope, active_containers
                )
                return RecordValue(nominal=nominal, display_name=display_name, fields=record_fields)
            finally:
                active_containers.remove(marker)
        case BoundaryEnum(nominal=nominal, display_name=display_name, variants=variants):
            marker = _enter_python_container(obj, active_containers, display_name)
            try:
                obj_fields = _expect_object(obj, display_name)
                case_val = obj_fields.get("$case")
                if not isinstance(case_val, str):
                    raise BoundaryViolation(
                        f"enum {display_name!r}: object must have a string '$case' field"
                    )
                variant = next((v for v in variants if v.name == case_val), None)
                if variant is None:
                    raise BoundaryViolation(f"enum {display_name!r}: unknown variant {case_val!r}")
                expected = {fname for fname, _ in variant.fields} | {"$case"}
                _check_exact_fields(f"{display_name}.{case_val}", expected, obj_fields)
                payload = _decode_boundary_fields(
                    variant.fields, obj_fields, scope, active_containers
                )
                return EnumValue(
                    nominal=nominal, display_name=display_name, variant=case_val, fields=payload
                )
            finally:
                active_containers.remove(marker)
        case BoundaryException(nominal=nominal, display_name=display_name, fields=fields):
            marker = _enter_python_container(obj, active_containers, display_name)
            try:
                obj_fields = _expect_object(obj, display_name)
                _check_exact_fields(display_name, {fname for fname, _ in fields}, obj_fields)
                exc_fields = _decode_boundary_fields(fields, obj_fields, scope, active_containers)
                return ExceptionValue(nominal=nominal, display_name=display_name, fields=exc_fields)
            finally:
                active_containers.remove(marker)
        case BoundarySealVar(var=var):
            if not isinstance(obj, SealedHandle):
                raise BoundaryViolation(
                    f"expected a sealed handle for type variable {var!r}, got {_typename(obj)}"
                )
            sealed_payload = scope._vault.open(obj)
            if sealed_payload.seal is not scope.seals.get(var):
                raise BoundaryViolation(
                    f"handle does not carry this call's seal for type variable {var!r}"
                )
            return sealed_payload.value
        case BoundaryRef(key=key):
            return _decode_boundary_value(scope.defs[key], obj, scope, active_containers)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _expect_object(obj: object, display_name: str) -> dict[str, object]:
    """Return *obj* as a ``str``-keyed dict, or raise ``BoundaryViolation``."""
    if type(obj) is not dict:
        raise BoundaryViolation(f"expected an object for {display_name!r}, got {_typename(obj)}")
    mapping = cast(dict[object, object], obj)
    result: dict[str, object] = {}
    for k, v in mapping.items():
        if not isinstance(k, str):
            raise BoundaryViolation(f"{display_name!r}: object key must be str, got {_typename(k)}")
        result[k] = v
    return result


def _check_exact_fields(
    display_name: str, expected: set[str], obj_fields: Mapping[str, object]
) -> None:
    """Raise ``BoundaryViolation`` unless *obj_fields* has exactly *expected* keys."""
    actual = set(obj_fields)
    if actual != expected:
        raise BoundaryViolation(
            f"{display_name!r}: field mismatch (expected {sorted(expected)}, got {sorted(actual)})"
        )


def _enter_python_container(obj: object, active: set[int], label: str) -> int:
    """Mark *obj* as active during decode, rejecting cyclic Python returns.

    The decode-direction counterpart to
    :func:`~agm.agl.semantics.cycles.enter_container`, which this module also
    uses for the encode direction. They are deliberately separate: a cycle in
    a Python value arriving from a companion module is a boundary violation
    (the companion returned something undecodable), whereas a cycle in an AgL
    value leaving for Python is the language-level ``CyclicValueError``.
    """
    marker = id(obj)
    if marker in active:
        raise BoundaryViolation(f"cyclic Python return value at {label}")
    active.add(marker)
    return marker


def _decode_boundary_fields(
    fields: tuple[tuple[str, BoundarySchema], ...],
    obj_fields: Mapping[str, object],
    scope: BoundaryScope,
    active_containers: set[int],
) -> dict[str, Value]:
    """Decode a nominal payload's ordered fields through the boundary schema."""
    return {
        field_name: _decode_boundary_value(
            field_schema, obj_fields[field_name], scope, active_containers
        )
        for field_name, field_schema in fields
    }


def _decode_scalar(kind: ScalarKind, obj: object) -> Value:
    """Strictly decode a Python scalar (or opaque json) into the matching leaf value."""
    match kind:
        case ScalarKind.TEXT:
            if isinstance(obj, str):
                return TextValue(obj)
            raise BoundaryViolation(f"expected text (str), got {_typename(obj)}")
        case ScalarKind.INT:
            if isinstance(obj, bool):
                raise BoundaryViolation("expected int, got bool")
            if isinstance(obj, int):
                return IntValue(obj)
            raise BoundaryViolation(f"expected int, got {_typename(obj)}")
        case ScalarKind.DECIMAL:
            if isinstance(obj, bool):
                raise BoundaryViolation("expected decimal, got bool")
            if isinstance(obj, Decimal):
                if not obj.is_finite():
                    raise BoundaryViolation("expected finite decimal")
                return DecimalValue(obj)
            if isinstance(obj, int):
                return DecimalValue(Decimal(obj))
            raise BoundaryViolation(f"expected decimal, got {_typename(obj)}")
        case ScalarKind.BOOL:
            if isinstance(obj, bool):
                return BoolValue(obj)
            raise BoundaryViolation(f"expected bool, got {_typename(obj)}")
        case ScalarKind.JSON:
            if not _is_json_shaped(obj):
                raise BoundaryViolation(f"expected a JSON-shaped value, got {_typename(obj)}")
            try:
                return JsonValue(copy.deepcopy(obj))
            except Exception as exc:
                raise BoundaryViolation(f"could not copy JSON-shaped value: {exc}") from exc
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _is_json_shaped(obj: object, active: set[int] | None = None) -> bool:
    """Return whether *obj* lies in the closed JSON-shape domain.

    Exact built-in ``dict``/``list`` plus ``str``/``int``/
    :class:`~decimal.Decimal`/``bool``/``None`` recursively; anything else
    (a ``float``, a :class:`SealedHandle`, an arbitrary object, or a
    ``dict``/``list`` subclass) is rejected.
    """
    if active is None:
        active = set()
    if isinstance(obj, Decimal):
        return obj.is_finite()
    if obj is None or isinstance(obj, (bool, str, int)):
        return True
    if type(obj) is list:
        items = cast(list[object], obj)
        marker = id(items)
        if marker in active:
            return False
        active.add(marker)
        try:
            return all(_is_json_shaped(e, active) for e in items)
        finally:
            active.remove(marker)
    if type(obj) is dict:
        mapping = cast(dict[object, object], obj)
        marker = id(mapping)
        if marker in active:
            return False
        active.add(marker)
        try:
            return all(
                isinstance(k, str) and _is_json_shaped(v, active) for k, v in mapping.items()
            )
        finally:
            active.remove(marker)
    return False
