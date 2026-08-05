"""Sealed handles and conversion walkers for the extern boundary.

This eval-free runtime module converts AgL values to Python arguments and
strictly converts Python return values back to AgL values. Arrays and dicts
cross as per-call live views: repeated occurrences reuse a view only when
their boundary schemas reconcile, and a matching returned view preserves its
original AgL container; built-in containers decode to new ones. A bare
type-variable occurrence is a sealed handle, while a generic container schema
can reconcile with a compatible concrete schema; the shared view then uses the
concrete schema and accepts concrete values. Distinct seals cannot share a
view. Sealed-handle opacity is a public FFI API property, not a sandbox or
security boundary for an unsandboxed Python companion. Nominal values are
rebuilt and ``json`` values are deep-copied. Each :class:`BoundaryScope`
carries one call's seals, recursive definitions, sealed-handle vault, and view
memo; when the call ends its views are revoked and its vault is released.
"""

from __future__ import annotations

import copy
import decimal
import operator
import weakref
from collections.abc import (
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    MutableMapping,
    MutableSequence,
    ValuesView,
)
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import (
    Callable,
    Generic,
    Protocol,
    SupportsIndex,
    TypeVar,
    assert_never,
    cast,
    overload,
    runtime_checkable,
)

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
    BoundaryVariantShape,
    ScalarKind,
)
from agm.agl.runtime.render import render_value
from agm.agl.runtime.serialize import value_to_json_obj
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
    """A value written through a boundary view violates its member schema."""


class BoundaryViewRevoked(RuntimeError):
    """A boundary view was used after its scope ended."""


class _SupportsLessThan(Protocol):
    """A value that can be ordered by Python's less-than operator."""

    def __lt__(self, other: object, /) -> bool: ...


@runtime_checkable
class _SupportsKeysAndGetItem(Protocol):
    """The mapping-like input shape accepted by ``MutableMapping.update``."""

    def keys(self) -> Iterable[str]: ...

    def __getitem__(self, key: str, /) -> object: ...


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

    Through the public FFI API, a Python companion may rearrange, count, and
    compare handles it receives, but cannot inspect or forge them: handles
    expose no value/seal attributes, and the public constructor rejects
    companion-created instances. Only the boundary encoder can mint an instance
    whose private id resolves in its call's boundary scope. This API property
    is not a sandbox or security boundary for arbitrary unsandboxed Python.

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

    This holder keeps a reference to the wrapped value — outliving the
    minting vault, which is released when the call ends — and derives the
    structural equality key lazily, on first ``__hash__``/``__eq__``, so a
    handle a companion retains still compares and renders.  A handle that
    merely crosses
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

    def release(self) -> None:
        """Drop every payload minted here, once no handle can be opened again."""
        self._payloads.clear()


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

_ContainerValue = TypeVar("_ContainerValue", ArrayValue, DictValue)
_View = TypeVar("_View", "AglArrayView", "AglDictView")


class BoundaryScope:
    """One call's seals, definitions, handle vault, compatible-view memo, and live flag."""

    def __init__(
        self,
        seals: Mapping[str, object] | None = None,
        defs: Mapping[str, BoundarySchema] | None = None,
    ) -> None:
        self.seals = seals if seals is not None else MappingProxyType({})
        self.defs = defs if defs else _NO_DEFS
        self._reconciled_defs: dict[str, BoundarySchema] = {}
        self._next_reconciled_def = 0
        self._vault = _HandleVault()
        self._array_views: dict[int, AglArrayView] = {}
        self._dict_views: dict[int, AglDictView] = {}
        self.live = True

    def array_view(self, value: ArrayValue, element_schema: BoundarySchema) -> AglArrayView:
        """Return this scope's view for *value*, or reconcile a compatible element schema."""
        return self._container_view(self._array_views, value, element_schema, AglArrayView)

    def dict_view(self, value: DictValue, value_schema: BoundarySchema) -> AglDictView:
        """Return this scope's view for *value*, or reconcile a compatible value schema."""
        return self._container_view(self._dict_views, value, value_schema, AglDictView)

    def _container_view(
        self,
        memo: dict[int, _View],
        value: _ContainerValue,
        member_schema: BoundarySchema,
        view_cls: Callable[[BoundaryScope, _ContainerValue, BoundarySchema], _View],
    ) -> _View:
        """Memoize one live view per container identity for the duration of this call."""
        member_schema = _resolve_boundary_ref(member_schema, self)
        marker = id(value)
        cached = memo.get(marker)
        if cached is not None:
            cached._reconcile_member_schema(member_schema)
            return cached
        view = view_cls(self, value, member_schema)
        memo[marker] = view
        return view

    def revoke(self) -> None:
        """End this call, making every view created for it unusable."""
        self.live = False
        self._array_views.clear()
        self._dict_views.clear()
        self._reconciled_defs.clear()
        # Releasing the vault drops this call's sealed AgL values, which a
        # companion-retained handle would otherwise pin for the process's life.
        self._vault.release()

    def begin_reconciled_schema(self) -> BoundaryRef:
        """Reserve a call-local reference for one recursively merged schema."""
        key = f"\x00reconciled_{self._next_reconciled_def}"
        self._next_reconciled_def += 1
        return BoundaryRef(key)

    def complete_reconciled_schema(self, ref: BoundaryRef, schema: BoundarySchema) -> None:
        """Make a completed recursively merged schema available through *ref*."""
        self._reconciled_defs[ref.key] = schema


_IteratorItem = TypeVar("_IteratorItem")


class _LiveIterator(Iterator[_IteratorItem]):
    """An iterator that validates its scope before every advance."""

    def __init__(self, scope: BoundaryScope, iterator: Iterator[_IteratorItem]) -> None:
        self._scope = scope
        self._iterator = iterator

    def __next__(self) -> _IteratorItem:
        _check_view_live(self._scope)
        return next(self._iterator)


class _AglArrayViewIterator(Iterator[object]):
    """A live array iterator that reads each not-yet-reached position lazily.

    *step* is ``1`` for forward iteration and ``-1`` for reverse.  Both walk
    until the next index leaves the array's current bounds, so elements a
    companion appends or removes mid-iteration are observed.
    """

    def __init__(self, view: AglArrayView, step: int) -> None:
        self._view = view
        self._step = step
        self._index = 0 if step > 0 else len(view._value.elements) - 1

    def __next__(self) -> object:
        _check_view_live(self._view._scope)
        if not 0 <= self._index < len(self._view._value.elements):
            raise StopIteration
        value = self._view[self._index]
        self._index += self._step
        return value


class _BoundaryContainerView(Generic[_ContainerValue]):
    """Shared scope, schema, and identity plumbing for a live container view.

    Subclasses add the Python container surface (sequence or mapping) and set
    :attr:`_member_label` to name their member position — the array element or
    the dict value — in an alias-conflict message.  Every public method checks
    liveness, so the helpers here do not repeat that check.
    """

    __slots__ = ("_scope", "_value", "_member_schema", "_encoded_member_schemas")

    _member_label: str
    _value: _ContainerValue

    def __init__(
        self, scope: BoundaryScope, value: _ContainerValue, member_schema: BoundarySchema
    ) -> None:
        self._scope = scope
        self._value = value
        self._member_schema = _resolve_boundary_ref(member_schema, scope)
        self._encoded_member_schemas = {self._member_schema}

    def _reconcile_member_schema(self, member_schema: BoundarySchema) -> None:
        """Record and adopt a compatible encoding schema without changing view identity."""
        _check_view_live(self._scope)
        reconciled = _reconcile_view_schema(self._member_schema, member_schema, self._scope)
        if reconciled is None:
            raise BoundaryViolation(f"aliased view has incompatible {self._member_label} schemas")
        self._encoded_member_schemas.add(member_schema)
        self._member_schema = reconciled

    def _accepts_decoded_member_schema(self, member_schema: BoundarySchema) -> bool:
        """Return whether this view may cross back at the requested schema."""
        _check_view_live(self._scope)
        return _view_schema_accepts_decode(
            self._member_schema, member_schema, self._encoded_member_schemas, self._scope
        )

    def _encode(self, value: Value) -> object:
        return _encode_boundary_value(self._member_schema, value, self._scope)

    def _decode(self, value: object) -> Value:
        return _decode_boundary_write(self._member_schema, value, self._scope)

    def __eq__(self, other: object) -> bool:
        _check_view_live(self._scope)
        return self is other

    def __hash__(self) -> int:
        _check_view_live(self._scope)
        return object.__hash__(self)

    def __repr__(self) -> str:
        _check_view_live(self._scope)
        return render_value(self._value)


class AglArrayView(_BoundaryContainerView[ArrayValue], MutableSequence[object]):
    """A live Python sequence view of an AgL array value."""

    __slots__ = ()

    _member_label = "array element"

    def __len__(self) -> int:
        _check_view_live(self._scope)
        return len(self._value.elements)

    @overload
    def __getitem__(self, index: SupportsIndex) -> object: ...

    @overload
    def __getitem__(self, index: slice[int | None, int | None, int | None]) -> list[object]: ...

    def __getitem__(
        self, index: SupportsIndex | slice[int | None, int | None, int | None]
    ) -> object | list[object]:
        _check_view_live(self._scope)
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
        _check_view_live(self._scope)
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
        _check_view_live(self._scope)
        if isinstance(index, SupportsIndex):
            del self._value.elements[operator.index(index)]
            return
        del self._value.elements[index]

    def insert(self, index: int, value: object) -> None:
        _check_view_live(self._scope)
        self._value.elements.insert(index, self._decode(value))

    def append(self, value: object) -> None:
        _check_view_live(self._scope)
        self._value.elements.append(self._decode(value))

    def clear(self) -> None:
        _check_view_live(self._scope)
        self._value.elements.clear()

    def extend(self, values: Iterable[object]) -> None:
        _check_view_live(self._scope)
        if values is self:
            self._value.elements.extend(self._value.elements.copy())
            return
        self._value.elements.extend(self._decode(item) for item in values)

    def pop(self, index: int = -1) -> object:
        _check_view_live(self._scope)
        return self._encode(self._value.elements.pop(index))

    def remove(self, value: object) -> None:
        _check_view_live(self._scope)
        for index, item in enumerate(self._value.elements):
            if self._encode(item) == value:
                del self._value.elements[index]
                return
        raise ValueError(f"{value!r} is not in list")

    def reverse(self) -> None:
        _check_view_live(self._scope)
        self._value.elements.reverse()

    def sort(self, *, key: Callable[[object], object] | None = None, reverse: bool = False) -> None:
        _check_view_live(self._scope)

        def sort_key(value: Value) -> _SupportsLessThan:
            encoded = self._encode(value)
            return cast(_SupportsLessThan, key(encoded) if key is not None else encoded)

        # Sorting into a fresh list leaves the AgL array untouched if a key
        # function or a comparison raises.
        self._value.elements[:] = sorted(self._value.elements, key=sort_key, reverse=reverse)

    def __iter__(self) -> Iterator[object]:
        _check_view_live(self._scope)
        return _AglArrayViewIterator(self, 1)

    def __reversed__(self) -> Iterator[object]:
        _check_view_live(self._scope)
        return _AglArrayViewIterator(self, -1)


_MISSING = object()


class AglDictView(_BoundaryContainerView[DictValue], MutableMapping[str, object]):
    """A live Python mapping view of an AgL dict value."""

    __slots__ = ()

    _member_label = "dict value"

    def __getitem__(self, key: str) -> object:
        _check_view_live(self._scope)
        self._check_key(key)
        return self._encode(self._value.entries[key])

    def __setitem__(self, key: str, value: object) -> None:
        _check_view_live(self._scope)
        self._check_key(key)
        self._value.entries[key] = self._decode(value)

    def __delitem__(self, key: str) -> None:
        _check_view_live(self._scope)
        self._check_key(key)
        del self._value.entries[key]

    def __contains__(self, key: object) -> bool:
        # ``Mapping.__contains__`` would probe through ``__getitem__``, so a
        # membership test would encode the value — minting a sealed handle or
        # a nested view that is thrown away immediately.
        _check_view_live(self._scope)
        self._check_key(key)
        return key in self._value.entries

    def __iter__(self) -> Iterator[str]:
        _check_view_live(self._scope)
        return _LiveIterator(self._scope, iter(tuple(self._value.entries)))

    def __reversed__(self) -> Iterator[str]:
        _check_view_live(self._scope)
        return _LiveIterator(self._scope, reversed(tuple(self._value.entries)))

    def keys(self) -> KeysView[str]:
        _check_view_live(self._scope)
        return KeysView(self)

    def items(self) -> ItemsView[str, object]:
        _check_view_live(self._scope)
        return ItemsView(self)

    def values(self) -> ValuesView[object]:
        _check_view_live(self._scope)
        return ValuesView(self)

    def __len__(self) -> int:
        _check_view_live(self._scope)
        return len(self._value.entries)

    def clear(self) -> None:
        _check_view_live(self._scope)
        self._value.entries.clear()

    def update(self, other: object = (), /, **kwargs: object) -> None:
        _check_view_live(self._scope)
        if isinstance(other, _SupportsKeysAndGetItem):
            for key in other.keys():
                self[key] = other[key]
        else:
            for key, value in cast(Iterable[tuple[str, object]], other):
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def pop(self, key: str, default: object = _MISSING) -> object:
        _check_view_live(self._scope)
        self._check_key(key)
        if default is _MISSING:
            return self._encode(self._value.entries.pop(key))
        if key not in self._value.entries:
            return default
        return self._encode(self._value.entries.pop(key))

    def popitem(self) -> tuple[str, object]:
        _check_view_live(self._scope)
        key, value = self._value.entries.popitem()
        return key, self._encode(value)

    @staticmethod
    def _check_key(key: object) -> None:
        if not isinstance(key, str):
            raise TypeError("dict view keys must be str")


def _reconcile_container_view_member_schema(
    existing: BoundarySchema, incoming: BoundarySchema
) -> BoundarySchema | None:
    """Choose one shared container-view member schema, or reject an incompatible pair.

    Used for both array element schemas and dict value schemas. Equal schemas
    retain their representation. A direct type-variable seal is less specific
    than a non-variable schema, so the latter represents both aliases.
    Distinct variable schemas and distinct non-variable schemas have no common
    representation.
    """
    if existing == incoming:
        return existing
    if isinstance(existing, BoundarySealVar):
        if isinstance(incoming, BoundarySealVar):
            return None
        return incoming
    if isinstance(incoming, BoundarySealVar):
        return existing
    return None


def _resolve_boundary_ref(schema: BoundarySchema, scope: BoundaryScope) -> BoundarySchema:
    """Resolve a finite ``BoundaryRef`` chain through this call's definitions."""
    if not isinstance(schema, BoundaryRef):
        return schema
    seen: set[str] = set()
    while isinstance(schema, BoundaryRef):
        key = schema.key
        if key in seen:
            raise BoundaryViolation(f"cyclic BoundaryRef chain at {key!r}")
        seen.add(key)
        try:
            schema = (
                scope._reconciled_defs[key] if key in scope._reconciled_defs else scope.defs[key]
            )
        except KeyError as exc:
            raise BoundaryViolation(f"unknown BoundaryRef {key!r}") from exc
    return schema


def _reconcile_view_schema(
    existing: BoundarySchema,
    incoming: BoundarySchema,
    scope: BoundaryScope,
    active: dict[tuple[int, int], BoundaryRef] | None = None,
) -> BoundarySchema | None:
    """Choose one shared container-view schema, or reject incompatible aliases.

    References resolve through the current scope at every structural position.
    A repeated recursive pair reuses a call-local reference to the enclosing
    merged schema, preserving a finite recursive representation.
    """
    existing = _resolve_boundary_ref(existing, scope)
    incoming = _resolve_boundary_ref(incoming, scope)
    if active is None:
        active = {}
    marker = (id(existing), id(incoming))
    if (recursive_ref := active.get(marker)) is not None:
        return recursive_ref
    try:
        reconciled = _reconcile_container_view_member_schema(existing, incoming)
        if reconciled is not None:
            return reconciled
        existing_is_less_specific = _view_schema_is_at_most_as_specific(existing, incoming, scope)
        incoming_is_less_specific = _view_schema_is_at_most_as_specific(incoming, existing, scope)
        if existing_is_less_specific and not incoming_is_less_specific:
            return incoming
        if incoming_is_less_specific and not existing_is_less_specific:
            return existing
        recursive_ref = scope.begin_reconciled_schema()
        active[marker] = recursive_ref
        if isinstance(existing, BoundaryArray) and isinstance(incoming, BoundaryArray):
            element = _reconcile_view_schema(existing.element, incoming.element, scope, active)
            reconciled = BoundaryArray(element) if element is not None else None
        elif isinstance(existing, BoundaryDict) and isinstance(incoming, BoundaryDict):
            value = _reconcile_view_schema(existing.value, incoming.value, scope, active)
            reconciled = BoundaryDict(value) if value is not None else None
        elif isinstance(existing, BoundaryRecord) and isinstance(incoming, BoundaryRecord):
            fields = _reconcile_boundary_fields(existing.fields, incoming.fields, scope, active)
            reconciled = (
                BoundaryRecord(existing.nominal, existing.display_name, fields)
                if existing.nominal == incoming.nominal
                and existing.display_name == incoming.display_name
                and fields is not None
                else None
            )
        elif isinstance(existing, BoundaryEnum) and isinstance(incoming, BoundaryEnum):
            variants = _reconcile_boundary_variants(
                existing.variants, incoming.variants, scope, active
            )
            reconciled = (
                BoundaryEnum(existing.nominal, existing.display_name, variants)
                if existing.nominal == incoming.nominal
                and existing.display_name == incoming.display_name
                and variants is not None
                else None
            )
        elif isinstance(existing, BoundaryException) and isinstance(incoming, BoundaryException):
            fields = _reconcile_boundary_fields(existing.fields, incoming.fields, scope, active)
            reconciled = (
                BoundaryException(existing.nominal, existing.display_name, fields)
                if existing.nominal == incoming.nominal
                and existing.display_name == incoming.display_name
                and fields is not None
                else None
            )
        else:
            reconciled = None
        if reconciled is not None:
            scope.complete_reconciled_schema(recursive_ref, reconciled)
        return reconciled
    finally:
        active.pop(marker, None)


def _view_schema_is_at_most_as_specific(
    existing: BoundarySchema,
    incoming: BoundarySchema,
    scope: BoundaryScope,
    active: set[tuple[int, int]] | None = None,
) -> bool:
    """Return whether *existing* can be represented by *incoming*.

    The relation is coinductive for recursive definitions: revisiting a pair
    is provisionally compatible while its enclosing structural comparison
    determines whether an actual generic/concrete difference exists.
    """
    existing = _resolve_boundary_ref(existing, scope)
    incoming = _resolve_boundary_ref(incoming, scope)
    if existing == incoming:
        return True
    if active is None:
        active = set()
    marker = (id(existing), id(incoming))
    if marker in active:
        return True
    active.add(marker)
    try:
        if isinstance(existing, BoundarySealVar):
            return not isinstance(incoming, BoundarySealVar)
        if isinstance(incoming, BoundarySealVar):
            return False
        if isinstance(existing, BoundaryArray) and isinstance(incoming, BoundaryArray):
            return _view_schema_is_at_most_as_specific(
                existing.element, incoming.element, scope, active
            )
        if isinstance(existing, BoundaryDict) and isinstance(incoming, BoundaryDict):
            return _view_schema_is_at_most_as_specific(
                existing.value, incoming.value, scope, active
            )
        if isinstance(existing, BoundaryRecord) and isinstance(incoming, BoundaryRecord):
            return (
                existing.nominal == incoming.nominal
                and existing.display_name == incoming.display_name
                and _boundary_fields_are_at_most_as_specific(
                    existing.fields, incoming.fields, scope, active
                )
            )
        if isinstance(existing, BoundaryEnum) and isinstance(incoming, BoundaryEnum):
            return (
                existing.nominal == incoming.nominal
                and existing.display_name == incoming.display_name
                and _boundary_variants_are_at_most_as_specific(
                    existing.variants, incoming.variants, scope, active
                )
            )
        if isinstance(existing, BoundaryException) and isinstance(incoming, BoundaryException):
            return (
                existing.nominal == incoming.nominal
                and existing.display_name == incoming.display_name
                and _boundary_fields_are_at_most_as_specific(
                    existing.fields, incoming.fields, scope, active
                )
            )
        return False
    finally:
        active.remove(marker)


def _boundary_fields_are_at_most_as_specific(
    existing: tuple[tuple[str, BoundarySchema], ...],
    incoming: tuple[tuple[str, BoundarySchema], ...],
    scope: BoundaryScope,
    active: set[tuple[int, int]],
) -> bool:
    """Compare matching nominal fields under the schema specificity relation."""
    return len(existing) == len(incoming) and all(
        existing_name == incoming_name
        and _view_schema_is_at_most_as_specific(existing_schema, incoming_schema, scope, active)
        for (existing_name, existing_schema), (incoming_name, incoming_schema) in zip(
            existing, incoming, strict=True
        )
    )


def _boundary_variants_are_at_most_as_specific(
    existing: tuple[BoundaryVariantShape, ...],
    incoming: tuple[BoundaryVariantShape, ...],
    scope: BoundaryScope,
    active: set[tuple[int, int]],
) -> bool:
    """Compare matching enum variants under the schema specificity relation."""
    return len(existing) == len(incoming) and all(
        existing_variant.name == incoming_variant.name
        and _boundary_fields_are_at_most_as_specific(
            existing_variant.fields, incoming_variant.fields, scope, active
        )
        for existing_variant, incoming_variant in zip(existing, incoming, strict=True)
    )


def _reconcile_boundary_fields(
    existing: tuple[tuple[str, BoundarySchema], ...],
    incoming: tuple[tuple[str, BoundarySchema], ...],
    scope: BoundaryScope,
    active: dict[tuple[int, int], BoundaryRef],
) -> tuple[tuple[str, BoundarySchema], ...] | None:
    """Reconcile ordered nominal fields only when their names align exactly."""
    if len(existing) != len(incoming):
        return None
    fields: list[tuple[str, BoundarySchema]] = []
    for (existing_name, existing_schema), (incoming_name, incoming_schema) in zip(
        existing, incoming, strict=True
    ):
        if existing_name != incoming_name:
            return None
        schema = _reconcile_view_schema(existing_schema, incoming_schema, scope, active)
        if schema is None:
            return None
        fields.append((existing_name, schema))
    return tuple(fields)


def _reconcile_boundary_variants(
    existing: tuple[BoundaryVariantShape, ...],
    incoming: tuple[BoundaryVariantShape, ...],
    scope: BoundaryScope,
    active: dict[tuple[int, int], BoundaryRef],
) -> tuple[BoundaryVariantShape, ...] | None:
    """Reconcile ordered enum variants only when names and fields align exactly."""
    if len(existing) != len(incoming):
        return None
    variants: list[BoundaryVariantShape] = []
    for existing_variant, incoming_variant in zip(existing, incoming, strict=True):
        if existing_variant.name != incoming_variant.name:
            return None
        fields = _reconcile_boundary_fields(
            existing_variant.fields, incoming_variant.fields, scope, active
        )
        if fields is None:
            return None
        variants.append(BoundaryVariantShape(existing_variant.name, fields))
    return tuple(variants)


def _view_schema_accepts_decode(
    view_schema: BoundarySchema,
    requested_schema: BoundarySchema,
    encoded_schemas: set[BoundarySchema],
    scope: BoundaryScope,
) -> bool:
    """Return whether an encoded view can safely return at *requested_schema*.

    The current resolved view schema is the normal exact-match rule. A view
    reconciled to a more concrete representation may also return at a less
    specific schema only if that exact schema was observed while encoding an
    alias in this scope.
    """
    if view_schema == requested_schema:
        return True
    return (
        requested_schema in encoded_schemas
        and _reconcile_view_schema(requested_schema, view_schema, scope) == view_schema
    )


def _check_view_live(scope: BoundaryScope) -> None:
    """Raise when a view's enclosing extern call has ended."""
    if not scope.live:
        raise BoundaryViewRevoked("boundary view is no longer live")


def _decode_boundary_write(schema: BoundarySchema, value: object, scope: BoundaryScope) -> Value:
    """Decode a view write, exposing schema violations as ``TypeError``."""
    _check_view_live(scope)
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
    """Encode one AgL value as a Python extern argument.

    Arrays and dicts become memoized live views in *scope*, so the companion
    observes mutations to their original AgL containers without a container
    copy. Other values retain the boundary's copy or sealed-handle rules.
    """
    return _encode_boundary_value(schema, value, scope if scope is not None else BoundaryScope())


def _encode_boundary_value(
    schema: BoundarySchema,
    value: Value,
    scope: BoundaryScope,
) -> object:
    """Encode a value using one call's boundary scope.

    Arrays and dicts become lazy live views, so encoding does not recursively
    walk container payloads. Nominal values are rebuilt and ``json`` values
    are deep-copied.
    """
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
            return scope.dict_view(value, val_schema)
        case BoundaryRecord(display_name=display_name, fields=fields):
            if not isinstance(value, RecordValue):
                raise BoundaryViolation(f"expected record {display_name!r}, got {_typename(value)}")
            return _encode_boundary_fields(fields, value.fields, scope)
        case BoundaryEnum(display_name=display_name, variants=variants):
            if not isinstance(value, EnumValue):
                raise BoundaryViolation(f"expected enum {display_name!r}, got {_typename(value)}")
            variant = next((item for item in variants if item.name == value.variant), None)
            if variant is None:
                raise BoundaryViolation(f"enum {display_name!r}: unknown variant {value.variant!r}")
            result: dict[str, object] = {"$case": value.variant}
            result.update(_encode_boundary_fields(variant.fields, value.fields, scope))
            return result
        case BoundaryException(display_name=display_name, fields=fields):
            if not isinstance(value, ExceptionValue):
                raise BoundaryViolation(
                    f"expected exception {display_name!r}, got {_typename(value)}"
                )
            return _encode_boundary_fields(fields, value.fields, scope)
        case BoundarySealVar(var=var):
            return scope._vault.make(value, scope.seals[var])
        case BoundaryRef(key=key):
            return _encode_boundary_value(scope.defs[key], value, scope)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _encode_boundary_fields(
    fields: tuple[tuple[str, BoundarySchema], ...],
    values: Mapping[str, Value],
    scope: BoundaryScope,
) -> dict[str, object]:
    """Encode a nominal payload's ordered fields through the boundary schema."""
    return {
        field_name: _encode_boundary_value(schema, values[field_name], scope)
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
    """Strictly decode one Python extern result against a boundary schema.

    A same-scope array or dict view with its exact resolved schema returns its
    original AgL container. A reconciled concrete view may additionally return
    at a generic schema observed while encoding an alias in the same scope.
    Exact built-in ``list`` and ``dict``
    values instead decode recursively into fresh AgL containers.
    """
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
            if isinstance(obj, AglArrayView):
                if obj._scope is not scope:
                    raise BoundaryViolation("array view belongs to another boundary scope")
                if not obj._accepts_decoded_member_schema(
                    _resolve_boundary_ref(elem_schema, scope)
                ):
                    raise BoundaryViolation("array view schema does not match its return schema")
                return obj._value
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
            if isinstance(obj, AglDictView):
                if obj._scope is not scope:
                    raise BoundaryViolation("dict view belongs to another boundary scope")
                if not obj._accepts_decoded_member_schema(_resolve_boundary_ref(val_schema, scope)):
                    raise BoundaryViolation("dict view schema does not match its return schema")
                return obj._value
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

    A cycle in a Python value arriving from a companion module is a boundary
    violation because the companion returned something undecodable. AgL
    containers leave through lazy views and require no encode-side walk.
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
