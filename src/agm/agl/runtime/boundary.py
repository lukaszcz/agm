"""Value-directed conversion at the AgL/Python extern boundary."""

from __future__ import annotations

import contextvars
import operator
from collections.abc import Callable, Iterable, Iterator, MutableMapping, MutableSequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, Self, SupportsIndex, cast, overload

from agm.agl.ir.ids import NominalId
from agm.agl.ir.program import NominalDescriptor, NominalKind
from agm.agl.runtime.render import render_value
from agm.agl.semantics.types import terminal_name
from agm.agl.semantics.values import (
    UNIT_VALUE,
    ArrayValue,
    BoolValue,
    DecimalValue,
    DictValue,
    ExceptionValue,
    IntValue,
    IrClosureValue,
    JsonValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
    value_equal,
)


class _SortKey(Protocol):
    """What :meth:`AglArrayView.sort` needs of a sort key: an ordering.

    Boundary representations are typed ``object``, so the keys a companion's
    ``sort`` produces carry no static ordering; this names the one operation
    the sort relies on, and ``TypeError`` from an unorderable pair surfaces to
    the companion exactly as it would for a plain Python list.
    """

    def __lt__(self, other: object, /) -> bool: ...


def _sort_key_of(pair: tuple[_SortKey, int]) -> _SortKey:
    """Project the key out of a ``(key, original position)`` sort pair."""
    return pair[0]


class BoundaryViolation(Exception):
    """A Python value has no representation in the value-directed boundary."""


class BoundaryTypeError(TypeError):
    """A value written through an AgL container view is unsupported."""


@dataclass(frozen=True, slots=True)
class AglJson:
    """The distinct Python representation of an AgL ``json`` value."""

    value: object


class AglException(Exception):
    """Carry an AgL exception value through companion Python frames.

    Companions construct this with an instance of a synthesized exception
    class. Callbacks construct it from their already-materialized
    :class:`ExceptionValue`. In either case, it carries the resulting AgL
    exception value rather than turning it into a Python exception.
    """

    def __init__(self, value: ExceptionValue | object) -> None:
        try:
            decoded = value if isinstance(value, ExceptionValue) else decode_boundary_value(value)
        except BoundaryViolation as exc:
            raise TypeError("AglException requires an AgL exception value") from exc
        if not isinstance(decoded, ExceptionValue):
            raise TypeError("AglException requires an AgL exception value")
        super().__init__(decoded.display_name)
        self.value = decoded


_FunctionEncoder = Callable[[IrClosureValue], object] | None

_IMMUTABLE_MESSAGE = "AgL nominal values are immutable"

# The closure encoder for the extent of one companion call. Encoding a closure
# needs an interpreter, which this evaluator-independent module never holds, so
# the extern-call chokepoint publishes one here rather than every crossing
# value carrying a copy: whatever a companion reaches -- an argument, a view it
# built itself, a closure nested in an array or dict -- encodes through the
# call it is running inside. Scoped to that call rather than captured, so
# nothing here outlives the interpreter that published it.
_ACTIVE_FUNCTION_ENCODER: contextvars.ContextVar["_FunctionEncoder"] = contextvars.ContextVar(
    "agl_active_function_encoder", default=None
)


@contextmanager
def active_function_encoder(encoder: "_FunctionEncoder") -> Iterator[None]:
    """Publish *encoder* as the ambient closure encoder for a call's extent."""
    token = _ACTIVE_FUNCTION_ENCODER.set(encoder)
    try:
        yield
    finally:
        _ACTIVE_FUNCTION_ENCODER.reset(token)


def _require_exact_fields(expected: tuple[str, ...], fields: dict[str, object]) -> None:
    """Check a companion-side construction names exactly the declared fields."""
    if fields.keys() != set(expected):
        raise TypeError(f"expected fields {expected!r}")


def _view_value(view: object) -> RecordValue:
    """Return the live ``RecordValue`` a record view is a window onto.

    Goes through ``object.__getattribute__`` because ``_AglRecordView``
    overrides ``__getattribute__`` to let a field named ``_agl_value`` shadow
    this slot for companion dot access.
    """
    return cast(RecordValue, object.__getattribute__(view, "_agl_value"))


class _AglNominal:
    """Base implementation shared by synthesized record, exception, and enum-variant classes.

    Field values live in one dict slot rather than one Python attribute per
    field, so construction cost stays flat regardless of nesting depth. Field
    access falls back to that dict through ``__getattr__``, which Python only
    consults once ordinary attribute lookup misses; a field whose name
    collides with an ``_agl_*`` class attribute is then reachable only
    through that attribute, not dot access.
    """

    __slots__ = ("_agl_values",)
    _agl_values: dict[str, object]
    _agl_nominal: NominalId
    _agl_kind: NominalKind
    _agl_fields: tuple[str, ...]
    _agl_descriptor: NominalDescriptor

    def __init__(self, **fields: object) -> None:
        _require_exact_fields(type(self)._agl_fields, fields)
        object.__setattr__(self, "_agl_values", fields)

    def __getattr__(self, name: str) -> object:
        try:
            return self._agl_values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(_IMMUTABLE_MESSAGE)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._agl_values == other._agl_values

    def __hash__(self) -> int:
        fields = type(self)._agl_fields
        return hash((type(self), tuple(self._agl_values[name] for name in fields)))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(...)"


class _AglRecordView:
    """Base implementation for synthesized live views of mutable record values.

    Unlike :class:`_AglNominal`, which snapshots its fields into a dict of
    already-encoded Python objects, a view keeps the underlying
    :class:`RecordValue` and encodes each field on read, so a companion sees
    AgL-side mutation and AgL sees the companion's writes.
    """

    __slots__ = ("_agl_value",)
    _agl_value: RecordValue
    _agl_nominal: NominalId
    _agl_kind: NominalKind
    _agl_fields: tuple[str, ...]
    _agl_descriptor: NominalDescriptor

    def __init__(self, **fields: object) -> None:
        expected = type(self)._agl_fields
        _require_exact_fields(expected, fields)
        descriptor = type(self)._agl_descriptor
        object.__setattr__(
            self,
            "_agl_value",
            RecordValue(
                descriptor.nominal,
                descriptor.display_name,
                {name: decode_boundary_value(fields[name]) for name in expected},
            ),
        )

    @classmethod
    def _from_value(cls, value: RecordValue) -> Self:
        view = object.__new__(cls)
        object.__setattr__(view, "_agl_value", value)
        return view

    def __getattribute__(self, name: str) -> object:
        # A declared field always wins over the storage slot, so a record whose
        # own field is named `_agl_value` stays reachable by dot access.
        if name in type(self)._agl_fields:
            return encode_boundary_value(_view_value(self).fields[name])
        return cast(object, object.__getattribute__(self, name))

    def __setattr__(self, name: str, value: object) -> None:
        if name not in type(self)._agl_descriptor.mutable_fields:
            raise AttributeError(_IMMUTABLE_MESSAGE)
        try:
            decoded = decode_boundary_value(value)
        except BoundaryViolation as exc:
            raise BoundaryTypeError(str(exc)) from exc
        _view_value(self).fields[name] = decoded

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return _view_value(self) == _view_value(other)

    def __repr__(self) -> str:
        return render_value(_view_value(self))


class _AglEnum:
    """Base class for one synthesized AgL enum: a pure namespace.

    Every member -- declared inline in the enum body or referenced by a
    qualified name -- is its own plain record class, set as an attribute of
    the enum class (``Step.Go``); the enum class itself carries no
    ``_agl_descriptor`` and is never instantiated, and no member class
    inherits from it (``issubclass(Step.Go, Step)`` is ``False``).
    """

    __slots__ = ()
    _agl_nominal: NominalId


def _nominal_class_name(descriptor: NominalDescriptor) -> str:
    return terminal_name(descriptor.display_name)


def _nominal_attrs(descriptor: NominalDescriptor) -> dict[str, object]:
    """Return the class-level attributes shared by every synthesized nominal shape."""
    return {
        "_agl_nominal": descriptor.nominal,
        "_agl_kind": descriptor.kind,
        "_agl_fields": descriptor.fields,
        "_agl_descriptor": descriptor,
        "__match_args__": descriptor.fields,
    }


def _class_namespace(attrs: dict[str, object]) -> dict[str, object]:
    """Return a synthesized class's namespace dict: no instance ``__dict__`` plus *attrs*."""
    namespace: dict[str, object] = {"__slots__": ()}
    namespace.update(attrs)
    return namespace


def _create_nominal(descriptor: NominalDescriptor, name: str) -> type[object]:
    """Create the synthesized class for one record or exception declaration.

    Called at most once per identity (see :func:`synthesize_nominal_classes`):
    an identity's layout is fixed at its declaration, so there is never a
    reason to build a second class for the same ``NominalId``.
    """
    # A record with a `var` field gets the live view base; every other nominal
    # shape snapshots. `_AglRecordView` declares `__eq__` and no `__hash__`, so
    # Python already makes it (and every subclass) unhashable.
    base = (
        _AglRecordView
        if descriptor.kind is NominalKind.RECORD and descriptor.mutable_fields
        else _AglNominal
    )
    return cast(
        type[object],
        type(name, (cast(type[object], base),), _class_namespace(_nominal_attrs(descriptor))),
    )


def _create_enum(
    descriptor: NominalDescriptor, name: str, variant_classes: dict[NominalId, type[object]]
) -> type[object]:
    """Create the synthesized enum class: a pure namespace over its members' own classes.

    Called at most once per identity, for the same reason as :func:`_create_nominal`.
    Every member -- inline or referenced -- already has its own plain record
    class by the time this runs (see :func:`synthesize_nominal_classes`); this
    only ever builds a class itself for a member whose own descriptor is
    reachable in neither *variant_classes* nor the current synthesis batch --
    a member record built standalone, elsewhere, from its own descriptor.
    """
    enum_cls = cast(
        type[object],
        type(
            name,
            (cast(type[object], _AglEnum),),
            _class_namespace({"_agl_nominal": descriptor.nominal}),
        ),
    )
    for variant in descriptor.variants:
        variant_cls = variant_classes.get(variant.member)
        if variant_cls is None:
            member_descriptor = NominalDescriptor(
                nominal=variant.member,
                module_id=descriptor.module_id,
                scope_path=(*descriptor.scope_path, descriptor.declared_name),
                declared_name=variant.name,
                kind=NominalKind.RECORD,
                fields=variant.fields,
            )
            variant_cls = _create_nominal(member_descriptor, variant.name)
            variant_classes[variant.member] = variant_cls
        setattr(enum_cls, variant.name, variant_cls)
    return enum_cls


#: Every identity a class has been synthesized for, keyed by ``NominalId``:
#: the encode direction's only registry, so it must outlive any single
#: caller's own class table. Nothing is ever removed. Within one program
#: image an entry can never go stale either, because a redeclaration always
#: mints a fresh identity of its own rather than reshaping an existing one.
#: Across images -- a fresh registry for a new program, or for a REPL after
#: ``:reset`` -- identity numbering restarts, so the same ``NominalId`` can
#: name a different declaration; the new image's own synthesis then takes
#: the entry over, matching the boundary's one-live-program-per-process
#: contract.
_NOMINAL_CLASSES: dict[NominalId, type[object]] = {}


def synthesize_nominal_classes(
    descriptors: Iterable[NominalDescriptor],
    existing: dict[NominalId, type[object]] | None = None,
) -> dict[NominalId, type[object]]:
    """Materialize the per-program Python classes used by extern companions, insert-only.

    A ``NominalId``'s layout is fixed at its declaration, so a class is
    synthesized for an identity at most once: when *existing* already holds
    a class for that identity, this reuses it unchanged rather than
    re-shaping it. A class a companion captured in a module global, a
    closure, or a default argument therefore keeps denoting the declaration
    it was captured from, permanently -- even after that declaration's name
    path is later reassigned to a fresh identity with a class of its own.
    The result also feeds the module-level registry
    :func:`encode_boundary_value` consults for the encode direction; see
    :data:`_NOMINAL_CLASSES` for its lifetime.
    """
    all_descriptors = tuple(descriptors)
    current = existing if existing is not None else {}
    result: dict[NominalId, type[object]] = {}
    pending_enums: list[NominalDescriptor] = []
    for descriptor in all_descriptors:
        reused = result.get(descriptor.nominal) or current.get(descriptor.nominal)
        if reused is not None:
            result[descriptor.nominal] = reused
            continue
        name = _nominal_class_name(descriptor)
        if descriptor.kind is NominalKind.ENUM:
            pending_enums.append(descriptor)
        else:
            result[descriptor.nominal] = _create_nominal(descriptor, name)
    for descriptor in pending_enums:
        variant_classes = {**current, **result}
        enum_cls = _create_enum(descriptor, _nominal_class_name(descriptor), variant_classes)
        result[descriptor.nominal] = enum_cls
        for variant in descriptor.variants:
            result.setdefault(variant.member, cast(type[object], getattr(enum_cls, variant.name)))
    _NOMINAL_CLASSES.update(result)
    return result


class AglArrayView(MutableSequence[object]):
    """A mutable, lazy Python view over one AgL array value."""

    __slots__ = ("_value",)

    def __init__(self, value: ArrayValue) -> None:
        self._value = value

    def __len__(self) -> int:
        return len(self._value.elements)

    @overload
    def __getitem__(self, index: SupportsIndex) -> object: ...

    @overload
    def __getitem__(self, index: slice[int | None, int | None, int | None]) -> list[object]: ...

    def __getitem__(
        self, index: SupportsIndex | slice[int | None, int | None, int | None]
    ) -> object | list[object]:
        if not isinstance(index, SupportsIndex):
            slice_index = index
            return [encode_boundary_value(value) for value in self._value.elements[slice_index]]
        return encode_boundary_value(self._value.elements[operator.index(index)])

    @overload
    def __setitem__(self, index: SupportsIndex, value: object) -> None: ...

    @overload
    def __setitem__(
        self, index: slice[int | None, int | None, int | None], value: Iterable[object]
    ) -> None: ...

    def __setitem__(
        self, index: SupportsIndex | slice[int | None, int | None, int | None], value: object
    ) -> None:
        try:
            if not isinstance(index, SupportsIndex):
                slice_index = index
                if not isinstance(value, Iterable):
                    raise TypeError("can only assign an iterable to an array slice")
                self._value.elements[slice_index] = [decode_boundary_value(item) for item in value]
            else:
                self._value.elements[operator.index(index)] = decode_boundary_value(value)
        except BoundaryViolation as exc:
            raise BoundaryTypeError(str(exc)) from exc

    def __delitem__(self, index: SupportsIndex | slice[int | None, int | None, int | None]) -> None:
        del self._value.elements[index]

    def insert(self, index: SupportsIndex, value: object) -> None:
        try:
            self._value.elements.insert(operator.index(index), decode_boundary_value(value))
        except BoundaryViolation as exc:
            raise BoundaryTypeError(str(exc)) from exc

    def extend(self, values: Iterable[object]) -> None:
        # Mirrors ``MutableSequence.extend``'s own self-aliasing guard: without
        # it, ``xs.extend(xs)`` (or ``xs += xs``, which routes through
        # ``__iadd__`` -> ``extend``) would append to the list it is still
        # iterating and never terminate. Decoding into a plain list first,
        # rather than feeding a generator straight to ``list.extend``, keeps a
        # decode failure part-way through from leaving *self* partially
        # mutated.
        if values is self:
            values = list(values)
        try:
            decoded = [decode_boundary_value(value) for value in values]
        except BoundaryViolation as exc:
            raise BoundaryTypeError(str(exc)) from exc
        self._value.elements.extend(decoded)

    def clear(self) -> None:
        self._value.elements.clear()

    def reverse(self) -> None:
        self._value.elements.reverse()

    def sort(self, *, key: Callable[[object], object] | None = None, reverse: bool = False) -> None:
        """Sort the underlying elements in place.

        *key*, when given, is applied to each element's encoded Python
        representation -- the shape a companion's own ``key=`` callable
        expects. Elements are paired with their computed key and original
        position, then reordered by a stable sort of those keys -- one encode
        pass per element, and no decode at all, unlike rebuilding every
        element through an encode-then-decode round trip.
        """
        elements = self._value.elements
        pairs = [
            (
                cast(
                    "_SortKey",
                    encode_boundary_value(value)
                    if key is None
                    else key(encode_boundary_value(value)),
                ),
                index,
            )
            for index, value in enumerate(elements)
        ]
        pairs.sort(key=_sort_key_of, reverse=reverse)
        elements[:] = [elements[index] for _, index in pairs]

    def __iter__(self) -> Iterator[object]:
        for value in self._value.elements:
            yield encode_boundary_value(value)

    def __contains__(self, value: object) -> bool:
        try:
            self.index(value)
        except ValueError:
            return False
        return True

    def index(self, value: object, start: int = 0, stop: int | None = None) -> int:
        """Find *value* using the language's equality relation.

        A companion sees encoded values, but array search must agree with AgL
        ``in`` rather than Python view identity or Python's JSON bool/number
        comparison. This intentionally mirrors ``list.index`` bounds.
        """
        try:
            probe = decode_boundary_value(value)
        except BoundaryViolation:
            raise ValueError(f"{value!r} is not in array") from None
        length = len(self._value.elements)
        lower = operator.index(start)
        upper = length if stop is None else operator.index(stop)
        if lower < 0:
            lower = max(lower + length, 0)
        else:
            lower = min(lower, length)
        if upper < 0:
            upper = max(upper + length, 0)
        else:
            upper = min(upper, length)
        for position in range(lower, upper):
            if value_equal(probe, self._value.elements[position]):
                return position
        raise ValueError(f"{value!r} is not in array")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AglArrayView) and self._value is other._value

    def __hash__(self) -> int:
        return id(self._value)

    def __repr__(self) -> str:
        return render_value(self._value)


class AglDictView(MutableMapping[str, object]):
    """A mutable, lazy Python view over one AgL dict value."""

    __slots__ = ("_value",)

    def __init__(self, value: DictValue) -> None:
        self._value = value

    def __getitem__(self, key: str) -> object:
        return encode_boundary_value(self._value.entries[key])

    def __setitem__(self, key: str, value: object) -> None:
        if not isinstance(key, str):
            raise TypeError("AgL dict keys must be str")
        try:
            self._value.entries[key] = decode_boundary_value(value)
        except BoundaryViolation as exc:
            raise BoundaryTypeError(str(exc)) from exc

    def __delitem__(self, key: str) -> None:
        del self._value.entries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._value.entries)

    def __len__(self) -> int:
        return len(self._value.entries)

    def clear(self) -> None:
        self._value.entries.clear()

    def popitem(self) -> tuple[str, object]:
        key, value = self._value.entries.popitem()
        return key, encode_boundary_value(value)

    def __contains__(self, key: object) -> bool:
        return key in self._value.entries

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AglDictView) and self._value is other._value

    def __hash__(self) -> int:
        return id(self._value)

    def __repr__(self) -> str:
        return render_value(self._value)


def encode_boundary_value(value: Value) -> object:
    """Encode an AgL value by its runtime subclass.

    An ``array``/``dict`` crosses as a live view (mutating it mutates the AgL
    value). A closure needs an interpreter to become a callable proxy, which
    the active extern call supplies through :func:`active_function_encoder`;
    the runtime boundary itself remains evaluator-independent. A ``json``
    payload crosses uncopied and unchecked: the companion is trusted to treat
    what it receives as read-only.
    """
    if isinstance(value, UnitValue):
        return None
    if isinstance(value, BoolValue):
        return value.value
    if isinstance(value, IntValue):
        return value.value
    if isinstance(value, DecimalValue):
        return value.value
    if isinstance(value, TextValue):
        return value.value
    if isinstance(value, JsonValue):
        return AglJson(value.raw)
    if isinstance(value, ArrayValue):
        return AglArrayView(value)
    if isinstance(value, DictValue):
        return AglDictView(value)
    if isinstance(value, IrClosureValue):
        encoder = _ACTIVE_FUNCTION_ENCODER.get()
        if encoder is None:
            raise BoundaryViolation("cannot encode an AgL function without an interpreter")
        return encoder(value)
    if isinstance(value, (RecordValue, ExceptionValue)):
        try:
            cls = _NOMINAL_CLASSES[value.nominal]
        except KeyError as exc:
            raise BoundaryViolation(f"unknown AgL nominal {value.display_name!r}") from exc
        if isinstance(value, RecordValue) and issubclass(cls, _AglRecordView):
            return cls._from_value(value)
        fields = {name: encode_boundary_value(field) for name, field in value.fields.items()}
        return cls(**fields)
    raise BoundaryViolation(f"cannot encode {type(value).__name__}")


def decode_boundary_value(obj: object) -> Value:
    """Decode a Python boundary representation by its concrete type.

    A synthesized nominal instance carries its own descriptor on its class
    (``_agl_descriptor``), so decoding needs no registry lookup: it resolves
    a nominal purely from ``type(obj)``, which is why a value built at
    companion import time, on a worker thread, or retained past the call
    that produced it all decode the same way.
    """
    if obj is None:
        return UNIT_VALUE
    if isinstance(obj, bool):
        return BoolValue(obj)
    if isinstance(obj, int):
        return IntValue(obj)
    if isinstance(obj, Decimal):
        return DecimalValue(obj)
    if isinstance(obj, str):
        return TextValue(obj)
    if isinstance(obj, AglJson):
        return JsonValue(obj.value)
    if isinstance(obj, AglArrayView):
        return obj._value
    if isinstance(obj, AglDictView):
        return obj._value
    # Imported lazily because externs depends on this module for normal
    # boundary conversion. Only proxies minted by the evaluator carry an AgL
    # closure; arbitrary Python callables remain unsupported.
    from agm.agl.runtime.externs import AglCallableProxy

    if isinstance(obj, AglCallableProxy):
        return obj._closure
    descriptor = cast(object, getattr(type(obj), "_agl_descriptor", None))
    if isinstance(descriptor, NominalDescriptor):
        if isinstance(obj, _AglRecordView):
            return _view_value(obj)
        nominal_obj = cast(_AglNominal, obj)
        fields = {
            name: decode_boundary_value(nominal_obj._agl_values[name])
            for name in type(nominal_obj)._agl_fields
        }
        if descriptor.kind is NominalKind.RECORD:
            return RecordValue(descriptor.nominal, descriptor.display_name, fields)
        # A descriptor reaches here only from :func:`_create_nominal`, which
        # builds a class for a record or an exception declaration; an enum
        # class is a pure namespace carrying no descriptor of its own.
        return ExceptionValue(descriptor.nominal, descriptor.display_name, fields)
    raise BoundaryViolation(f"unsupported Python extern value {type(obj).__name__}")
