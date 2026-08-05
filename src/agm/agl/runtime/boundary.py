"""Value-directed conversion at the AgL/Python extern boundary."""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Iterator, MutableMapping, MutableSequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, SupportsIndex, cast, overload

from agm.agl.ir.ids import NominalId
from agm.agl.ir.program import NominalDescriptor, NominalKind
from agm.agl.runtime.render import render_value
from agm.agl.semantics.values import (
    UNIT_VALUE,
    ArrayValue,
    BoolValue,
    DecimalValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    JsonValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
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
    _agl_variant: str

    def __init__(self, **fields: object) -> None:
        expected = type(self)._agl_fields
        if set(fields) != set(expected):
            raise TypeError(f"expected fields {expected!r}")
        object.__setattr__(self, "_agl_values", fields)

    def __getattr__(self, name: str) -> object:
        try:
            return self._agl_values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("AgL nominal values are immutable")

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._agl_values == other._agl_values

    def __hash__(self) -> int:
        fields = type(self)._agl_fields
        return hash((type(self), tuple(self._agl_values[name] for name in fields)))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(...)"


class _AglEnum:
    """Base class for one synthesized AgL enum."""

    __slots__ = ()
    _agl_nominal: NominalId


def _nominal_class_name(descriptor: NominalDescriptor) -> str:
    return descriptor.display_name.rsplit("::", maxsplit=1)[-1]


def _nominal_attrs(descriptor: NominalDescriptor, fields: tuple[str, ...]) -> dict[str, object]:
    """Return the class-level attributes shared by every synthesized nominal shape."""
    return {
        "_agl_nominal": descriptor.nominal,
        "_agl_kind": descriptor.kind,
        "_agl_fields": fields,
        "_agl_descriptor": descriptor,
        "__match_args__": fields,
    }


def _class_namespace(attrs: dict[str, object]) -> dict[str, object]:
    """Return a synthesized class's namespace dict: no instance ``__dict__`` plus *attrs*."""
    namespace: dict[str, object] = {"__slots__": ()}
    namespace.update(attrs)
    return namespace


def _class_members(cls: type[object]) -> dict[str, object]:
    """Return *cls*'s own namespace as a concretely typed mapping."""
    return cast(dict[str, object], vars(cls))


def _is_agl_nominal_class(value: object) -> bool:
    """Return whether *value* is itself a synthesized ``_AglNominal`` subclass."""
    try:
        return issubclass(cast("type[object]", value), _AglNominal)
    except TypeError:
        return False


def _update_or_create_nominal(
    descriptor: NominalDescriptor, name: str, existing_cls: type[object] | None
) -> type[object]:
    """Reuse *existing_cls* in place when it is already a plain nominal class.

    Reuse keeps every companion-held reference to the class -- a module
    global, a closure, a default argument -- valid across a redeclaration
    (for example, across a REPL ``:reset``) instead of stranding it on a
    stale class object.
    """
    attrs = _nominal_attrs(descriptor, descriptor.fields)
    if existing_cls is not None and not issubclass(existing_cls, _AglEnum):
        for attr_name, value in attrs.items():
            setattr(existing_cls, attr_name, value)
        return existing_cls
    return cast(
        type[object],
        type(name, (cast(type[object], _AglNominal),), _class_namespace(attrs)),
    )


def _update_or_create_enum(
    descriptor: NominalDescriptor, name: str, existing_cls: type[object] | None
) -> type[object]:
    """Reuse *existing_cls* and its variant classes in place when possible.

    Each surviving variant name is updated in place for the same reason
    :func:`_update_or_create_nominal` reuses a plain nominal class; a variant
    absent from *descriptor* is removed from the enum class entirely.
    """
    enum_cls: type[object]
    if existing_cls is not None and issubclass(existing_cls, _AglEnum):
        enum_cls = existing_cls
    else:
        enum_cls = cast(
            type[object],
            type(
                name,
                (cast(type[object], _AglEnum),),
                _class_namespace({"_agl_nominal": descriptor.nominal}),
            ),
        )

    live_variants: set[str] = set()
    for variant in descriptor.variants:
        live_variants.add(variant.name)
        attrs = {**_nominal_attrs(descriptor, variant.fields), "_agl_variant": variant.name}
        existing_variant: object | None = _class_members(enum_cls).get(variant.name)
        if _is_agl_nominal_class(existing_variant):
            for attr_name, value in attrs.items():
                setattr(existing_variant, attr_name, value)
        else:
            variant_cls = type(
                variant.name,
                (cast(type[object], _AglNominal), enum_cls),
                _class_namespace(attrs),
            )
            setattr(enum_cls, variant.name, variant_cls)

    stale_variants = [
        attr_name
        for attr_name, value in _class_members(enum_cls).items()
        if attr_name not in live_variants and _is_agl_nominal_class(value)
    ]
    for stale_name in stale_variants:
        delattr(enum_cls, stale_name)

    return enum_cls


_NOMINAL_CLASSES: dict[NominalId, type[object]] = {}


def synthesize_nominal_classes(
    descriptors: Iterable[NominalDescriptor],
    existing: dict[NominalId, type[object]] | None = None,
) -> dict[NominalId, type[object]]:
    """Materialize or update the per-program Python classes used by extern companions.

    A ``NominalId`` keeps the same class object across repeated calls: when
    *existing* already holds a class for that identity, this updates its
    shape in place instead of replacing it, so a class captured by a
    companion in a module global, a closure, or a default argument stays
    valid even after the nominal it names is redeclared. The result also
    becomes the module-level registry :func:`encode_boundary_value` consults
    for the encode direction; nothing ever removes an entry from it, matching
    the boundary's one-program-load-per-process contract.
    """
    current = existing if existing is not None else {}
    updated: dict[NominalId, type[object]] = {}
    for descriptor in descriptors:
        name = _nominal_class_name(descriptor)
        if descriptor.kind is NominalKind.ENUM:
            updated[descriptor.nominal] = _update_or_create_enum(
                descriptor, name, current.get(descriptor.nominal)
            )
        else:
            updated[descriptor.nominal] = _update_or_create_nominal(
                descriptor, name, current.get(descriptor.nominal)
            )
    _NOMINAL_CLASSES.update(updated)
    return updated


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
            probe = decode_boundary_value(value)
        except BoundaryViolation:
            return False
        return any(probe == item for item in self._value.elements)

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
    value).  A ``json`` payload crosses uncopied and unchecked: the companion
    is trusted to treat what it receives as read-only.
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
    if isinstance(value, (RecordValue, EnumValue, ExceptionValue)):
        try:
            cls = _NOMINAL_CLASSES[value.nominal]
        except KeyError as exc:
            raise BoundaryViolation(f"unknown AgL nominal {value.nominal.display_name!r}") from exc
        fields = {name: encode_boundary_value(field) for name, field in value.fields.items()}
        if isinstance(value, EnumValue):
            cls = getattr(cls, value.variant)
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
    descriptor = cast(object, getattr(type(obj), "_agl_descriptor", None))
    if isinstance(descriptor, NominalDescriptor):
        nominal_obj = cast(_AglNominal, obj)
        fields = {
            name: decode_boundary_value(nominal_obj._agl_values[name])
            for name in type(nominal_obj)._agl_fields
        }
        if descriptor.kind is NominalKind.RECORD:
            return RecordValue(descriptor.nominal, descriptor.display_name, fields)
        if descriptor.kind is NominalKind.EXCEPTION:
            return ExceptionValue(descriptor.nominal, descriptor.display_name, fields)
        return EnumValue(
            descriptor.nominal, descriptor.display_name, type(nominal_obj)._agl_variant, fields
        )
    raise BoundaryViolation(f"unsupported Python extern value {type(obj).__name__}")
