"""Value-directed conversion at the AgL/Python extern boundary."""

from __future__ import annotations

import operator
from collections.abc import Iterable, Iterator, MutableMapping, MutableSequence
from contextvars import ContextVar, Token
from dataclasses import dataclass
from decimal import Decimal
from typing import SupportsIndex, cast, overload

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


class BoundaryViolation(Exception):
    """A Python value has no representation in the value-directed boundary."""


class BoundaryTypeError(TypeError):
    """A value written through an AgL container view is unsupported."""


@dataclass(frozen=True, slots=True)
class AglJson:
    """The distinct Python representation of an AgL ``json`` value."""

    value: object


class _AglNominal:
    """Base implementation shared by synthesized record and exception classes."""

    __slots__ = ()
    _agl_nominal: NominalId
    _agl_kind: NominalKind
    _agl_fields: tuple[str, ...]
    _agl_variant: str

    def __init__(self, **fields: object) -> None:
        expected = self._agl_fields
        if set(fields) != set(expected):
            raise TypeError(f"expected fields {expected!r}")
        for name in expected:
            object.__setattr__(
                self, name, encode_boundary_value(decode_boundary_value(fields[name]))
            )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("AgL nominal values are immutable")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(...)"


class _AglEnum:
    """Base class for one synthesized AgL enum."""

    __slots__ = ()
    _agl_nominal: NominalId


def _nominal_class_name(descriptor: NominalDescriptor) -> str:
    return descriptor.display_name.rsplit("::", maxsplit=1)[-1]


def synthesize_nominal_classes(
    descriptors: Iterable[NominalDescriptor],
) -> tuple[dict[NominalId, type[object]], dict[type[object], NominalDescriptor]]:
    """Build the per-program Python classes used by extern companions."""
    by_nominal: dict[NominalId, type[object]] = {}
    by_type: dict[type[object], NominalDescriptor] = {}
    for descriptor in descriptors:
        name = _nominal_class_name(descriptor)
        if descriptor.kind is NominalKind.ENUM:
            enum_attrs: dict[str, object] = {"__slots__": (), "_agl_nominal": descriptor.nominal}
            enum_cls = cast(type[object], type(name, (cast(type[object], _AglEnum),), enum_attrs))
            by_nominal[descriptor.nominal] = enum_cls
            by_type[enum_cls] = descriptor
            for variant in descriptor.variants:
                variant_attrs: dict[str, object] = {
                    "__slots__": variant.fields,
                    "__match_args__": variant.fields,
                    "_agl_nominal": descriptor.nominal,
                    "_agl_kind": NominalKind.ENUM,
                    "_agl_fields": variant.fields,
                    "_agl_variant": variant.name,
                }
                variant_cls = cast(
                    type[object],
                    type(
                        variant.name,
                        (cast(type[object], _AglNominal), enum_cls),
                        variant_attrs,
                    ),
                )
                setattr(enum_cls, variant.name, variant_cls)
                by_type[variant_cls] = descriptor
            continue
        class_attrs: dict[str, object] = {
            "__slots__": descriptor.fields,
            "__match_args__": descriptor.fields,
            "_agl_nominal": descriptor.nominal,
            "_agl_kind": descriptor.kind,
            "_agl_fields": descriptor.fields,
        }
        cls = cast(
            type[object],
            type(
                name,
                (cast(type[object], _AglNominal),),
                class_attrs,
            ),
        )
        by_nominal[descriptor.nominal] = cls
        by_type[cls] = descriptor
    return by_nominal, by_type


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

    def append(self, value: object) -> None:
        self.insert(len(self), value)

    def extend(self, values: Iterable[object]) -> None:
        try:
            self._value.elements.extend(decode_boundary_value(value) for value in values)
        except BoundaryViolation as exc:
            raise BoundaryTypeError(str(exc)) from exc

    def clear(self) -> None:
        self._value.elements.clear()

    def reverse(self) -> None:
        self._value.elements.reverse()

    def sort(self, *, reverse: bool = False) -> None:
        values = [encode_boundary_value(value) for value in self._value.elements]
        values.sort(reverse=reverse)
        self._value.elements[:] = [decode_boundary_value(value) for value in values]

    def __iter__(self) -> Iterator[object]:
        for value in self._value.elements:
            yield encode_boundary_value(value)

    def __contains__(self, value: object) -> bool:
        return any(encode_boundary_value(item) == value for item in self._value.elements)

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


_NOMINAL_CLASSES: ContextVar[dict[NominalId, type[object]]] = ContextVar(
    "agl_nominal_classes", default={}
)
_NOMINAL_DESCRIPTORS: ContextVar[dict[type[object], NominalDescriptor]] = ContextVar(
    "agl_nominal_descriptors", default={}
)


def push_nominal_classes(
    classes: dict[NominalId, type[object]], descriptors: dict[type[object], NominalDescriptor]
) -> tuple[Token[dict[NominalId, type[object]]], Token[dict[type[object], NominalDescriptor]]]:
    """Install one registry's nominal mapping for the current extern call."""
    return _NOMINAL_CLASSES.set(classes), _NOMINAL_DESCRIPTORS.set(descriptors)


def pop_nominal_classes(
    tokens: tuple[
        Token[dict[NominalId, type[object]]], Token[dict[type[object], NominalDescriptor]]
    ],
) -> None:
    """Restore the nominal mapping that preceded an extern call."""
    classes, descriptors = tokens
    _NOMINAL_CLASSES.reset(classes)
    _NOMINAL_DESCRIPTORS.reset(descriptors)


def encode_boundary_value(value: Value) -> object:
    """Encode an AgL value by its runtime subclass."""
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
            cls = _NOMINAL_CLASSES.get()[value.nominal]
        except KeyError as exc:
            raise BoundaryViolation(f"unknown AgL nominal {value.nominal.display_name!r}") from exc
        fields = {name: encode_boundary_value(field) for name, field in value.fields.items()}
        if isinstance(value, EnumValue):
            cls = getattr(cls, value.variant)
        return cls(**fields)
    raise BoundaryViolation(f"cannot encode {type(value).__name__}")


def decode_boundary_value(obj: object) -> Value:
    """Decode a Python boundary representation by its concrete type."""
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
        _ensure_acyclic_python_containers(obj.value)
        return JsonValue(obj.value)
    if isinstance(obj, AglArrayView):
        return obj._value
    if isinstance(obj, AglDictView):
        return obj._value
    descriptor = _NOMINAL_DESCRIPTORS.get().get(type(obj))
    if descriptor is not None:
        nominal_obj = cast(_AglNominal, obj)
        fields = {
            name: decode_boundary_value(cast(object, object.__getattribute__(nominal_obj, name)))
            for name in nominal_obj._agl_fields
        }
        if descriptor.kind is NominalKind.RECORD:
            return RecordValue(descriptor.nominal, descriptor.display_name, fields)
        if descriptor.kind is NominalKind.EXCEPTION:
            return ExceptionValue(descriptor.nominal, descriptor.display_name, fields)
        return EnumValue(
            descriptor.nominal, descriptor.display_name, nominal_obj._agl_variant, fields
        )
    raise BoundaryViolation(f"unsupported Python extern value {type(obj).__name__}")


def _ensure_acyclic_python_containers(value: object, active: set[int] | None = None) -> None:
    """Reject a cyclic JSON wrapper without imposing a JSON-shape schema."""
    if not isinstance(value, (list, dict)):
        return
    seen = active if active is not None else set()
    marker = id(value)
    if marker in seen:
        raise BoundaryViolation("cyclic Python return value")
    seen.add(marker)
    try:
        items = value if isinstance(value, list) else value.values()
        for item in items:
            _ensure_acyclic_python_containers(item, seen)
    finally:
        seen.remove(marker)
