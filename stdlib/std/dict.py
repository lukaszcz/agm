"""Dictionary operations for ``std/dict``."""

from __future__ import annotations

from agl import AglException, array, nominals
from agl import dict as agl_dict

KeyError = nominals.std.prelude.KeyError
Option = nominals.std.option.Option
Pair = nominals.std.pair.Pair


def _none() -> object:
    return getattr(Option, "None")()


def _key_error(key: str) -> None:
    raise AglException(KeyError(message="dictionary key not found", key=key))


def size(values: object) -> int:
    return len(values)


def get(values: object, key: str) -> object:
    if key not in values:
        _key_error(key)
    return values[key]


def get_option(values: object, key: str) -> object:
    return Option.Some(value=values[key]) if key in values else _none()


def remove(values: object, key: str) -> object:
    if key not in values:
        _key_error(key)
    return values.pop(key)


def remove_option(values: object, key: str) -> object:
    return Option.Some(value=values.pop(key)) if key in values else _none()


def set(values: object, key: str, value: object) -> None:
    values[key] = value


def contains(values: object, key: str) -> bool:
    return key in values


def clear(values: object) -> None:
    values.clear()


def keys(values: object) -> object:
    return array(list(values))


def values(values_: object) -> object:
    return array([value for value in values_.values()])


def entries(values_: object) -> object:
    return array([Pair(first=key, second=value) for key, value in values_.items()])


def merge(values: object, other: object) -> object:
    merged = agl_dict({key: value for key, value in values.items()})
    merged.update(other)
    return merged


def merge_in_place(values: object, other: object) -> None:
    values.update(other)


def map_values(values: object, function: object) -> object:
    return agl_dict({key: function(value) for key, value in values.items()})


def filter(values: object, predicate: object) -> object:
    return agl_dict({key: value for key, value in values.items() if predicate(key, value)})


def filter_in_place(values: object, predicate: object) -> None:
    for key in list(values):
        if not predicate(key, values[key]):
            del values[key]


def each(values: object, function: object) -> None:
    for key, value in values.items():
        function(key, value)


def from_entries(values: object) -> object:
    return agl_dict({value.first: value.second for value in values})


__all__ = [
    "clear",
    "contains",
    "each",
    "entries",
    "filter",
    "filter_in_place",
    "from_entries",
    "get",
    "get_option",
    "keys",
    "map_values",
    "merge",
    "merge_in_place",
    "remove",
    "remove_option",
    "set",
    "size",
    "values",
]
