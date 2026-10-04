"""Dictionary operations for ``std/dict``."""

from __future__ import annotations

from typing import NoReturn

from agl import array, nominals, option_none, option_some
from agl import dict as agl_dict

from agm.agl.runtime.boundary import raise_key_error

DuplicateKeyError = nominals.std.errors.DuplicateKeyError
DuplicateKeys = nominals.std.dict.DuplicateKeys
KeyError = nominals.std.errors.KeyError
Pair = nominals.std.pair.Pair


def _key_error(key: object) -> NoReturn:
    raise_key_error(KeyError, "dictionary key not found", key)


def size(values: object) -> int:
    return len(values)


def get(values: object, key: object) -> object:
    if key not in values:
        _key_error(key)
    return values[key]


def get_option(values: object, key: object) -> object:
    return option_some(values[key]) if key in values else option_none()


def remove(values: object, key: object) -> object:
    if key not in values:
        _key_error(key)
    return values.pop(key)


def remove_option(values: object, key: object) -> object:
    return option_some(values.pop(key)) if key in values else option_none()


def set(values: object, key: object, value: object) -> None:
    values[key] = value


def contains(values: object, key: object) -> bool:
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
    merged = agl_dict({})
    for source in (values, other):
        for key, value in source.items():
            merged[key] = value
    return merged


def merge_in_place(values: object, other: object) -> None:
    values.update(other)


def map_values(values: object, function: object) -> object:
    result = agl_dict({})
    for key, value in values.items():
        result[key] = function(value)
    return result


def select(values: object, predicate: object) -> object:
    result = agl_dict({})
    for key, value in values.items():
        if predicate(key, value):
            result[key] = value
    return result


def select_in_place(values: object, predicate: object) -> None:
    for key in list(values):
        if not predicate(key, values[key]):
            del values[key]


def each(values: object, function: object) -> None:
    for key, value in values.items():
        function(key, value)


def from_entries(values: object, on_duplicate: object) -> object:
    result = agl_dict({})
    for pair in values:
        if pair.first in result:
            if isinstance(on_duplicate, DuplicateKeys.Raise):
                raise_key_error(DuplicateKeyError, "duplicate dictionary key", pair.first)
            if isinstance(on_duplicate, DuplicateKeys.KeepFirst):
                continue
        result[pair.first] = pair.second
    return result


__all__ = [
    "clear",
    "contains",
    "each",
    "entries",
    "select",
    "select_in_place",
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
