"""Array operations for ``std/array``."""

from __future__ import annotations

import builtins
from functools import cmp_to_key
from typing import Protocol

from agl import AglException, array, nominals

IndexError = nominals.std.errors.IndexError
Option = nominals.std.option.Option
Pair = nominals.std.pair.Pair


class _Comparator(Protocol):
    def __call__(self, left: object, right: object, /) -> int: ...


def _index_error(index: int, length: int, message: str = "array index out of range") -> None:
    raise AglException(IndexError(message=message, index=index, length=length))


def _element_at(values: object, index: int) -> object:
    length = len(values)
    if not -length <= index < length:
        _index_error(index, length)
    return values[index]


def _none() -> object:
    return getattr(Option, "None")()


def _option(value: object | None, found: bool) -> object:
    return Option.Some(value=value) if found else _none()


def size(values: object) -> int:
    return len(values)


def first(values: object) -> object:
    return _element_at(values, 0)


def first_option(values: object) -> object:
    return _option(values[0] if values else None, bool(values))


def last(values: object) -> object:
    return _element_at(values, -1)


def last_option(values: object) -> object:
    return _option(values[-1] if values else None, bool(values))


def push(values: object, value: object) -> None:
    values.append(value)


def _insert_index(values: object, index: int) -> int:
    length = len(values)
    if not -length <= index <= length:
        _index_error(index, length)
    return index


def insert(values: object, index: int, value: object) -> None:
    values.insert(_insert_index(values, index), value)


def pop(values: object) -> object:
    if not values:
        _index_error(-1, 0)
    return values.pop()


def pop_option(values: object) -> object:
    if not values:
        return _none()
    return Option.Some(value=values.pop())


def remove_at(values: object, index: int) -> object:
    value = _element_at(values, index)
    del values[index]
    return value


def clear(values: object) -> None:
    values.clear()


def contains(values: object, value: object) -> bool:
    return value in values


def _search(values: object, value: object) -> int:
    """Return the first index of *value*, or ``-1`` when it is absent."""
    try:
        return values.index(value)
    except ValueError:
        return -1


def index_of(values: object, value: object) -> int:
    index = _search(values, value)
    if index < 0:
        _index_error(-1, len(values), "array value not found")
    return index


def index_of_option(values: object, value: object) -> object:
    index = _search(values, value)
    return _option(index, index >= 0)


def count(values: object, predicate: object) -> int:
    return sum(predicate(value) for value in values)


def map(values: object, function: object) -> object:
    return array([function(value) for value in values])


def map_in_place(values: object, function: object) -> None:
    values[:] = [function(value) for value in values]


def select(values: object, predicate: object) -> object:
    return array([value for value in values if predicate(value)])


def select_in_place(values: object, predicate: object) -> None:
    values[:] = [value for value in values if predicate(value)]


def each(values: object, function: object) -> None:
    for value in values:
        function(value)


def fold(values: object, initial: object, function: object) -> object:
    result = initial
    for value in values:
        result = function(result, value)
    return result


def fold_right(values: object, initial: object, function: object) -> object:
    result = initial
    for value in reversed(values):
        result = function(result, value)
    return result


def any(values: object, predicate: object) -> bool:
    return builtins.any(predicate(value) for value in values)


def all(values: object, predicate: object) -> bool:
    return builtins.all(predicate(value) for value in values)


def find(values: object, predicate: object) -> object:
    for value in values:
        if predicate(value):
            return Option.Some(value=value)
    return _none()


def find_index(values: object, predicate: object) -> object:
    for index, value in builtins.enumerate(values):
        if predicate(value):
            return Option.Some(value=index)
    return _none()


def reverse(values: object) -> object:
    return array(list(reversed(values)))


def reverse_in_place(values: object) -> None:
    values.reverse()


def _sort(values: object, comparator: _Comparator) -> list[object]:
    return sorted(values, key=cmp_to_key(comparator))


def sort(values: object, comparator: _Comparator) -> object:
    return array(_sort(values, comparator))


def sort_in_place(values: object, comparator: _Comparator) -> None:
    values[:] = _sort(values, comparator)


def slice(values: object, start: int, end: int) -> object:
    return array(values[start:end])


def take(values: object, count: int) -> object:
    return array(values[: max(count, 0)])


def drop(values: object, count: int) -> object:
    return array(values[max(count, 0) :])


def concat(values: object, other: object) -> object:
    return array([*values, *other])


def concat_in_place(values: object, other: object) -> None:
    values.extend(other)


def zip(values: object, other: object) -> object:
    return array([Pair(first=left, second=right) for left, right in builtins.zip(values, other)])


def zip_with(values: object, other: object, function: object) -> object:
    return array([function(left, right) for left, right in builtins.zip(values, other)])


def enumerate(values: object) -> object:
    return array([Pair(first=index, second=value) for index, value in builtins.enumerate(values)])


def flatten(values: object) -> object:
    return array([value for inner in values for value in inner])


def unzip(values: object) -> object:
    first: list[object] = []
    second: list[object] = []
    for value in values:
        first.append(value.first)
        second.append(value.second)
    return Pair(first=array(first), second=array(second))


def repeat(value: object, count: int) -> object:
    return array([value] * max(count, 0))


def range(start: int, end: int) -> object:
    step = 1 if start <= end else -1
    return array(list(builtins.range(start, end + step, step)))


__all__ = [
    "all",
    "any",
    "clear",
    "concat",
    "concat_in_place",
    "contains",
    "count",
    "drop",
    "each",
    "enumerate",
    "select",
    "select_in_place",
    "find",
    "find_index",
    "first",
    "first_option",
    "flatten",
    "fold",
    "fold_right",
    "index_of",
    "index_of_option",
    "insert",
    "last",
    "last_option",
    "map",
    "map_in_place",
    "pop",
    "pop_option",
    "push",
    "range",
    "remove_at",
    "repeat",
    "reverse",
    "reverse_in_place",
    "size",
    "slice",
    "sort",
    "sort_in_place",
    "take",
    "unzip",
    "zip",
    "zip_with",
]
