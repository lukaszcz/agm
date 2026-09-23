"""``std/text`` companion operations, including runtime interpolation and text methods."""

from agl import array, nominals, option_none, option_some

from agm.agl.runtime.boundary import raise_index_error
from agm.util.interp import interp

IndexError = nominals.std.errors.IndexError


def _padding(length: int, fill: str) -> str:
    if length <= 0 or not fill:
        return ""
    return (fill * ((length + len(fill) - 1) // len(fill)))[:length]


def concat(left: str, right: str) -> str:
    return left + right


def size(value: str) -> int:
    return len(value)


def chars(value: str) -> object:
    return array(list(value))


def lines(value: str) -> object:
    return array(value.splitlines())


def join(values: object, separator: str) -> str:
    return separator.join(values)


def split(value: str, separator: str) -> object:
    return array(value.split(separator))


def trim(value: str) -> str:
    return value.strip()


def trim_start(value: str) -> str:
    return value.lstrip()


def trim_end(value: str) -> str:
    return value.rstrip()


def upper(value: str) -> str:
    return value.upper()


def lower(value: str) -> str:
    return value.lower()


def starts_with(value: str, prefix: str) -> bool:
    return value.startswith(prefix)


def ends_with(value: str, suffix: str) -> bool:
    return value.endswith(suffix)


def contains(value: str, substring: str) -> bool:
    return substring in value


def index_of(value: str, substring: str) -> int:
    index = value.find(substring)
    if index < 0:
        raise_index_error(IndexError, "text substring not found", -1, len(value))
    return index


def index_of_option(value: str, substring: str) -> object:
    index = value.find(substring)
    return option_some(index) if index >= 0 else option_none()


def replace(value: str, old: str, new: str) -> str:
    return value.replace(old, new)


def slice(value: str, start: int, end: int) -> str:
    return value[start:end]


def repeat(value: str, count: int) -> str:
    return value * max(count, 0)


def pad_start(value: str, length: int, fill: str) -> str:
    return _padding(length - len(value), fill) + value


def pad_end(value: str, length: int, fill: str) -> str:
    return value + _padding(length - len(value), fill)


__all__ = [
    "chars",
    "contains",
    "ends_with",
    "index_of",
    "index_of_option",
    "interp",
    "join",
    "lines",
    "lower",
    "pad_end",
    "pad_start",
    "repeat",
    "replace",
    "size",
    "slice",
    "split",
    "starts_with",
    "trim",
    "trim_end",
    "trim_start",
    "upper",
]
