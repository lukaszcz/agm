"""JSON parsing and inspection operations for ``std/json``."""

from decimal import Decimal

from agl import AglException, array, json, nominals

from agm.agl.runtime.codec import extract_json_text
from agm.agl.runtime.convert import StrictJsonParseError, parse_json_strict

JsonParseError = nominals.std.errors.JsonParseError
KeyError = nominals.std.errors.KeyError
Option = nominals.std.option.Option


def _none() -> object:
    return getattr(Option, "None")()


def _some(value: object) -> object:
    return Option.Some(value=value)


def _parse_error(raw: str, message: str) -> None:
    raise AglException(JsonParseError(message=message, raw=raw))


def _parse(raw: str, *, lenient: bool) -> object:
    candidate = extract_json_text(raw) if lenient else raw
    if candidate is None:
        _parse_error(raw, "Could not recover a single JSON value from the input.")
    try:
        return json(parse_json_strict(candidate))
    except StrictJsonParseError as exc:
        _parse_error(raw, exc.message)


def parse(raw: str) -> object:
    return _parse(raw, lenient=False)


def parse_option(raw: str) -> object:
    try:
        return _some(parse(raw))
    except AglException:
        return _none()


def parse_lenient(raw: str) -> object:
    return _parse(raw, lenient=True)


def parse_lenient_option(raw: str) -> object:
    try:
        return _some(parse_lenient(raw))
    except AglException:
        return _none()


def kind(value: object) -> str:
    raw = value.value
    if raw is None:
        return "null"
    if isinstance(raw, bool):
        return "bool"
    if isinstance(raw, int):
        return "int"
    if isinstance(raw, Decimal):
        return "decimal"
    if isinstance(raw, str):
        return "text"
    if isinstance(raw, list):
        return "array"
    return "object"


def size(value: object) -> int:
    raw = value.value
    return len(raw) if isinstance(raw, (dict, list)) else 0


def keys(value: object) -> object:
    raw = value.value
    return array(list(raw)) if isinstance(raw, dict) else array([])


def has(value: object, key: str) -> bool:
    raw = value.value
    return isinstance(raw, dict) and key in raw


def get(value: object, key: str) -> object:
    raw = value.value
    if not isinstance(raw, dict) or key not in raw:
        raise AglException(KeyError(message="JSON object key not found", key=key))
    return json(raw[key])


def get_option(value: object, key: str) -> object:
    raw = value.value
    return _some(json(raw[key])) if isinstance(raw, dict) and key in raw else _none()


__all__ = [
    "get",
    "get_option",
    "has",
    "keys",
    "kind",
    "parse",
    "parse_lenient",
    "parse_lenient_option",
    "parse_option",
    "size",
]
