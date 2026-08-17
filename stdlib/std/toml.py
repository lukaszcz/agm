"""TOML parsing and rendering operations for ``std/toml``."""

from __future__ import annotations

import tomllib
from datetime import date, datetime, time
from decimal import Decimal
from typing import cast

import tomlkit
from agl import AglException, json, nominals

Option = nominals.std.option.Option
TomlParseError = nominals.std.toml.TomlParseError
TomlRenderError = nominals.std.toml.TomlRenderError


def _none() -> object:
    return getattr(Option, "None")()


def _parse_error(raw: str) -> None:
    raise AglException(TomlParseError(message="Could not parse TOML.", raw=raw))


def _render_error(message: str) -> None:
    raise AglException(TomlRenderError(message=message))


def _json_value(value: object) -> object:
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def parse(raw: str) -> object:
    try:
        parsed = cast(dict[str, object], tomllib.loads(raw, parse_float=Decimal))
    except tomllib.TOMLDecodeError:
        _parse_error(raw)
    return json(_json_value(parsed))


def parse_option(raw: str) -> object:
    try:
        return Option.Some(value=parse(raw))
    except AglException:
        return _none()


_TOML_INT_MIN = -(2**63)
_TOML_INT_MAX = 2**63 - 1


def _validate_renderable(value: object) -> None:
    """Raise for JSON values outside TOML's value domain."""
    if value is None:
        _render_error("TOML cannot represent null values.")
    if isinstance(value, Decimal) and value.is_nan() and (
        value.is_snan() or value.as_tuple().digits
    ):
        _render_error("TOML cannot represent signaling or payload NaN values.")
    if type(value) is int and not _TOML_INT_MIN <= value <= _TOML_INT_MAX:
        _render_error("TOML integers must fit in a signed 64-bit value.")
    if isinstance(value, dict):
        for item in value.values():
            _validate_renderable(item)
    if isinstance(value, list):
        for item in value:
            _validate_renderable(item)


def _decimal_literal(value: Decimal) -> str:
    if value.is_nan():
        return "-nan" if value.is_signed() else "nan"
    if value.is_infinite():
        return "-inf" if value.is_signed() else "inf"
    if value.is_zero():
        return "-0.0" if value.is_signed() else "0.0"
    literal = str(value)
    return literal if "." in literal or "e" in literal.lower() else f"{literal}.0"


def _toml_value(value: object) -> object:
    if isinstance(value, Decimal):
        return tomlkit.parse(f"value = {_decimal_literal(value)}")["value"]
    if isinstance(value, dict):
        return {key: _toml_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_toml_value(item) for item in value]
    return value


def render(value: object) -> str:
    raw = value.value
    if not isinstance(raw, dict):
        _render_error("TOML documents must have a table root.")
    _validate_renderable(raw)
    try:
        return tomlkit.dumps(_toml_value(raw))
    except Exception:
        _render_error("Could not render TOML.")


__all__ = ["parse", "parse_option", "render"]
