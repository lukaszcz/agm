"""Companion contracts for the ``std/toml`` standard-library module."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

from agm.agl.ir.ids import NominalId
from agm.agl.ir.program import NominalDescriptor, NominalKind
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.boundary import AglException, AglJson, decode_boundary_value
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.semantics.values import JsonValue, TextValue

_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "packages" / "stdlib"
_TOML_MODULE = ModuleId(("std", "toml"))
_TOML_PARSE_ERROR = NominalId(9_500_001)
_TOML_RENDER_ERROR = NominalId(9_500_002)


class _TomlCompanion(Protocol):
    def parse(self, raw: str) -> object: ...

    def render(self, value: object) -> str: ...


def _toml_companion() -> _TomlCompanion:
    """Load ``std/toml`` through the same extern boundary as production."""
    registry = ExternRegistry()
    registry.set_nominals(
        {
            _TOML_PARSE_ERROR: NominalDescriptor(
                nominal=_TOML_PARSE_ERROR,
                module_id=_TOML_MODULE,
                scope_path=(),
                declared_name="TomlParseError",
                kind=NominalKind.EXCEPTION,
                fields=("message", "raw"),
            ),
            _TOML_RENDER_ERROR: NominalDescriptor(
                nominal=_TOML_RENDER_ERROR,
                module_id=_TOML_MODULE,
                scope_path=(),
                declared_name="TomlRenderError",
                kind=NominalKind.EXCEPTION,
                fields=("message",),
            ),
        },
    )
    module: ModuleType = registry.load_companion(_TOML_MODULE, _STDLIB_ROOT / "src" / "toml.py")
    return cast(_TomlCompanion, module)


def test_toml_parse_preserves_tables_arrays_decimals_and_iso_temporal_values() -> None:
    companion = _toml_companion()

    result = decode_boundary_value(
        companion.parse("""
ratio = 1.25
date = 1979-05-27
time = 07:32:00
instant = 1979-05-27T07:32:00Z

[service]
ports = [8000, 8001]
""")
    )

    assert result == JsonValue(
        {
            "ratio": Decimal("1.25"),
            "date": "1979-05-27",
            "time": "07:32:00",
            "instant": "1979-05-27T07:32:00+00:00",
            "service": {"ports": [8000, 8001]},
        }
    )


def test_toml_parse_failures_are_typed() -> None:
    companion = _toml_companion()

    with pytest.raises(AglException) as exc_info:
        companion.parse("broken = [")
    assert exc_info.value.value.fields["raw"] == TextValue("broken = [")


def test_toml_render_round_trips_tables_and_rejects_unrepresentable_json() -> None:
    companion = _toml_companion()
    original = {"ratio": Decimal("1.25"), "owner": {"name": "Ada"}, "ports": [8000, 8001]}

    rendered = companion.render(AglJson(original))

    assert decode_boundary_value(companion.parse(rendered)) == JsonValue(original)
    for invalid in (None, {"missing": None}, ["not", "a", "table"]):
        with pytest.raises(AglException) as exc_info:
            companion.render(AglJson(invalid))
        assert exc_info.value.value.nominal == _TOML_RENDER_ERROR


@pytest.mark.parametrize(
    ("value", "literal"),
    (
        (Decimal("0"), "0.0"),
        (Decimal("-0"), "-0.0"),
        (Decimal("0E+3"), "0.0"),
        (Decimal("-0E+3"), "-0.0"),
        (Decimal("1E+2"), "1E+2"),
        (Decimal("-NaN"), "-nan"),
        (Decimal("-Infinity"), "-inf"),
    ),
)
def test_toml_render_emits_standard_decimal_literals(value: Decimal, literal: str) -> None:
    companion = _toml_companion()

    rendered = companion.render(AglJson({"value": value}))

    assert rendered == f"value = {literal}"
    parsed = decode_boundary_value(companion.parse(rendered))
    assert isinstance(parsed, JsonValue)
    assert isinstance(parsed.raw, dict)
    parsed_value = parsed.raw["value"]
    assert isinstance(parsed_value, Decimal)
    if value.is_nan():
        assert parsed_value.is_nan()
        assert parsed_value.is_signed() == value.is_signed()
    else:
        assert parsed_value == value


@pytest.mark.parametrize("value", (-(2**63), 2**63 - 1))
def test_toml_render_accepts_signed_64_bit_integer_bounds(value: int) -> None:
    companion = _toml_companion()

    rendered = companion.render(AglJson({"value": value}))

    assert decode_boundary_value(companion.parse(rendered)) == JsonValue({"value": value})


@pytest.mark.parametrize(
    "value", (Decimal("sNaN"), Decimal("-sNaN42"), Decimal("NaN42"), Decimal("-NaN42"))
)
def test_toml_render_rejects_signaling_and_payload_nans_with_a_typed_error(value: Decimal) -> None:
    companion = _toml_companion()

    with pytest.raises(AglException) as exc_info:
        companion.render(AglJson({"value": value}))

    assert exc_info.value.value.nominal == _TOML_RENDER_ERROR


@pytest.mark.parametrize("value", (-(2**63) - 1, 2**63))
def test_toml_render_rejects_out_of_range_integers_with_a_typed_error(value: int) -> None:
    companion = _toml_companion()

    with pytest.raises(AglException) as exc_info:
        companion.render(AglJson({"value": value}))

    assert exc_info.value.value.nominal == _TOML_RENDER_ERROR
