"""Generic string-parsing helpers shared across AGM layers."""

from __future__ import annotations

import decimal
import re


def format_timeout(seconds: float) -> str:
    """Render parsed seconds in syntax accepted by :func:`parse_timeout`."""
    return f"{format(decimal.Decimal(str(seconds)), 'f')}s"


def parse_timeout(value: str) -> float:
    """Parse a human-readable timeout string into seconds.

    Supports plain numbers (treated as seconds) and durations with
    suffixes: ``s`` (seconds), ``m`` (minutes), ``h`` (hours).

    Examples::

        parse_timeout("30")    -> 30.0
        parse_timeout("30s")   -> 30.0
        parse_timeout("10m")   -> 600.0
        parse_timeout("2h")    -> 7200.0
    """
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(s|m|h)?", value.strip())
    if match is None:
        raise ValueError(f"invalid timeout format: {value!r}")
    amount = float(match.group(1))
    unit: str = match.group(2) or ""
    if unit == "m":
        return amount * 60
    if unit == "h":
        return amount * 3600
    return amount
