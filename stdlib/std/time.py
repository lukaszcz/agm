"""Clock, ISO-8601, and strftime operations for ``std/time``."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal

from agl import AglException, nominals

TimeParseError = nominals.std.time.TimeParseError
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MICROSECONDS_PER_SECOND = Decimal(1_000_000)
_NANOSECONDS_PER_SECOND = Decimal(1_000_000_000)


def _seconds_from_nanoseconds(nanoseconds: int) -> Decimal:
    return Decimal(nanoseconds) / _NANOSECONDS_PER_SECOND


def _epoch_seconds(value: datetime) -> Decimal:
    utc_value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    delta = utc_value - _EPOCH
    return Decimal(delta.days * 86_400 + delta.seconds) + Decimal(delta.microseconds).scaleb(-6)


def _datetime_from_epoch(epoch: Decimal) -> datetime:
    microseconds = int(
        (epoch * _MICROSECONDS_PER_SECOND).to_integral_value(rounding=ROUND_HALF_EVEN)
    )
    return _EPOCH + timedelta(microseconds=microseconds)


def _parse_error(raw: str) -> None:
    raise AglException(TimeParseError(message="Could not parse time.", raw=raw))


def now() -> Decimal:
    """Return the current Unix epoch timestamp with nanosecond source precision."""
    return _seconds_from_nanoseconds(time.time_ns())


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 timestamp."""
    return datetime.now(UTC).isoformat()


def monotonic() -> Decimal:
    """Return a monotonic elapsed-time clock in seconds."""
    return _seconds_from_nanoseconds(time.monotonic_ns())


def sleep(seconds: Decimal) -> None:
    """Pause the current thread for *seconds*."""
    time.sleep(float(seconds))


def parse_iso(raw: str) -> Decimal:
    """Parse an ISO-8601 timestamp, interpreting offset-free input as UTC."""
    try:
        return _epoch_seconds(datetime.fromisoformat(raw))
    except (OverflowError, ValueError):
        _parse_error(raw)


def format_iso(epoch: Decimal) -> str:
    """Format Unix epoch seconds as a UTC ISO-8601 timestamp."""
    return _datetime_from_epoch(epoch).isoformat()


def format(epoch: Decimal, fmt: str) -> str:
    """Format Unix epoch seconds in UTC with a strftime format."""
    return _datetime_from_epoch(epoch).strftime(fmt)


def parse(raw: str, fmt: str) -> Decimal:
    """Parse a strptime value, interpreting offset-free input as UTC."""
    try:
        return _epoch_seconds(datetime.strptime(raw, fmt))
    except (OverflowError, ValueError):
        _parse_error(raw)


__all__ = ["format", "format_iso", "monotonic", "now", "now_iso", "parse", "parse_iso", "sleep"]
