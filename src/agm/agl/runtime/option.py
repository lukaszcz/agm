"""Constructors for standard ``Option`` member-record values."""

from __future__ import annotations

from agm.agl.ir.builtin_nominals import BuiltinNominals, resolve_standard_member_name
from agm.agl.semantics.values import RecordValue, TextValue, Value


def some_value(value: Value, *, nominals: BuiltinNominals) -> RecordValue:
    """Build an ``Option::Some(value)`` member record."""
    member = nominals.resolve_standard_member("Option", "Some")
    return RecordValue(nominal=member.nominal, fields={"value": value})


def none_value(*, nominals: BuiltinNominals) -> RecordValue:
    """Build an ``Option::None`` member record."""
    member = nominals.resolve_standard_member("Option", "None")
    return RecordValue(nominal=member.nominal, fields={})


def option_text(value: RecordValue, *, nominals: BuiltinNominals) -> str | None:
    """Return the text an ``Option[text]`` member record carries, or ``None``."""
    if resolve_standard_member_name(value.nominal, "Option", ("Some",), nominals) is None:
        return None
    payload = value.fields["value"]
    assert isinstance(payload, TextValue)
    return payload.value
