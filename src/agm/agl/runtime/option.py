"""Constructors for ``std/core::Option`` member-record values."""

from __future__ import annotations

from agm.agl.ir.ids import NominalId
from agm.agl.semantics.type_table import OPTION_TYPE_DEF
from agm.agl.semantics.values import RecordValue, TextValue, Value

_OPTION_MEMBER_NOMINALS = {
    member.name: NominalId(member.decl_id) for member in OPTION_TYPE_DEF.members
}
_NONE_NOMINAL = _OPTION_MEMBER_NOMINALS["None"]
_SOME_NOMINAL = _OPTION_MEMBER_NOMINALS["Some"]


def some_value(value: Value) -> RecordValue:
    """Build a ``std/core::Option::Some(value)`` member record."""
    return RecordValue(
        nominal=_SOME_NOMINAL,
        display_name="Option::Some",
        fields={"value": value},
    )


def none_value() -> RecordValue:
    """Build a ``std/core::Option::None`` member record."""
    return RecordValue(
        nominal=_NONE_NOMINAL,
        display_name="Option::None",
        fields={},
    )


def option_text(value: RecordValue) -> str | None:
    """Return the text an ``Option[text]`` member record carries, or ``None``."""
    if value.nominal != _SOME_NOMINAL:
        return None
    payload = value.fields["value"]
    assert isinstance(payload, TextValue)
    return payload.value
