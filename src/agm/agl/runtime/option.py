"""Constructors for ``std/core::Option`` member-record values."""

from __future__ import annotations

from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS, BuiltinNominals
from agm.agl.semantics.values import RecordValue, TextValue, Value


def some_value(value: Value, *, nominals: BuiltinNominals = NO_BUILTIN_DECLARATIONS) -> RecordValue:
    """Build a ``std/core::Option::Some(value)`` member record."""
    member = nominals.resolve_standard_member("Option", "Some")
    return RecordValue(
        nominal=member.nominal,
        display_name=member.display_name,
        fields={"value": value},
    )


def none_value(*, nominals: BuiltinNominals = NO_BUILTIN_DECLARATIONS) -> RecordValue:
    """Build a ``std/core::Option::None`` member record."""
    member = nominals.resolve_standard_member("Option", "None")
    return RecordValue(
        nominal=member.nominal,
        display_name=member.display_name,
        fields={},
    )


def option_text(
    value: RecordValue, *, nominals: BuiltinNominals = NO_BUILTIN_DECLARATIONS
) -> str | None:
    """Return the text an ``Option[text]`` member record carries, or ``None``."""
    source_some = nominals.resolve_standard_member("Option", "Some").nominal
    fallback_some = NO_BUILTIN_DECLARATIONS.resolve_standard_member("Option", "Some").nominal
    if value.nominal not in {source_some, fallback_some} and value.display_name != "Option::Some":
        return None
    payload = value.fields["value"]
    assert isinstance(payload, TextValue)
    return payload.value
