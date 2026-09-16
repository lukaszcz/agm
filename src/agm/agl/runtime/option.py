"""Constructors for standard ``Option`` member-record values."""

from __future__ import annotations

from agm.agl.ir.builtin_nominals import BuiltinNominals, resolve_standard_member_name
from agm.agl.ir.ids import NominalId
from agm.agl.semantics.values import RecordValue, TextValue, Value


def _member(nominals: BuiltinNominals, name: str, *, declared: bool) -> NominalId:
    """Resolve an ``Option`` member's identity under the selected precedence.

    ``declared`` follows the program's own ``builtin enum Option`` when it
    declares one — the identity a value the language itself produces (an
    ``as?`` result) must carry, matching the type the checker gives that
    expression. The default is the standard representation every host-minted
    contract field uses.
    """
    resolve = nominals.resolve_member if declared else nominals.resolve_standard_member
    return resolve("Option", name).nominal


def some_value(value: Value, *, nominals: BuiltinNominals, declared: bool = False) -> RecordValue:
    """Build an ``Option::Some(value)`` member record."""
    nominal = _member(nominals, "Some", declared=declared)
    return RecordValue(nominal=nominal, fields={"value": value})


def none_value(*, nominals: BuiltinNominals, declared: bool = False) -> RecordValue:
    """Build an ``Option::None`` member record."""
    return RecordValue(nominal=_member(nominals, "None", declared=declared), fields={})


def option_text(value: RecordValue, *, nominals: BuiltinNominals) -> str | None:
    """Return the text an ``Option[text]`` member record carries, or ``None``."""
    if resolve_standard_member_name(value.nominal, "Option", ("Some",), nominals) is None:
        return None
    payload = value.fields["value"]
    assert isinstance(payload, TextValue)
    return payload.value
