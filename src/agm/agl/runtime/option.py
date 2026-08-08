"""Constructors for ``std/core::Option`` runtime values.

Shared by the config-value decoder (:mod:`agm.agl.runtime.params`) and the
agent-request effect builder (:mod:`agm.agl.eval.effects`) so the Option enum
value shape (nominal, variant names, fields) is spelled out exactly once.
"""

from __future__ import annotations

from agm.agl.ir.ids import NominalId
from agm.agl.ir.reserved_nominals import require_reserved_nominal_id
from agm.agl.semantics.values import EnumValue, TextValue, Value

_OPTION_NOMINAL = NominalId(require_reserved_nominal_id("Option"))


def some_value(value: Value) -> EnumValue:
    """Build a ``std/core::Option`` ``Some(value)`` runtime value."""
    return EnumValue(
        nominal=_OPTION_NOMINAL,
        display_name="Option",
        variant="Some",
        fields={"value": value},
    )


def none_value() -> EnumValue:
    """Build a ``std/core::Option`` ``None`` runtime value."""
    return EnumValue(
        nominal=_OPTION_NOMINAL,
        display_name="Option",
        variant="None",
        fields={},
    )


def option_text(value: EnumValue) -> str | None:
    """Return the text a ``std/core::Option[text]`` *value* carries, or ``None``.

    The decode counterpart of :func:`some_value` / :func:`none_value`: the
    ``None`` variant answers ``None``, and ``Some(t)`` answers with ``t``.
    Reading the shape back here rather than at each call site keeps the
    Option enum value shape spelled out exactly once in both directions.
    """
    if value.variant != "Some":
        return None
    payload = value.fields["value"]
    assert isinstance(payload, TextValue)
    return payload.value
