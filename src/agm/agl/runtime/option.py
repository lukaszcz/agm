"""Constructors for ``std/core::Option`` runtime values.

Shared by the config-value decoder (:mod:`agm.agl.runtime.params`) and the
agent-request effect builder (:mod:`agm.agl.eval.effects`) so the Option enum
value shape (nominal, variant names, fields) is spelled out exactly once.
"""

from __future__ import annotations

from agm.agl.ir.ids import NominalId
from agm.agl.ir.reserved_nominals import require_reserved_nominal_id
from agm.agl.semantics.values import EnumValue, Value

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
