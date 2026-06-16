"""Uniform value rendering for AgL string interpolation.

Every ``${expr}`` in any template — whether it appears in a ``prompt`` agent
call, a ``print`` statement, or an ``exec`` shell call — renders its value with
the same rules:

- ``text``               → verbatim string (no boundary markers, no quoting)
- ``int`` / ``decimal`` / ``bool`` → plain scalar text (see ``_scalar_text``)
- ``list`` / ``dict`` / record / enum / ``json`` / exception
                         → pretty-printed JSON, 2-space indent (see ``_pretty_json``)

No ``<dsl-value>`` boundary tags are ever added.  No shell quoting is applied
to ``exec`` interpolations — the rendered text is inserted verbatim.  The
``as <name>`` renderer-override syntax no longer exists; ``${expr}`` always
uses the rules above.

``render_value`` is the single entry point used by the interpreter for all
three evaluation contexts.
"""

from __future__ import annotations

from agm.agl.eval.values import (
    BoolValue,
    DecimalValue,
    IntValue,
    TextValue,
    Value,
)
from agm.agl.runtime.serialize import dumps_exact, value_to_json_obj


def _pretty_json(value: Value) -> str:
    """Render *value* as pretty-printed JSON (2-space indent).

    Uses the shared exact serializer so ``Decimal`` values are emitted as exact
    unquoted numeric text (never routed through binary ``float``; design §5.1).
    """
    return dumps_exact(value_to_json_obj(value), indent=2)


def _scalar_text(value: IntValue | DecimalValue | BoolValue) -> str:
    """Render an int, decimal, or bool value as a plain text string."""
    if isinstance(value, IntValue):
        return str(value.value)
    if isinstance(value, DecimalValue):
        # Use normalize() to drop trailing zeros (e.g. "1.50" → "1.5"),
        # but keep at least one decimal digit.
        d = value.value.normalize()
        # Avoid scientific notation (e.g. "1E+2" → "100").
        return format(d, "f")
    # BoolValue
    return "true" if value.value else "false"


def render_value(value: Value) -> str:
    """Render *value* for use inside any template interpolation (uniform rendering).

    Rules (applied identically in ``prompt``, ``print``, and ``exec`` contexts):
    - ``text``                        → verbatim string.
    - ``int`` / ``decimal`` / ``bool`` → plain scalar text via ``_scalar_text``.
    - Everything else (``list``, ``dict``, record, enum, ``json``, exception)
                                      → pretty-printed JSON, 2-space indent.

    No ``<dsl-value>`` boundary tags, no shell quoting.
    """
    if isinstance(value, TextValue):
        return value.value
    if isinstance(value, (IntValue, DecimalValue, BoolValue)):
        return _scalar_text(value)
    # Structured / json / exception: pretty JSON.
    return _pretty_json(value)


