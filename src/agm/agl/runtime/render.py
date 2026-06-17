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
    AgentValue,
    BoolValue,
    Closure,
    DecimalValue,
    IntValue,
    TextValue,
    UnitValue,
    Value,
)
from agm.agl.runtime.serialize import dumps_exact, value_to_json_obj

# Escape mapping mirroring AgL's JSON-style string-literal escape set, used by
# :func:`_quote_text` to produce a REPL-echo surface form for ``text`` values.
_TEXT_ESCAPES: dict[str, str] = {
    '"': '\\"',
    "\\": "\\\\",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\b": "\\b",
    "\f": "\\f",
}


def _quote_text(s: str) -> str:
    """Return *s* as a double-quoted AgL string literal surface form.

    Mirrors the escape set AgL string literals accept (``\\"``, ``\\\\``,
    ``\\n`` …, ``\\uXXXX``) so a ``text`` value round-trips visually in the
    REPL echo.  This is a display concern only — it never affects template
    interpolation, where ``text`` is always inserted verbatim.
    """
    out: list[str] = ['"']
    for ch in s:
        esc = _TEXT_ESCAPES.get(ch)
        if esc is not None:
            out.append(esc)
        elif ch < " ":
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


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


def _closure_surface(closure: Closure) -> str:
    """Return the human-readable surface form for a ``Closure`` value.

    Uses only the fields the ``Closure`` carries: the arity (from ``params``)
    and the declared return type (from ``return_type``).  The form is
    ``"<function/N -> T>"`` where N is the parameter count and T is the return
    type's canonical representation.

    This surface form is produced ONLY by ``render_value`` for REPL echo and
    ``:bindings``/``:params`` display — it is never reachable from ``print``,
    template interpolation, or ``exec``, which the type checker statically
    prevents (design D9).
    """
    arity = len(closure.params)
    return f"<function/{arity} -> {closure.return_type!r}>"


def render_value(value: Value) -> str:
    """Render *value* for use inside any template interpolation (uniform rendering).

    Rules (applied identically in ``prompt``, ``print``, and ``exec`` contexts):
    - ``text``                        → verbatim string.
    - ``int`` / ``decimal`` / ``bool`` → plain scalar text via ``_scalar_text``.
    - ``unit`` (``()``)               → the literal text ``"()"``.
    - ``agent``                       → ``"<agent NAME>"``.
    - ``function`` (``Closure``)      → ``"<function/N -> T>"``.
    - Everything else (``list``, ``dict``, record, enum, ``json``, exception)
                                      → pretty-printed JSON, 2-space indent.

    No ``<dsl-value>`` boundary tags, no shell quoting.

    ``Closure`` and ``AgentValue`` are only reachable here from REPL echo /
    ``:bindings`` (the type checker statically prevents them from appearing in
    ``print``, template interpolation, or ``exec`` — design D9).
    """
    if isinstance(value, TextValue):
        return value.value
    if isinstance(value, UnitValue):
        return "()"
    if isinstance(value, (IntValue, DecimalValue, BoolValue)):
        return _scalar_text(value)
    if isinstance(value, AgentValue):
        return f"<agent {value.name}>"
    if isinstance(value, Closure):
        return _closure_surface(value)
    # Structured / json / exception: pretty JSON.
    return _pretty_json(value)


def render_value_repl(value: Value) -> str:
    """Render *value* for REPL echo (``agl>`` and ``:bindings`` / ``:params``).

    Identical to :func:`render_value` except that a top-level ``text`` value is
    shown quoted (as an AgL string literal surface form) so the REPL echo of
    ``agl> \"aaa\"`` reads ``\"aaa\"``.  Text nested inside structured values
    is already quoted by the JSON renderer, so only the bare ``text`` case
    differs.  Template interpolation (``print`` / ``prompt`` / ``exec``) is
    unaffected — it always uses :func:`render_value` verbatim.
    """
    if isinstance(value, TextValue):
        return _quote_text(value.value)
    return render_value(value)


