"""AgL-native value rendering for string interpolation, print, and REPL echo.

Values render in the AgL syntax used to define them by default.  JSON output
is an explicit opt-in via ``as json``.  Two boolean axes drive the leaf cases:

- **top-level vs nested** — the caller passes the value at top level
  (``top_level=True``); every recursive child call uses ``top_level=False``.
- **interpolation vs REPL echo** — controls only the top-level ``text`` case:
  interpolation (``render_value``) leaves ``text`` verbatim; REPL echo
  (``render_value_repl``) quotes it as an AgL string literal.

Per-kind rules:

- ``text`` (top-level, interpolation)  → verbatim, no quotes
- ``text`` (top-level, REPL echo)      → quoted AgL string literal via ``_quote_text``
- ``text`` (nested, any mode)          → quoted AgL string literal via ``_quote_text``
- ``int`` / ``decimal`` / ``bool``     → ``_scalar_text`` at any depth
- ``unit``                             → ``()``
- ``agent``                            → ``<agent NAME>``
- ``function``                         → ``<function/N -> T>``
- ``json`` (top-level)                 → pretty JSON, 2-space indent
- ``json`` (nested)                    → compact JSON, single-line
- ``list``                             → ``[e1, e2, ...]``, children nested
- ``dict``                             → ``{"k1": v1, ...}``, keys always quoted
- record                               → ``TypeName(f1: v1, ...)`` declaration order
- enum   → ``TypeName.Variant(f1: v1, ...)``; nullary variant → ``TypeName.Variant``
- exception                            → ``TypeName(f1: v1, ...)`` all fields incl. ``trace_id``

Nominal values (record, enum, exception) carry their fields in declaration
order already — the interpreter normalizes them at construction time — so the
renderer simply walks ``value.fields`` and needs no type information.
"""

from __future__ import annotations

from agm.agl.eval.values import (
    AgentValue,
    BoolValue,
    ConstructorValue,
    DecimalValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    IrClosureValue,
    JsonValue,
    ListValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
)
from agm.agl.runtime.serialize import dumps_exact, value_to_json_obj

# ---------------------------------------------------------------------------
# Escape mapping for _quote_text
# ---------------------------------------------------------------------------

# JSON escape set extended with ``$`` so ``${`` cannot be read as interpolation
# inside a quoted string literal rendered into output.
_TEXT_ESCAPES: dict[str, str] = {
    '"': '\\"',
    "\\": "\\\\",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\b": "\\b",
    "\f": "\\f",
    "$": "\\$",
}


def _quote_text(s: str) -> str:
    """Return *s* as a double-quoted AgL string literal surface form.

    Applies the JSON escape set plus ``\\$`` (so ``${`` cannot read as
    interpolation) and ``\\uXXXX`` for remaining control characters.  Used
    for both nested ``text`` values and the top-level REPL-echo case so the
    two never diverge.
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


# ---------------------------------------------------------------------------
# Core recursive renderer
# ---------------------------------------------------------------------------


def _render(value: Value, *, top_level: bool, repl: bool) -> str:
    """Recursive AgL-native renderer.

    ``top_level=True`` for the outermost call; ``False`` for all children.
    ``repl=True`` enables REPL-echo quoting for a top-level ``text`` value.
    Nominal values are rendered straight from ``value.fields``, which the
    interpreter already keeps in declaration order.
    """
    if isinstance(value, TextValue):
        if top_level and not repl:
            # Interpolation context: verbatim.
            return value.value
        # REPL echo (top-level) or nested (any mode): quoted.
        return _quote_text(value.value)

    if isinstance(value, UnitValue):
        return "()"

    if isinstance(value, (IntValue, DecimalValue, BoolValue)):
        return _scalar_text(value)

    if isinstance(value, AgentValue):
        return f"<agent {value.name}>"

    if isinstance(value, ConstructorValue):
        if value.variant is not None:
            return f"<constructor {value.display_name}.{value.variant}>"
        return f"<constructor {value.display_name}>"

    if isinstance(value, IrClosureValue):
        return f"<function/{value.arity} -> {value.result_label}>"

    if isinstance(value, JsonValue):
        if top_level:
            return dumps_exact(value_to_json_obj(value), indent=2)
        return dumps_exact(value_to_json_obj(value), indent=None)

    if isinstance(value, ListValue):
        if not value.elements:
            return "[]"
        items = [_render(e, top_level=False, repl=repl) for e in value.elements]
        return "[" + ", ".join(items) + "]"

    if isinstance(value, DictValue):
        if not value.entries:
            return "{}"
        items = [
            f"{_quote_text(k)}: {_render(v, top_level=False, repl=repl)}"
            for k, v in value.entries.items()
        ]
        return "{" + ", ".join(items) + "}"

    if isinstance(value, (RecordValue, EnumValue, ExceptionValue)):
        prefix = (
            f"{value.display_name}.{value.variant}"
            if isinstance(value, EnumValue)
            else value.display_name
        )
        if not value.fields:
            return prefix if isinstance(value, EnumValue) else f"{prefix}()"
        field_parts = [
            f"{name}: {_render(v, top_level=False, repl=repl)}"
            for name, v in value.fields.items()
        ]
        return f"{prefix}(" + ", ".join(field_parts) + ")"

    # Exhaustiveness: the Value union is closed; all cases covered above.
    raise RuntimeError(f"render: unhandled value type {type(value).__name__}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_value(value: Value) -> str:
    """Render *value* for interpolation, ``print``, or ``as text``.

    Top-level ``text`` is verbatim (no quotes).  All other rendering follows
    the AgL-native rules: scalars as plain text, ``list``/``dict`` in AgL
    bracket/brace form, record/enum/exception as ``TypeName(field: value, ...)``
    with fields in declaration order.  ``json`` values render as pretty-printed
    JSON (2-space indent) at top level and compact single-line JSON when nested.
    """
    return _render(value, top_level=True, repl=False)


def render_value_repl(value: Value) -> str:
    """Render *value* for REPL echo (``agl>`` prompt and ``:bindings`` / ``:params``).

    Identical to :func:`render_value` except that a top-level ``text`` value is
    shown as a quoted AgL string literal so the REPL echo of ``"aaa"`` reads
    ``"aaa"``.  Text nested inside structured values is also quoted.  Template
    interpolation (``print`` / ``prompt`` / ``exec``) always uses
    :func:`render_value` verbatim.
    """
    return _render(value, top_level=True, repl=True)
