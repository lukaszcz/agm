"""Renderer registry and rendering implementations.

Two rendering contexts:

1. **Prompt interpolation** (``${expr}`` in templates — §2.12):
   - ``default`` renderer: type-directed, boundary-marked for ``text``,
     scalar text for numbers/bools, pretty JSON otherwise.
   - ``raw`` renderer: plain string conversion, no boundary markers.
   - ``json`` renderer: always pretty JSON regardless of type.
   - ``bullets`` renderer: list as ``- item\\n`` lines; other types as JSON.

2. **Console rendering** for ``print`` statements (§11.12):
   - ``text`` verbatim (no boundary markers),
   - scalars as plain text,
   - structured values (list, dict, record, enum, json) as pretty JSON,
   - exceptions as their diagnostic JSON.
   No boundary markers anywhere in console rendering.

The ``render_for_prompt`` and ``render_for_console`` functions are the main
entry points used by the interpreter.
"""

from __future__ import annotations

import json
from typing import Callable, assert_never

from agm.agl.eval.values import (
    BoolValue,
    DecimalValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    JsonValue,
    ListValue,
    RecordValue,
    TextValue,
    Value,
)

# ---------------------------------------------------------------------------
# JSON serialization helpers (Value → JSON-compatible object)
# ---------------------------------------------------------------------------


def _value_to_json_obj(value: Value) -> object:
    """Convert a ``Value`` to a JSON-serializable Python object.

    Used for pretty-JSON rendering of structured values.
    """
    if isinstance(value, TextValue):
        return value.value
    if isinstance(value, IntValue):
        return value.value
    if isinstance(value, DecimalValue):
        # Decimal → float-like repr; use str for exact representation.
        return float(value.value)
    if isinstance(value, BoolValue):
        return value.value
    if isinstance(value, JsonValue):
        return value.raw
    if isinstance(value, ListValue):
        return [_value_to_json_obj(e) for e in value.elements]
    if isinstance(value, DictValue):
        return {k: _value_to_json_obj(v) for k, v in value.entries.items()}
    if isinstance(value, RecordValue):
        return {k: _value_to_json_obj(v) for k, v in value.fields.items()}
    if isinstance(value, EnumValue):
        result: dict[str, object] = {"$case": value.variant}
        result.update({k: _value_to_json_obj(v) for k, v in value.fields.items()})
        return result
    if isinstance(value, ExceptionValue):
        return {k: _value_to_json_obj(v) for k, v in value.fields.items()}
    assert_never(value)  # pragma: no cover


def _pretty_json(value: Value) -> str:
    """Render *value* as pretty-printed JSON (2-space indent)."""
    obj = _value_to_json_obj(value)
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _scalar_text(value: Value) -> str:
    """Render a scalar value as a plain text string (no boundary markers)."""
    if isinstance(value, TextValue):
        return value.value
    if isinstance(value, IntValue):
        return str(value.value)
    if isinstance(value, DecimalValue):
        # Use normalize() to drop trailing zeros (e.g. "1.50" → "1.5"),
        # but keep at least one decimal digit.
        d = value.value.normalize()
        # Avoid scientific notation (e.g. "1E+2" → "100").
        return format(d, "f")
    if isinstance(value, BoolValue):
        return "true" if value.value else "false"
    if isinstance(value, JsonValue):
        return json.dumps(value.raw, ensure_ascii=False)
    return _pretty_json(value)


# ---------------------------------------------------------------------------
# Prompt interpolation renderers (§2.12)
# ---------------------------------------------------------------------------

# Type alias for renderer functions.
RendererFn = Callable[[Value, str | None], str]


def _render_default(value: Value, name: str | None) -> str:
    """Default type-directed renderer with boundary markers for ``text``.

    Rendering table (§2.12):
    - ``text``            → ``<dsl-value name="…" type="text">…</dsl-value>``
    - ``int``/``decimal``/``bool`` → scalar text (no boundary)
    - ``json``            → fenced pretty JSON (boundary-marked)
    - ``list[T]``         → fenced pretty JSON
    - records             → fenced pretty JSON
    - enums               → fenced pretty JSON with ``"$case"``
    - exceptions          → fenced pretty JSON
    - ``dict``            → fenced pretty JSON
    """
    if isinstance(value, TextValue):
        tag_name = name if name else "value"
        return (
            f'<dsl-value name="{tag_name}" type="text">\n'
            f"{value.value}\n"
            f"</dsl-value>"
        )
    if isinstance(value, (IntValue, DecimalValue, BoolValue)):
        return _scalar_text(value)
    # Structured / json: fenced pretty JSON with boundary markers.
    type_kind = _type_kind_str(value)
    tag_name = name if name else "value"
    inner = _pretty_json(value)
    return (
        f'<dsl-value name="{tag_name}" type="{type_kind}">\n'
        f"{inner}\n"
        f"</dsl-value>"
    )


def _type_kind_str(value: Value) -> str:
    """Return the AgL type-kind label for a value (used in boundary tags).

    The structured-value boundary ``type=`` attribute (``list``/``dict``/the
    record or enum ``<RecordName>``) is **provisional**: the M5 acceptance suite
    currently pins only the ``text`` boundary, so the exact labels emitted here
    for non-text values are not yet finalized against §2.12 pins and may change
    when M5 locks them in.
    """
    if isinstance(value, TextValue):
        return "text"
    if isinstance(value, IntValue):
        return "int"
    if isinstance(value, DecimalValue):
        return "decimal"
    if isinstance(value, BoolValue):
        return "bool"
    if isinstance(value, JsonValue):
        return "json"
    if isinstance(value, ListValue):
        return "list"
    if isinstance(value, DictValue):
        return "dict"
    if isinstance(value, RecordValue):
        return value.type_name
    if isinstance(value, EnumValue):
        return value.type_name
    if isinstance(value, ExceptionValue):
        return value.type_name
    assert_never(value)  # pragma: no cover


def _render_raw(value: Value, name: str | None) -> str:
    """``as raw``: plain string conversion, no boundary markers (§2.12)."""
    return _scalar_text(value)


def _render_json(value: Value, name: str | None) -> str:
    """``as json``: always pretty JSON, no boundary markers."""
    return _pretty_json(value)


def _render_bullets(value: Value, name: str | None) -> str:
    """``as bullets``: list items as ``- item`` lines; others as JSON."""
    if isinstance(value, ListValue):
        lines = []
        for elem in value.elements:
            lines.append(f"- {_scalar_text(elem)}")
        return "\n".join(lines)
    return _pretty_json(value)


# Registry: renderer name → function.
_RENDERERS: dict[str, RendererFn] = {
    "default": _render_default,
    "raw": _render_raw,
    "json": _render_json,
    "bullets": _render_bullets,
}


def render_for_prompt(value: Value, *, renderer_name: str | None, var_name: str | None) -> str:
    """Render *value* for use inside a prompt template (§2.12).

    ``renderer_name``  — the ``as X`` override (``None`` → ``"default"``).
    ``var_name``       — the variable name of the interpolated expression,
                         used as the ``name=`` attribute in boundary tags.
                         ``None`` when the expression is not a simple VarRef.
    """
    name = renderer_name if renderer_name is not None else "default"
    fn = _RENDERERS.get(name)
    if fn is None:
        # Unknown renderer: fall back to default (should not happen after
        # typecheck validation, but is defensive here).
        fn = _render_default
    return fn(value, var_name)


# ---------------------------------------------------------------------------
# Console rendering for ``print`` (§11.12)
# ---------------------------------------------------------------------------


def render_for_console(value: Value) -> str:
    """Render *value* for ``print`` console output (§11.12).

    Rules:
    - ``text`` → verbatim (no boundary markers).
    - ``int``, ``decimal``, ``bool`` → scalar plain text.
    - ``list``, ``dict``, record, enum, ``json``, exception → pretty JSON.

    No boundary markers are ever added (those are for prompt interpolation).
    """
    if isinstance(value, TextValue):
        return value.value
    if isinstance(value, (IntValue, DecimalValue, BoolValue)):
        return _scalar_text(value)
    # Structured: pretty JSON.
    return _pretty_json(value)
