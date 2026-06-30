"""AgL-native value rendering.

This module is the single implementation of value-to-text rendering used by
template interpolation, ``print``, ``as text``, the AgL ``render(...)`` builtin,
and REPL display.  Callers choose two display options:

- ``pretty``: render containers, nominal values, and JSON over multiple indented
  lines when ``True``; keep the output on one line when ``False``.
- ``quote_strings``: quote a top-level ``text`` value as an AgL string literal
  when ``True``; leave top-level text verbatim when ``False``.

Nested ``text`` values are always quoted so structured output remains parseable
as AgL surface syntax.  Nominal values (record, enum, exception) carry fields in
declaration order already, so rendering walks ``value.fields`` directly.
"""

from __future__ import annotations

from agm.agl.runtime.serialize import dumps_exact, value_to_json_obj
from agm.agl.semantics.values import (
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


def _indent(level: int) -> str:
    return "  " * level


def _quote_text(s: str) -> str:
    """Return *s* as a double-quoted AgL string literal surface form."""
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
    """Render an int, decimal, or bool value as plain text."""
    if isinstance(value, IntValue):
        return str(value.value)
    if isinstance(value, DecimalValue):
        # Drop trailing zeros without using scientific notation.
        return format(value.value.normalize(), "f")
    return "true" if value.value else "false"


def _shift_after_first(text: str, *, level: int) -> str:
    """Indent every line after the first by *level* indentation levels."""
    if "\n" not in text:
        return text
    prefix = _indent(level)
    lines = text.split("\n")
    return "\n".join((lines[0], *(prefix + line for line in lines[1:])))


def _render_child(value: Value, *, pretty: bool, level: int) -> str:
    return _render(
        value,
        pretty=pretty,
        quote_strings=False,
        top_level=False,
        level=level,
    )


def _render_sequence(
    open_token: str,
    close_token: str,
    items: list[str],
    *,
    level: int,
    pretty: bool,
) -> str:
    if not items:
        return f"{open_token}{close_token}"
    if not pretty:
        return f"{open_token}" + ", ".join(items) + f"{close_token}"
    item_indent = _indent(level + 1)
    close_indent = _indent(level)
    body = ",\n".join(item_indent + item for item in items)
    return f"{open_token}\n{body}\n{close_indent}{close_token}"


def _render(value: Value, *, pretty: bool, quote_strings: bool, top_level: bool, level: int) -> str:
    if isinstance(value, TextValue):
        if top_level and not quote_strings:
            return value.value
        return _quote_text(value.value)

    if isinstance(value, UnitValue):
        if not value.printable_in_repl:
            return "void"
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
        param_labels = value.param_labels or ("?",) * value.arity
        return f"<function: ({', '.join(param_labels)}) -> {value.result_label}>"

    if isinstance(value, JsonValue):
        rendered = dumps_exact(value_to_json_obj(value), indent=2 if pretty else None)
        return _shift_after_first(rendered, level=level) if pretty else rendered

    if isinstance(value, ListValue):
        items = [
            _render_child(element, pretty=pretty, level=level + 1)
            for element in value.elements
        ]
        return _render_sequence("[", "]", items, level=level, pretty=pretty)

    if isinstance(value, DictValue):
        items = [
            f"{_quote_text(key)}: {_render_child(child, pretty=pretty, level=level + 1)}"
            for key, child in value.entries.items()
        ]
        return _render_sequence("{", "}", items, level=level, pretty=pretty)

    if isinstance(value, RecordValue):
        items = [
            f"{name} = {_render_child(child, pretty=pretty, level=level + 1)}"
            for name, child in value.fields.items()
        ]
        return _render_sequence(
            f"{value.display_name}(", ")", items, level=level, pretty=pretty
        )

    if isinstance(value, (EnumValue, ExceptionValue)):
        prefix = (
            f"{value.display_name}.{value.variant}"
            if isinstance(value, EnumValue)
            else value.display_name
        )
        if not value.fields:
            return prefix if isinstance(value, EnumValue) else f"{prefix}()"
        items = [
            f"{name} = {_render_child(child, pretty=pretty, level=level + 1)}"
            for name, child in value.fields.items()
        ]
        return _render_sequence(f"{prefix}(", ")", items, level=level, pretty=pretty)

    raise RuntimeError(f"render: unhandled value type {type(value).__name__}")  # pragma: no cover


def render_value(
    value: Value,
    *,
    pretty: bool = False,
    quote_strings: bool = False,
) -> str:
    """Render *value* to AgL text.

    ``pretty=False`` keeps output single-line where possible. ``pretty=True``
    expands structured values and JSON over multiple lines with two-space
    indentation. ``quote_strings`` only controls top-level ``text`` values;
    nested text is always quoted.
    """
    return _render(
        value,
        pretty=pretty,
        quote_strings=quote_strings,
        top_level=True,
        level=0,
    )
