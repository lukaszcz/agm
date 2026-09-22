"""AgL-native value rendering.

This module is the single implementation of value-to-text rendering used by
template interpolation, ``print``, ``as text``, the AgL ``render(...)`` builtin,
and REPL display.  Callers choose two display options:

- ``pretty``: render containers, nominal values, and JSON over multiple indented
  lines when ``True``; keep the output on one line when ``False``.
- ``quote_strings``: quote a top-level ``text`` value as an AgL string literal
  when ``True``; leave top-level text verbatim when ``False``.

Nested ``text`` values are always quoted so structured output remains parseable
as AgL surface syntax.  A record or exception renders its nominal's
``positional_fields`` bare, then the rest as ``name = value``.
"""

from __future__ import annotations

from agm.agl.ir.program import ValueDescriptors
from agm.agl.runtime.serialize import dumps_exact, value_to_json_obj
from agm.agl.semantics.cycles import enter_value
from agm.agl.semantics.values import (
    ArrayValue,
    BoolValue,
    ConstructorValue,
    DecimalValue,
    DictValue,
    ExceptionValue,
    IntValue,
    IrClosureValue,
    JsonValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
)
from agm.agl.value_syntax.lexical import quote_text, scalar_text


def _indent(level: int) -> str:
    return "  " * level


def _shift_after_first(text: str, *, level: int) -> str:
    """Indent every line after the first by *level* indentation levels."""
    if "\n" not in text:
        return text
    prefix = _indent(level)
    lines = text.split("\n")
    return "\n".join((lines[0], *(prefix + line for line in lines[1:])))


def _render_child(
    value: Value,
    descriptors: ValueDescriptors,
    *,
    pretty: bool,
    level: int,
    active: "set[int] | None",
) -> str:
    return _render(
        value,
        descriptors,
        pretty=pretty,
        quote_strings=False,
        top_level=False,
        level=level,
        active=active,
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


def _has_top_level_arrow(label: str) -> bool:
    square_depth = 0
    paren_depth = 0
    for index, char in enumerate(label):
        if char == "[":
            square_depth += 1
        elif char == "]" and square_depth > 0:
            square_depth -= 1
        elif char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth > 0:
            paren_depth -= 1
        elif (
            char == "-"
            and square_depth == 0
            and paren_depth == 0
            and label[index : index + 2] == "->"
        ):
            return True
    return False


def _render_function_signature(param_labels: tuple[str, ...], result_label: str) -> str:
    if not param_labels:
        params = "()"
    elif len(param_labels) == 1:
        param = param_labels[0]
        params = f"({param})" if _has_top_level_arrow(param) else param
    else:
        params = f"({', '.join(param_labels)})"
    return f"{params} -> {result_label}"


def _render(
    value: Value,
    descriptors: ValueDescriptors,
    *,
    pretty: bool,
    quote_strings: bool,
    top_level: bool,
    level: int,
    active: "set[int] | None" = None,
) -> str:
    if isinstance(value, TextValue):
        if top_level and not quote_strings:
            return value.value
        return quote_text(value.value)

    if isinstance(value, UnitValue):
        return "()"

    if isinstance(value, (IntValue, DecimalValue, BoolValue)):
        return scalar_text(value.value)

    if isinstance(value, ConstructorValue):
        return f"<constructor {descriptors.nominals[value.nominal].display_name}>"

    if isinstance(value, IrClosureValue):
        function_desc = descriptors.functions[value.function_id]
        param_labels = function_desc.param_labels or ("?",) * len(function_desc.params)
        signature = _render_function_signature(param_labels, function_desc.result_label)
        return f"<function: {signature}>"

    if isinstance(value, JsonValue):
        rendered = dumps_exact(value_to_json_obj(value), indent=2 if pretty else None)
        return _shift_after_first(rendered, level=level) if pretty else rendered

    if isinstance(value, ArrayValue):
        active = enter_value(id(value), active)
        try:
            items = [
                _render_child(element, descriptors, pretty=pretty, level=level + 1, active=active)
                for element in value.elements
            ]
        finally:
            active.discard(id(value))
        return _render_sequence("[", "]", items, level=level, pretty=pretty)

    if isinstance(value, DictValue):
        active = enter_value(id(value), active)
        try:
            items = []
            for key, child in value.entries.items():
                rendered = _render_child(
                    child, descriptors, pretty=pretty, level=level + 1, active=active
                )
                items.append(f"{quote_text(key)}: {rendered}")
        finally:
            active.discard(id(value))
        return _render_sequence("{", "}", items, level=level, pretty=pretty)

    if isinstance(value, (RecordValue, ExceptionValue)):
        descriptor = descriptors.nominals[value.nominal]
        display_name = descriptor.display_name
        # A nullary constructor is an auto-value, so a fieldless record's bare
        # spelling round-trips as written: every fieldless record renders bare,
        # whether it is an enum member (`E::A`) or a standalone record (`Root`).
        # A fieldless exception keeps its parens, and neither can be part of a
        # cycle, so both answer before the guard is entered.
        if not value.fields:
            return display_name if isinstance(value, RecordValue) else f"{display_name}()"
        active = enter_value(id(value), active)
        try:
            positional: list[str] = []
            named: list[str] = []
            for name, child in value.fields.items():
                rendered = _render_child(
                    child, descriptors, pretty=pretty, level=level + 1, active=active
                )
                if name in descriptor.positional_fields:
                    positional.append(rendered)
                else:
                    named.append(f"{name} = {rendered}")
            items = positional + named
        finally:
            active.discard(id(value))
        return _render_sequence(f"{display_name}(", ")", items, level=level, pretty=pretty)

    raise RuntimeError(f"render: unhandled value type {type(value).__name__}")  # pragma: no cover


def render_value(
    value: Value,
    descriptors: ValueDescriptors,
    *,
    pretty: bool = False,
    quote_strings: bool = False,
) -> str:
    """Render *value* to AgL text, reading static spellings/labels from *descriptors*.

    ``pretty=False`` keeps output single-line where possible. ``pretty=True``
    expands structured values and JSON over multiple lines with two-space
    indentation. ``quote_strings`` only controls top-level ``text`` values;
    nested text is always quoted. ``unit`` always renders ``()``.
    """
    return _render(
        value,
        descriptors,
        pretty=pretty,
        quote_strings=quote_strings,
        top_level=True,
        level=0,
    )
