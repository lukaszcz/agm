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

3. **Shell-exec rendering** for ``exec`` templates (§4.12, §11.13):
   - Default: render the value to its plain-text representation (scalar text,
     compact JSON for structured values) then pass through ``shlex.quote``.
   - ``as raw``: bypass quoting, inserting the plain text verbatim — the
     caller is responsible for injection safety.
   - Other explicit renderers (e.g. ``as json``) are applied and then quoted
     unless the renderer is ``"raw"``.

The ``render_for_prompt``, ``render_for_console``, and ``render_for_shell``
functions are the main entry points used by the interpreter.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable, Mapping
from typing import assert_never

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
from agm.agl.runtime.serialize import dumps_exact, value_to_json_obj


def _pretty_json(value: Value) -> str:
    """Render *value* as pretty-printed JSON (2-space indent).

    Uses the shared exact serializer so ``Decimal`` values are emitted as exact
    unquoted numeric text (never routed through binary ``float``; design §5.1).
    """
    return dumps_exact(value_to_json_obj(value), indent=2)


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
        return dumps_exact(value.raw, indent=None)
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

    The ``type=`` attribute in structured-value boundary tags follows the §2.12
    convention:

    - Scalar values: ``"text"``, ``"int"``, ``"decimal"``, ``"bool"``, ``"json"``
    - Homogeneous collections: ``"list"``, ``"dict"``
    - Named types (records, enums, exceptions): the declared type name
      (e.g. ``"Review"``, ``"Status"``, ``"AgentParseError"``)
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

# Public constant: the set of built-in renderer names.
# ``WorkflowRuntime.run`` uses this as the authoritative source so that
# ``HostCapabilities.renderer_names`` is derived from the implementation,
# not from a duplicated literal in the runtime layer (CARRY-IN 1, M3b).
RENDERER_NAMES: frozenset[str] = frozenset(_RENDERERS)


def builtin_renderers() -> dict[str, RendererFn]:
    """Return a fresh mapping of the built-in renderer name → function.

    ``WorkflowRuntime.run`` merges this with any host-registered renderers to
    form the authoritative ``renderers`` table threaded into the interpreter,
    so registered renderers are actually invoked at interpolation time
    (F1, M3b).  A fresh copy is returned so callers cannot mutate the
    module-level registry.
    """
    return dict(_RENDERERS)


def render_for_prompt(
    value: Value,
    *,
    renderer_name: str | None,
    var_name: str | None,
    renderers: Mapping[str, RendererFn] | None = None,
) -> str:
    """Render *value* for use inside a prompt template (§2.12).

    ``renderer_name``  — the ``as X`` override (``None`` → ``"default"``).
    ``var_name``       — the variable name of the interpolated expression,
                         used as the ``name=`` attribute in boundary tags.
                         ``None`` when the expression is not a simple VarRef.
    ``renderers``      — the name → function table to resolve ``renderer_name``
                         against.  Built by ``WorkflowRuntime.run`` as
                         ``{**builtin_renderers(), **registered}`` so that
                         host-registered renderers are honoured (F1, M3b).
                         ``None`` falls back to the built-in renderers only.
    """
    if renderers is None:
        renderers = _RENDERERS
    name = renderer_name if renderer_name is not None else "default"
    fn = renderers.get(name)
    if fn is None:
        # Loud internal error.  After type-checking this is unreachable through
        # ``WorkflowRuntime.run``: the checker validates every explicit
        # ``as <name>`` against the registered renderer set, and ``default`` is
        # always present in ``renderers``.  A miss here means the renderers
        # table is inconsistent with the checker's capabilities — an internal
        # invariant violation, never a user-facing fallback (F2, M3b).
        raise AssertionError(
            f"Renderer {name!r} is not in the renderers table; the checker must "
            "reject unknown renderers before evaluation."
        )
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


# ---------------------------------------------------------------------------
# Shell-exec rendering for ``exec`` templates (§4.12, §11.13)
# ---------------------------------------------------------------------------


def _shell_plain_text(value: Value) -> str:
    """Render a value to its plain-text shell representation (no boundary markers).

    Scalars become plain text; structured values become compact (single-line) JSON
    so the shell sees a single token before quoting.
    """
    if isinstance(value, (TextValue, IntValue, DecimalValue, BoolValue)):
        return _scalar_text(value)
    # Structured values: compact JSON (no indent) so the whole thing is one token.
    from agm.agl.runtime.serialize import dumps_exact

    return dumps_exact(value_to_json_obj(value), indent=None)


def render_for_shell(
    value: Value,
    *,
    renderer_name: str | None,
    renderers: Mapping[str, RendererFn] | None = None,
) -> str:
    """Render *value* for use inside an ``exec`` shell template (§4.12, §11.13).

    ``renderer_name``
        The ``as X`` override on the interpolation segment (``None`` → default
        shell-quoted rendering).  ``"raw"`` bypasses quoting entirely.
        Any other explicit renderer (e.g. ``"json"``) is applied first, then the
        result is shell-quoted.
    ``renderers``
        The full renderer table (built-ins + host-registered).  ``None`` falls
        back to the built-in renderers.

    Shell-quoting semantics (design §4.12):
    - Default (``None`` or ``"default"``): render to plain text then
      ``shlex.quote`` — injection-safe.
    - ``as raw``: insert the plain text verbatim — quoting is bypassed;
      the caller accepts the injection risk.
    - ``as <other>``: apply the named renderer then ``shlex.quote`` the result.
    """
    if renderer_name is None or renderer_name == "default":
        # Default: plain text → shlex.quote.
        return shlex.quote(_shell_plain_text(value))

    if renderer_name == "raw":
        # ``as raw``: bypass quoting entirely, insert verbatim.
        return _shell_plain_text(value)

    # Explicit non-raw renderer: apply it (possibly with boundary tags if someone
    # registered a custom renderer), then shell-quote the whole result.
    if renderers is None:
        renderers = _RENDERERS
    fn = renderers.get(renderer_name)
    if fn is None:
        raise AssertionError(
            f"Renderer {renderer_name!r} is not in the renderers table; the checker must "
            "reject unknown renderers before evaluation."
        )
    rendered = fn(value, None)
    return shlex.quote(rendered)
