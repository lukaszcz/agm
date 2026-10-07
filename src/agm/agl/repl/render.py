"""Format an :class:`~agm.agl.repl.session.EntryResult` into REPL output text.

This renderer turns the pure-data ``EntryResult``
into the plain-text lines the console prints after evaluating one entry.  It is
deliberately small and styling-free so richer value formatting and echo handling
can evolve without rewriting the loop.

Channels mirror ``agm exec`` severity formatting; REPL diagnostics omit a
source filename:

- error diagnostics  → ``N:C: error: message``
- warnings           → ``N:C: warning: message``
- runtime raise      → ``AgL exception: <Type>: <message> at path:L:C`` (falls
  back to ``at line L[, col C]`` when the raise site's source is unknown)

On success, when ``echo`` is on, an entry's outcome is echoed Python-REPL style:

- ``expression`` → the rendered value (pretty display rendering);
- ``binding``    → ``name : Type = value``;
- ``declaration``→ a terse ``<name> declared`` confirmation;
- ``statement``  → nothing (its own ``print`` output already went to stdout).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from agm.agl.diagnostics import format_diagnostic

if TYPE_CHECKING:
    from agm.agl.ir.program import ValueDescriptors
    from agm.agl.repl.entry import EntryResult
    from agm.agl.semantics.types import Type
    from agm.agl.semantics.values import Value


def _render_value_or_cyclic_message(
    value: "Value", descriptors: "ValueDescriptors", *, pretty: bool, quote_strings: bool
) -> str:
    """Render *value*, or the ``AgL exception: CyclicValueError: ...`` line.

    Reference semantics lets a value bound at the REPL prompt become cyclic
    across entries (e.g. an indexed assignment that closes a cycle in a prior
    entry), so echoing an already-successful entry's value can still hit a
    cycle. There is no ``try``/``catch`` scope around a REPL echo — it runs
    after the entry's own evaluation already completed — so this reports the
    same one-line message a runtime raise would, via ``RunError.to_message``,
    rather than crashing the session.
    """
    from agm.agl.pipeline import RunError
    from agm.agl.runtime.render import render_value
    from agm.agl.semantics.cycles import CYCLE_MESSAGE, AglCyclicValue

    try:
        return render_value(value, descriptors, pretty=pretty, quote_strings=quote_strings)
    except AglCyclicValue:
        return RunError(
            type_name="CyclicValueError",
            fields={"message": CYCLE_MESSAGE},
        ).to_message()


def format_typed_value(
    name: "str | None", value_type: "Type", value: "Value", descriptors: "ValueDescriptors"
) -> str:
    """Format a ``name : Type = value`` line, or ``: Type = value`` when unnamed.

    This is the single source of truth for the binding/value display shared by
    the entry-echo path (:func:`_render_echo`) and the ``:bindings``
    meta-command, so the two never drift in how a value is rendered. A value
    that turned cyclic since its entry ran reports as a runtime-error line instead (see
    :func:`_render_value_or_cyclic_message`).
    """
    prefix = f"{name} :" if name is not None else ":"
    rendered = _render_value_or_cyclic_message(value, descriptors, pretty=True, quote_strings=True)
    return f"{prefix} {value_type!r} = {rendered}"


def _is_unit_entry(result: "EntryResult") -> bool:
    """Whether *result*'s checked type is ``unit`` — an entry that echoes nothing.

    Decided from the entry's checked static type, never from a value-carried flag.
    """
    from agm.agl.semantics.types import UnitType

    return result.kind in ("expression", "binding") and isinstance(result.value_type, UnitType)


def render_entry_result(
    result: "EntryResult",
    *,
    echo: bool,
    echo_unit: bool = False,
) -> str | None:
    """Return the text to print for *result*, or ``None`` when nothing to print.

    *echo* mirrors the session echo setting: when off, successful entries
    produce no echo line (errors and warnings are always reported regardless).
    A ``unit``-typed expression or binding never echoes (see
    :func:`_is_unit_entry`), unless *echo_unit* is set — the single place that
    decision is gated.
    """
    lines: list[str] = []

    # Warnings are advisory and always surfaced, on success or failure, ahead of
    # any error so the most actionable line (the error) is printed last.
    for diag in result.warnings:
        lines.append(format_diagnostic(diag, source_name=None))

    if not result.ok:
        lines.extend(_render_failure(result))
        if result.installed:
            lines.append(f"Installed before failure: {', '.join(result.installed)}")
        return "\n".join(lines) if lines else None

    if echo and (echo_unit or not _is_unit_entry(result)):
        echo_line = _render_echo(result)
        if echo_line is not None:
            lines.append(echo_line)

    return "\n".join(lines) if lines else None


def _render_failure(result: "EntryResult") -> list[str]:
    """Render a failed entry: a runtime raise, else pre-execution diagnostics.

    A runtime raise uses ``RunError.to_message`` — the same formatter ``agm
    exec`` prints (the REPL omits the trace id, which exec includes for
    correlation) — so the two never diverge.
    """
    if result.error is not None:
        return [result.error.to_message()]
    return [format_diagnostic(diag, source_name=None) for diag in result.diagnostics]


def _render_echo(result: "EntryResult") -> str | None:
    """Render the success echo line for *result*, or ``None`` for statements."""
    if result.kind == "type":
        # A bare type expression entered at the prompt is not a value; echo it
        # in the same ``<…>`` surface form used for other non-value echoes
        # (functions, agents, constructors) and tag it as a type.
        from agm.agl.repl.type_display import (
            format_type_echo_for_repl,
            format_type_text_echo_for_repl,
        )

        if result.type_display is not None:
            return format_type_text_echo_for_repl(result.type_display)
        return format_type_echo_for_repl(cast("Type", result.value_type), result.type_table)
    if result.kind == "expression":
        # A successful expression carries a value and descriptor view.
        return _render_value_or_cyclic_message(
            cast("Value", result.value),
            cast("ValueDescriptors", result.descriptors),
            pretty=True,
            quote_strings=result.quote_strings,
        )
    if result.kind == "binding":
        # A successful binding carries a value, type, and descriptor view; its
        # display is shared with ``:bindings``.
        return format_typed_value(
            result.name,
            cast("Type", result.value_type),
            cast("Value", result.value),
            cast("ValueDescriptors", result.descriptors),
        )
    if result.kind == "declaration":
        return f"{result.name} declared"
    # ``statement`` — nothing to echo (its own output already printed).
    return None
