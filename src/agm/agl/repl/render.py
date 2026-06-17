"""Format an :class:`~agm.agl.repl.session.EntryResult` into REPL output text.

This is the **minimal** M2 renderer.  It turns the pure-data ``EntryResult``
into the plain-text lines the console prints after evaluating one entry.  It is
deliberately small and styling-free so M3 can enrich it (richer value
formatting, the ``:set echo`` toggle already threads through the ``echo`` flag)
without rewriting the loop.

Channels mirror ``agm exec`` so REPL and batch output read identically:

- error diagnostics  → ``line N: message``
- warnings           → ``warning: line N: message``
- runtime raise      → ``AgL exception: <Type>: <message> at line L, col C``

On success, when ``echo`` is on, an entry's outcome is echoed Python-REPL style:

- ``expression`` → the rendered value (via the runtime console renderer);
- ``binding``    → ``name : Type = value``;
- ``declaration``→ a terse ``<name> declared`` confirmation;
- ``statement``  → nothing (its own ``print`` output already went to stdout).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agm.agl.eval.values import Value
    from agm.agl.repl.session import EntryResult
    from agm.agl.typecheck.types import Type


def format_typed_value(name: str, value_type: "Type", value: "Value") -> str:
    """Format a single ``name : Type = value`` line.

    This is the single source of truth for the binding/value display shared by
    the entry-echo path (:func:`_render_echo`) and the ``:bindings`` / ``:params``
    meta-commands, so the two never drift in how a value is rendered.
    """
    from agm.agl.runtime.render import render_value_repl

    return f"{name} : {value_type!r} = {render_value_repl(value)}"


def render_entry_result(
    result: "EntryResult", *, echo: bool, check_only: bool = False
) -> str | None:
    """Return the text to print for *result*, or ``None`` when nothing to print.

    *echo* mirrors the session echo setting: when off, successful entries
    produce no echo line (errors and warnings are always reported regardless).
    *check_only* selects the dry-run echo: a check-only result has a type but no
    value, so the echo shows the inferred type instead of a value.
    """
    lines: list[str] = []

    # Warnings are advisory and always surfaced, on success or failure, ahead of
    # any error so the most actionable line (the error) is printed last.
    for diag in result.warnings:
        lines.append(f"warning: line {diag.line}: {diag.message}")

    if not result.ok:
        lines.extend(_render_failure(result))
        return "\n".join(lines) if lines else None

    if echo:
        echo_line = _render_check_only(result) if check_only else _render_echo(result)
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
    return [f"line {diag.line}: {diag.message}" for diag in result.diagnostics]


def _render_check_only(result: "EntryResult") -> str | None:
    """Render the dry-run (type-only) echo for *result*, or ``None``.

    A ``check_only`` run never evaluates, so there is no value — the echo shows
    the inferred static type: ``name : Type`` for a binding, ``: Type`` for a
    bare expression.  Declarations confirm the declared name; statements have no
    type to show and echo nothing.
    """
    if result.kind == "expression":
        # A bare expression always carries a checked type on success.
        assert result.value_type is not None
        return f": {result.value_type!r}"
    if result.kind == "binding":
        assert result.name is not None
        assert result.value_type is not None
        return f"{result.name} : {result.value_type!r}"
    if result.kind == "declaration":
        assert result.name is not None
        return f"{result.name} declared"
    # ``statement`` — nothing to show.
    return None


def _render_echo(result: "EntryResult") -> str | None:
    """Render the success echo line for *result*, or ``None`` for statements."""
    from agm.agl.runtime.render import render_value_repl

    if result.kind == "expression":
        # A bare expression always carries a value and type on success.
        assert result.value is not None
        assert result.value_type is not None
        return render_value_repl(result.value)
    if result.kind == "binding":
        # A binding echoes ``name : Type = value`` (single-sourced helper so the
        # echo and ``:bindings`` listing never diverge).
        assert result.name is not None
        assert result.value is not None
        assert result.value_type is not None
        return format_typed_value(result.name, result.value_type, result.value)
    if result.kind == "declaration":
        assert result.name is not None
        return f"{result.name} declared"
    # ``statement`` — nothing to echo (its own output already printed).
    return None
