"""UI-free interactive core shared by both AgL REPL front ends.

:mod:`agm.agl.repl.console` (prompt_toolkit) and :mod:`agm.agl.repl.plain_console`
(plain line-oriented input/output) are thin front ends around the single
read-eval-print loop body defined here (:func:`run_repl_loop`), so meta-command
dispatch, theme-change handling, blank/comment no-ops, entry evaluation, and
result rendering exist exactly once. This module never imports prompt_toolkit;
it provides:

- :data:`PROMPT` / :data:`CONTINUATION` — the prompt spellings both front ends
  print (styled by the console, plain by :mod:`plain_console`);
- :func:`format_banner` — the startup banner;
- :func:`is_incomplete` — the multiline continuation predicate (delegates to
  the parser's structured incompleteness signal), shared so a pasted or
  editor-sent multi-line block accumulates identically on both front ends;
- :func:`make_console_confirm` — the stdlib-``input``/``print``-based agent-call
  confirmation callback, shared by both front ends;
- :func:`run_repl_loop` — the loop itself, parameterized by a reader/writer
  seam and an ``on_theme_change`` hook.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from agm.agl.parser import (
    has_open_raw_tail_block,
    has_unterminated_triple_quoted_string,
    is_incomplete_source,
)
from agm.agl.repl import meta as meta_mod
from agm.agl.repl import render as render_mod
from agm.agl.repl import session as session_mod
from agm.agl.repl.agentmode import AgentMode

if TYPE_CHECKING:
    from agm.agl.repl.agents import ConfirmDecision
    from agm.agl.repl.session import ReplSession


# ---------------------------------------------------------------------------
# Prompts and banner
# ---------------------------------------------------------------------------

PROMPT = "agl> "
CONTINUATION = "...> "


def format_banner(agent_mode: "AgentMode | None" = None) -> str:
    """Return the startup banner, noting the active agent-call mode.

    The first line is always ``AgL REPL …`` (a stable prefix other tooling and
    tests key on).  Subsequent lines state the prompt, how to get help, how to
    quit, and — when an :class:`AgentMode` is supplied — the current agent-call
    mode so the user knows up front whether live calls will prompt for
    confirmation.
    """
    lines = [
        "AgL REPL — an interactive read-eval-print loop for AgL.",
        f"  Enter AgL at the {PROMPT!r} prompt; a block continues on {CONTINUATION!r}.",
        "  Type :help for the meta-command list; :quit or Ctrl-D to exit.",
    ]
    if agent_mode is not None:
        if agent_mode.mode == "auto":
            lines.append("  Agent-call mode: auto (live calls fire without confirmation).")
        else:
            lines.append("  Agent-call mode: confirm (you approve each live agent call).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multiline continuation
# ---------------------------------------------------------------------------


def is_incomplete(text: str) -> bool:
    """Return ``True`` when *text* is a prefix of a valid entry (keep prompting).

    *text* is the current buffer content (no synthetic trailing newline).  Blank
    or whitespace-only input force-submits (returns ``False``) so pressing Enter
    on an empty prompt gives a fresh prompt rather than inserting a newline; the
    loop then no-ops the blank entry.  A trailing blank line — the user pressed
    Enter on an empty continuation line, so the buffer ends with ``\\n`` —
    likewise force-submits so the user can always escape a continuation even when
    the buffer is still syntactically incomplete.  Otherwise the structured
    parser signal decides, except that a registered raw-tail block stays open
    until its payload is closed by a blank line or dedent.

    Both front ends drive this same predicate — the prompt_toolkit console from
    its multiline Enter key binding, the plain console from its line-accumulating
    reader — so a pasted or editor-sent multi-line block accumulates identically
    on either.
    """
    if not text.strip():
        return False
    if text.endswith("\n") and not has_unterminated_triple_quoted_string(text):
        return False
    return is_incomplete_source(text) or has_open_raw_tail_block(text)


# ``has_runnable_statements`` (the blank/comment-only-entry predicate) lives in
# the UI-free ``session`` module so ``load_file`` can share it; re-exported here
# under its original name for the shared loop body and both front ends' tests.
has_runnable_statements = session_mod.has_runnable_statements


# ---------------------------------------------------------------------------
# Agent-call confirmation prompt
# ---------------------------------------------------------------------------

# How much of a rendered prompt to show inline before truncating; longer prompts
# offer a ``[v]iew`` option to print the full text.
_PROMPT_PREVIEW_CHARS = 200

# The reader the confirm prompt uses to read a line.  Injected so headless tests
# can script answers without a terminal; defaults to stdlib ``input``.
PromptReader = Callable[[str], str]


def make_console_confirm(
    *,
    reader: "PromptReader | None" = None,
    printer: Callable[[str], None] | None = None,
) -> "Callable[[str, str], ConfirmDecision]":
    """Return a confirm callback for :class:`~agm.agl.repl.agents.ConfirmingAgent`.

    The callback shows the *callee* and the rendered prompt (truncated, with a
    ``[v]iew`` option to print the full text), then reads ``[Y]es / [n]o /
    [a]lways`` and maps the answer to ``"yes"`` / ``"no"`` / ``"always"``.  An
    empty answer defaults to ``"yes"`` (the capitalised default).  Anything
    unrecognised re-asks.

    *reader* / *printer* are injected so headless tests drive it without a
    terminal; they default to stdlib ``input`` / ``print``.  Shared by both REPL
    front ends: neither needs styling for this prompt.
    """
    read: PromptReader = reader if reader is not None else input
    write: Callable[[str], None] = printer if printer is not None else print

    def confirm(callee: str, prompt: str) -> ConfirmDecision:
        write(f"Agent call to {callee!r}:")
        write(_preview_prompt(prompt))
        while True:
            answer = read("Run this agent call? [Y]es / [n]o / [a]lways: ").strip().lower()
            if answer in ("", "y", "yes"):
                return "yes"
            if answer in ("n", "no"):
                return "no"
            if answer in ("a", "always"):
                return "always"
            if answer in ("v", "view"):
                write(prompt)
                continue
            write("Please answer y(es), n(o), a(lways), or v(iew).")

    return confirm


def _preview_prompt(prompt: str) -> str:
    """Return the inline prompt preview, truncated with a ``[v]iew`` hint."""
    if len(prompt) <= _PROMPT_PREVIEW_CHARS:
        return prompt
    return f"{prompt[:_PROMPT_PREVIEW_CHARS]}… (truncated; type 'v' to view full)"


# ---------------------------------------------------------------------------
# The read-eval-print loop
# ---------------------------------------------------------------------------


def run_repl_loop(
    session: "ReplSession",
    *,
    reader: Callable[[], str],
    writer: Callable[[str], None],
    echo: bool = True,
    check_only: bool = False,
    agent_mode: "AgentMode | None" = None,
    theme: str = "auto",
    on_theme_change: Callable[[str], None] | None = None,
) -> None:
    """Run the read-eval-print loop against *session*; the core both front ends share.

    *reader* returns the next (possibly multiline) entry; it must raise
    ``EOFError`` on end of input (Ctrl-D, or closed stdin in plain mode) and may
    raise ``KeyboardInterrupt`` to cancel the entry in progress (Ctrl-C) without
    exiting the loop. *writer* prints one block of output text. Both are the
    front end's whole UI seam: the prompt_toolkit console wires
    ``prompt_session.prompt`` and ``print``; the plain console wires a
    line-accumulating reader over plain stdin/stdout and a plain ``print``-alike.

    A ``:`` line is routed to :func:`agm.agl.repl.meta.dispatch_meta`; a blank or
    comment-only entry is a no-op (fresh prompt, no error); any other entry is
    evaluated and its result rendered via
    :func:`agm.agl.repl.render.render_entry_result`.

    When *check_only* is set the REPL is in dry-run mode: each entry is run
    through the full static pipeline (parse / resolve / typecheck / match
    compilation) only — no evaluation, no agent/exec calls, and no bindings
    are persisted — and its inferred type is echoed.

    *theme* selects the initial colour palette; whenever ``:theme`` switches the
    active theme, *on_theme_change* is called with the new theme name. The
    prompt_toolkit console uses it to swap ``prompt_session.style`` and persist
    the choice; the plain console — which has no styling to swap — only
    persists.

    The loop is intentionally thin: formatting lives in ``render`` and meta
    handling in ``meta`` so both paths can evolve without touching it.
    """
    ctx = meta_mod.MetaContext(
        session=session,
        echo=echo,
        agent_mode=agent_mode if agent_mode is not None else AgentMode(),
        theme=theme,
    )

    writer(format_banner(ctx.agent_mode))
    current_theme = theme
    while True:
        try:
            entry = reader()
        except KeyboardInterrupt:
            # Ctrl-C cancels the current entry but never exits the REPL.
            continue
        except EOFError:
            # Ctrl-D (or closed stdin) exits.
            break

        if entry.lstrip().startswith(":"):
            outcome = meta_mod.dispatch_meta(entry, ctx)
            if outcome.text is not None:
                writer(outcome.text)
            if ctx.theme != current_theme:
                current_theme = ctx.theme
                if on_theme_change is not None:
                    on_theme_change(current_theme)
            if outcome.quit:
                break
            continue

        # Blank or comment-only entries have nothing to run; give a fresh prompt
        # without invoking the evaluator (whose parser would reject them).
        if not has_runnable_statements(entry):
            continue

        result = session.eval_entry(entry, check_only=check_only)
        rendered = render_mod.render_entry_result(result, echo=ctx.echo, check_only=check_only)
        if rendered is not None:
            writer(rendered)
