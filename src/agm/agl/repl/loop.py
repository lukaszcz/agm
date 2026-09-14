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

if TYPE_CHECKING:
    from agm.agl.repl.session import ReplSession


# ---------------------------------------------------------------------------
# Prompts and banner
# ---------------------------------------------------------------------------

PROMPT = "agl> "
CONTINUATION = "...> "


def format_banner() -> str:
    """Return the startup banner.

    The first line is always ``AgL REPL …`` (a stable prefix other tooling and
    tests key on); the rest state the prompt, how to get help, and how to quit.
    """
    return "\n".join(
        [
            "AgL REPL — an interactive read-eval-print loop for AgL.",
            f"  Enter AgL at the {PROMPT!r} prompt; a block continues on {CONTINUATION!r}.",
            "  Type :help for the meta-command list; :quit or Ctrl-D to exit.",
        ]
    )


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
# The read-eval-print loop
# ---------------------------------------------------------------------------


def run_repl_loop(
    session: "ReplSession",
    *,
    reader: Callable[[], str],
    writer: Callable[[str], None],
    echo: bool = True,
    check_only: bool = False,
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
        theme=theme,
    )

    writer(format_banner())
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
