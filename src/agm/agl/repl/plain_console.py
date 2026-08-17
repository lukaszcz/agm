"""Plain, styling-free line front end for the AgL REPL.

A sibling of :mod:`agm.agl.repl.console`: it shares the exact same
read-eval-print loop body (:func:`agm.agl.repl.loop.run_repl_loop`) and the
same prompt spellings (:data:`agm.agl.repl.loop.PROMPT` /
:data:`~agm.agl.repl.loop.CONTINUATION`), but reads lines from a plain text
stream instead of driving prompt_toolkit, and prints them with no styling or
ANSI escapes. This lets a comint buffer, a pipe, or any other non-terminal
consumer drive the REPL. This module never imports prompt_toolkit.

It also owns the engagement predicate (:func:`plain_mode_engaged`) deciding
when ``agm repl`` should use this front end instead of the rich console.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, TextIO

from agm.agl.repl.loop import CONTINUATION, PROMPT, is_incomplete, run_repl_loop

if TYPE_CHECKING:
    from agm.agl.repl.agentmode import AgentMode
    from agm.agl.repl.session import ReplSession


def plain_mode_engaged(*, stdin: TextIO, stdout: TextIO, env: Mapping[str, str]) -> bool:
    """Return whether ``agm repl`` should use the plain line front end.

    Engages when either *stdin* or *stdout* is not a real terminal, or when
    ``env["TERM"] == "dumb"`` (a terminal or embedding editor that presents a
    pty but cannot usefully render prompt_toolkit's styling). This is the
    auto-detection half of ``--plain``; the CLI ORs this predicate with the
    explicit flag, so there is no corresponding ``--no-plain`` — forcing
    prompt_toolkit onto a pipe has no valid use. No editor-specific detection
    is performed anywhere.
    """
    if env.get("TERM") == "dumb":
        return True
    return not (stdin.isatty() and stdout.isatty())


class PlainReader:
    """Read one (possibly multiline) entry from *stdin*, prompting on *stdout*.

    Accumulates continuation lines using :func:`agm.agl.repl.loop.is_incomplete`
    — the same predicate the prompt_toolkit Enter handler uses — so a pasted or
    editor-sent multi-line block accumulates exactly like the rich console's
    multiline entry: each line is appended and, if the accumulated text is
    still an incomplete prefix, another line is read after printing
    :data:`~agm.agl.repl.loop.CONTINUATION`. Closed *stdin* (EOF) raises
    ``EOFError`` — matching Ctrl-D — even mid-entry, so a pipe that closes
    exits the REPL cleanly rather than hanging. Over a pty, Ctrl-C interrupts
    the blocked read with ``KeyboardInterrupt``; a newline is written first so
    the loop's fresh prompt starts on its own line instead of gluing onto the
    cancelled entry's echoed input (matching the rich console, which is
    already on a fresh line by the time Ctrl-C reaches it).
    """

    def __init__(self, *, stdin: TextIO, stdout: TextIO) -> None:
        self._stdin = stdin
        self._stdout = stdout

    def __call__(self) -> str:
        text = ""
        prompt = PROMPT
        while True:
            self._stdout.write(prompt)
            self._stdout.flush()
            try:
                raw = self._stdin.readline()
            except KeyboardInterrupt:
                self._stdout.write("\n")
                self._stdout.flush()
                raise
            if raw == "":
                raise EOFError
            candidate = text + raw.rstrip("\n")
            if not is_incomplete(candidate):
                return candidate
            text = candidate + "\n"
            prompt = CONTINUATION


def run_plain_console(
    session: "ReplSession",
    *,
    echo: bool = True,
    check_only: bool = False,
    agent_mode: "AgentMode | None" = None,
    theme: str = "auto",
    on_theme_save: "Callable[[str], None] | None" = None,
    stdin: TextIO,
    stdout: TextIO,
) -> None:
    """Run the AgL REPL against *session* using plain, styling-free line input.

    Shares :func:`agm.agl.repl.console.run_console`'s public shape minus the
    prompt_toolkit-only parameters (``input``/``output``/``history_path``,
    which have no plain-mode equivalent — there is no command history in this
    front end): *stdin* / *stdout* replace prompt_toolkit's ``Input``/``Output``
    objects with plain text streams, so tests can drive it with ``StringIO`` or
    a pipe. *theme* only affects what ``:theme`` persists via *on_theme_save*,
    since plain output carries no colour to swap.
    """
    reader = PlainReader(stdin=stdin, stdout=stdout)

    def writer(text: str) -> None:
        stdout.write(text + "\n")
        stdout.flush()

    def on_theme_change(new_theme: str) -> None:
        if on_theme_save is not None:
            on_theme_save(new_theme)

    run_repl_loop(
        session,
        reader=reader,
        writer=writer,
        echo=echo,
        check_only=check_only,
        agent_mode=agent_mode,
        theme=theme,
        on_theme_change=on_theme_change,
    )
