"""Tests for the plain, styling-free REPL front end (``agm.agl.repl.plain_console``).

Drives :func:`run_plain_console` with plain ``io.StringIO`` streams — no
prompt_toolkit harness needed — and exercises the engagement predicate
directly. Assertions check user-visible plain-text output, never internals.
"""

from __future__ import annotations

import io
import subprocess
import sys
from collections.abc import Callable

import pytest

from agm.agl.repl import ReplSession
from agm.agl.repl.plain_console import PlainReader, plain_mode_engaged, run_plain_console

# ---------------------------------------------------------------------------
# prompt_toolkit containment
# ---------------------------------------------------------------------------


class TestPromptToolkitContainment:
    """The plain front end must never import prompt_toolkit, in-process import
    caching (this test file, ``conftest.py``, and every other test module
    already imported by the time these tests run) makes an in-process
    ``sys.modules`` check meaningless, so each check runs in a fresh
    subprocess that imports nothing else first.
    """

    def test_plain_console_does_not_import_prompt_toolkit(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import agm.agl.repl.plain_console, sys; print('prompt_toolkit' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "False"

    def test_loop_does_not_import_prompt_toolkit(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import agm.agl.repl.loop, sys; print('prompt_toolkit' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "False"


class _FakeStream:
    """A minimal stand-in for a stream, exposing only ``isatty``."""

    def __init__(self, is_tty: bool) -> None:
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


# ---------------------------------------------------------------------------
# Engagement predicate
# ---------------------------------------------------------------------------


class TestPlainModeEngaged:
    def test_tty_stdin_and_stdout_uses_the_rich_console(self) -> None:
        assert (
            plain_mode_engaged(stdin=_FakeStream(True), stdout=_FakeStream(True), env={}) is False
        )

    def test_non_tty_stdin_engages_plain(self) -> None:
        assert (
            plain_mode_engaged(stdin=_FakeStream(False), stdout=_FakeStream(True), env={}) is True
        )

    def test_non_tty_stdout_engages_plain(self) -> None:
        assert (
            plain_mode_engaged(stdin=_FakeStream(True), stdout=_FakeStream(False), env={}) is True
        )

    def test_term_dumb_engages_plain_even_on_a_tty(self) -> None:
        assert (
            plain_mode_engaged(
                stdin=_FakeStream(True), stdout=_FakeStream(True), env={"TERM": "dumb"}
            )
            is True
        )

    def test_both_non_tty_engages_plain(self) -> None:
        assert (
            plain_mode_engaged(stdin=_FakeStream(False), stdout=_FakeStream(False), env={}) is True
        )


# ---------------------------------------------------------------------------
# Driver helper
# ---------------------------------------------------------------------------


def drive_plain(
    input_text: str,
    *,
    session: ReplSession | None = None,
    echo: bool = True,
    check_only: bool = False,
    theme: str = "auto",
    on_theme_save: Callable[[str], None] | None = None,
) -> str:
    """Feed *input_text* to a headless plain REPL and return everything it printed.

    ``io.StringIO`` always raises EOF once exhausted, so there is no risk of
    hanging: an entry stream with no trailing ``:quit`` simply runs to a clean
    EOF exit, exactly like a closed pipe.
    """
    repl_session = session if session is not None else ReplSession()
    stdin = io.StringIO(input_text)
    stdout = io.StringIO()
    run_plain_console(
        repl_session,
        echo=echo,
        check_only=check_only,
        theme=theme,
        on_theme_save=on_theme_save,
        stdin=stdin,
        stdout=stdout,
    )
    return stdout.getvalue()


# ---------------------------------------------------------------------------
# Loop: prompts / submit / exit
# ---------------------------------------------------------------------------


class TestPlainLoop:
    def test_banner_is_printed(self) -> None:
        assert "AgL REPL" in drive_plain("")

    def test_primary_prompt_is_printed(self) -> None:
        assert "agl> " in drive_plain("")

    def test_eof_with_no_input_exits_cleanly(self) -> None:
        # No ":quit", no trailing newline: readline() hits EOF immediately.
        output = drive_plain("")
        assert "AgL REPL" in output

    def test_single_expression_submits_and_echoes(self) -> None:
        output = drive_plain('"hi"\n')
        assert "hi" in output

    def test_quit_meta_exits(self) -> None:
        output = drive_plain(":quit\nlet x = 1\n")
        # The entry after :quit is never read.
        assert "x : int = 1" not in output

    def test_exit_meta_exits(self) -> None:
        output = drive_plain(":exit\n")
        assert "AgL REPL" in output

    def test_whitespace_only_entry_is_ignored(self) -> None:
        output = drive_plain("   \nlet x = 1\n")
        assert "x : int = 1" in output

    def test_comment_only_entry_is_noop_no_error(self) -> None:
        output = drive_plain("# just a comment\nlet x = 1\n")
        assert ": error:" not in output.lower()
        assert "x : int = 1" in output


# ---------------------------------------------------------------------------
# Ctrl-C
# ---------------------------------------------------------------------------


class _InterruptOnceStdin(io.StringIO):
    """A stdin stand-in whose first ``readline()`` raises ``KeyboardInterrupt``.

    Every later ``readline()`` behaves like a normal ``StringIO`` reading
    *remaining* — simulating a pty where Ctrl-C interrupts one blocked read
    and typing resumes afterward.
    """

    def __init__(self, remaining: str) -> None:
        super().__init__(remaining)
        self._raised = False

    def readline(self, size: int = -1) -> str:
        if not self._raised:
            self._raised = True
            raise KeyboardInterrupt
        return super().readline(size)


class TestPlainReaderCtrlC:
    def test_reader_writes_newline_before_reraising(self) -> None:
        # PlainReader.__call__ must itself propagate KeyboardInterrupt (the
        # loop is what decides to catch and continue), but write a fresh
        # newline to stdout first so the cancelled entry's echoed input is not
        # glued to the next prompt.
        stdout = io.StringIO()
        reader = PlainReader(stdin=_InterruptOnceStdin(""), stdout=stdout)
        try:
            reader()
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("expected KeyboardInterrupt to propagate")
        assert stdout.getvalue() == "agl> \n"

    def test_ctrl_c_starts_a_new_line_then_loop_continues(self) -> None:
        # End to end through run_plain_console: the interrupted prompt gets its
        # own line, and the REPL keeps going — the next entry still evaluates.
        stdin = _InterruptOnceStdin("let x = 1\n")
        stdout = io.StringIO()
        run_plain_console(ReplSession(), stdin=stdin, stdout=stdout)
        output = stdout.getvalue()
        assert "agl> \nagl> " in output  # fresh line, not glued to next prompt
        assert "x : int = 1" in output


# ---------------------------------------------------------------------------
# Multiline continuation
# ---------------------------------------------------------------------------


class TestPlainMultiline:
    def test_continuation_prompt_appears_for_a_multiline_declaration(self) -> None:
        output = drive_plain("record R\n  x: int\n")
        assert "...> " in output

    def test_multiline_declaration_accumulates_and_declares(self) -> None:
        output = drive_plain("record R\n  x: int\n")
        assert "R declared" in output

    def test_blank_continuation_line_force_submits_incomplete(self) -> None:
        output = drive_plain("record R\n\n")
        assert "declared" not in output
        assert ": error:" in output.lower()

    def test_editor_sent_block_is_read_as_one_entry(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # An `if'/`else' block is complete after its first branch, so a reader
        # that submits at the first parseable prefix splits it in three and
        # reports the `else' as a stray. This is what `C-c C-r' sends.
        output = drive_plain(
            'let x = 5\nif x > 3 =>\n  print("big")\nelse =>\n  print("small")\n\n'
        )
        assert "error" not in output.lower()
        assert capsys.readouterr().out.split() == ["big"]

    def test_indented_line_keeps_the_entry_open(self, capsys: pytest.CaptureFixture[str]) -> None:
        # A layout block can always take one more line, so an indented last
        # line leaves the entry open however well the text parses: a `case'
        # parses after its first branch, and its second one is a stray on its
        # own.
        output = drive_plain('case 1 of\n  | 1 => print("one")\n  | _ => print("other")\n\n')
        assert "error" not in output.lower()
        assert capsys.readouterr().out.split() == ["one"]

    def test_pending_block_is_evaluated_at_end_of_input(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A pipe that closes without the terminating blank line still runs the
        # block it sent, rather than dropping it.
        drive_plain('if true =>\n  print("done")\n')
        assert capsys.readouterr().out.split() == ["done"]

    def test_dangling_operator_entry_continues_and_evaluates(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A line ending with an operator leaves the entry incomplete, so the
        # REPL prompts for more — and the continuation it invited must lex.
        output = drive_plain(
            "print ([1, 2, 3].map(fn v => v + 1) |>\n.fold(0, fn (t, v) => t + v))\n"
        )
        assert "...> " in output
        assert "error" not in output.lower()
        assert capsys.readouterr().out.split() == ["9"]

    def test_binder_suite_entry_continues_and_evaluates(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # `let x =` is incomplete on its own, so the REPL prompts for the suite
        # that supplies the binding's value.
        output = drive_plain("let total =\n  let a = 2\n  a + 3\nprint total\n")
        assert "...> " in output
        assert "error" not in output.lower()
        assert capsys.readouterr().out.split() == ["5"]

    def test_multiline_string_continues_through_blank_lines(self) -> None:
        # A blank line inside an open triple-quoted string must not force-submit
        # the still-open entry (unlike a blank line outside a string).
        output = drive_plain('let text = """first\n\nsecond"""\ntext\n')
        assert "first" in output
        assert "second" in output


# ---------------------------------------------------------------------------
# Meta-commands, echo, errors, dry-run
# ---------------------------------------------------------------------------


class TestPlainMetaAndEval:
    def test_help_meta_prints_commands(self) -> None:
        output = drive_plain(":help\n")
        assert ":help" in output
        assert ":quit" in output

    def test_unknown_meta_prints_error(self) -> None:
        output = drive_plain(":bogus\n")
        assert "Unknown command" in output
        assert ":bogus" in output

    def test_binding_echo_shows_name_type_value(self) -> None:
        output = drive_plain("let x = 5\n")
        assert "x : int = 5" in output

    def test_quiet_suppresses_echo(self) -> None:
        output = drive_plain("let x = 5\n", echo=False)
        assert "x : int = 5" not in output

    def test_error_entry_prints_diagnostic(self) -> None:
        output = drive_plain("let = 5\n")
        assert ": error:" in output.lower()

    def test_theme_switch_invokes_save_callback_only(self) -> None:
        saved: list[str] = []
        output = drive_plain(":theme dark\n", on_theme_save=saved.append)
        assert saved == ["dark"]
        assert "Theme" in output

    def test_theme_switch_without_save_callback_does_not_raise(self) -> None:
        # No on_theme_save was passed (defaults to None); the switch itself
        # still happens (plain mode has no styling to swap), it just persists
        # nothing.
        output = drive_plain(":theme dark\n")
        assert "Theme" in output


class TestPlainDryRun:
    def test_check_only_binding_shows_type_no_value(self) -> None:
        session = ReplSession()
        output = drive_plain("let x = 5\n", session=session, check_only=True)
        assert "x : int" in output
        assert "= 5" not in output
        assert session.bindings() == []

    def test_check_only_expression_shows_type(self) -> None:
        output = drive_plain("1 + 2\n", check_only=True)
        assert ": int" in output
        assert "3" not in output


# ---------------------------------------------------------------------------
# No styling
# ---------------------------------------------------------------------------


class TestPlainNoStyling:
    def test_no_ansi_escapes_in_output(self) -> None:
        output = drive_plain(
            'record R\n  x: int\nlet r = R(x = 1)\n:help\nlet bad = \n"unicode: héllo"\n'
        )
        assert "\x1b" not in output
