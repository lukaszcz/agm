"""Direct tests for the UI-free REPL loop core (``agm.agl.repl.loop``).

Both front ends (the prompt_toolkit console and the plain line console) drive
this module and are exercised through their own test files
(``tests/test_agl_repl_console.py``, ``tests/test_agl_repl_plain_console.py``),
which cover ``run_repl_loop`` end to end with a real ``on_theme_change`` hook.
This file covers what a front end cannot: calling :func:`run_repl_loop`
directly with a bare reader/writer seam, including the case neither front end
exercises — no ``on_theme_change`` hook at all — plus direct, driver-free tests
of the other names ``agm.agl.repl.loop`` exports: :func:`format_banner`,
:func:`is_incomplete`, :func:`has_runnable_statements`, and
:func:`make_console_confirm`.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from agm.agl.repl import ReplSession
from agm.agl.repl.agentmode import AgentMode
from agm.agl.repl.agents import ConfirmDecision
from agm.agl.repl.loop import (
    format_banner,
    has_runnable_statements,
    is_incomplete,
    make_console_confirm,
    run_repl_loop,
)


def _scripted_reader(entries: list[str]) -> Callable[[], str]:
    remaining = iter(entries)

    def reader() -> str:
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError from None

    return reader


def test_theme_switch_with_no_on_theme_change_hook_is_tolerated() -> None:
    # ``on_theme_change`` defaults to ``None``; a ``:theme`` switch must not
    # crash when there is nothing to notify.
    session = ReplSession()
    written: list[str] = []
    reader = _scripted_reader([":theme dark", ":quit"])

    run_repl_loop(session, reader=reader, writer=written.append)

    assert any("Theme" in line for line in written)


def test_on_theme_change_is_called_only_when_the_theme_actually_changes() -> None:
    session = ReplSession()
    written: list[str] = []
    changes: list[str] = []
    reader = _scripted_reader([":theme dark", ":theme dark", ":quit"])

    run_repl_loop(
        session,
        reader=reader,
        writer=written.append,
        on_theme_change=changes.append,
    )

    assert changes == ["dark"]  # the second, same-value switch triggers nothing


def test_entry_evaluation_and_rendering() -> None:
    session = ReplSession()
    written: list[str] = []
    reader = _scripted_reader(["1 + 2"])

    run_repl_loop(session, reader=reader, writer=written.append)

    assert any("3" in line for line in written)


def test_keyboard_interrupt_cancels_entry_without_exiting() -> None:
    session = ReplSession()
    written: list[str] = []
    calls = {"n": 0}

    def reader() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        if calls["n"] == 2:
            return "1 + 1"
        raise EOFError

    run_repl_loop(session, reader=reader, writer=written.append)

    assert any("2" in line for line in written)


# ---------------------------------------------------------------------------
# format_banner
# ---------------------------------------------------------------------------


def test_format_banner_starts_with_stable_prefix() -> None:
    # The first banner line is a stable prefix regardless of mode.
    assert format_banner().startswith("AgL REPL")
    assert format_banner(AgentMode(mode="auto")).startswith("AgL REPL")
    assert format_banner(AgentMode(mode="confirm")).startswith("AgL REPL")


# ---------------------------------------------------------------------------
# is_incomplete
# ---------------------------------------------------------------------------


class TestIsIncomplete:
    @pytest.mark.parametrize("source", ["exec$", "ask$::[text]", "let reply = ask$"])
    def test_raw_tail_headers_open_blocks(self, source: str) -> None:
        assert is_incomplete(source) is True

    @pytest.mark.parametrize(
        "source",
        [
            "exec$ echo hi",
            "exec$\n  echo hi\nnext",
            "let x: text = exec$\n    echo hi\n# trailing comment",
        ],
    )
    def test_closed_raw_tail_blocks_submit(self, source: str) -> None:
        assert is_incomplete(source) is False

    @pytest.mark.parametrize(
        "source",
        ["record R", "enum E", "case x of", "try", "do agent", "if x == 1 =>", "1 +"],
    )
    def test_incomplete_sources(self, source: str) -> None:
        assert is_incomplete(source) is True

    @pytest.mark.parametrize(
        "source",
        ["1 + 2", "let x = 1", "let = 5", "x == y", "record R\n  x: int"],
    )
    def test_complete_sources(self, source: str) -> None:
        assert is_incomplete(source) is False

    @pytest.mark.parametrize("source", ["", "   ", "\t", "  \n  "])
    def test_blank_input_force_submits(self, source: str) -> None:
        # Blank / whitespace-only input force-submits so Enter on an empty prompt
        # gives a fresh prompt instead of inserting a newline.
        assert is_incomplete(source) is False

    def test_blank_line_force_submits_incomplete(self) -> None:
        # A buffer still ending in an unfinished header is incomplete, but once
        # the user presses Enter on a blank continuation line (buffer ends with
        # a newline) ``is_incomplete`` force-submits.
        assert is_incomplete("record R") is True
        assert is_incomplete("record R\n") is False

    @pytest.mark.parametrize("quote", ['"""', "'''"])
    def test_unterminated_triple_quoted_string_keeps_prompting(self, quote: str) -> None:
        # An open triple-quoted string is a lexical EOF condition, but in the
        # REPL it is a natural multiline entry and must keep accepting lines
        # until the matching delimiter is typed.
        assert is_incomplete(f"let text = {quote}first") is True
        assert is_incomplete(f"let text = {quote}first\n") is True
        assert is_incomplete(f"let text = {quote}first\n\nsecond") is True
        assert is_incomplete(f"let text = {quote}first\n\nsecond{quote}") is False


# ---------------------------------------------------------------------------
# has_runnable_statements
# ---------------------------------------------------------------------------


class TestHasRunnableStatements:
    @pytest.mark.parametrize(
        "source",
        ["", "   ", "\t", "# a comment", "  # indented comment", "# one\n# two"],
    )
    def test_blank_or_comment_only_has_nothing_to_run(self, source: str) -> None:
        assert has_runnable_statements(source) is False

    @pytest.mark.parametrize(
        "source",
        ["1 + 1", "let x = 1", "# lead\nlet y = 2", "record R\n  x: int"],
    )
    def test_real_entry_has_statements(self, source: str) -> None:
        assert has_runnable_statements(source) is True

    def test_lexer_error_is_treated_as_runnable(self) -> None:
        # An odd/unlexable entry is conservatively runnable so it reaches the
        # evaluator and surfaces a real diagnostic rather than being dropped.
        assert has_runnable_statements("@") is True


# ---------------------------------------------------------------------------
# make_console_confirm
# ---------------------------------------------------------------------------
#
# Despite the name (kept for the callback's established meaning: it confirms a
# live agent call the way the console does), this callback is UI-free — it is
# shared by both front ends via agm.agl.repl.loop — so it is tested directly
# here with no console driven at all.


class TestMakeConsoleConfirm:
    def _confirm_factory(
        self, *answers: str
    ) -> tuple[Callable[[str, str], ConfirmDecision], list[str]]:
        """Build a confirm callback whose reader replays scripted answers."""
        replies = iter(answers)
        printed: list[str] = []
        confirm = make_console_confirm(
            reader=lambda _prompt: next(replies),
            printer=printed.append,
        )
        return confirm, printed

    def test_yes_no_always(self) -> None:
        confirm, _printed = self._confirm_factory("y", "n", "a")
        assert confirm("writer", "do it") == "yes"
        assert confirm("writer", "do it") == "no"
        assert confirm("writer", "do it") == "always"

    def test_empty_answer_defaults_to_yes(self) -> None:
        confirm, _printed = self._confirm_factory("")
        assert confirm("writer", "do it") == "yes"

    def test_unrecognised_reasks_then_accepts(self) -> None:
        confirm, printed = self._confirm_factory("huh?", "yes")
        assert confirm("writer", "do it") == "yes"
        assert any("y(es)" in line for line in printed)

    def test_view_prints_full_prompt_then_accepts(self) -> None:
        long_prompt = "X" * 500
        confirm, printed = self._confirm_factory("v", "y")
        assert confirm("writer", long_prompt) == "yes"
        # The truncated preview AND the full text both appear.
        assert any("truncated" in line for line in printed)
        assert any(long_prompt in line for line in printed)
