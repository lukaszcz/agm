"""Direct tests for the UI-free REPL loop core (``agm.agl.repl.loop``).

Both front ends (the prompt_toolkit console and the plain line console) drive
this module and are exercised through their own test files
(``tests/test_agl_repl_console.py``, ``tests/test_agl_repl_plain_console.py``),
which cover ``run_repl_loop`` end to end with a real ``on_theme_change`` hook.
This file covers what a front end cannot: calling :func:`run_repl_loop`
directly with a bare reader/writer seam, including the case neither front end
exercises — no ``on_theme_change`` hook at all — plus direct, driver-free tests
of the other names ``agm.agl.repl.loop`` exports: :func:`format_banner`,
:func:`is_incomplete`, and :func:`has_runnable_statements`.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from agm.agl.repl import ReplSession
from agm.agl.repl.loop import (
    format_banner,
    has_runnable_statements,
    is_incomplete,
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


def test_theme_switch_with_no_on_setting_change_hook_is_tolerated() -> None:
    # ``on_setting_change`` defaults to ``None``; a ``:theme`` switch must not
    # crash when there is nothing to notify.
    session = ReplSession()
    written: list[str] = []
    reader = _scripted_reader([":theme dark", ":quit"])

    run_repl_loop(session, reader=reader, writer=written.append)

    assert any("Theme" in line for line in written)


def test_on_setting_change_is_called_for_every_explicit_theme_switch() -> None:
    session = ReplSession()
    written: list[str] = []
    changes: list[tuple[str, str | bool]] = []
    reader = _scripted_reader([":theme dark", ":theme dark", ":quit"])

    run_repl_loop(
        session,
        reader=reader,
        writer=written.append,
        on_setting_change=lambda key, value: changes.append((key, value)),
    )

    # Both switches report, even the second (a no-op on the live session):
    # an explicit ``:theme`` always persists its target.
    assert changes == [("theme", "dark"), ("theme", "dark")]


def test_on_setting_change_reports_echo_and_echo_unit_changes() -> None:
    session = ReplSession()
    written: list[str] = []
    changes: list[tuple[str, str | bool]] = []
    reader = _scripted_reader([":set echo off", ":set echo-unit on", ":quit"])

    run_repl_loop(
        session,
        reader=reader,
        writer=written.append,
        on_setting_change=lambda key, value: changes.append((key, value)),
    )

    assert changes == [("echo", False), ("echo-unit", True)]


def test_info_uses_the_rich_front_end_highlighter_when_available() -> None:
    session = ReplSession()
    written: list[str] = []
    highlighted: list[str] = []
    reader = _scripted_reader(["let count = 1", ":info count", ":quit"])

    run_repl_loop(
        session,
        reader=reader,
        writer=written.append,
        highlighted_writer=lambda text, _ranges: highlighted.append(text),
    )

    assert highlighted == [
        "count is a binding.\nBinding:\n  let count\nType:\n  int\nValue:\n  1\n"
        "Location: <repl>:1:1"
    ]
    assert all("count is a binding" not in text for text in written)


def test_unit_entries_echo_nothing_by_default() -> None:
    session = ReplSession()
    written: list[str] = []
    reader = _scripted_reader(["()", 'print "x"', "let u = ()", ":quit"])

    run_repl_loop(session, reader=reader, writer=written.append)

    # Only the banner and print's own stdout-bound output land in ``written``
    # (``print`` here is the loop's writer callback, not the REPL entry's own
    # ``print``, whose effect is not captured by this reader/writer seam);
    # none of the three unit-typed entries produces an echo line.
    assert written == [format_banner()]


def test_set_echo_unit_on_makes_unit_entries_echo() -> None:
    session = ReplSession()
    written: list[str] = []
    reader = _scripted_reader([":set echo-unit on", "()", 'print "x"', "let u = ()", ":quit"])

    run_repl_loop(session, reader=reader, writer=written.append)

    # ``()`` and ``print "x"`` are both bare unit-typed expressions, echoed as
    # the bare value; ``let u = ()`` is a unit-typed binding, echoed with its
    # name and type. All three echo once echo-unit is on.
    assert written.count("()") == 2
    assert "u : unit = ()" in written


def test_set_echo_unit_off_suppresses_after_being_on() -> None:
    session = ReplSession()
    written: list[str] = []
    reader = _scripted_reader([":set echo-unit on", ":set echo-unit off", "()", ":quit"])

    run_repl_loop(session, reader=reader, writer=written.append)

    assert not any(": unit = ()" in line for line in written)


def test_dry_run_unit_entry_echoes_type_only_when_echo_unit_on() -> None:
    session = ReplSession()
    written: list[str] = []
    reader = _scripted_reader([":set echo-unit on", "()", ":quit"])

    run_repl_loop(session, reader=reader, writer=written.append, check_only=True)

    assert any(line == ": unit" for line in written)


def test_dry_run_unit_entry_echoes_nothing_by_default() -> None:
    session = ReplSession()
    written: list[str] = []
    reader = _scripted_reader(["()", ":quit"])

    run_repl_loop(session, reader=reader, writer=written.append, check_only=True)

    assert written == [format_banner()]


def test_entry_evaluation_and_rendering() -> None:
    session = ReplSession()
    written: list[str] = []
    reader = _scripted_reader(["1 + 2"])

    run_repl_loop(session, reader=reader, writer=written.append)

    assert any("3" in line for line in written)


def test_blank_and_comment_only_entries_are_never_evaluated() -> None:
    # A blank line (or a comment-only entry) has nothing to run: the loop must
    # give a fresh prompt without handing the entry to the evaluator, whose
    # parser rejects it.  Anything the evaluator produced would reach the
    # writer, so the banner must remain the only output.
    session = ReplSession()
    written: list[str] = []
    reader = _scripted_reader(["", "   ", "\t", "# just a comment", "  # indented"])

    run_repl_loop(session, reader=reader, writer=written.append)

    assert len(written) == 1
    assert written[0].startswith("AgL REPL")


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
    assert format_banner().startswith("AgL REPL")


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
        [
            "ask $",
            "ask::[text] $",
            "let reply = ask $",
            "ask $\n  echo hi",
            "ask $   \n  echo hi",
            "def f() =>\n  ask $\n    body",
            "let a = $\n  one\nlet b = $\n  two",
        ],
    )
    def test_dollar_verbatim_headers_open_blocks(self, source: str) -> None:
        assert is_incomplete(source) is True

    @pytest.mark.parametrize(
        "source",
        [
            "ask $ echo hi",
            "ask $\n  echo hi\nnext",
            "let x: text = ask $\n    echo hi\n# trailing comment",
            "ask $  # c",
            "ask $\n  echo\n  ",
            "let a = $\n  one\nlet b = $ two",
        ],
    )
    def test_closed_dollar_verbatim_blocks_submit(self, source: str) -> None:
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
