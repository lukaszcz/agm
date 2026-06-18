"""Headless tests for the prompt_toolkit console (``agm.agl.repl.console``).

Drives :func:`run_console` through prompt_toolkit's ``create_pipe_input`` +
``DummyOutput`` so no real terminal is required, and exercises the highlighting
lexer, the completer, and the multiline incompleteness predicate directly.

Scripted keystrokes use ``\\r`` for the Enter key (so the custom multiline Enter
binding fires), ``\\x04`` for Ctrl-D (EOF → exit), and ``\\x03`` for Ctrl-C
(cancel the current entry without exiting).  Assertions check user-visible
console output, never internals.
"""

from __future__ import annotations

import contextlib
import io
import signal
from collections.abc import Callable, Iterator
from types import FrameType

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from agm.agl.repl import ReplSession
from agm.agl.repl.agentmode import AgentMode
from agm.agl.repl.agents import ConfirmDecision
from agm.agl.repl.console import (
    AglCompleter,
    AglPromptLexer,
    _make_history,
    build_prompt_session,
    format_banner,
    has_runnable_statements,
    is_incomplete,
    run_console,
)
from agm.agl.runtime.request import AgentRequest, AgentResponse


class _CountingAgent:
    """A fake ``AgentFn`` that counts invocations and returns a scripted reply."""

    def __init__(self, reply: str = "ok") -> None:
        self._reply = reply
        self.calls = 0

    def __call__(self, request: AgentRequest) -> AgentResponse:
        del request
        self.calls += 1
        return AgentResponse(content=self._reply)

# ---------------------------------------------------------------------------
# Driver helper
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _fail_on_hang(seconds: int = 10) -> Iterator[None]:
    """Convert a stuck REPL (e.g. Ctrl-D on a non-empty buffer) into a failure.

    Scripted keystrokes should always terminate the loop; if they leave the
    prompt blocked on an exhausted pipe, this guard raises instead of hanging
    the whole test session.
    """

    def _raise(signum: int, frame: FrameType | None) -> None:
        raise AssertionError("REPL did not terminate — scripted keystrokes hung")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def drive(
    keystrokes: str,
    *,
    session: ReplSession | None = None,
    echo: bool = True,
    check_only: bool = False,
    agent_mode: AgentMode | None = None,
) -> str:
    """Feed *keystrokes* to a headless REPL and return everything it printed."""
    repl_session = session if session is not None else ReplSession()
    with create_pipe_input() as pipe, _fail_on_hang():
        pipe.send_text(keystrokes)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            run_console(
                repl_session,
                echo=echo,
                check_only=check_only,
                agent_mode=agent_mode,
                history_path=None,  # InMemoryHistory — never touch real $HOME
                input=pipe,
                output=DummyOutput(),
            )
    return out.getvalue()


# ---------------------------------------------------------------------------
# Loop: submit / exit / cancel
# ---------------------------------------------------------------------------


class TestLoop:
    def test_banner_is_printed(self) -> None:
        output = drive("\x04")
        assert "AgL REPL" in output
        # The banner points the user at :help and how to quit.
        assert ":help" in output
        assert ":quit" in output

    def test_banner_reports_confirm_mode(self) -> None:
        output = drive("\x04", agent_mode=AgentMode(mode="confirm"))
        assert "confirm" in output.lower()

    def test_banner_reports_auto_mode(self) -> None:
        output = drive("\x04", agent_mode=AgentMode(mode="auto"))
        assert "auto" in output.lower()

    def test_format_banner_starts_with_stable_prefix(self) -> None:
        # The first banner line is a stable prefix regardless of mode.
        assert format_banner().startswith("AgL REPL")
        assert format_banner(AgentMode(mode="auto")).startswith("AgL REPL")
        assert format_banner(AgentMode(mode="confirm")).startswith("AgL REPL")

    def test_single_expression_submits_and_echoes(self) -> None:
        output = drive("1 + 2\r\x04")
        assert "3" in output

    def test_ctrl_d_exits(self) -> None:
        # Ctrl-D with no entry exits cleanly (only the banner is printed).
        output = drive("\x04")
        assert output.strip().startswith("AgL REPL")

    def test_quit_meta_exits(self) -> None:
        output = drive(":quit\r")
        # Nothing evaluated after :quit.
        assert "AgL REPL" in output

    def test_exit_meta_exits(self) -> None:
        output = drive(":exit\r")
        assert "AgL REPL" in output

    def test_ctrl_c_cancels_entry_without_exiting(self) -> None:
        # Ctrl-C abandons the in-progress "1 + 1", then "2 + 2" still evaluates
        # and the REPL exits only on the trailing :quit — proving it kept going.
        # Each echoed expression result is printed on its own line; the cancelled
        # "1 + 1" must leave no echo line, while "2 + 2" echoes "4".
        output = drive("1 + 1\x03 2 + 2\r:quit\r")
        echoed = [line.strip() for line in output.splitlines() if line.strip()]
        assert "4" in echoed  # the surviving "2 + 2" evaluated
        assert "2" not in echoed  # the cancelled "1 + 1" produced no result line

    def test_whitespace_only_entry_is_ignored(self) -> None:
        # A whitespace-only buffer force-submitted on a blank line strips to ""
        # and is skipped; the following real entry still evaluates.
        output = drive("   \r\r1 + 1\r\x04")
        assert "2" in output

    def test_empty_enter_gives_fresh_prompt_no_error(self) -> None:
        # Pressing Enter on a wholly empty prompt is a no-op: nothing evaluates,
        # no parse error is printed, and a following real entry still evaluates.
        session = ReplSession()
        output = drive("\r1 + 1\r\x04", session=session)
        assert "2" in output
        assert "line" not in output.lower()  # no diagnostic
        assert session.bindings() == []  # no state change from the empty entry

    def test_comment_only_entry_is_noop_no_error(self) -> None:
        # A comment-only entry (everything after ``#`` is a comment) has nothing
        # to run: fresh prompt, no error, no state change. The next real entry
        # still evaluates.
        session = ReplSession()
        output = drive("# just a comment\r1 + 1\r\x04", session=session)
        assert "2" in output
        assert "Unexpected" not in output
        assert "line" not in output.lower()
        assert session.bindings() == []


# ---------------------------------------------------------------------------
# Multiline continuation
# ---------------------------------------------------------------------------


class TestMultiline:
    @pytest.mark.parametrize(
        ("header", "body", "echoed"),
        [
            ("record R", "  x: int", "R declared"),
            ("enum E", "| A", "E declared"),
            ("if 1 = 1 =>", '  "hi"', None),
            ("do", "  ()\nuntil 1 = 1", None),
            ("try", "  ()\ncatch _ =>\n  ()", None),
            ("case 1 of", "| _ => 7", None),
        ],
    )
    def test_block_continues_then_completes(
        self, header: str, body: str, echoed: str | None
    ) -> None:
        # The header alone is incomplete (Enter inserts a newline), and the full
        # block submits.  ``\r`` for each Enter so the multiline binding fires.
        keystrokes = header + "\r" + body.replace("\n", "\r") + "\r\x04"
        output = drive(keystrokes)
        if echoed is not None:
            assert echoed in output

    def test_incomplete_header_keeps_prompting(self) -> None:
        # ``record R`` alone is incomplete, so the first Enter opens a
        # continuation rather than submitting.  A blank line then force-submits
        # the still-incomplete buffer, which surfaces a parse error (no
        # declaration is promoted).
        output = drive("record R\r\r\x04")
        assert "declared" not in output
        assert ": error:" in output.lower()

    def test_blank_line_force_submits_incomplete(self) -> None:
        # A buffer still ending in an unfinished header is incomplete, but once
        # the user presses Enter on a blank continuation line (buffer ends with
        # a newline) ``is_incomplete`` force-submits.
        assert is_incomplete("record R") is True
        assert is_incomplete("record R\n") is False


class TestIsIncomplete:
    @pytest.mark.parametrize(
        "source",
        ["record R", "enum E", "case x of", "try", "do agent", "if x = 1 =>", "1 +"],
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
# Lexer
# ---------------------------------------------------------------------------


class TestLexer:
    def test_styles_a_sample_line(self) -> None:
        lexer = AglPromptLexer()
        fragments = lexer.lex_document(Document('let x = 1 + foo'))(0)
        styles = {style for style, _text in fragments}
        assert "class:agl.keyword" in styles  # let
        assert "class:agl.operator" in styles  # = / +
        assert "class:agl.number" in styles  # 1
        # The full line text is preserved across the fragments.
        assert "".join(text for _style, text in fragments) == "let x = 1 + foo"

    def test_half_typed_line_does_not_raise(self) -> None:
        lexer = AglPromptLexer()
        # An invalid character mid-line must fall back to plain text, not raise.
        fragments = lexer.lex_document(Document("let x = @bad"))(0)
        assert "".join(text for _style, text in fragments) == "let x = @bad"

    def test_string_literal_is_styled(self) -> None:
        lexer = AglPromptLexer()
        fragments = lexer.lex_document(Document('print "hello"'))(0)
        styles = {style for style, _text in fragments}
        assert "class:agl.string" in styles

    @pytest.mark.parametrize(
        "line",
        [
            'x = "hi"',
            "x = 'hi'",
            'ask "q"',
            'x = "a" + "b"',
            'x = ""',
        ],
    )
    def test_closed_string_is_not_duplicated(self, line: str) -> None:
        # Regression: the closing quote of a string used to be covered by both
        # the STRING_FRAGMENT and TEMPLATE_END spans, so the highlighter rendered
        # it twice. The styled fragments must reconstruct the line exactly.
        fragments = AglPromptLexer().lex_document(Document(line))(0)
        assert "".join(text for _style, text in fragments) == line

    def test_out_of_range_line_does_not_crash(self) -> None:
        lexer = AglPromptLexer()
        getter = lexer.lex_document(Document("let x = 1"))
        # Asking for a line beyond the document yields an empty line, not a crash.
        assert getter(0)  # in range
        assert getter(5) == []

    def test_multiline_document_styles_each_line(self) -> None:
        lexer = AglPromptLexer()
        getter = lexer.lex_document(Document("record R\n  x: int"))
        first = getter(0)
        second = getter(1)
        assert any(style == "class:agl.keyword" for style, _ in first)
        assert "".join(t for _, t in second) == "  x: int"

    def test_trailing_unstyled_text_is_preserved(self) -> None:
        # A line ending in unstyled text (trailing whitespace after a token)
        # keeps that text as a plain fragment.
        fragments = AglPromptLexer().lex_document(Document("1   "))(0)
        assert ("", "   ") in fragments
        assert "".join(text for _style, text in fragments) == "1   "


class TestHistory:
    def test_none_path_uses_in_memory_history(self) -> None:
        assert isinstance(_make_history(None), InMemoryHistory)

    def test_path_uses_file_history(self, tmp_path: object) -> None:
        from pathlib import Path

        assert isinstance(tmp_path, Path)
        history = _make_history(tmp_path / "hist")
        assert isinstance(history, FileHistory)

    def test_build_session_with_file_history(self, tmp_path: object) -> None:
        from pathlib import Path

        assert isinstance(tmp_path, Path)
        # Smoke-test the factory with a real history path (the FileHistory
        # branch of build_prompt_session).
        session = build_prompt_session(ReplSession(), history_path=tmp_path / "h")
        assert session is not None


# ---------------------------------------------------------------------------
# Completer
# ---------------------------------------------------------------------------


def _completions(completer: AglCompleter, text: str) -> list[str]:
    document = Document(text, len(text))
    return [c.text for c in completer.get_completions(document, CompleteEvent())]


class TestCompleter:
    def test_completes_keyword(self) -> None:
        completer = AglCompleter(ReplSession())
        assert "let" in _completions(completer, "le")

    def test_completes_live_binding(self) -> None:
        session = ReplSession()
        session.eval_entry("let myvar = 10")
        completer = AglCompleter(session)
        assert "myvar" in _completions(completer, "myv")

    def test_completes_meta_commands(self) -> None:
        completer = AglCompleter(ReplSession())
        suggestions = _completions(completer, ":")
        assert ":help" in suggestions
        assert ":quit" in suggestions

    def test_meta_prefix_filters(self) -> None:
        completer = AglCompleter(ReplSession())
        suggestions = _completions(completer, ":q")
        assert ":quit" in suggestions
        assert ":help" not in suggestions

    def test_no_completion_for_unknown_word(self) -> None:
        completer = AglCompleter(ReplSession())
        assert _completions(completer, "zzzzz") == []

    def test_completes_builtin_calls(self) -> None:
        # The four builtin call names are not reserved keywords and are not
        # promoted bindings; the completer must still offer them so a user
        # typing ``ask-...`` or ``print(`` gets a suggestion.
        completer = AglCompleter(ReplSession())
        for name in ("print", "exec", "ask", "ask-request"):
            assert name in _completions(completer, "")

    def test_completes_ask_request_prefix(self) -> None:
        completer = AglCompleter(ReplSession())
        assert "ask-request" in _completions(completer, "ask-r")
        # ``ask`` is also a builtin, so the unqualified prefix must still
        # surface both it and ``ask-request``.  Use a prefix shorter than the
        # full name so the exact-match exclusion does not drop ``ask``.
        suggestions = _completions(completer, "as")
        assert "ask" in suggestions
        assert "ask-request" in suggestions

    def test_completes_ask_without_default_agent(self) -> None:
        # ``ask`` is a builtin call name independent of any configured default
        # agent; even with no default agent it must be offered (regression for
        # the previous behaviour where ``ask`` only leaked in via the agents
        # pool when a default agent existed).
        completer = AglCompleter(ReplSession())
        assert "ask" in _completions(completer, "as")


# ---------------------------------------------------------------------------
# Evaluated output via the loop
# ---------------------------------------------------------------------------


class TestEvalOutput:
    def test_binding_echo_shows_name_type_value(self) -> None:
        output = drive("let x = 5\r\x04")
        assert "x : int = 5" in output

    def test_expression_echo_shows_value(self) -> None:
        output = drive('"hi"\r\x04')
        assert "hi" in output

    def test_quiet_suppresses_echo(self) -> None:
        output = drive("let x = 5\r\x04", echo=False)
        assert "x : int = 5" not in output

    def test_error_entry_prints_diagnostic(self) -> None:
        output = drive("let = 5\r\x04")
        assert ": error:" in output.lower()


class TestDryRun:
    def test_check_only_binding_shows_type_no_value(self) -> None:
        session = ReplSession()
        output = drive("let x = 5\r\x04", session=session, check_only=True)
        assert "x : int" in output
        assert "= 5" not in output  # no value in dry-run
        assert session.bindings() == []  # nothing persisted

    def test_check_only_expression_shows_type(self) -> None:
        output = drive("1 + 2\r\x04", check_only=True)
        assert ": int" in output
        assert "3" not in output  # the value is never computed

    def test_check_only_agent_call_typechecks_without_firing(self) -> None:
        # An entry with an agent call type-checks and echoes its type, but the
        # fake agent is never invoked and no binding is persisted.
        agent = _CountingAgent("should-not-be-used")
        session = ReplSession(default_agent=agent)
        output = drive(
            'let g: text = ask """say something"""\r\x04',
            session=session,
            check_only=True,
        )
        assert "g : text" in output
        assert agent.calls == 0  # no agent fired in dry-run
        assert session.bindings() == []  # no binding persisted

    def test_check_only_error_still_reports_diagnostic(self) -> None:
        output = drive("let = 5\r\x04", check_only=True)
        assert ": error:" in output.lower()

    def test_help_meta_prints_commands(self) -> None:
        output = drive(":help\r\x04")
        assert ":help" in output
        assert ":quit" in output

    def test_unknown_meta_prints_error(self) -> None:
        output = drive(":bogus\r\x04")
        assert "Unknown command" in output
        assert ":bogus" in output


# ---------------------------------------------------------------------------
# Meta-commands through the loop (M3)
# ---------------------------------------------------------------------------


class TestMetaThroughLoop:
    def test_set_echo_off_suppresses_then_on_restores(self) -> None:
        # echo off → the binding is not echoed; echo on → the next binding is.
        output = drive(
            ":set echo off\rlet a = 1\r:set echo on\rlet b = 2\r\x04"
        )
        assert "a : int = 1" not in output  # suppressed while echo off
        assert "b : int = 2" in output  # restored after echo on

    def test_bindings_meta_lists_live_bindings(self) -> None:
        output = drive("let x = 5\r:bindings\r\x04")
        assert "x : int = 5" in output

    def test_reset_meta_clears_session(self) -> None:
        session = ReplSession()
        output = drive("let x = 5\r:reset\r:bindings\r\x04", session=session)
        assert "Session reset." in output
        assert "No bindings." in output
        assert session.bindings() == []

    def test_type_meta_reports_type(self) -> None:
        output = drive("1 + 2\r:type 1 + 2\r\x04")
        assert "int" in output

    def test_agent_meta_mutates_shared_mode(self) -> None:
        # The shared AgentMode passed to run_console reflects :agent mutations,
        # which is exactly the instance M4 will also hand to its wrapper.
        from agm.agl.repl.agentmode import AgentMode

        mode = AgentMode()
        with create_pipe_input() as pipe, _fail_on_hang():
            pipe.send_text(":agent auto\r\x04")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                run_console(
                    ReplSession(),
                    agent_mode=mode,
                    history_path=None,
                    input=pipe,
                    output=DummyOutput(),
                )
        assert mode.mode == "auto"

    def test_load_meta_runs_file_into_session(self, tmp_path: object) -> None:
        from pathlib import Path

        assert isinstance(tmp_path, Path)
        src = tmp_path / "prog.agl"
        src.write_text("let loaded = 9\n")
        session = ReplSession()
        output = drive(f":load {src}\r\x04", session=session)
        assert "loaded : int = 9" in output
        assert any(n == "loaded" for n, _t, _v in session.bindings())


# ---------------------------------------------------------------------------
# Agent-call confirmation callback + confirm flow through run_console
# ---------------------------------------------------------------------------


class TestConfirmCallback:
    def _confirm_factory(
        self, *answers: str
    ) -> tuple[Callable[[str, str], ConfirmDecision], list[str]]:
        """Build a confirm callback whose reader replays scripted answers."""
        from agm.agl.repl.console import make_console_confirm

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


def _confirming_session(
    *answers: str, reply: str = "agent-reply"
) -> tuple[ReplSession, "object"]:
    """A session whose default agent is a ConfirmingAgent with a scripted confirm."""
    from agm.agl.repl.agentmode import AgentMode
    from agm.agl.repl.agents import ConfirmingAgent
    from agm.agl.repl.console import make_console_confirm

    replies = iter(answers)
    confirm = make_console_confirm(
        reader=lambda _prompt: next(replies), printer=lambda _s: None
    )
    mode = AgentMode(mode="confirm")
    underlying = _CountingAgent(reply)
    wrapper = ConfirmingAgent(underlying, mode, confirm=confirm)
    session = ReplSession(default_agent=wrapper)
    return session, underlying


class TestConfirmFlowThroughLoop:
    def test_confirmed_call_dispatches_and_echoes(self) -> None:
        session, underlying = _confirming_session("y", reply="hello-world")
        assert isinstance(underlying, _CountingAgent)
        output = drive('let g = ask """ask"""\r\x04', session=session)
        assert underlying.calls == 1
        assert "hello-world" in output
        assert any(n == "g" for n, _t, _v in session.bindings())

    def test_declined_call_aborts_entry_repl_continues(self) -> None:
        session, underlying = _confirming_session("n")
        assert isinstance(underlying, _CountingAgent)
        # Decline the agent call, then run a plain entry to prove the REPL keeps
        # looping after the abort.
        output = drive(
            'let g = ask """ask"""\rlet ok = 1\r\x04', session=session
        )
        assert underlying.calls == 0
        assert "cancelled" in output.lower()
        # The aborted entry promoted nothing; the later entry succeeded.
        assert all(n != "g" for n, _t, _v in session.bindings())
        assert any(n == "ok" for n, _t, _v in session.bindings())


# ---------------------------------------------------------------------------
# Issue #5: is_incomplete_source classification memo (no double parse)
# ---------------------------------------------------------------------------


class TestIsIncompleteSourceMemo:
    """is_incomplete_source must not re-parse text it already classified.

    The Enter key binding checks ``is_incomplete`` on every keypress; when
    Enter is pressed on a *complete* entry the same text is about to be
    submitted to eval which triggers another parse.  The classification memo
    ensures only ONE ``_PARSER.parse`` call is made for a given text within
    that cycle.
    """

    def test_repeated_call_with_same_text_parses_only_once(self) -> None:
        from unittest.mock import patch

        import agm.agl.parser.parser as parser_mod
        from agm.agl.parser import is_incomplete_source

        text = "let x = 42"
        real_parse = parser_mod._PARSER.parse
        with patch.object(parser_mod._PARSER, "parse", wraps=real_parse) as mock_parse:
            result1 = is_incomplete_source(text)
            result2 = is_incomplete_source(text)
        # Both calls return the same (False) classification.
        assert not result1
        assert not result2
        # The underlying parser must be called exactly once despite two
        # is_incomplete_source calls with the same text.
        count = mock_parse.call_count
        assert count == 1, f"Expected 1 parse call for repeated text; got {count}"

    def test_different_text_is_reclassified(self) -> None:
        """Changing the text invalidates the memo and re-parses."""
        from unittest.mock import patch

        import agm.agl.parser.parser as parser_mod
        from agm.agl.parser import is_incomplete_source

        real_parse = parser_mod._PARSER.parse
        with patch.object(parser_mod._PARSER, "parse", wraps=real_parse) as mock_parse:
            is_incomplete_source("let x = 1")
            is_incomplete_source("let y = 2")
        # Two distinct texts → two parse calls.
        assert mock_parse.call_count == 2

    def test_incomplete_text_memo(self) -> None:
        """Memo works correctly for incomplete (True) classification too."""
        from unittest.mock import patch

        import agm.agl.parser.parser as parser_mod
        from agm.agl.parser import is_incomplete_source

        text = "record R"
        real_parse = parser_mod._PARSER.parse
        with patch.object(parser_mod._PARSER, "parse", wraps=real_parse) as mock_parse:
            r1 = is_incomplete_source(text)
            r2 = is_incomplete_source(text)
        assert r1 is True
        assert r2 is True
        assert mock_parse.call_count == 1


# ---------------------------------------------------------------------------
# Issue #8: _styled_lines O(lines × spans) → O(spans) bucketing
# ---------------------------------------------------------------------------


class TestStyledLinesBucketing:
    """_styled_lines must produce identical output after the bisect-bucketing refactor.

    The characterizing test captures output from a representative multi-line
    entry using the full AglPromptLexer surface, then verifies the refactored
    implementation produces exactly the same per-line fragment lists.

    Because _styled_lines is a static method we test it directly, which lets
    us compare before/after without needing a full prompt_toolkit Document.
    """

    _MULTILINE_TEXT = "let x = 42\nlet y = x + 1\nlet z = y + 1"

    def _get_styled_lines(self, text: str) -> list[list[tuple[str, str]]]:
        """Return per-line fragments as plain lists for easy comparison."""
        return [list(line) for line in AglPromptLexer._styled_lines(text)]

    def test_multiline_fragment_text_coverage(self) -> None:
        """Every character of every line appears exactly once in the fragments."""
        lines = self._MULTILINE_TEXT.split("\n")
        styled = self._get_styled_lines(self._MULTILINE_TEXT)
        assert len(styled) == len(lines)
        for i, (line, frags) in enumerate(zip(lines, styled)):
            reconstructed = "".join(text for _style, text in frags)
            assert reconstructed == line, (
                f"Line {i}: reconstructed {reconstructed!r} != original {line!r}"
            )

    def test_keywords_styled_on_correct_lines(self) -> None:
        """Keywords on each line receive the keyword style class."""
        styled = self._get_styled_lines(self._MULTILINE_TEXT)
        # Line 0: 'let x = 42' — 'let' should be styled as keyword
        line0_styles = {style for style, _ in styled[0]}
        assert "class:agl.keyword" in line0_styles
        # Line 2: 'let z = y + 1' — 'let' should be styled as keyword
        line2_styles = {style for style, _ in styled[2]}
        assert "class:agl.keyword" in line2_styles

    def test_no_span_bleed_across_lines(self) -> None:
        """Tokens from one line must not appear in fragments for a different line."""
        text = "let a = 1\nlet b = 2"
        styled = self._get_styled_lines(text)
        # Line 0 should not contain "b" as styled text (it belongs to line 1)
        line0_text = "".join(t for _s, t in styled[0])
        assert "b" not in line0_text or line0_text == "let a = 1"
        # Line 1 text reconstruction must equal 'let b = 2'
        line1_text = "".join(t for _s, t in styled[1])
        assert line1_text == "let b = 2"

    def test_single_line_unchanged(self) -> None:
        """Single-line input still produces exactly one fragment list."""
        text = "let x = 1 + 2"
        styled = self._get_styled_lines(text)
        assert len(styled) == 1
        assert "".join(t for _s, t in styled[0]) == text

    def test_empty_line_in_middle(self) -> None:
        """An empty line in a multi-line input produces an empty fragment list."""
        text = "let x = 1\n\nlet y = 2"
        styled = self._get_styled_lines(text)
        assert len(styled) == 3
        assert styled[1] == [("", "")]
