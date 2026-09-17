"""Headless tests for the prompt_toolkit console (``agm.agl.repl.console``).

Drives :func:`run_console` through prompt_toolkit's ``create_pipe_input`` +
``DummyOutput`` so no real terminal is required, and exercises the highlighting
lexer and the completer directly. Tests for the UI-free predicates and helpers
``run_console`` delegates to (``format_banner``, ``is_incomplete``,
``has_runnable_statements``) live in
``tests/test_agl_repl_loop.py`` alongside the rest of ``agm.agl.repl.loop``'s
direct tests; this file covers only what actually needs the console driven.

Scripted keystrokes use ``\\r`` for the Enter key (so the custom multiline Enter
binding fires), ``\\x04`` for Ctrl-D (EOF → exit), and ``\\x03`` for Ctrl-C
(cancel the current entry without exiting).  Assertions check user-visible
console output, never internals.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterator
from contextlib import AbstractContextManager
from unittest.mock import patch

import pytest
from lark.lexer import Token
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput

import agm.agl.repl.console as console
from agm.agl.lexer import tokenize
from agm.agl.repl import ReplSession as _ReplSession
from agm.agl.repl.console import (
    AglCompleter,
    AglPromptLexer,
    _highlighted_agl_fragments,
    _make_history,
    build_prompt_session,
    run_console,
)
from agm.agl.runtime.request import AgentRequest, AgentResponse
from tests._process_helpers import FakeShell
from tests._timeouts import fail_if_slow


class ReplSession(_ReplSession):
    """Use the smallest session image for console-only behavior tests.

    The automatic prelude is covered by the session tests. Loading it for each
    headless console interaction competes with the full parallel suite and can
    consume the hang guard's budget before prompt_toolkit reads its input.
    """

    def __init__(self, **kwargs: object) -> None:
        default_stdlib = kwargs.pop("default_stdlib", False)
        super().__init__(default_stdlib=default_stdlib, **kwargs)


class _CountingAgent:
    """A fake ``AgentFn`` that counts invocations and returns a scripted reply."""

    def __init__(self, reply: str = "ok") -> None:
        self._reply = reply
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, request: AgentRequest) -> AgentResponse:
        self.calls += 1
        self.prompts.append(request.prompt)
        return AgentResponse(content=self._reply)


# ---------------------------------------------------------------------------
# Driver helper
# ---------------------------------------------------------------------------


def _fail_on_hang() -> AbstractContextManager[None]:
    """Convert a REPL that never returns into a failure rather than a wedge.

    `_scripted_input` closes the pipe, so running out of keystrokes ends the
    loop on its own and is no longer a way to hang.  What is left for this
    guard is a loop that stops consuming its input at all -- spinning, or
    waiting on something that is not the terminal -- which no amount of input
    will resolve.  That is a distinction between finite and infinite, so any
    finite budget makes it, and a generous one costs a passing run nothing
    while a tight one would report a merely loaded machine as a hang.
    """

    return fail_if_slow("REPL did not terminate — scripted keystrokes hung")


@contextlib.contextmanager
def _scripted_input(keystrokes: str) -> Iterator[PipeInput]:
    """Yield a pipe holding *keystrokes* and already at its end of input.

    Closing the write end leaves the buffered keystrokes readable but gives the
    loop a real end of input after them, so a script that never asks to quit
    ends rather than blocking on a pipe nothing will write to again.  Every
    console-driving test goes through here so that none of them can reintroduce
    that block by forgetting the close: the alternative backstop is the
    wall-clock deadline below, which has to stay loose enough for a loaded
    machine and so would turn a wedged loop into a slow, load-dependent
    failure somewhere else in the suite.
    """
    with create_pipe_input() as pipe, _fail_on_hang():
        pipe.send_text(keystrokes)
        pipe.close()
        yield pipe


def drive(
    keystrokes: str,
    *,
    session: ReplSession | None = None,
    echo: bool = True,
    check_only: bool = False,
) -> str:
    """Feed *keystrokes* to a headless REPL and return everything it printed."""
    repl_session = session if session is not None else ReplSession()
    # Session bootstrapping compiles the standard library, which is setup work
    # rather than a response to the scripted terminal input. Keep the hang
    # guard focused on the console loop it is meant to validate.
    assert repl_session.open() == ()
    with _scripted_input(keystrokes) as pipe:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            run_console(
                repl_session,
                echo=echo,
                check_only=check_only,
                history_path=None,  # InMemoryHistory — never touch real $HOME
                input=pipe,
                output=DummyOutput(),
            )
    return out.getvalue()


# ---------------------------------------------------------------------------
# Loop: submit / exit / cancel
# ---------------------------------------------------------------------------


class TestLoop:
    def test_exhausted_input_ends_the_loop(self) -> None:
        """A script that never asks to quit still terminates, at end of input.

        `drive` closes the write end of the pipe once the keystrokes are in it,
        so the loop reaches a real end-of-input and stops. Without that close
        the reader would block on a pipe that nothing will ever write to again,
        and the only thing standing between a mistyped script and a wedged test
        session would be a wall-clock deadline -- which has to be generous
        enough for a loaded machine, and so is a poor way to notice this.
        """
        output = drive("let x = 1\r")
        assert "AgL REPL" in output

    def test_banner_is_printed(self) -> None:
        output = drive("\x04")
        assert "AgL REPL" in output
        # The banner points the user at :help and how to quit.
        assert ":help" in output
        assert ":quit" in output

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
            ("do[0]", "  ()", None),
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

    @pytest.mark.parametrize("quote", ['"""', "'''"])
    def test_triple_quoted_string_continues_through_blank_lines(self, quote: str) -> None:
        # Pressing Enter on a blank line inside an open triple-quoted string
        # inserts another newline instead of force-submitting the broken entry.
        output = drive(f"let text = {quote}first\r\rsecond{quote}\rtext\r\x04")
        assert "first" in output
        assert "second" in output

    def test_raw_tail_block_continues_until_blank_line_then_executes(self) -> None:
        # A raw-tail header opens an interactive block. Enter after each content
        # line must keep collecting the payload; the blank continuation line
        # closes it and runs one shell call.
        shell = FakeShell([{"command": "echo one\necho two", "stdout": "one\ntwo\n"}])
        with patch("agm.core.process.run_capture_result", side_effect=shell):
            output = drive(
                "exec$\r  echo one\r  echo two\r\r\x04", session=ReplSession(default_stdlib=True)
            )

        assert ": error:" not in output.lower()
        # Exactly one shell call, carrying the whole payload, whose scripted
        # output comes back through the console as the entry's value.
        shell.assert_complete()
        assert "one" in output
        assert "two" in output

    def test_raw_tail_ask_block_continues_and_uses_mocked_default_agent(self) -> None:
        agent = _CountingAgent("mocked reply")
        output = drive(
            "ask$\r  summarize this\r\r\x04",
            session=ReplSession(agent_dispatcher=agent, default_stdlib=True),
        )

        assert agent.calls == 1
        assert agent.prompts == ["summarize this"]
        assert "mocked reply" in output


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------


class TestLexer:
    def test_info_output_is_highlighted_as_agl(self) -> None:
        session = ReplSession()
        assert session.eval_entry("record Issue()").ok
        text = "Issue is a record type.\nType:\n  record Issue"
        fragments = _highlighted_agl_fragments(
            text, session, ((0, len("Issue")), (len("Issue is a record type.\nType:\n"), len(text)))
        )
        styles = {style for style, _text in fragments}

        assert "class:agl.keyword" in styles
        assert "class:agl.type" in styles
        assert fragments[0] == ("class:agl.type", "Issue")
        assert ("class:agl.keyword", "is") not in fragments

    def test_info_type_is_highlighted_without_lexing_its_prose(self) -> None:
        text = "1 is a value.\nType:\n  int"
        type_start = len("1 is a value.\nType:\n")
        fragments = _highlighted_agl_fragments(text, agl_ranges=((0, 1), (type_start, len(text))))

        assert ("class:agl.number", "1") in fragments
        assert ("class:agl.type", "int") in fragments
        assert ("class:agl.keyword", "is") not in fragments

    def test_highlighted_fragment_preserves_newlines_and_trailing_prose(self) -> None:
        source = "let x = 1\nx"
        trailing = "\nLocation: <repl>:1:5"
        fragments = _highlighted_agl_fragments(source + trailing, agl_ranges=((0, len(source)),))

        assert ("", "\n") in fragments
        assert fragments[-1] == ("", trailing)

    def test_info_command_uses_the_highlighted_writer(self) -> None:
        output = drive("let count = 1\r:info count\r\x04")

        assert "count is a binding" in output
        assert "let count" in output

    def test_styles_a_sample_line(self) -> None:
        lexer = AglPromptLexer()
        fragments = lexer.lex_document(Document("let x = 1 + foo"))(0)
        styles = {style for style, _text in fragments}
        assert "class:agl.keyword" in styles  # let
        assert "class:agl.operator" in styles  # = / +
        assert "class:agl.number" in styles  # 1
        # The full line text is preserved across the fragments.
        assert "".join(text for _style, text in fragments) == "let x = 1 + foo"

    def test_half_typed_line_does_not_raise(self) -> None:
        lexer = AglPromptLexer()
        # An invalid character mid-line must fall back to plain text, not raise.
        source = "let x = \u200bbad"
        fragments = lexer.lex_document(Document(source))(0)
        assert "".join(text for _style, text in fragments) == source

    def test_token_without_source_position_is_not_styled(self) -> None:
        source = "let x = 1 + foo"
        expected = AglPromptLexer().lex_document(Document(source))(0)
        with patch(
            "agm.agl.repl.console.tokenize",
            side_effect=lambda text: iter([Token("NAME", "ghost"), *tokenize(text)]),
        ):
            fragments = AglPromptLexer().lex_document(Document(source))(0)
        assert fragments == expected

    @pytest.mark.parametrize("quote", ['"""', "'''"])
    def test_unterminated_triple_quoted_string_preserves_prefix_highlighting(
        self, quote: str
    ) -> None:
        lexer = AglPromptLexer()
        fragments = lexer.lex_document(Document(f"let x = {quote}"))(0)
        assert ("class:agl.keyword", "let") in fragments
        assert "".join(text for _style, text in fragments) == f"let x = {quote}"

    def test_string_literal_is_styled(self) -> None:
        lexer = AglPromptLexer()
        fragments = lexer.lex_document(Document('print "hello"'))(0)
        styles = {style for style, _text in fragments}
        assert "class:agl.string" in styles

    def test_partial_application_placeholders_are_styled_as_operators(self) -> None:
        fragments = AglPromptLexer().lex_document(Document("add(?, ?2)"))(0)

        assert ("class:agl.operator", "?") in fragments
        assert ("class:agl.operator", "?2") in fragments
        assert "".join(text for _style, text in fragments) == "add(?, ?2)"

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

    @pytest.mark.parametrize(
        "line",
        [
            '"%{x}"',
            '"a%{x}"',
            '"a%{x}b"',
            '"%{x}%{y}"',
            '"${PROJ_DIR}"',
            '"a${HOME}b"',
            '"${A}%{x}${B}"',
        ],
    )
    def test_string_interpolation_is_not_duplicated(self, line: str) -> None:
        # Interpolation tokens can overlap in source: a STRING_FRAGMENT spans the
        # following INTERP_START, and an environment hole's synthetic body tokens
        # share its name's characters. The prompt highlighter must still
        # partition the source so every typed character renders exactly once.
        fragments = AglPromptLexer().lex_document(Document(line))(0)
        assert "".join(text for _style, text in fragments) == line

    def test_out_of_range_line_does_not_crash(self) -> None:
        lexer = AglPromptLexer()
        getter = lexer.lex_document(Document("let x = 1"))
        # Asking for a line beyond the document yields an empty line, not a crash.
        assert getter(0)  # in range
        assert getter(5) == []

    def test_positionless_synthetic_token_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A lexer token without source offsets leaves the typed line intact."""
        monkeypatch.setattr(
            console,
            "tokenize",
            lambda _text: [Token("NAME", "x", start_pos=None, end_pos=None)],
        )

        fragments = AglPromptLexer().lex_document(Document("x"))(0)

        assert fragments == [("", "x")]

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

    def test_attribute_on_its_own_line_is_styled(self) -> None:
        # A declaration attribute may sit on the line above its target; the
        # `@` and the attribute name colour as one attribute.
        getter = AglPromptLexer().lex_document(Document("@arg-named\ndef f(x: int) -> int = x"))
        first = getter(0)
        assert ("class:agl.attribute", "@") in first
        assert ("class:agl.attribute", "arg-named") in first
        assert "".join(text for _style, text in first) == "@arg-named"

    def test_attribute_with_arguments_is_styled(self) -> None:
        # Same-line placement, with an argument list: the arguments keep their
        # own colours while the `@name` prefix colours as an attribute.
        line = '@doc("greets") def greet(x: text) -> text = x'
        fragments = AglPromptLexer().lex_document(Document(line))(0)
        assert ("class:agl.attribute", "@") in fragments
        assert ("class:agl.attribute", "doc") in fragments
        assert "class:agl.string" in {style for style, _text in fragments}
        assert "".join(text for _style, text in fragments) == line

    def test_attribute_on_a_parameter_is_styled(self) -> None:
        line = "def f(@arg-pos x: int) -> int = x"
        fragments = AglPromptLexer().lex_document(Document(line))(0)
        assert ("class:agl.attribute", "arg-pos") in fragments
        assert "".join(text for _style, text in fragments) == line

    def test_attributed_enum_member_still_colours_as_a_constructor(self) -> None:
        # An attribute prefix is transparent to the declaration walk, so the
        # member name after it is still recognised as a constructor.
        getter = AglPromptLexer().lex_document(
            Document('enum Outcome\n  | @doc("fine") Ok(value: int)')
        )
        member = getter(1)
        assert ("class:agl.attribute", "doc") in member
        assert self._style_of(member, "Ok") == "class:agl.constructor"

    def test_half_typed_attribute_argument_list_is_styled(self) -> None:
        # An unclosed argument list is normal mid-entry: it must still colour
        # and reconstruct the line exactly.
        line = '@doc("greets"'
        fragments = AglPromptLexer().lex_document(Document(line))(0)
        assert ("class:agl.attribute", "doc") in fragments
        assert "".join(text for _style, text in fragments) == line

    def test_attribute_argument_list_may_nest_calls(self) -> None:
        # The prefix spans the whole argument list, nested calls included, so
        # the member name after a nested `)` is still read as a constructor.
        getter = AglPromptLexer().lex_document(
            Document('enum E\n  | @doc(join("a", "b")) Ok(v: int)')
        )
        member = getter(1)
        assert ("class:agl.attribute", "doc") in member
        assert self._style_of(member, "Ok") == "class:agl.constructor"

    def test_unclosed_attribute_arguments_do_not_swallow_the_entry(self) -> None:
        # A half-typed argument list covers only the `@name` prefix, so the
        # declarations after it still colour as themselves.
        getter = AglPromptLexer().lex_document(Document('@doc("x"\nenum E\n  | A()\n  | B()'))
        assert self._style_of(getter(1), "E") == "class:agl.type"
        assert self._style_of(getter(2), "A") == "class:agl.constructor"
        assert self._style_of(getter(3), "B") == "class:agl.constructor"

    @pytest.mark.parametrize("line", ["@", "@ 1"])
    def test_at_sign_without_a_name_is_not_an_attribute(self, line: str) -> None:
        # Without a name after it there is no attribute yet — at the end of a
        # half-typed entry or before anything that is not a name — so the `@`
        # keeps its plain operator colour.
        fragments = AglPromptLexer().lex_document(Document(line))(0)
        assert ("class:agl.operator", "@") in fragments
        assert "".join(text for _style, text in fragments) == line

    @staticmethod
    def _style_of(fragments: object, word: str) -> str | None:
        """Return the style of the first fragment whose text equals *word*."""
        assert isinstance(fragments, list)
        for style, text in fragments:
            if text == word:
                assert isinstance(style, str)
                return style
        return None

    @pytest.mark.parametrize("name", ["int", "text", "bool", "array", "dict", "decimal"])
    def test_builtin_type_name_is_styled_as_type(self, name: str) -> None:
        # Builtin type spellings lex as plain NAME (capitalization is meaningless)
        # yet must colour as types — even without a session.
        fragments = AglPromptLexer().lex_document(Document(f"let x: {name} = y"))(0)
        assert self._style_of(fragments, name) == "class:agl.type"

    def test_plain_identifier_is_not_styled_as_type(self) -> None:
        # An ordinary NAME that is neither a builtin nor a declared type/constructor
        # stays a plain name, not a type.
        fragments = AglPromptLexer().lex_document(Document("let foo = bar"))(0)
        assert self._style_of(fragments, "foo") == "class:agl.name"
        assert self._style_of(fragments, "bar") == "class:agl.name"

    def test_record_name_colours_by_position(self) -> None:
        # A record name doubles as a type and a constructor. In a type annotation
        # it colours as a type; at a construction call (followed by `(`) it
        # colours as a constructor.
        session = ReplSession()
        result = session.eval_entry("record Pt\n  x: int\n  y: int")
        assert result.ok
        lexer = AglPromptLexer(session)

        annotation = lexer.lex_document(Document("let p: Pt = q"))(0)
        assert self._style_of(annotation, "Pt") == "class:agl.type"

        construction = lexer.lex_document(Document("Pt(x: 1, y: 2)"))(0)
        assert self._style_of(construction, "Pt") == "class:agl.constructor"

    def test_enum_variant_is_styled_as_constructor(self) -> None:
        # An enum variant is a pure constructor (never a type name), so it gets
        # the distinct constructor style — while the enum type itself is a type.
        session = ReplSession()
        result = session.eval_entry("enum Outcome\n  | ok(value: int)\n  | err(code: int)")
        assert result.ok
        lexer = AglPromptLexer(session)

        usage = lexer.lex_document(Document("let r: Outcome = ok(value: 1)"))(0)
        assert self._style_of(usage, "Outcome") == "class:agl.type"
        assert self._style_of(usage, "ok") == "class:agl.constructor"

    def test_explicit_type_arg_constructor_is_styled(self) -> None:
        # A constructor with an explicit type-argument override (`::[…]`) still
        # colours as a constructor (DCOLON look-ahead).
        session = ReplSession()
        assert session.eval_entry("enum Box\n  | wrap(value: int)").ok
        lexer = AglPromptLexer(session)
        usage = lexer.lex_document(Document("wrap::[int](value: 1)"))(0)
        assert self._style_of(usage, "wrap") == "class:agl.constructor"

    def test_type_alias_name_is_styled_as_type(self) -> None:
        # The aliased name in a `type` declaration colours as a type (and so do
        # its later references); no session needed.
        fragments = AglPromptLexer().lex_document(Document("type Id = int"))(0)
        assert self._style_of(fragments, "Id") == "class:agl.type"

    def test_in_progress_record_declaration_is_styled(self) -> None:
        # The name in a record header colours immediately, before the entry is
        # ever submitted (no session needed): positional declaration-site rule.
        fragments = AglPromptLexer().lex_document(Document("record Box"))(0)
        assert self._style_of(fragments, "Box") == "class:agl.type"

    def test_in_progress_enum_shares_name_with_constructor(self) -> None:
        # The reported case: `enum A | A` while typing. The header `A` is the
        # type, the variant `A` (after `|`) is the constructor — both coloured
        # distinctly even though they share a name and nothing is in the session.
        fragments = AglPromptLexer().lex_document(Document("enum A | A"))(0)
        type_a, ctor_a = (f for f in fragments if f[1] == "A")
        assert type_a[0] == "class:agl.type"
        assert ctor_a[0] == "class:agl.constructor"

    def test_later_pipe_is_not_a_variant(self) -> None:
        # A `|` in a `case` after an enum on the same entry must not be read as a
        # variant introducer: a bare pattern name has no lexical styling category.
        text = "enum A | A\ncase A(x: 1) of | y => y"
        lexer = AglPromptLexer()
        getter = lexer.lex_document(Document(text))
        second_line = getter(1)
        assert self._style_of(second_line, "y") == "class:agl.name"

    def test_declared_type_not_styled_without_session(self) -> None:
        # The same name is plain when no session backs the lexer (a user-defined
        # type is only known via session state).
        fragments = AglPromptLexer().lex_document(Document("let p: Pt = q"))(0)
        assert self._style_of(fragments, "Pt") == "class:agl.name"

    @pytest.mark.parametrize(
        ("source", "word"),
        [
            ("import std/text", "import"),
            ("use std/text::foo", "use"),
            ("export std/text", "export"),
            ("import std/text hiding foo", "hiding"),
        ],
    )
    def test_module_header_soft_keyword_is_styled(self, source: str, word: str) -> None:
        # Soft keywords are contextually promoted rather than reserved, so they
        # are absent from the reserved-word set; they must still colour like the
        # keywords they are inside their promotion window.
        fragments = AglPromptLexer().lex_document(Document(source))(0)
        assert self._style_of(fragments, word) == "class:agl.keyword"

    def test_scope_region_soft_keywords_are_styled(self) -> None:
        text = "scope a::b\n  let x = 1\nend a::b"
        getter = AglPromptLexer().lex_document(Document(text))
        assert self._style_of(getter(0), "scope") == "class:agl.keyword"
        assert self._style_of(getter(2), "end") == "class:agl.keyword"

    def test_soft_keyword_outside_its_window_stays_plain(self) -> None:
        # Outside their promotion window the same spellings are ordinary names.
        fragments = AglPromptLexer().lex_document(Document("let x = import + end"))(0)
        assert self._style_of(fragments, "import") == "class:agl.name"
        assert self._style_of(fragments, "end") == "class:agl.name"

    def test_module_header_path_is_styled_as_a_name(self) -> None:
        # The lexer merges a header path into one token; it is a name reference,
        # not an unstyled gap between the keyword and the rest of the line.
        fragments = AglPromptLexer().lex_document(Document("import std/text"))(0)
        assert self._style_of(fragments, "std/text") == "class:agl.name"

    def test_module_qualifier_is_styled_as_a_name(self) -> None:
        # `foo/bar::baz` used to colour only `baz`, leaving the merged qualifier
        # prefix as an unstyled hole.
        fragments = AglPromptLexer().lex_document(Document("let z = foo/bar::baz"))(0)
        assert self._style_of(fragments, "foo/bar::") == "class:agl.name"
        assert self._style_of(fragments, "baz") == "class:agl.name"

    def test_wildcard_module_tail_is_styled_as_an_operator(self) -> None:
        fragments = AglPromptLexer().lex_document(Document("import std/text/*"))(0)
        assert self._style_of(fragments, "/*") == "class:agl.operator"

    @pytest.mark.parametrize(
        ("source", "operator"),
        [
            ("let f: (int) -> int = g", "->"),
            ("var y := 1", ":="),
            ("let m = M::[int](1)", "::"),
        ],
    )
    def test_operator_is_styled(self, source: str, operator: str) -> None:
        fragments = AglPromptLexer().lex_document(Document(source))(0)
        assert self._style_of(fragments, operator) == "class:agl.operator"

    def test_trailing_comment_is_styled(self) -> None:
        fragments = AglPromptLexer().lex_document(Document("let x = 1  # note"))(0)
        assert self._style_of(fragments, "# note") == "class:agl.comment"
        assert self._style_of(fragments, "let") == "class:agl.keyword"
        assert "".join(text for _style, text in fragments) == "let x = 1  # note"

    def test_comment_only_line_is_styled(self) -> None:
        getter = AglPromptLexer().lex_document(Document("# note\nlet x = 1"))
        assert self._style_of(getter(0), "# note") == "class:agl.comment"
        assert self._style_of(getter(1), "let") == "class:agl.keyword"

    def test_hash_inside_a_string_is_not_styled_as_a_comment(self) -> None:
        source = 'let x = "a # b"'
        fragments = AglPromptLexer().lex_document(Document(source))(0)
        styles = {style for style, _text in fragments}
        assert "class:agl.comment" not in styles
        assert "".join(text for _style, text in fragments) == source

    def test_half_typed_string_after_a_comment_line_is_styled(self) -> None:
        # A half-typed entry does not tokenize, so highlighting falls back to
        # scanning for the open quote.  The scan must step over the preceding
        # comment line rather than reading its ``#`` as the start of one that
        # runs to the end of the entry.
        getter = AglPromptLexer().lex_document(Document('# note\nlet x = "abc'))
        assert self._style_of(getter(0), "# note") == "class:agl.comment"
        assert self._style_of(getter(1), '"abc') == "class:agl.string"

    def test_exception_name_colours_by_position(self) -> None:
        # An exception declares a type and a constructor, exactly like a record.
        lexer = AglPromptLexer()
        declaration = lexer.lex_document(Document("exception Boom\n  msg: text"))(0)
        assert self._style_of(declaration, "Boom") == "class:agl.type"

        construction = lexer.lex_document(Document('exception Boom\nraise Boom(msg: "x")'))(1)
        assert self._style_of(construction, "Boom") == "class:agl.constructor"


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

    def test_completes_soft_keywords(self) -> None:
        # Soft keywords are contextually promoted, so they are absent from the
        # reserved-word set the completer is built from; they must still be
        # offered — a user typing ``impo`` expects ``import``.
        completer = AglCompleter(ReplSession())
        for name in ("import", "use", "export", "hiding", "scope", "end"):
            assert name in _completions(completer, "")

    def test_completes_import_prefix(self) -> None:
        completer = AglCompleter(ReplSession())
        assert "import" in _completions(completer, "impo")

    def test_completes_builtin_calls(self) -> None:
        # Builtin call names are not reserved keywords and are not promoted
        # bindings; the completer must still offer them so a user typing
        # ``ask-...`` or ``print(`` gets a suggestion.
        completer = AglCompleter(ReplSession())
        for name in ("print", "render", "exec", "ask", "ask-request"):
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

    def test_deduplicates_builtin_default_agent_name(self) -> None:
        # A configured default agent is also named ``ask``; completion should
        # keep the builtin-first candidate once, not offer a duplicate.
        completer = AglCompleter(ReplSession(agent_dispatcher=_CountingAgent()))
        assert _completions(completer, "as").count("ask") == 1

    def test_deduplicates_a_name_from_multiple_completion_sources(self) -> None:
        class SessionWithBuiltinNamedBinding:
            def bindings(self) -> list[tuple[str, object, object]]:
                return [("print", object(), object())]

        completer = AglCompleter(SessionWithBuiltinNamedBinding())
        assert _completions(completer, "pr").count("print") == 1

    def test_no_completion_after_interpolation_opener(self) -> None:
        # Regression: when completions were offered with an empty word after
        # punctuation, typing `%{x}` interactively could accept a stale
        # completion and leave `%{%{x}` in the buffer.
        completer = AglCompleter(ReplSession())
        assert _completions(completer, "%{") == []

    @pytest.mark.parametrize("quote", ['"""', "'''"])
    def test_no_completion_inside_unterminated_triple_quoted_string(self, quote: str) -> None:
        completer = AglCompleter(ReplSession())
        assert _completions(completer, f"let prompt = {quote}le") == []
        assert _completions(completer, f"let prompt = {quote}first\n\nle") == []

    @pytest.mark.parametrize("quote", ['"', "'"])
    def test_no_completion_inside_unterminated_string(self, quote: str) -> None:
        completer = AglCompleter(ReplSession())
        assert _completions(completer, f"let prompt = {quote}le") == []
        assert _completions(completer, f"let prompt = {quote}escaped \\le") == []

    def test_comment_prefix_keeps_existing_word_completion_behavior(self) -> None:
        completer = AglCompleter(ReplSession())
        assert "let" in _completions(completer, '# "not a string" le')


# ---------------------------------------------------------------------------
# Evaluated output via the loop
# ---------------------------------------------------------------------------


class TestEvalOutput:
    def test_inline_raw_tail_exec_evaluates_through_the_console(self) -> None:
        shell = FakeShell([{"command": "echo hi", "stdout": "hi\n"}])
        with patch("agm.core.process.run_capture_result", side_effect=shell):
            output = drive("exec$ echo hi\r\x04", session=ReplSession(default_stdlib=True))

        assert ": error:" not in output.lower()
        # Exactly one shell call, and its scripted output is echoed as the value.
        shell.assert_complete()
        assert "hi" in output

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
        # An entry with an agent call type-checks and echoes its type, but the fake agent
        # is never invoked and no binding is persisted.
        agent = _CountingAgent("should-not-be-used")
        session = ReplSession(agent_dispatcher=agent, default_stdlib=True)
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
# Meta-commands through the loop
# ---------------------------------------------------------------------------


class TestMetaThroughLoop:
    def test_set_echo_off_suppresses_then_on_restores(self) -> None:
        # echo off → the binding is not echoed; echo on → the next binding is.
        output = drive(":set echo off\rlet a = 1\r:set echo on\rlet b = 2\r\x04")
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
        # This command needs no prelude declarations. Avoid compiling the full
        # standard library twice, which makes the headless-loop hang guard
        # spuriously fire when the suite runs in parallel.
        output = drive("1 + 2\r:type 1 + 2\r\x04", session=ReplSession())
        assert "int" in output

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

        text = "record MemoRecord"
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

    def test_infix_at_keyword_is_highlighted(self) -> None:
        # The contextual ``at`` keyword in an infix declaration must be styled
        # as a keyword (it is a plain NAME token, so it needs positional rules).
        fragments = AglPromptLexer().lex_document(Document("infixl |> at 5"))(0)
        assert ("class:agl.keyword", "at") in fragments
        # The relative-priority form: ``at`` is still a keyword (``prio`` already
        # is one as a true keyword).
        fragments = AglPromptLexer().lex_document(Document("infixr << at prio > + 1"))(0)
        assert ("class:agl.keyword", "at") in fragments

    def test_infix_at_not_highlighted_outside_infix_decl(self) -> None:
        # ``at`` is a contextual keyword: as a plain identifier elsewhere it must
        # NOT be styled as a keyword.
        fragments = AglPromptLexer().lex_document(Document("let at = 1"))(0)
        assert ("class:agl.keyword", "at") not in fragments

    def test_infix_decl_without_priority_clause(self) -> None:
        # An infix decl with no ``at priority`` clause: the operator is followed
        # by the next item, so nothing is mis-highlighted as the ``at`` keyword.
        fragments = AglPromptLexer().lex_document(Document("infixl |>\n1 + 2"))(0)
        assert ("class:agl.keyword", "infixl") in fragments
        assert not any(t == "at" for _s, t in fragments)

    def test_empty_line_in_middle(self) -> None:
        """An empty line in a multi-line input produces an empty fragment list."""
        text = "let x = 1\n\nlet y = 2"
        styled = self._get_styled_lines(text)
        assert len(styled) == 3
        assert styled[1] == [("", "")]


# ---------------------------------------------------------------------------
# Theme switching through the loop
# ---------------------------------------------------------------------------


class TestThemeThroughLoop:
    def _drive_with_save_tracking(
        self,
        keystrokes: str,
        *,
        initial_theme: str = "dark",
    ) -> tuple[str, list[tuple[str, "str | bool"]]]:
        """Run the loop, capture output and recorded on_setting_save calls."""
        saved: list[tuple[str, "str | bool"]] = []
        with _scripted_input(keystrokes) as pipe:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                run_console(
                    ReplSession(),
                    history_path=None,
                    theme=initial_theme,
                    on_setting_save=lambda key, value: saved.append((key, value)),
                    input=pipe,
                    output=DummyOutput(),
                )
        return out.getvalue(), saved

    def test_theme_command_reports_current_theme(self) -> None:
        output, _ = self._drive_with_save_tracking(":theme\r\x04", initial_theme="dark")
        assert "dark" in output

    def test_theme_switch_prints_confirmation(self) -> None:
        output, _ = self._drive_with_save_tracking(":theme light\r\x04")
        assert "light" in output

    def test_theme_switch_invokes_save_callback(self) -> None:
        _, saved = self._drive_with_save_tracking(":theme light\r\x04")
        assert saved == [("theme", "light")]

    def test_theme_switch_to_same_value_still_invokes_the_save_callback(self) -> None:
        # An explicit ``:theme dark`` still persists its target, even when it
        # already matches the active theme.
        _, saved = self._drive_with_save_tracking(":theme dark\r\x04", initial_theme="dark")
        assert saved == [("theme", "dark")]

    def test_multiple_theme_switches_all_saved(self) -> None:
        _, saved = self._drive_with_save_tracking(":theme light\r:theme dark\r:theme auto\r\x04")
        assert saved == [("theme", "light"), ("theme", "dark"), ("theme", "auto")]

    def test_unknown_theme_does_not_trigger_save(self) -> None:
        _, saved = self._drive_with_save_tracking(":theme neon\r\x04")
        assert saved == []

    def test_theme_switch_without_save_callback_does_not_raise(self) -> None:
        # on_setting_save=None (the default) — must not raise when theme changes.
        with _scripted_input(":theme light\r\x04") as pipe:
            with contextlib.redirect_stdout(io.StringIO()):
                run_console(
                    ReplSession(),
                    theme="dark",
                    on_setting_save=None,
                    history_path=None,
                    input=pipe,
                    output=DummyOutput(),
                )

    def test_set_echo_and_echo_unit_invoke_save_callback(self) -> None:
        _, saved = self._drive_with_save_tracking(":set echo off\r:set echo-unit on\r\x04")
        assert saved == [("echo", False), ("echo-unit", True)]

    def test_build_prompt_session_dark_theme(self) -> None:
        from agm.agl.repl.themes import DARK_THEME

        with create_pipe_input() as pipe:
            ps = build_prompt_session(ReplSession(), theme="dark", input=pipe, output=DummyOutput())
        assert ps.style is DARK_THEME

    def test_build_prompt_session_light_theme(self) -> None:
        from agm.agl.repl.themes import LIGHT_THEME

        with create_pipe_input() as pipe:
            ps = build_prompt_session(
                ReplSession(), theme="light", input=pipe, output=DummyOutput()
            )
        assert ps.style is LIGHT_THEME
