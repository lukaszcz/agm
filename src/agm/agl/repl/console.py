"""prompt_toolkit front end for the AgL REPL.

This module is the **only** place that touches prompt_toolkit; everything else
in :mod:`agm.agl.repl` is UI-free.  It provides:

- :class:`AglPromptLexer` — syntax highlighting that drives the *real* AgL lexer
  (:func:`agm.agl.lexer.tokenize`) so colours track the grammar exactly;
- :class:`AglCompleter` — completion fed from live session state (keywords,
  bindings, agents, meta-command names);
- :func:`is_incomplete` — the multiline continuation predicate (delegates to the
  parser's structured incompleteness signal);
- :func:`build_prompt_session` — a configured ``PromptSession`` with history,
  styling, and the multiline Enter binding;
- :func:`run_console` — the read-eval-print loop itself.

mypy note: the whole surface types cleanly under ``--strict`` /
``disallow_any_expr`` without any ``stubs/prompt_toolkit/`` shims.  The Enter key
binding only reads ``event.current_buffer`` (precisely typed as ``Buffer``); it
never touches ``event.app`` (whose ``Application[Any]`` type would trip the
strict ``Any`` rule).  Ctrl-C / Ctrl-D rely on prompt_toolkit's default
behaviour (``KeyboardInterrupt`` / ``EOFError``), so no custom binding needs to
reach into the application object.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style

from agm.agl.lexer import tokenize
from agm.agl.lexer.tokens import KEYWORDS
from agm.agl.parser import is_incomplete_source
from agm.agl.repl import meta as meta_mod
from agm.agl.repl import render as render_mod

if TYPE_CHECKING:
    from pathlib import Path

    from agm.agl.repl.session import ReplSession


# ---------------------------------------------------------------------------
# Prompts and banner
# ---------------------------------------------------------------------------

PROMPT = "agl> "
CONTINUATION = "...> "
BANNER = "AgL REPL — type :help for commands, :quit or Ctrl-D to exit."

# AgL keywords offered by the completer (the reserved-word set, sorted for a
# stable suggestion order).
_KEYWORDS: tuple[str, ...] = tuple(sorted(KEYWORDS))


# ---------------------------------------------------------------------------
# Syntax highlighting
# ---------------------------------------------------------------------------

# Map an AgL lexer token type to a prompt_toolkit style class.  The public
# ``tokenize`` helper emits lowercase keyword types (``"let"``), the operator /
# identifier constant names, and the synthetic template tokens.  The map is kept
# small and centralized; anything unmapped falls through to plain text.
_STRING_TOKENS: frozenset[str] = frozenset(
    {"TEMPLATE_START", "STRING_FRAGMENT", "TEMPLATE_END", "INTERP_START", "INTERP_END"}
)
_NUMBER_TOKENS: frozenset[str] = frozenset({"INT", "DECIMAL", "LOOP_BOUND"})
_OPERATOR_TOKENS: frozenset[str] = frozenset(
    {
        "ARROW", "EQ", "NEQ", "LE", "GE", "LT", "GT", "PLUS", "MINUS", "STAR",
        "SLASH", "LPAR", "RPAR", "LSQB", "RSQB", "LBRACE", "RBRACE", "COLON",
        "COMMA", "DOT", "PIPE", "SEMICOLON", "EQ_EQ",
    }
)


def _style_class_for(token_type: str) -> str | None:
    """Return the style class for an AgL token type, or ``None`` for plain text."""
    if token_type in KEYWORDS:
        return "class:agl.keyword"
    if token_type in _STRING_TOKENS:
        return "class:agl.string"
    if token_type in _NUMBER_TOKENS:
        return "class:agl.number"
    if token_type in _OPERATOR_TOKENS:
        return "class:agl.operator"
    if token_type == "TYPE_NAME":
        return "class:agl.type"
    if token_type == "VAR_NAME":
        return "class:agl.name"
    return None


# The Style mapping the style classes above to concrete colours.
AGL_STYLE: Style = Style.from_dict(
    {
        "agl.keyword": "bold #569cd6",
        "agl.string": "#ce9178",
        "agl.number": "#b5cea8",
        "agl.operator": "#d4d4d4",
        "agl.type": "#4ec9b0",
        "agl.name": "",
        "agl.banner": "italic #808080",
        "agl.prompt": "bold #569cd6",
    }
)


class AglPromptLexer(Lexer):
    """A prompt_toolkit lexer that drives the real AgL lexer for highlighting.

    Tokenizing a half-typed or invalid line is normal at the prompt, so any
    lexer error is swallowed and the affected document falls back to plain
    styling (no raise ever escapes ``lex_document``).
    """

    def lex_document(
        self, document: Document
    ) -> Callable[[int], StyleAndTextTuples]:
        styled = self._styled_lines(document.text)

        def get_line(lineno: int) -> StyleAndTextTuples:
            # prompt_toolkit only asks for in-range lines; guard defensively so a
            # stray request can never raise out of the highlighter.
            if 0 <= lineno < len(styled):
                return styled[lineno]
            return []

        return get_line

    @staticmethod
    def _styled_lines(text: str) -> list[StyleAndTextTuples]:
        """Tokenize *text* and return per-line styled ``(style, text)`` fragments.

        Styled spans are collected as absolute ``(start, end, style)`` offsets
        into *text*, then sliced per line.  On any lexer error the whole document
        falls back to plain text — a partially typed line must never raise out of
        the prompt.
        """
        spans = _styled_spans(text)
        line_starts = _line_start_offsets(text)
        result: list[StyleAndTextTuples] = []
        for index, line in enumerate(text.split("\n")):
            base = line_starts[index]
            line_spans = [
                (start - base, end - base, style)
                for start, end, style in spans
                if base <= start < base + len(line)
            ]
            result.append(_fragments_for_line(line, line_spans))
        return result


def _styled_spans(text: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, style)`` offsets for the styleable tokens in *text*.

    Synthetic zero-width tokens (INDENT/DEDENT/NEWLINE) and unstyled token types
    are skipped.  A lexer error on a half-typed entry yields no spans (plain
    text), never a raise.
    """
    spans: list[tuple[int, int, str]] = []
    try:
        for token in tokenize(text):
            style = _style_class_for(token.type)
            start = token.start_pos
            end = token.end_pos
            if style is None or start is None or end is None or end <= start:
                continue
            spans.append((start, end, style))
    except Exception:
        return []
    return spans


def _line_start_offsets(text: str) -> list[int]:
    """Return the absolute offset at which each line of *text* begins."""
    offsets = [0]
    for line in text.split("\n")[:-1]:
        offsets.append(offsets[-1] + len(line) + 1)
    return offsets


def _fragments_for_line(
    line: str, spans: list[tuple[int, int, str]]
) -> StyleAndTextTuples:
    """Turn a line and its styled spans into ordered ``(style, text)`` fragments."""
    if not spans:
        return [("", line)]
    spans = sorted(spans)
    fragments: StyleAndTextTuples = []
    cursor = 0
    for start, end, style in spans:
        if start > cursor:
            fragments.append(("", line[cursor:start]))
        fragments.append((style, line[start:end]))
        cursor = end
    if cursor < len(line):
        fragments.append(("", line[cursor:]))
    return fragments


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------


class AglCompleter(Completer):
    """Complete the current word from live session state and meta-commands.

    On a line whose first non-blank character is ``:`` the completer offers
    meta-command names; otherwise it offers AgL keywords, current binding names,
    and agent names.  All candidate sources are read live from the session, so
    a binding defined in an earlier entry is immediately completable.
    """

    def __init__(self, session: "ReplSession") -> None:
        self._session = session

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        del complete_event
        text_before = document.text_before_cursor
        if text_before.lstrip().startswith(":"):
            yield from self._meta_completions(document)
            return
        yield from self._word_completions(document)

    def _meta_completions(self, document: Document) -> Iterable[Completion]:
        word = document.text_before_cursor.lstrip()
        for name in meta_mod.meta_command_names():
            if name.startswith(word):
                yield Completion(name, start_position=-len(word))

    def _word_completions(self, document: Document) -> Iterable[Completion]:
        word = document.get_word_before_cursor()
        for candidate in self._candidates():
            if candidate.startswith(word) and candidate != word:
                yield Completion(candidate, start_position=-len(word))

    def _candidates(self) -> list[str]:
        names: list[str] = list(_KEYWORDS)
        names.extend(name for name, _type, _value in self._session.bindings())
        names.extend(self._session.agents())
        return names


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
    parser signal decides.
    """
    if not text.strip():
        return False
    if text.endswith("\n"):
        return False
    return is_incomplete_source(text)


def has_runnable_statements(text: str) -> bool:
    """Return ``True`` when *text* contains at least one statement to evaluate.

    Blank, whitespace-only, and comment-only entries (AgL comments run from a
    ``#`` to end of line) have nothing to run.  The check tokenizes *text* with
    the real AgL lexer and looks for any non-trivial token — the lexer skips
    whitespace and comments entirely and emits no tokens for blank/comment-only
    input, while synthetic layout tokens (``_NEWLINE`` / ``_INDENT`` /
    ``_DEDENT``) carry no statement, so they are ignored.  Any lexer error (a
    half-typed entry never reaches here, but be defensive) is treated as
    *runnable* so the entry flows on to ``eval_entry`` and surfaces a real
    diagnostic rather than being silently dropped.
    """
    try:
        return any(token.type not in _TRIVIAL_TOKENS for token in tokenize(text))
    except Exception:
        return True


# Layout-only token types that carry no statement to evaluate.
_TRIVIAL_TOKENS: frozenset[str] = frozenset({"_NEWLINE", "_INDENT", "_DEDENT"})


def _make_key_bindings() -> KeyBindings:
    """Build the Enter binding implementing AgL-aware multiline continuation.

    Reads only ``event.current_buffer`` (precisely typed ``Buffer``); it never
    touches ``event.app``, keeping the binding free of prompt_toolkit's
    ``Any``-typed application surface.
    """
    bindings = KeyBindings()

    @bindings.add("enter")
    def _on_enter(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        # Decide on the buffer as it stands; Enter either extends it (insert a
        # newline) or submits it.  ``is_incomplete`` treats a buffer already
        # ending in a newline (Enter on a blank line) as a force-submit.
        if is_incomplete(buffer.text):
            buffer.insert_text("\n")
        else:
            buffer.validate_and_handle()

    return bindings


# ---------------------------------------------------------------------------
# PromptSession factory
# ---------------------------------------------------------------------------


def _make_history(history_path: "Path | None") -> History:
    """Return a ``FileHistory`` at *history_path*, or an in-memory one when None."""
    if history_path is None:
        return InMemoryHistory()
    return FileHistory(str(history_path))


def build_prompt_session(
    session: "ReplSession",
    *,
    history_path: "Path | None" = None,
    input: Input | None = None,
    output: Output | None = None,
) -> "PromptSession[str]":
    """Construct the configured ``PromptSession`` for the REPL loop.

    *input* / *output* are forwarded to prompt_toolkit so headless tests can
    inject a pipe input + ``DummyOutput``; left ``None`` they default to the
    real terminal.
    """
    return PromptSession(
        message=[("class:agl.prompt", PROMPT)],
        lexer=AglPromptLexer(),
        completer=AglCompleter(session),
        history=_make_history(history_path),
        style=AGL_STYLE,
        multiline=True,
        key_bindings=_make_key_bindings(),
        prompt_continuation=_prompt_continuation,
        input=input,
        output=output,
    )


def _prompt_continuation(
    width: int, line_number: int, wrap_count: int
) -> StyleAndTextTuples:
    """Render the ``...> `` continuation prompt for multiline entries."""
    del width, line_number, wrap_count
    return [("class:agl.prompt", CONTINUATION)]


# ---------------------------------------------------------------------------
# The read-eval-print loop
# ---------------------------------------------------------------------------


def run_console(
    session: "ReplSession",
    *,
    echo: bool = True,
    check_only: bool = False,
    history_path: "Path | None" = None,
    input: Input | None = None,
    output: Output | None = None,
) -> None:
    """Run the interactive AgL REPL against *session*.

    Reads one (possibly multiline) entry per iteration.  ``EOFError`` (Ctrl-D)
    or a ``:quit`` / ``:exit`` meta-command exits the loop; ``KeyboardInterrupt``
    (Ctrl-C) cancels the current entry and keeps looping.  A ``:`` line is routed
    to :func:`agm.agl.repl.meta.dispatch_meta`; a blank or comment-only entry is
    a no-op (fresh prompt, no error); any other entry is evaluated and its result
    rendered via :func:`agm.agl.repl.render.render_entry_result`.

    When *check_only* is set the REPL is in dry-run mode: each entry is run
    through the full static pipeline (parse / resolve / typecheck) only — no
    evaluation, no agent/exec calls, and no bindings are persisted — and its
    inferred type is echoed.

    The loop is intentionally thin: formatting lives in ``render`` and meta
    handling in ``meta`` so M3 can extend both without touching it.
    """
    prompt_session = build_prompt_session(
        session, history_path=history_path, input=input, output=output
    )
    ctx = meta_mod.MetaContext(session=session, echo=echo)

    print(BANNER)
    while True:
        try:
            entry = prompt_session.prompt()
        except KeyboardInterrupt:
            # Ctrl-C cancels the current entry but never exits the REPL.
            continue
        except EOFError:
            # Ctrl-D exits.
            break

        if entry.lstrip().startswith(":"):
            outcome = meta_mod.dispatch_meta(entry, ctx)
            if outcome.text is not None:
                print(outcome.text)
            if outcome.quit:
                break
            continue

        # Blank or comment-only entries have nothing to run; give a fresh prompt
        # without invoking the evaluator (whose parser would reject them).
        if not has_runnable_statements(entry):
            continue

        result = session.eval_entry(entry, check_only=check_only)
        rendered = render_mod.render_entry_result(
            result, echo=ctx.echo, check_only=check_only
        )
        if rendered is not None:
            print(rendered)
