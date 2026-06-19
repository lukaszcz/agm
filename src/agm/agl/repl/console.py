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

import bisect
import re
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
from agm.agl.repl import session as session_mod
from agm.agl.repl.agentmode import AgentMode
from agm.agl.scope.symbols import BUILTIN_CALL_NAMES

if TYPE_CHECKING:
    from pathlib import Path

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

# AgL keywords offered by the completer (the reserved-word set, sorted for a
# stable suggestion order).
_KEYWORDS: tuple[str, ...] = tuple(sorted(KEYWORDS))

# Word pattern for completion: AgL identifiers, including the hyphenated
# built-in call name ``ask-request``.  prompt_toolkit's default word finder
# treats ``-`` as a boundary, so without this pattern typing ``ask-r`` yields
# a one-character word (``r``) and never matches ``ask-request``.
_IDENT_WORD: re.Pattern[str] = re.compile(r"[A-Za-z0-9_-]+")


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
    if token_type == "NAME":
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
        into *text*, then bucketed into per-line lists in a **single pass** over
        ``spans`` using ``bisect`` over the precomputed line-start offsets.
        This gives O(spans) total work instead of the O(lines × spans) cost of
        filtering the full span list once per line.

        On any lexer error the whole document falls back to plain text — a
        partially typed line must never raise out of the prompt.
        """
        spans = _styled_spans(text)
        lines = text.split("\n")
        line_starts = _line_start_offsets(text)

        # Bucket spans by line index in one pass: for each span find the line
        # whose start ≤ span.start via bisect_right, then subtract 1 to get
        # the containing line's index.  All spans from _styled_spans are within
        # the text, so line_index is always in [0, len(lines)-1].
        per_line: list[list[tuple[int, int, str]]] = [[] for _ in lines]
        for start, end, style in spans:
            line_index = bisect.bisect_right(line_starts, start) - 1
            base = line_starts[line_index]
            per_line[line_index].append((start - base, end - base, style))

        return [_fragments_for_line(line, per_line[i]) for i, line in enumerate(lines)]


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
        word = document.get_word_before_cursor(pattern=_IDENT_WORD)
        for candidate in self._candidates():
            if candidate.startswith(word) and candidate != word:
                yield Completion(candidate, start_position=-len(word))

    def _candidates(self) -> list[str]:
        names: list[str] = list(_KEYWORDS)
        # The built-in call names (print/exec/ask/ask-request) are reserved
        # call-site identifiers, not keywords, so they are absent from
        # ``KEYWORDS``; add them explicitly so the completer offers them.
        names.extend(BUILTIN_CALL_NAMES)
        names.extend(name for name, _type, _value in self._session.bindings())
        names.extend(self._session.agents())
        # De-duplicate while preserving the stable keyword-first ordering.  A
        # default agent makes ``ask`` appear in both the built-in names and the
        # agents pool; offering it twice is redundant.
        seen: set[str] = set()
        unique: list[str] = []
        for name in names:
            if name not in seen:
                seen.add(name)
                unique.append(name)
        return unique


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


# ``has_runnable_statements`` (the blank/comment-only-entry predicate) lives in
# the UI-free ``session`` module so ``load_file`` can share it; re-exported here
# under its original name for the console loop and its tests.
has_runnable_statements = session_mod.has_runnable_statements


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
    terminal; they default to stdlib ``input`` / ``print``.
    """
    read: PromptReader = reader if reader is not None else input
    write: Callable[[str], None] = printer if printer is not None else print

    def confirm(callee: str, prompt: str) -> "ConfirmDecision":
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


def run_console(
    session: "ReplSession",
    *,
    echo: bool = True,
    check_only: bool = False,
    agent_mode: "AgentMode | None" = None,
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
    # A shared, mutable agent-mode holder: ``:agent`` mutates it here, and M4
    # will pass this SAME instance to the confirming agent wrapper so the wrapper
    # observes the mutation.  Defaults to confirm-each-call per plan decision 2.
    ctx = meta_mod.MetaContext(
        session=session,
        echo=echo,
        agent_mode=agent_mode if agent_mode is not None else AgentMode(),
    )

    print(format_banner(ctx.agent_mode))
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
