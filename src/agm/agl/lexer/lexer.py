"""AglLexer: Lark custom lexer for the AgL DSL.

Composes :mod:`~agm.agl.lexer.scanner` (raw tokens) with
:mod:`~agm.agl.lexer.layout` (INDENT/DEDENT injection) to produce the full
token stream required by the Lark LALR parser.

The lexer is wired as::

    Lark(grammar, parser="lalr", lexer=AglLexer, propagate_positions=True)

The ``lex`` method receives ``lexer_state`` and ``parser_state`` from Lark.
We read the source text from ``lexer_state.text`` and yield
:class:`lark.lexer.Token` objects with full position information.

Grammar terminal names
----------------------
Lark creates uppercase terminal names from grammar string literals (e.g.
``"pass"`` → terminal ``PASS``).  Because the raw scanner emits lowercase
keyword types (``"pass"``, ``"let"``, etc.), ``lex()`` applies a one-to-one
remapping: ``GRAMMAR_TOKEN_REMAP`` maps each lowercase keyword type to its
uppercase equivalent so the LALR parser can look up the right parse-table
entry.

The public :func:`~agm.agl.lexer.tokenize` helper goes through
``layout(scan())`` directly and preserves the lowercase types that tests
and downstream tooling rely on.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

from lark.lexer import Lexer, LexerState, Token

from agm.agl.diagnostics import Diagnostic
from agm.agl.lexer.layout import layout
from agm.agl.lexer.scanner import _Scanner
from agm.agl.lexer.tokens import (
    GRAMMAR_TOKEN_REMAP,
    INDEX_LSQB,
    INT,
    LOOP_BOUND,
    LSQB,
    RSQB,
)

_INDEX_PREDECESSORS = frozenset(
    {
        "VAR_NAME",
        "TYPE_NAME",
        INT,
        "DECIMAL",
        "TRUE",
        "FALSE",
        "NULL",
        "TEMPLATE_END",
        "RPAR",
        RSQB,
        "RBRACE",
    }
)

# Ambient sink for TAB advisories produced during a Lark-driven parse.  The
# lexer scans the source exactly once (no separate TAB pass); when a sink is
# active, ``AglLexer.lex`` deposits the scan's TAB advisories into it so the
# caller (e.g. ``WorkflowRuntime.prepare``) can surface them alongside parse
# diagnostics.  A ``ContextVar`` keeps nested/reentrant parses isolated.
_TAB_WARNING_SINK: contextvars.ContextVar[list[Diagnostic] | None] = (
    contextvars.ContextVar("agl_tab_warning_sink", default=None)
)


@contextmanager
def tab_warning_collector() -> Iterator[list[Diagnostic]]:
    """Collect TAB advisories produced by parses within the ``with`` block.

    Yields a list that ``AglLexer.lex`` appends to as it scans.  The list is
    populated even when the parse fails (the scan runs to completion before the
    grammar is consulted), so callers get every TAB advisory on every path.
    """
    sink: list[Diagnostic] = []
    token = _TAB_WARNING_SINK.set(sink)
    try:
        yield sink
    finally:
        _TAB_WARNING_SINK.reset(token)


def _remap(tokens: Iterator[Token]) -> Iterator[Token]:
    """Remap lowercase keyword token types to the uppercase names the grammar expects.

    Also applies the ``do[N]`` merge: a ``DO LSQB INT RSQB`` sequence is
    collapsed into ``DO LOOP_BOUND(N)`` so the grammar can use a single terminal
    ``LOOP_BOUND`` in the ``loop_bound`` rule and avoid the LALR(1) conflict with
    ``lit_list`` (which also matches ``LSQB INT RSQB``).
    """
    buf: list[Token] = []
    for tok in tokens:
        mapped = GRAMMAR_TOKEN_REMAP.get(tok.type)
        if mapped is not None:
            tok = Token(
                mapped,
                str(tok),
                start_pos=tok.start_pos,
                line=tok.line,
                column=tok.column,
                end_line=tok.end_line,
                end_column=tok.end_column,
                end_pos=tok.end_pos,
            )
        buf.append(tok)
        # Check for the LOOP_BOUND merge pattern: DO LSQB INT RSQB.
        # We need 4 tokens in the buffer to detect this.
        if len(buf) >= 4:
            t0, t1, t2, t3 = buf[-4], buf[-3], buf[-2], buf[-1]
            if (
                t0.type == "DO"
                and t1.type == LSQB
                and t2.type == INT
                and t3.type == RSQB
            ):
                # Merge LSQB INT RSQB → LOOP_BOUND, flush DO + LOOP_BOUND
                lb_tok = Token(
                    LOOP_BOUND,
                    str(t2),  # value is the integer string
                    start_pos=t1.start_pos,
                    line=t1.line,
                    column=t1.column,
                    end_line=t3.end_line,
                    end_column=t3.end_column,
                    end_pos=t3.end_pos,
                )
                buf[-3:] = [lb_tok]
                # Flush all but the newly added LOOP_BOUND — it'll flush in
                # subsequent iterations. Actually flush up to len-1.
                while len(buf) > 1:
                    yield buf.pop(0)
                continue
        # Flush tokens that can no longer be part of a loop-bound merge.
        # Keep up to 3 tokens in the buffer (we need 4 to detect the pattern).
        while len(buf) > 3:
            yield buf.pop(0)
    # Flush remaining buffer
    yield from buf


def _remap_index_brackets(tokens: list[Token]) -> list[Token]:
    """Turn adjacent expression brackets into INDEX_LSQB for the parser."""
    result: list[Token] = []
    previous: Token | None = None
    for tok in tokens:
        if (
            tok.type == LSQB
            and previous is not None
            and previous.type in _INDEX_PREDECESSORS
            and previous.end_pos == tok.start_pos
        ):
            tok = Token(
                INDEX_LSQB,
                str(tok),
                start_pos=tok.start_pos,
                line=tok.line,
                column=tok.column,
                end_line=tok.end_line,
                end_column=tok.end_column,
                end_pos=tok.end_pos,
            )
        result.append(tok)
        previous = tok
    return result


class AglLexer(Lexer):
    """Custom Lark lexer for AgL.

    Accepted by the Lark parser via ``lexer=AglLexer``; ``lexer_conf`` is
    received but not used (the grammar's terminal regex patterns are not needed
    because we generate all tokens ourselves).

    ``__future_interface__ = 1`` tells Lark to call ``lex(lexer_state,
    parser_state)`` directly (interface v1), which matches the method
    signature.  Without this, Lark wraps the lexer as interface=0 and calls
    ``lex(text)`` — the wrong arity.
    """

    __future_interface__ = 1

    def __init__(self, lexer_conf: object) -> None:
        # lexer_conf is accepted but unused; the scanner handles all tokenization.
        pass

    def lex(self, lexer_state: LexerState, parser_state: object) -> Iterator[Token]:
        """Tokenize ``lexer_state.text`` and yield Lark tokens.

        Keyword token types are remapped from lowercase (scanner convention)
        to the uppercase names expected by the Lark grammar.
        """
        # lexer_state.text may be a str or a lark.lexer.TextSlice; extract
        # the raw string for the scanner.
        raw = lexer_state.text
        if isinstance(raw, str):
            source = raw
        else:
            # TextSlice: use the underlying .text attribute.
            source = str(raw.text) if hasattr(raw, "text") else str(raw)
        # Drive the scan to completion up front (materialized) so the single
        # lex pass records EVERY TAB advisory before the grammar is consulted —
        # the advisories are then complete even if the parse later fails.  The
        # ``finally`` deposits whatever was collected, including on a LexError.
        scanner = _Scanner(source)
        try:
            tokens = _remap_index_brackets(list(_remap(layout(scanner.scan()))))
        finally:
            sink = _TAB_WARNING_SINK.get()
            if sink is not None:
                sink.extend(scanner.tab_warnings)
        return iter(tokens)
