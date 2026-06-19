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
    DCOLON,
    DOT,
    GRAMMAR_TOKEN_REMAP,
    HIDING,
    IMPORT,
    INDEX_LSQB,
    INT,
    LOOP_BOUND,
    LSQB,
    MODPATH,
    MODQUAL,
    PRIVATE,
    QUALIFIED,
    RSQB,
    TYPE_NAME,
    USING,
    VAR_NAME,
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


_ITEM_START_TYPES = frozenset({"_NEWLINE", "_INDENT", "_DEDENT", "SEMICOLON"})


def _promote_soft_keywords(tokens: list[Token]) -> list[Token]:
    """Contextually promote soft keywords in the post-layout token stream.

    Rules:
    - 'import' → IMPORT when preceded by start-of-file / _NEWLINE / _INDENT /
      _DEDENT / SEMICOLON (i.e. at item-start position).
    - 'private' → PRIVATE under the same item-start condition.
    - 'qualified' → QUALIFIED, 'using' → USING, 'hiding' → HIDING only
      within an import declaration line (after IMPORT has been emitted on
      the current logical line, up to the next line/statement terminator).
    """
    result: list[Token] = []
    in_import_line = False
    prev_type: str | None = None  # None means start-of-stream

    for tok in tokens:
        tt = tok.type
        tv = str(tok)

        # Track import-line window: close on line/stmt terminators
        if tt in ("_NEWLINE", "_INDENT", "_DEDENT", "SEMICOLON"):
            in_import_line = False

        if tt == VAR_NAME:
            at_item_start = prev_type is None or prev_type in _ITEM_START_TYPES
            if tv == "import" and at_item_start:
                tok = Token(
                    IMPORT, tv,
                    start_pos=tok.start_pos, line=tok.line, column=tok.column,
                    end_line=tok.end_line, end_column=tok.end_column, end_pos=tok.end_pos,
                )
                in_import_line = True
            elif tv == "private" and at_item_start:
                tok = Token(
                    PRIVATE, tv,
                    start_pos=tok.start_pos, line=tok.line, column=tok.column,
                    end_line=tok.end_line, end_column=tok.end_column, end_pos=tok.end_pos,
                )
            elif in_import_line:
                if tv == "qualified":
                    tok = Token(
                        QUALIFIED, tv,
                        start_pos=tok.start_pos, line=tok.line, column=tok.column,
                        end_line=tok.end_line, end_column=tok.end_column, end_pos=tok.end_pos,
                    )
                elif tv == "using":
                    tok = Token(
                        USING, tv,
                        start_pos=tok.start_pos, line=tok.line, column=tok.column,
                        end_line=tok.end_line, end_column=tok.end_column, end_pos=tok.end_pos,
                    )
                elif tv == "hiding":
                    tok = Token(
                        HIDING, tv,
                        start_pos=tok.start_pos, line=tok.line, column=tok.column,
                        end_line=tok.end_line, end_column=tok.end_column, end_pos=tok.end_pos,
                    )

        result.append(tok)
        prev_type = tok.type

    return result


def _merge_modpath(tokens: list[Token]) -> list[Token]:
    """Merge import module paths into single MODPATH tokens.

    Pattern: immediately following an IMPORT token, consume
    VAR_NAME (DOT VAR_NAME)* into a single MODPATH token whose value
    is the dotted path (e.g. "foo.bar", "utils").

    This eliminates the LALR(1) shift/reduce conflict between
    ``var_ref : VAR_NAME`` / ``postfix: postfix DOT ...`` (expression grammar)
    and the ``module_path : VAR_NAME (DOT VAR_NAME)*`` (import grammar).
    By merging the path in the lexer, the grammar sees a single MODPATH token
    rather than the raw VAR_NAME DOT ... sequence.
    """
    result: list[Token] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok.type == IMPORT and i + 1 < n and tokens[i + 1].type == VAR_NAME:
            result.append(tok)
            i += 1
            # Absorb VAR_NAME (DOT VAR_NAME)*
            j = i
            seg_parts: list[str] = [str(tokens[j])]
            j += 1
            while j + 1 < n and tokens[j].type == DOT and tokens[j + 1].type == VAR_NAME:
                seg_parts.append(str(tokens[j + 1]))
                j += 2
            modpath_value = ".".join(seg_parts)
            first_tok = tokens[i]
            last_tok = tokens[j - 1]
            merged = Token(
                MODPATH,
                modpath_value,
                start_pos=first_tok.start_pos,
                line=first_tok.line,
                column=first_tok.column,
                end_line=last_tok.end_line,
                end_column=last_tok.end_column,
                end_pos=last_tok.end_pos,
            )
            result.append(merged)
            # If next token is DOT STAR, absorb into a separate STAR token
            # (wildcard tail). We keep DOT STAR as two tokens for the grammar.
            i = j
            continue
        result.append(tok)
        i += 1
    return result


def _merge_modqual(tokens: list[Token]) -> list[Token]:
    """Merge module-qualifier prefixes into single MODQUAL tokens.

    Pattern: (VAR_NAME | TYPE_NAME) (DOT VAR_NAME)* DCOLON where the token
    AFTER DCOLON is NOT LSQB.

    Merges the prefix including '::' into a single MODQUAL token whose value
    is the dotted qualifier (e.g. "foo.bar", "A", "A.baz"). The DCOLON itself
    is consumed into the MODQUAL token.

    The `next != LSQB` guard preserves the existing typed-call atom
    `callee::[T](args)` (VAR_NAME DCOLON LSQB must stay intact).

    A leading '::name' (empty qualifier, D9 self-reference) has no preceding
    name, so no merge fires; the bare DCOLON is handled by the grammar.
    """
    result: list[Token] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        tt = tok.type
        # Check if we start a potential module qualifier:
        # (VAR_NAME | TYPE_NAME) (DOT VAR_NAME)* DCOLON (not followed by LSQB)
        if tt in (VAR_NAME, TYPE_NAME):
            # Scan ahead: collect (DOT VAR_NAME)* then DCOLON
            j = i + 1
            while j + 1 < n and tokens[j].type == DOT and tokens[j + 1].type == VAR_NAME:
                j += 2
            # Now tokens[j] should be DCOLON (if this is a qualifier)
            if j < n and tokens[j].type == DCOLON:
                # Check the token after DCOLON is not LSQB
                next_after = tokens[j + 1].type if j + 1 < n else None
                if next_after != LSQB:
                    # Merge tokens[i..j] (inclusive of DCOLON at j) into MODQUAL
                    # Build the qualifier string: segments joined with '.'
                    seg_parts: list[str] = [str(tokens[i])]
                    k = i + 1
                    while k < j:
                        # skip DOT, take the name
                        k += 1  # skip DOT
                        seg_parts.append(str(tokens[k]))
                        k += 1
                    qualifier_value = ".".join(seg_parts)
                    first_tok = tokens[i]
                    last_tok = tokens[j]  # the DCOLON
                    merged = Token(
                        MODQUAL,
                        qualifier_value,
                        start_pos=first_tok.start_pos,
                        line=first_tok.line,
                        column=first_tok.column,
                        end_line=last_tok.end_line,
                        end_column=last_tok.end_column,
                        end_pos=last_tok.end_pos,
                    )
                    result.append(merged)
                    i = j + 1
                    continue
        result.append(tok)
        i += 1
    return result


def apply_module_passes(tokens: list[Token]) -> list[Token]:
    """Apply soft-keyword promotion, import path merging, and module-qualifier merging."""
    return _merge_modqual(_merge_modpath(_promote_soft_keywords(tokens)))


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
            after_remap = list(_remap(layout(scanner.scan())))
            tokens = _remap_index_brackets(apply_module_passes(after_remap))
        finally:
            sink = _TAB_WARNING_SINK.get()
            if sink is not None:
                sink.extend(scanner.tab_warnings)
        return iter(tokens)
