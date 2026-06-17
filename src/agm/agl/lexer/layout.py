"""INDENT / DEDENT filter for the AgL lexer.

Consumes the raw token stream from :mod:`agm.agl.lexer.scanner` and injects
synthetic ``_INDENT`` and ``_DEDENT`` tokens according to the indentation rules
described in plan §3.4.

Key rules
---------
- The indent stack starts at ``[0]``.
- Inside brackets/parens/braces (``paren_level > 0``) ``_NEWLINE`` tokens are
  suppressed (implicit line continuation).
- On a ``_NEWLINE`` token: compare the carried indentation width against the
  stack top.  Deeper → emit ``_INDENT``.  Same → emit ``_NEWLINE`` as-is.
  Shallower → emit one or more ``_DEDENT`` s and verify alignment.
- ``|``/``else``/``catch``/``until``-continuation rule (§3.4): when the first
  significant token on the next line is ``|``, ``else``, ``catch``, or
  ``until``, the ``_NEWLINE`` is suppressed and only the ``_DEDENT`` s needed to
  pop the stack to levels strictly greater than the keyword's column are
  emitted.  These lines never push an indent.
- At EOF: unwind remaining indent levels with ``_DEDENT`` s.
- Template tokens never produce ``_NEWLINE`` s (the scanner does not emit them
  inside templates).
- Misaligned-dedent diagnostics are positioned at the **first real token** on the
  offending line (i.e. the lookahead ``sig`` token), not at the ``_NEWLINE`` that
  precedes it.  This ensures the reported source line matches the line the user
  actually misindented.
"""

from __future__ import annotations

from typing import Iterator

from lark.lexer import Token

from agm.agl.diagnostics import SourceSpan
from agm.agl.lexer.errors import LexError
from agm.agl.lexer.tokens import (
    DEDENT,
    INDENT,
    INTERP_END,
    INTERP_START,
    KW_CATCH,
    KW_ELSE,
    KW_UNTIL,
    LBRACE,
    LPAR,
    LSQB,
    NEWLINE,
    PIPE,
    RBRACE,
    RPAR,
    RSQB,
    TEMPLATE_END,
    TEMPLATE_START,
)

# Tokens that increase paren depth (newlines suppressed inside these)
_OPEN_BRACKETS = {LPAR, LSQB, LBRACE, INTERP_START, TEMPLATE_START}
# Tokens that decrease paren depth
_CLOSE_BRACKETS = {RPAR, RSQB, RBRACE, INTERP_END, TEMPLATE_END}


def _synthetic(typ: str, value: str, ref: Token) -> Token:
    """Create a synthetic layout token borrowing position from *ref*.

    Position rule (the coherent ruling for layout tokens): synthetic
    ``_NEWLINE``/``_INDENT``/``_DEDENT`` tokens are positioned at the newline
    that introduces the next line — i.e. they borrow the full
    line/column/start_pos/end_pos of the originating ``_NEWLINE`` ``ref`` token
    (which the scanner positions at the ``\\n`` character itself).  At EOF the
    ``ref`` is a synthetic newline anchored at ``len(text)``.
    """
    return Token(
        typ,
        value,
        start_pos=ref.start_pos,
        line=ref.line,
        column=ref.column,
        end_line=ref.end_line,
        end_column=ref.end_column,
        end_pos=ref.end_pos,
    )


def layout(tokens: Iterator[Token]) -> Iterator[Token]:
    """Inject ``_INDENT``/``_DEDENT`` tokens into *tokens*.

    Parameters
    ----------
    tokens:
        Raw token stream from the scanner (contains ``_NEWLINE`` tokens with
        indentation widths as values).

    Yields
    ------
    Token
        The augmented token stream with layout tokens injected.
    """
    indent_stack: list[int] = [0]
    paren_level = 0

    # One-token lookahead buffer for the branch/compound continuation rule.
    buffered: list[Token] = []
    stream = iter(tokens)

    def _next() -> Token | None:
        if buffered:
            return buffered.pop(0)
        try:
            return next(stream)
        except StopIteration:
            return None

    def _peek_next() -> Token | None:
        """Peek at the next token without consuming it."""
        tok = _next()
        if tok is not None:
            buffered.insert(0, tok)
        return tok

    # Reference token for EOF dedents
    last_real: Token | None = None

    while True:
        tok = _next()
        if tok is None:
            break

        # Track bracket/template depth
        if tok.type in _OPEN_BRACKETS:
            paren_level += 1
            last_real = tok
            yield tok
            continue

        if tok.type in _CLOSE_BRACKETS:
            if paren_level > 0:
                paren_level -= 1
            last_real = tok
            yield tok
            continue

        if tok.type != NEWLINE:
            last_real = tok
            yield tok
            continue

        # --- _NEWLINE token ---

        # Suppressed inside brackets/templates
        if paren_level > 0:
            continue

        indent_width = int(str(tok))

        # Peek at the next token to check for the continuation rule.
        sig = _peek_next()

        if sig is not None and sig.type in (PIPE, KW_ELSE, KW_CATCH, KW_UNTIL):
            # Continuation rule: suppress the _NEWLINE and emit only the DEDENTs
            # needed to pop the stack to levels strictly greater than the
            # keyword's column.  These lines never push an indent.
            kw_col = (sig.column or 1) - 1  # convert 1-based column to 0-based

            while len(indent_stack) > 1 and indent_stack[-1] > kw_col:
                indent_stack.pop()
                yield _synthetic(DEDENT, "", tok)

            # Suppress the _NEWLINE — the keyword continues the current construct
            continue

        current_level = indent_stack[-1]

        if indent_width > current_level:
            # Indent: push new level, emit _INDENT
            indent_stack.append(indent_width)
            yield _synthetic(INDENT, "", tok)
        elif indent_width == current_level:
            # Same level: emit _NEWLINE as-is
            yield tok
        else:
            # Dedent: pop levels and emit _DEDENT for each
            while len(indent_stack) > 1 and indent_stack[-1] > indent_width:
                indent_stack.pop()
                yield _synthetic(DEDENT, "", tok)

            if indent_stack[-1] != indent_width:
                # Misaligned dedent — the new indentation is not on the stack.
                # Use the lookahead token (``sig``) to position the diagnostic
                # on the *offending* line rather than the preceding ``_NEWLINE``
                # (which sits on the line before the misaligned content).
                err_ref = sig if sig is not None else tok
                start_off = err_ref.start_pos if err_ref.start_pos is not None else 0
                end_off = err_ref.end_pos if err_ref.end_pos is not None else start_off
                span = SourceSpan(
                    start_line=err_ref.line or 1,
                    start_col=err_ref.column or 1,
                    end_line=err_ref.end_line or err_ref.line or 1,
                    end_col=err_ref.end_column or err_ref.column or 1,
                    start_offset=start_off,
                    end_offset=end_off,
                )
                raise LexError(
                    f"Misaligned dedent: expected indentation {indent_stack[-1]}, "
                    f"got {indent_width}",
                    span=span,
                )
            # Emit _NEWLINE at the restored level
            yield tok

    # EOF: unwind remaining indent levels with _DEDENT tokens.  ``last_real`` is
    # always set here in practice (reaching this loop body requires a pushed
    # indent, which requires a real token); the fallback only guards the
    # impossible empty-stream case and still carries concrete EOF positions.
    ref = (
        last_real
        if last_real is not None
        else Token(
            NEWLINE,
            "0",
            start_pos=0,
            line=1,
            column=1,
            end_line=1,
            end_column=1,
            end_pos=0,
        )
    )
    while len(indent_stack) > 1:
        indent_stack.pop()
        yield _synthetic(DEDENT, "", ref)
