"""AglSyntaxError: parse-layer error with SourceSpan and friendly message.

This module maps Lark ``UnexpectedInput`` family and lexer ``LexError`` to
``AglSyntaxError`` carrying a ``SourceSpan`` and a user-facing message.

Special case: if the unexpected token is ``EQ_EQ`` (``==``), the message is
``'Use `=` for equality.'`` per design §2.3.
"""

from __future__ import annotations

from agm.agl.diagnostics import AglError
from agm.agl.syntax.spans import SourceSpan

# Token type for the "==" error token (mirrors tokens.EQ_EQ).
_EQ_EQ = "EQ_EQ"


class AglSyntaxError(AglError):
    """A syntax error produced by the parser layer.

    Carries a :class:`~agm.agl.syntax.spans.SourceSpan` pinpointing the
    offending location in the source.  The ``span`` attribute is guaranteed
    non-None (unlike the base ``AglError``).
    """

    def __init__(self, message: str, *, span: SourceSpan) -> None:
        super().__init__(message, span=span)

    @property
    def source_span(self) -> SourceSpan:
        """Always-non-None span; avoids repeated ``assert span is not None``."""
        assert self.span is not None
        return self.span


def _span_from_token(
    token_line: int,
    token_col: int,
    token_pos: int,
    token_end_line: int | None,
    token_end_col: int | None,
    token_end_pos: int | None,
) -> SourceSpan:
    """Build a SourceSpan from Lark Token position fields.

    Falls back to single-character synthetic positions when optional fields
    are absent.
    """
    end_line = token_end_line if token_end_line is not None else token_line
    end_col = token_end_col if token_end_col is not None else token_col + 1
    end_pos = token_end_pos if token_end_pos is not None else token_pos + 1
    return SourceSpan(
        start_line=token_line,
        start_col=token_col,
        end_line=end_line,
        end_col=end_col,
        start_offset=token_pos,
        end_offset=end_pos,
    )


def _make_eq_eq_error(span: SourceSpan) -> AglSyntaxError:
    return AglSyntaxError("Use `=` for equality.", span=span)


def syntax_error_from_lark(
    exc: Exception,
    *,
    filename: str = "<agl>",
) -> AglSyntaxError:
    """Convert a Lark parse exception to ``AglSyntaxError``.

    Handles:
    - ``lark.exceptions.UnexpectedToken`` (token type mismatch)
    - ``lark.exceptions.UnexpectedCharacters`` (lexer-level character error)
    - ``lark.exceptions.UnexpectedEOF`` (premature end-of-file)
    - ``agm.agl.lexer.errors.LexError`` (custom lexer error)
    - Generic fallback for any other exception.
    """
    from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedToken

    from agm.agl.lexer.errors import LexError

    if isinstance(exc, LexError):
        # LexError already carries a SourceSpan.
        assert exc.span is not None
        return AglSyntaxError(str(exc), span=exc.span)

    if isinstance(exc, UnexpectedToken):
        tok = exc.token
        line = tok.line if tok.line is not None else 1
        col = tok.column if tok.column is not None else 1
        pos = tok.start_pos if tok.start_pos is not None else 0
        span = _span_from_token(line, col, pos, tok.end_line, tok.end_column, tok.end_pos)
        if tok.type == _EQ_EQ:
            return _make_eq_eq_error(span)
        return AglSyntaxError(
            f"Unexpected token {tok!r} at line {line}, column {col}.",
            span=span,
        )

    if isinstance(exc, UnexpectedCharacters):
        line = exc.line if exc.line is not None else 1
        col = exc.column if exc.column is not None else 1
        pos = exc.pos_in_stream if exc.pos_in_stream is not None else 0
        span = SourceSpan(
            start_line=line,
            start_col=col,
            end_line=line,
            end_col=col + 1,
            start_offset=pos,
            end_offset=pos + 1,
        )
        return AglSyntaxError(
            f"Unexpected character at line {line}, column {col}.",
            span=span,
        )

    if isinstance(exc, UnexpectedEOF):
        # No position info; use (1, 1) as a fallback.
        span = SourceSpan(
            start_line=1, start_col=1, end_line=1, end_col=1,
            start_offset=0, end_offset=0,
        )
        return AglSyntaxError("Unexpected end of input.", span=span)

    # Generic fallback.
    span = SourceSpan(
        start_line=1, start_col=1, end_line=1, end_col=1,
        start_offset=0, end_offset=0,
    )
    return AglSyntaxError(str(exc), span=span)
