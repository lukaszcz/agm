"""Token positions as source spans."""

from __future__ import annotations

from typing import cast

from lark.lexer import Token

from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceId, SourceSpan


def token_span(tok: Token, source: SourceId = UNKNOWN_SOURCE) -> SourceSpan:
    """Return *tok*'s span in *source*.

    Every token the AgL lexer emits, and every token Lark borrows a position
    from one, carries all six position fields; Lark merely types them optional.
    """
    return SourceSpan(
        start_line=cast(int, tok.line),
        start_col=cast(int, tok.column),
        end_line=cast(int, tok.end_line),
        end_col=cast(int, tok.end_column),
        start_offset=cast(int, tok.start_pos),
        end_offset=cast(int, tok.end_pos),
        source=source,
    )
