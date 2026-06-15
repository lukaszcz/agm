"""AgL custom lexer package (Component 1).

Public API
----------
- :class:`AglLexer` — Lark ``Lexer`` subclass; wire as
  ``Lark(grammar, parser="lalr", lexer=AglLexer)``.
- :class:`LexError` — span-aware lexical error raised by the scanner /
  layout filter.
- :func:`tokenize` — convenience helper: tokenize a source string and return
  the full token list (useful for tests and diagnostics).
"""

from __future__ import annotations

from typing import Iterator

from lark.lexer import Token

from agm.agl.lexer.errors import LexError
from agm.agl.lexer.layout import layout
from agm.agl.lexer.lexer import AglLexer
from agm.agl.lexer.scanner import lex_tab_warnings, scan

__all__ = ["AglLexer", "LexError", "lex_tab_warnings", "tokenize"]


def tokenize(source: str) -> Iterator[Token]:
    """Tokenize *source* and yield :class:`lark.lexer.Token` objects.

    This is the public convenience entry point for tests and tooling; the
    Lark parser uses :class:`AglLexer` directly.
    """
    return layout(scan(source))
