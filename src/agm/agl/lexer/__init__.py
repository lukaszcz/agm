"""AgL custom lexer package.

Public API
----------
- :class:`AglLexer` — Lark ``Lexer`` subclass; wire as
  ``Lark(grammar, parser="lalr", lexer=AglLexer)``.
- :class:`LexError` — span-aware lexical error raised by the scanner /
  layout filter.
- :class:`IncompleteInputError` — a :class:`LexError` subtype raised only when
  more input could complete the construct; :class:`UnterminatedTripleQuotedStringError`
  narrows it to an unclosed triple-quoted string.
- :func:`tokenize` — convenience helper: tokenize a source string and return
  the full token list (useful for tests and diagnostics).
- :func:`lex_comment_spans` — offsets of the ``#`` comments the scan skipped
  (comments carry no token, so highlighters read them from here).
- :class:`SpacedQualifier` / :func:`spaced_qualifier_collector` — lexical
  advisories for qualifier runs broken by whitespace before ``::``.
"""

from __future__ import annotations

from typing import Iterator

from lark.lexer import Token

from agm.agl.lexer.errors import IncompleteInputError, LexError, UnterminatedTripleQuotedStringError
from agm.agl.lexer.layout import layout
from agm.agl.lexer.lexer import (
    AglLexer,
    apply_module_passes,
    spaced_qualifier_collector,
    tab_warning_collector,
    unclosed_scope_path,
)
from agm.agl.lexer.scanner import lex_comment_spans, lex_tab_warnings, scan
from agm.agl.syntax.advisories import SpacedQualifier

__all__ = [
    "AglLexer",
    "IncompleteInputError",
    "LexError",
    "SpacedQualifier",
    "UnterminatedTripleQuotedStringError",
    "lex_comment_spans",
    "lex_tab_warnings",
    "spaced_qualifier_collector",
    "tab_warning_collector",
    "unclosed_scope_path",
    "tokenize",
]


def tokenize(source: str) -> Iterator[Token]:
    """Tokenize *source* and yield :class:`lark.lexer.Token` objects.

    This is the public convenience entry point for tests and tooling; the
    Lark parser uses :class:`AglLexer` directly.
    """
    return iter(apply_module_passes(list(layout(scan(source))), source))
