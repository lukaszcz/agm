"""Span-aware lexical error for the AgL lexer."""

from __future__ import annotations

from agm.agl.diagnostics import AglError, SourceSpan


class LexError(AglError):
    """A lexical error with an associated source span.

    Raised by the scanner or layout filter when the input is lexically invalid
    (unknown escape sequence, misaligned dedent, etc.).  The ``span``
    attribute carries 1-based line/column information so the runtime can emit
    a :class:`~agm.agl.diagnostics.Diagnostic` with a precise source location.
    """

    span: SourceSpan

    def __init__(self, message: str, *, span: SourceSpan) -> None:
        super().__init__(message, span=span)


class IncompleteInputError(LexError):
    """Lexing ran out of input where more input could still complete the construct.

    A REPL checks for this type to decide whether to keep prompting instead of
    reporting the error immediately.
    """


class UnterminatedTripleQuotedStringError(IncompleteInputError):
    """Lexing reached end of input inside an unclosed triple-quoted string."""
