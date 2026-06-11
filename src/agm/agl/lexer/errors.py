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

    def __init__(self, message: str, *, span: SourceSpan) -> None:
        super().__init__(message, span=span)
