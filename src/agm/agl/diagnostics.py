"""Diagnostic types for the AgL pipeline.

`Diagnostic` is the user-visible error record: a human-readable message and
a 1-based source location.  `AglError` is the base class for all fatal pipeline
errors raised *before* evaluation begins (lex, parse, scope, typecheck).

``SourceSpan`` is defined in ``agm.agl.syntax.spans`` and re-exported here
for backward compatibility (the lexer and other callers use
``from agm.agl.diagnostics import SourceSpan``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Re-export the canonical definition so existing callers keep working.
from agm.agl.syntax.spans import SourceSpan as SourceSpan


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A single pre-execution diagnostic.

    ``line`` and ``column`` are 1-based.  ``end_line``/``end_column`` are
    optional end-exclusive positions when the diagnostic has a range.
    ``message`` is a human-readable description.
    ``severity`` is ``"error"`` (the default) or ``"warning"``: warnings are
    reported to the user but, unlike errors, do not cause the run to fail (see
    ``RunResult.ok`` and the exit-code contract).
    """

    message: str
    line: int
    column: int | None = None
    end_line: int | None = None
    end_column: int | None = None
    severity: Literal["error", "warning"] = "error"


def diagnostic_from_span(
    message: str,
    span: SourceSpan,
    *,
    severity: Literal["error", "warning"] = "error",
) -> Diagnostic:
    """Build a diagnostic pinned to a concrete source span."""
    return Diagnostic(
        message=message,
        line=span.start_line,
        column=span.start_col,
        end_line=span.end_line,
        end_column=span.end_col,
        severity=severity,
    )


def format_diagnostic_location(diagnostic: Diagnostic) -> str:
    """Return a compact source location for a diagnostic."""
    if diagnostic.column is None:
        return f"line {diagnostic.line}"
    if diagnostic.end_line is None or diagnostic.end_column is None:
        return f"line {diagnostic.line}:{diagnostic.column}"
    if (
        diagnostic.end_line == diagnostic.line
        and diagnostic.end_column > diagnostic.column + 1
    ):
        return f"line {diagnostic.line}:{diagnostic.column}-{diagnostic.end_column - 1}"
    if diagnostic.end_line != diagnostic.line:
        return (
            f"line {diagnostic.line}:{diagnostic.column}-"
            f"{diagnostic.end_line}:{diagnostic.end_column}"
        )
    return f"line {diagnostic.line}:{diagnostic.column}"


def format_diagnostic(diagnostic: Diagnostic) -> str:
    """Return a user-visible diagnostic line without severity prefix."""
    return f"{format_diagnostic_location(diagnostic)}: {diagnostic.message}"


class AglError(Exception):
    """Base class for all fatal AgL pipeline errors."""

    def __init__(self, message: str, *, span: SourceSpan | None = None) -> None:
        super().__init__(message)
        self.span = span

    def to_diagnostic(self) -> Diagnostic:
        if self.span is None:
            return Diagnostic(message=str(self), line=1)
        return diagnostic_from_span(str(self), self.span)
