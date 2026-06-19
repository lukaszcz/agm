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
from agm.agl.syntax.spans import UNKNOWN_SOURCE as UNKNOWN_SOURCE
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
    ``source_label`` is the display label of the source file this diagnostic
    originated from (e.g. a canonical file path, ``"<agl>"``, ``"<repl>"``).
    When set, it takes precedence over the ambient ``source_name`` argument
    passed to :func:`format_diagnostic_location` / :func:`format_diagnostic`.
    Populated automatically by :func:`diagnostic_from_span` from
    ``span.source.label``.
    """

    message: str
    line: int
    column: int | None = None
    end_line: int | None = None
    end_column: int | None = None
    severity: Literal["error", "warning"] = "error"
    source_label: str | None = None


def diagnostic_from_span(
    message: str,
    span: SourceSpan,
    *,
    severity: Literal["error", "warning"] = "error",
) -> Diagnostic:
    """Build a diagnostic pinned to a concrete source span.

    When ``span.source`` is a real (non-default) ``SourceId``, the diagnostic's
    ``source_label`` is populated from ``span.source.label`` so that multi-file
    diagnostics identify their origin file.  When the span carries the default
    ``UNKNOWN_SOURCE``, ``source_label`` is left as ``None`` so that callers
    can still supply a ``source_name`` argument to the formatting functions —
    preserving full backward compatibility for single-source callers.
    """
    source_label: str | None = (
        span.source.label if span.source is not UNKNOWN_SOURCE else None
    )
    return Diagnostic(
        message=message,
        line=span.start_line,
        column=span.start_col,
        end_line=span.end_line,
        end_column=span.end_col,
        severity=severity,
        source_label=source_label,
    )


def format_diagnostic_location(
    diagnostic: Diagnostic, *, source_name: str | None = "<agl>"
) -> str:
    """Return a compiler-style source location for a diagnostic.

    The effective source identifier is determined in priority order:
    1. ``diagnostic.source_label`` (set by :func:`diagnostic_from_span` from
       the span's ``SourceId``) — takes precedence when set.
    2. The ``source_name`` argument — used when ``source_label`` is ``None``.
    """
    effective_source: str | None = (
        diagnostic.source_label if diagnostic.source_label is not None else source_name
    )
    prefix = f"{effective_source}:" if effective_source is not None else ""
    if diagnostic.column is None:
        return f"{prefix}{diagnostic.line}"
    if diagnostic.end_line is None or diagnostic.end_column is None:
        return f"{prefix}{diagnostic.line}:{diagnostic.column}"
    if (
        diagnostic.end_line == diagnostic.line
        and diagnostic.end_column > diagnostic.column + 1
    ):
        return (
            f"{prefix}{diagnostic.line}:"
            f"{diagnostic.column}-{diagnostic.end_column - 1}"
        )
    if diagnostic.end_line != diagnostic.line:
        return (
            f"{prefix}{diagnostic.line}:{diagnostic.column}-"
            f"{diagnostic.end_line}:{diagnostic.end_column}"
        )
    return f"{prefix}{diagnostic.line}:{diagnostic.column}"


def format_diagnostic(
    diagnostic: Diagnostic, *, source_name: str | None = "<agl>"
) -> str:
    """Return a user-visible diagnostic line with source and severity."""
    return (
        f"{format_diagnostic_location(diagnostic, source_name=source_name)}: "
        f"{diagnostic.severity}: {diagnostic.message}"
    )


class AglError(Exception):
    """Base class for all fatal AgL pipeline errors."""

    def __init__(self, message: str, *, span: SourceSpan | None = None) -> None:
        super().__init__(message)
        self.span = span

    def to_diagnostic(self) -> Diagnostic:
        if self.span is None:
            return Diagnostic(message=str(self), line=1)
        return diagnostic_from_span(str(self), self.span)
