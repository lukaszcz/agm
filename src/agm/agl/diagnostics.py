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
from pathlib import Path
from typing import Literal, Sequence

# Re-export the canonical definition so existing callers keep working.
from agm.agl.syntax.spans import UNKNOWN_SOURCE as UNKNOWN_SOURCE
from agm.agl.syntax.spans import SourceSpan as SourceSpan
from agm.core.path import display_path


@dataclass(frozen=True, slots=True)
class RelatedDiagnostic:
    """A non-recursive source location that explains a primary diagnostic.

    Related diagnostics are always rendered as ``note:`` entries and cannot
    themselves have related locations. This keeps a diagnostic's provenance
    deterministic and bounded while still allowing a primary error to point at
    every relevant source constraint.
    """

    message: str
    line: int
    column: int | None = None
    end_line: int | None = None
    end_column: int | None = None
    source_label: str | None = None


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
    related: tuple[RelatedDiagnostic, ...] = ()


def _source_label_from_span(span: SourceSpan) -> str | None:
    """Return the display label encoded in *span*, if it has one."""
    return span.source.label if span.source is not UNKNOWN_SOURCE else None


def related_diagnostic_from_span(message: str, span: SourceSpan) -> RelatedDiagnostic:
    """Build a source-aware related note from a semantic message/span pair."""
    return RelatedDiagnostic(
        message=message,
        line=span.start_line,
        column=span.start_col,
        end_line=span.end_line,
        end_column=span.end_col,
        source_label=_source_label_from_span(span),
    )


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
    return Diagnostic(
        message=message,
        line=span.start_line,
        column=span.start_col,
        end_line=span.end_line,
        end_column=span.end_col,
        severity=severity,
        source_label=_source_label_from_span(span),
    )


def _format_diagnostic_location(
    *,
    line: int,
    column: int | None,
    end_line: int | None,
    end_column: int | None,
    source_label: str | None,
    source_name: str | None,
) -> str:
    """Format fields shared by primary diagnostics and related notes.

    A source label supplied by a span takes precedence over the ambient
    ``source_name``; the latter is used for source-less diagnostics.
    """
    effective_source: str | None = source_label if source_label is not None else source_name
    prefix = (
        f"{display_path(Path(effective_source))}:" if effective_source is not None else ""
    )
    if column is None:
        return f"{prefix}{line}"
    if end_line is None or end_column is None:
        return f"{prefix}{line}:{column}"
    if end_line == line and end_column > column + 1:
        return f"{prefix}{line}:{column}-{end_column - 1}"
    if end_line != line:
        return f"{prefix}{line}:{column}-{end_line}:{end_column}"
    return f"{prefix}{line}:{column}"


def format_diagnostic_location(
    diagnostic: Diagnostic | RelatedDiagnostic, *, source_name: str | None = "<agl>"
) -> str:
    """Return a compiler-style source location for a diagnostic or related note."""
    return _format_diagnostic_location(
        line=diagnostic.line,
        column=diagnostic.column,
        end_line=diagnostic.end_line,
        end_column=diagnostic.end_column,
        source_label=diagnostic.source_label,
        source_name=source_name,
    )


def format_diagnostic(
    diagnostic: Diagnostic, *, source_name: str | None = "<agl>"
) -> str:
    """Return a user-visible primary diagnostic followed by its related notes."""
    primary = (
        f"{format_diagnostic_location(diagnostic, source_name=source_name)}: "
        f"{diagnostic.severity}: {diagnostic.message}"
    )
    notes = (
        f"  {format_diagnostic_location(note, source_name=source_name)}: note: {note.message}"
        for note in diagnostic.related
    )
    return "\n".join((primary, *notes))


class AglError(Exception):
    """Base class for all fatal AgL pipeline errors.

    ``related`` carries semantic ``(message, SourceSpan)`` pairs until the
    error is converted into a source-aware :class:`Diagnostic`.
    """

    def __init__(
        self,
        message: str,
        *,
        span: SourceSpan | None = None,
        related: Sequence[tuple[str, SourceSpan]] = (),
    ) -> None:
        super().__init__(message)
        self.span = span
        self.related = tuple(related)

    def to_diagnostic(self) -> Diagnostic:
        related = tuple(
            related_diagnostic_from_span(message, span) for message, span in self.related
        )
        if self.span is None:
            return Diagnostic(message=str(self), line=1, related=related)
        primary = diagnostic_from_span(str(self), self.span)
        return Diagnostic(
            message=primary.message,
            line=primary.line,
            column=primary.column,
            end_line=primary.end_line,
            end_column=primary.end_column,
            severity=primary.severity,
            source_label=primary.source_label,
            related=related,
        )
