"""Diagnostic types for the AgL pipeline.

`Diagnostic` is the user-visible error record: a human-readable message and
a 1-based source line.  `AglError` is the base class for all fatal pipeline
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

    ``line`` is 1-based.  ``message`` is a human-readable description.
    ``severity`` is ``"error"`` (the default) or ``"warning"``: warnings are
    reported to the user but, unlike errors, do not cause the run to fail (see
    ``RunResult.ok`` and the exit-code contract).
    """

    message: str
    line: int
    severity: Literal["error", "warning"] = "error"


class AglError(Exception):
    """Base class for all fatal AgL pipeline errors."""

    def __init__(self, message: str, *, span: SourceSpan | None = None) -> None:
        super().__init__(message)
        self.span = span

    def to_diagnostic(self) -> Diagnostic:
        line = self.span.start_line if self.span is not None else 1
        return Diagnostic(message=str(self), line=line)
