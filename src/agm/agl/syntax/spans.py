"""Canonical SourceSpan definition for the AgL AST.

This is the single source of truth for ``SourceSpan``.  The diagnostics
module re-exports it so that ``from agm.agl.diagnostics import SourceSpan``
continues to work for callers (including the lexer).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Location of a token or node in the source text.

    Position convention (the single ruling for the whole AgL pipeline):

    - Lines are 1-based (the first line is line 1).
    - Columns are 1-based (the first column is column 1).
    - ``start_offset``/``end_offset`` are 0-based character offsets into the
      *normalized* source string (universal newlines; see ``scanner`` module
      docstring) and are end-exclusive: the span covers
      ``source[start_offset:end_offset]``.

    All four offset/line/column fields are required; there are no sentinel
    defaults.  Synthetic nodes must still supply real positions.
    """

    start_line: int
    start_col: int
    end_line: int
    end_col: int
    start_offset: int
    end_offset: int
