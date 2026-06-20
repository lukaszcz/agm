"""Canonical SourceSpan definition for the AgL AST.

This is the single source of truth for ``SourceSpan``.  The diagnostics
module re-exports it so that ``from agm.agl.diagnostics import SourceSpan``
continues to work for callers (including the lexer).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field


@dataclass(frozen=True, slots=True)
class SourceId:
    """Identity of a source text (file, REPL entry, inline string, etc.).

    ``label`` is the display string used in diagnostics:
    - canonical file path for file-based modules (``"/path/to/foo.agl"``)
    - ``"<command>"`` for ``exec -c`` inline source
    - ``"<repl>"`` for REPL entries
    - ``"<agl>"`` (the default / unknown source)
    """

    label: str


#: Sentinel used when no source identity is known (the default).
UNKNOWN_SOURCE: SourceId = SourceId("<agl>")


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

    ``source`` identifies which source file/text this span belongs to.  It is
    excluded from equality and hashing (``compare=False``) so that structurally
    identical spans from different source files still compare equal — consistent
    with how ``span`` and ``node_id`` are handled in AST nodes.  Defaults to
    ``UNKNOWN_SOURCE`` so that the hundreds of existing synthetic-span sites
    require no changes.
    """

    start_line: int
    start_col: int
    end_line: int
    end_col: int
    start_offset: int
    end_offset: int
    source: SourceId = dc_field(default=UNKNOWN_SOURCE, compare=False)
