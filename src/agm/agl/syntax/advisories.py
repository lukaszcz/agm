"""Lexical advisories: facts the lexer observes but the grammar cannot express.

A qualifier such as ``app/config::x`` is recognized only when every part of the
run is byte-adjacent in the source.  Whitespace before the ``::`` silently makes
the run parse as something else entirely, and by the time the AST exists the
whitespace is gone.  The lexer sees it directly, so it records it here for later
passes to turn into a diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

from agm.agl.syntax.spans import SourceSpan


@dataclass(frozen=True, slots=True)
class SpacedQualifier:
    """One ``[/] name (/ name)*`` run separated from a following ``::`` by whitespace.

    Attributes
    ----------
    segments:
        The adjacent name run preceding the ``::`` (e.g. ``("app", "config")``).
    anchored:
        Whether the run began with a leading ``/`` adjacent to its first name.
    run_start_offset:
        Source offset where the run begins, including any leading ``/``.  The
        half-open range ``[run_start_offset, dcolon_offset)`` covers exactly the
        text that would have become a qualifier.
    dcolon_span:
        Span of the ``::`` that failed to join the run.
    member:
        The name immediately after the ``::`` — the member the run would have to
        contribute for the tight spelling to mean anything.
    member_text:
        Source spelling of the whole reference following the ``::``
        (e.g. ``"E[int]::X"``).
    type_qualified:
        Whether the reference reads as ``Type::Ctor``, making ``member`` the name
        of a type rather than of the referenced binding itself.
    """

    segments: tuple[str, ...]
    anchored: bool
    run_start_offset: int
    dcolon_span: SourceSpan
    member: str
    member_text: str
    type_qualified: bool

    @property
    def dcolon_offset(self) -> int:
        """Source offset of the ``::`` that whitespace kept out of the run."""
        return self.dcolon_span.start_offset

    def covers(self, offset: int) -> bool:
        """Return whether *offset* falls inside the run that precedes the ``::``."""
        return self.run_start_offset <= offset < self.dcolon_offset
