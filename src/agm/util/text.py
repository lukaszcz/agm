"""Shared text helpers for AGM.

This is a small, dependency-free leaf module so that *both* the lexer
(``agm.agl.lexer.scanner``) and lowering/runtime diagnostics
can share the universal-newline normalization without either depending on the
other (the eval pass must never import the lexer — see ``agm/agl/CLAUDE.md``).
The prose helpers are here for the same reason: every help surface that
summarizes authored documentation — the ``agm`` overview, a command group's
listing, ``agm pkg info`` — renders it identically without depending on each
other.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterable

__all__ = [
    "first_paragraph",
    "format_description_column",
    "keep_indented_paragraphs",
    "normalize_newlines",
]

#: Columns of blank left margin every listed entry carries, and the gap
#: separating an entry's name from its description.
_INDENT = 2
_GAP = 2


def normalize_newlines(text: str) -> str:
    """Convert CRLF and lone CR line endings to LF (universal-newline style).

    Every ``\\r\\n`` and every lone ``\\r`` becomes a single ``\\n``.  The
    scanner normalizes its source at entry; span offsets index into this
    normalized text, so the evaluator must normalize identically before slicing
    source by offset.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def first_paragraph(text: str) -> str:
    """Return *text*'s opening paragraph as one line.

    Lines up to the first blank one join with single spaces, so prose wrapped
    across source lines reads as a whole sentence where a listing has room for
    one line only. Returns the empty text when *text* holds no prose.
    """
    paragraph: list[str] = []
    for line in normalize_newlines(text).split("\n"):
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        paragraph.append(stripped)
    return " ".join(paragraph)


#: Click's marker line: the paragraph it opens is printed without rewrapping.
_CLICK_VERBATIM = "\b"


def keep_indented_paragraphs(text: str) -> str:
    """Return authored help *text* for Click, keeping every indented block's layout.

    Click rewraps each paragraph of help text into one flowing block. A
    paragraph holding an indented line — an example, a list, a table — is
    opened with Click's verbatim marker so it prints as written, while prose
    still wraps to the terminal. The text's common indentation is removed first.
    """
    paragraphs: list[list[str]] = [[]]
    for line in textwrap.dedent(normalize_newlines(text)).split("\n"):
        if line.strip():
            paragraphs[-1].append(line)
        elif paragraphs[-1]:
            paragraphs.append([])
    return "\n\n".join(
        "\n".join([_CLICK_VERBATIM, *lines] if any(line[:1].isspace() for line in lines) else lines)
        for lines in paragraphs
        if lines
    )


def format_description_column(entries: Iterable[tuple[str, str]], *, width: int) -> list[str]:
    """Render ``(name, description)`` *entries* as an aligned two-column listing.

    Descriptions start at one column, shared by every entry, and wrap under
    themselves rather than against the left margin. A description is never
    split mid-word, so one narrower than a single word simply overruns
    *width*; an entry without one lists its name alone.
    """
    listing = tuple(entries)
    if not listing:
        return []
    column = _INDENT + max(len(name) for name, _ in listing) + _GAP
    lines: list[str] = []
    for name, description in listing:
        head = f"{' ' * _INDENT}{name.ljust(column - _INDENT - _GAP)}{' ' * _GAP}"
        if not description:
            lines.append(head.rstrip())
            continue
        wrapped = textwrap.wrap(
            description,
            width=max(width - column, 1),
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.append(head + wrapped[0])
        lines.extend(" " * column + line for line in wrapped[1:])
    return lines
