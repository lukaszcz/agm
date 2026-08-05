"""Shared runtime ``%{name}`` string interpolation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final, TypeAlias

from agm.util.ident import is_identifier

__all__ = [
    "INTERP_OPEN",
    "INTERP_TRIGGER",
    "Hole",
    "InterpolationError",
    "Literal",
    "Segment",
    "interp",
    "interp_preserving",
    "interp_segments",
    "split_template",
]

INTERP_TRIGGER: Final[str] = "%"
INTERP_OPEN: Final[str] = f"{INTERP_TRIGGER}{{"


@dataclass(frozen=True, slots=True)
class Literal:
    """A verbatim portion of an interpolation template."""

    text: str


@dataclass(frozen=True, slots=True)
class Hole:
    """A named interpolation hole and its trigger offset in the template."""

    name: str
    offset: int = field(default=0, compare=False)


Segment: TypeAlias = Literal | Hole


class InterpolationError(ValueError):
    """A malformed interpolation template or unavailable interpolation value."""

    kind: str
    text: str
    offset: int
    context: str | None

    def __init__(self, kind: str, text: str, offset: int) -> None:
        self.kind = kind
        self.text = text
        self.offset = offset
        # Callers that interpolate several templates in a row set *context* to
        # say which one failed; the offset alone is ambiguous across them.
        self.context = None
        super().__init__(f"{kind} {text!r} at offset {offset}")

    def __str__(self) -> str:
        message = super().__str__()
        return f"{message} ({self.context})" if self.context else message


def _scan(text: str, *, lenient: bool) -> tuple[list[Segment], bool]:
    """Split *text* into segments, and report whether any hole was malformed.

    In lenient mode malformed and unterminated holes remain literal text and
    set the flag; in strict mode they raise instead, so the flag is always
    ``False``.
    """
    # Every construct the scan recognises — a ``%{`` hole and its ``\%{``
    # escape — contains the trigger, so text without one is wholly literal.
    if INTERP_TRIGGER not in text:
        return ([Literal(text)] if text else []), False

    malformed_hole = False
    segments: list[Segment] = []
    literal: list[str] = []
    position = 0

    def add_literal(part: str) -> None:
        if part:
            literal.append(part)

    def flush_literal() -> None:
        if literal:
            segments.append(Literal("".join(literal)))
            literal.clear()

    while position < len(text):
        # Jump straight to the next trigger; everything before it is literal.
        trigger = text.find(INTERP_TRIGGER, position)
        if trigger == -1:
            add_literal(text[position:])
            break
        if not text.startswith(INTERP_OPEN, trigger):
            add_literal(text[position : trigger + 1])
            position = trigger + 1
        elif trigger > position and text[trigger - 1] == "\\":
            # The backslash is still unconsumed, so this ``%{`` is escaped.
            add_literal(text[position : trigger - 1])
            add_literal(INTERP_OPEN)
            position = trigger + len(INTERP_OPEN)
        else:
            end = text.find("}", trigger + len(INTERP_OPEN))
            if end == -1:
                if not lenient:
                    raise InterpolationError("unterminated hole", text[trigger:], trigger)
                malformed_hole = True
                add_literal(text[position:])
                break
            name = text[trigger + len(INTERP_OPEN) : end]
            if is_identifier(name):
                add_literal(text[position:trigger])
                flush_literal()
                segments.append(Hole(name, trigger))
            elif not lenient:
                raise InterpolationError("invalid hole name", name, trigger)
            else:
                malformed_hole = True
                add_literal(text[position : end + 1])
            position = end + 1

    flush_literal()
    return segments, malformed_hole


def split_template(text: str, *, lenient: bool = False) -> list[Segment]:
    """Split *text* into literal and named-hole segments.

    In lenient mode malformed and unterminated holes remain literal text.
    """
    return _scan(text, lenient=lenient)[0]


def interp_segments(
    segments: Iterable[Segment],
    variables: Mapping[str, str],
    *,
    lenient: bool = False,
) -> str:
    """Render *segments*, resolving each named hole from *variables*.

    A hole with no matching variable raises, or is preserved verbatim in
    lenient mode.
    """
    parts: list[str] = []
    for segment in segments:
        if isinstance(segment, Literal):
            parts.append(segment.text)
        elif segment.name in variables:
            parts.append(variables[segment.name])
        elif lenient:
            parts.append(f"{INTERP_OPEN}{segment.name}}}")
        else:
            raise InterpolationError("missing variable", segment.name, segment.offset)
    return "".join(parts)


def interp(template: str, variables: Mapping[str, str]) -> str:
    """Strictly interpolate named holes in *template* from *variables*."""
    return interp_segments(split_template(template), variables)


def interp_preserving(template: str, variables: Mapping[str, str]) -> tuple[str, bool]:
    """Interpolate leniently, reporting whether anything stayed unresolved.

    The flag is set when a hole names an unavailable variable or is malformed —
    that is, whenever the result still carries interpolation syntax that this
    mapping could not fill in.
    """
    segments, malformed_hole = _scan(template, lenient=True)
    unresolved = malformed_hole or any(
        isinstance(segment, Hole) and segment.name not in variables for segment in segments
    )
    return interp_segments(segments, variables, lenient=True), unresolved
