"""Shared runtime ``%{name}`` string interpolation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final, TypeAlias

__all__ = [
    "IDENT_STOP",
    "INTERP_OPEN",
    "INTERP_TRIGGER",
    "Hole",
    "InterpolationError",
    "Literal",
    "Segment",
    "assemble",
    "interp",
    "interp_lenient",
    "is_identifier_start",
    "is_interp_name",
    "split_template",
]

INTERP_TRIGGER: Final[str] = "%"
INTERP_OPEN: Final[str] = f"{INTERP_TRIGGER}{{"

# Characters that terminate an identifier scan. An identifier starts with a
# (Unicode) letter or ``_`` and then greedily consumes every character that is
# not in this set. The stop set retains structural punctuators and operators as
# standalone delimiters, while allowing names such as ``ask-prompt``, ``ask?``,
# ``do-it-now!``, ``a+b``, and ``foo\"bar``.
IDENT_STOP: Final[frozenset[str]] = frozenset(
    {
        " ",
        "\t",
        "\n",
        "\r",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        ":",
        ",",
        ".",
        "|",
        ";",
        "/",
        "@",
        "=",
    }
)


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

    def __init__(self, kind: str, text: str, offset: int) -> None:
        self.kind = kind
        self.text = text
        self.offset = offset
        super().__init__(f"{kind} {text!r} at offset {offset}")


def is_identifier_start(character: str) -> bool:
    """Return whether *character* may start an AgL identifier."""
    return character.isalpha() or character == "_"


def is_interp_name(name: str) -> bool:
    """Return whether *name* is a complete AgL identifier."""
    return (
        bool(name)
        and is_identifier_start(name[0])
        and not any(character in IDENT_STOP for character in name)
    )


def split_template(text: str, *, lenient: bool = False) -> list[Segment]:
    """Split *text* into literal and named-hole segments.

    In lenient mode malformed and unterminated holes remain literal text.
    """
    segments: list[Segment] = []
    literal: list[str] = []
    position = 0

    def flush_literal() -> None:
        if literal:
            segments.append(Literal("".join(literal)))
            literal.clear()

    while position < len(text):
        if text.startswith(f"\\{INTERP_OPEN}", position):
            literal.append(INTERP_OPEN)
            position += len(INTERP_OPEN) + 1
        elif text.startswith(INTERP_OPEN, position):
            end = text.find("}", position + len(INTERP_OPEN))
            if end == -1:
                malformed = text[position:]
                if not lenient:
                    raise InterpolationError("unterminated hole", malformed, position)
                literal.append(malformed)
                position = len(text)
            else:
                name = text[position + len(INTERP_OPEN) : end]
                if is_interp_name(name):
                    flush_literal()
                    segments.append(Hole(name, position))
                elif not lenient:
                    raise InterpolationError("invalid hole name", name, position)
                else:
                    literal.append(text[position : end + 1])
                position = end + 1
        else:
            literal.append(text[position])
            position += 1

    flush_literal()
    return segments


def assemble(segments: Iterable[Segment], resolve: Callable[[str], str]) -> str:
    """Assemble *segments*, resolving each named hole in order."""
    parts: list[str] = []
    for segment in segments:
        if isinstance(segment, Literal):
            parts.append(segment.text)
        else:
            parts.append(resolve(segment.name))
    return "".join(parts)


def interp(template: str, variables: Mapping[str, str]) -> str:
    """Strictly interpolate named holes in *template* from *variables*."""
    segments = split_template(template)
    offsets = iter(segment.offset for segment in segments if isinstance(segment, Hole))

    def resolve(name: str) -> str:
        offset = next(offsets)
        if name in variables:
            return variables[name]
        raise InterpolationError("missing variable", name, offset)

    return assemble(segments, resolve)


def interp_lenient(template: str, variables: Mapping[str, str]) -> str:
    """Interpolate known named holes and preserve all other input verbatim."""

    def resolve(name: str) -> str:
        if name in variables:
            return variables[name]
        return f"{INTERP_OPEN}{name}}}"

    return assemble(split_template(template, lenient=True), resolve)
