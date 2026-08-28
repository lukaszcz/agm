"""AgL identifier grammar shared by lexing and interpolation."""

from __future__ import annotations

from typing import Final

# Characters that terminate an identifier scan. An identifier starts with a
# (Unicode) letter or ``_`` and then greedily consumes every character that is
# not in this set. The stop set retains structural punctuators and operators as
# standalone delimiters, while allowing names such as ``ask-prompt``, ``ask?``,
# ``do-it-now!``, ``exec$``, ``a+b``, and ``foo\"bar``.
IDENT_STOP: Final[frozenset[str]] = frozenset(
    {" ", "\t", "\n", "\r", "(", ")", "[", "]", "{", "}", ":", ",", ".", "|", ";", "/", "@", "="}
)


def is_identifier_start(character: str) -> bool:
    """Return whether *character* may start an AgL identifier."""
    return character.isalpha() or character == "_"


def is_identifier(name: str) -> bool:
    """Return whether *name* is a complete AgL identifier."""
    return (
        bool(name)
        and is_identifier_start(name[0])
        and not any(character in IDENT_STOP for character in name)
    )
