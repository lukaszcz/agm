"""The shared surface form for AgL text literals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

INTERP_TRIGGER: Final[str] = "%"
INTERP_OPEN: Final[str] = f"{INTERP_TRIGGER}{{"

# Each pair is (escape character, decoded literal character).  The pairs are
# bijective, so the encoder below can be derived from the same declaration.
_ESCAPE_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ('"', '"'),
    ("\\", "\\"),
    ("b", "\b"),
    ("f", "\f"),
    ("n", "\n"),
    ("r", "\r"),
    ("t", "\t"),
    (INTERP_TRIGGER, INTERP_TRIGGER),
)

# AgL accepts these JSON escape spellings but never emits them.
_DECODE_ONLY_ESCAPES: Final[tuple[tuple[str, str], ...]] = (("'", "'"), ("/", "/"))

ESCAPE_DECODE: Final[Mapping[str, str]] = dict((*_ESCAPE_PAIRS, *_DECODE_ONLY_ESCAPES))
ESCAPE_ENCODE: Final[Mapping[str, str]] = {
    literal: f"\\{escaped}" for escaped, literal in _ESCAPE_PAIRS
}


def quote_text(value: str) -> str:
    """Return *value* as a double-quoted AgL text-literal surface form."""
    out: list[str] = ['"']
    for character in value:
        escaped = ESCAPE_ENCODE.get(character)
        if escaped is not None:
            out.append(escaped)
        elif character < " ":
            out.append(f"\\u{ord(character):04x}")
        else:
            out.append(character)
    out.append('"')
    return "".join(out)


__all__ = [
    "ESCAPE_DECODE",
    "ESCAPE_ENCODE",
    "INTERP_OPEN",
    "INTERP_TRIGGER",
    "quote_text",
]
