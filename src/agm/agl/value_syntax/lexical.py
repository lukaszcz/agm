"""AgL literal lexical rules: text escapes, numbers, and environment holes.

Pure functions over ``(source, offset)`` shared by the frontend scanner and the
value-syntax reader, so both scan literals identically.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from agm.agl.value_syntax.errors import ValueSyntaxError
from agm.util.ident import IDENT_STOP, is_identifier_start
from agm.util.interp import INTERP_TRIGGER

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

_HEX_DIGITS: Final[str] = "0123456789abcdefABCDEF"


def quote_text(value: str) -> str:
    """Return *value* as a double-quoted AgL text-literal surface form."""
    out: list[str] = ['"']
    for index, character in enumerate(value):
        if character == "$" and value.startswith("${", index):
            out.append("\\$")
            continue
        escaped = ESCAPE_ENCODE.get(character)
        if escaped is not None:
            out.append(escaped)
        elif character < " ":
            out.append(f"\\u{ord(character):04x}")
        else:
            out.append(character)
    out.append('"')
    return "".join(out)


def is_ascii_digit(ch: str) -> bool:
    """Return True iff *ch* is an ASCII digit (``0``-``9``)."""
    return "0" <= ch <= "9"


def decode_escape(source: str, offset: int) -> tuple[str, int]:
    """Decode the backslash escape at *offset* in *source*.

    *offset* indexes the backslash. Returns (decoded text, offset just past the
    escape): the ``\\${`` template escape (decodes to ``"${"``), the escape
    table, or ``\\uXXXX``. Raises :class:`ValueSyntaxError` for end-of-input
    after the backslash, an incomplete or invalid ``\\uXXXX``, or an unknown
    escape.
    """
    if source.startswith("${", offset + 1):
        return "${", offset + 3
    pos = offset + 1
    if pos >= len(source):
        raise ValueSyntaxError("Unexpected end of input after backslash", offset, pos)
    ch = source[pos]
    pos += 1
    if ch in ESCAPE_DECODE:
        return ESCAPE_DECODE[ch], pos
    if ch == "u":
        digits = ""
        for _ in range(4):
            if pos >= len(source):
                raise ValueSyntaxError("Incomplete \\uXXXX escape", offset, pos)
            digit = source[pos]
            pos += 1
            if digit not in _HEX_DIGITS:
                raise ValueSyntaxError(
                    f"Invalid hex digit in \\uXXXX escape: {digit!r}", offset, pos
                )
            digits += digit
        return chr(int(digits, 16)), pos
    raise ValueSyntaxError(f"Unknown escape sequence: \\{ch}", offset, pos)


def scan_number(source: str, offset: int) -> tuple[bool, int]:
    """Scan ``[0-9]+`` or ``[0-9]+.[0-9]+`` (ASCII only) at *offset*.

    *offset* indexes the first digit. Returns (is_decimal, end offset).
    """
    end = offset
    while end < len(source) and is_ascii_digit(source[end]):
        end += 1
    if (
        end < len(source)
        and source[end] == "."
        and end + 1 < len(source)
        and is_ascii_digit(source[end + 1])
    ):
        end += 1
        while end < len(source) and is_ascii_digit(source[end]):
            end += 1
        return True, end
    return False, end


def scan_name(source: str, offset: int) -> int | None:
    """Return the end offset of the identifier starting at *offset*.

    Returns None if *offset* is not an identifier start.
    """
    if offset >= len(source) or not is_identifier_start(source[offset]):
        return None
    end = offset + 1
    while end < len(source) and source[end] not in IDENT_STOP:
        end += 1
    return end


def environment_hole_name(source: str, offset: int) -> str | None:
    """Return the environment name when *offset* starts a ``${NAME}`` hole."""
    if not source.startswith("${", offset):
        return None
    start = offset + 2
    end = scan_name(source, start)
    if end is None or source[end : end + 1] != "}":
        return None
    return source[start:end]


__all__ = [
    "ESCAPE_DECODE",
    "ESCAPE_ENCODE",
    "decode_escape",
    "environment_hole_name",
    "is_ascii_digit",
    "quote_text",
    "scan_name",
    "scan_number",
]
