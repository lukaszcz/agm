"""Recursive-descent reader for AgL value syntax.

Grammar (whitespace, including newlines, is insignificant between tokens)::

    value ::= INT | DECIMAL | "-" (INT | DECIMAL) | "true" | "false" | "null" | text
            | "[" [value ("," value)* [","]] "]"
            | "{" [key ":" value ("," key ":" value)* [","]] "}"
            | ctor ["(" [arg ("," arg)* [","]] ")"]
    key   ::= text | NAME
    ctor  ::= NAME | NAME "::" NAME
    arg   ::= NAME "=" value | value
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Final

from agm.agl.keywords import is_plain_name
from agm.agl.value_syntax import lexical
from agm.agl.value_syntax.errors import ValueSyntaxError
from agm.agl.value_syntax.nodes import (
    ArrayNode,
    BoolNode,
    CtorNode,
    DecimalNode,
    DictEntry,
    DictNode,
    IntNode,
    NullNode,
    TextNode,
    ValueArg,
    ValueNode,
)
from agm.util.interp import INTERP_OPEN

__all__ = ["read_ctor_head", "read_value"]

# The scanner's own whitespace set (space, TAB, and the two newline forms it
# normalizes away); not `str.isspace()`, which admits other Unicode space
# characters that are errors here, as everywhere else in AgL source.
_WHITESPACE: Final[frozenset[str]] = frozenset(" \t\n\r")


def read_value(source: str) -> ValueNode:
    """Parse *source* as one AgL value-syntax literal.

    Surrounding whitespace (including newlines) is allowed; any leftover
    content after the value is a :class:`ValueSyntaxError`.
    """
    try:
        pos = _skip_ws(source, 0)
        value, pos = _parse_value(source, pos)
        pos = _skip_ws(source, pos)
    except RecursionError as exc:
        raise ValueSyntaxError("value is nested too deeply", 0, len(source)) from exc
    if pos != len(source):
        raise ValueSyntaxError("unexpected trailing input", pos, len(source))
    return value


def _skip_ws(source: str, pos: int) -> int:
    while pos < len(source) and source[pos] in _WHITESPACE:
        pos += 1
    return pos


def _parse_value(source: str, pos: int) -> tuple[ValueNode, int]:
    pos = _skip_ws(source, pos)
    if pos >= len(source):
        raise ValueSyntaxError("unexpected end of input", pos, pos)
    ch = source[pos]
    if lexical.is_ascii_digit(ch):
        return _parse_number(source, pos)
    if ch == "-":
        return _parse_negative(source, pos)
    if ch in "\"'":
        return _parse_text(source, pos)
    if ch == "[":
        return _parse_array(source, pos)
    if ch == "{":
        return _parse_dict(source, pos)
    end = lexical.scan_name(source, pos)
    if end is None:
        raise ValueSyntaxError(f"unexpected character {ch!r}", pos, pos + 1)
    word = source[pos:end]
    if word == "true":
        return BoolNode(True, pos, end), end
    if word == "false":
        return BoolNode(False, pos, end), end
    if word == "null":
        return NullNode(pos, end), end
    return _parse_ctor(source, pos, word, end)


def _parse_number(source: str, pos: int) -> tuple[ValueNode, int]:
    is_decimal, end = lexical.scan_number(source, pos)
    text = source[pos:end]
    if is_decimal:
        return DecimalNode(Decimal(text), pos, end), end
    return IntNode(int(text), pos, end), end


def _parse_negative(source: str, pos: int) -> tuple[ValueNode, int]:
    digit_pos = _skip_ws(source, pos + 1)
    if digit_pos >= len(source) or not lexical.is_ascii_digit(source[digit_pos]):
        raise ValueSyntaxError("expected a digit after '-'", pos, digit_pos)
    is_decimal, end = lexical.scan_number(source, digit_pos)
    text = "-" + source[digit_pos:end]
    if is_decimal:
        return DecimalNode(Decimal(text), pos, end), end
    return IntNode(int(text), pos, end), end


def _read_quoted_text(source: str, pos: int) -> tuple[str, int]:
    """Consume a quoted text literal at *pos* (the opening quote); return (text, end)."""
    quote = source[pos]
    end = pos + 1
    parts: list[str] = []
    while True:
        if end >= len(source):
            raise ValueSyntaxError("unterminated text literal", pos, end)
        ch = source[end]
        if ch == quote:
            return "".join(parts), end + 1
        if ch == "\n" or ch == "\r":
            raise ValueSyntaxError("newline is not allowed in a value text literal", pos, end)
        if ch == "\\":
            decoded, end = lexical.decode_escape(source, end)
            parts.append(decoded)
            continue
        if source.startswith(INTERP_OPEN, end):
            raise ValueSyntaxError(
                "interpolation is not allowed in values", end, end + len(INTERP_OPEN)
            )
        hole = lexical.environment_hole_name(source, end)
        if hole is not None:
            raise ValueSyntaxError(
                "interpolation is not allowed in values", end, end + len(hole) + 3
            )
        parts.append(ch)
        end += 1


def _parse_text(source: str, pos: int) -> tuple[ValueNode, int]:
    text, end = _read_quoted_text(source, pos)
    return TextNode(text, pos, end), end


def _parse_delimited[T](
    source: str,
    open_pos: int,
    close: str,
    parse_item: Callable[[str, int], tuple[T, int]],
    what: str,
) -> tuple[tuple[T, ...], int]:
    """Parse a *close*-delimited, comma-separated list starting at *open_pos*.

    *open_pos* indexes the opening delimiter. Trailing commas are allowed.
    Returns (items, end just past *close*); raises "unterminated {what}" (with
    offsets within ``[0, len(source)]``) if the input ends first.
    """
    scan_pos = _skip_ws(source, open_pos + 1)
    items: list[T] = []
    if scan_pos < len(source) and source[scan_pos] == close:
        return tuple(items), scan_pos + 1
    while True:
        if scan_pos >= len(source):
            raise ValueSyntaxError(f"unterminated {what}", open_pos, scan_pos)
        item, scan_pos = parse_item(source, scan_pos)
        items.append(item)
        scan_pos = _skip_ws(source, scan_pos)
        if scan_pos >= len(source):
            raise ValueSyntaxError(f"unterminated {what}", open_pos, scan_pos)
        if source[scan_pos] == ",":
            scan_pos = _skip_ws(source, scan_pos + 1)
            if scan_pos < len(source) and source[scan_pos] == close:
                return tuple(items), scan_pos + 1
            continue
        if source[scan_pos] == close:
            return tuple(items), scan_pos + 1
        raise ValueSyntaxError(f"expected ',' or {close!r}", scan_pos, scan_pos + 1)


def _parse_array(source: str, pos: int) -> tuple[ValueNode, int]:
    items, end = _parse_delimited(source, pos, "]", _parse_value, "array")
    return ArrayNode(items, pos, end), end


def _parse_dict(source: str, pos: int) -> tuple[ValueNode, int]:
    entries, end = _parse_delimited(source, pos, "}", _parse_dict_entry, "dict")
    return DictNode(entries, pos, end), end


def _parse_dict_entry(source: str, pos: int) -> tuple[DictEntry, int]:
    entry_start = pos
    if pos < len(source) and source[pos] in "\"'":
        key, scan_pos = _read_quoted_text(source, pos)
    else:
        name_end = lexical.scan_name(source, pos)
        if name_end is None:
            raise ValueSyntaxError("expected a dict key", pos, pos + 1)
        scan_pos = name_end
        key = source[pos:scan_pos]
        if not is_plain_name(key):
            raise ValueSyntaxError(f"{key!r} is not a valid dict key", pos, scan_pos)
    scan_pos = _skip_ws(source, scan_pos)
    if scan_pos >= len(source) or source[scan_pos] != ":":
        end = min(scan_pos + 1, len(source))
        raise ValueSyntaxError("expected ':' after dict key", scan_pos, end)
    scan_pos = _skip_ws(source, scan_pos + 1)
    value, scan_pos = _parse_value(source, scan_pos)
    return DictEntry(key, value, entry_start, scan_pos), scan_pos


def _read_ctor_qualifier(
    source: str, start: int, word: str, pos: int
) -> tuple[str | None, str, int]:
    """Read an optional ``word :: NAME`` qualifier chain following *word*.

    *word* is the already-scanned leading name, ending at *pos*. Returns
    ``(qualifier, name, end)`` -- *qualifier* is ``None`` and *name* is
    *word* itself when no ``::`` follows, with whitespace (including
    newlines) allowed around it exactly like everywhere else in this
    grammar. Raises :class:`ValueSyntaxError` for an invalid constructor
    name, on either side of ``::``.
    """
    if not is_plain_name(word):
        raise ValueSyntaxError(f"{word!r} is not a valid constructor name", start, pos)
    dcolon_pos = _skip_ws(source, pos)
    if not source.startswith("::", dcolon_pos):
        return None, word, pos
    member_start = _skip_ws(source, dcolon_pos + 2)
    member_end = lexical.scan_name(source, member_start)
    if member_end is None:
        raise ValueSyntaxError("expected a name after '::'", member_start, member_start)
    member_word = source[member_start:member_end]
    if not is_plain_name(member_word):
        raise ValueSyntaxError(
            f"{member_word!r} is not a valid constructor name", member_start, member_end
        )
    return word, member_word, member_end


def _parse_ctor(source: str, start: int, word: str, pos: int) -> tuple[ValueNode, int]:
    qualifier, name, pos = _read_ctor_qualifier(source, start, word, pos)
    peek = _skip_ws(source, pos)
    args: tuple[ValueArg, ...] | None = None
    end = pos
    if peek < len(source) and source[peek] == "(":
        args, end = _parse_args(source, peek)
    return CtorNode(qualifier, name, args, start, end), end


def read_ctor_head(source: str) -> tuple[str | None, str] | None:
    """Return ``(qualifier, name)`` when *source* opens with a constructor call.

    Skips leading whitespace, then reads exactly the qualifier/name prefix
    :func:`_parse_ctor` reads -- the same whitespace handling around ``::``
    and before ``(`` -- so a caller probing for an attempted call (the host
    Agent-text dispatch) can never diverge from how the reader itself treats
    the same text. Returns ``None`` for anything that is not a name opening
    a call: a bare name with no ``(``, a malformed qualifier chain, or text
    that is not a name at all.
    """
    pos = _skip_ws(source, 0)
    end = lexical.scan_name(source, pos)
    if end is None:
        return None
    word = source[pos:end]
    try:
        qualifier, name, after = _read_ctor_qualifier(source, pos, word, end)
    except ValueSyntaxError:
        return None
    peek = _skip_ws(source, after)
    if peek >= len(source) or source[peek] != "(":
        return None
    return qualifier, name


def _parse_args(source: str, pos: int) -> tuple[tuple[ValueArg, ...], int]:
    return _parse_delimited(source, pos, ")", _parse_arg, "argument list")


def _parse_arg(source: str, pos: int) -> tuple[ValueArg, int]:
    start = pos
    after_name = lexical.scan_name(source, pos)
    if after_name is not None:
        word = source[pos:after_name]
        ws = _skip_ws(source, after_name)
        if ws < len(source) and source[ws] == "=" and not source.startswith("==", ws):
            if not is_plain_name(word):
                raise ValueSyntaxError(f"{word!r} is not a valid argument name", start, after_name)
            value, end = _parse_value(source, _skip_ws(source, ws + 1))
            return ValueArg(word, value, start, end), end
    value, end = _parse_value(source, pos)
    return ValueArg(None, value, start, end), end
