"""Check, and optionally fix, the layout style of AgL source in this repository.

AgL has significant indentation but no formatter, so the conventions that keep
programs readable are enforced here instead. The rules are deliberately few and
about layout alone -- none changes what a program means:

``fields``
    A record or exception declares its fields in the indented block form, one
    per line with its attributes, and one with no fields of its own has no body
    at all. Rewrites parenthesized, inline, and empty ``()`` field lists; the
    modules in ``FIELDS_EXEMPT`` exercise those forms on purpose.
``indent``
    A scope region's items sit one level (two spaces) in from their ``scope``
    header, and the ``end`` closer returns to the header's own column.
``end-blank``
    A blank line follows a region's closer, unless the next line closes an
    enclosing region.
``program-blank``
    A blank line precedes a ``program def``.
``scope-blank``
    A blank line precedes a ``scope`` header.

The last two attach to the comment block and the attributes written above a
declaration rather than splitting them from the declaration they belong to.

AgL appears in three places, and all three are checked:

* ``.agl`` files, whole;
* ```` ```agl ```` fenced blocks in Markdown documentation;
* AgL program sources written as Python string literals in the test suite --
  either a triple-quoted block or a run of implicitly concatenated one-line
  literals. Restyling those edits the literal in place, re-encoding what it
  inserts in the literal's own quoting.

Structure comes from AgL's own lexer, so a ``scope`` or ``record`` in a comment,
a string, or a `$` verbatim literal's payload is never mistaken for a
declaration. A file the lexer rejects -- the deliberate lexical-rejection
fixtures -- carries no structure to check and is skipped.

Usage::

    uv run python tools/agl_style.py --check   # report violations, exit 1 on any
    uv run python tools/agl_style.py --fix     # rewrite the offending files
"""

from __future__ import annotations

import difflib
import io
import operator
import re
import sys
import textwrap
import token as token_module
import tokenize as py_tokenize
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from lark.lexer import Token

from agm.agl.diagnostics import AglError
from agm.agl.lexer import LexError, lex_comment_spans, tokenize
from agm.agl.parser.parser import parse_program_unresolved
from agm.agl.syntax.nodes import Program

INDENT_WIDTH = 2

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("docs", "src", "packages/stdlib", "tests", "tools")
SKIP_DIRS = frozenset({".git", ".venv", "__pycache__", "dist", "htmlcov", "node_modules"})
# Modules exercising the parenthesized, inline, and empty `()` field-list forms
# on purpose, which the `fields` rule would rewrite away.
FIELDS_EXEMPT: frozenset[Path] = frozenset(
    map(
        Path,
        (
            "tests/agl/programs/types/bodyless_declarations.agl",
            "tests/test_agl_lexer.py",
            "tests/test_agl_parser.py",
            "tests/test_agl_repl_loop.py",
        ),
    )
)


@dataclass(frozen=True)
class Violation:
    """One style rule broken at one line of one file."""

    path: Path
    line: int  # 1-based, in the containing file
    rule: str
    message: str

    def render(self) -> str:
        location = self.path
        if location.is_absolute() and location.is_relative_to(REPO_ROOT):
            location = location.relative_to(REPO_ROOT)
        return f"{location}:{self.line}: {self.rule}: {self.message}"


# ---------------------------------------------------------------------------
# AgL structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """A ``scope``/``end`` pair, by 0-based line index, with its nested regions."""

    header: int
    closer: int
    children: tuple[Region, ...]


_ITEM_START_TYPES = frozenset({"_NEWLINE", "_INDENT", "_DEDENT", "SEMICOLON"})


@dataclass(frozen=True)
class Structure:
    """Where an AgL text's regions and ``program def`` declarations begin."""

    regions: tuple[Region, ...]
    program_lines: frozenset[int]  # 0-based lines starting a `program def`
    attribute_lines: frozenset[int]  # 0-based lines belonging to an attribute


def _nest(markers: Sequence[tuple[int, bool]], start: int) -> tuple[list[Region], int]:
    """Fold a flat header/closer marker list into regions from *start* onward."""
    regions: list[Region] = []
    index = start
    while index < len(markers):
        line, is_header = markers[index]
        if not is_header:
            return regions, index
        children, index = _nest(markers, index + 1)
        if index >= len(markers):
            # Unclosed region: no closer to align a body against.
            return regions, index
        regions.append(Region(line, markers[index][0], tuple(children)))
        index += 1
    return regions, index


def structure_of(source: str) -> Structure | None:
    """Locate *source*'s scope regions and ``program def`` lines, or None if it cannot lex."""
    try:
        tokens = list(tokenize(source))
    except LexError:
        return None
    markers: list[tuple[int, bool]] = []
    program_lines: list[int] = []
    attribute_lines: set[int] = set()
    attribute_start: int | None = None
    previous: str | None = None
    for tok in tokens:
        line = tok.line
        if line is None:  # pragma: no cover - only the synthetic layout tokens
            continue
        if tok.type == "AT":
            attribute_start = line - 1
        elif attribute_start is not None and tok.type == "_NEWLINE":
            attribute_lines.update(range(attribute_start, line))
            attribute_start = None
        if tok.type in {"SCOPE", "END"}:
            markers.append((line - 1, tok.type == "SCOPE"))
        elif tok.type == "program" and (previous is None or previous in _ITEM_START_TYPES):
            program_lines.append(line - 1)
        previous = tok.type
    regions, _ = _nest(markers, 0)
    return Structure(tuple(regions), frozenset(program_lines), frozenset(attribute_lines))


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _is_blank(line: str) -> bool:
    return not line.strip()


def _is_comment(line: str) -> bool:
    return line.lstrip().startswith("#")


def _is_attribute(line: str) -> bool:
    return line.lstrip().startswith("@")


def _shift(lines: list[str], span: range, amount: int) -> None:
    """Shift every non-blank line in *span* by *amount* columns, in place."""
    if amount == 0:
        return
    for index in span:
        if _is_blank(lines[index]):
            continue
        lines[index] = " " * amount + lines[index] if amount > 0 else lines[index][-amount:]


def _apply_indent(lines: list[str], regions: Iterable[Region]) -> list[Violation]:
    """Indent each region's body one level under its header, outermost first."""
    violations: list[Violation] = []
    for region in regions:
        body = range(region.header + 1, region.closer)
        if region.header == region.closer or not body:
            continue
        header_indent = _indent_of(lines[region.header])
        content = [index for index in body if not _is_blank(lines[index])]
        if not content:
            continue
        base = min(_indent_of(lines[index]) for index in content)
        amount = header_indent + INDENT_WIDTH - base
        if amount != 0:
            violations.append(
                Violation(
                    Path(),
                    region.header + 1,
                    "indent",
                    "a scope region's items belong one level in from their `scope` header",
                )
            )
            _shift(lines, body, amount)
        closer_indent = _indent_of(lines[region.closer])
        if closer_indent != header_indent:
            violations.append(
                Violation(
                    Path(),
                    region.closer + 1,
                    "indent",
                    "a region's `end` belongs at its `scope` header's column",
                )
            )
            lines[region.closer] = " " * header_indent + lines[region.closer].lstrip()
        violations.extend(_apply_indent(lines, region.children))
    return violations


def _closer_lines(regions: Iterable[Region]) -> Iterator[int]:
    for region in regions:
        yield region.closer
        yield from _closer_lines(region.children)


def _header_lines(regions: Iterable[Region]) -> Iterator[int]:
    for region in regions:
        yield region.header
        yield from _header_lines(region.children)


def _prefix_block_start(lines: list[str], index: int, attribute_lines: frozenset[int]) -> int:
    """The first line of the prefix documenting *index*, or *index* itself.

    A declaration's own lines run back over the comment block above it and over
    the attributes written on the lines between: a blank line belongs before
    that whole prefix, never inside it.
    """
    start = index
    indent = _indent_of(lines[index])
    while start > 0:
        previous_index = start - 1
        previous = lines[previous_index]
        if previous_index in attribute_lines:
            start -= 1
            continue
        if _indent_of(previous) != indent or not (_is_comment(previous) or _is_attribute(previous)):
            break
        start -= 1
    return start


def _blank_before(lines: list[str], index: int, attribute_lines: frozenset[int]) -> int | None:
    """Where a blank line is owed before *index*, or None when one is not."""
    target = _prefix_block_start(lines, index, attribute_lines)
    if target == 0 or _is_blank(lines[target - 1]):
        return None
    return target


def _apply_blank_lines(lines: list[str], structure: Structure) -> list[Violation]:
    """Insert the blank lines the vertical-spacing rules require."""
    closers = set(_closer_lines(structure.regions))
    headers = set(_header_lines(structure.regions))
    wanted: dict[int, tuple[str, str]] = {}

    for index in sorted(closers):
        following = index + 1
        if following >= len(lines) or _is_blank(lines[following]) or following in closers:
            continue
        wanted[following] = ("end-blank", "a blank line belongs after a region's `end`")

    for index in sorted(headers):
        target = _blank_before(lines, index, structure.attribute_lines)
        if target is not None:
            wanted.setdefault(
                target, ("scope-blank", "a blank line belongs before a `scope` header")
            )

    for index in sorted(structure.program_lines):
        target = _blank_before(lines, index, structure.attribute_lines)
        if target is not None:
            wanted.setdefault(
                target, ("program-blank", "a blank line belongs before a `program def`")
            )

    violations = [
        Violation(Path(), index + 1, rule, message)
        for index, (rule, message) in sorted(wanted.items())
    ]
    for index in sorted(wanted, reverse=True):
        lines.insert(index, "")
    return violations


# ---------------------------------------------------------------------------
# Field declarations
# ---------------------------------------------------------------------------

_DECLARATION_TYPES = frozenset({"record", "exception"})
_OPENER_TYPES = frozenset(
    {"LPAR", "LSQB", "INDEX_LSQB", "TYPEARG_LSQB", "DO_LSQB", "LBRACE", "CALL_LBRACE"}
)
_CLOSER_TYPES = frozenset({"RPAR", "RSQB", "RBRACE"})
_LINE_END_TYPES = frozenset({"_NEWLINE", "_DEDENT"})
_INLINE_END_TYPES = _LINE_END_TYPES | {"_INDENT", "SEMICOLON"}


@dataclass(frozen=True)
class FieldList:
    """A record or exception declaration whose fields are not in the block form."""

    keyword: int  # token index of `record`/`exception`
    head_end: int  # token index of the head's last token
    body_end: int  # token index of the list's last token
    fields: tuple[tuple[int, int], ...]  # inclusive token-index span of each field
    punctuation: frozenset[int]  # the `=`, parentheses, and separating commas
    restylable: bool  # whether the declaration's line ends with the list


def _start(tok: Token) -> int:
    assert tok.start_pos is not None
    return tok.start_pos


def _end(tok: Token) -> int:
    assert tok.end_pos is not None
    return tok.end_pos


def _start_line(tok: Token) -> int:
    """*tok*'s 0-based line."""
    assert tok.line is not None
    return tok.line - 1


def _after_group(tokens: Sequence[Token], index: int) -> int:
    """The index just past the bracket group opening at *index*."""
    depth = 0
    while index < len(tokens):
        kind = tokens[index].type
        if kind in _OPENER_TYPES:
            depth += 1
        elif kind in _CLOSER_TYPES:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return index


def _after_name(tokens: Sequence[Token], index: int) -> int:
    """The index just past a possibly scope-qualified name starting at *index*."""
    while index < len(tokens) and tokens[index].type == "MODQUAL":
        index += 1
    return index + 1


def _field_list_at(tokens: Sequence[Token], keyword: int) -> FieldList | None:
    """The non-block field list of the declaration at *keyword*, if it has one."""
    index = _after_name(tokens, keyword + 1)
    if index < len(tokens) and tokens[index].type in _OPENER_TYPES - {"LPAR"}:
        index = _after_group(tokens, index)
    if index < len(tokens) and tokens[index].type == "extends":
        index = _after_name(tokens, index + 1)
    head_end = index - 1
    punctuation: set[int] = set()
    if index < len(tokens) and tokens[index].type == "EQ":
        punctuation.add(index)
        index += 1
    if index >= len(tokens) or tokens[index].type in _INLINE_END_TYPES:
        return None  # the block form, or no body at all
    parenthesized = tokens[index].type == "LPAR"
    if parenthesized:
        punctuation.add(index)
        stop = _after_group(tokens, index) - 1
        punctuation.add(stop)
        index += 1
        after = stop + 1
    else:
        stop = index
        depth = 0
        while stop < len(tokens) and (depth or tokens[stop].type not in _INLINE_END_TYPES):
            if tokens[stop].type in _OPENER_TYPES:
                depth += 1
            elif tokens[stop].type in _CLOSER_TYPES:
                depth -= 1
            stop += 1
        after = stop
    fields: list[tuple[int, int]] = []
    start = index
    depth = 0
    for position in range(index, stop + 1):
        kind = tokens[position].type if position < stop else "COMMA"
        if kind in _OPENER_TYPES:
            depth += 1
        elif kind in _CLOSER_TYPES:
            depth -= 1
        elif kind == "COMMA" and depth == 0:
            if position < stop:
                punctuation.add(position)
            if position > start:
                fields.append((start, position - 1))
            start = position + 1
    return FieldList(
        keyword,
        head_end,
        after - 1,
        tuple(fields),
        frozenset(punctuation),
        after >= len(tokens) or tokens[after].type in _LINE_END_TYPES,
    )


def _starts_item(tokens: Sequence[Token], index: int) -> bool:
    """Whether the declaration keyword at *index* opens an item, past its prefix.

    The prefix is a ``builtin`` modifier and the attributes before it.
    """
    index -= 1
    if index >= 0 and tokens[index].type == "builtin":
        index -= 1
    while index >= 0:
        if tokens[index].type == "RPAR":
            depth = 0
            while index >= 0:
                depth += tokens[index].type in _CLOSER_TYPES
                depth -= tokens[index].type in _OPENER_TYPES
                index -= 1
                if depth == 0:
                    break
        if index >= 1 and tokens[index].type == "NAME" and tokens[index - 1].type == "AT":
            index -= 2
        else:
            break
    return index < 0 or tokens[index].type in _ITEM_START_TYPES


def field_lists(tokens: Sequence[Token]) -> list[FieldList]:
    """Every record/exception declaration in *tokens* not in the block form."""
    found = (
        _field_list_at(tokens, index)
        for index, tok in enumerate(tokens)
        if tok.type in _DECLARATION_TYPES and _starts_item(tokens, index)
    )
    return [field_list for field_list in found if field_list is not None]


def _field_text(source: str, tokens: Sequence[Token], span: tuple[int, int]) -> str | None:
    """One field's tokens on a single line, or None if they cannot share one."""
    parts: list[str] = []
    for index in range(span[0], span[1] + 1):
        tok = tokens[index]
        text = source[_start(tok) : _end(tok)]
        if "\n" in text:
            return None
        if index > span[0]:
            gap = source[_end(tokens[index - 1]) : _start(tok)]
            parts.append(" " if "\n" in gap else gap)
        parts.append(text)
    return "".join(parts)


def _block_lines(
    source: str,
    tokens: Sequence[Token],
    field_list: FieldList,
    comments: Sequence[tuple[int, int]],
) -> list[str] | None:
    """*field_list*'s fields as block lines, keeping its comments, or None.

    A comment trailing a field's line stays on that field's line; any other
    comment in the list gets its own line ahead of the field that follows it.
    """
    texts: list[str] = []
    for span in field_list.fields:
        text = _field_text(source, tokens, span)
        if text is None:
            return None
        texts.append(text)
    leading: list[list[str]] = [[] for _ in range(len(texts) + 1)]
    begin = _end(tokens[field_list.head_end])
    end = _end(tokens[field_list.body_end])
    for comment_start, comment_end in comments:
        if not begin <= comment_start < end:
            continue
        comment = source[comment_start:comment_end]
        if any(
            _start(tokens[first]) < comment_start < _end(tokens[last])
            for first, last in field_list.fields
        ):
            return None  # a comment splitting one field's tokens
        previous = max(index for index, tok in enumerate(tokens) if _end(tok) <= comment_start)
        before = sum(1 for first, _ in field_list.fields if first <= previous)
        if before and "\n" not in source[_end(tokens[previous]) : comment_start]:
            texts[before - 1] += " " + comment
        else:
            leading[before].append(comment)
    lines: list[str] = []
    for comments_ahead, text in zip(leading, [*texts, None]):
        lines.extend(comments_ahead)
        if text is not None:
            lines.append(text)
    return lines


def _apply_fields(source: str, tokens: Sequence[Token]) -> tuple[str, list[Violation]]:
    """Rewrite each non-block field list into the block form, or drop an empty one."""
    lines = source.split("\n")
    comments = lex_comment_spans(source)
    violations: list[Violation] = []
    edits: list[tuple[int, int, str]] = []
    for field_list in field_lists(tokens):
        line = _start_line(tokens[field_list.keyword])
        block = _block_lines(source, tokens, field_list, comments)
        if not field_list.restylable or block is None:
            violations.append(
                Violation(
                    Path(),
                    line + 1,
                    "fields",
                    "declare these fields in the indented block form by hand",
                )
            )
            continue
        margin = " " * (_indent_of(lines[line]) + INDENT_WIDTH)
        violations.append(
            Violation(
                Path(),
                line + 1,
                "fields",
                "fields belong in the indented block form, one per line"
                if block
                else "a declaration without fields has no `()`",
            )
        )
        # A comment trailing the whole list describes the declaration: it stays
        # on the declaration's own line.
        end = _end(tokens[field_list.body_end])
        trailing = next(
            (
                source[end:comment_end]
                for comment_start, comment_end in comments
                if comment_start >= end and source[end:comment_start].strip(" ") == ""
            ),
            "",
        )
        edits.append(
            (
                _end(tokens[field_list.head_end]),
                end + len(trailing),
                trailing + "".join(f"\n{margin}{text}" for text in block),
            )
        )
    for start, end, text in reversed(edits):
        source = source[:start] + text + source[end:]
    return source, violations


# ---------------------------------------------------------------------------
# Restyling one AgL text
# ---------------------------------------------------------------------------


_LAYOUT_TYPES = frozenset({"_NEWLINE", "_INDENT", "_DEDENT"})


def _meaning(source: str) -> Program | list[tuple[str, str]] | None:
    """What *source* means, blind to the layout and field-list forms style may change.

    A source that parses means its syntax tree, which records neither. A
    fragment that only lexes means its token stream less the layout tokens and
    the punctuation of non-block field lists.
    """
    try:
        return parse_program_unresolved(source)
    except AglError:
        pass
    try:
        tokens = list(tokenize(source))
    except LexError:
        return None
    punctuation = {index for field_list in field_lists(tokens) for index in field_list.punctuation}
    return [
        (tok.type, str(tok))
        for index, tok in enumerate(tokens)
        if tok.type not in _LAYOUT_TYPES and index not in punctuation
    ]


def restyle(source: str, *, fields: bool = True) -> tuple[str, list[Violation]]:
    """Return *source* in the repository's AgL layout style, with what it broke.

    A text the lexer cannot read comes back untouched: without a token stream
    there is no way to tell a region header from the word ``scope`` in a string.
    *fields* applies the ``fields`` rule.

    Only layout and field-list forms move, so the result must mean exactly what
    it meant before and keep no field list the rule could rewrite. Anything
    else -- a shifted line that turned out to be the inside of a multi-line
    string, say -- is a bug in this tool rather than a style fix, and the
    original is kept instead.

    A margin common to every line, as a snippet indented with its Python code
    has, is set aside while restyling.
    """
    lines = source.split("\n")
    width = min((_indent_of(line) for line in lines if not _is_blank(line)), default=0)
    if width and all(line[:width].strip(" ") == "" for line in lines):
        styled, violations = restyle("\n".join(line[width:] for line in lines), fields=fields)
        margin = " " * width
        return "\n".join(margin + line if line else line for line in styled.split("\n")), violations
    try:
        tokens = list(tokenize(source))
    except LexError:
        return source, []
    styled, violations = _apply_fields(source, tokens) if fields else (source, [])
    structure = structure_of(styled)
    if structure is not None:
        lines = styled.split("\n")
        violations.extend(_apply_indent(lines, structure.regions))
        # Indentation moves no lines, so the structure's line numbers still hold.
        violations.extend(_apply_blank_lines(lines, structure))
        styled = "\n".join(lines)
    if styled != source and (
        structure is None
        or _meaning(styled) != _meaning(source)
        or (fields and _rewritable(styled))
    ):
        return source, [
            Violation(
                Path(),
                violations[0].line,
                "manual",
                "restyling this source would change what it means; fix it by hand",
            )
        ]
    return styled, violations


def _rewritable(source: str) -> bool:
    """Whether *source* still holds a field list the ``fields`` rule would rewrite."""
    tokens = list(tokenize(source))
    return any(
        field_list.restylable
        and _block_lines(source, tokens, field_list, lex_comment_spans(source)) is not None
        for field_list in field_lists(tokens)
    )


def _relocate(violations: Iterable[Violation], path: Path, offset: int) -> list[Violation]:
    """Retarget violations found in an extracted fragment onto its containing file."""
    return [
        Violation(path, violation.line + offset, violation.rule, violation.message)
        for violation in violations
    ]


# ---------------------------------------------------------------------------
# Markdown documents
# ---------------------------------------------------------------------------

_AGL_FENCE = re.compile(r"^([ \t]*)```agl\s*$")


def restyle_markdown(source: str, path: Path, *, fields: bool) -> tuple[str, list[Violation]]:
    """Restyle every ```` ```agl ```` block in a Markdown document."""
    lines = source.split("\n")
    violations: list[Violation] = []
    added = 0  # lines restyling has added so far, to report against the original
    index = 0
    while index < len(lines):
        match = _AGL_FENCE.match(lines[index])
        if match is None:
            index += 1
            continue
        margin = str(match.group(1))
        opening = index
        index += 1
        body: list[str] = []
        while index < len(lines) and lines[index].strip() != "```":
            body.append(lines[index])
            index += 1
        if index >= len(lines):
            break
        block = "\n".join(line[len(margin) :] if line.startswith(margin) else line for line in body)
        styled, found = restyle(block, fields=fields)
        if found:
            violations.extend(_relocate(found, path, opening + 1 - added))
            replacement = [margin + line if line else line for line in styled.split("\n")]
            lines[opening + 1 : index] = replacement
            added += len(replacement) - len(body)
            index = opening + 1 + len(replacement)
        index += 1
    return "\n".join(lines), violations


# ---------------------------------------------------------------------------
# AgL inside Python string literals
# ---------------------------------------------------------------------------

_ESCAPES = {
    "\n": "",
    "\\": "\\",
    "'": "'",
    '"': '"',
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
}


@dataclass(frozen=True)
class Literal:
    """One Python string literal, decoded."""

    value: str
    offsets: list[int]  # source offset per decoded character, plus the closing quote's
    opening: int  # source offset of the literal's first character, prefix included
    prefix: str
    quote: str
    triple: bool

    @property
    def raw(self) -> bool:
        return "r" in self.prefix.lower()


def _decode(literal: str, origin: int) -> Literal | None:
    """Decode one Python string literal at source offset *origin*.

    Returns None for a literal whose prefix (``f``, ``b``) makes it unusable as
    a plain AgL snippet.
    """
    index = 0
    while index < len(literal) and literal[index] not in "\"'":
        index += 1
    prefix = literal[:index].lower()
    if "f" in prefix or "b" in prefix:
        return None
    raw = "r" in prefix
    quote = literal[index]
    width = 3 if literal[index : index + 3] == quote * 3 else 1
    body_start = index + width
    body_end = len(literal) - width
    value: list[str] = []
    offsets: list[int] = []
    position = body_start
    while position < body_end:
        char = literal[position]
        if char != "\\" or raw or position + 1 >= body_end:
            value.append(char)
            offsets.append(origin + position)
            position += 1
            continue
        escape = literal[position + 1]
        if escape in _ESCAPES:
            decoded = _ESCAPES[escape]
            if decoded:
                value.append(decoded)
                offsets.append(origin + position)
            position += 2
            continue
        # Numeric and named escapes are not line starts and never carry an
        # AgL structure position, so they only have to keep the value's shape.
        span = 2
        if escape == "x":
            span = 4
        elif escape == "u":
            span = 6
        elif escape == "U":
            span = 10
        elif escape == "N" and literal[position + 2 : position + 3] == "{":
            span = literal.index("}", position) - position + 1
        chunk = literal[position : position + span]
        try:
            decoded = chunk.encode().decode("unicode_escape")
        except (UnicodeDecodeError, ValueError):  # pragma: no cover - malformed source
            return None
        for character in decoded:
            value.append(character)
            offsets.append(origin + position)
        position += span
    offsets.append(origin + body_end)
    return Literal("".join(value), offsets, origin, literal[:index], quote, width == 3)


@dataclass(frozen=True)
class LiteralRun:
    """A run of implicitly concatenated Python string literals, decoded."""

    value: str
    offsets: list[int]  # per decoded character, plus a past-the-end entry
    start_line: int  # 1-based line of the run's first literal
    parts: tuple[tuple[int, Literal], ...]  # each literal, by its first index in `value`

    def spans_lines_at(self, offset: int) -> bool:
        """Whether *offset* sits in a literal whose quotes let it hold real newlines."""
        return any(
            part.triple and part.offsets[0] <= offset < part.offsets[-1] for _, part in self.parts
        )

    def part_at(self, index: int) -> tuple[int, Literal]:
        """The literal holding `value[index]`, or the last one past the end."""
        return next((start, part) for start, part in reversed(self.parts) if start <= index)


def _literal_runs(source: str) -> Iterator[LiteralRun]:
    """Yield each run of adjacent Python string literals in *source*."""
    lines = source.splitlines(keepends=True)
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))

    def offset(row: int, column: int) -> int:
        return line_starts[row - 1] + column

    try:
        tokens = list(py_tokenize.generate_tokens(io.StringIO(source).readline))
    except (py_tokenize.TokenError, IndentationError, SyntaxError):  # pragma: no cover
        return
    run: list[Literal] = []
    run_line = 0

    def collect() -> LiteralRun:
        starts = [0]
        for part in run:
            starts.append(starts[-1] + len(part.value))
        return LiteralRun(
            "".join(part.value for part in run),
            [position for part in run for position in part.offsets[:-1]] + [run[-1].offsets[-1]],
            run_line,
            tuple(zip(starts, run)),
        )

    for tok in tokens:
        if tok.type == token_module.STRING:
            decoded = _decode(tok.string, offset(tok.start[0], tok.start[1]))
            if decoded is None:
                run = []
                continue
            if not run:
                run_line = tok.start[0]
            run.append(decoded)
            continue
        if tok.type in {token_module.NL, token_module.COMMENT}:
            continue
        if run:
            yield collect()
            run = []
    if run:  # pragma: no cover - a module never ends inside an expression
        yield collect()


_REGION = re.compile(r"^[ \t]*scope[ \t]+[^\n]*$", re.M)


def _looks_like_agl(value: str) -> bool:
    """Whether a string literal is an AgL snippet carrying a scope region or a declaration.

    The text tests only prefilter; the lexer decides. A declaration only counts
    where its keyword opens an item, and in a literal that parses as a program,
    as prose mentioning a ``record`` does not.
    """
    if "\n" in value and _REGION.search(value) is not None:
        return structure_of(value) is not None
    if not any(keyword in value for keyword in _DECLARATION_TYPES):
        return False
    source = textwrap.dedent(value)
    try:
        if not field_lists(list(tokenize(source))):
            return False
        parse_program_unresolved(source)
    except AglError:
        return False
    return True


# Same-offset edits are applied highest rank first, and each insertion pushes
# the previous one right, so the lower-ranked blank line ends up ahead of the
# indentation it shares an offset with.
_INDENT_RANK = 1
_BLANK_RANK = 0


def _splice(source: str, run: LiteralRun, styled: str) -> str | None:
    """Write *styled* back into the literal *run* occupies, or None if it cannot.

    Restyling only ever inserts spaces at the start of an AgL line or inserts a
    whole blank line, so the edit reduces to insertions at line starts. Where a
    literal opens its own physical source line the blank line becomes a new
    literal of the same shape; elsewhere it is an escaped newline in place.
    """
    old_lines = run.value.split("\n")
    new_lines = styled.split("\n")
    line_start: list[int] = []
    position = 0
    for line in old_lines:
        line_start.append(position)
        position += len(line) + 1

    # (offset, rank, text): an empty text deletes the character at `offset`.
    # Insertions at one offset land in `rank` order, so a blank line inserted
    # before a line that is also being indented keeps the indentation with the
    # line rather than with the blank.
    edits: list[tuple[int, int, str]] = []
    old_index = 0
    for new_line in new_lines:
        if old_index < len(old_lines) and new_line.lstrip() == old_lines[old_index].lstrip():
            added = _indent_of(new_line) - _indent_of(old_lines[old_index])
            start = line_start[old_index]
            if added > 0:
                edits.append((run.offsets[start], _INDENT_RANK, " " * added))
            for removed in range(-added):
                edits.append((run.offsets[start + removed], _INDENT_RANK, ""))
            old_index += 1
            continue
        if new_line != "":
            return None
        # A blank line inserted before the line `old_index` still to be matched.
        if old_index >= len(old_lines):
            return None
        offset = run.offsets[line_start[old_index]]
        if run.spans_lines_at(offset):
            # A triple-quoted literal holds the blank line directly.
            edits.append((offset, _BLANK_RANK, "\n"))
            continue
        line_begin = source.rfind("\n", 0, offset) + 1
        opener = source[line_begin:offset]
        quote = opener.strip().lstrip("(")
        if quote in {'"', "'"}:
            # One AgL line per concatenated literal: the blank line is its own.
            indent = opener[: len(opener) - len(opener.lstrip())]
            edits.append((line_begin, _BLANK_RANK, f"{indent}{quote}\\n{quote}\n"))
        else:
            # Several AgL lines in one literal: an escaped newline in place.
            edits.append((offset, _BLANK_RANK, "\\n"))
    if old_index != len(old_lines):
        return None

    result = source
    for offset, _, text in sorted(edits, key=operator.itemgetter(0, 1), reverse=True):
        if text:
            result = result[:offset] + text + result[offset:]
        else:
            result = result[:offset] + result[offset + 1 :]
    return result


def _owns_line(source: str, part: Literal) -> bool:
    """Whether *part* opens its physical line, and so sits inside brackets."""
    line_begin = source.rfind("\n", 0, part.opening) + 1
    return source[line_begin : part.opening].strip() in {"", "("}


def _encode(source: str, part: Literal, text: str, at_end: bool) -> str | None:
    """*text* written inside *part*'s quotes, or None if its quoting cannot hold it.

    A newline in a one-line literal that opens its own source line splits the
    literal in two, the second opening the next source line; elsewhere it is
    an escaped newline. *at_end* says whether the text lands at the literal's
    closing quote, where a split would only leave an empty literal behind.
    """
    if part.raw:
        if "\\" in text or part.quote in text or ("\n" in text and not part.triple):
            return None
        return text
    split = not part.triple and _owns_line(source, part)
    line_begin = source.rfind("\n", 0, part.opening) + 1
    reopen = f"{part.quote}\n{' ' * (part.opening - line_begin)}{part.prefix}{part.quote}"
    encoded: list[str] = []
    for index, char in enumerate(text):
        if char == "\\" or char == part.quote:
            encoded.append("\\" + char)
        elif char == "\n" and part.triple:
            encoded.append(char)
        elif char == "\n":
            last = at_end and index == len(text) - 1
            encoded.append("\\n" + reopen if split and not last else "\\n")
        elif char == "\t":
            encoded.append("\\t")
        elif not char.isprintable():
            return None
        else:
            encoded.append(char)
    return "".join(encoded)


def _splice_anywhere(source: str, run: LiteralRun, styled: str) -> str | None:
    """Write *styled* back into *run* by its character diff, or None if it cannot.

    Unlike :func:`_splice`, edits may fall anywhere in a line, as the ``fields``
    rule's do; each lands in the literal holding it, re-encoded for its quotes.
    """
    edits: list[tuple[int, int, str]] = []  # source span replaced, in source order
    matcher = difflib.SequenceMatcher(None, run.value, styled, autojunk=False)
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        position = old_start
        text = styled[new_start:new_end]
        while True:
            part_start, part = run.part_at(position)
            part_end = part_start + len(part.value)
            stop = min(old_end, part_end)
            end_offset = part.offsets[-1] if stop == part_end else run.offsets[stop]
            encoded = _encode(source, part, text, stop == part_end) if text else ""
            if encoded is None:
                return None
            edits.append((run.offsets[position], end_offset, encoded))
            text = ""
            position = stop
            if position >= old_end:
                break
    result = source
    for start, end, encoded in reversed(edits):
        result = result[:start] + encoded + result[end:]
    return result


def restyle_python(source: str, path: Path, *, fields: bool) -> tuple[str, list[Violation]]:
    """Restyle every AgL snippet written as a string literal in a Python module.

    Writing back through quotes and escapes is where this tool can get a literal
    subtly wrong, so every rewritten snippet must then read exactly as intended
    rather than the edit arithmetic being trusted. One re-read of the module
    confirms them all at once; should any be off, each is confirmed on its own
    so that only the culprit is left for a manual fix.
    """
    result, violations, expected = _restyle_literals(source, path, fields, confirm_each=False)
    runs = [run.value for run in _literal_runs(result)]
    if all(index < len(runs) and runs[index] == value for index, value in expected.items()):
        return result, violations
    result, violations, _ = _restyle_literals(source, path, fields, confirm_each=True)
    return result, violations


def _restyle_literals(
    source: str, path: Path, fields: bool, *, confirm_each: bool
) -> tuple[str, list[Violation], dict[int, str]]:
    """Restyle *source*'s snippets, with the value each rewritten run should read.

    Snippets are rewritten last to first, so an edit never moves the source
    offsets of a snippet still to come. An edit neither merges nor splits
    runs, so a run keeps its index.
    """
    found_per_run: list[list[Violation]] = []
    expected: dict[int, str] = {}
    result = source
    runs = list(_literal_runs(source))
    for index in reversed(range(len(runs))):
        run = runs[index]
        if not _looks_like_agl(run.value):
            continue
        styled, found = restyle(run.value, fields=fields)
        if not found:
            continue
        spliced = (
            result
            if styled == run.value
            else _splice(result, run, styled) or _splice_anywhere(result, run, styled)
        )
        if spliced is not None and confirm_each:
            confirmed = list(_literal_runs(spliced))
            if index >= len(confirmed) or confirmed[index].value != styled:
                spliced = None
        if spliced is None:
            found_per_run.append(
                [
                    Violation(
                        path,
                        run.start_line,
                        "manual",
                        "an AgL snippet needs restyling that cannot be applied automatically",
                    )
                ]
            )
            continue
        found_per_run.append(_relocate(found, path, run.start_line - 1))
        expected[index] = styled
        result = spliced
    violations = [violation for found in reversed(found_per_run) for violation in found]
    return result, violations, expected


# ---------------------------------------------------------------------------
# Driving the tree
# ---------------------------------------------------------------------------


def _sources(root: Path) -> Iterator[Path]:
    for directory in SOURCE_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if SKIP_DIRS & set(path.parts) or not path.is_file():
                continue
            if path.suffix in {".agl", ".md", ".py"}:
                yield path
    readme = root / "README.md"
    if readme.is_file():
        yield readme


def restyle_file(path: Path) -> tuple[str, list[Violation]] | None:
    """Restyle one file by kind, or None when it holds no AgL to restyle."""
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    fields = not (path.is_relative_to(REPO_ROOT) and path.relative_to(REPO_ROOT) in FIELDS_EXEMPT)
    if path.suffix == ".agl":
        styled, violations = restyle(source, fields=fields)
        return styled, _relocate(violations, path, 0)
    if path.suffix == ".md":
        return restyle_markdown(source, path, fields=fields)
    if path.name == Path(__file__).name:
        return None
    return restyle_python(source, path, fields=fields)


USAGE = "usage: agl_style.py (--check | --fix) [path ...]"


def main(argv: Sequence[str]) -> int:
    """Check or fix the AgL layout style of *argv*'s paths, or of the whole tree."""
    flags = [argument for argument in argv if argument.startswith("-")]
    if flags not in ([["--check"]] + [["--fix"]]):
        print(USAGE, file=sys.stderr)
        return 2
    fixing = flags == ["--fix"]

    given = [Path(argument).resolve() for argument in argv if not argument.startswith("-")]
    violations: list[Violation] = []
    changed: list[Path] = []
    for path in given or list(_sources(REPO_ROOT)):
        outcome = restyle_file(path)
        if outcome is None:
            continue
        styled, found = outcome
        violations.extend(found)
        if fixing and styled != path.read_text(encoding="utf-8"):
            path.write_text(styled, encoding="utf-8")
            changed.append(path)

    for violation in violations:
        print(violation.render())
    if fixing:
        print(f"restyled {len(changed)} file(s)")
        return 0
    if violations:
        print(f"\n{len(violations)} AgL style violation(s); run `just agl-style-fix` to apply.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
