"""Check, and optionally fix, the layout style of AgL source in this repository.

AgL has significant indentation but no formatter, so the conventions that keep
programs readable are enforced here instead. The rules are deliberately few and
purely about vertical layout -- nothing reflows a line's contents:

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
  literals. Restyling those edits the literal in place, adding indentation
  inside the quotes and whole lines between them.

Structure comes from AgL's own lexer, so a ``scope`` in a comment, a string, or
a `$` verbatim literal's payload is never mistaken for a region header. A file
the lexer rejects -- the deliberate lexical-rejection fixtures -- carries no
structure to check and is skipped.

Usage::

    uv run python tools/agl_style.py --check   # report violations, exit 1 on any
    uv run python tools/agl_style.py --fix     # rewrite the offending files
"""

from __future__ import annotations

import io
import operator
import re
import sys
import token as token_module
import tokenize as py_tokenize
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from agm.agl.lexer import LexError, tokenize

INDENT_WIDTH = 2

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("docs", "src", "packages/stdlib", "tests", "tools")
SKIP_DIRS = frozenset({".git", ".venv", "__pycache__", "dist", "htmlcov", "node_modules"})


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
# Restyling one AgL text
# ---------------------------------------------------------------------------


_LAYOUT_TYPES = frozenset({"_NEWLINE", "_INDENT", "_DEDENT"})


def _meaning(source: str) -> list[tuple[str, str]] | None:
    """The token stream *source* means, with the layout tokens style may move."""
    try:
        return [(tok.type, str(tok)) for tok in tokenize(source) if tok.type not in _LAYOUT_TYPES]
    except LexError:
        return None


def restyle(source: str) -> tuple[str, list[Violation]]:
    """Return *source* in the repository's AgL layout style, with what it broke.

    A text the lexer cannot read comes back untouched: without a token stream
    there is no way to tell a region header from the word ``scope`` in a string.

    Layout is the only thing that moves, so the result must mean exactly what it
    meant before. Anything else -- a shifted line that turned out to be the
    inside of a multi-line string, say -- is a bug in this tool rather than a
    style fix, and the original is kept instead.
    """
    structure = structure_of(source)
    if structure is None:
        return source, []
    lines = source.split("\n")
    violations = _apply_indent(lines, structure.regions)
    # Indentation moves no lines, so the structure's line numbers still hold.
    violations.extend(_apply_blank_lines(lines, structure))
    styled = "\n".join(lines)
    if violations and _meaning(styled) != _meaning(source):
        return source, [
            Violation(
                Path(),
                structure.regions[0].header + 1 if structure.regions else 1,
                "manual",
                "restyling this source would change what it means; fix it by hand",
            )
        ]
    return styled, violations


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


def restyle_markdown(source: str, path: Path) -> tuple[str, list[Violation]]:
    """Restyle every ```` ```agl ```` block in a Markdown document."""
    lines = source.split("\n")
    violations: list[Violation] = []
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
        styled, found = restyle(block)
        if found:
            violations.extend(_relocate(found, path, opening + 1))
            replacement = [margin + line if line else line for line in styled.split("\n")]
            lines[opening + 1 : index] = replacement
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


def _decode(literal: str, origin: int) -> tuple[str, list[int], bool] | None:
    """Decode one Python string literal to its value, offsets, and quote width.

    *origin* is the literal's offset in the enclosing source, so the returned
    offsets address that source directly; the flag reports whether the literal
    is triple-quoted, and so may carry newlines of its own. Returns None for a
    literal whose prefix (``f``, ``b``) makes it unusable as a plain AgL snippet.
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
    return "".join(value), offsets, width == 3


@dataclass(frozen=True)
class LiteralRun:
    """A run of implicitly concatenated Python string literals, decoded."""

    value: str
    offsets: list[int]  # per decoded character, plus a past-the-end entry
    start_line: int  # 1-based line of the run's first literal
    triple_spans: tuple[tuple[int, int], ...]  # source spans of the run's triple-quoted parts

    def spans_lines_at(self, offset: int) -> bool:
        """Whether *offset* sits in a literal whose quotes let it hold real newlines."""
        return any(start <= offset < end for start, end in self.triple_spans)


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
    run: list[tuple[str, list[int], bool]] = []
    run_line = 0

    def collect() -> LiteralRun:
        return LiteralRun(
            "".join(value for value, _, _ in run),
            [position for _, positions, _ in run for position in positions[:-1]] + [run[-1][1][-1]],
            run_line,
            tuple((positions[0], positions[-1]) for _, positions, triple in run if triple),
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
    """Whether a string literal is an AgL snippet carrying a scope region."""
    return "\n" in value and _REGION.search(value) is not None and structure_of(value) is not None


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


def restyle_python(source: str, path: Path) -> tuple[str, list[Violation]]:
    """Restyle every AgL snippet written as a string literal in a Python module."""
    violations: list[Violation] = []
    result = source
    while True:
        for run in _literal_runs(result):
            if not _looks_like_agl(run.value):
                continue
            styled, found = restyle(run.value)
            if not found:
                continue
            spliced = _splice(result, run, styled)
            # Writing back through quotes and escapes is where this tool can
            # get a literal subtly wrong, so confirm the snippet now reads
            # exactly as intended rather than trusting the edit arithmetic.
            if spliced is not None and not any(
                other.value == styled for other in _literal_runs(spliced)
            ):
                spliced = None
            if spliced is None:
                violations.append(
                    Violation(
                        path,
                        run.start_line,
                        "manual",
                        "an AgL snippet needs restyling that cannot be applied automatically",
                    )
                )
                continue
            violations.extend(_relocate(found, path, run.start_line - 1))
            result = spliced
            break
        else:
            return result, violations


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
    if path.suffix == ".agl":
        styled, violations = restyle(source)
        return styled, _relocate(violations, path, 0)
    if path.suffix == ".md":
        return restyle_markdown(source, path)
    if path.name == Path(__file__).name:
        return None
    return restyle_python(source, path)


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
