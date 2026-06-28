"""Mode-stack raw scanner for the AgL lexer.

Produces a stream of :class:`lark.lexer.Token` objects from raw source text.
The scanner handles:

- CODE mode: keywords, identifiers, numbers, operators (maximal munch), and
  horizontal whitespace / ``#`` comments.
- Template mode: single- and triple-quoted string literals with
  ``${...}`` interpolation and the JSON escape set plus ``\\$``.
- Layout signalling: ``_NEWLINE`` tokens carrying the next real line's leading
  indentation width (tabs expanded at ``tab_len=4``, comments skipped).

The layout filter (``layout.py``) consumes this stream and injects
``_INDENT``/``_DEDENT`` tokens; together they form the full token stream fed
to the Lark parser.

Newline normalization
---------------------
The scanner normalizes line endings at entry, universal-newline style: every
``\\r\\n`` and every lone ``\\r`` is converted to a single ``\\n`` *before* any
scanning happens.  Layout measurement, string scanning, and triple-quoted
dedent therefore all operate on the normalized text.

**Offset convention (accepted ruling):** after normalization, every
``start_pos``/``end_pos`` on a token and every ``start_offset``/``end_offset``
on a :class:`SourceSpan` refers to an index into the *normalized* text, not the
original bytes.  Offsets are 0-based and end-exclusive; lines and columns are
1-based.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from lark.lexer import Token

from agm.agl.diagnostics import Diagnostic, SourceSpan
from agm.agl.lexer.errors import LexError
from agm.agl.lexer.tokens import (
    ARROW,
    ASSIGN,
    AT,
    COLON,
    COMMA,
    DCOLON,
    DECIMAL,
    DOT,
    EQ,
    EQ_EQ,
    GE,
    GT,
    INT,
    INTERP_END,
    INTERP_START,
    KEYWORDS,
    LBRACE,
    LE,
    LPAR,
    LSQB,
    LT,
    MINUS,
    NAME,
    NEQ,
    NEWLINE,
    PIPE,
    PLUS,
    RBRACE,
    RPAR,
    RSQB,
    SEMICOLON,
    SLASH,
    STAR,
    STRING_FRAGMENT,
    TEMPLATE_END,
    TEMPLATE_START,
    THIN_ARROW,
)
from agm.util.text import normalize_newlines

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Characters that terminate an identifier scan.  An identifier starts with
# a (Unicode) letter or ``_`` and then greedily consumes every character that
# is NOT in this set.  Whitespace and the structural punctuators/operators
# that must remain standalone delimiters (brackets, field access ``.``,
# separators ``,`` ``;``, type/arg ``:``, ``|``, ``/``, and ``@``) are listed
# here.
# Everything else — including ``-``, ``?``, ``!``, ``<``, ``>``, ``=``,
# the string quotes ``"`` and ``'``, the arithmetic operators ``+`` and ``*``,
# and arbitrary Unicode letters/digits — is an identifier-continuation
# character, so names like ``ask-prompt``, ``ask?``, ``do-it-now!``, ``a+b``
# and ``foo"bar`` scan as a single token.  Operator tokens (``->``, ``=>``,
# ``::``, ``!=``, ``<=``, ``>=``, ``==``, field access ``.``, etc.) still lex as
# operators whenever they appear as standalone, whitespace-delimited tokens:
# spaces (or another stop character) break the identifier before the
# operator's first character.  A leading ``"`` or ``'`` (or one after
# whitespace) still starts a string template because the identifier-start
# predicate requires a letter or ``_``.
_IDENT_STOP: frozenset[str] = frozenset({
    # whitespace
    " ", "\t", "\n", "\r",
    # structural punctuators / operators that stay standalone delimiters.
    # String quotes (" and ') and the operator characters + * are NOT stop
    # characters: they may appear inside an identifier (e.g. ``foo"bar``,
    # ``a+b``, ``n*x``).  A leading " or ' (or one after whitespace) still
    # starts a string template because the identifier-start predicate requires
    # a letter or _.
    # @ is a stop character so that @pos/@std/@named lex as two tokens (AT NAME)
    # rather than gluing into the preceding identifier.
    "(", ")", "[", "]", "{", "}",
    ":", ",", ".", "|", ";", "/", "@",
})

_TAB_LEN = 4


def _is_ascii_digit(ch: str) -> bool:
    """Return True iff *ch* is an ASCII digit (``0``–``9``)."""
    return "0" <= ch <= "9"

# Single-char operator table (must not overlap with maximal-munch multi-char ops).
# NOTE: "-" is intentionally absent — it is handled in the multi-char operator
# section of _scan_one_code_token so that "->" (THIN_ARROW) takes priority via
# maximal munch before falling back to MINUS.
_SINGLE_OPS: dict[str, str] = {
    "(": LPAR,
    ")": RPAR,
    "[": LSQB,
    "]": RSQB,
    "{": LBRACE,
    "}": RBRACE,
    ":": COLON,
    ",": COMMA,
    ".": DOT,
    "|": PIPE,
    ";": SEMICOLON,
    "+": PLUS,
    "*": STAR,
    "/": SLASH,
    "@": AT,
}

# JSON escape decoding table (excluding \uXXXX and \$, handled separately)
_JSON_ESCAPES: dict[str, str] = {
    '"': '"',
    "'": "'",
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


# ---------------------------------------------------------------------------
# Triple-quoted template segments (typed union)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _LitSeg:
    """A literal-text segment of a triple-quoted template.

    ``text`` is the raw (pre-dedent) literal text; ``start_pos``/``start_line``/
    ``start_col`` mark the first source character of the segment in the
    normalized text (used to position the synthesised ``STRING_FRAGMENT``).
    """

    text: str
    start_pos: int
    start_line: int
    start_col: int


@dataclass(frozen=True, slots=True)
class _InterpSeg:
    """An interpolation hole of a triple-quoted template.

    ``tokens`` are the code tokens scanned inside ``${...}`` followed by the
    closing ``INTERP_END`` token (already carrying real positions).
    ``start_pos``/``start_line``/``start_col`` mark the ``$`` of ``${``.
    """

    tokens: list[Token]
    start_pos: int
    start_line: int
    start_col: int


# ---------------------------------------------------------------------------
# Scanner state
# ---------------------------------------------------------------------------


class _Scanner:
    """Stateful scanner: processes ``source`` from left to right."""

    def __init__(self, source: str) -> None:
        # Universal-newline normalization: CRLF and lone CR both become LF.
        # All offsets henceforth refer to this normalized text.
        self._src = normalize_newlines(source)
        self._pos = 0
        self._line = 1
        self._col = 1  # 1-based column
        # Offset of the first character of the current line (after newline
        # normalization).  Used to report TAB warnings with a SIMPLE 1-based
        # column (``pos - line_start + 1``) — distinct from ``self._col``, which
        # is tab-EXPANDED after indentation.  Updated only at newline boundaries.
        self._line_start_pos = 0
        # TAB-character advisories accumulated during this single scan (one per
        # ``\t`` in code, indentation, a comment, or an interpolation hole —
        # literal string content is exempt).  The lexer is the sole producer of
        # these; there is no separate TAB scan pass.
        self._tab_warnings: list[Diagnostic] = []
        # True once at least one real (non-layout) token has been emitted; used
        # to suppress the leading ``_NEWLINE`` of comment/blank-only prefixes.
        self._emitted_real = False

    @property
    def tab_warnings(self) -> list[Diagnostic]:
        """TAB advisories collected so far during the scan."""
        return self._tab_warnings

    def _record_tab(self) -> None:
        """Record a TAB advisory for the ``\\t`` at the current scan position.

        Callers invoke this with ``self._pos`` pointing AT the tab character;
        the reported column is the simple (non-expanded) 1-based offset within
        the current line.
        """
        col = self._pos - self._line_start_pos + 1
        self._tab_warnings.append(
            Diagnostic(
                message=(
                    f"TAB character at column {col} is not allowed;"
                    " use spaces for indentation"
                ),
                line=self._line,
                column=col,
                end_line=self._line,
                end_column=col + 1,
                severity="warning",
            )
        )

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _peek(self, offset: int = 0) -> str:
        idx = self._pos + offset
        if idx < len(self._src):
            return self._src[idx]
        return ""

    def _at_end(self) -> bool:
        return self._pos >= len(self._src)

    def _advance(self, *, in_string: bool = False) -> str:
        ch = self._src[self._pos]
        # A TAB inside string-literal content is allowed; everywhere else (code,
        # indentation, comments, interpolation) a literal TAB is advised against.
        if ch == "\t" and not in_string:
            self._record_tab()
        self._pos += 1
        if ch == "\n":
            self._line += 1
            self._col = 1
            self._line_start_pos = self._pos
        else:
            self._col += 1
        return ch

    def _span_here(self) -> SourceSpan:
        return SourceSpan(
            start_line=self._line,
            start_col=self._col,
            end_line=self._line,
            end_col=self._col,
            start_offset=self._pos,
            end_offset=self._pos,
        )

    def _make_token(
        self,
        typ: str,
        value: str,
        start_pos: int,
        start_line: int,
        start_col: int,
    ) -> Token:
        return Token(
            typ,
            value,
            start_pos=start_pos,
            line=start_line,
            column=start_col,
            end_line=self._line,
            end_column=self._col,
            end_pos=self._pos,
        )

    # ------------------------------------------------------------------
    # Indentation measurement
    # ------------------------------------------------------------------

    def _measure_indentation(self) -> int:
        """Return the leading indentation width of the current line.

        Advances past the leading whitespace and any full-line comments,
        but does NOT consume non-whitespace characters.  Returns the
        computed width of the *next real content line* (blank lines and
        comment-only lines are skipped).
        """
        while True:
            col = 0
            saved_pos = self._pos
            saved_line = self._line
            # TAB advisories recorded by THIS probe iteration; dropped if the
            # line turns out to be a whitespace-only EOF tail (restored below
            # and re-scanned in code mode, which records them once).
            saved_warn_count = len(self._tab_warnings)
            # Measure leading horizontal whitespace on this line
            while self._pos < len(self._src) and self._src[self._pos] in (" ", "\t"):
                ch = self._src[self._pos]
                if ch == "\t":
                    col += _TAB_LEN - (col % _TAB_LEN)
                    self._record_tab()
                else:
                    col += 1
                self._pos += 1

            # What's at the current position after whitespace?
            if self._pos >= len(self._src):
                # EOF — no real line follows; restore position and drop the
                # probe's TAB advisories (they re-record on the code-mode rescan).
                self._pos = saved_pos
                self._line = saved_line
                del self._tab_warnings[saved_warn_count:]
                return 0

            ch = self._src[self._pos]
            if ch == "\n":
                # Blank line — skip it and try the next line
                self._pos += 1
                self._line += 1
                self._col = 1
                self._line_start_pos = self._pos
                continue
            if ch == "#":
                # Comment-only line — skip to end of line and try next
                while self._pos < len(self._src) and self._src[self._pos] != "\n":
                    if self._src[self._pos] == "\t":
                        self._record_tab()
                    self._pos += 1
                if self._pos < len(self._src):
                    self._pos += 1
                    self._line += 1
                    self._col = 1
                    self._line_start_pos = self._pos
                continue
            # Real content line found; update column counter
            self._col = col + 1
            return col

    # ------------------------------------------------------------------
    # Escape decoding
    # ------------------------------------------------------------------

    def _decode_escape(self) -> str:
        """Decode a backslash escape; the ``\\`` has already been consumed.

        Returns the decoded character(s).
        Raises :class:`LexError` for unknown escapes.
        """
        esc_line = self._line
        esc_col = self._col
        # The backslash sits one position before the current scan position.
        esc_offset = self._pos - 1
        if self._at_end():
            span = SourceSpan(esc_line, esc_col, esc_line, esc_col, esc_offset, self._pos)
            raise LexError("Unexpected end of input after backslash", span=span)
        ch = self._advance()
        if ch in _JSON_ESCAPES:
            return _JSON_ESCAPES[ch]
        if ch == "$":
            return "$"
        if ch == "u":
            # \uXXXX
            hex_digits = ""
            for _ in range(4):
                if self._at_end():
                    span = SourceSpan(
                        esc_line, esc_col, self._line, self._col, esc_offset, self._pos
                    )
                    raise LexError("Incomplete \\uXXXX escape", span=span)
                d = self._advance()
                if d not in "0123456789abcdefABCDEF":
                    span = SourceSpan(
                        esc_line, esc_col, self._line, self._col, esc_offset, self._pos
                    )
                    raise LexError(
                        f"Invalid hex digit in \\uXXXX escape: {d!r}", span=span
                    )
                hex_digits += d
            return chr(int(hex_digits, 16))
        span = SourceSpan(esc_line, esc_col, self._line, self._col, esc_offset, self._pos)
        raise LexError(f"Unknown escape sequence: \\{ch}", span=span)

    # ------------------------------------------------------------------
    # Template sub-scanner
    # ------------------------------------------------------------------

    def _scan_template(
        self, start_pos: int, start_line: int, start_col: int, quote: str = '"'
    ) -> Iterator[Token]:
        """Scan a template (single- or triple-quoted) starting just after the opening quote.

        *quote* is either ``'"'`` or ``"'"``; it is the delimiter character already
        consumed by the caller.

        Yields:
            ``TEMPLATE_START``, zero or more (``STRING_FRAGMENT`` |
            ``INTERP_START`` … ``INTERP_END``), ``TEMPLATE_END``.
        """
        triple = self._peek() == quote and self._peek(1) == quote
        if triple:
            self._advance()
            self._advance()
        yield self._make_token(TEMPLATE_START, quote, start_pos, start_line, start_col)

        if triple:
            yield from self._scan_triple_template(quote)
        else:
            yield from self._scan_single_template(quote)

    def _scan_single_template(self, quote: str = '"') -> Iterator[Token]:
        """Scan the body of a single-line template delimited by *quote*, yielding tokens."""
        frag_start_pos = self._pos
        frag_start_line = self._line
        frag_start_col = self._col
        buf: list[str] = []

        while True:
            if self._at_end():
                span = self._span_here()
                raise LexError("Unterminated string literal", span=span)
            ch = self._peek()
            if ch == quote:
                # End of template. Emit the literal fragment *before* consuming
                # the closing quote so its end_pos stops at the quote instead of
                # spanning it; an overlap would otherwise duplicate the quote in
                # span consumers such as the REPL syntax highlighter.
                yield self._make_token(
                    STRING_FRAGMENT,
                    "".join(buf),
                    frag_start_pos,
                    frag_start_line,
                    frag_start_col,
                )
                quote_pos, quote_line, quote_col = self._pos, self._line, self._col
                self._advance()
                yield self._make_token(
                    TEMPLATE_END, quote, quote_pos, quote_line, quote_col
                )
                return
            if ch == "\n":
                span = SourceSpan(
                    self._line, self._col, self._line, self._col, self._pos, self._pos
                )
                raise LexError("Unterminated single-line string literal", span=span)
            if ch == "\\":
                self._advance()
                buf.append(self._decode_escape())
            elif ch == "$" and self._peek(1) == "{":
                # Start of interpolation
                interp_pos = self._pos
                interp_line = self._line
                interp_col = self._col
                self._advance()  # consume '$'
                self._advance()  # consume '{'
                yield self._make_token(
                    STRING_FRAGMENT,
                    "".join(buf),
                    frag_start_pos,
                    frag_start_line,
                    frag_start_col,
                )
                yield self._make_token(
                    INTERP_START, "${", interp_pos, interp_line, interp_col
                )
                buf = []
                yield from self._scan_interp_code()
                frag_start_pos = self._pos
                frag_start_line = self._line
                frag_start_col = self._col
            else:
                # Literal string content: a TAB here is allowed (not advised).
                self._advance(in_string=True)
                buf.append(ch)

    def _scan_interp_code(self) -> Iterator[Token]:
        """Scan code tokens inside ``${...}`` up to and including the closing ``}``.

        Tracks nested ``{...}`` so that a dict literal inside the interpolation
        does not prematurely close it.  Yields all code tokens then an
        ``INTERP_END`` token.
        """
        depth = 1
        while True:
            if self._at_end():
                span = self._span_here()
                raise LexError("Unterminated interpolation", span=span)
            # Skip horizontal whitespace
            if self._peek() in (" ", "\t"):
                self._advance()
                continue
            # Newlines are not permitted inside an interpolation in v1.
            if self._peek() == "\n":
                span = SourceSpan(
                    self._line, self._col, self._line, self._col, self._pos, self._pos
                )
                raise LexError(
                    "newline is not allowed inside an interpolation", span=span
                )
            if self._peek() == "{":
                depth += 1
                start_pos = self._pos
                start_line = self._line
                start_col = self._col
                self._advance()
                yield self._make_token(LBRACE, "{", start_pos, start_line, start_col)
                continue
            if self._peek() == "}":
                depth -= 1
                if depth == 0:
                    # Closing interpolation
                    end_pos = self._pos
                    end_line = self._line
                    end_col = self._col
                    self._advance()
                    yield self._make_token(
                        INTERP_END, "}", end_pos, end_line, end_col
                    )
                    return
                start_pos = self._pos
                start_line = self._line
                start_col = self._col
                self._advance()
                yield self._make_token(RBRACE, "}", start_pos, start_line, start_col)
                continue
            # Scan a code token
            yield from self._scan_one_code_token()

    def _scan_triple_template(self, quote: str = '"') -> Iterator[Token]:
        """Scan the body of a triple-quoted template, yielding tokens.

        Triple-quoted dedent rule (§10.1):
        1. Collect the raw content until the closing triple-quote, tracking
           interpolation holes as opaque segments.
        2. Apply the dedent rule to the combined literal skeleton (replacing
           each interpolation hole with a placeholder).
        3. Emit tokens: STRING_FRAGMENT for each literal segment, with
           INTERP_START/inner-tokens/INTERP_END around each hole.

        Interpolation holes occupy their position in the text and are never
        dedented; only surrounding literal whitespace is stripped.

        Positions: the dedent transformation changes fragment *text*, but the
        synthesised tokens are positioned at their original source locations
        (the first source character of each literal segment, the ``$`` of each
        interpolation, the closing triple-quote).  All positions therefore
        point INTO the template's true normalized-source range, never ``None``.
        """
        # Strictly alternating literal/interp segments (always lit-first and
        # lit-last), each carrying its first-source-character position.
        segments: list[_LitSeg | _InterpSeg] = []
        current_lit: list[str] = []
        lit_start_pos = self._pos
        lit_start_line = self._line
        lit_start_col = self._col

        while True:
            if self._at_end():
                span = self._span_here()
                raise LexError("Unterminated triple-quoted string literal", span=span)
            ch = self._peek()
            if ch == quote and self._peek(1) == quote and self._peek(2) == quote:
                # End of triple-quoted string; record the closing-quote position.
                close_pos = self._pos
                close_line = self._line
                close_col = self._col
                self._advance()
                self._advance()
                self._advance()
                segments.append(
                    _LitSeg("".join(current_lit), lit_start_pos, lit_start_line, lit_start_col)
                )
                break
            if ch == "\\":
                self._advance()
                decoded = self._decode_escape()
                current_lit.append(decoded)
            elif ch == "$" and self._peek(1) == "{":
                # Start interpolation; remember the '$' position.
                interp_start_pos = self._pos
                interp_start_line = self._line
                interp_start_col = self._col
                self._advance()  # consume '$'
                self._advance()  # consume '{'
                segments.append(
                    _LitSeg("".join(current_lit), lit_start_pos, lit_start_line, lit_start_col)
                )
                current_lit = []
                interp_tokens = list(self._scan_interp_code())
                segments.append(
                    _InterpSeg(
                        interp_tokens,
                        interp_start_pos,
                        interp_start_line,
                        interp_start_col,
                    )
                )
                lit_start_pos = self._pos
                lit_start_line = self._line
                lit_start_col = self._col
            else:
                # Literal string content: a TAB here is allowed (not advised).
                self._advance(in_string=True)
                current_lit.append(ch)

        # Build combined literal text (literals only — holes contribute zero
        # chars) and record where each literal segment boundary falls within it.
        # We use character-offset arithmetic so that no in-band marker is
        # needed and literal content containing any byte sequence is safe.
        lit_segs = [seg for seg in segments if isinstance(seg, _LitSeg)]
        interp_segs = [seg for seg in segments if isinstance(seg, _InterpSeg)]

        combined = "".join(seg.text for seg in lit_segs)
        # Boundaries: boundary[i] is the start offset of lit_segs[i] in combined.
        boundaries: list[int] = []
        offset = 0
        for seg in lit_segs:
            boundaries.append(offset)
            offset += len(seg.text)

        # Build an indent probe for hole-aware min-indent measurement.
        # Each interpolation hole is replaced by a single non-whitespace
        # placeholder ("X") so that a line whose only non-whitespace content
        # is a hole (e.g. "  ${x}") is treated as non-blank.  The probe is
        # used ONLY for measuring indentation — never for reassembly — so a
        # placeholder collision with literal content is irrelevant.
        indent_probe = "".join(
            seg.text if isinstance(seg, _LitSeg) else "X" for seg in segments
        )
        # Apply the same step-1 (leading-newline drop) that
        # _apply_triple_dedent_with_map applies, so the line split matches.
        probe_body = indent_probe[1:] if indent_probe.startswith("\n") else indent_probe
        probe_min_indent = _compute_min_indent(probe_body.split("\n"))

        dedented, pos_map = _apply_triple_dedent_with_map(combined, probe_min_indent)

        # Use the position map to find where each literal segment starts in dedented.
        def _mapped(pre: int) -> int:
            """Map a pre-dedent offset to its post-dedent offset."""
            return pos_map[pre] if pre < len(pos_map) else len(dedented)

        # Emit STRING_FRAGMENT (+ INTERP_START / inner tokens / INTERP_END) per part.
        for part_idx, lit_seg in enumerate(lit_segs):
            seg_start = _mapped(boundaries[part_idx])
            # End of this literal segment = start of next segment's boundary.
            next_boundary = boundaries[part_idx + 1] if part_idx + 1 < len(boundaries) else offset
            seg_end = _mapped(next_boundary)
            lit_text = dedented[seg_start:seg_end]
            yield Token(
                STRING_FRAGMENT,
                lit_text,
                start_pos=lit_seg.start_pos,
                line=lit_seg.start_line,
                column=lit_seg.start_col,
                end_line=lit_seg.start_line,
                end_column=lit_seg.start_col,
                end_pos=lit_seg.start_pos + len(lit_text),
            )
            if part_idx < len(interp_segs):
                interp_seg = interp_segs[part_idx]
                yield Token(
                    INTERP_START,
                    "${",
                    start_pos=interp_seg.start_pos,
                    line=interp_seg.start_line,
                    column=interp_seg.start_col,
                    end_line=interp_seg.start_line,
                    end_column=interp_seg.start_col + 2,
                    end_pos=interp_seg.start_pos + 2,
                )
                # All inner tokens (already positioned) plus the trailing
                # INTERP_END token, re-yielded with its real positions intact.
                yield from interp_seg.tokens

        yield Token(
            TEMPLATE_END,
            quote,
            start_pos=close_pos,
            line=close_line,
            column=close_col,
            end_line=close_line,
            end_column=close_col + 3,
            end_pos=close_pos + 3,
        )

    # ------------------------------------------------------------------
    # Code token scanning
    # ------------------------------------------------------------------

    def _scan_one_code_token(self) -> Iterator[Token]:
        """Scan exactly one code-mode token from the current position."""
        start_pos = self._pos
        start_line = self._line
        start_col = self._col
        ch = self._advance()

        # Identifiers and keywords: start with a (Unicode) letter or ``_``,
        # then greedily consume every character that is not whitespace and not
        # an operator/punctuator delimiter (see ``_IDENT_STOP``).  This admits
        # arbitrary Unicode letters/digits as well as the symbol characters
        # ``-``, ``?``, ``!``, so names like ``ask-prompt`` or ``do-it-now!``
        # scan as a single token.  Operator tokens (``->``, ``=>``, ``!=``,
        # ``<=``, ``>=``, field access ``.``, etc.) still lex as operators when
        # they appear as standalone, whitespace-delimited tokens.
        if ch.isalpha() or ch == "_":
            while not self._at_end() and self._peek() not in _IDENT_STOP:
                self._advance()
            word = self._src[start_pos : self._pos]
            if word in KEYWORDS:
                typ = word
            else:
                typ = NAME
            yield self._make_token(typ, word, start_pos, start_line, start_col)
            return

        # Numbers — ASCII digits only.  ``str.isdigit()`` admits non-ASCII
        # digit characters (e.g. fullwidth ``２`` or Arabic-Indic ``٠``); the
        # language defines numeric literals as ``[0-9]+`` / ``[0-9]+.[0-9]+``, so
        # the scan is restricted to the ASCII range.  Non-ASCII digits therefore
        # fall through to the ``Unexpected character`` path (or, when they follow
        # a letter, become part of a greedy identifier — see _IDENT_STOP).
        if _is_ascii_digit(ch):
            while not self._at_end() and _is_ascii_digit(self._peek()):
                self._advance()
            if self._peek() == "." and (
                self._pos + 1 < len(self._src) and _is_ascii_digit(self._src[self._pos + 1])
            ):
                self._advance()  # '.'
                while not self._at_end() and _is_ascii_digit(self._peek()):
                    self._advance()
                value = self._src[start_pos:self._pos]
                yield self._make_token(DECIMAL, value, start_pos, start_line, start_col)
            else:
                value = self._src[start_pos:self._pos]
                yield self._make_token(INT, value, start_pos, start_line, start_col)
            return

        # Strings/templates
        if ch == '"':
            yield from self._scan_template(start_pos, start_line, start_col, quote='"')
            return
        if ch == "'":
            yield from self._scan_template(start_pos, start_line, start_col, quote="'")
            return

        # Multi-char operators (maximal munch)
        if ch == "=" and self._peek() == ">":
            self._advance()
            yield self._make_token(ARROW, "=>", start_pos, start_line, start_col)
            return
        if ch == "=" and self._peek() == "=":
            self._advance()
            yield self._make_token(EQ_EQ, "==", start_pos, start_line, start_col)
            return
        if ch == "=":
            yield self._make_token(EQ, "=", start_pos, start_line, start_col)
            return
        if ch == "!" and self._peek() == "=":
            self._advance()
            yield self._make_token(NEQ, "!=", start_pos, start_line, start_col)
            return
        if ch == "<" and self._peek() == "=":
            self._advance()
            yield self._make_token(LE, "<=", start_pos, start_line, start_col)
            return
        if ch == "<":
            yield self._make_token(LT, "<", start_pos, start_line, start_col)
            return
        if ch == ">" and self._peek() == "=":
            self._advance()
            yield self._make_token(GE, ">=", start_pos, start_line, start_col)
            return
        if ch == ">":
            yield self._make_token(GT, ">", start_pos, start_line, start_col)
            return
        # "->" is THIN_ARROW (v2 function/return type); bare "-" is MINUS.
        # Maximal munch: check the next character before deciding.
        if ch == "-" and self._peek() == ">":
            self._advance()
            yield self._make_token(THIN_ARROW, "->", start_pos, start_line, start_col)
            return
        if ch == "-":
            yield self._make_token(MINUS, "-", start_pos, start_line, start_col)
            return

        # ":=" is destructive assignment and "::" introduces typed calls.
        if ch == ":" and self._peek() == "=":
            self._advance()
            yield self._make_token(ASSIGN, ":=", start_pos, start_line, start_col)
            return
        # "::" is DCOLON (type-argument introducer for typed calls, e.g.
        # ask-request::[Review](...)); bare ":" is COLON.  Maximal munch:
        # check the next character before falling back to the single-char op.
        if ch == ":" and self._peek() == ":":
            self._advance()
            yield self._make_token(DCOLON, "::", start_pos, start_line, start_col)
            return
        # Single-char operators
        if ch in _SINGLE_OPS:
            yield self._make_token(_SINGLE_OPS[ch], ch, start_pos, start_line, start_col)
            return

        # Unknown character
        span = SourceSpan(start_line, start_col, self._line, self._col, start_pos, self._pos)
        raise LexError(f"Unexpected character: {ch!r}", span=span)

    # ------------------------------------------------------------------
    # Main scanning loop (CODE mode)
    # ------------------------------------------------------------------

    def scan(self) -> Iterator[Token]:
        """Yield all tokens in CODE mode (the top-level entry point).

        Produces ``_NEWLINE`` tokens that carry the next real line's indentation
        width as their value.  The layout filter converts these into
        ``_INDENT``/``_DEDENT``/``_NEWLINE`` tokens.
        """
        while not self._at_end():
            ch = self._peek()

            # Horizontal whitespace — skip
            if ch in (" ", "\t"):
                self._advance()
                continue

            # Comments — skip to end of line
            if ch == "#":
                while not self._at_end() and self._peek() != "\n":
                    self._advance()
                continue

            # Newline — emit _NEWLINE with next real line's indentation
            if ch == "\n":
                newline_offset = self._pos  # position of the '\n' itself
                newline_line = self._line
                newline_col = self._col
                self._advance()  # consume the newline
                # Measure indentation of next real line
                indent_width = self._measure_indentation()
                # Suppress a leading _NEWLINE: the grammar's block_stmts cannot
                # consume a _NEWLINE before any real statement token (Python
                # tokenizer style).  Blank/comment-only prefixes thus emit none.
                if not self._emitted_real:
                    continue
                yield Token(
                    NEWLINE,
                    str(indent_width),
                    start_pos=newline_offset,
                    line=newline_line,
                    column=newline_col,
                    end_line=newline_line,
                    end_column=newline_col + 1,
                    end_pos=newline_offset + 1,
                )
                continue

            # All other tokens
            self._emitted_real = True
            yield from self._scan_one_code_token()


# ---------------------------------------------------------------------------
# Triple-quoted dedent rule
# ---------------------------------------------------------------------------


def _compute_min_indent(lines: list[str]) -> int:
    """Return the minimum leading whitespace count of non-blank lines."""
    min_ind: int | None = None
    for line in lines:
        if not line.strip():
            continue  # blank lines don't contribute
        indent = len(line) - len(line.lstrip(" \t"))
        if min_ind is None or indent < min_ind:
            min_ind = indent
    return min_ind if min_ind is not None else 0


def _apply_triple_dedent_with_map(text: str, min_indent: int) -> tuple[str, list[int]]:
    """Apply the triple-quoted dedent rule and return a position map.

    Rule:
    1. Remove one leading ``\\n`` if present.
    2. Strip the minimum common indentation of all non-blank lines.
    3. Remove one trailing ``\\n`` if present (after dedent).

    This order (dedent after leading-strip, trailing-strip after dedent)
    produces the natural result for the common pattern where the closing
    delimiter's indentation defines the common indent level.

    *min_indent* is supplied by the triple-template scanner as a hole-aware
    value measured from a probe string that treats each interpolation hole as
    a non-whitespace character, preventing hole-only lines from being
    classified as blank (it must NOT be computed from *text*, which has the
    holes removed).

    Returns ``(dedented, pos_map)`` where ``pos_map[i]`` is the index in
    *dedented* that corresponds to position *i* in *text*.  The map has
    length ``len(text) + 1``; the extra entry maps the past-the-end position.
    Removed positions map to the output index of the next kept character, so
    callers can locate boundaries from the original string within the result
    without relying on any in-band sentinel marker.
    """
    kept = [True] * len(text)

    # Step 1: drop one leading newline.
    pre_start = 0
    if text.startswith("\n"):
        kept[0] = False
        pre_start = 1

    # Step 2: strip the minimum common indentation.  Every character in
    # line[:min_indent] is whitespace: non-blank lines carry at least
    # min_indent leading whitespace by construction, and blank lines are
    # whitespace throughout.  When *min_indent* is supplied by the caller
    # (hole-aware measurement), a hole line's literal prefix in *text* may
    # be shorter than min_indent (the hole consumed part of the line); the
    # ``min(min_indent, len(line))`` guard already handles that safely.
    lines = text[pre_start:].split("\n")
    pos = pre_start
    for line in lines:
        for offset in range(min(min_indent, len(line))):
            kept[pos + offset] = False
        pos += len(line) + 1  # +1 for the '\n' separator (or past-end)

    # Step 3: drop one trailing newline (after dedent normalises the content).
    kept_indices = [i for i, k in enumerate(kept) if k]
    if kept_indices and text[kept_indices[-1]] == "\n":
        kept[kept_indices[-1]] = False

    # Build pos_map: pos_map[i] = output index of input position i.
    pos_map: list[int] = []
    out = 0
    for k in kept:
        pos_map.append(out)
        if k:
            out += 1
    pos_map.append(out)  # past-the-end entry

    dedented = "".join(ch for i, ch in enumerate(text) if kept[i])
    return dedented, pos_map


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def scan(source: str) -> Iterator[Token]:
    """Yield raw tokens from *source* (code mode, with ``_NEWLINE`` signals)."""
    return _Scanner(source).scan()


def lex_tab_warnings(source: str) -> list[Diagnostic]:
    """Return a ``Diagnostic`` warning for every TAB character in *source*.

    Drives the real lexer scan (the single source of TAB-detection truth) and
    returns the advisories it accumulated — there is NO separate TAB scan pass.
    One warning is emitted per ``\\t`` in code, indentation, a comment, or an
    interpolation hole (literal string content is exempt), with a simple 1-based
    column.

    The full scan over *source* must succeed (lex-valid input); callers that may
    pass lex-invalid source should instead read advisories from the parse path
    (``tab_warning_collector``), which surfaces whatever was collected.
    """
    scanner = _Scanner(source)
    for _ in scanner.scan():
        pass
    return scanner.tab_warnings
