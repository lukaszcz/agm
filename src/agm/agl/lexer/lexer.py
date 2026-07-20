"""AglLexer: Lark custom lexer for the AgL DSL.

Composes :mod:`~agm.agl.lexer.scanner` (raw tokens) with
:mod:`~agm.agl.lexer.layout` (INDENT/DEDENT injection) to produce the full
token stream required by the Lark LALR parser.

The lexer is wired as::

    Lark(grammar, parser="lalr", lexer=AglLexer, propagate_positions=True)

The ``lex`` method receives ``lexer_state`` and ``parser_state`` from Lark.
We read the source text from ``lexer_state.text`` and yield
:class:`lark.lexer.Token` objects with full position information.

Grammar terminal names
----------------------
Lark creates uppercase terminal names from grammar string literals (e.g.
``"pass"`` → terminal ``PASS``).  Because the raw scanner emits lowercase
keyword types (``"pass"``, ``"let"``, etc.), ``lex()`` applies a one-to-one
remapping: ``GRAMMAR_TOKEN_REMAP`` maps each lowercase keyword type to its
uppercase equivalent so the LALR parser can look up the right parse-table
entry.

The public :func:`~agm.agl.lexer.tokenize` helper goes through
``layout(scan())`` directly and preserves the lowercase types that tests
and downstream tooling rely on.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

from lark.lexer import Lexer, LexerState, Token

from agm.agl.diagnostics import Diagnostic, SourceSpan
from agm.agl.lexer.layout import layout
from agm.agl.lexer.scanner import _Scanner
from agm.agl.lexer.tokens import (
    CALL_LBRACE,
    DCOLON,
    DO_LSQB,
    EXPORT,
    GRAMMAR_TOKEN_REMAP,
    HIDING,
    IMPORT,
    INDEX_LSQB,
    INT,
    LBRACE,
    LSQB,
    MODPATH,
    MODQUAL,
    NAME,
    OP_NAME,
    OPEN,
    PRIVATE,
    RSQB,
    SLASH,
    TYPEARG_LSQB,
    USING,
    WILDCARD,
)
from agm.agl.syntax.advisories import SpacedQualifier

_INDEX_PREDECESSORS = frozenset(
    {
        "NAME",
        OP_NAME,
        INT,
        "DECIMAL",
        "TRUE",
        "FALSE",
        "NULL",
        "TEMPLATE_END",
        "RPAR",
        RSQB,
        "RBRACE",
    }
)

# Ambient sink for TAB advisories produced during a Lark-driven parse.  The
# lexer scans the source exactly once (no separate TAB pass); when a sink is
# active, ``AglLexer.lex`` deposits the scan's TAB advisories into it so the
# caller (e.g. ``PipelineDriver.prepare``) can surface them alongside parse
# diagnostics.  A ``ContextVar`` keeps nested/reentrant parses isolated.
_TAB_WARNING_SINK: contextvars.ContextVar[list[Diagnostic] | None] = contextvars.ContextVar(
    "agl_tab_warning_sink", default=None
)


@contextmanager
def tab_warning_collector() -> Iterator[list[Diagnostic]]:
    """Collect TAB advisories produced by parses within the ``with`` block.

    Yields a list that ``AglLexer.lex`` appends to as it scans.  The list is
    populated even when the parse fails (the scan runs to completion before the
    grammar is consulted), so callers get every TAB advisory on every path.
    """
    sink: list[Diagnostic] = []
    token = _TAB_WARNING_SINK.set(sink)
    try:
        yield sink
    finally:
        _TAB_WARNING_SINK.reset(token)


# Ambient sink for spaced-qualifier advisories, mirroring the TAB sink above.
# ``_merge_modqual`` deposits into it as it walks the token stream, so the
# advisories are complete whether or not the grammar later accepts the program.
_SPACED_QUALIFIER_SINK: contextvars.ContextVar[list[SpacedQualifier] | None] = (
    contextvars.ContextVar("agl_spaced_qualifier_sink", default=None)
)


@contextmanager
def spaced_qualifier_collector() -> Iterator[list[SpacedQualifier]]:
    """Collect spaced-qualifier advisories produced by lexing within the ``with`` block.

    Yields a list that the module-qualifier merge pass appends to.  Callers keep
    the result alongside the parsed module so later passes can explain a
    reference that whitespace turned into an unrelated expression.
    """
    sink: list[SpacedQualifier] = []
    token = _SPACED_QUALIFIER_SINK.set(sink)
    try:
        yield sink
    finally:
        _SPACED_QUALIFIER_SINK.reset(token)


def _remap(tokens: Iterator[Token]) -> Iterator[Token]:
    """Remap lowercase keyword token types to the uppercase names the grammar expects."""
    for tok in tokens:
        mapped = GRAMMAR_TOKEN_REMAP.get(tok.type)
        if mapped is not None:
            tok = Token(
                mapped,
                str(tok),
                start_pos=tok.start_pos,
                line=tok.line,
                column=tok.column,
                end_line=tok.end_line,
                end_column=tok.end_column,
                end_pos=tok.end_pos,
            )
        yield tok


_ITEM_START_TYPES = frozenset({"_NEWLINE", "_INDENT", "_DEDENT", "SEMICOLON"})


def _retype(tok: Token, new_type: str) -> Token:
    """Return a copy of *tok* with a new token type, preserving value and span."""
    return Token(
        new_type,
        str(tok),
        start_pos=tok.start_pos,
        line=tok.line,
        column=tok.column,
        end_line=tok.end_line,
        end_column=tok.end_column,
        end_pos=tok.end_pos,
    )


def _promote_soft_keywords(tokens: list[Token]) -> list[Token]:
    """Contextually promote soft keywords in the post-layout token stream.

    Rules:
    - 'open' → OPEN only at item-start and only directly before 'import'.
    - 'import' → IMPORT at item-start, or immediately after OPEN.
    - 'private' → PRIVATE and 'export' → EXPORT at item-start.
    - 'using' → USING and 'hiding' → HIDING within import or export declarations.
    """
    result: list[Token] = []
    in_module_header = False
    prev_type: str | None = None  # None means start-of-stream

    for index, tok in enumerate(tokens):
        tt = tok.type
        tv = str(tok)

        # Track the module-header window: close on line/stmt terminators
        if tt in ("_NEWLINE", "_INDENT", "_DEDENT", "SEMICOLON"):
            in_module_header = False

        if tt == NAME:
            at_item_start = prev_type is None or prev_type in _ITEM_START_TYPES
            if (
                tv == "open"
                and at_item_start
                and index + 1 < len(tokens)
                and tokens[index + 1].type == NAME
                and str(tokens[index + 1]) == "import"
            ):
                tok = _retype(tok, OPEN)
            elif tv == "import" and (at_item_start or prev_type == OPEN):
                tok = _retype(tok, IMPORT)
                in_module_header = True
            elif tv == "export" and at_item_start:
                tok = _retype(tok, EXPORT)
                in_module_header = True
            elif tv == "private" and at_item_start:
                tok = _retype(tok, PRIVATE)
            elif in_module_header:
                if tv == "using":
                    tok = _retype(tok, USING)
                elif tv == "hiding":
                    tok = _retype(tok, HIDING)

        result.append(tok)
        prev_type = tok.type

    return result


def _merge_modpath(tokens: list[Token]) -> list[Token]:
    """Merge module-header paths into single MODPATH tokens.

    Pattern: immediately following an IMPORT or EXPORT token, consume
    NAME (SLASH NAME)* into a single MODPATH token whose value is the slash
    path (e.g. "foo/bar", "utils"). Every pair in the run is adjacent in the
    source, the rule module qualifiers obey: ``a/b`` is a path, ``a / b`` is
    division, so a spaced header separator ends the path and fails to parse.

    This keeps slash-separated module headers distinct from division in the
    expression grammar. By merging the path in the lexer, the grammar sees a
    single MODPATH token rather than the raw NAME SLASH ... sequence.
    """
    result: list[Token] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok.type in (IMPORT, EXPORT) and i + 1 < n and tokens[i + 1].type == NAME:
            result.append(tok)
            i += 1
            # Absorb NAME (SLASH NAME)*. Module path segments may begin with
            # either lowercase or uppercase letters.
            j = i
            last_seg = tokens[j]
            seg_parts: list[str] = [str(last_seg)]
            j += 1
            while (
                j + 1 < n
                and tokens[j].type == SLASH
                and tokens[j + 1].type == NAME
                and last_seg.end_pos == tokens[j].start_pos
                and tokens[j].end_pos == tokens[j + 1].start_pos
            ):
                last_seg = tokens[j + 1]
                seg_parts.append(str(last_seg))
                j += 2
            modpath_value = "/".join(seg_parts)
            first_tok = tokens[i]
            last_tok = tokens[j - 1]
            merged = Token(
                MODPATH,
                modpath_value,
                start_pos=first_tok.start_pos,
                line=first_tok.line,
                column=first_tok.column,
                end_line=last_tok.end_line,
                end_column=last_tok.end_column,
                end_pos=last_tok.end_pos,
            )
            result.append(merged)
            # A byte-adjacent ``/*`` is the wildcard tail. The raw scanner makes
            # it an OP_NAME; retype it as the grammar's single WILDCARD token.
            # Whitespace anywhere in the tail leaves the operator alone, so the
            # header fails to parse — the same adjacency rule module qualifiers
            # obey.
            i = j
            if (
                i < n
                and tokens[i].type == OP_NAME
                and str(tokens[i]) == "/*"
                and tokens[i].start_pos == merged.end_pos
            ):
                tail = tokens[i]
                result.append(
                    Token(
                        WILDCARD,
                        "/*",
                        start_pos=tail.start_pos,
                        line=tail.line,
                        column=tail.column,
                        end_line=tail.end_line,
                        end_column=tail.end_column,
                        end_pos=tail.end_pos,
                    )
                )
                i += 1
            continue
        result.append(tok)
        i += 1
    return result


def _member_reference(
    tokens: list[Token], dcolon_index: int, source: str
) -> tuple[str, str, bool] | None:
    """Describe the reference following a ``::`` as ``(member, text, type_qualified)``.

    Consumes a ``NAME``, an optional balanced ``[...]`` type-argument group, and
    an optional ``:: NAME`` variant tail, then slices the source across the run.
    Returns ``None`` when no name follows the ``::`` and there is nothing to
    quote back to the author.
    """
    n = len(tokens)
    index = dcolon_index + 1
    if index >= n or tokens[index].type != NAME:
        return None
    member = str(tokens[index])
    start = tokens[index].start_pos
    end = tokens[index].end_pos
    index += 1
    if index < n and tokens[index].type == LSQB:
        depth = 0
        while index < n:
            if tokens[index].type == LSQB:
                depth += 1
            elif tokens[index].type == RSQB:
                depth -= 1
                if depth == 0:
                    end = tokens[index].end_pos
                    index += 1
                    break
            index += 1
    type_qualified = (
        index + 1 < n and tokens[index].type == DCOLON and tokens[index + 1].type == NAME
    )
    if type_qualified:
        end = tokens[index + 1].end_pos
    return member, source[start:end], type_qualified


def _token_span(tok: Token) -> SourceSpan:
    """Build a :class:`SourceSpan` from a token's position fields.

    The span carries no ``SourceId``; consumers stamp their own module identity
    onto it, since the lexer never learns which source it is scanning.
    """
    line = tok.line if tok.line is not None else 1
    col = tok.column if tok.column is not None else 1
    start = tok.start_pos if tok.start_pos is not None else 0
    return SourceSpan(
        start_line=line,
        start_col=col,
        end_line=tok.end_line if tok.end_line is not None else line,
        end_col=tok.end_column if tok.end_column is not None else col + len(str(tok)),
        start_offset=start,
        end_offset=tok.end_pos if tok.end_pos is not None else start + len(str(tok)),
    )


def _record_spaced_qualifier(
    tokens: list[Token],
    dcolon_index: int,
    run_start: Token,
    segments: list[str],
    *,
    source: str,
    seen: set[int],
) -> None:
    """Deposit one spaced-qualifier advisory into the ambient sink, if any is active.

    The first run recorded for a given ``::`` wins.  ``_merge_modqual`` scans
    left to right, so that is the longest run reaching the ``::`` — the whole
    route rather than one of its suffixes.
    """
    sink = _SPACED_QUALIFIER_SINK.get()
    dcolon = tokens[dcolon_index]
    dcolon_offset = dcolon.start_pos
    assert dcolon_offset is not None
    if sink is None or dcolon_offset in seen:
        return
    reference = _member_reference(tokens, dcolon_index, source)
    if reference is None:
        return
    member, member_text, type_qualified = reference
    seen.add(dcolon_offset)
    run_start_offset = run_start.start_pos
    assert run_start_offset is not None
    sink.append(
        SpacedQualifier(
            segments=tuple(segments),
            anchored=run_start.type == SLASH,
            run_start_offset=run_start_offset,
            dcolon_span=_token_span(dcolon),
            member=member,
            member_text=member_text,
            type_qualified=type_qualified,
        )
    )


def _merge_modqual(tokens: list[Token], source: str) -> list[Token]:
    """Merge byte-adjacent slash-qualified prefixes into ``MODQUAL`` tokens.

    A qualifier is ``[SLASH] NAME (SLASH NAME)* DCOLON`` with every pair in
    that run adjacent in the source. The token value retains its optional
    leading slash so the AST builder can distinguish anchored references.
    ``NAME DCOLON LSQB`` remains unmerged for typed calls.

    A run whose only defect is whitespace before the ``::`` emits its tokens
    unchanged — the grammar must not see a qualifier there — but is recorded as
    a :class:`~agm.agl.syntax.advisories.SpacedQualifier` advisory so a later
    pass can explain the mis-parse.
    """
    result: list[Token] = []
    seen_dcolons: set[int] = set()
    i = 0
    n = len(tokens)
    while i < n:
        start = tokens[i]
        name_index = i
        if start.type == SLASH:
            name_index += 1
            if (
                name_index >= n
                or tokens[name_index].type != NAME
                or start.end_pos != tokens[name_index].start_pos
            ):
                result.append(start)
                i += 1
                continue
        elif start.type != NAME:
            result.append(start)
            i += 1
            continue

        last_name = tokens[name_index]
        segments = [str(last_name)]
        j = name_index + 1
        while (
            j + 1 < n
            and tokens[j].type == SLASH
            and tokens[j + 1].type == NAME
            and last_name.end_pos == tokens[j].start_pos
            and tokens[j].end_pos == tokens[j + 1].start_pos
        ):
            last_name = tokens[j + 1]
            segments.append(str(last_name))
            j += 2

        if j < n and tokens[j].type == DCOLON and last_name.end_pos != tokens[j].start_pos:
            _record_spaced_qualifier(
                tokens, j, start, segments, source=source, seen=seen_dcolons
            )

        if (
            j < n
            and tokens[j].type == DCOLON
            and last_name.end_pos == tokens[j].start_pos
            and (j + 1 >= n or tokens[j + 1].type != LSQB)
        ):
            qualifier_value = "/".join(segments)
            if start.type == SLASH:
                qualifier_value = "/" + qualifier_value
            last_tok = tokens[j]
            result.append(
                Token(
                    MODQUAL,
                    qualifier_value,
                    start_pos=start.start_pos,
                    line=start.line,
                    column=start.column,
                    end_line=last_tok.end_line,
                    end_column=last_tok.end_column,
                    end_pos=last_tok.end_pos,
                )
            )
            i = j + 1
            continue

        result.append(start)
        i += 1
    return result


def apply_module_passes(tokens: list[Token], source: str) -> list[Token]:
    """Apply soft-keyword promotion, import path merging, and module-qualifier merging."""
    return _merge_modqual(_merge_modpath(_promote_soft_keywords(tokens)), source)


def _remap_adjacent_brackets(tokens: list[Token]) -> list[Token]:
    """Turn adjacent expression brackets/braces into parser-only suffix tokens.

    A ``[`` immediately following the ``do`` keyword opens a loop bound; it is
    retagged ``DO_LSQB`` so the LALR grammar can tell ``do[expr]`` (the bound)
    apart from a ``do`` body that starts with a list literal — without that
    distinct terminal the optional ``loop_bound`` and a list-literal body both
    begin with ``LSQB``, which is the conflict this resolves.  Whitespace
    between ``do`` and ``[`` is allowed (``do [n]`` works); a newline is not,
    because layout inserts a token between them.
    """
    result: list[Token] = []
    previous: Token | None = None
    for tok in tokens:
        if tok.type == LSQB and previous is not None and previous.type == "DO":
            tok = _retype(tok, DO_LSQB)
        elif (
            tok.type == LSQB
            and previous is not None
            and previous.type in _INDEX_PREDECESSORS
            and previous.end_pos == tok.start_pos
        ):
            tok = _retype(tok, INDEX_LSQB)
        elif (
            tok.type == LBRACE
            and previous is not None
            and previous.type in _INDEX_PREDECESSORS
            and previous.end_pos == tok.start_pos
        ):
            tok = _retype(tok, CALL_LBRACE)
        result.append(tok)
        previous = tok
    return result


def _mark_typearg_lsqb(tokens: list[Token]) -> list[Token]:
    """Retag ``Type[T]::Ctor`` opening brackets for the parser.

    Only the outer ``[`` immediately adjacent to a preceding ``NAME`` is
    retagged, and only when its matching ``]`` is followed by ``::`` and then a
    non-``[`` token.  The guard preserves value-position type application such
    as ``xs[i]::[T]``.
    """
    result = list(tokens)
    n = len(result)
    open_types = {LSQB, INDEX_LSQB, TYPEARG_LSQB, DO_LSQB}
    for i, tok in enumerate(result):
        if tok.type != INDEX_LSQB:
            continue
        if i == 0 or result[i - 1].type != NAME:
            continue
        depth = 1
        j = i + 1
        while j < n:
            tt = result[j].type
            if tt in open_types:
                depth += 1
            elif tt == RSQB:
                depth -= 1
                if depth == 0:
                    after = result[j + 1].type if j + 1 < n else None
                    after_next = result[j + 2].type if j + 2 < n else None
                    if after == DCOLON and after_next != LSQB:
                        result[i] = _retype(tok, TYPEARG_LSQB)
                    break
            j += 1
    return result


class AglLexer(Lexer):
    """Custom Lark lexer for AgL.

    Accepted by the Lark parser via ``lexer=AglLexer``; ``lexer_conf`` is
    received but not used (the grammar's terminal regex patterns are not needed
    because we generate all tokens ourselves).

    ``__future_interface__ = 1`` tells Lark to call ``lex(lexer_state,
    parser_state)`` directly, which matches the method signature.  Without
    this, Lark wraps the lexer with the older interface and calls ``lex(text)``
    — the wrong arity.
    """

    __future_interface__ = 1

    def __init__(self, lexer_conf: object) -> None:
        # lexer_conf is accepted but unused; the scanner handles all tokenization.
        pass

    def lex(self, lexer_state: LexerState, parser_state: object) -> Iterator[Token]:
        """Tokenize ``lexer_state.text`` and yield Lark tokens.

        Keyword token types are remapped from lowercase (scanner convention)
        to the uppercase names expected by the Lark grammar.
        """
        # lexer_state.text may be a str or a lark.lexer.TextSlice; extract
        # the raw string for the scanner.
        raw = lexer_state.text
        if isinstance(raw, str):
            source = raw
        else:
            # TextSlice: use the underlying .text attribute.
            source = str(raw.text) if hasattr(raw, "text") else str(raw)
        # Drive the scan to completion up front (materialized) so the single
        # lex pass records EVERY TAB advisory before the grammar is consulted —
        # the advisories are then complete even if the parse later fails.  The
        # ``finally`` deposits whatever was collected, including on a LexError.
        scanner = _Scanner(source)
        try:
            after_remap = list(_remap(layout(scanner.scan())))
            tokens = _mark_typearg_lsqb(
                _remap_adjacent_brackets(apply_module_passes(after_remap, source))
            )
        finally:
            sink = _TAB_WARNING_SINK.get()
            if sink is not None:
                sink.extend(scanner.tab_warnings)
        return iter(tokens)
