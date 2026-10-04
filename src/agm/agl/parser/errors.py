"""Parse-layer ``AglSyntaxError`` construction with SourceSpan and friendly message.

This module maps Lark's ``UnexpectedToken`` (the only error the LALR parser
raises over the AgL lexer) and lexer ``LexError`` to ``AglSyntaxError``
carrying a ``SourceSpan`` and a user-facing message.

Special cases:
- If the unexpected token is any comparison operator (``==``, ``!=``, ``<``,
  ``<=``, ``>``, ``>=``) and that operator is NOT in the expected set, the
  parser has already consumed one complete comparison expression and a second
  one was chained, which is non-associative in AgL.  A targeted
  "comparisons are non-associative; parenthesize" message is emitted instead of
  the generic "Unexpected token" fallback.
- The fallback quotes the unexpected token's spelling, so a token that has none
  is named instead: end of input, end of block, and indentation all reach the
  parser as zero-width layout tokens.
- Every message produced from a Lark exception (not a ``LexError``) gets a
  uniform final pass: a spacing hint is appended when a NAME token ending in
  ``$`` appears on the offending token's line before it, e.g. ``exec$ date``
  lexes as one NAME, not ``exec`` applied to a ``$ date`` verbatim literal;
  and a piping hint is appended when the offending token is a `$` literal
  opener immediately preceded by two operand-ending tokens (a further
  juxtaposed argument, e.g. ``print exec $ date``), since juxtaposition never
  chains.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Sequence

from lark import Tree

from agm.agl.diagnostics import AglSyntaxError as AglSyntaxError
from agm.agl.diagnostics import dollar_spacing_hint, piping_hint
from agm.agl.lexer.operators import OPERAND_END_TYPES, scanner_token_type
from agm.agl.lexer.positions import token_span
from agm.agl.lexer.tokens import NAME, VERBATIM_END, VERBATIM_START
from agm.agl.syntax.spans import SourceSpan

if TYPE_CHECKING:
    from lark.exceptions import UnexpectedToken
    from lark.lexer import Token

# All comparison operator token types (mirrors tokens.py).  Equality is ``==``
# (``EQ_EQ``); ``=`` (``EQ``) is a binder / named-arg separator, not a comparison.
_CMP_OPS: frozenset[str] = frozenset({"EQ_EQ", "NEQ", "LT", "LE", "GT", "GE"})

# Compound forms (``if`` / ``case`` / ``try``) whose statement spelling is only
# valid in an indented block (a *suite*), never inline after ``=>``, ``until``,
# or in a ``case`` *expression* branch.  When
# one of these tokens is the unexpected token, the parser was at a position
# where the grammar's bar-safe inline forms forbid a nested compound statement.
_INLINE_BLOCKED: frozenset[str] = frozenset({"IF", "CASE", "TRY"})

# Lark terminal names that begin a *statement* (a suite element).  When any of
# these is in the expected set, the parser was expecting an indented block /
# statement position — so a blocked compound there should be written as an
# indented block (a suite), not parenthesized as an expression.  ``_INDENT``
# itself signals "a suite may begin here".
_STMT_STARTERS: frozenset[str] = frozenset(
    {
        "LET",
        "VAR",
        "SET",
        "PASS",
        "PRINT",
        "RAISE",
        "DO",
        "INPUT",
        "ENUM",
        "RECORD",
        "TYPE",
        "_INDENT",
    }
)

_ELSE_BEFORE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])else\s*$")
# A genuine placeholder token (`?` or `?N`) starts a fresh token, so it is only
# ever preceded by whitespace, a structural delimiter (see the lexer's
# ``_IDENT_STOP``), or the start of input.  The negative lookbehind excludes any
# identifier-body character, so a name that merely ends in ``?`` (predicate names
# like ``empty?`` or the ``as?`` keyword) does not masquerade as a placeholder.
_PLACEHOLDER_BEFORE_TOKEN_RE = re.compile(r"(?<![^\s(){}\[\]:,.|;/@=])\?[0-9]*\s*$")

# Terminals that end an item.  When one is expected, the parser had a complete
# item in hand, so an unexpected indent is stray rather than a missing body.
_ITEM_ENDERS: frozenset[str] = frozenset({"$END", "_DEDENT", "_NEWLINE", "SEMICOLON"})

# Terminals that carry no spelling: quoting their value would print ``''``.  The
# LALR parser reports a premature end of input as an unexpected ``$END`` rather
# than raising Lark's own ``UnexpectedEOF``, and a source that runs out inside an
# indented block reaches ``_DEDENT`` first.  Layout tokens with a message of
# their own are handled ahead of this fallback naming.
_ZERO_WIDTH_TOKEN_NAMES: dict[str, str] = {
    "$END": "end of input",
    "_DEDENT": "end of block",
    "_INDENT": "indentation",
}


# An inline `=>` body is a single item: no binders, no `;` sequence.  Both are
# legal once the body is parenthesized or written as an indented block, so the
# diagnostic names those two fixes rather than the grammar rule.
_INLINE_BODY_OPENERS: frozenset[str] = frozenset({"let", "var"})

# Tokens that end a `try` body.  Arriving at one while `catch` is still
# expected means the `try` has no `catch` clause of its own.
_TRY_BODY_ENDERS: frozenset[str] = frozenset({"_NEWLINE", "_DEDENT", "$END"})


def _end_of_source_span(source_text: str) -> SourceSpan:
    """Span pinpointing the end of *source_text*."""
    offset = len(source_text)
    last_newline = source_text.rfind("\n")
    line = source_text.count("\n") + 1
    col = offset - last_newline
    return SourceSpan(
        start_line=line,
        start_col=col,
        end_line=line,
        end_col=col + 1,
        start_offset=offset,
        end_offset=offset + 1,
    )


def _make_chained_comparison_error(span: SourceSpan) -> AglSyntaxError:
    """Targeted diagnostic for chained comparisons.

    All comparison operators (``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``) are
    non-associative in AgL: ``x == y == z``, ``1 < 2 < 3``, ``a <= b != c`` are
    all parse errors.  When the parser sees a comparison operator as the
    *unexpected* token AND that operator is absent from the *expected* set, a
    full comparison expression was already consumed and a second was chained —
    the friendly message below is emitted instead of the generic fallback.
    """
    return AglSyntaxError(
        "Comparisons are non-associative; parenthesize explicitly, e.g. `(x == y) == z`.",
        span=span,
    )


def _make_inline_compound_error(
    keyword: str, span: SourceSpan, *, stmt_context: bool
) -> AglSyntaxError:
    """Targeted diagnostic for a compound form blocked inline.

    ``if`` / ``case`` / ``try`` may not appear directly in a bar-safe inline
    position (an inline ``=>`` branch body, an inline ``catch`` body, after
    ``until``, or as a ``case`` *expression* branch).  ``stmt_context`` selects
    the honest, actionable guidance:

    - ``True``: a statement was expected here, so the compound must be written
      as an indented block (a suite).
    - ``False``: an expression was expected here, so a ``case`` expression must
      be parenthesized; ``if``/``try`` have no expression form at all.
    """
    if stmt_context:
        guidance = f"`{keyword}` is not allowed inline here; write it as an indented block instead."
    elif keyword == "case":
        guidance = (
            "`case` is not allowed inline here; "
            "parenthesize the case expression, e.g. `(case x of ...)`."
        )
    else:
        guidance = f"`{keyword}` is not allowed inline here; write it as an indented block instead."
    return AglSyntaxError(guidance, span=span)


def _completed_try(exc: UnexpectedToken) -> bool:
    """Whether the parser had just completed a ``try`` expression.

    Lets a diagnostic ask the parser what it had just finished reading rather
    than guessing from source text.
    """
    top = exc.interactive_parser.parser_state.value_stack[-1]
    return isinstance(top, Tree) and top.data == "try_expr"


def _inline_body_guidance(value: str, source_text: str, token_pos: int) -> str | None:
    """Guidance for a binder opening an inline ``=>`` body, which takes one item.

    Returns ``None`` unless the binder really does open the body introduced by
    the nearest preceding ``=>`` on the same line, so that a stray ``let``
    elsewhere still takes the generic path.
    """
    arrow = source_text.rfind("=>", 0, token_pos)
    if arrow < 0 or "\n" in source_text[arrow:token_pos]:
        return None
    if source_text[arrow + 2 : token_pos].strip():
        return None
    return (
        f"a `{value}` binding cannot be an inline `=>` body; "
        f"parenthesize it (`({value} x = ...; x)`) or use an indented block."
    )


def _make_missing_else_arrow_error(span: SourceSpan) -> AglSyntaxError:
    return AglSyntaxError("Missing `=>` after `else`.", span=span)


def _make_placeholder_position_error(span: SourceSpan) -> AglSyntaxError:
    return AglSyntaxError(
        "placeholder is only allowed as a whole parenthesized call argument.",
        span=span,
    )


def _is_missing_arrow_after_else(*, source_text: str, token_pos: int, expected: set[str]) -> bool:
    return (
        expected == {"ARROW"} and _ELSE_BEFORE_TOKEN_RE.search(source_text[:token_pos]) is not None
    )


def _is_placeholder_position_error(*, token_type: str, source_text: str, token_pos: int) -> bool:
    return (
        token_type in {"PLACEHOLDER", "PLACEHOLDER_NUM"}
        or _PLACEHOLDER_BEFORE_TOKEN_RE.search(source_text[:token_pos]) is not None
    )


# Zero-width layout/end token types that carry no line of their own: the
# spacing hint's anchor line for one of these is borrowed from the last real
# token before it, the same way Lark borrows $END's reported position.
_LAYOUT_TOKEN_TYPES: frozenset[str] = frozenset({"_INDENT", "_DEDENT", "_NEWLINE", "$END"})


def _dollar_spacing_hint(
    tokens: Sequence[Token],
    *,
    offending_type: str,
    line: int,
    pos: int,
) -> str:
    """Hint suffix when a ``$``-suffixed NAME precedes the error on its anchor line.

    *tokens* is the parse's single materialized token pass (see
    :func:`~agm.agl.lexer.token_collector`). The anchor line is *line* (the
    offending token's own line), except for a zero-width layout/end token —
    ``_INDENT``/``_DEDENT``/``_NEWLINE``/``$END`` have no line of their own, so
    their anchor is the line of the last real token strictly before *pos*.
    Reports the first ``$``-suffixed NAME on the anchor line before *pos*,
    delegating the stem check to :func:`~agm.agl.diagnostics.dollar_spacing_hint`.
    """
    spans = [(tok, token_span(tok)) for tok in tokens]
    anchor_line = line
    if offending_type in _LAYOUT_TOKEN_TYPES:
        anchor_line = 0
        for tok, span in spans:
            if tok.type not in _LAYOUT_TOKEN_TYPES and span.start_offset < pos:
                anchor_line = span.start_line
    for tok, span in spans:
        if tok.type != NAME or span.start_line != anchor_line or span.start_offset >= pos:
            continue
        hint = dollar_spacing_hint(str(tok))
        if hint:
            return hint
    return ""


# Token types that can END an operand: the lexer's own inventory, plus
# ``VERBATIM_END``.  A `$` literal runs to the end of its line, so nothing can
# follow it in operator position and the lexer leaves it out; here it closes an
# operand like any other literal.  Used to recognize a `$` literal opener
# rejected as a further juxtaposed argument (juxtaposition applies exactly one
# argument, so a THIRD juxtaposed token — name or `$` literal alike — is always
# rejected the same way).  Membership is tested in scanner form, since these
# tokens carry the grammar spellings of the reserved words.
_OPERAND_ENDING_TOKEN_TYPES: frozenset[str] = OPERAND_END_TYPES | {VERBATIM_END}


def _piping_hint(
    tokens: Sequence[Token],
    *,
    offending_type: str,
    line: int,
    pos: int,
) -> str:
    """Hint suffix when a `$` literal opener is rejected as a further juxtaposed argument.

    Fires only when the offending token is a `$` literal opener (a
    ``VERBATIM_START``) and the two tokens immediately preceding it on its own
    line are both operand-ending (see :data:`_OPERAND_ENDING_TOKEN_TYPES`) —
    the shape of ``f x $ y``, where ``f x`` is already a full juxtaposed
    application and the `$` literal was meant as a further, piped argument.
    Delegates the wording to :func:`~agm.agl.diagnostics.piping_hint`.
    """
    if offending_type != VERBATIM_START:
        return ""
    preceding = [
        tok
        for tok in tokens
        if (span := token_span(tok)).start_offset < pos and span.start_line == line
    ]
    if len(preceding) < 2:
        return ""
    if (
        scanner_token_type(preceding[-1].type) in _OPERAND_ENDING_TOKEN_TYPES
        and scanner_token_type(preceding[-2].type) in _OPERAND_ENDING_TOKEN_TYPES
    ):
        return piping_hint()
    return ""


def syntax_error_from_lark(
    exc: UnexpectedToken, *, source_text: str, tokens: Sequence[Token]
) -> AglSyntaxError:
    """Convert the parser's ``UnexpectedToken`` on *source_text* to ``AglSyntaxError``.

    *tokens* is the parse's materialized token pass (see
    :func:`~agm.agl.lexer.token_collector`); every message gets one uniform
    final pass appending :func:`_dollar_spacing_hint` or :func:`_piping_hint`.
    """
    error = _unexpected_token_error(exc, source_text=source_text)
    offending_type = exc.token.type
    hint = _dollar_spacing_hint(
        tokens,
        offending_type=offending_type,
        line=error.span.start_line,
        pos=error.span.start_offset,
    ) or _piping_hint(
        tokens,
        offending_type=offending_type,
        line=error.span.start_line,
        pos=error.span.start_offset,
    )
    if not hint:
        return error
    return AglSyntaxError(f"{error}{hint}", span=error.span)


def _unexpected_token_error(exc: UnexpectedToken, *, source_text: str) -> AglSyntaxError:
    """Build the syntax error for an unexpected token, without the spacing hint."""
    tok = exc.token
    if tok.type == "$END" and tok.end_pos is None:
        # Lark borrows $END's position from the last token unless that
        # token is falsy (zero-width, e.g. VERBATIM_END); its synthetic
        # (1, 1) fallback has no end position.
        span = _end_of_source_span(source_text)
    else:
        span = token_span(tok)
    pos = span.start_offset
    if _is_missing_arrow_after_else(
        source_text=source_text, token_pos=pos, expected=set(exc.expected)
    ):
        return _make_missing_else_arrow_error(span)
    if tok.type == "$END" and "END" in exc.expected:
        from agm.agl.lexer import unclosed_scope_path

        return AglSyntaxError(
            f"Missing scope closer; expected 'end {unclosed_scope_path(source_text)}'.",
            span=span,
        )
    if _is_placeholder_position_error(token_type=tok.type, source_text=source_text, token_pos=pos):
        return _make_placeholder_position_error(span)
    # Chained comparison detection: the unexpected token is
    # a comparison operator AND that operator is NOT in the expected set.
    # When the operator IS expected, we are still before the first comparison
    # (valid start of, e.g., ``x == y``); when it is absent, a full comparison
    # expression was already consumed and the parser cannot continue — the
    # user chained comparisons such as ``x == y == z``, ``1 < 2 < 3``, or
    # ``a <= b != c``.
    if tok.type in _CMP_OPS and tok.type not in exc.expected:
        return _make_chained_comparison_error(span)
    # Bar-safe inline-form rejections: a
    # nested ``if`` / ``case`` / ``try`` appears where the grammar's inline
    # forms forbid it (inline ``=>``/``catch`` body, after ``until``, or a
    # ``case`` expression branch).  Differentiate "needs a suite" (a
    # statement position) from "needs parentheses" (an expression position)
    # by whether the expected set contains any statement starter.
    if tok.type in _INLINE_BLOCKED:
        stmt_context = bool(_STMT_STARTERS & set(exc.expected))
        return _make_inline_compound_error(tok.value, span, stmt_context=stmt_context)
    # A binder written directly in an inline `=>` body, which takes a
    # single item.  It is legal parenthesized or as an indented block.
    if tok.value in _INLINE_BODY_OPENERS:
        guidance = _inline_body_guidance(str(tok.value), source_text, pos)
        if guidance is not None:
            return AglSyntaxError(guidance, span=span)
    # The `try` body ended while a `catch` was still expected.
    if tok.type in _TRY_BODY_ENDERS and "CATCH" in exc.expected:
        return AglSyntaxError(
            "a `try` expression requires at least one `catch` clause.",
            span=span,
        )
    # A nested `try` closed a marked body's final item: it consumed every
    # `catch`, so the enclosing `try` has none and the body has no value.
    if tok.type in _TRY_BODY_ENDERS and "SEMICOLON" in exc.expected and _completed_try(exc):
        return AglSyntaxError(
            "a nested `try` at the end of a `try` body takes the enclosing "
            "`catch` clauses; parenthesize it.",
            span=span,
        )
    if tok.type == "_INDENT" and bool(_ITEM_ENDERS & set(exc.expected)):
        # A complete item was already in hand (its terminators are expected),
        # so this indentation is stray — never a misplaced `$` literal,
        # whose payload the lexer would have consumed on the header line.
        return AglSyntaxError(
            "Unexpected indentation; this line is more indented than its block.",
            span=span,
        )
    if tok.type == "_NEWLINE":
        if "_INDENT" in exc.expected:
            return AglSyntaxError(
                "Expected an indented block or inline expression after this line.",
                span=span,
            )
        return AglSyntaxError("Unexpected newline.", span=span)
    zero_width = _ZERO_WIDTH_TOKEN_NAMES.get(tok.type)
    if zero_width is not None:
        return AglSyntaxError(f"Unexpected {zero_width}.", span=span)
    return AglSyntaxError(f"Unexpected {tok.value!r}.", span=span)
