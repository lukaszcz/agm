"""AglSyntaxError: parse-layer error with SourceSpan and friendly message.

This module maps Lark ``UnexpectedInput`` family and lexer ``LexError`` to
``AglSyntaxError`` carrying a ``SourceSpan`` and a user-facing message.

Special cases:
- If the unexpected token is any comparison operator (``==``, ``!=``, ``<``,
  ``<=``, ``>``, ``>=``) and that operator is NOT in the expected set, the
  parser has already consumed one complete comparison expression and a second
  one was chained, which is non-associative in AgL.  A targeted
  "comparisons are non-associative; parenthesize" message is emitted instead of
  the generic "Unexpected token" fallback.
"""

from __future__ import annotations

import re

from agm.agl.diagnostics import AglError
from agm.agl.syntax.spans import SourceSpan

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
_MODULE_HEADER_RE = re.compile(
    r"(?:^|[;\n])[ \t]*(?P<kind>(?:(?:open[ \t]+)?import|export))[ \t]+(?P<path>[^;\n]*)$"
)
_MODULE_PATH_BEFORE_DOT_RE = re.compile(r"[^\s/]+(?:[ \t]*/[ \t]*[^\s/]+)*[ \t]*$")


class AglSyntaxError(AglError):
    """A syntax error produced by the parser layer.

    Carries a :class:`~agm.agl.syntax.spans.SourceSpan` pinpointing the
    offending location in the source.  The ``span`` attribute is guaranteed
    non-None (unlike the base ``AglError``).
    """

    def __init__(self, message: str, *, span: SourceSpan) -> None:
        super().__init__(message, span=span)

    @property
    def source_span(self) -> SourceSpan:
        """Always-non-None span; avoids repeated ``assert span is not None``."""
        assert self.span is not None
        return self.span


def _span_from_token(
    token_line: int,
    token_col: int,
    token_pos: int,
    token_end_line: int | None,
    token_end_col: int | None,
    token_end_pos: int | None,
) -> SourceSpan:
    """Build a SourceSpan from Lark Token position fields.

    Falls back to single-character synthetic positions when optional fields
    are absent.
    """
    end_line = token_end_line if token_end_line is not None else token_line
    end_col = token_end_col if token_end_col is not None else token_col + 1
    end_pos = token_end_pos if token_end_pos is not None else token_pos + 1
    return SourceSpan(
        start_line=token_line,
        start_col=token_col,
        end_line=end_line,
        end_col=end_col,
        start_offset=token_pos,
        end_offset=end_pos,
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


def _make_missing_else_arrow_error(span: SourceSpan) -> AglSyntaxError:
    return AglSyntaxError("Missing `=>` after `else`.", span=span)


def _make_placeholder_position_error(span: SourceSpan) -> AglSyntaxError:
    return AglSyntaxError(
        "placeholder is only allowed as a whole parenthesized call argument.",
        span=span,
    )


def _is_missing_arrow_after_else(
    *, source_text: str | None, token_pos: int, expected: set[str]
) -> bool:
    return (
        source_text is not None
        and expected == {"ARROW"}
        and _ELSE_BEFORE_TOKEN_RE.search(source_text[:token_pos]) is not None
    )


def _is_placeholder_position_error(
    *, token_type: str, source_text: str | None, token_pos: int
) -> bool:
    return token_type in {"PLACEHOLDER", "PLACEHOLDER_NUM"} or (
        source_text is not None
        and _PLACEHOLDER_BEFORE_TOKEN_RE.search(source_text[:token_pos]) is not None
    )


def _module_header_migration_error(
    *,
    token_type: str,
    token_value: str,
    source_text: str | None,
    token_pos: int,
    span: SourceSpan,
) -> AglSyntaxError | None:
    if source_text is None:
        return None
    header = _MODULE_HEADER_RE.search(source_text[:token_pos])
    if header is None:
        return None
    module_kind = str(header.group("kind"))
    noun = "import" if module_kind.endswith("import") else "export"
    path_before_token = str(header.group("path"))
    if token_type == "DOT" and _MODULE_PATH_BEFORE_DOT_RE.fullmatch(path_before_token):
        return AglSyntaxError("Module paths use `/` between segments.", span=span)
    if token_type == "SLASH" and not path_before_token.strip():
        return AglSyntaxError(
            f"An {noun} module path must not start with `/`.", span=span
        )
    if token_type == "NAME" and token_value == "qualified":
        message = (
            "`qualified` was removed; imports are qualified by default."
            if noun == "import"
            else "`qualified` was removed; exports name modules directly."
        )
        return AglSyntaxError(message, span=span)
    return None


def syntax_error_from_lark(
    exc: Exception,
    *,
    filename: str = "<agl>",
    source_text: str | None = None,
) -> AglSyntaxError:
    """Convert a Lark parse exception to ``AglSyntaxError``.

    Handles:
    - ``lark.exceptions.UnexpectedToken`` (token type mismatch)
    - ``lark.exceptions.UnexpectedCharacters`` (lexer-level character error)
    - ``lark.exceptions.UnexpectedEOF`` (premature end-of-file)
    - ``agm.agl.lexer.errors.LexError`` (custom lexer error)
    - Generic fallback for any other exception.
    """
    from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedToken

    from agm.agl.lexer.errors import LexError

    if isinstance(exc, LexError):
        # LexError already carries a SourceSpan.
        assert exc.span is not None
        return AglSyntaxError(str(exc), span=exc.span)

    if isinstance(exc, UnexpectedToken):
        tok = exc.token
        line = tok.line if tok.line is not None else 1
        col = tok.column if tok.column is not None else 1
        pos = tok.start_pos if tok.start_pos is not None else 0
        span = _span_from_token(line, col, pos, tok.end_line, tok.end_column, tok.end_pos)
        if _is_missing_arrow_after_else(
            source_text=source_text, token_pos=pos, expected=set(exc.expected)
        ):
            return _make_missing_else_arrow_error(span)
        if _is_placeholder_position_error(
            token_type=tok.type, source_text=source_text, token_pos=pos
        ):
            return _make_placeholder_position_error(span)
        migration_error = _module_header_migration_error(
            token_type=tok.type,
            token_value=str(tok),
            source_text=source_text,
            token_pos=pos,
            span=span,
        )
        if migration_error is not None:
            return migration_error
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
        if tok.type == "_NEWLINE":
            if "_INDENT" in exc.expected:
                return AglSyntaxError(
                    "Expected an indented block or inline expression after this line.",
                    span=span,
                )
            return AglSyntaxError("Unexpected newline.", span=span)
        return AglSyntaxError(
            f"Unexpected {tok.value!r}.",
            span=span,
        )

    if isinstance(exc, UnexpectedCharacters):
        line = exc.line if exc.line is not None else 1
        col = exc.column if exc.column is not None else 1
        pos = exc.pos_in_stream if exc.pos_in_stream is not None else 0
        span = SourceSpan(
            start_line=line,
            start_col=col,
            end_line=line,
            end_col=col + 1,
            start_offset=pos,
            end_offset=pos + 1,
        )
        return AglSyntaxError(
            "Unexpected character.",
            span=span,
        )

    if isinstance(exc, UnexpectedEOF):
        # No position info; use (1, 1) as a fallback.
        span = SourceSpan(
            start_line=1,
            start_col=1,
            end_line=1,
            end_col=1,
            start_offset=0,
            end_offset=0,
        )
        return AglSyntaxError("Unexpected end of input.", span=span)

    # Generic fallback.
    span = SourceSpan(
        start_line=1,
        start_col=1,
        end_line=1,
        end_col=1,
        start_offset=0,
        end_offset=0,
    )
    return AglSyntaxError(str(exc), span=span)
