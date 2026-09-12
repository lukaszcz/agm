"""Operator position: whether a token stands as an operator or as a name.

One inventory of the token types that close an operand, begin one, or mark a
name, plus the predicates that read them.  Two passes ask the same question and
must never disagree: the layout filter joins the lines a line break would
otherwise wedge between an operator and its operand
(:mod:`agm.agl.lexer.layout`), and the soft-keyword promotion pass then decides
whether an operator word is an operator or a member name
(:mod:`agm.agl.lexer.lexer`).

The inventories are kept in canonical *scanner* form, so the public tokenizer's
lowercase keywords and the Lark stream's remapped keywords share them; use
:func:`scanner_token_type` on any type that may already be remapped.
"""

from __future__ import annotations

from lark.lexer import Token

from agm.agl.keywords import OPERATOR_SOFT_KEYWORDS, PREFIX_SOFT_KEYWORDS
from agm.agl.lexer.tokens import (
    ASSIGN,
    COLON,
    DCOLON,
    DECIMAL,
    DOT,
    EQ,
    EQ_EQ,
    GE,
    GRAMMAR_TOKEN_REMAP,
    GT,
    INT,
    LBRACE,
    LE,
    LPAR,
    LSQB,
    LT,
    MINUS,
    MODQUAL,
    NAME,
    NEQ,
    NOT,
    OP_NAME,
    PLUS,
    RBRACE,
    RPAR,
    RSQB,
    SLASH,
    STAR,
    TEMPLATE_END,
    TEMPLATE_START,
    THIN_ARROW,
)

_GRAMMAR_TOKEN_UNMAP = {
    grammar_type: scanner_type for scanner_type, grammar_type in GRAMMAR_TOKEN_REMAP.items()
}


def scanner_token_type(token_type: str) -> str:
    """Return the public scanner spelling for a possibly parser-remapped token type."""
    return _GRAMMAR_TOKEN_UNMAP.get(token_type, token_type)


# Canonical scanner token types that delimit closed expressions and begin infix
# operands.
EXPRESSION_END_TYPES = frozenset(
    {
        NAME,
        OP_NAME,
        INT,
        DECIMAL,
        "true",
        "false",
        "null",
        TEMPLATE_END,
        RPAR,
        RSQB,
        RBRACE,
        "break",
        "continue",
    }
)
INFIX_OPERAND_START_TYPES = frozenset(
    {
        NAME,
        OP_NAME,
        INT,
        DECIMAL,
        "true",
        "false",
        "null",
        TEMPLATE_START,
        LPAR,
        LSQB,
        LBRACE,
        MODQUAL,
        DCOLON,
        "break",
        "continue",
        NOT,
        MINUS,
    }
)
SLASH_LEFT_OPERAND_TYPES = EXPRESSION_END_TYPES | {MODQUAL}

# Token types that close an operand, and so put the next token in operator
# position.  An operator name does not close one (it expects an operand of its
# own), and neither do `break`/`continue`, which end a statement rather than a
# value.
OPERAND_END_TYPES = EXPRESSION_END_TYPES - {OP_NAME, "break", "continue"}

# A soft operator word followed by one of these spells a name: `=` opens a
# named argument or a default, `:` a field or parameter type.  No operand
# starts with either.
NAME_MARKER_TYPES = frozenset({EQ, COLON})

# Positions where only a name can stand, whatever an operand could otherwise
# do there.  Infix words are excluded from these anyway (none closes an
# operand); the prefix word `not` needs them spelled out.
NAME_ONLY_PREV_TYPES = frozenset({DOT, DCOLON, "def", "record", "enum", "exception", "type", "as"})

# Symbolic tokens that are operators wherever an operand can follow them.  `=`
# and `=>` are excluded: both also open an indented suite, so a line break after
# one must keep opening a block.  A bare operator name is a value rather than an
# operator, and `*` spells a wildcard header, so the operand test below is what
# separates those readings.
SYMBOLIC_OPERATOR_TYPES = frozenset(
    {
        OP_NAME,
        PLUS,
        MINUS,
        STAR,
        SLASH,
        LT,
        LE,
        GT,
        GE,
        EQ_EQ,
        NEQ,
        DOT,
        THIN_ARROW,
        ASSIGN,
        "as",
        "as?",
    }
)


def promotes_as_operator(word: str, prev_type: str | None, next_type: str | None) -> bool:
    """Whether an operator word stands in operator position rather than naming a member.

    An infix word is an operator exactly when it follows an operand; the prefix
    word ``not`` exactly when it precedes one.  Either way a following ``=`` or
    ``:`` marks a name, and a preceding ``.``, ``::`` or declaration keyword
    means only a name can stand there.

    The neighbours are compared in scanner form: this runs after the parser path
    has remapped reserved keywords to their grammar spellings, and the token
    inventories above are canonical scanner types.
    """
    prev = scanner_token_type(prev_type) if prev_type is not None else None
    if next_type is not None and scanner_token_type(next_type) in NAME_MARKER_TYPES:
        return False
    if prev in NAME_ONLY_PREV_TYPES:
        return False
    if word in PREFIX_SOFT_KEYWORDS:
        return prev not in OPERAND_END_TYPES
    return prev in OPERAND_END_TYPES


def stands_as_operator(token: Token, prev_type: str | None, next_type: str | None) -> bool:
    """Whether *token* stands as an operator, given the tokens around it.

    True for a symbolic operator whose left neighbour closes an operand, and for
    an operator word in its promotion window.  The layout filter asks this of the
    tokens on either side of a line break, with the break itself ignored, so its
    answer is the one promotion reaches later.
    """
    kind = scanner_token_type(token.type)
    if kind in SYMBOLIC_OPERATOR_TYPES:
        prev = scanner_token_type(prev_type) if prev_type is not None else None
        return prev in OPERAND_END_TYPES
    if kind == NAME and str(token) in OPERATOR_SOFT_KEYWORDS:
        return promotes_as_operator(str(token), prev_type, next_type)
    return False
