"""Canonical token-type name constants for the AgL lexer.

This is the **single source of truth** for all token type names emitted by the
custom lexer and declared in ``grammar/agl.lark``.  It also defines the
tree-sitter portability contract: the ``externals`` set of a future
tree-sitter grammar maps 1-to-1 to the synthetic tokens below.

Token categories
----------------
Layout (synthetic, ``%declare`` in grammar):
    Produced by the INDENT/DEDENT filter; never appear in source text directly.

Templates (synthetic, ``%declare`` in grammar):
    Produced by the template sub-scanner for string literals and interpolation.

Keywords:
    Reserved words that are always keywords.

Contextual keywords:
    ``prompt`` and ``exec`` are NOT reserved; they lex as plain VAR_NAME tokens.
    The scope pass gives them their built-in meaning.

Identifiers:
    TYPE_NAME -- starts with upper-case letter (records, enums, type aliases).
    VAR_NAME  -- starts with lower-case letter or underscore (variables, fields).
    Policy: ``_`` (the wildcard) is NOT a distinct token -- it lexes as a plain
    VAR_NAME; wildcard interpretation happens at the grammar / AST-builder level.

Keyword convention:
    A keyword's token *type* is the keyword string itself (e.g. the ``let``
    token has type ``"let"``), matching the prototype grammar's string
    terminals -- not a synthetic ``KW_*`` name.  The ``KW_*`` constants below
    are just readable aliases for those literal strings.

Numbers:
    INT     -- integer literals.
    DECIMAL -- decimal (fixed-point) literals.  No float type in AgL.

Operators / punctuation:
    Named constants for every single- and multi-char operator.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Layout tokens (synthetic; produced by INDENT/DEDENT filter)
# ---------------------------------------------------------------------------
NEWLINE = "_NEWLINE"
INDENT = "_INDENT"
DEDENT = "_DEDENT"

# ---------------------------------------------------------------------------
# Template / interpolation tokens (synthetic)
# ---------------------------------------------------------------------------
TEMPLATE_START = "TEMPLATE_START"
STRING_FRAGMENT = "STRING_FRAGMENT"
INTERP_START = "INTERP_START"  # "${" sequence
INTERP_END = "INTERP_END"  # "}" that closes an interpolation
TEMPLATE_END = "TEMPLATE_END"

# ---------------------------------------------------------------------------
# Keywords (always reserved)
# ---------------------------------------------------------------------------
KW_RECORD = "record"
KW_ENUM = "enum"
KW_TYPE = "type"
KW_INPUT = "input"
KW_AGENT = "agent"
KW_LET = "let"
KW_VAR = "var"
KW_SET = "set"
KW_DO = "do"
KW_UNTIL = "until"
KW_IF = "if"
KW_ELSE = "else"
KW_CASE = "case"
KW_OF = "of"
KW_TRY = "try"
KW_CATCH = "catch"
KW_RAISE = "raise"
KW_AS = "as"
KW_PASS = "pass"
KW_PRINT = "print"
KW_AND = "and"
KW_OR = "or"
KW_NOT = "not"
KW_IS = "is"
KW_IN = "in"
KW_TRUE = "true"
KW_FALSE = "false"
KW_NULL = "null"

# Set of all reserved keyword strings (used by the scanner for fast lookup).
KEYWORDS: frozenset[str] = frozenset(
    {
        KW_RECORD,
        KW_ENUM,
        KW_TYPE,
        KW_INPUT,
        KW_AGENT,
        KW_LET,
        KW_VAR,
        KW_SET,
        KW_DO,
        KW_UNTIL,
        KW_IF,
        KW_ELSE,
        KW_CASE,
        KW_OF,
        KW_TRY,
        KW_CATCH,
        KW_RAISE,
        KW_AS,
        KW_PASS,
        KW_PRINT,
        KW_AND,
        KW_OR,
        KW_NOT,
        KW_IS,
        KW_IN,
        KW_TRUE,
        KW_FALSE,
        KW_NULL,
    }
)

# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------
TYPE_NAME = "TYPE_NAME"  # /[A-Z][A-Za-z0-9_]*/
VAR_NAME = "VAR_NAME"  # /[a-z_][A-Za-z0-9_]*/

# ---------------------------------------------------------------------------
# Numbers (no float type; decimal is exact fixed-point)
# ---------------------------------------------------------------------------
INT = "INT"
DECIMAL = "DECIMAL"  # /[0-9]+\.[0-9]+/

# ---------------------------------------------------------------------------
# Operators and punctuation
# ---------------------------------------------------------------------------
ARROW = "ARROW"  # =>
EQ = "EQ"  # =
NEQ = "NEQ"  # !=
LE = "LE"  # <=
GE = "GE"  # >=
LT = "LT"  # <
GT = "GT"  # >
PLUS = "PLUS"  # +
MINUS = "MINUS"  # -
STAR = "STAR"  # *
SLASH = "SLASH"  # /
LPAR = "LPAR"  # (
RPAR = "RPAR"  # )
LSQB = "LSQB"  # [
RSQB = "RSQB"  # ]
LBRACE = "LBRACE"  # {
RBRACE = "RBRACE"  # }
COLON = "COLON"  # :
COMMA = "COMMA"  # ,
DOT = "DOT"  # .
PIPE = "PIPE"  # |
SEMICOLON = "SEMICOLON"  # ;

# Error token emitted for "==" so the parser can emit a friendly diagnostic.
EQ_EQ = "EQ_EQ"  # ==

# Synthetic token emitted by the lexer when it sees do[N].
# The lexer merges LSQB INT RSQB → LOOP_BOUND right after DO, so the grammar
# can use a single token instead of a three-token sequence, eliminating the
# LALR(1) conflict with lit_list (which also matches LSQB INT RSQB).
LOOP_BOUND = "LOOP_BOUND"  # [N] immediately after do — value is the integer string

# ---------------------------------------------------------------------------
# Grammar token-type mapping
#
# Lark grammar rules use string literals for keywords (e.g. ``"pass"``),
# which Lark auto-creates as uppercase terminal names (``PASS``).  The
# custom AglLexer must therefore emit those uppercase names when interfacing
# with the Lark parser (the ``AglLexer.lex()`` interface).
#
# The raw scanner (``scanner.py``) emits lowercase keyword types following
# the ``keyword string == token type`` convention documented above.  The
# ``tokenize()`` public helper preserves that lowercase stream; only the
# parser-facing ``AglLexer.lex()`` method applies this mapping.
# ---------------------------------------------------------------------------
GRAMMAR_TOKEN_REMAP: dict[str, str] = {
    kw: kw.upper() for kw in KEYWORDS
}
