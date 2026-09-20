"""Canonical token-type name constants for the AgL lexer.

This is the token contract for names emitted by the custom lexer and declared
in ``grammar/agl.lark``. Reserved keyword names come from the dependency-free
canonical inventory in :mod:`agm.agl.keywords`. This module also defines the
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
    ``ask`` and ``exec`` are NOT reserved; they lex as plain NAME tokens.
    The scope pass gives them their built-in meaning.

Soft keywords:
    The spellings in ``SOFT_KEYWORDS`` are promoted to the token types in
    ``SOFT_KEYWORD_TOKENS`` inside their promotion window, and lex as plain
    NAME tokens outside it.  Unlike a reserved word, a soft keyword's token
    type is its spelling UPPER-cased, because the grammar ``%declare``s these
    tokens rather than writing them as string terminals.  The operator words
    are soft: their promotion window is operator position, so they spell
    member and field names everywhere else.

Identifiers:
    NAME -- any identifier (case-neutral: both upper- and lower-case start are NAME).
    OP_NAME -- any operator-name identifier, formed from operator characters.
    PLACEHOLDER -- bare ``?``.
    PLACEHOLDER_NUM -- ``?`` immediately followed by ASCII digits.
    Policy: ``_`` (the wildcard) is NOT a distinct token -- it lexes as a plain
    NAME; wildcard interpretation happens at the grammar / AST-builder level.

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

from agm.agl.keywords import (
    KEYWORDS,
    KW_AS_QUESTION,
    KW_END,
    KW_EXPORT,
    KW_HIDING,
    KW_IMPORT,
    KW_NOT,
    KW_SCOPE,
    KW_USE,
    SOFT_KEYWORDS,
)

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
INTERP_START = "INTERP_START"  # "%{" sequence
INTERP_END = "INTERP_END"  # "}" that closes an interpolation
TEMPLATE_END = "TEMPLATE_END"
VERBATIM_START = "VERBATIM_START"  # "$" opening a verbatim text literal
VERBATIM_END = "VERBATIM_END"  # closes a verbatim text literal (empty value)

# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------
NAME = "NAME"  # any identifier (case-neutral)
OP_NAME = "OP_NAME"  # operator-name identifier
PLACEHOLDER = "PLACEHOLDER"  # bare ? placeholder argument
PLACEHOLDER_NUM = "PLACEHOLDER_NUM"  # numbered placeholder argument, e.g. ?1

# ---------------------------------------------------------------------------
# Numbers (no float type; decimal is exact fixed-point)
# ---------------------------------------------------------------------------
INT = "INT"
DECIMAL = "DECIMAL"  # /[0-9]+\.[0-9]+/

# ---------------------------------------------------------------------------
# Operators and punctuation
# ---------------------------------------------------------------------------
THIN_ARROW = "THIN_ARROW"  # -> (function return type / function type)
ARROW = "ARROW"  # =>
ASSIGN = "ASSIGN"  # :=
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
AT = "AT"  # @
LPAR = "LPAR"  # (
RPAR = "RPAR"  # )
LSQB = "LSQB"  # [
INDEX_LSQB = "INDEX_LSQB"  # [ immediately adjacent to an expression-ending token
TYPEARG_LSQB = "TYPEARG_LSQB"  # [ opening Type[T]::Ctor type arguments
RSQB = "RSQB"  # ]
LBRACE = "LBRACE"  # {
CALL_LBRACE = "CALL_LBRACE"  # { immediately adjacent to an expression-ending token
RBRACE = "RBRACE"  # }
COLON = "COLON"  # :
DCOLON = "DCOLON"  # :: (type-argument introducer for typed calls)
COMMA = "COMMA"  # ,
DOT = "DOT"  # .
PIPE = "PIPE"  # |
SEMICOLON = "SEMICOLON"  # ;

# Equality operator token for "==".
EQ_EQ = "EQ_EQ"  # ==

# Synthetic token: the "[" that opens a loop bound, i.e. the first "[" after the
# "do" keyword.  Retagging it ``DO_LSQB`` lets the grammar distinguish the
# ``do[expr]`` bound from a ``do`` body that starts with an array literal —
# without it, the optional ``loop_bound`` and an array-literal body both begin
# with ``LSQB`` (the LALR(1) conflict this resolves).
DO_LSQB = "DO_LSQB"  # [ opening a do-loop bound

# ---------------------------------------------------------------------------
# Module system tokens (contextual / synthetic — %declare in grammar)
# ---------------------------------------------------------------------------
# A promoted soft keyword's token type is its spelling upper-cased, so the
# inventory in :mod:`agm.agl.keywords` stays the single source of truth.
IMPORT = KW_IMPORT.upper()  # contextual: 'import' at item-start
USE = KW_USE.upper()  # contextual: 'use' at item-start
HIDING = KW_HIDING.upper()  # contextual: 'hiding' in a module header
EXPORT = KW_EXPORT.upper()  # contextual: 'export' at item-start
SCOPE = KW_SCOPE.upper()  # contextual: 'scope' at item-start before a scope path
END = KW_END.upper()  # contextual: 'end' at item-start while a scope region is open
NOT = KW_NOT.upper()  # contextual: 'not' before an operand
MODQUAL = "MODQUAL"  # synthetic: merged qualifier prefix (e.g. "foo/bar::")
MODPATH = "MODPATH"  # synthetic: merged module path in a header (e.g. "foo/bar")
WILDCARD = "WILDCARD"  # synthetic: adjacent "/*" tail of a wildcard module header

#: Token types a promoted soft keyword can carry.  Highlighters classify against
#: this set the way they classify reserved words against ``KEYWORDS``.
SOFT_KEYWORD_TOKENS: frozenset[str] = frozenset(kw.upper() for kw in SOFT_KEYWORDS)

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
GRAMMAR_TOKEN_REMAP: dict[str, str] = {kw: kw.upper() for kw in KEYWORDS if kw != KW_AS_QUESTION}
# `as?` contains `?` which is not valid in an uppercase terminal name; map it
# explicitly to the declared terminal name AS_QUESTION.
GRAMMAR_TOKEN_REMAP[KW_AS_QUESTION] = "AS_QUESTION"
