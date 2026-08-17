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

from agm.agl.keywords import KEYWORDS, KW_AS_QUESTION

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
RAW_TAIL_NAME = "RAW_TAIL_NAME"
RAW_TAIL_START = "RAW_TAIL_START"
RAW_FRAGMENT = "RAW_FRAGMENT"
RAW_TAIL_END = "RAW_TAIL_END"

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
IMPORT = "IMPORT"  # contextual: 'import' at item-start
USE = "USE"  # contextual: 'use' at item-start
HIDING = "HIDING"  # contextual: 'hiding' in a module header
EXPORT = "EXPORT"  # contextual: 'export' at item-start
SCOPE = "SCOPE"  # contextual: 'scope' at item-start before a scope path
END = "END"  # contextual: 'end' at item-start while a scope region is open
MODQUAL = "MODQUAL"  # synthetic: merged qualifier prefix (e.g. "foo/bar::")
MODPATH = "MODPATH"  # synthetic: merged module path in a header (e.g. "foo/bar")
WILDCARD = "WILDCARD"  # synthetic: adjacent "/*" tail of a wildcard module header

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
