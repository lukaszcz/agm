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

from agm.agl import keywords as _keywords

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

# Reserved keyword constants and their canonical inventory are imported from
# ``agm.agl.keywords`` and re-exported here with the rest of the token contract.
KEYWORDS = _keywords.KEYWORDS
KW_RECORD = _keywords.KW_RECORD
KW_ENUM = _keywords.KW_ENUM
KW_EXCEPTION = _keywords.KW_EXCEPTION
KW_TYPE = _keywords.KW_TYPE
KW_BUILTIN = _keywords.KW_BUILTIN
KW_EXTERN = _keywords.KW_EXTERN
KW_EXTENDS = _keywords.KW_EXTENDS
KW_PARAM = _keywords.KW_PARAM
KW_PROGRAM = _keywords.KW_PROGRAM
KW_LET = _keywords.KW_LET
KW_VAR = _keywords.KW_VAR
KW_DEF = _keywords.KW_DEF
KW_FN = _keywords.KW_FN
KW_DO = _keywords.KW_DO
KW_UNTIL = _keywords.KW_UNTIL
KW_DONE = _keywords.KW_DONE
KW_BREAK = _keywords.KW_BREAK
KW_CONTINUE = _keywords.KW_CONTINUE
KW_FOR = _keywords.KW_FOR
KW_WHILE = _keywords.KW_WHILE
KW_IF = _keywords.KW_IF
KW_ELSE = _keywords.KW_ELSE
KW_CASE = _keywords.KW_CASE
KW_OF = _keywords.KW_OF
KW_TRY = _keywords.KW_TRY
KW_CATCH = _keywords.KW_CATCH
KW_RAISE = _keywords.KW_RAISE
KW_RETURN = _keywords.KW_RETURN
KW_AS = _keywords.KW_AS
KW_AS_QUESTION = _keywords.KW_AS_QUESTION
KW_AND = _keywords.KW_AND
KW_OR = _keywords.KW_OR
KW_NOT = _keywords.KW_NOT
KW_IS = _keywords.KW_IS
KW_IN = _keywords.KW_IN
KW_TRUE = _keywords.KW_TRUE
KW_FALSE = _keywords.KW_FALSE
KW_NULL = _keywords.KW_NULL
KW_INFIXL = _keywords.KW_INFIXL
KW_INFIXR = _keywords.KW_INFIXR
KW_PRIO = _keywords.KW_PRIO
KW_TO = _keywords.KW_TO
KW_DOWNTO = _keywords.KW_DOWNTO
KW_BY = _keywords.KW_BY
KW_WITH = _keywords.KW_WITH

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
OPEN = "OPEN"  # contextual: 'open' directly before an item-start import
IMPORT = "IMPORT"  # contextual: 'import' at item-start
USING = "USING"  # contextual: 'using' in an import or export declaration
HIDING = "HIDING"  # contextual: 'hiding' in an import or export declaration
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
