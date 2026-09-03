"""Dependency-free canonical inventory of AgL keywords, reserved and soft.

`KEYWORDS` holds the reserved words, which are always keywords; `SOFT_KEYWORDS`
holds the spellings the lexer promotes contextually and that stay ordinary names
everywhere else.

Every syntax-highlighting definition mirrors both inventories and must be
updated alongside them: `config/micro/agl.yaml` (Micro), `config/emacs/agl-mode.el`
(`agl-keywords`, `agl-soft-keywords`, and the other keyword-inventory constants
near the top of that file), and the REPL prompt highlighter in
`agm.agl.repl.console`.
"""

from __future__ import annotations

KW_RECORD = "record"
KW_ENUM = "enum"
KW_EXCEPTION = "exception"
KW_TYPE = "type"
KW_BUILTIN = "builtin"
KW_EXTERN = "extern"
KW_EXTENDS = "extends"
KW_PROGRAM = "program"
KW_LET = "let"
KW_VAR = "var"
KW_DEF = "def"  # function declaration keyword
KW_FN = "fn"  # lambda keyword
KW_DO = "do"
KW_UNTIL = "until"
KW_DONE = "done"
KW_BREAK = "break"
KW_CONTINUE = "continue"
KW_FOR = "for"
KW_WHILE = "while"
KW_IF = "if"
KW_ELSE = "else"
KW_CASE = "case"
KW_OF = "of"
KW_TRY = "try"
KW_CATCH = "catch"
KW_RAISE = "raise"
KW_RETURN = "return"
KW_AS = "as"
KW_AS_QUESTION = "as?"
# `pass` is a plain identifier (role taken by `()`).
# `print` is an ordinary function name (NAME).
KW_AND = "and"
KW_OR = "or"
KW_NOT = "not"
KW_IS = "is"
KW_IN = "in"
KW_TRUE = "true"
KW_FALSE = "false"
KW_NULL = "null"
KW_INFIXL = "infixl"
KW_INFIXR = "infixr"
KW_PRIO = "prio"
KW_TO = "to"
KW_DOWNTO = "downto"
KW_BY = "by"
KW_WITH = "with"

# Soft keywords: promoted to their own token type only inside a promotion
# window (item start, or a module header), and ordinary names everywhere else.
# `at`, which introduces an infix priority, is contextual too but is never
# promoted -- it stays a NAME token and is recognized by the AST builder.
KW_IMPORT = "import"
KW_USE = "use"
KW_EXPORT = "export"
KW_HIDING = "hiding"
KW_SCOPE = "scope"
KW_END = "end"

KEYWORDS: frozenset[str] = frozenset(
    {
        KW_RECORD,
        KW_ENUM,
        KW_EXCEPTION,
        KW_TYPE,
        KW_BUILTIN,
        KW_EXTERN,
        KW_EXTENDS,
        KW_PROGRAM,
        KW_LET,
        KW_VAR,
        KW_DEF,
        KW_FN,
        KW_DO,
        KW_UNTIL,
        KW_DONE,
        KW_BREAK,
        KW_CONTINUE,
        KW_FOR,
        KW_WHILE,
        KW_IF,
        KW_ELSE,
        KW_CASE,
        KW_OF,
        KW_TRY,
        KW_CATCH,
        KW_RAISE,
        KW_RETURN,
        KW_AS,
        KW_AS_QUESTION,
        KW_AND,
        KW_OR,
        KW_NOT,
        KW_IS,
        KW_IN,
        KW_TRUE,
        KW_FALSE,
        KW_NULL,
        KW_INFIXL,
        KW_INFIXR,
        KW_PRIO,
        KW_TO,
        KW_DOWNTO,
        KW_BY,
        KW_WITH,
    }
)

SOFT_KEYWORDS: frozenset[str] = frozenset(
    {
        KW_IMPORT,
        KW_USE,
        KW_EXPORT,
        KW_HIDING,
        KW_SCOPE,
        KW_END,
    }
)
