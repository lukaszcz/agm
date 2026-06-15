# AgL Grammar Reference

This document describes the token alphabet, layout rules, and operator
precedence table for the AgL (Agent Language) DSL.  It is the portability
contract (plan §12): a future parser replacement (e.g. tree-sitter) must
produce the same token stream and honour the same precedence rules.

All information below is derived from
`src/agm/agl/lexer/tokens.py` and `src/agm/agl/grammar/agl.lark`.

---

## Token Alphabet

### Layout tokens (synthetic)

Produced by the INDENT/DEDENT filter (`agm.agl.lexer.layout`).  They never
appear in source text directly.  The leading underscore causes Lark to filter
them from parse trees.

| Token | Meaning |
|-------|---------|
| `_NEWLINE` | Statement separator at the current indent level |
| `_INDENT` | Start of a new indented block |
| `_DEDENT` | End of an indented block |

### Template / interpolation tokens (synthetic)

Produced by the template sub-scanner inside string literals.  Inside any open
bracket (`(`, `[`, `{`, `${…}`) the paren level is positive and `_NEWLINE`
tokens are suppressed (implicit continuation).

| Token | Meaning |
|-------|---------|
| `TEMPLATE_START` | Opening `"` (or `"""`) of a template string |
| `STRING_FRAGMENT` | Literal text fragment inside a template |
| `INTERP_START` | `${` — start of an interpolation expression |
| `INTERP_END` | `}` — end of an interpolation expression |
| `TEMPLATE_END` | Closing `"` (or `"""`) of a template string |

### Keywords

All keywords below are **always reserved**: the scanner never emits them as
`VAR_NAME`.  The token *type* is the keyword string itself (e.g. the `let`
keyword has token type `"let"`); the `KW_*` constants in `tokens.py` are
readable aliases.  When the Lark parser receives the token stream the custom
`AglLexer.lex()` method uppercases keyword types (e.g. `"let"` → `"LET"`).

`record` `enum` `type` `input` `agent` `let` `var` `set` `do` `until` `if`
`else` `case` `of` `try` `catch` `raise` `as` `pass` `print` `and` `or` `not`
`is` `in` `true` `false` `null`

`agent` is reserved (it leads an `agent` declaration) but it is still accepted
as a *field name* — record/enum field definitions, named constructor arguments,
dict shorthand keys, and postfix field access — via the `field_name`
nonterminal (`field_name: VAR_NAME | AGENT`).  This keeps built-in exception
fields such as `AgentCallError.agent` usable.  It cannot be used as a variable
binder (`let`/`var`/`set`/`input`/patterns/catch).

**Contextual keywords** — `prompt` and `exec` are NOT reserved; they lex as
plain `VAR_NAME` tokens.  The scope pass gives them their built-in meaning.

**Type-annotation keywords** — `text`, `json`, `bool`, `int`, `decimal`,
`list`, `dict` are NOT reserved; they also lex as `VAR_NAME`.  The AST builder
maps them to `TypeExpr` nodes inside type-annotation positions.

### Identifiers

| Token | Pattern | Used for |
|-------|---------|----------|
| `TYPE_NAME` | `/[A-Z][A-Za-z0-9_]*/` | Record/enum types, qualified constructors |
| `VAR_NAME` | `/[a-z_][A-Za-z0-9_]*/` | Variables, field names, agent names (the reserved word `agent` is also accepted as a field name via `field_name`) |

`_` (the wildcard) is NOT a distinct token — it lexes as a plain `VAR_NAME`;
wildcard interpretation (`WildcardPattern`) happens in the AST builder.

### Numbers

No float type in AgL.  Decimal arithmetic is exact fixed-point.

| Token | Matches |
|-------|---------|
| `INT` | `/[0-9]+/` |
| `DECIMAL` | `/[0-9]+\.[0-9]+/` |

### Operators and punctuation

| Token | Source | Token | Source |
|-------|--------|-------|--------|
| `ARROW` | `=>` | `EQ` | `=` |
| `NEQ` | `!=` | `LT` | `<` |
| `LE` | `<=` | `GT` | `>` |
| `GE` | `>=` | `PLUS` | `+` |
| `MINUS` | `-` | `STAR` | `*` |
| `SLASH` | `/` | `LPAR` | `(` |
| `RPAR` | `)` | `LSQB` | `[` |
| `RSQB` | `]` | `LBRACE` | `{` |
| `RBRACE` | `}` | `COLON` | `:` |
| `COMMA` | `,` | `DOT` | `.` |
| `PIPE` | `\|` | `SEMICOLON` | `;` |

**Special tokens:**

| Token | Source | Purpose |
|-------|--------|---------|
| `EQ_EQ` | `==` | Error token; not in any grammar rule.  The parser emits `"Use \`=\` for equality."` |
| `LOOP_BOUND` | `[N]` after `do` | Merged from `LSQB INT RSQB` by the lexer to eliminate the LALR(1) conflict with list literals.  The token's value is the integer string `N`; the AST builder rejects zero or negative values. |

---

## Layout Rules

AgL uses significant indentation (Python-style).  The INDENT/DEDENT filter in
`agm.agl.lexer.layout` transforms the raw token stream:

1. **Indent stack** starts at `[0]`.
2. **Inside brackets** — `paren_level > 0` when any of `( [ { ${` is open
   (including template interpolations).  While `paren_level > 0`, `_NEWLINE`
   tokens are suppressed (implicit line continuation).  Closing brackets
   `} ) ] }` decrement the level.
3. **On a `_NEWLINE`:**
   - Indentation deeper than the stack top → emit `_INDENT`, push new level.
   - Same depth → emit `_NEWLINE` as-is.
   - Shallower → emit one `_DEDENT` per popped level; if the resulting top
     does not equal the new depth, a `LexError` ("misaligned dedent") is raised.
4. **`|`/`catch`/`until`-continuation rule** — when the first significant token
   on the next line is `|`, `catch`, or `until`, the `_NEWLINE` is suppressed
   and only the `_DEDENT`s needed to pop the stack to levels strictly greater
   than the keyword's column are emitted.  The line never pushes an indent.
   This rule lets `|` branches, `catch` clauses, and `until` continuations
   align with their enclosing keyword without triggering a new block.
5. **At EOF** — remaining indent levels are unwound with `_DEDENT` tokens.

---

## Operator Precedence and Associativity

Listed from **lowest** to **highest** binding power (bottom of the table binds
tightest).  This matches the `agl.lark` grammar's operator chain.

| Level | Operators | Associativity |
|-------|-----------|---------------|
| `or` | `or` | left |
| `and` | `and` | left |
| `not` | `not` (unary prefix) | right (unary) |
| comparison | `=` `!=` `<` `<=` `>` `>=` `in` `is` `is not` | **non-associative** |
| additive | `+` `-` | left |
| multiplicative | `*` `/` | left |
| unary minus | `-` (unary prefix) | right (unary) |
| access | `.field` `.Type` `ctor(…)` (postfix) | left |
| atom | literals, identifiers, `(expr)`, templates, agent calls | — |

**Notes on the comparison level:**

- `=` is the equality operator (not assignment; assignment is `let`/`var`/`set`).
- `in` tests membership: `x in list`.
- `is` / `is not` test against a constructor type: `v is Some`, `v is not None`.
  `is` and `is not` are at the same precedence level as the other comparison
  operators.
- All comparison operators are **non-associative**.  The grammar encodes this by
  allowing at most one binary comparison per expression; a second chained
  operator is an unexpected token.  The parser emits a targeted diagnostic:
  *"Comparisons are non-associative; parenthesize explicitly, e.g. `(x = y) = z`."*

**Case expressions** (`case expr of | …`) are the lowest-precedence expression
form.  In positions where a `|` would otherwise be ambiguous (the `bar_expr`
positions: `if` branches, `catch` bodies, assignment RHS, `do…until`
conditions), a `case` expression must be wrapped in parentheses.
