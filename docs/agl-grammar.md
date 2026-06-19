# AgL Grammar Reference

This document describes the token alphabet, layout rules, and operator
precedence table for the AgL (Agent Language) DSL. It is the portability
contract for the parser: a future parser replacement (e.g. tree-sitter) must
produce the same token stream and honour the same precedence rules.

---

## Token Alphabet

### Layout tokens (synthetic)

Produced by the INDENT/DEDENT filter. They never appear in source text
directly.

| Token | Meaning |
|-------|---------|
| `_NEWLINE` | Item separator at the current indent level |
| `_INDENT` | Start of a new indented block |
| `_DEDENT` | End of an indented block |

### Template / interpolation tokens (synthetic)

Produced by the template sub-scanner inside string literals. Inside any open
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

The following words are **always reserved**: the scanner never emits them as
`VAR_NAME`.

```
record enum type input agent config def fn let var do until if else
case of try catch raise as and or not is in true false null unit
```

**Removed from v1:** `pass` and `print` are no longer reserved. `pass`'s
role is taken by the unit literal `()`. `print` is a contextual built-in
resolved by the scope pass (it lexes as `VAR_NAME`).

`agent` is reserved (it leads an `agent` declaration) but is still accepted
as a *field name* via a `field_name` production in the grammar. It cannot be
used as a variable binder, pattern binder, or catch binder.

**Contextual keywords** — `ask`, `exec`, and `print` are NOT reserved; they
lex as plain `VAR_NAME` tokens. The scope pass gives them their built-in
meaning.

**Type-annotation keywords** — `text`, `json`, `bool`, `int`, `decimal`,
`list`, `dict`, and `unit` are NOT reserved; they lex as `VAR_NAME` and are
mapped to type nodes in type-annotation positions.

### Identifiers

| Token | Start | Used for |
|-------|-------|----------|
| `TYPE_NAME` | an uppercase letter | Record/enum types, qualified constructors, exception types |
| `VAR_NAME` | a lowercase letter or `_` | Variables, fields, agent names, function names, parameter names |

An identifier starts with a (Unicode) letter or `_` and then greedily
consumes every character that is not whitespace and not a structural
operator/punctuator delimiter (`( ) [ ] { } : , . | ; /`).  Operator
characters such as `- ? ! < > = + *` and the string quotes `" '` are
identifier-continuation characters, so `ask-prompt`, `ask?`, `do-it-now!`,
`a+b`, and `foo"bar` are single tokens; spaces (or a delimiter) around an
operator make it parse as an operator.  See
[Lexical structure](agl/reference/lexical-structure.md) for the full
disambiguation table.

`_` (the wildcard) lexes as a plain `VAR_NAME`; wildcard interpretation
happens in the AST builder.

### Numbers

No float type in AgL. Decimal arithmetic is exact fixed-point.

| Token | Matches |
|-------|---------|
| `INT` | `/[0-9]+/` |
| `DECIMAL` | `/[0-9]+\.[0-9]+/` |

### Operators and punctuation

| Token | Source | Token | Source |
|-------|--------|-------|--------|
| `ARROW` | `=>` | `THIN_ARROW` | `->` |
| `ASSIGN` | `:=` | `EQ` | `=` |
| `NEQ` | `!=` | | |
| `LT` | `<` | `LE` | `<=` |
| `GT` | `>` | `GE` | `>=` |
| `PLUS` | `+` | `MINUS` | `-` |
| `STAR` | `*` | `SLASH` | `/` |
| `LPAR` | `(` | `RPAR` | `)` |
| `LSQB` | `[` | `RSQB` | `]` |
| `LBRACE` | `{` | `RBRACE` | `}` |
| `COLON` | `:` | `COMMA` | `,` |
| `DOT` | `.` | `PIPE` | `\|` |
| `SEMICOLON` | `;` | | |

`THIN_ARROW` (`->`) is a new token in v2. It is used for:
- function return-type annotations: `def f(x: int) -> text = …`
- function type expressions: `(int, int) -> text`
- lambda return types (optional): `fn(x: int) -> text => …`

`ARROW` (`=>`) is used for branch and lambda bodies only.

**Special tokens:**

| Token | Source | Purpose |
|-------|--------|---------|
| `EQ_EQ` | `==` | Error token; not in any grammar rule. The parser emits *"Use `=` for equality."* |
| `LOOP_BOUND` | `[N]` after `do` | Merged from `LSQB INT RSQB` by the lexer to eliminate the LALR(1) conflict with list literals. The token's value is the integer string `N`. |
| `INDEX_LSQB` | adjacent `[` after an expression | Distinguishes indexing (`xs[0]`) from spaced list-literal call sugar (`f [0]`). |

**Module system tokens (contextual / synthetic):**

The following tokens are produced by post-layout lexer passes; they are NOT
reserved keywords and remain valid identifiers outside their promotion windows.

| Token | Promotion condition | Purpose |
|-------|---------------------|---------|
| `IMPORT` | `import` at item-start (after `_NEWLINE`, `_INDENT`, `_DEDENT`, `;`, or stream start) | Leads an `import_decl` |
| `PRIVATE` | `private` at item-start | Visibility modifier on `def`/`record`/`enum`/`type` |
| `QUALIFIED` | `qualified` within an import line | Import mode modifier |
| `USING` | `using` within an import line | Selective import clause |
| `HIDING` | `hiding` within an import line | Exclusion import clause |
| `MODQUAL` | `name(DOT name)*::` where the token after `::` is not `[` | Module qualifier prefix; value is the dotted qualifier string (e.g. `"foo.bar"`) |
| `MODPATH` | `name(DOT name)*` immediately after `IMPORT` | Module path in an import declaration; value is the dotted path string |

`MODQUAL` carries the dotted qualifier string up to but not including `::`.
`DCOLON` is consumed into the `MODQUAL` token.
The `next != [` guard preserves the existing typed-call form `callee::[T](args)`.
A leading `::name` (empty qualifier; self-reference) has no preceding name, so
no merge fires; the bare `DCOLON` is handled by the grammar directly.

---

## Layout Rules

AgL uses significant indentation (Python-style).

1. **Indent stack** starts at `[0]`.
2. **Inside brackets** — `paren_level > 0` when any of `( [ { ${` is open.
   While `paren_level > 0`, `_NEWLINE` tokens are suppressed. Closing
   brackets `} ) ]` decrement the level.
3. **On a `_NEWLINE`:**
   - Deeper than stack top → emit `_INDENT`, push new level.
   - Same depth → emit `_NEWLINE`.
   - Shallower → emit one `_DEDENT` per popped level; if the resulting top
     does not equal the new depth, a `LexError` ("misaligned dedent") is
     raised.
4. **`|`/`else`/`catch`/`until`-continuation rule** — when the first
   significant token on the next line is `|`, `else`, `catch`, or `until`,
   the `_NEWLINE` is suppressed and only the `_DEDENT`s needed to pop the stack
   to levels strictly greater than the keyword's column are emitted. This rule
   lets branches, pipe-less `else` clauses, catch clauses, and `until`
   continuations align with their enclosing keyword without triggering a new
   block.
5. **At EOF** — remaining indent levels are unwound with `_DEDENT` tokens.

---

## Operator Precedence and Associativity

Listed from **lowest** to **highest** binding power (bottom binds tightest).

| Level | Operators / Forms | Associativity |
|-------|-------------------|---------------|
| `case_expr`, `if_expr` | loosest expression forms | — |
| `or` | `or` | left |
| `and` | `and` | left |
| `not` | `not` (unary prefix) | right (unary) |
| comparison | `=` `!=` `<` `<=` `>` `>=` `in` `is` `is not` | **non-associative** |
| additive | `+` `-` | left |
| multiplicative | `*` `/` | left |
| unary minus | `-` (unary prefix) | right (unary) |
| **application** | single-arg sugar `f x` (non-chaining) | — |
| postfix | `.field`, `.Type`, `f(args)` call, `[index]` | left |
| atom | literals, names, `()`, `(expr)`, templates, lambdas | — |

**Application (juxtaposition sugar)** is a new precedence level in v2. It
binds tighter than all binary operators: `print x + 1` parses as
`(print x) + 1`. It is **non-chaining**: `f g x` is a parse error.
`juxt_arg` (the sugar argument) excludes `(`-led forms — so `f(x)` is always
a parenthesized call, never `f` applied to `(x)` — and excludes unary `-`,
so `f -1` is subtraction.

**`case` and `if` expressions** are the lowest-precedence forms. In
`or_expr` positions (branch bodies, `until` conditions, call arguments) a
`case` or `if` expression must be wrapped in parentheses.

**Comparison operators** are **non-associative**: at most one binary
comparison per expression; a second chained operator is a parse error.

### Removed from v1

- The `[options]` bracket cluster after agent/`exec` calls — replaced by
  named arguments inside parentheses.
- The bar-safe statement stratification (`closed_stmt`, `open_stmt`,
  `bar_safe_stmt`, `bar_expr`) — branch bodies and `until` conditions now
  reference `or_expr` directly. A `case` or `if` expression in those
  positions must be parenthesized.
- `print_stmt` as a distinct grammar production — `print` is now a built-in
  function name.
- `pass` as a keyword — replaced by the unit literal `()`.

### New in v2

- `THIN_ARROW` (`->`) — return type and function type arrow.
- `def` and `fn` as reserved keywords.
- `unit` as a non-reserved type keyword.
- `func_type`: `"(" type_list? ")" "->" type_expr`.
- `lambda_expr`: `"fn" "(" params? ")" ("->" type_expr)? "=>" expr`.
- `func_def`: `"def" VAR_NAME "(" params? ")" "->" type_expr "=" expr`.
- `juxt` production for single-arg sugar (non-chaining, concrete rule).
- `"(" ")"` as the unit literal (unified with the empty argument list of a
  zero-arg call).
- Named arguments `VAR_NAME ":" expr` in `arg_list` (reusing the
  constructor shape).

### Module system additions

- `import_decl` as a new top-level declaration: `import MODPATH [.*] [qualified] [as ALIAS] [using…|hiding…]`.
- `MODQUAL::name` — module-qualified value/type/constructor reference.
- `::name` — self-reference (current module).
- `private` modifier on `def`/`record`/`enum`/`type` declarations.
- Soft keywords: `import`, `private`, `qualified`, `using`, `hiding` — contextually promoted by the lexer; remain valid identifiers outside their promotion windows.
