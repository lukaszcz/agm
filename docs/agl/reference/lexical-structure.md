# Lexical Structure

[← Index](index.md)

## Source text

AgL source is Unicode text. Line endings are normalized before scanning:
every `\r\n` and every lone `\r` is treated as a single `\n`. Source
locations (lines and columns) are 1-based.

## Comments

A `#` begins a comment that runs to the end of the line. There are no block
comments.

```agl
# This is a comment
let x = 1   # so is this
```

## Layout: indentation, newlines, continuation

AgL uses significant indentation (Python-style). Newlines separate items in
a block; an increase in indentation opens a nested block and a decrease
closes it.

The layout rules:

1. **Indentation width.** Leading spaces count 1 column each; a tab advances
   to the next multiple of 4 columns. A dedent must return to a level
   previously in effect — a misaligned dedent is a lexical error.
2. **Blank lines and comment-only lines** are ignored for layout purposes.
3. **Implicit continuation inside brackets.** While any `(`, `[`, `{`, or
   `${` interpolation is open, newlines do not terminate the item; the
   logical line continues until the bracket closes. List literals, dictionary
   literals, constructor argument lists, and function call argument lists may
   therefore span multiple lines.
4. **Branch-marker continuation.** When the first token of a line is `|`,
   `else`, `catch`, `until`, or `done`, the line continues the enclosing
   construct instead of starting a new item, and may align with the enclosing
   keyword without opening a new block. This is what lets `if`/`case` branches,
   `else` branches, `catch` clauses, enum variants, and the `until`/`done`
   terminator of a loop sit at the same indentation as the construct that owns
   them:

   ```agl
   if
     | status is Complete => ()
     | status is Blocked => let report = ask("Explain ${status}", agent = critic)
     | else => ()

   do[5]
     let r: Review = ask("Review ${artifact}", agent = reviewer)
   until r is Pass
   ```

A semicolon `;` also separates items in a block; see
[Program structure](program-structure.md).

## Keywords

The following words are **always reserved** and can never be used as
variable, agent, or function names:

```text
record enum type param program agent config def fn let var for while do until done
if else case of try catch raise return break continue exception extends builtin extern as as?
and or not is in to downto by true false null
```

**`as?`** is a single reserved keyword/token — the `?` is part of the
lexeme. There is no whitespace permitted between `as` and `?`; with
whitespace, `as` is the cast keyword and `?` starts a separate operator name.
`as?` is always reserved and cannot be used as an identifier.

`agent` is reserved (it leads an `agent` declaration) but is accepted
as a **field name** (record/enum field definitions, named constructor
arguments, dict shorthand keys, postfix field access, and pattern field keys).
It cannot be used as a variable binder, pattern binder, or catch binder.

`to`, `downto`, and `by` are reserved (they introduce the range tail of a
`for` clause) but are still accepted as **field names** (record/enum field
definitions, named constructor arguments, dict shorthand keys, postfix field
access, and pattern field keys). They cannot be used as variable, pattern, or
catch binders. This preserves existing uses such as `tagged(by: value)`.

**Contextual keywords** — `print`, `ask`, and `exec` are NOT reserved; they
lex as plain `NAME` tokens and are given their built-in meaning during scope
resolution. They may not be declared with `let`, `var`, or `param`, may not be
declared as agents or functions, and may not appear as pattern or catch
binders — but they remain legal as field names.

**Type-annotation keywords** — `text`, `json`, `bool`, `int`, `decimal`,
`list`, `dict`, and `unit` are **not** reserved; they are recognized
contextually in type positions. `fn` is reserved (it introduces a lambda).
`def` is reserved (it introduces a function declaration). `builtin` is
reserved for standard-library declarations that are implemented by the host.
`extern` is reserved for declarations implemented by a companion Python file
(see [Python FFI](ffi.md)).

**Module system soft keywords** — `import`, `private`, `qualified`, `using`,
and `hiding` are **not reserved**. They remain valid identifiers in all
positions except:

| Keyword | Promoted to | Window |
|---------|-------------|--------|
| `import` | `IMPORT` | At item-start (after newline, indent, dedent, `;`, or stream start) |
| `private` | `PRIVATE` | At item-start |
| `qualified` | `QUALIFIED` | Within an import line (after `import` keyword, before the next newline or `;`) |
| `using` | `USING` | Within an import line |
| `hiding` | `HIDING` | Within an import line |

Examples where they remain plain identifiers:

```agl
let import = 1          # 'import' not at item-start → VAR_NAME
let using = "hello"     # 'using' not in import line → VAR_NAME
def private() -> text = "x"  # 'private' not at item-start → VAR_NAME
```

## Module qualifiers

`::` separates a **module qualifier** from the name it qualifies. In value
and type position:

```agl
foo.bar::thing       # module foo.bar, name thing
::name               # current module, name name (self-reference)
A.baz::y             # alias-rooted qualifier, name y
foo.bar::Color::Red   # module foo.bar, enum Color, variant Red
```

A qualifier is a dotted module path followed by `::` immediately before the
name it qualifies. Module path segments are dot-separated lowercase names.
A leading `::` with no preceding path is the **self-reference** form — it
refers to the current module.

The type-argument form `callee::[T]` and typed-call form `callee::[T](args)`
(e.g. `ask-request::[Review](…)`) are distinct constructs — they are NOT module
qualifiers.

## Identifiers

An identifier starts with a letter (any Unicode letter, not just ASCII) or
`_`, and then continues for as long as the next character is **not** whitespace
and **not** a structural operator/punctuator delimiter.  The delimiter
characters that terminate an identifier are:

```
(  )  [  ]  {  }  :  ,  .  |  ;  /
```

The string quotes `"` and `'`, and the arithmetic operators `+` and `*`, are
**not** delimiters: they may appear inside an identifier (e.g. `foo"bar`,
`a+b`, `n*x`).  A leading `"` or `'` (or one preceded by whitespace) still
starts a string template because an identifier must begin with a letter or
`_`.

Every other character is an identifier-continuation character.  In particular
the operator characters `-`, `?`, `!`, `<`, `>`, `=` may appear *inside*
an identifier, so names like `ask-prompt`, `ask?`, and `do-it-now!` scan as a
single token.

Operator names are a second lexical class of identifier: the grammar terminal
`OP_NAME`. They start with an operator character and continue while the next
character is also an operator-name character. Operator-name characters are
Unicode punctuation or symbol characters, except AgL structural delimiters:
parentheses, brackets, braces, `:`, `,`, `.`, `;`, quotes, `@`, `#`, and `_`.
Exact reserved operator and punctuation tokens such as `=`, `==`, `!=`, `<`,
`<=`, `>`, `>=`, `->`, `=>`, `:=`, `::`, `+`, `-`, `*`, `/`, `|`, `.`, `:`,
and `@` keep their syntactic meaning. Non-reserved standalone runs such as
`==>`, `>>`, `|>`, `<|`, `>=>`, `%$`, `%?`, `~`, and `⊕` are operator names.

AgL has two lexical classes of identifier: `NAME` and `OP_NAME`. Both are
ordinary names in declaration and reference positions, so they can name
variables, functions, and constructors.

| Token | Start | Used for |
| ----- | ----- | -------- |
| `NAME` | a letter (any Unicode letter, not just ASCII) or `_` | Every kind of name: types, constructors, variables, fields, agents, functions, parameters, type parameters |
| `OP_NAME` | an operator-name character | Variables, functions, constructors, and other grammar positions that accept a name |

**Capitalization carries no syntactic or semantic meaning.** The case of an
identifier's first letter never classifies it: `option` and `Option`, `some`
and `Some`, `box` and `Box` are all equally valid as type names, value names,
constructors, or functions. Whether a name denotes a type or a value is
determined entirely by how it is declared and the position it appears in, not
by its spelling.

Type names and value names live in **separate namespaces**, so a `record` or
`enum` declaration may introduce a type name and a same-spelled value
constructor without collision (see
[Bindings and scope](bindings-and-scope.md)).

The single underscore `_` is lexically an ordinary `NAME`; in pattern
and `catch` positions it is interpreted as the wildcard
([Pattern matching](pattern-matching.md)).

### Operator disambiguation

Because many operator characters are also identifier-continuation characters,
whether such a run is part of a word-starting `NAME`, a standalone `OP_NAME`,
or a sequence of operator tokens depends on maximal munch plus reserved-token
disambiguation.

| Source | Tokens | |
| ----- | ----- | - |
| `ask-prompt` | `NAME "ask-prompt"` | one identifier |
| `a - b` | `NAME "a"`, `MINUS "-"`, `NAME "b"` | spaces break the identifier, `-` is an operator |
| `a.b` | `NAME "a"`, `DOT "."`, `NAME "b"` | `.` is a delimiter, always an operator |
| `a -> b` | `NAME "a"`, `THIN_ARROW "->"`, `NAME "b"` | arrow operator, whitespace-delimited |
| `a->b` | `NAME "a->b"` | one identifier (no spaces) |
| `x == 3` | `NAME "x"`, `EQ_EQ "=="`, `INT "3"` | equality operator, whitespace-delimited |
| `x != 3` | `NAME "x"`, `NEQ "!="`, `INT "3"` | not-equal operator, whitespace-delimited |
| `>>` | `OP_NAME ">>"` | standalone operator name |
| `|>` | `OP_NAME "|>"` | standalone operator name |
| `%$` | `OP_NAME "%$"` | standalone operator name |
| `a+b` | `NAME "a+b"` | one identifier (`+` is not a delimiter) |
| `a + b` | `NAME "a"`, `PLUS "+"`, `NAME "b"` | spaces break the identifier, `+` is an operator |
| `n*x` | `NAME "n*x"` | one identifier (`*` is not a delimiter) |
| `foo"bar` | `NAME "foo\"bar"` | one identifier (`"` is not a delimiter) |

This mirrors a Lisp-like maximal-munch identifier rule: scan for as long as
possible until a disallowed character.  Use spaces around operators when you
want them parsed as operators.

## Numbers

There are two numeric token forms and **no floating-point tokens**:

| Token | Pattern | Type |
| ----- | ------- | ---- |
| `INT` | `[0-9]+` | `int` (arbitrary precision) |
| `DECIMAL` | `[0-9]+\.[0-9]+` | `decimal` (exact) |

A decimal literal requires digits on both sides of the dot. There is no
exponent notation. Negative numbers are written with the unary minus
operator: `-3` is `-` applied to the literal `3`.

## Strings and templates

All string literals are **templates**: they may contain `${expr}`
interpolation. Both `"` and `'` are valid delimiter characters, giving four
forms:

- `"…"` / `'…'` — single-line.
- `"""…"""` / `'''…'''` — triple-quoted, multi-line, subject to the dedent rule.

Escape sequences, triple-quoted dedent normalization, and interpolation
semantics are covered in [Strings and interpolation](strings-and-interpolation.md).

## Operators and punctuation

```text
=>   ->   =   ==   !=   <   <=   >   >=
::   +   -   *   /   @
(   )   [   ]   {   }
:   ,   .   |   ;
```

`->` is the **return/function-type arrow** (distinct from `=>`). It appears
in function type annotations (`(int) -> text`), `def` return type annotations
(`def f(x: int) -> text = …`), and `fn` lambda return types
(`fn(x: int) -> text => …`). `=>` is the **branch/lambda-body arrow** — it
separates a branch condition or pattern from its body.

`::` serves two distinct roles: as the **module-qualifier separator** (see
[Module qualifiers](#module-qualifiers) above) and as the **type-argument
introducer** `callee::[Type]` / `callee::[Type](args)`. It is
a maximal-munch token distinct from two `:` delimiters. The two uses are
disambiguated by context: a `::` immediately preceded by a name or dotted path
is the qualifier form; a `::` following a `NAME` and immediately followed
by `[` is the type-argument form.

`==` is the **equality operator** (with `!=` for inequality). A single `=` is
never a comparison: it separates a binder or named argument from its value
(`let x = …`, `f(name = …)`, `R(field = …)`), and `:=` is destructive
assignment.

Multi-character operators are matched greedily.

A `[` that immediately follows `do` — with or without intervening whitespace
(`do[n]` and `do [n]` are equivalent) — opens the loop bound `[expr]`. This
is what distinguishes the bound from a list literal that could otherwise begin
the loop body. As a consequence, a `do` body cannot itself *begin* with a bare
list literal; parenthesize it (`do ([item1, item2]) until …`) if needed.

An adjacent `[` after an expression-ending token starts indexing. Whitespace
keeps the bracket as a list literal, so `xs[0]` indexes while `f [0]` is the
single-argument call sugar `f([0])`.

## Zone markers

`@` is a token used exclusively in **zone markers** inside parameter and field lists.
The three markers are:

| Marker | Equivalent | Zone opened |
|--------|-----------|------------|
| `@pos` | (none) | Positional-only (must be first in the list) |
| `@std` | `/` | Standard (positional-or-named) |
| `@named` | `*` | Named-only |

`pos`, `std`, and `named` are **ordinary identifiers** everywhere except
immediately after `@` inside a parameter or field list. An unrecognized name
after `@` (e.g. `@foo`) is a static error.

```agl
def f(x: int, @std, y: int) -> int = x + y   # @std same as /
def g(a: int, /, b: int, @named, c: int) -> int = ...  # mixing / and @named
```

See [Functions](functions.md) and [Types](types.md) for the full zone semantics.

## Operator precedence

From loosest to tightest binding (the bottom binds tightest):

| Level | Operators | Associativity |
| ----- | --------- | ------------- |
| 1 | `or` | left |
| 2 | `and` | left |
| 3 | `not` (prefix) | — |
| 4 | `==` `!=` `<` `<=` `>` `>=` `in` `is` `is not` | **non-associative** |
| 5 | `+` `-` | left |
| 6 | `*` `/` | left |
| 7 | `as` `as?` (cast / convertibility test) | left |
| 8 | `-` (unary prefix) | — |
| 9 | function application (single-arg sugar) | **non-chaining** |
| 10 | `.field` access, `[index]`, `( args )` call | left |
| 11 | atoms: literals, names, `( expr )`, `()` unit, templates | — |

User-defined symbolic infix operators are declared with `infixl` or `infixr`:

```agl
infixl |> at 45
infixr << at prio > + 1
```

Priorities are integers where lower numbers bind looser and higher numbers bind
tighter. A priority can be a literal integer or relative to an existing operator
with `prio <op> + N` / `prio <op> - N`; omitted priority uses the `+`/`-` level.
User infix use lowers to a normal two-argument function call, so the operator
must also be declared as a function with the same name.

**Cast operators (level 7)** — `as` and `as?` — sit between unary `-` and
`* /`. They are left-associative: `x as json as text` = `(x as json) as text`.
See [Types](types.md#casts-and-convertibility) and
[Expressions](expressions.md#casts-as-and-as) for semantics and examples.

**Application (level 9)** is the single-argument call sugar (`print x`,
`ask "…"`, `f val`). It binds tighter than all binary operators:
`print x + 1` parses as `(print x) + 1`. Application is **non-chaining**:
`f g x` is a parse error — only one juxtaposition per expression. A nested
postfix call can be the single sugar argument, so `f g(x)` parses as
`f(g(x))`.

Because `OP_NAME` after an expression is parsed as an infix operator position,
an operator-name value used as an argument should be parenthesized:
`print(>>)`, not `print >>`.

**Calls with parentheses (level 10)** are left-associative postfix and
support multiple arguments: `f(a, b)`.

`case` and `if` expressions sit **below all of this**: they are the loosest
expression forms. In positions where a following `|` would be ambiguous
(branch bodies, `if`/`until` conditions) they must be parenthesized.

All comparison operators are non-associative: `x == y == z`, `1 < 2 < 3`, and
`a <= b != c` are parse errors with a targeted diagnostic.
