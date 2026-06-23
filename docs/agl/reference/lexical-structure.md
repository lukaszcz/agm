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
   `else`, `catch`, or `until`, the line continues the enclosing construct
   instead of starting a new item, and may align with the enclosing keyword
   without opening a new block. This is what lets `if`/`case` branches, `else`
   branches, `catch` clauses, enum variants, and the `until` clause of a loop
   sit at the same indentation as the construct that owns them:

   ```agl
   if
     | status is Complete => ()
     | status is Blocked => let report = ask("Explain ${status}", agent: critic)
     | else => ()

   do[5]
     let r: Review = ask("Review ${artifact}", agent: reviewer)
   until r is Pass
   ```

A semicolon `;` also separates items in a block; see
[Program structure](program-structure.md).

## Keywords

The following words are **always reserved** and can never be used as
variable, agent, or function names:

```text
record enum type param program agent config def fn let var do until if else
case of try catch raise as as? and or not is in true false null
```

**`as?`** is a single reserved keyword/token — the `?` is part of the
lexeme. There is no whitespace permitted between `as` and `?`; with
whitespace, `as` is the cast keyword and `?` would start a separate (invalid)
token. `as?` is always reserved and cannot be used as an identifier.

**Removed from v1:** `pass` and `print` are **no longer reserved** in v2.
`pass`'s role is taken by the unit literal `()`. `print` is a built-in
function name, looked up like any variable — it is a contextual keyword, not
a reserved word.

`agent` is reserved (it leads an `agent` declaration) but is still accepted
as a **field name** (record/enum field definitions, named constructor
arguments, dict shorthand keys, and postfix field access). It cannot be used
as a variable binder, pattern binder, or catch binder.

**Contextual keywords** — `ask` and `exec` are NOT reserved; they lex as
plain `NAME` tokens and are given their built-in meaning during scope
resolution. They may not be declared with `let`, `var`, or `param`, may not be
declared as agents or functions, and may not appear as pattern or catch
binders — but they remain legal as field names.

**Type-annotation keywords** — `text`, `json`, `bool`, `int`, `decimal`,
`list`, `dict`, and `unit` are **not** reserved; they are recognized
contextually in type positions. `fn` is reserved (it introduces a lambda).
`def` is reserved (it introduces a function declaration).

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
foo.bar::Color.Red   # module foo.bar, enum Color, variant Red
```

A qualifier is a dotted module path followed by `::` immediately before the
name it qualifies. Module path segments are dot-separated lowercase names.
A leading `::` with no preceding path is the **self-reference** form — it
refers to the current module.

The typed-call form `callee::[T](args)` (e.g. `ask-request::[Review](…)`) is
a distinct construct — it is NOT a module qualifier.

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

There is a **single** lexical class of identifier — the grammar terminal
`NAME`. It is used for every name in the language: type names, record/enum
declarations and their variant constructors, type aliases, variables, fields,
agent names, function names, parameter names, and type-parameter names.

| Token | Start | Used for |
| ----- | ----- | -------- |
| `NAME` | a letter (any Unicode letter, not just ASCII) or `_` | Every kind of name: types, constructors, variables, fields, agents, functions, parameters, type parameters |

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

Because the operator characters `-`, `<`, `>`, `=`, `!`, `+`, `*` are also
identifier-continuation characters, whether such a run is an identifier or a
sequence of operator tokens depends entirely on **whitespace** (or another
delimiter).  Whitespace and the structural punctuators above are the only
characters that break an identifier scan.

| Source | Tokens | |
| ----- | ----- | - |
| `ask-prompt` | `NAME "ask-prompt"` | one identifier |
| `a - b` | `NAME "a"`, `MINUS "-"`, `NAME "b"` | spaces break the identifier, `-` is an operator |
| `a.b` | `NAME "a"`, `DOT "."`, `NAME "b"` | `.` is a delimiter, always an operator |
| `a -> b` | `NAME "a"`, `THIN_ARROW "->"`, `NAME "b"` | arrow operator, whitespace-delimited |
| `a->b` | `NAME "a->b"` | one identifier (no spaces) |
| `x != 3` | `NAME "x"`, `NEQ "!="`, `INT "3"` | not-equal operator, whitespace-delimited |
| `x!=3` | `NAME "x!=3"` | one identifier (no spaces) |
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
=>   ->   =   !=   <   <=   >   >=
::   +   -   *   /
(   )   [   ]   {   }
:   ,   .   |   ;
```

`->` is the **return/function-type arrow** (distinct from `=>`). It appears
in function type annotations (`(int) -> text`), `def` return type annotations
(`def f(x: int) -> text = …`), and `fn` lambda return types
(`fn(x: int) -> text => …`). `=>` is the **branch/lambda-body arrow** — it
separates a branch condition or pattern from its body.

`::` serves two distinct roles: as the **module-qualifier separator** (see
[Module qualifiers](#module-qualifiers) above) and as the **typed-call
introducer** `callee::[Type](args)` (e.g. `ask-request::[Review](…)`). It is
a maximal-munch token distinct from two `:` delimiters. The two uses are
disambiguated by context: a `::` immediately preceded by a name or dotted path
is the qualifier form; a `::` following a `VAR_NAME` and immediately followed
by `[` is the typed-call form.

`==` is recognized as a distinct token solely so it can be rejected with
the targeted error **"Use `=` for equality."** — it is not part of the
language.

Multi-character operators are matched greedily.

The loop bound `[N]` immediately after `do` is lexed as a single unit so it
never conflicts with a list literal.

An adjacent `[` after an expression-ending token starts indexing. Whitespace
keeps the bracket as a list literal, so `xs[0]` indexes while `f [0]` is the
single-argument call sugar `f([0])`.

## Operator precedence

From loosest to tightest binding (the bottom binds tightest):

| Level | Operators | Associativity |
| ----- | --------- | ------------- |
| 1 | `or` | left |
| 2 | `and` | left |
| 3 | `not` (prefix) | — |
| 4 | `=` `!=` `<` `<=` `>` `>=` `in` `is` `is not` | **non-associative** |
| 5 | `+` `-` | left |
| 6 | `*` `/` | left |
| 7 | `as` `as?` (cast / convertibility test) | left |
| 8 | `-` (unary prefix) | — |
| 9 | function application (single-arg sugar) | **non-chaining** |
| 10 | `.field` access, `.Variant` qualification, `[index]`, `( args )` call | left |
| 11 | atoms: literals, names, `( expr )`, `()` unit, templates | — |

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

**Calls with parentheses (level 10)** are left-associative postfix and
support multiple arguments: `f(a, b)`, `f(a)(b)` (curried — not yet
supported, but syntactically the grammar allows it for future use).

`case` and `if` expressions sit **below all of this**: they are the loosest
expression forms. In positions where a following `|` would be ambiguous
(branch bodies, `if`/`until` conditions) they must be parenthesized.

All comparison operators are non-associative: `x = y = z`, `1 < 2 < 3`, and
`a <= b != c` are parse errors with a targeted diagnostic.
