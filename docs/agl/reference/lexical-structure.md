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
   `catch`, or `until`, the line continues the enclosing construct instead of
   starting a new item, and may align with the enclosing keyword without
   opening a new block. This is what lets `if`/`case` branches, `catch`
   clauses, enum variants, and the `until` clause of a loop sit at the same
   indentation as the construct that owns them:

   ```agl
   if | status is Complete => ()
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
record enum type input agent config def fn let var set do until if else
case of try catch raise as and or not is in true false null unit
```

**Removed from v1:** `pass` and `print` are **no longer reserved** in v2.
`pass`'s role is taken by the unit literal `()`. `print` is a built-in
function name, looked up like any variable — it is a contextual keyword, not
a reserved word.

`agent` is reserved (it leads an `agent` declaration) but is still accepted
as a **field name** (record/enum field definitions, named constructor
arguments, dict shorthand keys, and postfix field access). It cannot be used
as a variable binder, pattern binder, or catch binder.

**Contextual keywords** — `ask` and `exec` are NOT reserved; they lex as
plain `VAR_NAME` tokens and are given their built-in meaning by the scope
pass. They may not be declared with `let`, `var`, or `input`, may not be
declared as agents or functions, and may not appear as pattern or catch
binders — but they remain legal as field names.

**Type-annotation keywords** — `text`, `json`, `bool`, `int`, `decimal`,
`list`, `dict`, and `unit` are **not** reserved; they are recognized
contextually in type positions. `fn` is reserved (it introduces a lambda).
`def` is reserved (it introduces a function declaration).

## Identifiers

Identifiers are ASCII:

| Token | Pattern | Used for |
| ----- | ------- | -------- |
| `TYPE_NAME` | `[A-Z][A-Za-z0-9_]*` | Record, enum, alias, and exception type names; enum variant constructors |
| `VAR_NAME` | `[a-z_][A-Za-z0-9_]*` | Variables, fields, inputs, agent names, function names, parameter names |

Capitalization is significant: a name starting with an uppercase letter is a
type/constructor name; anything else is a value-level name.

The single underscore `_` is lexically an ordinary `VAR_NAME`; in pattern
and `catch` positions it is interpreted as the wildcard
([Pattern matching](pattern-matching.md)).

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
+   -   *   /
(   )   [   ]   {   }
:   ,   .   |   ;
```

`->` is the **return/function-type arrow** (distinct from `=>`). It appears
in function type annotations (`(int) -> text`), `def` return type annotations
(`def f(x: int) -> text = …`), and `fn` lambda return types
(`fn(x: int) -> text => …`). `=>` is the **branch/lambda-body arrow** — it
separates a branch condition or pattern from its body.

`==` is recognized as a distinct token solely so it can be rejected with
the targeted error **"Use `=` for equality."** — it is not part of the
language.

Multi-character operators are matched greedily.

The loop bound `[N]` immediately after `do` is lexed as a single unit so it
never conflicts with a list literal.

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
| 7 | `-` (unary prefix) | — |
| 8 | function application (single-arg sugar) | **non-chaining** |
| 9 | `.field` access, `.Variant` qualification, `( args )` call | left |
| 10 | atoms: literals, names, `( expr )`, `()` unit, templates | — |

**Application (level 8)** is the single-argument call sugar (`print x`,
`ask "…"`, `f val`). It binds tighter than all binary operators:
`print x + 1` parses as `(print x) + 1`. Application is **non-chaining**:
`f g x` is a parse error — only one juxtaposition per expression. To chain,
use parentheses: `f(g(x))`.

**Calls with parentheses (level 9)** are left-associative postfix and
support multiple arguments: `f(a, b)`, `f(a)(b)` (curried — not yet
supported, but syntactically the grammar allows it for future use).

`case` and `if` expressions sit **below all of this**: they are the loosest
expression forms. In positions where a following `|` would be ambiguous
(branch bodies, `if`/`until` conditions) they must be parenthesized.

All comparison operators are non-associative: `x = y = z`, `1 < 2 < 3`, and
`a <= b != c` are parse errors with a targeted diagnostic.
