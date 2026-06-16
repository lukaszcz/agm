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

AgL uses significant indentation (Python-style). Newlines separate
statements; an increase in indentation opens a nested block (a *suite*) and a
decrease closes it.

The layout rules:

1. **Indentation width.** Leading spaces count 1 column each; a tab advances
   to the next multiple of 4 columns. A dedent must return to an indentation
   level previously in effect — a misaligned dedent is a lexical error.
2. **Blank lines and comment-only lines** are ignored for layout purposes.
3. **Implicit continuation inside brackets.** While any `(`, `[`, `{`, or
   `${` interpolation is open, newlines do not terminate the statement; the
   logical line continues until the bracket closes. List literals, dictionary
   literals, constructor argument lists, and call-option lists may therefore
   span multiple lines.
4. **Branch-marker continuation.** When the first token of a line is `|`,
   `catch`, or `until`, the line continues the enclosing construct instead of
   starting a new statement, and may align with the enclosing keyword without
   opening a new block. This is what lets `if`/`case` branches, `catch`
   clauses, enum variants, and the `until` clause of a loop sit at the same
   indentation as the construct that owns them:

   ```agl
   if status is Complete => pass
   | status is Blocked => let report = critic "Explain ${status}"
   | else => pass

   do[5]
     let r: Review = reviewer "Review ${artifact}"
   until r is Pass
   ```

A semicolon `;` also separates statements on one line; see
[Program structure](program-structure.md) for the inline-form rules.

## Keywords

The following 28 words are **reserved** and can never be used as variable or
agent names:

```text
record enum type input agent let var set do until if else case of
try catch raise as pass print and or not is in true false null
```

Reserved words are also rejected as field names, with one exception: `agent`
remains legal as a **field name** (record/enum field definitions, named
constructor arguments, dict shorthand keys, and field access) so that the
built-in exception fields `AgentCallError.agent` and `AgentParseError.agent`
stay usable. It still cannot be used as a variable, input, pattern/`catch`
binder, or agent name.

Two further names are **contextual keywords**, not reserved:

- `prompt` — in call position it denotes the built-in default agent
  ([Agent calls](agent-calls.md)).
- `exec` — in call position it denotes the built-in shell executor
  ([Shell execution](shell-execution.md)).

`prompt` and `exec` may not be declared with `let`, `var`, or `input`, may
not be bound by patterns or `catch` binders, and may not be declared as
agent names — but they remain legal as record/enum **field names**:
`Continue(prompt: text)` is a valid variant declaration. To destructure such
a field, bind it under another name — `Continue(prompt: p)` — since the
shorthand pattern `Continue(prompt)` would bind the reserved name and is
rejected.

The built-in type names `text`, `json`, `bool`, `int`, `decimal`, `list`,
and `dict` are likewise **not** reserved; they are recognized contextually in
type positions.

## Identifiers

Identifiers are ASCII:

| Token | Pattern | Used for |
| ----- | ------- | -------- |
| `TYPE_NAME` | `[A-Z][A-Za-z0-9_]*` | Record, enum, alias, and exception type names; enum variant constructors |
| `VAR_NAME` | `[a-z_][A-Za-z0-9_]*` | Variables, fields, inputs, agent names, call options, renderers |

Capitalization is significant: a name starting with an uppercase letter is a
type/constructor name; anything else is a value-level name.

The single underscore `_` is lexically an ordinary `VAR_NAME`; in pattern and
`catch` positions it is interpreted as the wildcard
([Pattern matching](pattern-matching.md)).

## Numbers

There are two numeric token forms and **no floating-point tokens**:

| Token | Pattern | Type |
| ----- | ------- | ---- |
| `INT` | `[0-9]+` | `int` (arbitrary precision) |
| `DECIMAL` | `[0-9]+\.[0-9]+` | `decimal` (exact) |

A decimal literal requires digits on both sides of the dot (`1.5`, not `1.`
or `.5`). There is no exponent notation. Negative numbers are written with
the unary minus operator: `-3` is `-` applied to the literal `3`.

## Strings and templates

All string literals are **templates**: they may contain `${expr}`
interpolation. Both `"` and `'` are valid delimiter characters, giving four
forms:

- `"…"` / `'…'` — single-line; an unescaped newline inside is a lexical error.
- `"""…"""` / `'''…'''` — triple-quoted, multi-line, subject to the dedent rule.

The two delimiter styles are interchangeable: `"it's done"` and `'"quoted"'`
are both legal. The closing delimiter must match the opening one.

The escape sequences are the JSON set plus `\'`, `\"`, and `\$`:

```text
\"  \'  \\  \/  \b  \f  \n  \r  \t  \uXXXX  \$
```

`\'` and `\"` each produce the respective quote character; both work in either
quote style. `\$` produces a literal dollar sign (suppressing interpolation).
A `$` not followed by `{` is literal without any escaping. **Any other
backslash sequence is a lexical error**, as is an incomplete or malformed
`\uXXXX` escape.

Interpolation has the form `${expr}` or `${expr as renderer}`, where
`renderer` is a lowercase identifier. Newlines are not permitted inside an
interpolation. A nested `{ … }` (for example a dictionary literal) inside an
interpolation is balanced correctly and does not close it.

### Triple-quoted dedent

The content of a `"""…"""` or `'''…'''` template is normalized in three steps:

1. Remove one leading newline, if present.
2. Strip the *minimum common indentation* of all non-blank lines. A line
   whose only non-whitespace content is an interpolation hole counts as
   non-blank. Interpolation holes themselves are never dedented; only
   surrounding literal whitespace is stripped.
3. Remove one trailing newline, if present.

This makes the natural layout — content indented to match the surrounding
code, closing `"""` on its own line — produce text without the incidental
indentation:

```agl
let report = critic """
Review the artifact.

Artifact:
${artifact}
"""
```

See [Strings and interpolation](strings-and-interpolation.md) for
interpolation semantics and rendering.

## Operators and punctuation

```text
=>   =   !=   <   <=   >   >=
+   -   *   /
(   )   [   ]   {   }
:   ,   .   |   ;
```

`==` is recognized as a distinct token solely so it can be rejected with the
targeted error **"Use `=` for equality."** — it is not part of the language.

Multi-character operators are matched greedily: `=>`, `!=`, `<=`, `>=`, and
`==` are recognized before `=`, `<`, `>`.

The loop bound `[N]` immediately after `do` is lexed as a single unit so it
never conflicts with a list literal; `N` must be a positive integer
([Control flow](control-flow.md)).

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
| 8 | `.field` access, `.Variant` qualification, constructor payload `( … )` | left |
| 9 | atoms: literals, names, calls, templates, `( expr )` | — |

All comparison operators are non-associative: `x = y = z`, `1 < 2 < 3`, and
`a <= b != c` are parse errors with the targeted diagnostic
**"Comparisons are non-associative; parenthesize explicitly, e.g.
`(x = y) = z`."** Use explicit boolean composition instead:

```agl
if x = y and y = z => pass
```

`case` expressions sit below all of this: they are the loosest expression
form and must be parenthesized in *bar-safe* positions
([Program structure](program-structure.md)).
