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
   to the next multiple of 4 columns. A dedent must return to an active prior
   indentation level — a misaligned dedent is a lexical error.
2. **Blank lines and comment-only lines** are ignored for layout purposes.
3. **Implicit continuation inside brackets.** While any `(`, `[`, `{`, or
   `%{` interpolation is open, newlines do not terminate the item; the
   logical line continues until the bracket closes. Array literals, dictionary
   literals, constructor argument lists, and function call argument lists may
   therefore span multiple lines.
4. **Continuation markers.** None of `|`, `else`, `catch`, `until`, `done`,
   `->`, `=>`, and `=` can begin a construct, so a line whose first token is
   one of them always continues the line before it rather than starting a new
   item, and may sit at any column — including the enclosing construct's —
   without opening a new block.

   This is what lets `if`/`case` branches, `else` branches, `catch` clauses,
   enum members, and the `until`/`done` terminator of a loop align with the
   construct that owns them:

   <!-- agl-check: fragment -->
   ```agl
   if
     | status is Complete => ()
     | status is Blocked => (let report = critic.ask("Explain %{status}"); print report)
     | else => ()

   var r: Review = Pass
   do[5]
     r := reviewer.ask("Review %{artifact}")
   until r is Pass
   ```

   The same rule lets a long signature or binding wrap before its return type,
   its `=`, or a branch arrow:

   ```agl
   def describe(status: text)
       -> text
       = case status of
         | "ok"
           => "all clear"
         | _
           => "needs attention"

   program def main()
       -> unit
       = print (describe "ok")
   ```
5. **Operator continuation.** A line break is ignored wherever it would
   separate an operator from its operand. This holds for every operator —
   built-in, word (`and`, `or`, `not`, `is`, `in`, `to`, `downto`, `step`,
   `with`, `as`), and user-defined — as well as for `.`, `:=`, and a
   signature's `->`.

   A line ending with an operator that still awaits its operand is unfinished,
   so the next line continues it whatever its indentation:

   ```agl
   program def main() -> unit =
     let scaled = [1, 2, 3].map(fn v => v * 2) |>
     .fold(0, fn (total, v) => total + v)
     let wide = 1 +
         2
     print (scaled + wide)
   ```

   A line that is **more indented** than its block and opens with an operator
   continues the line before it instead of opening one:

   ```agl
   program def main() -> unit =
     let report = [1, 2, 3]
       .filter(fn v => v > 1)
       .map(fn v => v * 2)
       |> .fold(0, fn (total, v) => total + v)
     print report
   ```

   The indentation is what makes the second form a continuation. At its block's
   own level a line opening with `.`, `-`, or `not` is an item in its own right,
   so a block ending in a leading-dot method value or a negation keeps its
   meaning:

   ```agl
   record Box(value: int)

   def Box::scale(self, factor: int) -> Box = Box(value = self.value * factor)

   def doubler() -> (Box) -> Box =
     let factor = 2
     .scale(factor)

   program def main() -> unit = print (doubler()(Box(value = 3)).value)
   ```

   Neither form applies to a module header, which spells a path rather than an
   expression.

A semicolon `;` also separates items in a block; see
[Program structure](program-structure.md).

## Keywords

The following words are **always reserved** and can never be used as
variable or function names:

```text
record enum type program def fn let var for while do until done
if else case of try catch raise return break continue exception extends builtin extern as as?
true false null
infixl infixr prio
```

**`as?`** is a single reserved keyword/token — the `?` is part of the
lexeme. There is no whitespace permitted between `as` and `?`; with
whitespace, `as` is the cast keyword and `?` starts a separate placeholder
spelling. `as?` is always reserved and cannot be used as an identifier.

`agent` is an ordinary identifier, including in binding, pattern, and field
positions.

`by` carries no syntactic role: the range stride is spelled `step`.

`self` is a **contextual identifier**, not a keyword. It is special only as the
first parameter of a `def` whose enclosing path resolves to a record, enum,
exception, or built-in receiver type; that includes foreign receiver names
made bare-visible by an import or `use`. Everywhere else it is an ordinary
identifier, including as a field name or an annotated function parameter.

**Contextual keywords** — `print`, `ask`, and `exec` are NOT reserved; they
lex as plain `NAME` tokens and are given their built-in meaning during scope
resolution. They may not be declared with `let` or `var`, may not be
declared as functions or function parameters, and may not appear as pattern or catch
binders — but they remain legal as field and method names, which live in a
type's own member namespace. The distinct raw-tail spellings
`exec$` and `ask$` are reserved for their raw forms and cannot be used as names.

**Type-annotation keywords** — `text`, `json`, `bool`, `int`, `decimal`,
`array`, `dict`, and `unit` are **not** reserved; they are recognized
contextually in type positions. `fn` is reserved (it introduces a lambda).
`def` is reserved (it introduces a function declaration). `builtin` is
reserved for standard-library declarations that are implemented by the host.
`extern` is reserved for declarations implemented by a companion Python file
(see [Python FFI](ffi.md)).

Soft keywords are **not reserved**: each is promoted to its own token only
inside a promotion window, and remains a valid identifier everywhere else.

**Module and scope soft keywords** — `import`, `use`, `export`, `hiding`,
`scope`, and `end` remain valid identifiers in all positions except:

| Keyword | Promoted to | Window |
|---------|-------------|--------|
| `import` | `IMPORT` | At item-start |
| `use` | `USE` | At item-start when followed by a use declaration form |
| `export` | `EXPORT` | At item-start |
| `hiding` | `HIDING` | Within an import, use, or export declaration |
| `scope` | `SCOPE` | At item-start, before a complete `NAME (:: NAME)*` scope path |
| `end` | `END` | At a region's layout level, while that region is open, before a complete closer path ending the item |

A scope closer must repeat exactly the path of the region it closes. At other
layout levels, `end` remains a name;
for example it can be a record field or the first expression in a declaration
suite.

**Operator soft keywords** — `and`, `or`, `not`, `is`, `in`, `to`, `downto`,
`step`, and `with` spell the built-in operators. An operator's spelling only
has to be unambiguous where an operand could stand instead, so each is promoted
in operator position alone:

| Keywords | Promoted to | Window |
|----------|-------------|--------|
| `and` `or` `is` `in` `to` `downto` `step` `with` | `AND` `OR` `IS` `IN` `TO` `DOWNTO` `STEP` `WITH` | Directly after a token that closes an operand |
| `not` | `NOT` | Directly before a token that can start an operand |

A following `=` or `:` marks a name either way — no operand starts with
either — and after `.`, `::` or a declaration keyword only a name can stand.
So `Option::or`, `flag.not()`, `R(step = 2)` and a `with: int` field all name
members the way any other spelling does, while `a or b`, `not ready` and
`1 to 9 step 2` read as operators. The one position an operator word cannot
reach is an enum variant name, where a case branch may legitimately begin with
`not`.

Examples where they remain plain identifiers:

```agl
let import = 1          # 'import' not at item-start → VAR_NAME
let export = "hello"    # 'export' not at item-start → VAR_NAME
let use = "hello"       # 'use' without a declaration form → VAR_NAME
record R(end: int)            # 'end' is a field name, not a closer
```

## Qualifier chains

`::` separates qualifier-chain segments from the member they select. A chain
can begin with a module route, continue through named scopes or types, and end
at a value, type, or enum member:

<!-- agl-check: fragment -->
```agl
foo/bar::thing              # suffix module route, member thing
/foo/bar::thing             # anchored module route
Point::distance             # local scope member
foo/bar::Geometry::Point    # module route, then scope members
::Outer::Inner::name        # path anchored at the current module root
```

A slash-separated path before the first `::` is a module route. A leading `/`
anchors that route to the complete module path; otherwise it may be a suffix
route or an alias. Subsequent `::` segments name scopes or types. A single
leading segment can be either a local scope/type or a module route; use `/` for
the module reading or `::` for the current-module reading when both would
resolve. Scope segments never suffix-match.

Every route and chain segment is byte-adjacent through `::`: `foo/bar::thing`
is a qualifier, while `foo / bar::thing` is division followed by a separate
qualifier. `/` is the division operator only when written with whitespace on
**both** sides. This matches `+`, `-` and `*`, which are identifier characters
and so already need surrounding space to read as operators (`a+b` is one
name). Left unspaced, `/` separates route segments, so a `/` that touches an
operand on exactly one side is rejected:

<!-- agl-check: error -->
```agl
let q = a / b        # division
let r = a/b::thing   # qualifier
let s = a/ b         # error: reads as a path, but the segments are split
let t = a /b         # error: same
```

A type-owning chain segment may carry type arguments, as in
`Option[int]::Some`; type arguments on a plain scope segment are a static
error. The type-argument form `callee::[T]` and typed-call form
`callee::[T](args)` (e.g. `ask::[Review](…)`) instead apply to the
complete callee and are not qualifier segments.
## Identifiers

An identifier starts with a letter (any Unicode letter, not just ASCII) or
`_`, and then continues for as long as the next character is **not** whitespace
and **not** a structural operator/punctuator delimiter.  The delimiter
characters that terminate an identifier are:

```
(  )  [  ]  {  }  :  ,  .  |  ;  /  @  =
```

The string quotes `"` and `'`, and the arithmetic operators `+` and `*`, are
**not** delimiters: they may appear inside an identifier (e.g. `foo"bar`,
`a+b`, `n*x`).  A leading `"` or `'` (or one preceded by whitespace) still
starts a string template because an identifier must begin with a letter or
`_`.

Every other character is an identifier-continuation character.  In particular
the operator characters `-`, `?`, `!`, `$`, `<`, `>` may appear *inside*
an identifier, so names like `ask-prompt`, `ask?`, `exec$`, and `do-it-now!`
scan as a single token.  Note that `=` and `@` **are** delimiters, so `a=b`
scans as three tokens and `@arg-pos` as two — the attribute name that follows
`@` is one identifier, hyphens included.

Operator names are a second lexical class of identifier: the grammar terminal
`OP_NAME`. They start with an operator character and continue while the next
character is also an operator-name character. Operator-name characters are
Unicode punctuation or symbol characters, except AgL structural delimiters:
parentheses, brackets, braces, `:`, `,`, `.`, `;`, quotes, `@`, `#`, and `_`.
Exact reserved operator and punctuation tokens such as `=`, `==`, `!=`, `<`,
`<=`, `>`, `>=`, `->`, `=>`, `:=`, `::`, `+`, `-`, `*`, `/`, `|`, `.`, `:`,
and `@` keep their syntactic meaning.

The placeholder spellings `?` and `?<digits>` (for example `?1` and `?12`)
are also reserved tokens. They are used only as whole call arguments for
partial application ([Functions](functions.md#partial-application)). The
numbered form requires the digits to be immediately adjacent: `?1` is one
numbered placeholder token, while `? 1` is a bare `?` placeholder followed by
an integer literal. A standalone `?` is therefore not available as a user
operator name. Longer operator-name runs that merely contain `?`, such as
`??`, `?=`, `%?`, and `>=>`, are still ordinary `OP_NAME` tokens, and
word-starting identifiers containing `?` after the first character, such as
`valid?` and `ask?`, are unaffected.

Non-reserved standalone runs such as `==>`, `>>`, `|>`, `<|`, `>=>`, `%$`,
`%?`, `~`, and `⊕` are operator names.

AgL has two lexical classes of identifier: `NAME` and `OP_NAME`. Both are
ordinary names in declaration and reference positions, so they can name
variables, functions, and constructors.

| Token | Start | Used for |
| ----- | ----- | -------- |
| `NAME` | a letter (any Unicode letter, not just ASCII) or `_` | Every kind of name: types, constructors, variables, fields, functions, parameters, type parameters |
| `OP_NAME` | an operator-name character | Variables, functions, constructors, and other grammar positions that accept a name |

**Capitalization carries no syntactic or semantic meaning.** The case of an
identifier's first letter never classifies it: `option` and `Option`, `some`
and `Some`, `box` and `Box` are all equally valid as type names, value names,
constructors, or functions. Whether a name denotes a type or a value is
determined entirely by how it is declared and the position it appears in, not
by its spelling. Names are still matched exactly: `Some` and `some` are two
distinct names, not two spellings of one.

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
| `valid?` | `NAME "valid?"` | one identifier (`?` is allowed after the first character) |
| `?` | `PLACEHOLDER "?"` | reserved placeholder spelling, not an operator name |
| `?1` | `PLACEHOLDER_NUM "?1"` | numbered placeholder; no whitespace before the digits |
| `? 1` | `PLACEHOLDER "?"`, `INT "1"` | not a numbered placeholder |
| `??` | `OP_NAME "??"` | longer `?`-containing operator names are unaffected |

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

All string literals are **templates**: they may contain `%{expr}` and
`${NAME}` interpolation. Write `\${` for a literal `${`. Both `"` and `'`
are valid delimiter characters, giving four forms:

- `"…"` / `'…'` — single-line.
- `"""…"""` / `'''…'''` — triple-quoted, multi-line, subject to the dedent rule.

Escape sequences, triple-quoted dedent normalization, and interpolation
semantics are covered in [Strings and interpolation](strings-and-interpolation.md).

## Raw-tail forms

`exec$` and `ask$` begin raw-tail calls, either directly or after a `.`
projection: `receiver.ask$ prompt`. The lexer emits a `RAW_TAIL_NAME`, then
`RAW_TAIL_START`, one or more `RAW_FRAGMENT` and interpolation-token runs, and
`RAW_TAIL_END`. Optional type arguments must be byte-adjacent to the raw name:
`ask$::[T]` and `receiver.ask$::[T]`. In `ask$ ::[T]` or
`receiver.ask$ ::[T]`, the spaced `::[T]` instead begins the payload. The
payload is either the rest of that line or a following indented block. In both
cases it is one template: its text is verbatim except that `%{expr}`
interpolates and `\%{` is a literal `%{`. Inline payloads discard trailing
spaces and tabs; block payloads drop the blank lines that trail the last
content line.

A raw-tail call requires a nonempty inline payload or a block with at least one
nonblank line. It is only recognized at bracket depth zero and must occupy a
line-final expression position. Its payload therefore owns `#`, `;`, quotes,
parentheses, dollar forms, and ordinary backslashes rather than treating them
as AgL syntax. The [Grammar](grammar.md#raw-tail-calls) lists the allowed
positions; [Shell execution](shell-execution.md#raw-tail-exec) and [Agent
calls](agent-calls.md#raw-tail-ask) describe the two forms.

## Operators and punctuation

```text
=>   ->   =   ==   !=   <   <=   >   >=
::   +   -   *   /   @
(   )   [   ]   {   }
:   ,   .   |   ;
```

`->` is the **return/function-type arrow** (distinct from `=>`). It appears
in function type annotations (`int -> text`), `def` return type annotations
(`def f(x: int) -> text = …`), and `fn` lambda return types
(`fn(x: int) -> text => …`). `=>` is the **branch/lambda-body arrow** — it
separates a branch condition or pattern from its body.

`::` serves two distinct roles: as the **qualifier-chain separator** (see
[Qualifier chains](#qualifier-chains) above) and as the **type-argument
introducer** `callee::[Type]` / `callee::[Type](args)`. It is a maximal-munch
token distinct from two `:` delimiters. The uses are disambiguated by context:
a `::` after a tightly written route or chain segment is a qualifier separator;
a `::` following a complete callee and immediately followed by `[` introduces
type arguments.

`==` is the **equality operator** (with `!=` for inequality). A single `=` is
never a comparison: it separates a binder or named argument from its value
(`let x = …`, `f(name = …)`, `R(field = …)`), and `:=` is destructive
assignment.

Multi-character operators are matched greedily.

A `[` that immediately follows `do` — with or without intervening whitespace
(`do[n]` and `do [n]` are equivalent) — opens the loop bound `[expr]`. This
is what distinguishes the bound from an array literal that could otherwise begin
the loop body. As a consequence, a `do` body cannot itself *begin* with a bare
array literal; parenthesize it (`do ([item1, item2]) until …`) if needed.

An adjacent `[` after an expression-ending token starts indexing. Whitespace
keeps the bracket as an array literal, so `xs[0]` indexes while `f [0]` is the
single-argument call sugar `f([0])`.

## Attributes

`@` introduces an **attribute** on a declaration — `@name`, or `@name(args)`
with an ordinary call argument list. `@` is a delimiter, so the name after it
is a separate identifier token and the attribute name may be hyphenated. The
same spelling is an ordinary identifier everywhere else.

An attribute prefixes what it belongs to, and several attributes may be written
in a row. A declaration, a field, and a parameter each take their attributes on
the line above or in front of them on the same line. An enum member takes its
attributes after the member's `|`, on the member's own line.

```agl
@arg-pos
def f(x: int, y: int) -> int = x + y

@doc("Formats one entry.")
def g(a: int, @arg-named key: text) -> text = "%{a}: %{key}"

record Entry
  @arg-named
  label: text
  @arg-named count: int

enum Shape
  | @doc("a rectangle") @arg-pos Rect(width: int, height: int)
  | Empty
```

Catalog and semantics: [Attributes](attributes.md).

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
| 11 | atoms: literals, names, `( expr )`, `(op)` operator values, `()` unit, templates, `break`, `continue` | — |

The record-update operator `with` binds looser than every level in the table,
on both sides; see
[Record update](expressions.md#record-update).

### Prelude combinators

The standard library provides four functional combinators through the prelude:
`|>` is left-associative at priority 5 and passes a value to a function; `<|`
is right-associative at priority 4 and applies a function to a value; `>>` is
left-associative at priority 60 and composes functions left-to-right; `<<` is
right-associative at priority 60 and composes them right-to-left. Thus
`increment <| 2 |> double` is `increment(double(2))`. A chain cannot mix `>>`
and `<<` without parentheses because they have opposite associativity at the
same priority.

User-defined symbolic infix operators are declared with `infixl` or `infixr`:

```agl
infixl |> at 45
infixr << at prio > + 1
```

Priorities are integers where lower numbers bind looser and higher numbers bind
tighter. A priority can be a literal integer or relative to an existing builtin,
local operator, operator made bare-visible by an import wildcard or tail, or an
operator member made bare by `use` (with the `std/prelude` prelude included);
omitted priority uses the `+`/`-` level. A plain qualified import does not make
an operator's fixity available. User infix
use lowers to a normal two-argument function call, so the operator must also be
declared as a function with the same name. Two visible declarations for one
operator must agree on fixity, and operators at one priority cannot mix left and
right associativity in a chain.

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
`print(>>)`, not `print >>`. A built-in symbolic operator is a value only in
parentheses of its own, such as `(+)`; see
[Operators as function values](expressions.md#operators-as-function-values).

**Calls with parentheses (level 10)** are left-associative postfix and
support multiple arguments: `f(a, b)`.

`case` and `if` expressions sit **below all of this**: they are the loosest
expression forms. In positions where a following `|` would be ambiguous
(branch bodies, `if`/`until` conditions) they must be parenthesized. A loop
body is not such a position — the loop terminator closes it — so they may
appear there directly.

All comparison operators are non-associative: `x == y == z`, `1 < 2 < 3`, and
`a <= b != c` are parse errors with a targeted diagnostic.
