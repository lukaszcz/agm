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
   `%{` interpolation is open, newlines do not terminate the item; the
   logical line continues until the bracket closes. Array literals, dictionary
   literals, constructor argument lists, and function call argument lists may
   therefore span multiple lines.
4. **Branch-marker continuation.** When the first token of a line is `|`,
   `else`, `catch`, `until`, or `done`, the line continues the enclosing
   construct instead of starting a new item, and may align with the enclosing
   keyword without opening a new block. This is what lets `if`/`case` branches,
   `else` branches, `catch` clauses, enum variants, and the `until`/`done`
   terminator of a loop sit at the same indentation as the construct that owns
   them:

   <!-- agl-check: fragment -->
   ```agl
   if
     | status is Complete => ()
     | status is Blocked => (let report = ask("Explain %{status}", agent = critic); print report)
     | else => ()

   var r: Review = Pass
   do[5]
     r := ask("Review %{artifact}", agent = reviewer)
   until r is Pass
   ```

A semicolon `;` also separates items in a block; see
[Program structure](program-structure.md).

## Keywords

The following words are **always reserved** and can never be used as
variable, agent, or function names:

```text
record enum type param program agent def fn let var for while do until done
if else case of try catch raise return break continue exception extends builtin extern as as?
and or not is in to downto by with true false null
infixl infixr prio
```

**`as?`** is a single reserved keyword/token — the `?` is part of the
lexeme. There is no whitespace permitted between `as` and `?`; with
whitespace, `as` is the cast keyword and `?` starts a separate placeholder
spelling. `as?` is always reserved and cannot be used as an identifier.

`agent` is reserved (it leads an `agent` declaration) but is accepted
as a **field name** (record/enum field definitions, named constructor
arguments, dict shorthand keys, postfix field access, and pattern field keys).
It cannot be used as a variable binder, pattern binder, or catch binder.

`to`, `downto`, and `by` are reserved (they introduce the range tail of a
`for` clause) but are still accepted as **field names** (record/enum field
definitions, named constructor arguments, dict shorthand keys, postfix field
access, and pattern field keys). They cannot be used as variable, pattern, or
catch binders. This preserves existing uses such as `tagged(by: value)`.

`with` (the record-update operator) is fully reserved: unlike `to`, `downto`,
and `by`, it is not accepted as a field name.

**Contextual keywords** — `print`, `ask`, and `exec` are NOT reserved; they
lex as plain `NAME` tokens and are given their built-in meaning during scope
resolution. They may not be declared with `let`, `var`, or `param`, may not be
declared as agents or functions, and may not appear as pattern or catch
binders — but they remain legal as field names. The distinct raw-tail spellings
`exec!` and `ask!` are reserved for their raw forms and cannot be used as names.

**Type-annotation keywords** — `text`, `json`, `bool`, `int`, `decimal`,
`array`, `dict`, and `unit` are **not** reserved; they are recognized
contextually in type positions. `fn` is reserved (it introduces a lambda).
`def` is reserved (it introduces a function declaration). `builtin` is
reserved for standard-library declarations that are implemented by the host.
`extern` is reserved for declarations implemented by a companion Python file
(see [Python FFI](ffi.md)).

**Module and scope soft keywords** — `open`, `import`, `export`, `using`,
`hiding`, `scope`, and `end` are **not reserved**. They remain valid
identifiers in all positions except:

| Keyword | Promoted to | Window |
|---------|-------------|--------|
| `open` | `OPEN` | At item-start, before an import or scope reference |
| `import` | `IMPORT` | At item-start, or directly after `open` |
| `export` | `EXPORT` | At item-start |
| `using` | `USING` | Within an import, export, or open declaration |
| `hiding` | `HIDING` | Within an import, export, or open declaration |
| `scope` | `SCOPE` | At item-start, before a complete `NAME (:: NAME)*` scope path |
| `end` | `END` | At a region's layout level, while that region is open, before a complete closer path ending the item |

A scope closer must repeat exactly the path of the region it closes. At other
layout levels, `end` remains a name;
for example it can be a record field or the first expression in a declaration
suite.

Examples where they remain plain identifiers:

```agl
let import = 1          # 'import' not at item-start → VAR_NAME
let export = "hello"    # 'export' not at item-start → VAR_NAME
let using = "hello"     # 'using' not in an import/export declaration → VAR_NAME
record R(end: int)            # 'end' is a field name, not a closer
```

## Qualifier chains

`::` separates qualifier-chain segments from the member they select. A chain
can begin with a module route, continue through named scopes or types, and end
at a value, type, or enum variant:

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

The positional-parameter marker `/` ([Functions](functions.md)) touches no
operand and is unaffected.

A type-owning chain segment may carry type arguments, as in
`Option[int]::Some`; type arguments on a plain scope segment are a static
error. The type-argument form `callee::[T]` and typed-call form
`callee::[T](args)` (e.g. `ask-request::[Review](…)`) instead apply to the
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
the operator characters `-`, `?`, `!`, `<`, `>` may appear *inside*
an identifier, so names like `ask-prompt`, `ask?`, and `do-it-now!` scan as a
single token.  Note that `=` and `@` **are** delimiters, so `a=b` scans as
three tokens and `@std` as two.

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

All string literals are **templates**: they may contain `%{expr}`
interpolation. Both `"` and `'` are valid delimiter characters, giving four
forms:

- `"…"` / `'…'` — single-line.
- `"""…"""` / `'''…'''` — triple-quoted, multi-line, subject to the dedent rule.

Escape sequences, triple-quoted dedent normalization, and interpolation
semantics are covered in [Strings and interpolation](strings-and-interpolation.md).

## Raw-tail forms

`exec!` and `ask!` begin raw-tail calls. The lexer emits a `RAW_TAIL_NAME`,
then `RAW_TAIL_START`, one or more `RAW_FRAGMENT` and interpolation-token
runs, and `RAW_TAIL_END`. Optional type arguments must be byte-adjacent to the
name: `exec!::[T]` and `ask!::[T]`. In `exec! ::[T]` or `ask! ::[T]`, the
spaced `::[T]` instead begins the payload. The payload is either the rest of
that line or a following indented block. In both cases it is one template: its
text is verbatim except that `%{expr}` interpolates and `\%{` is a literal
`%{`. Inline payloads discard trailing spaces and tabs; block payloads drop the
blank lines that trail the last content line.

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

<!-- agl-check: fragment -->
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
| 11 | atoms: literals, names, `( expr )`, `()` unit, templates, `break`, `continue` | — |

The record-update operator `with` binds looser than every level in the table,
on both sides; see
[Record update](expressions.md#record-update).

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
(branch bodies, `if`/`until` conditions) they must be parenthesized. A loop
body is not such a position — the loop terminator closes it — so they may
appear there directly.

All comparison operators are non-associative: `x == y == z`, `1 < 2 < 3`, and
`a <= b != c` are parse errors with a targeted diagnostic.
