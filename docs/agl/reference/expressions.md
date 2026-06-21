# Expressions

[← Index](index.md)

This chapter covers literals, constructors, field access, operators, calls,
and `case`/`if` expressions, together with the static typing rules and runtime
semantics of each. Operator precedence is tabulated in
[Lexical structure](lexical-structure.md).

In AgL v2 **everything is an expression**: there is no separate statement
category. Former "statements" — bindings, `:=`, `print`, `if` without
`else`, loops — are all expressions with well-defined types. A block
(function body, branch body, or the program top level) is a sequence of
items whose value is the value of its last item.

## Literals

```agl
42            # int
1.5           # decimal
true false    # bool
null          # json
()            # unit — the single value of type unit
"text ${x}"   # template (type text; see Strings and interpolation)
```

The unit literal `()` is both the value of type `unit` and the empty
argument list of a zero-argument call — the two are syntactically unified.

### List literals

```agl
let issues = ["missing tests", "unclear API"]
```

Elements must share a type, up to `int → decimal` widening. Under an expected
type, each element is checked against the expected element type. An **empty
list requires an annotation**:

```agl
let items: list[Issue] = []
```

### Dictionary literals

```agl
let metadata: dict[text, json] = {
  "source": "reviewer",
  "attempt": 2,
}
```

Keys are literal strings; an unquoted identifier key is shorthand for the same
string. Interpolated keys are rejected. Duplicate keys are a
static error.

## Constructors

```ebnf
constructor ::= NAME ("." NAME)? type_args? ("(" named_args? ")")?
type_args   ::= "::" "[" type_expr ("," type_expr)* "]"
named_args  ::= NAME ":" expr ("," NAME ":" expr)* ","?
```

A record constructor or enum variant, when invoked **directly** by name, takes
**named** arguments only — never positional. (A constructor escaped through a
variable becomes a positional callable value; see
[Constructors as values](#constructors-as-values).) The optional `::[…]`
pins the type arguments of a generic constructor (see
[Generic constructors](#generic-constructors)).

### Record construction

```agl
Issue(title: "Bug", severity: 2, description: "…")
```

Every declared field must be supplied; unknown and duplicate fields are
static errors.

### Enum variant construction

Qualified or unqualified:

```agl
Review.Pass
Review.Fail(issues: ["missing tests"])

let review: Review = Pass           # resolved by expected type
```

An unqualified variant name resolves when the expected type is an enum
containing it, or when exactly one declared enum has a variant of that name.
A nullary variant is constructed by writing its name alone (no parentheses);
its payload variants take named arguments.

### Unqualified variant ambiguity

If **two or more** declared enums each have a variant of the same unqualified
name, a bare reference to that name is a **static ambiguity error** — even
when the expected type, the payload, or an explicit `::[…]` would in principle
single out one enum. Disambiguate by qualifying with the owning enum:

```agl
enum Holder[T]
  | empty
  | tagged(by: T)

enum Other
  | tagged(name: text)

let h: Holder[int] = Holder.tagged(by: 7)   # qualified; unqualified 'tagged' is an error
```

A nearer binding (a `let`/`var`/parameter of the same name) **shadows** a
constructor or an overloaded set of constructors entirely; within that scope
the name refers to the binding, not the constructor.

### Generic constructors

The constructors of a generic record or enum ([Generics](generics.md)) are
generic too. Their type arguments are normally inferred — from the payload
arguments, the expected type, or both:

```agl
record Box[T]
  value: T

let bi: Box[int] = Box(value: 5)        # T = int, inferred from the payload
let bt: Box[text] = Box(value: "hi")    # same definition, T = text
```

Pin the instantiation explicitly with `::[…]` when inference cannot (or
should not) determine it:

```agl
let be = Box::[int](value: 99)
```

Nullary variants of a generic enum carry no payload to infer from, so they
need an expected type (or an explicit `::[…]`):

```agl
enum Option[T]
  | none
  | some(value: T)

let e: Option[int] = none          # T = int, fixed by the annotation
let s = some::[int](value: 1)      # T pinned explicitly
let q = Option.some::[int](value: 2) # qualification disambiguates the owner
```

### Constructors as values

A record constructor or an enum variant is an **ordinary value binding**: it
can be stored, passed to a function, and called like any other function value.
When a constructor is reached **through a variable** rather than written
directly, it is a positional callable — its arguments are supplied positionally
in **declaration order**, since a function value has no named parameters
([Functions](functions.md)):

```agl
let mk: (int) -> Box[int] = Box     # the constructor as a value
let made = mk(1)                    # called positionally
print made.value
```

This lets constructors be passed to higher-order functions:

```agl
def apply[A, B](x: A, f: (A) -> B) -> B = f(x)

let built = apply(42, mk)           # mk applied inside a generic HOF
print built.value
```

A **generic** constructor used as a value needs an expected type to fix its
instantiation, exactly like a generic `def` used as a value (a bare
`let f = some` is a static error). The annotation on `mk` above supplies it.

Nullary enum variants are likewise ordinary values:

```agl
let n: Option[int] = none           # the nullary variant as a value
```

### Exception construction

Built-in exception types are constructed like records:

```agl
raise Abort(message: "Cannot continue.")
```

## Field access

`expr.field` reads a field of a **record**, **exception**, or `ExecResult`
value:

```agl
let sev = issue.severity
let code = res.exit_code
```

Field access is statically checked. It does not apply to enums (use pattern
matching to extract variant payloads), dictionaries, or lists.

## Indexing

`expr[index]` reads from a list or dictionary:

```agl
let third = xs[2]
let last = xs[-1]
let value = metadata["source"]
```

Indexing is a postfix operator and may be chained with calls and field access:

```agl
let cell = matrix[0][1]
let name = rows[0].name
let item = make_items()[0]
```

Whitespace matters. `xs[0]` is indexing because the `[` is adjacent to `xs`.
`f [0]` remains the single-argument call sugar `f([0])`.

List indexes must be `int`. Negative indexes count from the end, as in
Python: `xs[-1]` selects the last element. An out-of-range list index raises
catchable `IndexError` with `index`, `length`, and `message` fields.

Dictionary indexes must be `text`. Missing keys raise catchable `KeyError`
with `key` and `message` fields.

## Calls

All calls use the same uniform parenthesized syntax. This applies equally to
user `def`s, built-in functions (`ask`, `exec`, `print`), and function values
stored in bindings:

```ebnf
call_expr ::= postfix_expr type_args? "(" arg_list? ")"
type_args ::= "::" "[" type_expr ("," type_expr)* "]"
arg_list  ::= arg ("," arg)* ","?
arg       ::= expr                  (* positional *)
            | NAME ":" expr         (* named *)
```

**Single-argument sugar.** When there is exactly one positional argument and
no named arguments, the parentheses may be dropped:

```agl
print review          # equivalent to print(review)
ask "Hello?"          # equivalent to ask("Hello?")
print res.stdout      # field-access path is valid sugar argument
```

A *call result* as the lone argument still needs parentheses:

```agl
print(classify(x))    # compound arg: parens required
```

Application binds **tighter than all operators**:

```agl
print x + 1           # parsed as (print x) + 1
```

For details on named arguments, defaults, and function types, see
[Functions](functions.md). For `ask`'s named parameters, see
[Agent calls](agent-calls.md).

## `print`

`print` is a built-in function that accepts one argument of any type, writes
its rendered value (followed by a newline) to the host's standard output, and
returns `unit`. The argument may have any type that has a rendering:

```agl
print "Review round failed; retrying."
print review                           # renders review in AgL form
print(classify(-4))                    # compound argument needs parens
print res.stdout                       # field-access chain as sugar arg
```

`print` cannot be bound as a function value (`let f = print` is a static
error, because `print`'s type is not yet fully expressible in v1).

## `parse_json`

`parse_json` is a built-in function that parses a `text` value as a strict
JSON document and returns the resulting `json` value:

```text
parse_json(input: text) -> json
```

It uses **strict JSON parsing**: the input must be exactly one well-formed
JSON value with nothing but surrounding whitespace — no Markdown fences, no
prose, no repair. On success it returns the parsed JSON tree. On failure it
raises a catchable `JsonParseError` ([Exceptions](exceptions.md)).

```agl
let v: json = parse_json('{"key": 42}')   # json dict
let n: json = parse_json("42")             # json number 42
let b: json = parse_json("true")           # json boolean true

# parse_json("42") ≠ "42" as json — the former parses; the latter embeds
let embedded: json = "42" as json     # the JSON string "42" (wraps text)
let parsed: json   = parse_json("42") # the JSON number 42 (interprets text)
```

**Contrast with `text as json`.** Because `text` is already JSON-shaped,
`"42" as json` wraps the text in JSON representation — it yields the JSON
**string** `"42"`. `parse_json("42")` instead interprets the characters of
the text and yields the JSON **number** `42`. Use `parse_json` when you have
a text value that contains serialized JSON and you want to traverse or
validate its structure.

```agl
try
  let data: json = parse_json(raw_output)
  # use data...
catch JsonParseError as e =>
  print "Malformed JSON: ${e.raw}"
```

`parse_json` cannot be bound as a function value (`let f = parse_json` is a
static error, because `parse_json`'s type is not yet fully expressible in v1).

## Operators

### Equality: `=` and `!=`

A single `=` is **equality**, not assignment. Both operands must have the
same type after `int → decimal` widening. Equality is full value equality
([Types](types.md)).

`=` is non-associative; `x = y = z` is a parse error.

### Ordering: `<` `<=` `>` `>=`

Both operands must be numeric or both `text`. Text ordering is lexicographic
by code point.

### Membership: `in`

```agl
issue in issues          # element membership:  issues: list[T]
"source" in metadata     # key membership:      metadata: dict[text, V]
"missing" in body        # substring:           both text
```

### Arithmetic: `+` `-` `*` `/` and unary `-`

1. `+ - *` on two `int` values yield `int`; if either is `decimal`, the
   result is `decimal`.
2. `/` **always yields `decimal`**, even for two `int` operands.
3. Division by zero raises `ArithmeticError` at runtime.
4. `+` on two `text` values is concatenation.
5. Unary `-` negates an `int` or `decimal`.

### Boolean: `and`, `or`, `not`

Operands must be `bool`. `and` and `or` short-circuit.

### Casts: `as` and `as?`

```agl
EXPR as T     # cast: convert EXPR to type T
EXPR as? T    # convertibility test: bool, never raises
```

`as` converts the value to the named type; `as?` tests whether the
conversion would succeed. The full conversion matrix and semantics are in
[Types](types.md#casts-and-convertibility).

**Precedence.** Cast operators sit between unary `-` (tighter) and `* /`
(looser). They are **left-associative**:

| Expression | Parsed as |
| ---------- | --------- |
| `-1 as text` | `(-1) as text` — unary minus binds tighter |
| `2 * 3 as text` | `2 * (3 as text)` — `*` is looser than cast |
| `1 + 2 as text` | `1 + (2 as text)` — `+` is looser than cast |
| `f x as int` | `(f x) as int` — application binds tighter |
| `x as json as text` | `(x as json) as text` — left-associative |
| `a as? int and b` | `(a as? int) and b` — `and` is looser than cast |

**`as?` is a single token**: the `?` is part of the keyword and must be
adjacent to `as` (no whitespace). `as` and `as?` are always reserved
keywords — they cannot be used as variable names.

Examples:

```agl
let n: int = raw_value as int          # raises CastError if not an int
let ok: bool = raw_value as? int       # true when cast would succeed

let s: text = some_int as text         # total — always succeeds
let j: json = my_record.count as json  # total — int is JSON-shaped

# left-associativity chains
let t: text = some_int as json as text   # (some_int as json) as text

# convertibility test without exception handling
if count_json as? int =>
  let n: int = count_json as int
  print n
```

A `text` cast from a fallible source reads the value and formats it as text;
it is always total regardless of the source type (any data value has a text
rendering). For structured source types (`list`, records, enums, exceptions …)
`as text` produces the same AgL-form text that `print` would render.
For scalar types (`bool`, `int`, `decimal`) it produces the plain scalar text.

### Variant tests: `is`, `is not`

```agl
review is Pass
status is Status.Blocked     # qualified; aliases resolve transparently
```

The left operand must have enum type; the variant must belong to that enum.

## `case` expressions

A `case` **expression** selects among single-expression branches:

```ebnf
case_expr ::= "case" expr "of" ("|" pattern "=>" or_expr)+
```

```agl
let next: text = case action of
  | Stop => "Stop."
  | Continue(prompt) => prompt
  | Escalate(reason) => "Investigate blocker:\n${reason}"
```

All branch result types must agree after `int → decimal` widening. An outer
expected type propagates into every branch. If no pattern matches,
`MatchError` is raised.

A `case` expression is the loosest expression form: in positions where a
following `|` could be ambiguous (branch bodies, `if`/`until` conditions) it
must be parenthesized. A bare `case` at block level is always a `case`
**statement form** (same grammar production — the same `case` keyword leads
both).

## `if` expressions

An `if` **expression** selects among branches by boolean condition:

```ebnf
if_expr        ::= "if" "|"? if_expr_branch ("|" if_expr_branch)* if_else_branch
if_expr_branch ::= or_expr "=>" or_expr
if_else_branch ::= "|"? "else" "=>" or_expr
```

```agl
let label: text = if | score > 90 => "A" | score > 75 => "B" | else => "C"
```

The `else` branch is **required** — an `if` without `else` is a
**statement** (type `unit`, below), not an expression. All branch result
types must agree after `int → decimal` widening.

## Expressions that yield `unit`

The following expressions have type `unit` — they exist for their side
effect, not their value:

| Form | Type | Notes |
|------|------|-------|
| `print(e)` | `unit` | writes to stdout |
| `x := e` | `unit` | mutates `x` |
| `if c => body` (no `else`) | `unit` | branch body must be `unit` |
| `do … until c` | `unit` | loops run for effect |
| `()` | `unit` | the unit literal itself |

An `if` without `else` always has type `unit`, and each branch body must also
have type `unit`. A `case` or `do` loop likewise yields `unit`.

`unit` values may appear anywhere in a block, including as the final
expression. A function declared `-> unit` has its body checked against
`unit`.

## `let` and `var` as expressions

`let` and `var` are **binders**: they bind a name and scope it over the
*continuation* — the remaining items in the block. The type and value of a
`let`/`var` binding is the type and value of the continuation, not of the
bound expression:

```agl
let x = 3
let y = x + 1
y                 # the block's value is y, an int
```

A block that ends in a `let` or `var` with no continuation is a **static
error** — a binder must be followed by at least one more expression.

```agl
def broken() -> int =
  let x = 1        # static error: 'let' must be followed by an expression
```

## `raise` as an expression

`raise expr` diverges — it never yields a value. Its type is the **bottom
type** (assignable to any expected type), so `raise` may appear in any
expression position, including as the initializer of a `let`/`var` or as the
body of a branch suite:

```agl
let x: int = raise Abort(message: "Cannot continue.")

case status of
  | Ok => ()
  | Error(reason) =>
      raise Abort(message: reason)
```

## `try` as an expression

`try … catch …` yields a value: the body and every handler must agree on a
common type (with `int → decimal` widening). This type becomes the `try`
expression's type:

```agl
let result: decimal =
  try
    10 / 0
  catch ArithmeticError =>
    -1
```

## Expected-type propagation

The checker propagates an expected type top-down where it helps:

| Context | Propagated expectation |
| ------- | ---------------------- |
| `let x: T = e` / `var x: T = e` | `T` into `e` |
| `x := e` | declared type of `x` into `e` |
| Constructor argument | declared field type |
| `list[T]` / `dict[text, V]` expectation | element/value type into each element |
| `case` / `if` expression with outer expectation | into every branch |
| `ask` / typed `exec` | becomes the call's target type |
| Function call | each parameter type into the corresponding argument |
| Function body | `-> RetType` propagated in |

Propagation resolves unqualified variant constructors (`let r: Review = Pass`),
types empty containers, and gives agent calls their output contracts. Where
no expectation exists, inference is bottom-up, and an untyped `ask` defaults
to `text`.
