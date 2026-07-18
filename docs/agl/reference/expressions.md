# Expressions

[← Index](index.md)

This chapter covers literals, constructors, field access, operators, calls,
and `case`/`if` expressions, together with the static typing rules and runtime
semantics of each. Operator precedence is tabulated in
[Lexical structure](lexical-structure.md).

In AgL **everything is an expression**: there is no separate statement
category. Bindings, `:=`, `print`, `if` without `else`, and loops are all
expressions with well-defined types. A block (function body, branch body, or
the program top level) is a sequence of items whose value is the value of its
last item.

## Literals

```agl
42            # int
1.5           # decimal
true false    # bool
null          # json
()            # unit — printable unit value
"text ${x}"   # template (type text; see Strings and interpolation)
```

The unit literal `()` is the printable unit value. `void` is the same unit
value with REPL echo suppressed; it compares equal to `()`. `()` is also the
empty argument list of a zero-argument call — the two are syntactically
unified.

### List literals

```agl
let issues = ["missing tests", "unclear API"]
```

Elements must share a type, up to `int → decimal` widening. Under an expected
type, each element is checked against the expected element type. An empty list
may obtain its element type from an expected container type or another
constraint in the same enclosing expression:

```agl
def choose[T](left: T, right: T) -> T = right
let items = choose([], [1])
```

If no such constraint determines the element type before the enclosing
expression ends, add an annotation:

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
string. Interpolated keys are rejected. Duplicate keys are a static error. An
empty dictionary may obtain its value type from an expected dictionary type or
another constraint in the same enclosing expression; otherwise it needs an
annotation:

```agl
let metadata: dict[text, json] = {}
```

## Constructors

```ebnf
constructor ::= constructor_ref value_type_args? constructor_args?
constructor_ref ::= name
                  | qual_prefix name
                  | qual_prefix? type_ref "::" name
type_ref    ::= name ("[" type_expr ("," type_expr)* "]")?
qual_prefix ::= module_path "::" | "::"
constructor_args ::= "(" (ctor_arg ("," ctor_arg)* ","?)? ")"
value_type_args ::= "::" "[" type_expr ("," type_expr)* "]"
ctor_arg    ::= expr              (* positional *)
              | field_name "=" expr
```

Constructor arguments follow the same **positional-greedy** binding as function
calls — positional arguments fill positional-capable (pos-only/standard) field
slots left to right; named arguments (`field = value`) follow. The optional
value-position `::[…]` pins the type arguments of a generic constructor (see
[Generic constructors](#generic-constructors)).

**Per-type field zones.** Record fields, enum payload fields, and an
exception's own fields default to the **standard** zone (positional or named),
regardless of payload arity. Markers (`/`, `*`, `@pos`, `@std`, `@named`) can
constrain fields to a different zone. An exception's inherited `message` field
is named-only.

**Bare-name shorthand.** A bare name `x` in a positional slot that lands on a
**named-only** field (where positional binding is impossible) is reinterpreted as
`x = x`. This shorthand applies in any call context — functions and constructors
alike — whenever a named-only parameter is in play.

### Record construction

```agl
Issue(title = "Bug", severity = 2, description = "...")
```

Every declared field must be supplied; unknown and duplicate fields are
static errors.

### Enum variant construction

Qualified or unqualified:

```agl
Review::Pass
Review::Fail(issues = ["missing tests"])

let review: Review = Pass           # resolved by expected type
```

An unqualified variant name resolves when the expected type is an enum
containing it, or when exactly one declared enum has a variant of that name.
A nullary variant is constructed by writing its name alone (no parentheses).
Payload variants use positional-greedy binding. Every unmarked payload field
is standard (positional or named), regardless of the number of fields.

```agl
enum Result
  | Ok(value: int)
  | Err(reason: text, fatal: bool)

let ok = Ok(42)
let ok2 = Ok(value = 42)
let err = Err("bad", false)
let named_err = Err(reason = "bad", fatal = false)
```

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

let h: Holder[int] = Holder::tagged(by = 7)   # qualified; unqualified 'tagged' is an error
```

A nearer binding (a `let`/`var`/parameter of the same name) **shadows** a
constructor or an overloaded set of constructors entirely; within that scope
the name refers to the binding, not the constructor.

### Generic constructors

The constructors of a generic record or enum ([Generics](generics.md)) are
generic too. Their type arguments are normally inferred — from payload
arguments, the expected type, or other evidence in the surrounding expression:

```agl
record Box[T]
  value: T

let bi: Box[int] = Box(value = 5)        # T = int, inferred from the payload
let bt: Box[text] = Box(value = "hi")    # same definition, T = text
```

Pin the instantiation explicitly with `::[…]` when inference cannot (or
should not) determine it:

```agl
let be = Box::[int](value = 99)
```

This also applies when a constructor value or partial constructor is an
argument to another call: a sibling argument may determine its type arguments.

```agl
record Box[T]
  value: T

def build[T](factory: (T) -> Box[T], value: T) -> Box[T] = factory(value)
let b = build(Box(value = ?), 5)  # the partial is int -> Box[int]
```

Nullary variants of a generic enum carry no payload to infer from, so they
need contextual evidence (or an explicit `::[…]`):

```agl
enum Option[T]
  | none
  | some(value: T)

let e: Option[int] = none          # T = int, fixed by the annotation
let s = some::[int](value = 1)      # T pinned explicitly
let q = Option[int]::some(value = 2) # qualification disambiguates the owner
```

### Constructors as values

A record constructor or an enum variant is an **ordinary value binding**: it
can be stored, passed to a function, and called like any other function value.
When a constructor is reached **through a variable** rather than written
directly, it is a positional callable — its arguments are supplied positionally
in **declaration order**, since a function value has no named parameters
([Functions](functions.md)):

```agl
let mk: int -> Box[int] = Box     # the constructor as a value
let made = mk(1)                    # called positionally
print made.value
```

This lets constructors be passed to higher-order functions:

```agl
def apply[A, B](x: A, f: A -> B) -> B = f(x)

let built = apply(42, mk)           # mk applied inside a generic HOF
print built.value
```

A **generic** constructor used as a value needs expression-local evidence to
fix its instantiation, exactly like a generic `def` used as a value. An
annotation supplies that evidence, and a surrounding higher-order call may
supply it through another argument or its result. A bare `let f = some` is a
static error because the binding has no such evidence.

Nullary enum variants are likewise ordinary values:

```agl
let n: Option[int] = none           # the nullary variant as a value
```

### Exception construction

Built-in exception types are constructed like records:

```agl
raise Abort(message = "Cannot continue.")
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
user `def`s, built-in functions (`ask`, `exec`, `print`, `render`), and
function values stored in bindings:

```ebnf
call_expr ::= postfix_expr type_args? "(" arg_list? ")"
type_args ::= "::" "[" type_expr ("," type_expr)* "]"
arg_list        ::= arg ("," arg)* ","?
arg             ::= expr                         (* positional *)
                  | placeholder_arg              (* positional hole *)
                  | NAME "=" expr                (* named *)
                  | NAME "=" placeholder_arg     (* named hole *)
placeholder_arg ::= "?" | "?<digits>"
```

**Single-argument sugar.** When there is exactly one positional argument and
no named arguments, the parentheses may be dropped:

```agl
print review          # equivalent to print(review)
ask "Hello?"          # equivalent to ask("Hello?")
print res.stdout      # field-access path is valid sugar argument
print classify(x)     # equivalent to print(classify(x))
f Opt::Some(x = 1)      # equivalent to f(Opt::Some(x = 1))
```

Application binds **tighter than all operators**:

```agl
print x + 1           # parsed as (print x) + 1
```

With user-defined symbolic infix operators, `OP_NAME` after an expression is an
operator position. To pass an operator-name value as a call argument, use
parentheses: `print(>>)`.

For details on named arguments, defaults, function types, and partial
application with placeholder arguments, see [Functions](functions.md). For
`ask`'s named parameters, see [Agent calls](agent-calls.md).

## `print`

`print` is a built-in function that accepts one argument of any type, writes
its rendered value (followed by a newline) to the host's standard output, and
returns `void`. It renders with `pretty = false` and `quote_strings = false`:

```agl
print "Review round failed; retrying."
print review                           # renders review in AgL form
print(classify(-4))                    # compound argument needs parens
print res.stdout                       # field-access chain as sugar arg
```

`print` cannot be bound as a function value (`let f = print` is a static
error, because built-ins are only valid in call position).

## `render`

`render` is a built-in function that converts any value to `text` using the
same renderer as interpolation, `print`, casts to `text`, and REPL echo.

```agl
render(value: T, pretty: bool = true, quote_strings: bool = true) -> text
```

`pretty` selects single-line versus multi-line indented rendering for
structured values and JSON. `quote_strings` controls only a top-level `text`
argument; when it is `false`, rendering text is identity.

```agl
render("hi")                         # "\"hi\""
render("hi", quote_strings = false)  # "hi"
render([1, 2])                       # "[\n  1,\n  2\n]"
render([1, 2], pretty = false)       # "[1, 2]"
```

`render` cannot be bound as a function value (`let f = render` is a static
error, because built-ins are only valid in call position).

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
static error, because `parse_json`'s type is not fully expressible).

## Operators

### Equality: `==` and `!=`

`==` is **equality** (a single `=` is never a comparison — it is a
binder/named-argument separator). Both operands must have the same type after
`int → decimal` widening. Equality is full value equality
([Types](types.md)).

Operands whose type is, or transitively contains, a function, agent, or
`unit` value are a static error — this applies to bare values as well as to
containers (`list`, `dict`), records, enums, or exceptions that hold such
a type at any depth.

`==` is non-associative; `x == y == z` is a parse error.

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

**Recursive cast targets.** A [recursive record or enum](types.md#recursive-types)
works as a `text`/`json` cast target exactly like any other: the value is
validated and decoded through the same JSON Schema (`$defs`/`$ref` for the
recursive parts) used at the agent-call boundary, and `as`/`as?` behave
normally, including inside a container target such as `list[Tree]`. The same
finite-schema restriction applies as for an agent output type: a
[polymorphically recursive](generics.md#recursive-generic-types) generic type
whose reachable instantiations never close cannot be used as a cast target
either — a static error at the `as`/`as?` expression, not a runtime failure.

### Variant tests: `is`, `is not`

```agl
review is Pass
status is Status::Blocked     # qualified; aliases resolve transparently
```

The left operand must have enum type; the variant must belong to that enum.

## `case` expressions

A `case` expression selects among pattern branches whose bodies may be a
single expression or an indented suite:

```ebnf
case_expr   ::= "case" or_expr "of" "|"? case_branch ("|" case_branch)*
case_branch ::= pattern "=>" (suite | or_expr)
```

```agl
let next: text = case action of
  Stop => "Stop."
  | Continue(prompt) => prompt
  | Escalate(reason) => "Investigate blocker:\n${reason}"
```

All branch result types must agree after `int → decimal` widening. An outer
expected type propagates into every branch. The patterns must be exhaustive
and non-redundant, as described in [Pattern matching](pattern-matching.md).

A `case` expression is one of the loosest expression forms: in positions where
a following `|` could be ambiguous (branch bodies, `if`/`until` conditions) it
must be parenthesized. The same expression form is valid at block level; AgL
has no separate `case` statement.

## `if` expressions

An `if` **expression** selects among branches by boolean condition:

```ebnf
if_expr        ::= "if" "|"? if_expr_branch ("|" if_expr_branch)* if_else_branch?
if_expr_branch ::= or_expr "=>" (suite | or_expr)
if_else_branch ::= "|"? "else" "=>" (suite | or_expr)
```

```agl
let label: text = if | score > 90 => "A" | score > 75 => "B" | else => "C"
```

The `else` branch is optional. With `else`, all branch result types must agree
after `int → decimal` widening. Without `else`, the `if` remains an expression
but has type `unit`, as described below.

## Expressions with type `unit`

The following expressions have type `unit` — they exist for their side
effect, not their value:

| Form | Type | Notes |
|------|------|-------|
| `print(e)` | `unit` | writes to stdout and returns `void` |
| `x := e` | `unit` | mutates `x` and returns `void` |
| `if c => body` (no `else`) | `unit` | branch body must be `unit`; returns `void` |
| loop expressions | `unit` | loops run for effect and return `void` |
| `return` | bottom expression returning `()` | valid only in a `unit` function |
| `()` | `unit` | the printable unit literal |

An `if` without `else` always has type `unit`, and each branch body must also
have type `unit`. A loop likewise has type `unit` and returns `void`; a `case`
has the common type of its branch bodies.

`unit` values may appear anywhere in a block, including as the final
expression. A function declared `-> unit` has its body checked against
`unit`. In the REPL, a final `void` result is not echoed; a final explicit
`()` is echoed as `()`. `print(void)` still prints `void`.

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

## Divergent expressions

`raise expr` and `return expr` diverge — they never yield a value at their
expression site. Their type is the **bottom type** (assignable to any expected
type), so they may appear in expression positions such as a branch suite or the
initializer of a `let`/`var`:

```agl
let x: int = raise Abort(message = "Cannot continue.")
let y: int = if done => (return 0) | else => 1

case status of
  | Ok => ()
  | Error(reason) =>
      raise Abort(message = reason)
```

`raise` is described in [Exceptions](exceptions.md); `return` is described in
[Functions](functions.md).

## `break` and `continue` as expressions

`break` and `continue` are loop-control expressions. Both have the **bottom
type** (assignable to any expected type), so they may appear in any expression
position inside a loop:

```agl
let x: int = if done => break else => count
```

`break` exits the innermost enclosing loop immediately; the loop produces
`unit`. `continue` skips the remainder of the current iteration's body —
including the `until` condition — and restarts the loop body from the top.

Both are checked to appear inside a loop body or `until` condition. Using
`break` or `continue` outside a loop — including inside a `fn` or `lambda`
that is defined inside a loop — is a static error.

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
types empty containers, and gives agent calls their output contracts. A target
that depends on sibling constraints is resolved with the enclosing expression
before its codec and schema are chosen. Where no expectation exists, inference
is bottom-up, and an untyped `ask` defaults to `text`.
