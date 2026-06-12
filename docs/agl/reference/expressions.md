# Expressions

[← Index](index.md)

This chapter covers literals, constructors, field access, operators, and
`case` expressions, together with the static typing rules and runtime
semantics of each. Operator precedence is tabulated in
[Lexical structure](lexical-structure.md).

## Literals

```agl
42            # int
1.5           # decimal
true false    # bool
null          # json
"text ${x}"   # template (type text; see Strings and interpolation)
```

### List literals

```agl
let issues = ["missing tests", "unclear API"]
```

Elements must share a type, up to `int → decimal` widening: `[1, 2.5]` has
type `list[decimal]`; `[1, "a"]` is a static error
(*"List literal elements have inconsistent types"*). Under an expected type
(`let xs: list[Issue] = […]`), each element is checked against the expected
element type instead. An **empty list requires an annotation**:

```agl
let items: list[Issue] = []
```

Trailing commas are permitted, and a bracketed literal may span lines
(implicit continuation).

### Dictionary literals

```agl
let metadata: dict[text, json] = {
  "source": "reviewer",
  "attempt": 2,
}
```

Keys are literal strings; an unquoted lowercase identifier key is shorthand
for the same string (`source:` ≡ `"source":`). Interpolated keys are
rejected (*"dict keys must be literal strings (no interpolation)."*).
Duplicate keys are a static error. Value typing mirrors list literals:
values unify with `int → decimal` widening, an expected type propagates into
each value, and an empty literal requires an annotation. As the example
shows, values of *mixed* JSON-shaped types (here `text` and `int`) do not
unify on their own — a `dict[text, json]` annotation provides the common
value type. The result type is `dict[text, V]`.

## Constructors

```ebnf
constructor ::= TYPE_NAME ("." TYPE_NAME)? ("(" named_args? ")")?
named_args  ::= VAR_NAME ":" expr ("," VAR_NAME ":" expr)* ","?
```

Constructor arguments are **named only** — there is no positional
construction. A constructor takes a single argument list
(`Issue(a: 1)(b: 2)` is a syntax error), and constructor syntax may only
follow a type name (there is no method-call syntax).

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

let review: Review = Pass                       # resolved by expected type
let review: Review = Fail(issues: ["…"])
```

An unqualified variant name resolves when the expected type is an enum
containing it, or when exactly one declared enum has a variant of that name.
Otherwise:

- If several enums declare the variant and no expected type disambiguates,
  it is a static error: *"Constructor 'Pass' is ambiguous … Use a qualified
  name (e.g. EnumName.Pass)."*
- If no type declares it, *"Unknown constructor 'Pass'."*

Qualifiers resolve alias-transparently: with `type Status = Review`,
`Status.Pass` constructs `Review.Pass` ([Types](types.md)).

### Exception construction

Built-in exception types are constructed like records and are typically used
with `raise`:

```agl
raise Abort(message: "Cannot continue.")
```

The abstract base `Exception` is **not constructible**. The `trace_id` field
of every exception is supplied by the runtime and is not required (nor
expected) in source. See [Exceptions](exceptions.md).

## Field access

`expr.field` reads a field of a **record** or **exception** value:

```agl
let sev = issue.severity
catch AgentParseError as e =>
  print e.raw
```

Field access is statically checked: the object must have record or exception
type and must declare the field. Field access does **not** apply to enums
(use [pattern matching](pattern-matching.md) to extract variant payloads),
nor to dictionaries or lists (no indexing in v1).

## Operators

### Equality: `=` and `!=`

A single `=` is **equality**, not assignment (`==` is rejected: *"Use `=`
for equality."*). Both operands must have the same type after `int → decimal`
widening; otherwise the comparison is a static error. Equality is full value
equality ([Types](types.md)).

`=` is non-associative; `x = y = z` is a parse error. When an initializer is
itself an equality, parenthesizing is recommended for readability:

```agl
let same = (x = y)
```

### Ordering: `<` `<=` `>` `>=`

Both operands must be numeric (`int`/`decimal`, mixed allowed) or both
`text`. Text ordering is lexicographic by code point. Anything else —
including booleans — is a static error.

### Membership: `in`

`in` is statically overloaded by the right operand's type; the result is
always `bool`:

```agl
issue in issues          # element membership:  issues: list[T], issue assignable to T
"source" in metadata     # key membership:      metadata: dict[text, V], left: text
"missing tests" in body  # substring:           both text
```

Any other right-operand type is a static error. List membership uses the
same value equality as `=`, including widening: `1 in [1.0]` is true.

### Arithmetic: `+` `-` `*` `/` and unary `-`

1. `+ - *` on two `int` values yield `int`; if either operand is `decimal`,
   the result is `decimal`.
2. `/` **always yields `decimal`**, even for two `int` operands. Division is
   exact up to 28 significant digits with banker's rounding
   ([Types](types.md)).
3. Division by zero raises `ArithmeticError` at runtime.
4. `+` on two `text` values is concatenation. Mixing `text` with a number is
   a static error — use interpolation instead.
5. Unary `-` negates an `int` or `decimal`.

There is no `%`, no integer division, no `len`, and no indexing.

### Boolean: `and`, `or`, `not`

Operands must be `bool` (a static error otherwise — there is no truthiness).
`and` and `or` **short-circuit**: the right operand is not evaluated when the
left already decides the result.

### Variant tests: `is`, `is not`

`expr is Variant` tests which variant an enum value holds, without
destructuring; `is not` negates it:

```agl
review is Pass
review is not Fail
status is Status.Blocked     # qualified; aliases resolve transparently
```

The left operand must have enum type, the named variant must belong to that
enum, and a qualifier (if present) must resolve to the operand's enum — each
checked statically. Style: use `is` for variant tests and `=` for
scalar/full-value equality (for a nullary variant, `review = Pass` is also
legal value equality).

## `case` expressions

A `case` **expression** selects among single-expression branches:

```ebnf
case_expr ::= "case" expr "of" ("|" pattern "=>" bar_safe_expr)+
```

```agl
let next_prompt: text = case action of
  | Stop => "Stop."
  | Continue(prompt) => prompt
  | Escalate(reason) => "Investigate blocker:\n${reason}"
```

Semantics:

1. The scrutinee is evaluated once.
2. Patterns are tried in order ([Pattern matching](pattern-matching.md));
   the first match's branch expression is evaluated in a fresh branch scope
   with the pattern's bindings, and its value is the result.
3. If no pattern matches, `MatchError` is raised.

Branch bodies are expressions only — no statements. All branch result types
must agree after `int → decimal` widening (a mix of `int` and `decimal`
branches yields `decimal`); any other mix is a static error. An outer
expected type propagates into every branch.

A `case` expression is the loosest expression form: in bar-safe positions
(branch bodies, `if`/`until` conditions, and as the trailing expression of an
inline statement) it must be parenthesized
([Program structure](program-structure.md)). A bare `case` at statement level
is always a `case` *statement* ([Control flow](control-flow.md)).

## Expected-type propagation

The checker propagates an expected type top-down where it helps:

| Context | Propagated expectation |
| ------- | ---------------------- |
| `let x: T = e` / `var x: T = e` | `T` into `e` |
| `set x = e` | declared type of `x` into `e` |
| Constructor argument | declared field type |
| `list[T]` / `dict[text, V]` expectation | element/value type into each element |
| `json` expectation over a container literal | `json` into elements |
| `case` expression with outer expectation | into every branch |
| Agent call / `exec` | becomes the call's target type ([Agent calls](agent-calls.md)) |

Propagation is what resolves unqualified variant constructors
(`let r: Review = Pass`), types empty containers, and gives agent calls
their output contracts. Where no expectation exists, inference is bottom-up,
and an untyped agent call defaults to `text`.

Inside template interpolation `${…}`, a non-empty container *literal* is
checked as a JSON-shaped structure (so mixed-kind literals like
`${ {kind: "demo", n: 1} }` are legal), but this fabricated `json`
expectation applies to literal structure only — a non-literal child
expression (a variable, call, or `case`) is checked on its own and must
merely be JSON-renderable; an agent call inside an interpolation still
defaults to `text`.
