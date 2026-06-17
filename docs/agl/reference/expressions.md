# Expressions

[← Index](index.md)

This chapter covers literals, constructors, field access, operators, calls,
and `case`/`if` expressions, together with the static typing rules and runtime
semantics of each. Operator precedence is tabulated in
[Lexical structure](lexical-structure.md).

In AgL v2 **everything is an expression**: there is no separate statement
category. Former "statements" — bindings, `set`, `print`, `if` without
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

Keys are literal strings; an unquoted lowercase identifier key is shorthand
for the same string. Interpolated keys are rejected. Duplicate keys are a
static error.

## Constructors

```ebnf
constructor ::= TYPE_NAME ("." TYPE_NAME)? ("(" named_args? ")")?
named_args  ::= VAR_NAME ":" expr ("," VAR_NAME ":" expr)* ","?
```

Constructor arguments are **named only**. Constructors may only follow a
type name.

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

## Calls

All calls use the same uniform parenthesized syntax. This applies equally to
user `def`s, built-in functions (`ask`, `exec`, `print`), and function values
stored in bindings:

```ebnf
call_expr ::= postfix_expr "(" arg_list? ")"
arg_list  ::= arg ("," arg)* ","?
arg       ::= expr                  (* positional *)
            | VAR_NAME ":" expr     (* named *)
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
print review                           # renders review as pretty JSON
print(classify(-4))                    # compound argument needs parens
print res.stdout                       # field-access chain as sugar arg
```

`print` cannot be bound as a function value (`let f = print` is a static
error, because `print`'s type is not yet fully expressible in v1).

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
| `set x = e` | `unit` | mutates `x` |
| `if c => body` (no `else`) | `unit` | branch may not run |
| `do … until c` | `unit` | loops run for effect |
| `()` | `unit` | the unit literal itself |

An `if` without `else` always has type `unit`, and the branch body's value
is discarded. A `case` or `do` loop likewise yields `unit`.

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
| `set x = e` | declared type of `x` into `e` |
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
