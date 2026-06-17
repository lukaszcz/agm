# Functions

[← Index](index.md)

AgL supports **user-defined functions**: named `def` declarations at the
program root and anonymous `fn` expressions. Functions are first-class
values; they may be stored in bindings, passed as arguments, and returned
from other functions. The type of a function value is written
`(A, B) -> C`.

## `def` — named function declarations

```ebnf
func_def   ::= "def" VAR_NAME "(" params? ")" "->" type_expr "=" expr
params     ::= param ("," param)* ","?
param      ::= VAR_NAME ":" type_expr ("=" expr)?
```

A `def` is a top-level declaration. The body is a single expression — which
may be a block (a sequence of items ending in an expression):

```agl
def classify(n: int) -> text =
  if n > 0 => "pos"
  | n < 0  => "neg"
  | else   => "zero"

def summarize(doc: text, limit: int = 3) -> text =
  let head = ask "Summarize: ${doc}"
  let tagged = "[${limit}] ${head}"
  tagged
```

### Return type

The `-> RetType` annotation is **required** on every `def`. The body is
checked against it; a mismatch is a static error. There is no `return`
keyword — the body's value is its last expression.

### Parameters

Parameters are listed with explicit types. A parameter may carry a
**default value** (`param: type = expr`). Defaulted parameters must follow
all required parameters:

```agl
def greet(name: text, greeting: text = "Hello") -> text =
  "${greeting}, ${name}!"
```

### Scope and forward references

`def` declarations are collected before any expressions are evaluated, so
every top-level `def` is in scope for every other (and for itself). Mutual
recursion among top-level `def`s is therefore unrestricted:

```agl
def is_even(n: int) -> bool =
  if n = 0 => true | else => is_odd(n - 1)

def is_odd(n: int) -> bool =
  if n = 0 => false | else => is_even(n - 1)
```

`def` is **not** a valid declaration inside a block (`do` body, `if`
branch, etc.); it is root-only. A static error is raised if a `def` is
nested.

## `fn` — anonymous functions (lambdas)

```ebnf
lambda_expr ::= "fn" "(" params? ")" ("->" type_expr)? "=>" expr
```

`fn` produces a function value. The return type annotation is **optional**:
when omitted it is inferred from the body. Parameter types are always
required.

```agl
let double = fn(x: int) => x * 2
let add    = fn(x: int, y: int) -> int => x + y
let greet  = fn(name: text) -> text => "Hello, ${name}!"
```

A lambda is an ordinary expression and may appear anywhere an expression is
accepted — in a binding, as a call argument, or in a list:

```agl
let ops: list[(int) -> int] = [fn(x: int) => x + 1, fn(x: int) => x * 2]
```

When used in juxtaposition position (as the right operand of an operator or
the lone argument to a single-arg call), a lambda must be parenthesized:

```agl
# Correct: parenthesized lambda in operator position
let result = (fn(x: int) => x + 1)(5)
```

### Lambdas are not self-recursive

A lambda's name (the binding introduced by `let`) is not in scope inside
the lambda body. Local recursion is expressed via a top-level `def`. The
restriction is intentional: lambda return-type inference is bottom-up and
safe precisely because the body never depends on the lambda's own type.

## Calling functions

All calls use the same uniform parenthesized syntax:

```ebnf
call_expr ::= postfix_expr "(" arg_list? ")"
arg_list  ::= arg ("," arg)* ","?
arg       ::= expr                   (* positional *)
            | VAR_NAME ":" expr      (* named *)
```

**Single-argument sugar.** When there is exactly one positional argument
and no named arguments, the parentheses may be dropped and the argument
written directly after the callee:

```agl
print review          # equivalent to print(review)
ask "Hello?"          # equivalent to ask("Hello?")
print res.stdout      # field-access path is a valid sugar argument
```

The sugar applies only to simple atoms and field-access chains — not to
call-result expressions. When the sole argument is itself a call, the
parentheses are required:

```agl
print(classify(x))    # compound arg: parens required
```

Application binds **tighter than all operators**:

```agl
print x + 1           # parsed as (print x) + 1
```

### Required (positional) arguments

Required parameters must be supplied positionally, in declaration order:

```agl
def add(x: int, y: int) -> int = x + y

let r = add(3, 4)     # x=3, y=4
```

### Named and defaulted arguments

Parameters that carry a default may be omitted at a declared-name call site.
When supplied, they are named:

```agl
def format_msg(text: text, prefix: text = "[INFO]") -> text =
  "${prefix} ${text}"

format_msg("Done.")              # prefix uses its default
format_msg("Done.", prefix: "!") # prefix supplied by name
```

Named arguments may be given in any order after all required positional
arguments. Unknown names, duplicates, and supplying a required positional
argument both positionally and by name are static errors.

Named arguments are only available when calling a **declared name** — a
`def` or a built-in. A function *value* (bound in a `let` or passed as an
argument) has a purely positional type and is called with positional
arguments only.

### Calling function values

A function value is called like any other call. The callee is an expression
of function type:

```agl
let g: (int) -> text = classify
let label = g(7)               # positional call of a function value
```

## Function types

The type of a function value is `(A, B, …) -> C` — a list of parameter
types in parentheses, a `->`, and the result type:

```ebnf
func_type ::= "(" type_list? ")" "->" type_expr
type_list ::= type_expr ("," type_expr)* ","?
```

```agl
let f: (int) -> text = classify
let g: (int, int) -> int = add
let h: () -> bool = fn() => true
```

Function types are assignable by **exact structural match**: the number of
parameters, their types (in order), and the result type must all agree. No
variance or subtyping applies in v1.

Named and defaulted arguments are erased from the function *value* type.
A `def` with optional parameters still has a fully positional function type;
only the declared name retains the named/default information at call sites.

## Opacity

Function values have **no rendering, no JSON encoding, and no equality**.
Interpolating a function value in a template (`"${f}"`) is a static error.
Printing a function value (`print f`) is a static error. Storing a function
value in a `json` slot, passing it to `ask`, or using it where a JSON-shaped
type is expected are all static errors. These restrictions exist because
function values are capability handles, not data.

## Recursion and the call-depth limit

Top-level `def`s may call themselves and each other without restriction
at the language level. The runtime enforces a **call-depth limit** (default
256, host-configurable). Exceeding it raises `RecursionError`
([Exceptions](exceptions.md)):

```agl
def fact(n: int) -> int =
  if n <= 1 => 1 | else => n * fact(n - 1)

let r = fact(10)     # fine
let s = fact(10000)  # raises RecursionError at depth limit
```

`RecursionError` is catchable with `try`/`catch`. The limit counts
activation frames across all `def` calls including mutual recursion.

## Prelude types used with functions

The built-in `ParsePolicy` enum and the `ExecResult` record are described
in the chapters that cover `ask` ([Agent calls](agent-calls.md)) and `exec`
([Shell execution](shell-execution.md)), but they are ordinary values that
can be stored in bindings and passed to functions:

```agl
def make_policy(retries: int) -> ParsePolicy =
  if retries = 0 => ParsePolicy.Abort | else => Retry(n: retries)
```

## Complete example

```agl
record Issue
  title: text
  severity: int

enum Review
  | Pass
  | Fail(issues: list[text])

agent reviewer

def summarize_issues(issues: list[text]) -> text =
  "Issues found:\n${issues}"

def review_artifact(artifact: text, max_retries: int = 2) -> Review =
  let policy = if max_retries > 0 => Retry(n: max_retries) | else => Abort
  let r: Review = ask(
    "Review this artifact:\n${artifact}",
    agent: reviewer,
    on_parse_error: policy
  )
  r

input spec: text
let artifact: text = ask "Implement ${spec}"
let result = review_artifact(artifact)

case result of
  | Pass => print "Accepted."
  | Fail(issues) => print(summarize_issues(issues))
```
