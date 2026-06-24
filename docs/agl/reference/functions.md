# Functions

[← Index](index.md)

AgL supports **user-defined functions**: named `def` declarations at the
program root and anonymous `fn` expressions. Functions are first-class
values; they may be stored in bindings, passed as arguments, and returned
from other functions. The type of a function value is written
`(A, B) -> C`.

## `def` — named function declarations

```ebnf
func_def   ::= "def" NAME type_params? "(" params? ")" "->" type_expr "=" expr
             | "builtin" "def" NAME type_params? "(" params? ")" "->" type_expr
type_params ::= "[" NAME ("," NAME)* "]"
params     ::= param ("," param)* ","?
param      ::= NAME ":" type_expr ("=" expr)?
```

A `def` is a top-level declaration. The body is a single expression — which
may be a block (a sequence of items ending in an expression):

```agl
def classify(n: int) -> text =
  if
    | n > 0 => "pos"
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

### Built-in functions

`builtin def` declares a function implemented by the host, so it has no body.
The declared name and signature must match a recognized built-in exactly. This
form is used by `std.core`; ordinary programs normally call those declarations
through the default standard-library import instead of redeclaring them.

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
  if n = 0 => true else => is_odd(n - 1)

def is_odd(n: int) -> bool =
  if n = 0 => false else => is_even(n - 1)
```

`def` is **not** a valid declaration inside a block (`do` body, `if`
branch, etc.); it is root-only. A static error is raised if a `def` is
nested.

## `fn` — anonymous functions (lambdas)

```ebnf
lambda_expr ::= "fn" "(" params? ")" ("->" type_expr)? "=>" expr
params      ::= param ("," param)* ","?
param       ::= NAME ":" type_expr ("=" expr)?
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

## Generic functions

A `def` may declare **type parameters** in square brackets after its name,
making it polymorphic over those types. This is prenex (rank-1) parametric
polymorphism: the type parameters are universally quantified over the whole
declaration. See [Generics](generics.md) for the full treatment; this section
covers the function-specific surface.

```agl
def id[T](x: T) -> T = x

def fst[A, B](a: A, b: B) -> A = a
```

A type parameter is an ordinary name; it may be used anywhere a type may
appear within the declaration — parameter types, the return type, and any
annotation **nested inside the body**:

```agl
def singleton[T](x: T) -> list[T] =
  let single: list[T] = [x]
  single

def via_lambda[A](x: A) -> A =
  let g: (A) -> A = fn(y: A) -> A => y
  g(x)
```

Here `list[T]` is an annotation on an inner `let`, and the lambda's parameter
and return types refer to the enclosing `A`. A type variable is in scope
throughout the body of the `def` that introduces it.

### Inference and explicit type arguments

At a call site the type arguments are normally **inferred** — from the
argument types and from the expected type of the call:

```agl
print(id(5))          # T = int, inferred from the argument
print(id("hi"))       # T = text
print(fst("x", 9))    # A = text, B = int
```

When inference is insufficient or you want to pin the instantiation, supply
the type arguments explicitly with the typed-call form `::[…]`:

```agl
print(id::[int](5))
```

The type-argument list must match the declared type parameters in number and
order.

### A generic `def` as a first-class value

A generic `def` can be used as a value, but only where an **expected type**
fixes its instantiation — typically an annotated binding. The expected
function type determines the type arguments:

```agl
def id[T](x: T) -> T = x

let f: (text) -> text = id      # T = text, fixed by the annotation
print(f("via value"))
```

A bare `let f = id` with no expected type is a **static error**: there is
nothing to infer the type arguments from. (Calling `id` directly, where the
arguments drive inference, needs no annotation.)

### Strict parametricity

Inside a generic `def`, a value whose static type is a bare type variable
`T` is **opaque**. The body knows nothing about `T` beyond the fact that
values of it exist, so such a value can only be passed to other functions,
returned, or stored. It may **not** be:

- compared with `=`, `!=`, or the ordering operators,
- used in arithmetic,
- printed or interpolated in a template,
- field- or index-accessed,
- tested with `is` / `is not`.

```agl
def bad[T](x: T, y: T) -> bool = x = y   # static error: '=' on type variable T
```

Each of these is a static error. This *parametricity* guarantee means a
generic function treats its type-variable values uniformly regardless of the
concrete type they are instantiated at. (The restriction applies only to the
bare type variable itself — a value of a concrete or composite type such as
`list[T]` supports every operation that type normally allows.)

## Calling functions

All calls use the same uniform parenthesized syntax:

```ebnf
call_expr ::= postfix_expr type_args? "(" arg_list? ")"
type_args ::= "::" "[" type_expr ("," type_expr)* "]"
arg_list  ::= arg ("," arg)* ","?
arg       ::= expr                   (* positional *)
            | NAME ":" expr          (* named *)
```

**Single-argument sugar.** When there is exactly one positional argument
and no named arguments, the parentheses may be dropped and the argument
written directly after the callee:

```agl
print review          # equivalent to print(review)
ask "Hello?"          # equivalent to ask("Hello?")
print res.stdout      # field-access path is a valid sugar argument
print classify(x)     # equivalent to print(classify(x))
f Opt.Some(x: 1)      # equivalent to f(Opt.Some(x: 1))
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

Function values have **opaque rendering, no JSON encoding, and no equality**.
Rendering a function value, interpolating it in a template, or printing it
produces a diagnostic surface form such as `<function: (int, int) -> int>`.
Storing a function value in a `json` slot, passing it to `ask`, or using it
where a JSON-shaped type is expected are static errors. These restrictions
exist because function values are capability handles, not data.

The REPL echoes bare function values with the same opaque rendering. That
display is available through AgL rendering, but it is not JSON data.
programs through `print`, interpolation, `as text`, or JSON conversion.

## Recursion and the call-depth limit

Top-level `def`s may call themselves and each other without restriction
at the language level. The runtime enforces a **call-depth limit** (default
256, host-configurable). Exceeding it raises `RecursionError`
([Exceptions](exceptions.md)):

```agl
def fact(n: int) -> int =
  if n <= 1 => 1 else => n * fact(n - 1)

let r = fact(10)     # fine
let s = fact(10000)  # raises RecursionError at depth limit
```

`RecursionError` is catchable with `try`/`catch`. The limit counts
activation frames across all `def` calls including mutual recursion.

## Standard core types used with functions

The built-in `ParsePolicy` enum and the `ExecResult` record are described
in the chapters that cover `ask` ([Agent calls](agent-calls.md)) and `exec`
([Shell execution](shell-execution.md)), but they are ordinary values that
can be stored in bindings and passed to functions:

```agl
def make_policy(retries: int) -> ParsePolicy =
  if retries = 0 => ParsePolicy.Abort else => Retry(n: retries)
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
  let policy = if max_retries > 0 => Retry(n: max_retries) else => Abort
  let r: Review = ask(
    "Review this artifact:\n${artifact}",
    agent: reviewer,
    on_parse_error: policy
  )
  r

param spec: text
let artifact: text = ask "Implement ${spec}"
let result = review_artifact(artifact)

case result of
  | Pass => print "Accepted."
  | Fail(issues) => print(summarize_issues(issues))
```
