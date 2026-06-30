# Functions

[← Index](index.md)

AgL supports **user-defined functions**: named `def` declarations at the
program root and anonymous `fn` expressions. Functions are first-class
values; they may be stored in bindings, passed as arguments, and returned
from other functions. The type of a function value is written
`(A, B) -> C`.

## `def` — named function declarations

```ebnf
func_def   ::= "def" NAME type_params? "(" param_list? ")" "->" type_expr "=" expr
             | "builtin" "def" NAME type_params? "(" param_list? ")" "->" type_expr
type_params  ::= "[" NAME ("," NAME)* "]"
param_list   ::= param_entry ("," param_entry)* ","?
param_entry  ::= param | param_marker
param        ::= NAME ":" type_expr ("=" expr)?
param_marker ::= "/" | "*" | "@" NAME    (* @pos, @std, @named *)
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

`builtin` is a declaration modifier: it may precede `def` on the same line or
on the line directly above it (the newline after the modifier is
insignificant).

### Parameters

Parameters are listed with explicit types. Each parameter belongs to one of
three **zones** that determine how arguments at the call site are matched:

| Zone | Binding | Notation in the list |
|------|---------|---------------------|
| **Positional-only** | Positional argument only; cannot be passed by name | Parameters before a `/` or `@std` marker, or after an `@pos` marker |
| **Standard** | Positional or named | Parameters after a `/` or `@std` marker, or before `*`/`@named` |
| **Named-only** | Named argument only (or bare-name shorthand) | Parameters after a `*` or `@named` marker |

For `def`/`builtin def`/lambda, the **default zone is standard**: a
parameter list with no markers has all parameters in the standard zone
(positional or named). Markers switch zones at the boundary they appear at:

```agl
def f(x: int, /, y: int) -> int = x + y          # x pos-only, y standard
def g(x: int, /, y: int, *, z: int) -> int = ...  # x pos-only, y std, z named-only
def h(x: int, @std, y: int, @named, z: int) -> int = ...  # same as g
def simple(x: int, y: int) -> int = x + y         # both standard (default)
```

`/` and `@std` are interchangeable (both mean "end of positional-only zone");
`*` and `@named` are interchangeable (both mean "end of standard zone"). A list
may use `/`/`*` and `@`-markers freely mixed. `@pos` opens the positional-only
zone and has no punctuation equivalent; it must come first.

At most one `/`/`@std` and one `*`/`@named` may appear, in zone order. `@pos`
must be the first entry. Violations are static errors.

**Defaults.** A parameter may carry a default value (`param: type = expr`).
Only positional-fillable (pos-only or standard) parameters are subject to the
ordering constraint: no *required* pos-only/standard parameter may follow a
*defaulted* pos-only/standard one. Named-only defaults may appear in any order:

```agl
def greet(name: text, greeting: text = "Hello") -> text =
  "${greeting}, ${name}!"

def with_named_default(x: int, *, tag: text = "ok") -> text =
  "${tag}: ${x}"   # tag is named-only; its default is unconstrained
```

### Scope and forward references

`def` declarations are collected before any expressions are evaluated, so
every top-level `def` is in scope for every other (and for itself). Mutual
recursion among top-level `def`s is therefore unrestricted:

```agl
def is_even(n: int) -> bool =
  if n == 0 => true else => is_odd(n - 1)

def is_odd(n: int) -> bool =
  if n == 0 => false else => is_even(n - 1)
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
def bad[T](x: T, y: T) -> bool = x == y   # static error: '==' on type variable T
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
            | NAME "=" expr          (* named *)
```

**Single-argument sugar.** When there is exactly one positional argument
and no named arguments, the parentheses may be dropped and the argument
written directly after the callee:

```agl
print review          # equivalent to print(review)
ask "Hello?"          # equivalent to ask("Hello?")
print res.stdout      # field-access path is a valid sugar argument
print classify(x)     # equivalent to print(classify(x))
f Opt.Some(x = 1)      # equivalent to f(Opt.Some(x = 1))
```

Application binds **tighter than all operators**:

```agl
print x + 1           # parsed as (print x) + 1
```

### Calling functions

Arguments are matched **positional-greedy**: positional arguments fill
positional-capable (pos-only and standard) slots left to right, in
declaration order. Named arguments use `name = value` and may follow the
positional arguments in any order. Positional arguments must precede named
arguments at the call site.

```agl
def add(x: int, y: int) -> int = x + y
let r = add(3, 4)          # x=3, y=4 (both standard — positional or named)
let s = add(3, y = 4)      # x=3 positional, y=4 named

def f(x: int, /, y: int) -> int = x + y
let a = f(1, 2)            # x=1 positional-only, y=2 positional
let b = f(1, y = 2)        # x=1 positional-only, y=2 named (standard)
# f(x = 1, y = 2) is an error — x is positional-only

def g(x: int, *, z: int) -> int = x + z
let c = g(5, z = 3)        # x=5 (standard), z=3 named-only
# g(5, 3) is an error — z is named-only, positional not permitted
```

**Named-only shorthand.** When a bare variable name `x` appears in a
positional argument slot but all positional-capable parameters are already
filled, it is reinterpreted as the named argument `x = x` — but only if
`x` is a bare name (not an expression). A non-bare expression in that
position is an error:

```agl
def h(a: int, *, key: text) -> text = "${a}: ${key}"
let key = "hello"
print(h(1, key))           # key is bare name, lands on named-only 'key' → key = key
print(h(1, key = key))     # explicit form, identical result
```

**Defaults.** Defaulted parameters may be omitted. Named-only defaults may be
supplied in any order:

```agl
def format_msg(text: text, prefix: text = "[INFO]") -> text =
  "${prefix} ${text}"

format_msg("Done.")              # prefix uses its default
format_msg("Done.", prefix = "!") # prefix supplied by name
```

Unknown names, duplicates, and supplying a positional-only parameter by name
are static errors.

**Named arguments at declared-name sites only.** Named arguments are
available when calling a **declared name** (`def` or built-in). A function
*value* (bound in a `let` or passed as an argument) has a purely positional
type and is called with positional arguments only.

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
variance or subtyping applies.

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
256, configurable via the `max_call_depth` config pragma). Exceeding it raises
`RecursionError` ([Exceptions](exceptions.md)):

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
  if retries == 0 => ParsePolicy.Abort else => Retry(n = retries)
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
  let policy = if max_retries > 0 => Retry(n = max_retries) else => Abort
  let r: Review = ask(
    "Review this artifact:\n${artifact}",
    agent = reviewer,
    on_parse_error = policy
  )
  r

param spec: text
let artifact: text = ask "Implement ${spec}"
let result = review_artifact(artifact)

case result of
  | Pass => print "Accepted."
  | Fail(issues) => print(summarize_issues(issues))
```
