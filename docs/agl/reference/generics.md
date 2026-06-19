# Generics

[← Index](index.md)

AgL supports **prenex (rank-1) parametric polymorphism**: declarations may
abstract over types using type parameters, and those parameters are
instantiated at concrete types where the declaration is used. Type parameters
appear only at the top of a declaration — there are no nested or
higher-ranked type quantifiers.

This page collects the whole generics story: declaring type parameters, type
application, inference and the explicit `::[…]` override, generic constructors
and constructor values, what may be done with a value of a type parameter,
invariance, and the rules around names. Identifier capitalization is
irrelevant throughout (`Box`/`box`, `Option`/`option`, `some`/`Some` are all
equally valid); see [Lexical structure](lexical-structure.md).

## Declaring type parameters

`def`, `record`, `enum`, and `type` aliases may declare type parameters in a
bracketed list immediately after the declared name. Each parameter is an
ordinary `NAME` that is in scope **as a type** throughout the declaration's
body.

```agl
def id[T](x: T) -> T = x

record Box[T]
  value: T

enum Option[T]
  | none
  | some(value: T)

type Pair[A, B] = dict[text, json]
```

A declaration may have several parameters (`def apply[A, B](…)`,
`enum Outcome[T, E]`). Inside the body, a type parameter may be used anywhere
a type is expected — including as a field type, a function parameter or result
type, the element type of `list[T]`, or in a `let` annotation:

```agl
def singleton[T](x: T) -> list[T] =
  let single: list[T] = [x]
  single
```

## Type application

A generic declaration is **used** by applying it to type arguments. The
applied-type syntax is `Name[arg, …]`:

```agl
record Box[T]
  value: T
enum Outcome[T, E]
  | ok(value: T)
  | err(error: E)

let bi: Box[int] = Box(value: 1)
let bt: Box[text] = Box(value: "hi")
let nested: Box[Box[int]] = Box(value: Box(value: 7))
print nested.value.value
```

`Box[int]`, `Option[text]`, `Outcome[int, text]`, and the nested
`Box[Box[int]]` are all applied types. The built-in `list[T]` and
`dict[text, V]` use exactly the same form.

## Inference and the explicit `::[…]` override

Type arguments are normally **inferred** from the argument types and the
expected (contextual) type, so generic code reads like ordinary code:

```agl
def id[T](x: T) -> T = x
record Box[T]
  value: T

print(id(5))                       # T inferred = int
print(id("hi"))                    # T inferred = text
let bi: Box[int] = Box(value: 5)   # T inferred from the payload
```

When inference cannot determine the arguments, or to pin them explicitly, pass
type arguments at the call site with the `::[…]` typed-call form, listing one
argument per type parameter:

```agl
def apply[A, B](x: A, f: (A) -> B) -> B = f(x)
record Box[T]
  value: T
enum Option[T]
  | none
  | some(value: T)

print(id::[int](9))
let be = Box::[int](value: 99)
let s = some::[int](value: 8)
let _r = apply::[int, int](10, fn(n: int) -> int => n + 1)
print be.value
```

A nullary variant can be inferred purely from the expected type:

```agl
enum Option[T]
  | none
  | some(value: T)
let e: Option[int] = none          # T inferred from the annotation
```

## Constructors as values; generic constructor values

Record constructors and enum variants are **ordinary value bindings** (see
[Bindings and scope](bindings-and-scope.md)). Direct construction uses
**named** arguments; a constructor reached through a variable is a normal
function value, called **positionally** in declaration field order:

```agl
record Box[T]
  value: T
let direct: Box[int] = Box(value: 1)   # named, at the construction site
let mk: (int) -> Box[int] = Box        # the constructor as a value
let one = mk(1)                         # called positionally
print one.value
```

A **generic** constructor or generic `def` used as a first-class value has
nothing to infer from, so it needs an **expected-type annotation** that pins
the instantiation:

```agl
def id[T](x: T) -> T = x
record Box[T]
  value: T

let f: (int) -> int = id               # annotation instantiates T = int
let mk: (int) -> Box[int] = Box        # annotation instantiates T = int
print(f(7))
let made = mk(2)
print made.value
```

Such a value behaves like any monomorphic function value afterwards — it can
be passed to a higher-order function and called there:

```agl
def apply[A, B](x: A, f: (A) -> B) -> B = f(x)
record Box[T]
  value: T
let mk: (int) -> Box[int] = Box
let made = apply(42, mk)
print made.value
```

## Strict parametricity

A value whose static type is a **bare type parameter** `T` is **opaque**: the
generic body knows nothing about it beyond that it exists. You may only
**pass it, return it, and store it**. Every operation that would inspect its
contents is a static error. You cannot:

- compare it with `=`, `!=`, `<`, … (`x = x` on a `T` is rejected),
- do arithmetic on it,
- `print` it or interpolate it in a template (a `T` has no rendering),
- access a field (`x.foo`) or index (`x[0]`) of it,
- test it with `is` / `is not`.

```agl
def bad[T](x: T) -> bool = x = x     # static error: '=' not permitted on 'T'
```

This guarantees a generic definition behaves uniformly at every instantiation:
the body cannot branch on the actual type. (Once a type parameter is *applied*
inside a known constructor — e.g. a `Box[T]` value — the surrounding structure
is fully usable; only the bare `T` payload is opaque.)

## Invariance

Type arguments are **invariant**: an applied type matches another only when
its type arguments match **exactly**. There is no variance or subtyping
through a type argument, and the `int → decimal` widening does not propagate
inside one:

```agl
let xs: list[int] = [1, 2]
# let ys: list[decimal] = xs   # static error: list[int] ≠ list[decimal]
print xs[0]
```

`list[int]` is not assignable to `list[decimal]` or `list[json]`, and
`Box[int]` is a different type from `Box[text]`.

## Unqualified variant ambiguity

If two enums declare the same unqualified variant name, an unqualified
reference to that name is a **static ambiguity error** — regardless of
payload, surrounding context, or explicit type arguments. Disambiguate by
**qualifying** the reference (and patterns) with the owning enum:

```agl
enum Option[T]
  | none
  | some(value: T)

def render(o: Option[int]) -> text =
  case o of
    | Option.none => "missing"
    | Option.some(value) => "found ${value}"

let d: Option[int] = Option.some(value: 11)
let line = render(d)
print line
```

Qualification works in expression, pattern, and `is`-test positions
(`Option.some(value: 1)`, `case … | Option.none => …`,
`probe is Option.some`). A nearer ordinary binding (a `let`, `var`, or
function parameter) **shadows** a constructor or overload set, exactly like
any other shadowing (see [Bindings and scope](bindings-and-scope.md)).

## No generic agent targets

An `ask` call's response type becomes a wire-serialized output contract, and a
bare type parameter has no wire representation. Therefore a generic type
parameter (or a type containing one in an unresolved position) may not be used
as an `ask` response type. Instantiate the generic at a concrete type before
the value crosses the agent boundary. See [Agent calls](agent-calls.md) for
how response types drive output contracts.

## No runtime cost (erasure)

Generics add **no run-time type information**. Type parameters are a
compile-time-only abstraction: one generic body, and one generic constructor,
serves every instantiation. `Box[int]` and `Box[text]` are produced by the
same constructor at run time, and a generic `def` runs the identical code
whatever its type arguments. This is a property of the language, observable as
the absence of any per-instantiation runtime behavior — there is no
reflection over a value's type arguments at run time.

## See also

- [Types](types.md) — applied types, invariance, the `type_expr` grammar.
- [Bindings and scope](bindings-and-scope.md) — constructors as value
  bindings, overload sets, shadowing, namespaces.
- [Functions](functions.md) — function values and higher-order functions.
- [Pattern matching](pattern-matching.md) — qualified variant patterns.
