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
applied-type syntax is `Name[arg, …]`; imported declarations may use an open
name or a module-qualified name such as `lib::Box[int]`:

```agl
record Box[T]
  value: T
enum Outcome[T, E]
  | ok(value: T)
  | err(error: E)

let bi: Box[int] = Box(value = 1)
let bt: Box[text] = Box(value = "hi")
let nested: Box[Box[int]] = Box(value = Box(value = 7))
print nested.value.value
```

`Box[int]`, `Option[text]`, `Outcome[int, text]`, and the nested
`Box[Box[int]]` are all applied types. The built-in `list[T]` and
`dict[text, V]` use exactly the same form.

```agl
import containers

def unwrap(box: Box[int]) -> int = box.value
let open_box: Box[int] = Box(value = 1)
let qualified_box: containers::Box[int] = containers::Box(value = 2)
```

## Inference and the explicit `::[…]` override

Type arguments are normally **inferred** from the argument types and the
expected (contextual) type, so generic code reads like ordinary code:

```agl
def id[T](x: T) -> T = x
record Box[T]
  value: T

print(id(5))                       # T inferred = int
print(id("hi"))                    # T inferred = text
let bi: Box[int] = Box(value = 5)   # T inferred from the payload
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
let be = Box::[int](value = 99)
let s = some::[int](value = 8)
let qs = Option.some::[int](value = 13)
let _r = apply::[int, int](10, fn(n: int) -> int => n + 1)
print be.value
```

The same `::[…]` suffix can instantiate a generic function as a value without
calling it:

```agl
let int_id = id::[int]
print(int_id(9))
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
positional-greedy binding — positional arguments fill positional-capable slots
left to right, then named arguments follow. A constructor reached through a
variable is a normal function value, called **positionally** in declaration
field order:

```agl
record Box[T]
  value: T
let direct: Box[int] = Box(value = 1)   # named, at the construction site
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

### Pinning a generic constructor value with `::[…]`

Instead of relying on an expected-type annotation, you can instantiate a
bare generic constructor value explicitly with the same `::[…]` suffix used
for generic functions. A payload variant becomes a function value; a nullary
variant constructs its value directly, with no parentheses:

```agl
enum Option[T]
  | none
  | some(value: T)

let mk: (int) -> Option[int] = some::[int]   # ≡ fn (x: int) => some(x)
let v = mk(7)
let z: Option[int] = none::[int]            # nullary value, no call needed
```

The qualified forms `Option.some::[int]` and `Option.none::[int]` work the
same way. The result is an ordinary function value (payload) or nominal value
(nullary) and can be passed and called like any other.

## Strict parametricity

A value whose static type is a **bare type parameter** `T` is **opaque**: the
generic body knows nothing about it beyond that it exists. You may only
**pass it, return it, and store it**. Every operation that would inspect its
contents is a static error. You cannot:

- compare it with `==`, `!=`, `<`, … (`x == x` on a `T` is rejected),
- do arithmetic on it,
- access a field (`x.foo`) or index (`x[0]`) of it,
- test it with `is` / `is not`.

```agl
def bad[T](x: T) -> bool = x == x     # static error: '==' not permitted on 'T'
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

## Recursive generic types

A generic record or enum may reference itself, or another declaration that
in turn reaches back to it, in its own field or variant definitions — the
same recursion rule as [Recursive types](types.md#recursive-types), extended
to generics. The self-reference's type argument is not required to be the
declaration's own type parameter unchanged; it may be a different type built
from that parameter (polymorphic recursion):

```agl
record Pair[A, B]
  first: A
  second: B

enum Perfect[T]
  | Single(value: T)
  | Succ(next: Perfect[Pair[T, T]])
```

`Perfect[T]`'s `Succ` variant carries a `Perfect[Pair[T, T]]`, not a
`Perfect[T]` — each `Succ` layer doubles the "roundness" of the payload type
one level further. This is unrestricted: a recursive reference's argument may
combine any number of type parameters, containers, and other generic
declarations, and different references (in a mutually recursive group of
declarations) may each recurse at a different argument.

The [inhabitation](types.md#inhabitation) rule applies exactly as for a
non-generic recursive type: `Single` is the base-case variant that makes
`Perfect[T]` constructible for every `T`. Constructing, matching, comparing,
and folding a value works exactly like any other recursive type — a value is
always a finite tree, regardless of how many argument levels its declaration
can grow through:

```agl
let level0: Perfect[int] = Single(value = 1)
let level1: Perfect[int] = Succ(next = Single(value = Pair(first = 1, second = 2)))

def shape[T](p: Perfect[T]) -> text =
  case p of
    | Single(value) => "leaf"
    | Succ(next) => "deeper"

print shape(level0)   # "leaf"
print shape(level1)   # "deeper"
```

A generic function may recurse alongside such a type's own growth, calling
itself at a new instantiation to process the nested payload — the same
generic function definition serves every level, exactly like any other
generic function called at different type arguments; see
[Functions](functions.md) for generic function calls in general.

## Unqualified variant ambiguity

If two enums declare the same unqualified variant name, an unqualified
reference to that name is a **static ambiguity error** — regardless of
payload, surrounding context, or explicit type arguments. Disambiguate by
**qualifying** the reference (and patterns) with the owning enum:

```agl
enum Option[T]
  | none
  | some(value: T)

def describe_option(o: Option[int]) -> text =
  case o of
    | Option.none => "missing"
    | Option.some(value) => "found ${value}"

let d: Option[int] = Option.some(value = 11)
let line = describe_option(d)
print line
```

Qualification works in expression, pattern, and `is`-test positions
(`Option.some(value = 1)`, `case … | Option.none => …`,
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
