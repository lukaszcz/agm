# Types

[← Index](index.md)

AgL is statically typed with nominal user types, a small set of built-ins,
and directed implicit coercions. The full program is scope-resolved and
type-checked before any expression executes; checking stops at the first
error, and a program with a static error never runs.

## Built-in types

```text
unit
text
json
bool
int
decimal
array[T]
dict[text, T]
() -> B
A -> B
(A, B, …) -> C
```

Type expressions:

```ebnf
type_expr ::= "unit"
            | "text" | "json" | "bool" | "int" | "decimal"
            | name
            | name "[" type_expr ("," type_expr)* "]"   (* applied type *)
            | qualifier_chain name "[" type_expr ("," type_expr)* "]"
                                                               (* qualified applied type *)
            | qualifier_chain name                            (* qualified type *)
            | "array" "[" type_expr "]"
            | "dict" "[" "text" "," type_expr "]"
            | func_type

func_type ::= type_atom "->" type_expr
            | "(" type_list? ")" "->" type_expr
type_atom ::= "unit" | "text" | "json" | "bool" | "int" | "decimal"
            | name
            | name "[" type_expr ("," type_expr)* "]"
            | qualifier_chain name "[" type_expr ("," type_expr)* "]"
            | qualifier_chain name
            | "array" "[" type_expr "]"
            | "dict" "[" "text" "," type_expr "]"
type_list       ::= type_expr ("," type_expr)* ","?
qualifier_chain ::= "::" qualifier_segment* | qualifier_segment+
qualifier_segment ::= ["/"] NAME ("/" NAME)* "::"
                    | NAME "[" type_expr ("," type_expr)* "]" "::"
```

A bare `name` in type position names a built-in type, a user type, an alias,
or — inside a generic declaration — one of that declaration's type parameters.
`name "[" … "]"` is an **applied type**: it instantiates a generic declaration
at concrete type arguments, e.g. `Box[int]`, `Option[text]`,
`Outcome[int, text]`, or nested `Box[Box[int]]`. A `qualifier_chain` may precede
the complete type name before its brackets, as in `mylib::Box[int]` or
`Geometry::Box[int]`; without brackets it forms a qualified type such as
`mylib::Point` or `Geometry::Point`. The built-in `array[T]` and `dict[text, V]`
are the same applied-type form. See [Named scopes](scopes.md) for scope-path
resolution.

An inline enum member may also be selected from an applied enum owner:
`Source[text]::Member` specializes every owner parameter captured by
`Member`. That selection is already concrete, so it cannot take another type
application; use `Source::Member[T]` when applying the member directly.

`dict[text, T]` keys are always `text`, and the key position must be spelled
literally as `text`. There are no union types, no string-literal types, and no
optional/nullable types; model alternatives and optionality with enums.

User declarations may themselves be **generic** — `record`, `enum`, `type`
aliases, and `def` functions can declare type parameters. See
[Generics](generics.md).

### `unit`

`unit` is the type of expressions that exist only for their side effect and
produce no meaningful value. The printable unit value is written `()` — the
empty argument list. Statement-like expressions return `void`, which has type
`unit`, compares equal to `()`, and is not echoed by the REPL. Side-effecting
expressions such as `print(…)`, `:=`, an `if` without an `else` branch, and
loops all have type `unit`.

```agl
program def main() -> unit =
  let _: unit = print "hello"
```

`unit` cannot be JSON-encoded or stored in a `json` slot; it renders and
interpolates as `()` — or `void` for the value produced by statement-like
effects. The literal `()` is both the unit value and the empty argument list
of a zero-argument call — the two are syntactically unified.

### `text`

An immutable Unicode string. Untyped `ask` results default to `text`
([Agent calls](agent-calls.md)).

### Numbers: `int` and `decimal`

There is **no binary floating-point type**.

- `int` — an arbitrary-precision integer.
- `decimal` — an exact decimal number. Literals with a fractional part, such
  as `1.5`, are `decimal`.

Arithmetic is performed under a fixed decimal context: 28 significant digits
with banker's rounding (round-half-even). This context is part of the
language semantics and does not vary by host.

On the JSON wire both kinds are plain JSON numbers, parsed and emitted
exactly. A wire number with an integral value (such as `1.0`) satisfies an
`int` target; a non-integral number does not.

### `bool`

`true` or `false`. Booleans never coerce to or from numbers.

### `json`

`json` holds any *JSON-shaped* value: `null`, booleans, numbers, text, and
arrays/dictionaries of JSON-shaped values. The literal `null` has type `json`.

Records, enums, exceptions, and functions are **not** JSON-shaped.

`null` is not assignable to `text`, `int`, `decimal`, `bool`, records, or
enums. Use an enum for optionality:

```agl
enum MaybeText
  | None
  | Some(value: text)
```

### `array[T]` and `dict[text, T]`

Homogeneous containers, and **mutable reference values**: binding, assignment,
passing as an argument, and storing in a field never copy an array or dict —
every alias shares the same underlying object. Elements and values are read
with indexing (`xs[0]`, `metadata["key"]`). An array or dict can be updated
in place through an index with `:=` — see [Bindings and scope](bindings-and-scope.md#--destructive-assignment)
for the assignment-root rules and evaluation order. There is no `len`
operator.

Records, enums, and exceptions are immutable — none of their fields can be
reassigned, and `with` ([Expressions](expressions.md)) builds a new value
rather than mutating one — but they too are shared by reference: two
bindings holding "the same" record share it, so a record or enum that holds
an array or dict makes that container's mutability observable through every
binding that reaches it.

See [Bindings and Scope](bindings-and-scope.md) for the `:=` indexed-assignment
rules, and [Foreign Function Interface](ffi.md) for boundary behavior. Arrays
and dicts cross the FFI as live views; immutable kinds cross as independent
values. Use
`copy` or `shallow_copy` before the call, or an explicit `as json` cast, when
a boundary snapshot is needed. (`as json` builds an independent container
snapshot — see [Casts and convertibility](#casts-and-convertibility) below —
and `with` ([Expressions](expressions.md)) builds a shallow copy of a record.
See [Copying values](#copying-values) below for `copy`/`shallow_copy`.)

#### Cycles

Because an array or dict is a mutable reference value, an indexed assignment
can make one hold a reference back to a container that (transitively)
contains it — a genuine reference cycle:

```agl
record Node(children: array[Node])
program def main() -> unit =
  var xs: array[Node] = [Node(children = [])]
  let n = Node(children = xs)
  xs[0] := n
```

A record, enum, or exception cannot be self-referential on its own — every
field is fixed at construction — so a cycle is always closed through at
least one array or dict, however many nominal layers it passes through.

Rendering (`print`, `render`, string interpolation, REPL echo), `as text`,
and `as json` walk a value's containers and therefore raise the catchable
`CyclicValueError` ([Exceptions](exceptions.md#cyclicvalueerror)) if the walk
re-enters a container already on its own path. An `extern def` call can pass a
cyclic array or dict as a live view; rendering that view in its companion
raises `CyclicValueError`. `as?` never
raises: it returns `false` when the corresponding conversion fails, so `as?
text` and `as? json` on a cyclic value evaluate to `false` instead. `copy` is
the exception: it is the one deep, structure-rebuilding walk that traverses a
cyclic value to completion instead of raising — see [Copying
values](#copying-values) below. A **shared** (diamond) structure — the same
array or dict reachable twice from different paths, but never from itself —
is not a cycle and renders normally.

Equality (`==`) is different: it never raises. Comparing two values assumes a
pair already being compared is equal, so `==` on a cyclic array or dict
terminates and answers the same structural relation co-inductively — the
comparison never needs a base case for the parts of the structure it has
already matched up.

Trace logging is a narrower exception still: a traced cyclic value is
recorded as a placeholder marker rather than raising — see
[Tracing](host-environment.md#tracing).

### Copying values

Because binding is by reference, a program that wants an independent value
must ask for one explicitly. `std/core` declares both host functions with
this signature:

<!-- agl-check: fragment -->
```agl
builtin def copy[T](value: T) -> T
builtin def shallow_copy[T](value: T) -> T
```

Both are generic and identity-typed — `T -> T` — so they compose with any
value and never change its type. An explicit type argument
(`copy::[decimal](5)`) is accepted and behaves like passing that value to any
other declaration expecting the explicit type: the argument must be
assignable to it. Like `print`, `render`, and `parse_json`, neither `copy`
nor `shallow_copy` can be bound as a function value — both are only valid in
call position.

`shallow_copy` rebuilds exactly **one** level: a new array, dict, record,
enum, or exception holding the *same* element or field references as the
original. Replacing the copy's own top-level contents does not affect the
original, but mutating a container reached *through* the copy — one level
down or deeper — is observed through the original too, because that nested
container is the same shared object. Every other value kind (`int`,
`decimal`, `bool`, `text`, `json`, `unit`, a function value) is returned as-is:
primitives are immutable, so there is nothing to detach. `shallow_copy`
never recurses, so it can never loop and never raises, even on a cyclic
value.

```agl
program def main() -> unit =
  var inner = [1, 2]
  var outer = [inner]
  let copied = shallow_copy(outer)
  copied[0][0] := 9
  let _ = print(outer)
```

`copy` is deep: every array, dict, record, enum, and exception reachable from
the value is rebuilt with independently copied contents, all the way down.
Function values reached along the way are returned as-is, exactly as for
`shallow_copy`: a container reachable only through a function value's captured
environment stays shared after the copy.
A `json` leaf copies as an independent value. **Sharing is preserved**: if
two fields or array slots pointed at the same array before the copy, they
still point at the same (new) array after it — a diamond copies to a
diamond, never to two independent arrays. `copy` is also the **one deep,
structure-rebuilding walk in the language that traverses a cyclic value to
completion**: it terminates and produces an isomorphic, independent cycle
rather than the `CyclicValueError` ([Exceptions](exceptions.md#cyclicvalueerror))
that every other container walk raises on one. The result is a genuinely
separate structure — printing or otherwise walking the *copy* of a cyclic
value still raises, exactly as it would for the original.

```agl
program def main() -> unit =
  var shared = [1]
  var pair = [shared, shared]
  let copied = copy(pair)
  copied[0][0] := 9
  let _ = print(pair)
  let _ = print(copied)
```

### Function types: `A -> B` and `(A, B, …) -> C`

A function value has a positional function type. The parameters appear as a
type or a parenthesized comma-separated list of types; the arrow `->` separates
the parameters from the result type. A single parameter may omit parentheses,
and chained arrows associate to the right:

<!-- agl-check: fragment -->
```agl
let f: int -> text = classify        # one param, text result
let g: (int, int) -> int = add       # two params, int result
let h: () -> bool = fn() => true     # zero params
let k: int -> text -> bool = chain   # int -> (text -> bool)
```

Function type assignability is by **exact structural match** — same parameter
count, same parameter types in order, same result type. No variance or
subtyping applies.

Named and defaulted arguments are a property of declared names (`def`s and
built-ins), not of function value types. The value type is purely positional.

Function values have **opaque rendering, no JSON encoding, and no equality**.
A function value can be rendered, interpolated, or printed as an opaque handle
such as `<function: int -> int>`, but cannot be stored in a `json` slot or
compared with `==` or `!=`. These restrictions exist because function values
are capability handles, not data.

See [Functions](functions.md) for the declaration and call syntax.

## Standard core types

The following types are defined by `std/core`. Every loaded entry and library
module except `std/core` itself receives an automatic `import std/core::*`, unless
`--no-stdlib` disables it or an explicit import whose expansion includes `std/core`
supplies the core contribution instead.

### `Option[T]`

A generic enum for optional values:

```text
enum Option[T]
  | None
  | Some(value: T)
```

`null` is only a value of type `json`; ordinary AgL types are not nullable.
Use `Option[T]` when a value may be absent.

### `ExecResult`

A structured record returned by `exec` when no target annotation is given
(or when the annotation is explicitly `ExecResult`):

```text
stdout:    text
exit_code: int
stderr:    text
timed_out: bool
```

Field access works normally: `res.stdout`, `res.exit_code`, etc.

### `ParsePolicy`

An enum used as the `on_parse_error` argument to `ask` and typed `exec`:

```text
enum ParsePolicy
  | Abort
  | Retry(n: int)
```

`Abort` is the portable default. `Retry(n: N)` permits up to `N`
corrective retries after the initial attempt.

### `Agent`

`Agent` is a built-in enum describing an agent backend. Its members are the
record types `AgentCommand(command)`, `AgentClaude(model, thinking)`,
`AgentCodex(model, thinking)`, and `AgentPi(provider, model, thinking)`.
Like every enum, `Agent` values have equality, rendering, and JSON casts; a
member record exposes its fields when used at its record type. Its standard-core `ask` and `ask-request` members are call-only builtin
methods, so `agent.ask(...)` and `agent.ask-request(...)` select that agent
for the operation; see [Agent calls](agent-calls.md) for dispatch behavior.

### `AgentRequest`

`AgentRequest` is the first-attempt text request that `ask-request` builds
without dispatching an agent (see [Agent calls](agent-calls.md)):

```text
record AgentRequest
  agent:               Agent
  prompt:              text
  target_type:         Option[text]
  format_instructions: Option[text]
  json_schema:         Option[json]
  attempt:             int
  previous_error:      Option[text]
  metadata:            json
```

`target_type` is always `Some("text")`; `format_instructions` and
`json_schema` are `None` because `ask-request` has a fixed text contract.
`previous_error` is `None` because it constructs only the first-attempt
request.

## Members of nominal types

A method is a member of a record, enum, or exception's nominal type. It is
available on every value of that type wherever the value is used, so calling it
does not require an import of the module that declared the type. See
[Methods](functions.md#methods) for declaration and call syntax.

In the REPL, redeclaring a record, enum, or exception starts a new
declaration rather than changing the existing one: a name always resolves to
its most recently declared owner, but a value built before the redeclaration
keeps working against the declaration it was built from — its fields or enum members, its methods, and equality with other values of that same
declaration are all unaffected. The new declaration starts with no methods of
its own; declare them again to use them on values of the new declaration.

The two declarations are unrelated types that happen to share a name, and
they are written and displayed identically: neither is usable where the other
is expected, and comparing values across them is a type error. Every spelling
that names the type — a constructor call, a type annotation, a `catch`
clause, a type-qualified constructor pattern — means the declaration in
effect where it is written, so one written after the redeclaration does not
apply to an earlier value. A bare member pattern is directed by the value being matched instead, so an
earlier value can still be destructured.

A failed entry that would have redeclared the type changes nothing — the
previous declaration, its methods, and every binding built from it remain in
effect. Redeclaring a record referenced by an existing enum does not change
that enum's member set; redeclaring an enum creates new identities for its
inline member records.

## Record types

A `record` declares a nominal product type. The fields are written in an
indented block, one per line:

```agl
record Issue
  title: text
  severity: int
  description: text
```

All fields are required. A fieldless record uses an empty parenthesized field
list (`record R1()`). Its constructor reference in value position constructs an
`R1` value; see [Expressions](expressions.md#fieldless-constructor-references).
By default, record fields are **standard**: they may be supplied positionally
or as `field = value`:

<!-- agl-check: fragment -->
```agl
let positional = Issue("Missing tests", 3, "No failure-path tests exist.")
let named = Issue(
  title = "Missing tests",
  severity = 3,
  description = "No failure-path tests exist."
)
```

**Zone markers** constrain fields to positional-only or named-only zones. Markers appear
in the parenthesized field list, as comma-separated zone entries, or in the indented block
form (as a leading marker on the header line or on its own line between fields):

```agl
# Inline / parenthesized forms: marker as comma entry
record Pair[T1, T2](fst: T1, snd: T2)                # both fields standard
record R(x: int, /, y: int)                          # x pos-only, y standard
record NamedPair(*, fst: int, snd: int)              # both fields named-only

# Block form: own-line marker between fields
record Mixed
  id: int
  *
  value: int
  label: text
```

<!-- agl-check: fragment -->
```agl
let p = Pair(1, 2)             # positional (both fields are standard)
let q = Pair(1, snd = 2)       # first positional, second named
let r = R(0, y = 1)            # x positional-only, y named (standard)
# R(x = 0, y = 1) is an error — x is positional-only
```

Definition-time rules: at most one `/`/`@std` and one `*`/`@named` per list, in zone
order; `@pos` must lead. Violations are static errors.

Two record types with identical fields are still distinct types (nominal
typing). Two record types from different modules are also distinct even if
they have the same name and the same fields — see
[Module- and scope-qualified type identity](#module--and-scope-qualified-type-identity).
A record may be generic — `record Box[T]` then a field `value: T`
(see [Generics](generics.md)).

`builtin record` is the body-equivalent form for host-recognized nominal record
types in `std/core`. The name and full field shape must match a recognized
built-in type exactly, and that complete scoped name may be declared only once
across the whole program. A program loading the default standard library, as
it does unless started with `--no-stdlib`, cannot redeclare a `std/core` type
at the same path (see [Built-in functions](functions.md#built-in-functions)).

## Enum types

An `enum` declares a closed nominal union of record types. A value of an enum
is one of its member-record values; constructing a member does not wrap or
retag the record. A bare member name declares a new record in the enum's
scope; it is either fieldless or carries named, typed fields:

```agl
enum FixResult
  | Complete(output: text)
  | Changed(output: text)
  | Blocked(reason: text, recoverable: bool)
```

A qualified member spelling instead references an existing record. The
reference may apply the enum's type parameters, and aliases are transparent:

```agl
record Saved(id: int)
record Box[T](value: T)

enum Result[T] = ::Saved | ::Box[T] | Fresh(value: T)
```

A reference must name a record, including through a transparent type alias.
An enum may not name the same member declaration twice, even with different
type arguments, and its members must have distinct terminal names. The enum's
scope contains only its inline declarations: `Result::Fresh` is available,
while `Result::Saved` is not; `Saved` remains reachable at its original
declaration path. Referencing a record does not re-export it. Every member's
terminal name is also an injected constructor and pattern candidate wherever
the enum is visible.

Each member is a record type. An inline member may appear in field, parameter,
return, and generic-argument positions such as `array[Result::Fresh]`; a
referenced member retains its own record type and declaration path. Member
records support record construction, field access, methods, `with`, casts, and
standalone JSON decoding exactly like other records. `with` applies while a
value has its member-record type, not after it has widened to the enum. A
member value widens to an enum only in a known enum-typed slot, so its inferred
type remains the member record type. An inline member captures exactly the enum type parameters
used by its fields, in enum declaration order. Thus `Tree[T]::Leaf` is
fieldless and non-generic, while `Tree[T]::Node(value: T)` captures `T`.

Enums are the intended model for agent outcomes. An enum establishes a
same-named scope for its declared members, so `Review::Pass` is a qualified
member access. The unqualified member spelling remains available under the ordinary
constructor rules.

```agl
enum Review
  | Pass
  | Fail(issues: array[text])
```

**Member field zones.** Member-record fields are standard by default,
regardless of arity. Single-value and multi-field members may both be
constructed positionally or by name:

```agl
enum Result
  | Ok(value: int)
  | Err(reason: text, fatal: bool)

let ok = Ok(42)
let ok2 = Ok(value = 42)
let err = Err("bad", false)
let named_err = Err(reason = "bad", fatal = false)
```

Zone markers are also available on inline member fields:

```agl
enum Triple
  | T(*, a: int, b: int, c: int)   # all fields named-only
```

Construction, qualification, and ambiguity rules are covered in
[Expressions](expressions.md); destructuring in
[Pattern matching](pattern-matching.md); the JSON wire shape (the `"$case"`
tag) in [Agent calls](agent-calls.md).

`builtin enum` similarly declares a host-recognized nominal enum type. Its
member names and fields must match the built-in shape exactly.

The `builtin` modifier behaves like a decorator on a type declaration: it may
sit on the same line as the `record`, `enum`, or `exception` keyword or on the
line directly above it (the newline after the modifier is insignificant).

A `builtin` type declaration also names the type the host produces values of
under that name: an unannotated `exec` returns that program's
`builtin record ExecResult`, `ask-request` builds its
`builtin record AgentRequest`, and a built-in exception the host raises is its
`builtin exception` of that name. Such a value carries that declaration's own
path, so it renders under that path and a `catch` clause naming the declaration
matches it. A name the program declares no `builtin` for keeps the standard
core type described in [Standard core types](#standard-core-types). A `catch`
clause naming the standard declaration of a name the program declares its own
`builtin` for is rejected, wherever in the program it is written, because the
host raises the program's own declaration under that name instead.

The host fills the fields of such a value itself, always with standard
values, so any field of one whose type is itself a nominal type must keep the
standard identity for that name — including a nominal nested inside a type
argument, such as the `Option` in `AgentRequest`'s `target_type:
Option[text]`. The fields subject to this are `AgentRequest`'s `agent: Agent`
and its `Option`-typed fields, and the `agent: Agent` field of the built-in
exceptions `AgentCallError` and `AgentParseError`; `ExecResult` has no
nominal field. So a program may declare its own `builtin enum Agent`, or —
without the standard library — its own `Option`, and it may separately
declare a `builtin record AgentRequest` or a `builtin exception
AgentCallError`; but writing both so that the redeclaration is what the
contract's own field resolves to makes the pair unusable together. That is
reported where the contract is used: at an `ask`/`ask-request` call for
`AgentRequest`, and at a `catch` clause naming the exception.

The agent an `ask`/`ask-request` dispatches to must likewise be a standard
`Agent`, whether it is supplied as the `agent` argument or as the receiver of
`agent.ask(...)` / `agent.ask-request(...)`. A value of a program's own
differently-scoped `Agent` is an ordinary type mismatch there.

## Recursive types

A record, enum, or exception may reference its own type, directly or through
another declaration, in its own fields or enum-member fields:

```agl
enum Tree
  | Leaf
  | Node(value: int, left: Tree, right: Tree)

record Category
  name: text
  subcategories: array[Category]

exception Retryable extends Exception
  causes: array[Retryable]
```

Mutual recursion — records, enums, and exceptions referencing each other in
any combination — is allowed, and a recursive type may span any number of
imported modules: a cycle through a cross-module reference is exactly as legal
as one within a single module.

### Inhabitation

Recursion is legal only when it is possible to build a finite value — the
type must be **inhabited**. Recursion is well-founded when at least one of
the following breaks the chain:

- an enum member that does not need another value of the same (or a
  mutually recursive) type — a **base case**, such as `Leaf` above;
- an `array[T]`/`dict[text, T]` field whose element type is the recursive
  type — the empty array or dict is always a value, regardless of `T`, as
  with `Category.subcategories` above.

A record or exception whose every required field, or an enum whose every
member, needs another value of the same or a mutually recursive declaration
with no such escape has no finite value and is rejected:

<!-- agl-check: error -->
```agl
record Node
  next: Node
# Record type 'Node' is uninhabitable: every value of 'Node' would be
# infinite. Recursion must be guarded by an enum base-case member or an
# `array`/`dict` field.
```

The same rule rejects an enum whose only member carries itself, an exception
whose required fields contain an unguarded cycle, and a mutually recursive
pair with no base case or guard anywhere in the cycle (for example
`record A { b: B }` / `record B { a: A }`, with no array/dict field and no enum
alternative on either side). Abstract exception roots are inhabited only by
constructible descendants, not by their own fields alone. The error is
reported at the declaration and names its kind (`Record type`/`Enum type`/
`Exception type`).

Generic recursive types — a declaration referencing itself at a different
type argument, such as `Expr[T]` referencing `Expr[array[T]]` in its own body —
are constructible under the same rule; see [Generics](generics.md) for the
generics-specific recursion rules.

### Recursive aliases are not allowed

Unlike records, enums, and exceptions, a `type` alias may not be recursive,
directly or through another alias — see [Type aliases](#type-aliases). An
alias has no nominal identity of its own to anchor a cycle; recursion must
always pass through a named `record`, `enum`, or `exception`.

## Module- and scope-qualified type identity

Every `record`, `enum`, and `exception` declaration introduces its own type,
named by its **defining module, scope path, and name**. Two types with the
same name are distinct when they come from different modules or different
named scopes, including a module root and a scope in that same module:

<!-- agl-check: fragment -->
```agl
# In foo.agl
record Point
  x: int
  y: int

# In bar.agl
record Point
  x: int
  y: int

# In entry.agl
import foo
import bar

let p: foo::Point = foo::Point(x = 0, y = 0)
# The next line is a static error: bar::Point is not the same type as foo::Point
# let q: bar::Point = p
```

This is **deep nominal identity**: two `Point` types from different modules or
scope paths are never interchangeable regardless of structural similarity.
Because a type is its declaration rather than its name, declaring one name
twice — which only the REPL allows — yields two distinct types sharing a
module, scope path, and name; see [Members of nominal
types](#members-of-nominal-types) for how that is resolved.

### Qualified type references

A type from an imported module may be named explicitly using the module
qualifier. A type in a named scope uses its complete scope path, such as
`A::Token`, in annotations and constructor expressions:

<!-- agl-check: fragment -->
```agl
import mylib
import mylib as M

# These are equivalent:
let p1: mylib::Point = mylib::origin()
let p2: M::Point     = M::origin()

scope A
record Token()
end A
let token: A::Token = A::Token()
```

A plain import provides suffix routes; `as` provides its alias route. Qualified
type references work in annotations, cast targets, and constructor expressions.
An import tail or `use` declaration also brings the selected type name into bare
scope, so `Point` resolves to `mylib::Point` when it has one bare contribution.

Generic imported types retain the same qualification rules. Apply type
arguments after the complete qualified name:

<!-- agl-check: fragment -->
```agl
import mylib::*

let p1: Box[int] = Box(value = 1)
let p2: mylib::Box[int] = mylib::Box(value = 2)
```

### Self-reference: `::TypeName`

Inside a module, `::TypeName` refers to the **current module's own** type
named `TypeName`. This resolves directly in the module root, bypassing any
shadow introduced by a bare import contribution:

```agl
# In mylib.agl
record Node
  value: int

def make() -> ::Node = Node(value = 0)   # refers to this module's own Node
```

## Type aliases

`type` declares a transparent alias:

<!-- agl-check: fragment -->
```agl
type Status = Review
type Issues = array[Issue]
type Metadata = dict[text, json]
```

Aliases never create a new nominal type: a value of type `Status` *is* a
value of type `Review`. Aliases are transparent everywhere, including
qualified member access. Alias chains resolve transitively.

## Type parameters and applied types

`record`, `enum`, `type`, and `def` declarations may take **type parameters**
in a bracketed list immediately after the declared name:

```agl
record Box[T]
  value: T

enum Option[T]
  | none
  | some(value: T)

type Pair[A, B] = dict[text, json]
```

Each type parameter is an ordinary `name` in scope as a type throughout the
declaration's body. A generic type is **used** by applying it to type
arguments — `Box[int]`, `Option[text]`, `Outcome[int, text]` — producing a
distinct concrete type for each instantiation.

### Invariance

Type arguments are **invariant**: an applied type matches another only when
their type arguments match exactly, with no variance or subtyping. The
`int → decimal` widening (below) does **not** propagate through type
arguments.

```agl
let xs: array[int] = [1, 2]
# let ys: array[decimal] = xs   # static error: array[int] ≠ array[decimal]
```

`Box[int]` and `Box[text]` are unrelated types, and `array[int]` is not
assignable to `array[json]`. The full generics model — declaration syntax,
inference, the `::[…]` override, and what may be done with a value of a type
parameter — is covered in [Generics](generics.md).

## Declaration validity

The following are static errors:

1. A user type whose name duplicates another user type, a built-in type name,
   or a built-in exception name ([Exceptions](exceptions.md)).
2. Duplicate record fields, duplicate enum member declarations or terminal
   names, or duplicate fields within one inline member.
3. References to unknown types in records, enums, aliases, or `param`
   declarations.
4. Cyclic aliases.
5. An **uninhabitable** record, enum, or exception — see
   [Recursive types](#recursive-types).

Type declarations are valid at the module root and in [named scope
regions](scopes.md), but not in ordinary expression blocks.

## Assignability and coercion

Typing is exact nominal matching with these implicit coercions:

1. **`int` widens to `decimal`.** An `int` value is accepted wherever a
   `decimal` is expected. Mixed arithmetic yields `decimal`, and `1 == 1.0`
   is true.
2. **A `json` target accepts any *scalar* JSON-shaped value** — `null`,
   `bool`, `int`, `decimal`, or `text` — storing it in canonical `json`
   representation.
3. **An enum member record widens to an enum that declares it.** This applies only
   when checking against a known enum slot; it never finds a common enum while
   inferring a mixed expression.
4. There are no other implicit conversions. In particular, an `array` or
   `dict` value — even one that is JSON-shaped — is never implicitly absorbed
   into `json`: an implicit conversion never copies a data structure, and
   converting a container to `json` builds one. Use an explicit `as json`
   cast (see [Casts and convertibility](#casts-and-convertibility) below).
5. Equality (`==`, `!=`) and ordering comparisons require both operands to
   have the *same* type after rule 1. Operands whose type is, or transitively
   contains, a function or `unit` value are a static error — see
   [Values and equality](#values-and-equality) below.
6. All branches of a `case` expression must have the same type after rule 1.

For explicit, user-requested conversions between types, see
[Casts and convertibility](#casts-and-convertibility) below.

## Casts and convertibility

AgL provides two cast operators:

- **`EXPR as T`** — converts the value of `EXPR` to type `T`. If the
  conversion cannot succeed at runtime it raises `CastError`
  ([Exceptions](exceptions.md)).
- **`EXPR as? T`** — tests whether the same conversion would succeed without
  raising. It yields `bool`: `true` on success and `false` on failure.

For an enum member record, these operators are identity casts rather than
parsing conversions. A member value may be cast up to an enum that declares
it only when that enum declares the member record; this upcast is a
compile-time-checked no-op. An enum value may be cast down only to one of that
enum's declared member records; this downcast checks the runtime nominal
identity and returns the same record value on success.

The target type `T` is a type expression written the same way as any other
type annotation (`int`, `array[text]`, `MyRecord`, etc.).

Both operators are left-associative and sit at a precedence level between
unary `-` and `* /`. See [Lexical structure](lexical-structure.md) and
[Expressions](expressions.md) for the full precedence table and examples.

### Conversion matrix

The table below defines every permitted and rejected source–target pair.
**Static cast error** means the combination is rejected before the program
runs. A **total** conversion has no conformance failure and therefore does not
raise `CastError`; rendering or JSON conversion can still raise
`CyclicValueError` when it walks a reference cycle. A **fallible** conversion
may raise `CastError`.

| Target type | Permitted source types | Outcome |
| ----------- | ---------------------- | ------- |
| `text` | any data type (`text`, `json`, `bool`, `int`, `decimal`, `array[E]`, `dict[text,V]`, record, enum, exception) | total for conformance — renders the value to its AgL-form text representation; a cyclic walk raises `CyclicValueError` |
| `json` | any type with a JSON representation — see [Convertibility to `json`](#convertibility-to-json) | total for conformance — canonicalizes the value to `json`; a cyclic walk raises `CyclicValueError` |
| `bool` | `bool` | total (no-op) |
| `bool` | `text`, `json` | fallible — value must be a JSON boolean |
| `int` | `int` | total (no-op) |
| `int` | `decimal` | fallible — decimal must have no fractional part |
| `int` | `text`, `json` | fallible — value must be an integral number |
| `decimal` | `decimal` | total (no-op) |
| `decimal` | `int` | total (widening, same as the implicit coercion) |
| `decimal` | `text`, `json` | fallible — value must be a number |
| `array[E]` | identical `array[E]` | total (no-op) |
| `array[E]` | `text`, `json` | fallible — strict JSON parse then element validation |
| `dict[text,V]` | identical `dict[text,V]` | total (no-op) |
| `dict[text,V]` | `text`, `json` | fallible — strict JSON parse then value validation |
| record `R` | same record `R` | total (no-op) |
| record `R` | `text`, `json` | fallible — strict JSON parse then field validation |
| enum `E` | same enum `E` | total (no-op) |
| member record `R` of enum `E` | `E` | total compile-time-checked identity upcast (no-op) |
| enum `E` | declared member record `R` | fallible identity downcast — checks that the runtime member is `R` |
| enum `E` | `text`, `json` | fallible — strict JSON parse then member validation |
| any type | `unit`, function type | **static cast error** |
| `unit`, function type | any type | **static cast error** |

Any source–target combination not listed above is a static cast error. In
particular: `bool as int`, `int as bool`, `bool as decimal`, `decimal as bool`
are all static errors — booleans never convert to or from numbers.

### Convertibility to `json`

A type converts to `json` with `as json` iff no **non-data** type — `unit` or
a function type — is reachable from it:

- the scalars `text`, `json`, `bool`, `int`, `decimal` always convert;
- `array[E]`/`dict[text, V]` converts iff `E`/`V` does;
- a record, enum, or exception converts iff no non-data type is reachable
  from its declaration, transitively through its fields (and, for an
  exception, through its `extends` ancestors and its catchable descendants,
  since a value statically typed as a base may hold a descendant at
  runtime);
- `unit` and function values never convert.

This makes `array[R] as json`, `dict[text, R] as json`, nested containers
(`array[array[R]]`, `dict[text, array[R]]`), and a recursive declaration such
as `record Node(tag: int, children: array[Node])` all convert, exactly as a
bare `R as json` does — a container converts whenever its element type does,
with no special case for a nominal element.

A **free type variable never converts**, so `T as json`, `array[T] as json`,
and `Box[T] as json` are static errors inside a generic `def`: type
arguments are erased, so at the point the cast is checked there is no way to
know whether the eventual instantiation of `T` will carry a non-data value.

### Total vs fallible casts

A **total** cast has no conformance failure, so it does not raise `CastError`.
Rendering or JSON conversion still raises `CyclicValueError` when it walks a
reference cycle; the corresponding `as?` expression yields `false` instead.
Redundant casts to the same type are accepted with no warning and are no-ops;
`int as decimal` is the accepted widening conversion. Casting an array or dict
to its own type (`xs as array[int]`) is a true no-op: it yields the *same*
value, not a copy, so a mutation through the result is visible through `xs`
and vice versa. This differs from `as json` on a container, which builds an
independent snapshot (see [`array[T]` and `dict[text, T]`](#arrayt-and-dicttext-t)
above).

A **fallible** cast may raise `CastError` if the value does not conform to
the target type. The `as?` form instead returns whether that cast would
succeed, without handling an exception:

<!-- agl-check: fragment -->
```agl
let parses_as_int: bool = some_json as? int
```

### Strict parsing in text and json casts

When the source type is `text` or `json` and the target is a type that
requires structure (`bool`, `int`, `decimal`, array, dict, record, or enum),
the cast parses the text (or validates the JSON tree) using **strict JSON
parsing**: the input must be exactly one well-formed JSON value with no
surrounding prose, no Markdown fences, and no recovery. This contrasts with
agent-output parsing, which uses lenient recovery by default.

### `decimal as int` integrality

`decimal as int` succeeds only when the decimal value has no fractional part:
`3.0 as int` yields `3`, while `3.5 as int` raises `CastError`.

### Nominal types `as json` — structural encoding

Records, enums, and exceptions can be explicitly cast to `json` with `as
json`, and so can any `array`/`dict` built from them. This is a structural
conversion:

- **record** → a JSON object with one key per field, in declaration order.
- **enum** → a JSON object with a `"$case"` key holding the terminal member
  name, plus one key per member-record field. The same record in a
  record-typed slot has no `"$case"` key.
- **exception** → a JSON object with all fields in declaration order.
- **`array[E]`/`dict[text, V]`** → the JSON array/object obtained by
  converting each element/value the same way — so `array[R] as json` is a
  JSON array of record objects, and a nested `array[array[R]]` or
  `dict[text, array[R]]` converts to the matching nested JSON shape.

This encoding is exactly what the decode direction (`json as R`, `text as
array[R]`, and the other record/enum/array/dict rows in the [conversion
matrix](#conversion-matrix) above) accepts, so a round trip through `json`
recovers the original value: `(rs as json) as array[R] == rs`.

This is an **explicit cast only**. Nominal values are not JSON-shaped and are
not implicitly assignable to `json`; `as json` must be written explicitly:

```agl
record R
  x: int

program def main() -> unit =
  let r: R = R(x = 1)
  let _ = print r
  let _ = print(r as json)
  let _ = print render(r as json, pretty = true)
```

A record (or exception) with a field of type `unit` or a function type cannot
be converted — see [Convertibility to
`json`](#convertibility-to-json) above for the static error this produces and
how it names the offending field. A JSON-convertible recursive value can be
converted to `json` when its runtime value is finite, including a growing
polymorphic-recursive value such as `Perfect[int]`. This does not make that
type eligible for a JSON-schema boundary; those positions require a finite
schema as described in [Generics](generics.md#the-finite-schema-boundary).

### `text as json` — embedding, not parsing

Because `text` is already JSON-shaped, `"42" as json` produces the JSON
**string** `"42"` — it wraps the text in JSON representation and does not
interpret it as a JSON value. This is a total, no-parse cast.

To parse the *contents* of a text as JSON, use the built-in
**`parse_json`** function ([Expressions](expressions.md)). `parse_json("42")`
produces the JSON number `42` and raises `JsonParseError`
([Exceptions](exceptions.md)) on malformed input.

## Values and equality

Every **data** type has full value equality (`==` / `!=`):

- Scalars compare by value; `int` and `decimal` compare numerically.
- Arrays compare element-wise; dictionaries compare by key set and per-key
  values.
- Records compare by nominal type and field values; enum values compare by
  their member-record nominal type and field values.
- `json` values compare structurally.

Function types and `unit` have **no equality**. A comparison involving one of
these types is a static error. This rule is **transitive**: an `array`, `dict`,
`record`, `enum`, or `exception` that (at any depth) contains a function or
`unit` value likewise has no equality and cannot be used with `==`/`!=`. For
example, comparing two `array[int -> int]` values with `==` is a static error.

See [Expressions](expressions.md) for the operator rules and
[Pattern matching](pattern-matching.md) for enum-member tests with `is`.
