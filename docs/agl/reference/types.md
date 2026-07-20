# Types

[← Index](index.md)

AgL is statically typed with nominal user types, a small set of built-ins,
and exactly one implicit coercion. The full program is scope-resolved and
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
list[T]
dict[text, T]
agent
() -> B
A -> B
(A, B, …) -> C
```

Type expressions:

```ebnf
type_expr ::= "unit"
            | "text" | "json" | "bool" | "int" | "decimal"
            | NAME
            | NAME "[" type_expr ("," type_expr)* "]"   (* applied type *)
            | "list" "[" type_expr "]"
            | "dict" "[" "text" "," type_expr "]"
            | func_type

func_type ::= type_atom "->" type_expr
            | "(" type_list? ")" "->" type_expr
type_atom ::= "unit" | "text" | "json" | "bool" | "int" | "decimal"
            | NAME
            | NAME "[" type_expr ("," type_expr)* "]"
            | "list" "[" type_expr "]"
            | "dict" "[" "text" "," type_expr "]"
type_list ::= type_expr ("," type_expr)* ","?
```

A bare `NAME` in type position names a built-in type, a user type, an alias,
or — inside a generic declaration — one of that declaration's type parameters.
`NAME "[" … "]"` is an **applied type**: it instantiates a generic declaration
at concrete type arguments, e.g. `Box[int]`, `Option[text]`,
`Outcome[int, text]`, or nested `Box[Box[int]]`. The built-in `list[T]` and
`dict[text, V]` are the same applied-type form.

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
let _: unit = print "hello"
```

`unit` cannot be rendered, JSON-encoded, or interpolated. The literal `()` is
both the unit value and the empty argument list of a zero-argument call —
the two are syntactically unified.

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
lists/dictionaries of JSON-shaped values. The literal `null` has type `json`.

Records, enums, exceptions, functions, and agents are **not** JSON-shaped.

`null` is not assignable to `text`, `int`, `decimal`, `bool`, records, or
enums. Use an enum for optionality:

```agl
enum MaybeText
  | None
  | Some(value: text)
```

### `list[T]` and `dict[text, T]`

Homogeneous containers. Elements and values are read with indexing
(`xs[0]`, `metadata["key"]`). A mutable `var` binding can be updated through
an index with `:=`; this replaces the binding with a new list or dictionary
value. There is no `len` operator.

### `agent`

`agent` is an opaque type for declared agent values. Every `agent`
declaration introduces a name of this type. Agent values may be stored in
bindings, passed to functions, and held in lists, but have **no fields, no
operators, no JSON encoding, and only opaque rendering**. Rendering or
interpolating an agent value produces a handle such as `<agent reviewer>`.
Storing it where a JSON-shaped type is expected is a static error.

See [Agent calls](agent-calls.md) for how agent values are used with `ask`.

### Function types: `A -> B` and `(A, B, …) -> C`

A function value has a positional function type. The parameters appear as a
type or a parenthesized comma-separated list of types; the arrow `->` separates
the parameters from the result type. A single parameter may omit parentheses,
and chained arrows associate to the right:

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
compared with `=`. These restrictions exist because function values are
capability handles, not data.

See [Functions](functions.md) for the declaration and call syntax.

## Standard core types

The following types are defined by `std/core`, which the automatic prelude
opens in every loaded entry and library module except `std/core` itself.

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

### `AgentRequest`

`AgentRequest` is the first-attempt request that the matching `ask` call would
dispatch to its agent (see [Agent calls](agent-calls.md)):

```text
record AgentRequest
  agent:               text
  prompt:              text
  target_type:         Option[text]
  format_instructions: Option[text]
  json_schema:         Option[json]
  attempt:             int
  previous_error:      Option[text]
  metadata:            json
```

`target_type` is `None` for a `unit` response target and `Some("Review")`,
`Some("text")`, etc. otherwise. `format_instructions` and `json_schema` are
`None` when no such contract data applies. `previous_error` is `None` for
`ask-request` because it constructs only the first-attempt request.

## Record types

A `record` declares a nominal product type. The fields are written in an
indented block, one per line:

```agl
record Issue
  title: text
  severity: int
  description: text
```

All fields are required. By default, record fields are **standard**: they may be
supplied positionally or as `field = value`:

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
[Module-qualified type identity](#module-qualified-type-identity).
A record may be generic — `record Box[T]` then a field `value: T`
(see [Generics](generics.md)).

`builtin record` is the body-equivalent form for host-recognized nominal record
types in `std/core`. The name and full field shape must match a recognized
built-in type exactly.

## Enum types

An `enum` declares a tagged union (algebraic data type). Variants are
introduced by `|`; each variant is either nullary or carries named, typed
fields:

```agl
enum FixResult
  | Complete(output: text)
  | Changed(output: text)
  | Blocked(reason: text, recoverable: bool)
```

Enums are the intended model for agent outcomes:

```agl
enum Review
  | Pass
  | Fail(issues: list[text])
```

**Variant field zones.** Payload fields are standard by default, regardless of
payload arity. Common single-value variants and multi-field variants may both be
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

Zone markers are also available on variant payloads:

```agl
enum Triple
  | T(*, a: int, b: int, c: int)   # all fields named-only
```

Construction, qualification, and ambiguity rules are covered in
[Expressions](expressions.md); destructuring in
[Pattern matching](pattern-matching.md); the JSON wire shape (the `"$case"`
tag) in [Agent calls](agent-calls.md).

`builtin enum` similarly declares a host-recognized nominal enum type. Its
variant names and payload fields must match the built-in shape exactly.

The `builtin` and `private` modifiers behave like decorators on a type
declaration: a modifier may sit on the same line as the `record`/`enum` keyword
or on the line directly above it (the newline after the modifier is
insignificant).

## Recursive types

A record, enum, or exception may reference its own type, directly or through
another declaration, in its own field or variant definitions:

```agl
enum Tree
  | Leaf
  | Node(value: int, left: Tree, right: Tree)

record Category
  name: text
  subcategories: list[Category]

exception Retryable extends Exception
  causes: list[Retryable]
```

Mutual recursion — records, enums, and exceptions referencing each other in
any combination — is allowed, and a recursive type may span any number of
imported modules: a cycle through a cross-module reference is exactly as legal
as one within a single module.

### Inhabitation

Recursion is legal only when it is possible to build a finite value — the
type must be **inhabited**. Recursion is well-founded when at least one of
the following breaks the chain:

- an enum variant that does not need another value of the same (or a
  mutually recursive) type — a **base case**, such as `Leaf` above;
- a `list[T]`/`dict[text, T]` field whose element type is the recursive
  type — the empty list or dict is always a value, regardless of `T`, as
  with `Category.subcategories` above.

A record or exception whose every required field, or an enum whose every
variant, needs another value of the same or a mutually recursive
declaration with no such escape has no finite value and is rejected:

```agl
record Node
  next: Node
# Record type 'Node' is uninhabitable: every value of 'Node' would be
# infinite. Recursion must be guarded by an enum base-case variant or a
# list/dict field.
```

The same rule rejects an enum whose only variant carries itself, an exception
whose required fields contain an unguarded cycle, and a mutually recursive
pair with no base case or guard anywhere in the cycle (for example
`record A { b: B }` / `record B { a: A }`, with no list/dict field and no enum
alternative on either side). Abstract exception roots are inhabited only by
constructible descendants, not by their own fields alone. The error is
reported at the declaration and names its kind (`Record type`/`Enum type`/
`Exception type`).

Generic recursive types — a declaration referencing itself at a different
type argument, such as `Expr[T]` referencing `Expr[list[T]]` in its own body —
are constructible under the same rule; see [Generics](generics.md) for the
generics-specific recursion rules.

### Recursive aliases are not allowed

Unlike records, enums, and exceptions, a `type` alias may not be recursive,
directly or through another alias — see [Type aliases](#type-aliases). An
alias has no nominal identity of its own to anchor a cycle; recursion must
always pass through a named `record`, `enum`, or `exception`.

## Module-qualified type identity

Record types and enum types are identified by **both their name and their
defining module**. Two types with the same name are distinct types if they
come from different modules:

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

This is **deep nominal identity**: two `Point` types from different modules are
never interchangeable regardless of structural similarity.

### Qualified type references

A type from an imported module may be named explicitly using the module
qualifier:

```agl
import mylib
import mylib as M

# These are equivalent:
let p1: mylib::Point = mylib::origin()
let p2: M::Point     = M::origin()
```

A plain import provides suffix routes; `as` provides its alias route. Qualified
type references work in annotations, cast targets, and constructor expressions.
An open import also brings the type name into bare scope, so `Point` resolves
to `mylib::Point` if `mylib` is open-imported and no other open import clashes.

Generic imported types retain the same qualification rules. Apply type
arguments after the complete qualified name:

```agl
open import mylib

let p1: Box[int] = Box(value = 1)
let p2: mylib::Box[int] = mylib::Box(value = 2)
```

### Self-reference: `::TypeName`

Inside a module, `::TypeName` refers to the **current module's own** type
named `TypeName`. This resolves directly in the module root, bypassing any
shadow introduced by an open import:

```agl
# In mylib.agl
record Node
  value: int

def make() -> ::Node = Node(value = 0)   # refers to this module's own Node
```

## Type aliases

`type` declares a transparent alias:

```agl
type Status = Review
type Issues = list[Issue]
type Metadata = dict[text, json]
```

Aliases never create a new nominal type: a value of type `Status` *is* a
value of type `Review`. Aliases are transparent everywhere, including
qualified variant access. Alias chains resolve transitively.

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

Each type parameter is an ordinary `NAME` in scope as a type throughout the
declaration's body. A generic type is **used** by applying it to type
arguments — `Box[int]`, `Option[text]`, `Outcome[int, text]` — producing a
distinct concrete type for each instantiation.

### Invariance

Type arguments are **invariant**: an applied type matches another only when
their type arguments match exactly, with no variance or subtyping. The
`int → decimal` widening (below) does **not** propagate through type
arguments.

```agl
let xs: list[int] = [1, 2]
# let ys: list[decimal] = xs   # static error: list[int] ≠ list[decimal]
```

`Box[int]` and `Box[text]` are unrelated types, and `list[int]` is not
assignable to `list[json]`. The full generics model — declaration syntax,
inference, the `::[…]` override, and what may be done with a value of a type
parameter — is covered in [Generics](generics.md).

## Declaration validity

Before any value checking, the following are rejected as static errors:

1. A user type whose name duplicates another user type, a built-in type name,
   or a built-in exception name ([Exceptions](exceptions.md)).
2. Duplicate record fields, duplicate enum variants, or duplicate fields
   within one variant.
3. References to unknown types in records, enums, aliases, or `param`
   declarations.
4. Cyclic aliases.
5. An **uninhabitable** record, enum, or exception — see
   [Recursive types](#recursive-types).

Type declarations are valid only at the program root.

## Assignability and coercion

Typing is exact nominal matching with **one** implicit coercion:

1. **`int` widens to `decimal`.** An `int` value is accepted wherever a
   `decimal` is expected. Mixed arithmetic yields `decimal`, and `1 == 1.0`
   is true.
2. There are no other implicit conversions.
3. A `json` target accepts any JSON-shaped value. When a JSON-shaped value of
   a more specific static type is bound into a `json` slot, it is stored in
   canonical `json` representation.
4. Equality (`==`, `!=`) and ordering comparisons require both operands to
   have the *same* type after rule 1. Operands whose type is, or transitively
   contains, a function, agent, or `unit` value are a static error — see
   [Values and equality](#values-and-equality) below.
5. All branches of a `case` expression must have the same type after rule 1.

For explicit, user-requested conversions between types, see
[Casts and convertibility](#casts-and-convertibility) below.

## Casts and convertibility

AgL provides two cast operators:

- **`EXPR as T`** — converts the value of `EXPR` to type `T`. If the
  conversion cannot succeed at runtime it raises `CastError`
  ([Exceptions](exceptions.md)).
- **`EXPR as? T`** — tests whether the value of `EXPR` is convertible to `T`
  without actually performing the conversion. Always yields `bool`; never
  raises. Equivalent to asking whether `EXPR as T` would succeed.

The target type `T` is a type expression written the same way as any other
type annotation (`int`, `list[text]`, `MyRecord`, etc.).

Both operators are left-associative and sit at a precedence level between
unary `-` and `* /`. See [Lexical structure](lexical-structure.md) and
[Expressions](expressions.md) for the full precedence table and examples.

### Conversion matrix

The table below defines every permitted and rejected source–target pair.
**Static cast error** means the combination is rejected before the program
runs; these are never runtime failures. **Total** conversions always succeed
at runtime; **fallible** ones may raise `CastError`.

| Target type | Permitted source types | Outcome |
| ----------- | ---------------------- | ------- |
| `text` | any data type (`text`, `json`, `bool`, `int`, `decimal`, `list[E]`, `dict[text,V]`, record, enum, exception) | total — renders the value to its AgL-form text representation |
| `json` | `text`, `json`, `bool`, `int`, `decimal`, `list[E]`, `dict[text,V]` | total — canonicalizes the value to `json` |
| `json` | record, enum, exception | total — structural JSON encoding (record → field object; enum → object with `"$case"` tag; exception → all fields including `trace_id`) |
| `bool` | `bool` | total (no-op) |
| `bool` | `text`, `json` | fallible — value must be a JSON boolean |
| `int` | `int` | total (no-op) |
| `int` | `decimal` | fallible — decimal must have no fractional part |
| `int` | `text`, `json` | fallible — value must be an integral number |
| `decimal` | `decimal` | total (no-op) |
| `decimal` | `int` | total (widening, same as the implicit coercion) |
| `decimal` | `text`, `json` | fallible — value must be a number |
| `list[E]` | identical `list[E]` | total (no-op) |
| `list[E]` | `text`, `json` | fallible — strict JSON parse then element validation |
| `dict[text,V]` | identical `dict[text,V]` | total (no-op) |
| `dict[text,V]` | `text`, `json` | fallible — strict JSON parse then value validation |
| record `R` | same record `R` | total (no-op) |
| record `R` | `text`, `json` | fallible — strict JSON parse then field validation |
| enum `E` | same enum `E` | total (no-op) |
| enum `E` | `text`, `json` | fallible — strict JSON parse then variant validation |
| any type | `unit`, `agent`, function type | **static cast error** |
| `unit`, `agent`, function type | any type | **static cast error** |

Any source–target combination not listed above is a static cast error. In
particular: `bool as int`, `int as bool`, `bool as decimal`, `decimal as bool`
are all static errors — booleans never convert to or from numbers.

### Total vs fallible casts

A **total** cast is guaranteed to succeed at runtime; it never raises and has
no runtime cost beyond the conversion itself. Redundant total casts (casting
a value to its own type, or `int as decimal`) are accepted and are no-ops;
no warning is emitted.

A **fallible** cast may raise `CastError` if the value does not conform to
the target type. The `as?` form lets you probe convertibility without
handling an exception:

```agl
let ok: bool = some_json as? int   # true if the value is an integral number
```

### Strict parsing in text and json casts

When the source type is `text` or `json` and the target is a type that
requires structure (`bool`, `int`, `decimal`, list, dict, record, or enum),
the cast parses the text (or validates the JSON tree) using **strict JSON
parsing**: the input must be exactly one well-formed JSON value with no
surrounding prose, no Markdown fences, and no recovery. This contrasts with
agent-output parsing, which uses lenient recovery by default.

### `decimal as int` integrality

`decimal as int` succeeds only when the decimal value has no fractional part:
`3.0 as int` yields `3`, while `3.5 as int` raises `CastError`.

### Nominal types `as json` — structural encoding

Records, enums, and exceptions can be explicitly cast to `json` with `as json`.
This is a structural conversion:

- **record** → a JSON object with one key per field, in declaration order.
- **enum** → a JSON object with a `"$case"` key holding the variant name, plus
  one key per variant field.
- **exception** → a JSON object with all fields including `trace_id`, in
  declaration order.

This is an **explicit cast only**. Nominal values are not JSON-shaped and are
not implicitly assignable to `json`; `as json` must be written explicitly:

```agl
record R
  x: int

let r: R = R(x = 1)
print r              # → R(x = 1)      (AgL render form — the default)

# A json value printed (or interpolated) directly is top-level, so it renders
# as pretty, multi-line JSON (2-space indent):
print(r as json)     # → {
                     #      "x": 1
                     #    }
```

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
- Lists compare element-wise; dictionaries compare by key set and per-key
  values.
- Records and enums compare by type, variant (for enums), and field values.
- `json` values compare structurally.

**Opaque types** — `agent`, function types, and `unit` — have **no equality**.
A comparison involving one of these types is a static error. This rule is
**transitive**: a `list`, `dict`, `record`, `enum`, or `exception` that (at
any depth) contains a function, agent, or `unit` value likewise has no
equality and cannot be used with `==`/`!=`. For example, comparing two
`list[int -> int]` values with `==` is a static error, as is a
`record` with an `agent` field compared with `==`.

See [Expressions](expressions.md) for the operator rules and
[Pattern matching](pattern-matching.md) for variant tests with `is`.
