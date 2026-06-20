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
(A, B, …) -> C
```

Type expressions:

```ebnf
type_expr ::= "unit"
            | "text" | "json" | "bool" | "int" | "decimal"
            | TYPE_NAME
            | "list" "[" type_expr "]"
            | "dict" "[" "text" "," type_expr "]"
            | func_type

func_type ::= "(" type_list? ")" "->" type_expr
type_list ::= type_expr ("," type_expr)* ","?
```

`dict[text, T]` is the only dictionary form: keys are always `text`, and the
key position must be spelled literally as `text`. There are no union types,
no string-literal types, no optional/nullable types, and no user-defined
generics. Model alternatives and optionality with enums.

### `unit`

`unit` is the type of expressions that exist only for their side effect and
produce no meaningful value. Its single value is written `()` — the empty
argument list. Side-effecting expressions such as `print(…)`, `:=`, an
`if` without an `else` branch, and `do … until` loops all have type `unit`.

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
operators, no rendering, and no JSON encoding**. Interpolating an agent
value in a template or storing it where a JSON-shaped type is expected is a
static error.

See [Agent calls](agent-calls.md) for how agent values are used with `ask`.

### Function types: `(A, B, …) -> C`

A function value has a positional function type. The parameters appear as a
parenthesized comma-separated list of types; the arrow `->` separates the
parameters from the result type:

```agl
let f: (int) -> text = classify      # one param, text result
let g: (int, int) -> int = add       # two params, int result
let h: () -> bool = fn() => true     # zero params
```

Function type assignability is by **exact structural match** — same parameter
count, same parameter types in order, same result type. No variance or
subtyping applies.

Named and defaulted arguments are a property of declared names (`def`s and
built-ins), not of function value types. The value type is purely positional.

Function values have **no rendering, no JSON encoding, and no equality**.
A function value cannot be interpolated in a template, printed, stored in a
`json` slot, or compared with `=`. These restrictions exist because function
values are capability handles, not data.

See [Functions](functions.md) for the declaration and call syntax.

## Prelude types

The following types are defined by the language prelude and are available
without any `record` or `enum` declaration.

### `ExecResult`

A structured record returned by `exec` when no target annotation is given
(or when the annotation is explicitly `ExecResult`):

```text
stdout:    text
stderr:    text
exit_code: int
timed_out: bool
```

Field access works normally: `res.stdout`, `res.exit_code`, etc.

### `ParsePolicy`

An enum used as the `on_parse_error:` argument to `ask` and typed `exec`:

```text
enum ParsePolicy
  | Abort
  | Retry(n: int)
```

`Abort` is the portable default. `Retry(n: N)` permits up to `N`
corrective retries after the initial attempt.

### `AgentRequest` and `OutputContract`

The records surfaced by `ask-request` (see [Agent calls](agent-calls.md)).
`AgentRequest` is the first-attempt request that the matching `ask` call would
dispatch to its agent:

```text
record AgentRequest
  agent:           text
  prompt:          text
  attempt:         int
  output_contract: OutputContractOption
```

```text
enum OutputContractOption
  | None
  | Some(value: OutputContract)
```

`None` is used for a `unit` response target. `Some` contains the contract for
all response types that are parsed.

`OutputContract` carries the materialized contract for the call site:

```text
record OutputContract
  target_type:         text
  codec_name:          text
  strict_json:         json   (* null when the codec is not JSON-based *)
  format_instructions: text
  json_schema:         json   (* null when no schema applies *)
  structured_exec:     bool
```

`target_type` is the type's display name (e.g. `"Review"`, `"text"`).

## Record types

A `record` declares a nominal product type. The fields are written in an
indented block, one per line:

```agl
record Issue
  title: text
  severity: int
  description: text
```

All fields are required. Records are constructed with named arguments only:

```agl
let issue = Issue(
  title: "Missing tests",
  severity: 3,
  description: "No failure-path tests exist."
)
```

Two record types with identical fields are still distinct types (nominal
typing).

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

Construction, qualification, and ambiguity rules are covered in
[Expressions](expressions.md); destructuring in
[Pattern matching](pattern-matching.md); the JSON wire shape (the `"$case"`
tag) in [Agent calls](agent-calls.md).

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

## Declaration validity

Before any value checking, the following are rejected as static errors:

1. A user type whose name duplicates another user type, a built-in type name,
   or a built-in exception name ([Exceptions](exceptions.md)).
2. Duplicate record fields, duplicate enum variants, or duplicate fields
   within one variant.
3. References to unknown types in records, enums, aliases, or `param`
   declarations.
4. Cyclic aliases.
5. Directly or indirectly **recursive records or enums**. Recursive nominal
   types are not part of v1.

Type declarations are valid only at the program root.

## Assignability and coercion

Typing is exact nominal matching with **one** implicit coercion:

1. **`int` widens to `decimal`.** An `int` value is accepted wherever a
   `decimal` is expected. Mixed arithmetic yields `decimal`, and `1 = 1.0`
   is true.
2. There are no other implicit conversions.
3. A `json` target accepts any JSON-shaped value. When a JSON-shaped value of
   a more specific static type is bound into a `json` slot, it is stored in
   canonical `json` representation.
4. Equality (`=`, `!=`) and ordering comparisons require both operands to
   have the *same* type after rule 1.
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

let r: R = R(x: 1)
print r              # → R(x: 1)       (AgL form — the default)

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

Every **data** type has full value equality (`=` / `!=`):

- Scalars compare by value; `int` and `decimal` compare numerically.
- Lists compare element-wise; dictionaries compare by key set and per-key
  values.
- Records and enums compare by type, variant (for enums), and field values.
- `json` values compare structurally.

**Opaque types** — `agent` and function types — have **no equality**. A
comparison involving an agent or function value is a static error.

`unit` has one value (`()`), and `() = ()` is therefore trivially true, but
comparing `unit` values is rarely useful in practice.

See [Expressions](expressions.md) for the operator rules and
[Pattern matching](pattern-matching.md) for variant tests with `is`.
