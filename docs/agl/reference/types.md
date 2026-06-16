# Types

[← Index](index.md)

AgL is statically typed with nominal user types, a small set of built-ins,
and exactly one implicit coercion. The full program is scope-resolved and
type-checked before any statement executes; checking stops at the first
error, and a program with a static error never runs.

## Built-in types

```text
text
json
bool
int
decimal
list[T]
dict[text, T]
```

Type expressions:

```ebnf
type_expr ::= "text" | "json" | "bool" | "int" | "decimal"
            | TYPE_NAME
            | "list" "[" type_expr "]"
            | "dict" "[" "text" "," type_expr "]"
```

`dict[text, T]` is the only dictionary form: keys are always `text`, and the
key position must be spelled literally as `text`. There are no union types,
no string-literal types, no optional/nullable types, and no user-defined
generics. Model alternatives and optionality with enums.

### `text`

An immutable Unicode string. Untyped agent-call results default to `text`
([Agent calls](agent-calls.md)).

### Numbers: `int` and `decimal`

There is **no binary floating-point type**.

- `int` — an arbitrary-precision integer.
- `decimal` — an exact decimal number. Literals with a fractional part, such
  as `1.5`, are `decimal`.

Arithmetic is performed under a fixed decimal context: 28 significant digits
with banker's rounding (round-half-even). Results of operations whose exact
value needs more than 28 significant digits — such as `1 / 3` — are rounded
to 28 significant digits. This context is part of the language semantics and
does not vary by host.

On the JSON wire both kinds are plain JSON numbers, parsed and emitted
exactly, never through a binary float. A wire number with an integral value
(such as `1.0`) satisfies an `int` target; a non-integral number does not.

### `bool`

`true` or `false`. Booleans never coerce to or from numbers — including
inside `json` values, where `true` is never equal to `1`.

### `json`

`json` holds any *JSON-shaped* value: `null`, booleans, numbers, text, and
lists/dictionaries of JSON-shaped values. The literal `null` has type `json`.

Records, enums, and exceptions are **not** JSON-shaped and are not assignable
to `json`. When interpolated in a template, structured values render as pretty
JSON automatically ([Strings and interpolation](strings-and-interpolation.md)).

`null` is not assignable to `text`, `int`, `decimal`, `bool`, records, or
enums. Use an enum for optionality:

```agl
enum MaybeText
  | None
  | Some(value: text)
```

### `list[T]` and `dict[text, T]`

Homogeneous containers. There is no indexing, no `len`, and no element
access syntax in v1; lists and dictionaries are consumed via interpolation,
membership tests (`in`), equality, and agent prompts. See
[Expressions](expressions.md) for literals and operations.

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

The variant list may follow on the same line or on subsequent lines aligned
with (or indented under) the `enum` keyword — lines beginning with `|`
continue the declaration ([Lexical structure](lexical-structure.md)).

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
qualified variant access — if `type Status = Review`, then `Status.Pass`,
`status is Status.Pass`, and the pattern `Status.Pass` all resolve to
`Review.Pass`. Alias chains (aliases of aliases) resolve transitively.

An alias of a structural type (such as `type Issues = list[Issue]`) cannot be
used as a constructor qualifier, since it does not name an enum.

## Declaration validity

Before any value checking, the following are rejected as static errors:

1. A user type whose name duplicates another user type, a built-in type name
   (`text`, `json`, `bool`, `int`, `decimal`), or a built-in exception name
   ([Exceptions](exceptions.md)).
2. Duplicate record fields, duplicate enum variants, or duplicate fields
   within one variant.
3. References to unknown types in records, enums, aliases, or `input`
   declarations.
4. Cyclic aliases (`type A = B` with `type B = A`).
5. Directly or indirectly **recursive records or enums** — including
   recursion routed through an alias. Recursive nominal types are not part
   of v1.

Duplicate constructor arguments, duplicate pattern fields, duplicate call
options, and duplicate dictionary-literal keys are also static errors;
silently keeping one occurrence never happens.

Type declarations are valid only at the program root; a `record`, `enum`, or
`type` inside a nested block is a static error.

## Assignability and coercion

Typing is exact nominal matching with **one** implicit coercion:

1. **`int` widens to `decimal`.** An `int` value is accepted wherever a
   `decimal` is expected. Mixed arithmetic yields `decimal`
   ([Expressions](expressions.md)), and `1 = 1.0` is true.
2. There are no other implicit conversions — not between `text` and numbers,
   not between `bool` and anything, not from records/enums to `json`.
3. A `json` target accepts any JSON-shaped value (rule 3 above describes the
   shape). When a JSON-shaped value of a more specific static type is bound
   into a `json` slot, it is stored in canonical `json` representation, so
   rendering and equality behave uniformly regardless of which literal it
   came from.
4. Equality (`=`, `!=`) and ordering comparisons require both operands to
   have the *same* type after rule 1. Note that this is stricter than
   assignability: `json` compares only with `json` — comparing a `json`
   value with a `text` or `int` value is a static error even though that
   value would be *assignable* to `json`.
5. All branches of a `case` expression must have the same type after rule 1;
   no other coercion to a common type exists
   ([Expressions](expressions.md)).

## Values and equality

Every type has full value equality (`=` / `!=`):

- Scalars compare by value; `int` and `decimal` compare numerically.
- Lists compare element-wise; dictionaries compare by key set and per-key
  values.
- Records and enums compare by type, variant (for enums), and field values.
- `json` values compare structurally with numeric equivalence (`1` equals
  `1.0`) but with booleans kept distinct from numbers (`true` never equals
  `1`).

See [Expressions](expressions.md) for the operator rules and
[Pattern matching](pattern-matching.md) for variant tests with `is`.
