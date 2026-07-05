# Exceptions

[← Index](index.md)

Runtime failures in AgL are **exceptions** — typed values that propagate up
through expressions, blocks, and calls until caught by a `try`/`catch` or,
uncaught, terminate the program. There are no sentinel return values.

## The exception model

Every exception is a value of one of the built-in exception types listed
below. All of them conceptually extend the abstract base type `Exception`,
which declares two fields present on every exception:

```text
message: text     # human-readable description
trace_id: text    # links the exception to its record in the run trace
```

`Exception` itself is **not constructible** — `raise Exception(…)` is a
static error. It exists for typing: a wildcard catch binds its variable as
`Exception`, so only `message`, `trace_id`, and whole-value interpolation
(`${e}`) are available there. Accessing a subtype field such as `e.raw`
requires catching the concrete type.

Programs may declare concrete exception types:

```agl
exception DeployError extends Exception
  service: text
  exit_code: int
```

An exception extends exactly one base exception type. Constructor fields include
the inherited fields first, followed by fields declared on the subtype.
`builtin exception` is the standard-library form for host-recognized exception
types; the name, base, and fields must match the recognized shape exactly.

### Recursive exceptions

An exception's fields may reference its own type, and may participate in
mutual recursion with records, enums, and other exceptions, under the same
inhabitation rule as records and enums (see
[Recursive types](types.md#recursive-types)). Because an exception is a
product type — every declared field is required — a field cannot directly
self-reference without a base case:

```agl
exception ValidationError extends Exception
  field_name: text
  causes: list[ValidationError]   # legal: guarded by list[...]
```

```agl
exception Broken extends Exception
  child: Broken   # rejected: uninhabitable, no list/dict guard
```

A `list`/`dict` field is the only guard available for a required field —
there is no enum-style alternative variant to fall back on. The same
inhabitation rule also covers the `extends` chain itself: an `extends`
cycle (two exceptions each extending the other) is rejected as uninhabitable
for the same reason a field cycle is.

Exception values support field access (`e.raw`), equality, and rendering.
In interpolation and `print` an exception renders in **AgL record form**,
including all fields (`message`, `trace_id`, and any type-specific fields)
in declaration order — for example:

```
CastError(message = "cannot parse \"x\" as int", trace_id = "evt-7", source_type = "text", target_type = "int", raw = "x")
```

See [Strings and interpolation](strings-and-interpolation.md) for the uniform
rendering rules. Use `e as json` to obtain the JSON object of the exception's
fields; use `e as text` to obtain the same AgL-form string.

## `try` / `catch`

`try … catch …` is an **expression** — its type is the unified type of the
body and all catch handler bodies (with `int → decimal` widening). A
`try`/`catch` in a value position binds a typed result:

```ebnf
try_expr      ::= "try" block catch_clause+
catch_clause  ::= "catch" catch_pattern "=>" branch_body
catch_pattern ::= NAME ("as" NAME)?
                | "_" ("as" NAME)?
```

```agl
try
  let review: Review = ask(
    "Review ${artifact}",
    agent = reviewer,
    on_parse_error = Retry(n = 2)
  )
catch AgentParseError as e =>
  let report = ask("Explain invalid output:\n${e.raw}", agent = critic)
  raise e
catch _ as e =>
  print "unexpected: ${e.message}"
```

Semantics:

1. The `try` body executes.
2. If it completes without an exception, every `catch` is skipped; the
   body's value is the result.
3. If an exception is raised, catch clauses are tested **in order**; the
   first clause whose pattern matches handles it.
4. The catch body runs in a fresh scope; `as name` binds the exception as
   an immutable, handler-local value typed by the caught type.
5. If no clause matches, the exception propagates outward.
6. An exception raised inside a catch body propagates normally.

Catch patterns:

- `catch SomeError` / `catch SomeError as e` — matches exactly that built-in
  exception type.
- `catch _` / `catch _ as e` — matches anything; `e` has type `Exception`.
- `catch Exception as e` — equivalent to `catch _ as e`.

There is no `finally`.

## `raise`

```ebnf
raise_expr ::= "raise" expr
```

The operand must be an exception value (statically checked). `raise`
**diverges** — it never yields a value. Its type is the bottom type,
assignable to any expected type:

```agl
let x: int = if condition => 1 else => raise Abort(message = "!")
raise Abort(message = "Cannot continue without repository access.")
```

Exception fields are **named-only by default** — each is supplied as
`field = value` or via the bare-name shorthand. Zone markers (`/`, `*`,
`@pos`, `@std`, `@named`) may opt an exception's own fields into the standard
or positional-only zone, but the inherited `message` field is always
named-only, so exceptions are in practice constructed by name (or shorthand).

Any concrete built-in exception type is constructible with named arguments
for its fields; `trace_id` is injected by the runtime and is not written
in source when omitted. The same construction rule applies to user-declared
exception types. `Abort` is the conventional type for user-initiated failures.

## Built-in exception catalog

Field lists below are in addition to the base `message` and `trace_id`.

### `AgentCallError`

An agent **transport** failure: the agent could not run. Not eligible for
`on_parse_error` retries ([Agent calls](agent-calls.md)).

```text
agent: text       # the callee name
cause: text       # "spawn_failure" | "nonzero_exit" | "timeout"
metadata: json    # host details: exit code, stderr tail, elapsed seconds
```

### `AgentParseError`

Structured agent (or typed `exec`) output failed parsing or validation after
all attempts allowed by the parse policy.

```text
agent: text             # callee name ("exec" for shell calls)
target_type: text       # the contract's target type, e.g. "Review"
expected_schema: json   # the derived JSON Schema
raw: text               # the last attempt's raw output
normalized_raw: text    # the recovered/normalized JSON text (or the raw output)
validation_errors: json # structured error records (category, message, path, field)
attempts: int           # total attempts made
metadata: json
```

### `ExecError`

A shell command failed to run, exited nonzero, or timed out in the **parsed
form** of `exec` ([Shell execution](shell-execution.md)). Also raised by the
structured form on spawn failure.

```text
command: text     # the rendered command
exit_code: int    # -1 for spawn failure or timeout without an exit status
stdout: text
stderr: text
timed_out: bool
```

### `MaxIterationsExceeded`

A **bounded** loop (`[n]`, with `n ≥ 1`) exhausted its bound before an exit
condition triggered ([Control flow](control-flow.md)). An unbounded loop (no
`[n]`) never raises this exception.

```text
limit: int                  # the bound in effect
condition: text             # source text of the until-condition
last_condition_value: bool  # the condition's final value
metadata: json
```

### `RecursionError`

A user-defined function call exceeded the runtime call-depth limit
([Functions](functions.md)).

```text
limit: int    # the depth limit that was exceeded
```

`RecursionError` is catchable. It is raised when the call-depth counter
exceeds the configured limit (default 256) while evaluating a chain of
user-function activations.

### `MatchError`

A `case` with no matching branch and no wildcard
([Pattern matching](pattern-matching.md)).

```text
scrutinee_type: text   # type name of the unmatched value
scrutinee: json        # structural JSON encoding of the unmatched value
```

### `ArithmeticError`

Raised by division by zero.

```text
operation: text    # the operator, e.g. "/"
```

### `TypeError`

```text
(base fields only)
```

### `IndexError`

Raised by out-of-range list indexing or indexed list assignment.

```text
index: int
length: int
```

### `KeyError`

Raised by missing dictionary keys during indexing or indexed dictionary
assignment.

```text
key: text
```

### `UndefinedVariableError`

```text
name: text
```

### `ImmutableBindingError`

```text
name: text
operation: text
```

`TypeError`, `UndefinedVariableError`, and `ImmutableBindingError` are
prevented statically in normal programs — type errors, reads of undefined
names, and `:=` on immutable bindings are all static errors — but the types
exist, are catchable, and may be constructed and raised explicitly.

### `CastError`

A fallible `as` cast failed at runtime: the source value did not conform to
the target type.

```text
source_type: text   # name of the source type, e.g. "json"
target_type: text   # name of the target type, e.g. "int"
raw: text           # text representation of the value that failed to convert
```

`CastError` is raised by `as` casts that are fallible (see
[Types](types.md#casts-and-convertibility)). The `as?` form never raises —
it reports failure as `false`.

### `JsonParseError`

The `parse_json` built-in received text that is not a well-formed JSON
document ([Expressions](expressions.md#parse_json)).

```text
raw: text   # the input text that failed to parse
```

### `RangeError`

Raised when a range `for` step (`by k`) evaluates to a non-positive `int`
(`k ≤ 0`) at loop entry. Carries only the base fields. It is catchable.

```text
(base fields only)
```

### `Abort`

The general-purpose user abort; carries only the base fields.

```text
(base fields only)
```

## Where exceptions come from

| Source | Exception |
| ------ | --------- |
| Out-of-range list index access or assignment | `IndexError` |
| Missing dictionary key access or assignment | `KeyError` |
| Agent transport failure | `AgentCallError` |
| Invalid structured output after all attempts | `AgentParseError` |
| Failing/timed-out shell command (parsed form) | `ExecError` |
| Spawn failure (either exec form) | `ExecError` |
| Loop bound exhausted | `MaxIterationsExceeded` |
| Non-positive range `for` step (`by k` with `k ≤ 0`) | `RangeError` |
| Call-depth limit exceeded | `RecursionError` |
| Non-exhaustive `case` at runtime | `MatchError` |
| Division by zero | `ArithmeticError` |
| Fallible `as` cast — source does not conform to target type | `CastError` |
| `parse_json` — input is not well-formed JSON | `JsonParseError` |
| `raise` of a constructed or re-raised value | any concrete type |

An exception that reaches the top of the program uncaught terminates the
run; the host reports the exception's type, fields, and source location, and
records it in the trace ([Host environment](host-environment.md)).
