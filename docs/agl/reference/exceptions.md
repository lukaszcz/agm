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
requires catching the concrete type. User-defined exception types do not
exist in v1.

Exception values support field access (`e.raw`), equality, and rendering:
in interpolation and `print` an exception renders as the JSON object of its
fields ([Strings and interpolation](strings-and-interpolation.md)).

## `try` / `catch`

`try … catch …` is an **expression** — its type is the unified type of the
body and all catch handler bodies (with `int → decimal` widening). A
`try`/`catch` in a value position binds a typed result:

```ebnf
try_expr      ::= "try" block catch_clause+
catch_clause  ::= "catch" catch_pattern "=>" branch_body
catch_pattern ::= TYPE_NAME ("as" VAR_NAME)?
                | "_" ("as" VAR_NAME)?
```

```agl
try
  let review: Review = ask(
    "Review ${artifact}",
    agent: reviewer,
    on_parse_error: Retry(n: 2)
  )
catch AgentParseError as e =>
  let report = ask("Explain invalid output:\n${e.raw}", agent: critic)
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

There is no `finally` in v1.

## `raise`

```ebnf
raise_expr ::= "raise" expr
```

The operand must be an exception value (statically checked). `raise`
**diverges** — it never yields a value. Its type is the bottom type,
assignable to any expected type:

```agl
let x: int = if condition => 1 else => raise Abort(message: "!")
raise Abort(message: "Cannot continue without repository access.")
```

Any concrete built-in exception type is constructible with named arguments
for its fields; `trace_id` is injected by the runtime and is not written
in source. `Abort` is the conventional type for user-initiated failures.

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

A `do … until` loop exhausted its bound ([Control flow](control-flow.md)).

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
scrutinee: json        # JSON rendering of the unmatched value
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
names, and `set` on immutable bindings are all static errors — but the types
exist, are catchable, and may be constructed and raised explicitly.

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
| Call-depth limit exceeded | `RecursionError` |
| Non-exhaustive `case` at runtime | `MatchError` |
| Division by zero | `ArithmeticError` |
| `raise` of a constructed or re-raised value | any concrete type |

An exception that reaches the top of the program uncaught terminates the
run; the host reports the exception's type, fields, and source location, and
records it in the trace ([Host environment](host-environment.md)).
