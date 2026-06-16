# Exceptions

[← Index](index.md)

Runtime failures in AgL are **exceptions** — typed values that propagate up
through statements, loops, branches, and agent calls until caught by a
`try`/`catch` or, uncaught, terminate the program. There are no sentinel
return values.

## The exception model

Every exception is a value of one of the built-in exception types listed
below. All of them conceptually extend the abstract base type `Exception`,
which declares two fields present on every exception:

```text
message: text     # human-readable description
trace_id: text    # links the exception to its record in the run trace
```

`Exception` itself is **not constructible** — `raise Exception(…)` is a
static error (*use a concrete type such as `Abort`*). It exists for typing:
a wildcard catch binds its variable as `Exception`, so only `message`,
`trace_id`, and whole-value interpolation (`${e}`) are available there.
Accessing a subtype field such as `e.raw` requires catching the concrete
type. User-defined exception types do not exist in v1.

Exception values support field access (`e.raw`), equality, and rendering:
in interpolation and `print` an exception renders as the JSON object of its
fields ([Strings and interpolation](strings-and-interpolation.md)).

## `try` / `catch`

```ebnf
try_stmt      ::= "try" try_body catch_clause+
catch_clause  ::= "catch" catch_pattern "=>" catch_body
catch_pattern ::= TYPE_NAME ("as" VAR_NAME)?
                | "_" ("as" VAR_NAME)?
```

```agl
try
  let review: Review = reviewer[on_parse_error: retry[2]] "Review ${artifact}"
catch AgentParseError as e =>
  let report = critic "Explain invalid output:\n${e.raw}"
  raise e
catch _ as e =>
  print "unexpected: ${e.message}"
```

Semantics:

1. The `try` body executes (in its own scope).
2. If it completes without an exception, every `catch` is skipped.
3. If an exception is raised, catch clauses are tested **in order**; the
   first clause whose pattern matches handles it.
4. The catch body runs in a fresh scope; `as name` binds the exception as an
   immutable, handler-local value typed by the caught type.
5. If no clause matches, the exception propagates outward.
6. An exception raised inside a catch body propagates normally (it is not
   re-tested against this `try`'s clauses).

Catch patterns are intentionally simple:

- `catch SomeError` / `catch SomeError as e` — matches exactly that built-in
  exception type. The name must be a known exception type (a static error
  otherwise; a lowercase word other than `_` is rejected with *"… is not an
  exception type name"*).
- `catch _` / `catch _ as e` — matches anything; `e` has type `Exception`.
- `catch Exception as e` — equivalent to `catch _ as e`.

There is no `finally` in v1.

Inline form: a `catch` body on one line is a single bar-safe statement;
anything more needs a suite. A `catch` keyword may start its own line at the
`try`'s indentation (branch-marker continuation,
[Lexical structure](lexical-structure.md)).

## `raise`

```ebnf
raise_stmt ::= "raise" expr
```

The operand must be an exception value (statically checked):

- **Re-raise** an exception in scope: `raise e` — legal even when `e` has the
  abstract type `Exception` (from a wildcard catch).
- **Construct and raise** a built-in exception:

  ```agl
  raise Abort(message: "Cannot continue without repository access.")
  ```

Any concrete built-in exception type is constructible with named arguments
for its fields ([Expressions](expressions.md)); `trace_id` is injected by
the runtime and is not written in source. `Abort` is the conventional type
for user-initiated failures.

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

A shell command failed to run, exited nonzero, or timed out
([Shell execution](shell-execution.md)).

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
names, and `set` on immutable bindings are all compile-time errors — but the
types exist, are catchable, and may be constructed and raised explicitly.

### `Abort`

The general-purpose user abort; carries only the base fields.

```text
(base fields only)
```

## Where exceptions come from

| Source | Exception |
| ------ | --------- |
| Agent transport failure | `AgentCallError` |
| Invalid structured output after all attempts | `AgentParseError` |
| Failing/timed-out shell command | `ExecError` |
| Loop bound exhausted | `MaxIterationsExceeded` |
| Non-exhaustive `case` at runtime | `MatchError` |
| Division by zero | `ArithmeticError` |
| `raise` of a constructed or re-raised value | any concrete type |

An exception that reaches the top of the program uncaught terminates the
run; the host reports the exception's type, fields, and source location, and
records it in the trace ([Host environment](host-environment.md)).
