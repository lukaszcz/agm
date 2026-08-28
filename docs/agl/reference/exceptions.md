# Exceptions

[← Index](index.md)

Runtime failures in AgL are **exceptions** — typed values that propagate up
through expressions, blocks, and calls until caught by a `try`/`catch` or,
uncaught, terminate the program. There are no sentinel return values.

## The exception model

Every exception is a value of a concrete built-in or user-declared exception
type. Every exception type extends the abstract base type `Exception`, which
declares one field present on every exception:

```text
message: text     # human-readable description
```

`Exception` itself is **not constructible** — `raise Exception(…)` is a
static error. It exists for typing: a wildcard catch binds its variable as
`Exception`, so only `message` and whole-value interpolation (`%{e}`) are
available there. Accessing a subtype field such as `e.raw` requires catching
the concrete type.

Programs may declare concrete exception types:

```agl
exception DeployError extends Exception
  service: text
  exit_code: int
```

An exception extends exactly one base exception type. Constructor fields include
the inherited fields first, followed by fields declared on the subtype.
`trace_id` is not reserved: a user-declared exception may use it as one of its
own fields. Exception fields do not accept the `var` marker and cannot be
reassigned. `builtin exception` is the standard-library form for host-recognized
exception types; the name, base, and fields must match the recognized shape
exactly.

### Methods

An exception type may declare methods. A method declared on an exception is
available on every subtype in its `extends` chain:

```agl
exception Problem extends Exception
  code: int

exception DetailedProblem extends Problem
  detail: text

def Problem::label(self) -> text = "problem %{self.code}"
def DetailedProblem::detail_label(self) -> text = self.label() + ": %{self.detail}"
program def main() -> unit =
  let problem = DetailedProblem(message = "failed", code = 7, detail = "network")
  let _ = print(problem.label())
  let _ = print(problem.detail_label())
```

A method name is declared only once in an exception hierarchy. A subtype cannot
use the name of a method declared by one of its base types, and a base type
cannot use a name declared by one of its descendants.

The built-in `Exception` type declares no methods. A wildcard `catch _ as e
=>` binds `e` as `Exception`, so no method is available through that binding.

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
  causes: array[ValidationError]   # legal: guarded by array[...]
```

<!-- agl-check: error -->
```agl
exception Broken extends Exception
  child: Broken   # rejected: uninhabitable, no array/dict guard
```

For a required exception field, recursion must be guarded by a field type
that itself has a finite base value, such as `array`/`dict` (empty collection)
or an enum/option-style type with a base-case member. The same inhabitation
rule also covers the `extends` chain itself: an `extends` cycle (two exceptions
each extending the other) is rejected as uninhabitable for the same reason a
field cycle is.

Exception values support field access (`e.raw`), equality, and rendering.
In interpolation and `print` an exception renders in **AgL record form**,
including all fields (`message` and any type-specific fields) in declaration
order — for example:

```
CastError(message = "cannot parse \"x\" as int", source_type = "text", target_type = "int", raw = "x")
```

See [Strings and interpolation](strings-and-interpolation.md) for the uniform
rendering rules. Use `e as json` to obtain the JSON object of the exception's
fields; use `e as text` to obtain the same AgL-form string.

## `try` / `catch`

`try … catch …` is an **expression** — its type is the unified type of the
body and all catch handler bodies (with `int → decimal` widening). A
`try`/`catch` in a value position binds a typed result:

```ebnf
try_expr          ::= "try" try_body catch_clause+
try_body          ::= suite | (marked_item ";")* try_tail
try_tail          ::= or_expr | inline_assign | try_letvar_decl | raise_expr
                    | return_expr | if_expr | case_expr | loop
try_letvar_decl   ::= "let" pattern type_ann? "=" try_value
                    | "var" decl_head type_ann? "=" try_value
try_value         ::= or_expr | raise_expr | return_expr | if_expr | case_expr | loop
catch_clause      ::= "catch" catch_pattern "=>" branch_body
catch_pattern ::= name ("as" name)?
                | "_" ("as" name)?
branch_body   ::= suite | closed_item
```

`branch_body` is the same body form an `if` or `case` branch takes — a suite
or a single item. Because `catch` marks where a `try` body ends, an inline
`try` body is a full `;` sequence, binders and `assign_target := or_expr` assignments included. Its final
item may be a `let` or `var`; then the try body has type `unit` unless the
initializer exits, in which case it has bottom type. See
[Inline bodies](grammar.md#inline-bodies).

<!-- agl-check: fragment -->
```agl
try
  let review: Review = reviewer.ask(
    "Review %{artifact}",
    on_parse_error = Retry(n = 2)
  )
  print "reviewed: %{review}"
catch AgentParseError as e =>
  let report = critic.ask("Explain invalid output:\n%{e.raw}")
  raise e
catch _ as e =>
  print "unexpected: %{e.message}"
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

A `return` inside a `try` body or a catch body is not an exception and is never
caught by `catch`; it unwinds to the nearest enclosing function.

Catch patterns:

- `catch SomeError` / `catch SomeError as e` — matches exactly the named
  exception type, whether built-in or user-declared. It does not match that
  type's subtypes. `SomeError` names whichever declaration of that name is in
  scope where the `catch` clause is written; in the REPL, a name that is
  later redeclared keeps naming its original declaration in a `catch` clause
  written before the redeclaration, while a `catch` clause written afterward
  names the new one.
- `catch _` / `catch _ as e` — matches anything; `e` has type `Exception`.
- `catch Exception as e` — equivalent to `catch _ as e`.

There is no `finally`.

## `raise`

```ebnf
raise_expr ::= "raise" or_expr
```

The operand must be an exception value (statically checked). It is an
`or_expr`, so an open form such as `if`, `case`, `try`, a loop, another
`raise`, or a lambda must be parenthesized. `raise` **diverges** — it never
yields a value. Its type is the bottom type, assignable to any expected type:

<!-- agl-check: fragment -->
```agl
let x: int = if condition => 1 else => raise Abort(message = "!")
raise Abort(message = "Cannot continue without repository access.")
```

An exception's own fields are **standard by default** and may be supplied
positionally or by name. Zone markers (`/`, `*`, `@pos`, `@std`, `@named`)
constrain an exception's own fields. The inherited `message` field is
named-only, so it is supplied by name when constructing an exception with
fields:

```agl
exception DeployError extends Exception
  service: text
  exit_code: int

program def main() -> unit =
  let _ = raise DeployError("api", 1, message = "deployment failed")
```

Any concrete built-in exception type is constructible with named arguments
for its fields. The same construction rule applies to user-declared exception
types. `Abort` is the conventional type for user-initiated failures.

Exception values support the
[record update](expressions.md#record-update) operator. Field names are
checked against the binding's static exception type, but the copy preserves
the value's **concrete** exception type — updating through a
`catch Exception as e` binding and re-raising keeps the original exception
type catchable:

<!-- agl-check: fragment -->
```agl
catch Exception as e =>
  raise (e with message = "while deploying: %{e.message}")
```

## Standard library exceptions

### `UnwrapError`

`std/option` raises `UnwrapError` when `Option[T].unwrap()` is called on
`None`; `std/result` raises the same exception when `Result[T, E].unwrap()` is
called on `Err`. It carries only the inherited `message` field.

### `FsError`

`std/fs` raises `FsError` for a failed filesystem operation. In addition to
`message`, it carries `path: text` and `operation: text`.

## Built-in exception catalog

Field lists below are in addition to the base `message`.

### `AgentCallError`

An agent **transport** failure: the agent could not run. Not eligible for
`on_parse_error` retries ([Agent calls](agent-calls.md)).

```text
agent: Agent      # the selected backend
cause: text       # "spawn_failure" | "nonzero_exit" | "timeout" | "interpolation_failure" | "protocol_failure" | "invalid_agent"
metadata: json    # host details: exit code, stderr tail, elapsed seconds
```

### `AgentParseError`

Structured agent output failed parsing or validation after all attempts
allowed by the parse policy.

```text
agent: Agent            # selected backend
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
or unit form** of `exec` ([Shell execution](shell-execution.md)). The
structured form raises it on spawn failure or timeout, but represents a
nonzero exit as `ExecResult` data.

```text
command: text     # the rendered command
exit_code: int    # -1 for spawn failure or timeout without an exit status
stdout: text
stderr: text
timed_out: bool
```

### `SessionError`

A catchable session lifecycle or capability failure. It covers opening or
obtaining the default session, use of a closed or unknown session, an
unsupported transport or lifecycle operation, and a host failure while running
`compact`, `reset`, `fork`, `stats`, `set-name`, or `close`. The backend
capabilities for each agent/transport combination are listed in
[Sessions](agent-calls.md#sessions).

```text
operation: text   # the host operation that failed, e.g. "open", "default",
                  # "ask", "compact", "reset", "fork", "stats",
                  # "set-name", or "close"
```

A prompt transport failure from `Session::ask` is instead `AgentCallError`;
output received from a prompt that fails its requested contract is
`AgentParseError`.

### `ExternError`

An `extern def` call failed: the companion Python callable raised, or returned
a Python value that has no AgL boundary representation
([Python FFI](ffi.md)).

```text
function: text       # the extern's declared name
python_type: text    # the raising Python exception's class name; empty for
                      # a return-value contract violation
```

### `MaxIterationsExceeded`

A **bounded** loop (`[n]`, with `n ≥ 1`) exhausted its bound before an exit
condition triggered ([Control flow](control-flow.md)). An unbounded loop (no
`[n]`) raises this only when the host `max-iters` safety valve is active and its
cap is reached (see [Control flow](control-flow.md)); with the valve off, an
unbounded loop never raises it.

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

An explicitly raised match-related failure. Source `case` expressions are
required to be exhaustive and never raise `MatchError` implicitly
([Pattern matching](pattern-matching.md)). Like any concrete exception,
`MatchError` may be constructed, raised, and caught by user code.

```text
scrutinee_type: text   # type name of the rejected value
scrutinee: json        # structural JSON encoding of the rejected value
```

### `ArithmeticError`

Raised by division by zero.

```text
operation: text    # the operator, e.g. "/"
```

### `TypeError`

Raised by an engine-setting write the host cannot accept — a negative
`max-iters`, or a `timeout` whose text is not a duration ([Host
environment](host-environment.md#engine-settings)).

```text
(base fields only)
```

### `IndexError`

Raised by out-of-range array or text indexing, indexed array assignment, or
`array::index-of`/`text::index-of` when the searched value is absent.

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

`UndefinedVariableError` and `ImmutableBindingError` are prevented statically
in normal programs — reads of undefined names and `:=` on immutable bindings
are both static errors — but the types exist, are catchable, and may be
constructed and raised explicitly. So is a type error, which is why
`TypeError` reaches a running program only from an engine-setting write whose
value the host rejects.

### `CastError`

A fallible `as` cast failed at runtime: the source value did not conform to
the target type. This includes an enum-to-member identity downcast when the
value has a different member identity.

```text
source_type: text   # name of the source type, e.g. "json"
target_type: text   # name of the target type, e.g. "int"
raw: text           # text representation of the value that failed to convert
```

`CastError` is raised by `as` casts that are fallible (see
[Types](types.md#casts-and-convertibility)). The `as?` form never raises —
it reports whether the cast would succeed as a `bool`.

### `JsonParseError`

A `std/json` parsing function received text that is not a well-formed JSON
document ([Modules](modules.md#stdjson)).

```text
raw: text   # the input text that failed to parse
```

### `TomlParseError`

A `std/toml` parsing function received text that is not a well-formed TOML
document ([Modules](modules.md#stdtoml)).

```text
raw: text   # the input text that failed to parse
```

### `TomlRenderError`

`std/toml::render` received a JSON value that TOML cannot represent: a
non-object root, a value containing `null`, an integer outside TOML's signed
64-bit range, or a signaling/payload `decimal` NaN. It carries only the base
fields.

```text
(base fields only)
```

### `RegexError`

A `std/regex` operation received a pattern that Python `re` cannot compile
([Modules](modules.md#stdregex)).

```text
pattern: text   # the pattern that failed to compile
```

### `RangeError`

Raised when `int.pow` receives a negative exponent, `decimal.sqrt` receives a
negative receiver, `decimal.pow` receives a negative exponent on a zero base,
or a range `for` step (`by k`) evaluates to a non-positive `int` (`k ≤ 0`) at
loop entry. Carries only the base fields. It is catchable.

```text
(base fields only)
```

### `Abort`

The general-purpose user abort; carries only the base fields.

```text
(base fields only)
```

### `CyclicValueError`

Raised when rendering (`print`, `render`, string interpolation, REPL echo),
`as text`, or `as json` encounters a genuine reference cycle, including one
closed through a record or enum-member `var` field. A cyclic array, dictionary,
or mutable record can be passed to an `extern def`; this error arises if its
companion calls Python `repr()` on the corresponding view. Carries only the
base fields. See [Cycles](types.md#cycles) for how a cycle arises, which
operations raise this and which tolerate a cycle instead (`as?`, `copy`), and
how equality and tracing treat one.

```text
(base fields only)
```

## Where exceptions come from

| Source | Exception |
| ------ | --------- |
| Out-of-range array/text index access, array indexed assignment, or an absent `array::index-of`/`text::index-of` search | `IndexError` |
| Missing dictionary key access or assignment | `KeyError` |
| Agent transport failure | `AgentCallError` |
| Invalid structured output after all attempts | `AgentParseError` |
| Failing shell command (parsed or unit form) | `ExecError` |
| Timed-out shell command (any exec form) | `ExecError` |
| Spawn failure (either exec form) | `ExecError` |
| Session open/default failure; unsupported transport or lifecycle capability; closed or unknown session; or failed `compact`, `reset`, `fork`, `stats`, `set-name`, or `close` | `SessionError` (`operation` identifies the failed host operation) |
| Session prompt transport failure | `AgentCallError` |
| Extern (Python FFI) companion raised, or its return value violated the contract | `ExternError` |
| Loop bound exhausted | `MaxIterationsExceeded` |
| Negative `int.pow` exponent, negative `decimal.sqrt` receiver, negative `decimal.pow` exponent on a zero base, or non-positive range `for` step (`by k` with `k ≤ 0`) | `RangeError` |
| Call-depth limit exceeded | `RecursionError` |
| Explicit `raise MatchError(...)` | `MatchError` |
| Division by zero | `ArithmeticError` |
| Engine-setting write the host rejects (negative `max-iters`, unparseable `timeout`) | `TypeError` |
| Fallible `as` cast — source does not conform to target type | `CastError` |
| `std/json` parsing — input is not well-formed JSON | `JsonParseError` |
| `std/toml` parsing — input is not well-formed TOML | `TomlParseError` |
| `std/toml` rendering — root is not an object, a value is `null`, an integer is outside signed 64-bit range, or a `decimal` NaN is signaling/payload | `TomlRenderError` |
| `std/regex` pattern compilation — Python `re` rejects the pattern | `RegexError` |
| Rendering, `as text`, or `as json` encounters a reference cycle, including a record-closed cycle; or an extern companion `repr()`s the corresponding cyclic view | `CyclicValueError` |
| `raise` of a constructed or re-raised value | any concrete type |

An exception that reaches the top of the program uncaught terminates the
run; the host reports the exception's type, fields, and source location, and
records it in the trace ([Host environment](host-environment.md)).
