# Host Environment

[← Index](index.md)

An AgL program does not run in a vacuum: a **host** embeds the language,
supplies the agents, provides external param values, executes shell commands, and
records the trace. This chapter specifies the contract between a program and
its host — what a program may assume, and which knobs are host-configurable.

## The execution pipeline

A conforming host processes a program in this order:

1. **Parse** — syntax errors abort here.
2. **Name resolution** — scope errors abort here.
3. **Static validation** against the host's *capability catalog*. Any type
   error, non-exhaustive case, or redundant case arm aborts here. Independent
   advisory diagnostics are collected separately.
4. **Param validation** — externally provided values are checked against the
   program's `param` declarations.
5. **Contract materialization** — every agent-call and `exec` site's output
   contract (codec, schema, format instructions) is built.
6. **Execution.**

A failure in steps 1–5 means **nothing executes**: no statement runs, no
agent is called, no shell command is spawned. Warnings (for example a useless
`on_parse_error` on a `text` target) are reported on every path and never
prevent execution.

## Agents

Each `ask` evaluates an `Agent` enum value that selects the backend command.
The value may be passed explicitly or supplied by the
`std/config::default-agent` setting. `AgentCommand` carries a command string;
the provider variants carry their provider-specific fields. The host dispatches
the selected value and does not contribute agent names or reconcile a registry.

Per dispatch, an agent receives the rendered prompt, the output contract
(format instructions plus derived JSON Schema, so schema-capable backends
can use native structured output), the attempt number, and — on corrective
retries — the previous invalid output with its validation errors
([Agent calls](agent-calls.md)). The agent returns raw text. Hosts must pass
the rendered prompt through verbatim, with no second template or
environment-variable expansion.

Transport failures (spawn failure, nonzero exit, timeout) surface as the
catchable `AgentCallError` with an enumerated `cause`; exit 0 with empty
output is a valid empty response ([Exceptions](exceptions.md)).

## Codecs

The built-in codecs are `text` and `json`. Hosts may register additional
codecs (selectable per call with `format`). Built-in names are reserved;
duplicate registrations are host configuration errors.

Each registration declares which type kinds it supports, and every
`format` option is validated against this **capability catalog** before
execution — an unsupported codec/type combination is a static error, not a
runtime surprise.

## Params

Entry-module parameters are declared with `param`
([Bindings and scope](bindings-and-scope.md)) and may be supplied by the
host as named external values at run start. An entry-module param declared as a
member of a named scope region ([Named scopes](scopes.md#parameters)) is
supplied under its full path spelling — `param Deploy::region` is named
`Deploy::region` by the host, e.g. `--Deploy::region` on the CLI. During the
M2 interim, non-entry param declarations are accepted and type-checked, but
every use is a static error because discovery and lowering omit them until M3.
In a config file the same key must be quoted, since `::` is not a legal bare
TOML key:

```toml
[demo]
"Deploy::region" = "prod"
```

Validation happens after type checking and **before any statement executes**:

- a required param (no default) for which no external value is provided,
- an external value supplied for a name that is not a declared param, and
- an external value that fails its declared type

are each **host invocation errors** — reported like static failures, not
catchable in-language. Optional params whose external value is absent have
their default expression evaluated at that point in declaration order before
execution begins.

`text` params take their external value verbatim. A param of any other type
is parsed from its JSON representation **strictly** (externally supplied
values are not chatty agent output, so no lenient recovery applies) and
validated against the declared type.

The declared type must be JSON-wire-serializable, including for a param whose
default is always used. Runtime-only values such as `unit` and functions are not valid program param
types because the executable always
includes external-decoder metadata for every declared param. A
[recursive](types.md#recursive-types) record or enum param decodes normally,
subject to the same finite-schema restriction as an agent output type or cast
target — see [Generics](generics.md#the-finite-schema-boundary).

## Host-configurable settings

### Engine settings

The standard-library module `std/config` exposes the following fixed engine
settings as mutable bindings ([Bindings and scope](bindings-and-scope.md)). Each
`builtin var` declaration may supply its portable default with a constant
initializer; the host uses that declared value only when it has no seed for the
key:

| Setting | AgL type | Portable default |
| --- | -------- | ---------------- |
| `log` | `bool` | `false` |
| `strict-json` | `bool` | `false` (lenient recovery) |
| `max-iters` | `int` | `0` (off) |
| `default-agent` | `Agent` | `AgentClaude("sonnet", "medium")` |
| `log-file` | `Option[text]` | `None` |
| `timeout` | `Option[text]` | `None` |

Import `std/config` and read or write a setting through a qualified target
(`std/config::max-iters`); writing zero disables that safety valve.
`default-agent` is a typed `Agent` value used by `ask` when its `agent` option
is omitted. The `Option[text]` settings (`log-file`, `timeout`) take a
`Some("…")` or `None` value.

### Precedence

`agm exec` resolves initial values as:

```
setting X:  source (std/config::X := e)  >  CLI --X  >  [<program>].X  >  [exec].X  >  declared default
param   Y:  CLI --Y                       >  [<program>].Y  >  source default (param Y = e) >  required error
```

The CLI flag and config-file layers supply a setting's **initial** value; a
source write to `std/config::X` overrides them from its program point onward.
A program that never writes a setting keeps the value chosen by the CLI/config
layers.

`agm repl` resolves engine settings as source writes > CLI > `[exec]` > declared
default. It has no per-param CLI options or per-program config table. REPL
parameters therefore require source defaults.

### Config-file schema

`[exec]` holds global engine defaults with kebab field names (`strict-json`,
`max-iters`, `log-file`). For `agm exec`, a `[<program>]` top-level section is
keyed by the `.agl` file stem and overrides both engine settings and param
values. Inline `-c` source has no per-program config section.

`agm repl` does not read `[<program>]` tables. They never override REPL engine
settings.

### Positional effect

The host applies each effective initial setting before execution. Thus a declared `log` or `log-file` default configures the trace service when
no CLI/config seed is supplied. Every setting takes effect
**positionally** thereafter: a write to `std/config::X` governs the statements
that follow it, in program order, and does not affect statements before it. A
completed write remains effective if a later expression fails. Writing `log` or
`log-file` updates the trace destination used by subsequent calls. Assigning
`Some(path)` to `log-file`
enables logging; a later `log := false` disables it while retaining the path.
Writing `strict-json`, `max-iters`, or `timeout` changes subsequent agent-output
parsing, unbounded loops, or `exec` calls, respectively. A write the engine
cannot accept — a negative `max-iters`, or a `timeout` whose text is not a
duration — raises the catchable `TypeError`
([Exceptions](exceptions.md#typeerror)) and leaves the setting unchanged.
Trace output is best-effort: a filesystem failure disables tracing for the rest
of the run without rolling back the assigned `log` or `log-file` value.

### Error surface for `timeout`

- A bad `--timeout`, `[<program>].timeout`, or `[exec].timeout` value is caught
  before execution (exit 1 pre-execution error).
- A bad duration in `std/config::timeout := Some("…")` is evaluated at runtime;
  a bad value raises the catchable `TypeError`
  ([Exceptions](exceptions.md#typeerror)), terminating the run (exit 2) when
  uncaught.
- A CLI, `[<program>]`, or `[exec]` timeout initially seeds both shell execution
  and agent idle timeout. A source write to the `timeout` setting changes only
  the **shell-exec** timeout; agent idle timeout cannot be changed mid-program.
- Reading `timeout` returns the exact `Option[text]` value assigned or supplied
  initially; duration parsing does not normalize its text.

### `--no-log-file` semantics

`--no-log-file` clears the initial `log-file` value. It does **not** suppress a
trace configured elsewhere — a `[exec] log-file` path or an auto path from
`--log` still applies. Use `--no-log` to disable tracing entirely.

### Other host-configurable defaults

| Setting | Portable default | Used when |
| ------- | ---------------- | --------- |
| Default parse policy | `abort` | call without `on_parse_error` |
| Default JSON parsing mode | lenient recovery | JSON-codec call without `strict_json` |
| Agent idle timeout | host-defined | every agent dispatch |

Source-level call options always override host defaults — in both
directions: `strict_json = false` forces lenient parsing even under a strict
host default.

## Tracing

While tracing is active, a conforming host writes one JSON object per line.
Every record has the envelope `ts`, `run_id`, and `kind`: `ts` is an
ISO-8601 local timestamp with an offset, and `run_id` distinguishes runs that
append to the same file. Positional source writes may enable or disable
tracing, so records outside the active interval (including run start or end)
can be absent.

Tracing records only observable boundaries:

- run start and end (with success/failure);
- stdout emitted by `print`;
- each agent request and response. A request records the fully composed prompt,
  selected agent and payload, attempt information, and output contract; a
  response records its full content or transport/cancellation outcome;
- every `exec` invocation (command, exit code, duration, stdout, stderr, and
  timeout flag);
- an exception only when it escapes the program uncaught.

Ordinary expression evaluation is not traced. Trace records and exception
values have no host-generated `trace_id`; a user-declared exception may still
have a field with that name.

## Results and termination

A run ends in one of three ways:

1. **Success** — all statements executed; the host can observe the final
   bindings, each scoped one under its full path spelling.
2. **Pre-execution failure** — a static error, param-validation error, or
   host configuration error; nothing was executed.
3. **Uncaught exception** — the program started and an exception reached the
   top. The host reports the exception's type name, fields, and the source
   location of the raise site. A field holding a value with a reference cycle
   ([Types](types.md#cycles)), or a value of a kind with no JSON
   representation, is reported as a placeholder marker, so reporting a failure
   never fails.

## Static call inventory

Because contracts are materialized before execution, a host can present a
complete static inventory of a program's agent-call and `exec` sites — for
each: callee, target type, codec, schema presence, parse policy, and source
location — without running anything. This supports dry-run inspection of a
workflow's external interactions.
