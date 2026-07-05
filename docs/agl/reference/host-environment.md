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
3. **Type checking** against the host's *capability catalog* — type errors
   abort here; warnings are collected.
4. **Param validation** — externally provided values are checked against the
   program's `param` declarations.
5. **Contract materialization** — every agent-call and `exec` site's output
   contract (codec, schema, format instructions) is built.
6. **Execution.**

A failure in steps 1–5 means **nothing executes**: no statement runs, no
agent is called, no shell command is spawned. Static checking stops at the
first error. Warnings (for example a non-exhaustive `case`, or a useless
`on_parse_error` on a `text` target) are reported on every path and never
prevent execution.

## Agents

**The program owns the set of named agents.** Every named agent a program
calls must be declared at the program root with an `agent` declaration
([Bindings and scope](bindings-and-scope.md)); a call to an undeclared name
is a static binding error. The host does *not* contribute names and there is
**no implicit fallback** that makes arbitrary names resolve. Two names need
no declaration:

- **The default agent** backs the contextual keyword `ask`.
- **`exec`** denotes the shell executor ([Shell execution](shell-execution.md)).

The host's role is to **supply a backing** — the actual agent that runs — for
each declared name. A declaration may also carry an optional *runner hint*, an
opaque static string the host may use to launch the agent; the host ignores
it if it has its own backing, and host configuration for a given name always
takes precedence over the source hint. The hint is never interpreted by the
language (no interpolation; see [Bindings and scope](bindings-and-scope.md)).

Because the program owns the names and the host owns the backings, two
mismatches are **host configuration errors**, reported before anything
executes:

- a backing supplied for a name the program never declares (*registered but
  undeclared*), and
- a declared agent for which the host provides neither a dedicated backing
  nor a default agent to fall back on (*declared but unbacked*).

An `ask` call requires a default agent to be configured, or it is a static
error. The names `ask` and `exec` can never be declared as agents.

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

Each registration declares which type kinds it supports, and the
type-checker validates every `format` option against this **capability
catalog** before execution — an unsupported codec/type combination is a
static error, not a runtime surprise.

## Params

Program parameters are declared with `param`
([Bindings and scope](bindings-and-scope.md)) and may be supplied by the
host as named external values at run start. Validation happens after type
checking and **before any statement executes**:

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
default is always used. Runtime-only values such as `unit`, agents, and
functions are not valid program param types because the executable always
includes external-decoder metadata for every declared param. A
[recursive](types.md#recursive-types) record or enum param decodes normally,
subject to the same finite-schema restriction as an agent output type or cast
target — see [Generics](generics.md#the-finite-schema-boundary).

## Host-configurable settings

### Engine keys

A program may declare any of the following fixed engine keys with `config`
([Bindings and scope](bindings-and-scope.md)):

| Key | AgL type | Portable default |
| --- | -------- | ---------------- |
| `log` | `bool` | `false` |
| `strict-json` | `bool` | `false` (lenient recovery) |
| `max-iters` | `int` | `5` |
| `runner` | `text` | host floor runner |
| `log-file` | `Option[text]` | `none` |
| `timeout` | `Option[text]` | `none` |

For an `Option[T]` key (`log-file`, `timeout`) a bare inner `T` value is
accepted and projected into `some(value)` automatically.

### Precedence chains

The two declaration forms have distinct precedence rules:

```
config X:   CLI --X  >  source (config X = e)  >  [<program>].X  >  [exec].X  >  engine default
param  Y:   CLI --Y  >  [<program>].Y           >  source default (param Y = e) >  required error
```

A bare `config KEY` (no value) contributes no source value and falls through
to the program-section / exec-section / engine-default layers.

### Config-file schema

`[exec]` holds global engine defaults with kebab field names (`strict-json`,
`max-iters`, `log-file`). A `[<program>]` top-level section — keyed by the
`program NAME` declaration or the `.agl` file stem — overrides both engine-key
values and param values for that program. A file stem that matches a reserved
host section name (e.g. `exec`, `loop`) is an error unless an explicit `program
NAME` declaration is present. Inline `-c` programs with no `program` declaration
have no config section.

### Effect-at-binding and start-resolved keys

The six engine keys are divided into two groups:

- **Effect-at-binding** (`strict-json`, `max-iters`, `timeout`): take effect
  from the point the `config` declaration executes. Expressions after the
  declaration see the updated setting; expressions before do not.
- **Start-resolved** (`runner`, `log`, `log-file`): resolved before the program
  runs. Declare them at the top so the agent factory and trace infrastructure
  see the chosen values before any expression evaluates.

### Error surface for `timeout`

- A bad `--timeout` or `[exec].timeout` value is caught before execution
  (exit 1 pre-execution error).
- A bad source `config timeout = "…"` value is a runtime-evaluated expression;
  a bad value raises a runtime error (exit 2).
- Source `config timeout` governs only the **shell-exec** timeout. The agent
  idle timeout is always start-resolved from the CLI or `[exec]` and cannot be
  changed mid-program.

### `--no-log-file` semantics

`--no-log-file` sets the in-program `config log-file` binding to `none`. It
does **not** suppress a trace configured elsewhere — a `[exec] log-file` path
or an auto path from `--log` still applies to the trace infrastructure. Use
`--no-log` to disable tracing entirely.

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

A conforming host traces execution so runs can be audited and debugged. The
trace records:

- run start and end (with success/failure);
- every agent call attempt (agent, attempt number, rendered prompt) and
  every parse result (raw output, normalized output, error summary);
- every `exec` invocation (command, exit code, duration, stdout, stderr,
  timeout flag);
- every `:=` mutation and every `print`;
- every raised exception.

Every exception value carries a `trace_id` field linking it to the
corresponding trace record. The normalized (recovered) JSON of a lenient
parse is traced alongside the raw output.

## Results and termination

A run ends in one of three ways:

1. **Success** — all statements executed; the host can observe the final
   root-scope bindings.
2. **Pre-execution failure** — a static error, param-validation error, or
   host configuration error; nothing was executed.
3. **Uncaught exception** — the program started and an exception reached the
   top. The host reports the exception's type name, fields, and the source
   location of the raise site.

## Static call inventory

Because contracts are materialized before execution, a host can present a
complete static inventory of a program's agent-call and `exec` sites — for
each: callee, target type, codec, schema presence, parse policy, and source
location — without running anything. This supports dry-run inspection of a
workflow's external interactions.
