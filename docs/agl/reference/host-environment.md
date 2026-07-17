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

### Engine settings

The standard-library module `std.config` exposes the following fixed engine
settings as mutable bindings ([Bindings and scope](bindings-and-scope.md)):

| Setting | AgL type | Portable default |
| --- | -------- | ---------------- |
| `log` | `bool` | `false` |
| `strict-json` | `bool` | `false` (lenient recovery) |
| `max-iters` | `int` | `0` (off) |
| `runner` | `text` | host floor runner |
| `log-file` | `Option[text]` | `None` |
| `timeout` | `Option[text]` | `None` |

Import `std.config` and read or write a setting through a qualified target
(`std.config::max-iters`); writing zero disables that safety valve. The
`Option[text]` settings (`log-file`, `timeout`) take a `Some("…")` or `None` value.

### Precedence

A setting's effective value is resolved as:

```
setting X:  source (std.config::X := e)  >  CLI --X  >  [<program>].X  >  [exec].X  >  engine default
param   Y:  CLI --Y                       >  [<program>].Y  >  source default (param Y = e) >  required error
```

The CLI flag and the config-file layers supply the setting's **initial** value;
a source write to `std.config::X` overrides them from its program point onward. A
program that never writes a setting keeps the value chosen by the CLI/config
layers.

### Config-file schema

`[exec]` holds global engine defaults with kebab field names (`strict-json`,
`max-iters`, `log-file`). A `[<program>]` top-level section — keyed by the
`program NAME` declaration or the `.agl` file stem — overrides both engine
settings and param values for that program. A file stem that matches a reserved
host section name (e.g. `exec`, `loop`) is an error unless an explicit `program
NAME` declaration is present. Inline `-c` programs with no `program` declaration
have no config section.

### Positional effect

Every setting takes effect **positionally**: a write to `std.config::X` governs
the statements that follow it, in program order, and does not affect statements
before it. A completed write remains effective if a later expression fails.
Writing `runner`, `log`, or `log-file` repoints the default agent and the trace
destination used by subsequent calls. Assigning `Some(path)` to `log-file`
enables logging; a later `log := false` disables it while retaining the path.
Writing `strict-json`, `max-iters`, or `timeout` changes subsequent agent-output
parsing, unbounded loops, or `exec` calls, respectively.
A host that rejects a `runner` reconfiguration leaves the setting's prior value
in place. Trace output is best-effort: a filesystem failure disables tracing for
the rest of the run without rolling back the assigned `log` or `log-file` value.

### Error surface for `timeout`

- A bad `--timeout`, `[<program>].timeout`, or `[exec].timeout` value is caught
  before execution (exit 1 pre-execution error).
- A bad duration in `std.config::timeout := Some("…")` is evaluated at runtime;
  a bad value raises a runtime error (exit 2).
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

While tracing is active, a conforming host records execution so runs can be
audited and debugged. Positional source writes may enable or disable tracing,
so records outside the active interval (including run start or end) can be
absent. The trace can contain:

- run start and end (with success/failure);
- every agent call attempt (agent, attempt number, rendered prompt) and
  every parse result (raw output, normalized output, error summary);
- every `exec` invocation (command, exit code, duration, stdout, stderr,
  timeout flag);
- every `:=` mutation, including engine-setting writes, and every `print`;
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
