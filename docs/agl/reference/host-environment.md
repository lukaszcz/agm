# Host Environment

[← Index](index.md)

An AgL program does not run in a vacuum: a **host** embeds the language,
supplies the agents, provides external values for a selected program's
signature and module parameters, executes shell commands, and records the trace. This chapter
specifies the contract between a program and its host — what a program may
assume, and which knobs are host-configurable.

## The execution pipeline

A conforming host processes a program in this order:

1. **Parse** — syntax errors abort here.
2. **Name resolution** — scope errors abort here.
3. **Static validation** against the host's *capability catalog*. Any type
   error, non-exhaustive case, or redundant case arm aborts here. Independent
   advisory diagnostics are collected separately.
4. **Host-input validation** — externally provided values are checked against
   the selected `program def`'s own value parameters and every module
   parameter in its import closure.
5. **Contract materialization** — every agent-call and `exec` site's output
   contract (codec, schema, format instructions) is built.
6. **Execution.**

A failure in steps 1–5 means **nothing executes**: no statement runs, no
agent is called, no shell command is spawned. Warnings (for example a useless
`on-parse-error` on a `text` target) are reported on every path and never
prevent execution.

## Agents

Each `ask` evaluates an `Agent` member `RecordValue` that selects the backend command; an `Agent`-typed value is that member record, not an enum wrapper. The record may be passed explicitly or supplied by the `std/config::default-agent` setting. `AgentCommand` carries a command string; the provider member records carry their provider-specific fields. The host dispatches the selected member record and does not contribute agent names or reconcile a registry.

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

## Program arguments

A `program def`'s own value parameters ([Functions](functions.md#parameters))
are its external inputs. The host supplies them as named external values at
run start, resolved for the **selected** program only — a program reached
only through an import keeps its parameters as ordinary function arguments,
supplied by its caller, not by the host. Binding is ordinary call semantics:
the host binds only the parameters it has an external value for, and every
other parameter uses its own default, evaluated exactly as an omitted call
argument is — at call time, after every module's static bindings have
initialized, so a default may read module-level state. An exception raised by
a default therefore surfaces as an ordinary uncaught program exception, not a
host invocation error.

A positional-only or standard parameter accepts a positional CLI token; a
standard or named-only parameter accepts `--name value`, `--name=value`, or,
for `bool`, `Option[T]`, and `Optional[T]`, the negated form `--no-name`. An
`Optional[T]` parameter additionally reads the exact value `default` as `Default`;
other positive values become `Some`. A parameter given a
one-letter spelling by `@opt-short` also accepts `-n value` and `-nvalue`, and
groups with other one-letter flags — `-abc` — where only the last letter of a
group may take a value. A `program def`'s
parameter list defaults to the **named-only** zone, so a plain `name: text`
parameter is addressed only by `--name`; an `@arg-pos` parameter opens a
positional slot. A bare `--` ends option parsing, so a later
`--`-prefixed token is collected positionally instead of being read as a flag.
Supplying the same parameter twice (by any combination of position and name)
is a usage error, as is a flag naming no declared parameter.

Validation happens after type checking and **before any statement executes**;
it checks only the externally supplied values themselves, not a parameter's
default — a default's own evaluation is distinct and happens later, at the
entry call, as described above. Each parameter's effective value resolves as:

```
CLI token (--name / positional)  >  @opt-env variable  >  qualified config table  >  declared default
```

A **positional-only** parameter has no `--flag`, so a config-table entry
naming it can never reach the argument binder — it falls back to its declared
default and is reported with a distinct positional-only warning, not the
generic "not a declared program argument" one. A required parameter (no
default) for which no external value is provided is a **host invocation
error** — reported like a static failure, not catchable in-language, before
any statement executes.

`text` parameters take their external value verbatim. A parameter of any
other type reads its external text as one **strict JSON value or AgL value
syntax literal** (externally supplied values are not chatty agent output, so
no lenient recovery applies), validated against the declared type: a token
that parses as strict JSON is read as JSON; any other token is read as one
[value syntax](#value-syntax) literal instead. An `Agent`-typed value
([Agents](#agents) above) instead reads compact shorthand, then a JSON
object, then an `Agent` member constructor call, and otherwise falls back to
a verbatim command — see [host Agent syntax](../../commands/agl.md#host-agent-syntax).

A CLI token, an `@opt-env` variable's value, and a qualified config-table
string ([Config-file schema](#config-file-schema) below) must all be valid
Unicode. A value holding a surrogate code point is a **host invocation
error** naming the parameter, before any statement executes.

The process environment a host exposes through `std/env::environ` is
likewise required to be valid Unicode in both variable names and values: a
host that finds one it cannot decode fails before the program runs, naming
the variable, rather than letting the failure surface later at a print or
`exec` call. Unset or fix the offending variable, or exclude it with
`env -u NAME agm exec …`.

### Value syntax

A `text`-to-structured-type cast (`as`/`as?`, [Types](types.md#strict-parsing-in-text-and-json-casts))
accepts the same value-syntax literal alongside strict JSON, with no lenient
recovery either way; a cast to `Agent` accepts shorthand, a JSON object, or a
member constructor call, but never falls back to a verbatim command the way a
host `Agent` parameter does.

A value-syntax literal is a data-only subset of AgL's own expression syntax:
an integer, decimal, `true`/`false`, a quoted text literal, an `[item, ...]`
array, a `{key: value, ...}` dict of quoted-or-bare keys, or a constructor
reference/call — `Name`, `Name()`, or `Name(arg, ..., field = value, ...)`. A
constructor's arguments bind against its declared fields by the same
[zone rules](functions.md#parameters) an ordinary call uses, except that a
named-only field always takes an explicit `field = value`: value syntax has
no variables, so the bare-name shorthand an ordinary call allows for a
named-only argument does not apply. A bare or called name matches a record's
or enum member's declared name or its own
[`@name`](attributes.md#name-and-json-name) alias. An optional qualifier is
the single name immediately enclosing the record's declaration: its innermost
scope (`Geo::Point(...)`) or, for an inline enum member, its enum
(`Shape::Square(...)`). A top-level record takes no qualifier. Where an enum
type is expected, the enum's own name also qualifies any of its members. A
duplicate dict key is an error. Nesting is
unrestricted — a constructor argument, array item, or dict value may itself
be any value-syntax literal, including another constructor call. `null` and
a heterogeneous (mixed-type) array or dict are legal only in a `json`-typed
slot, read as plain data with no constructor calls, since a `json` value has
no declared type to resolve one against.

A constructor argument for a field with a declared
[default](types.md#record-types) may be omitted; the field then fills from
that default, exactly as an omitted argument does in AgL source. A
constructor whose every field has a default may be spelled bare (`Name`) or
with an empty call (`Name()`), constructing it with every default. For

```agl
record Retry
  count: int = 3

record Config
  retry: Retry = Retry
  verbose: bool = false
```

`Retry`, `Retry()`, `Config(verbose = true)`, and the bare `Config` all
construct a value with `retry` filled from its default. The same rule governs
strict JSON: an object may omit a key for a defaulted field, and the derived
JSON Schema ([Agent calls](agent-calls.md#derived-json-schema)) excludes such
a field from `"required"`.

A parameter annotated [`path`](types.md#type-aliases) — directly, as
`Option[path]` or `Optional[path]`, or through an alias — takes its value exactly as the
corresponding `text` parameter does. A host presents that value as a
filesystem location: its value placeholder defaults to `PATH`, and a host
offering completion completes it from the filesystem.

The declared type must be JSON-wire-serializable, including for a parameter
whose default is always used. Runtime-only values such as `unit` and
functions are not valid program-argument types, whether or not the host ever
supplies a value for the parameter. A [recursive](types.md#recursive-types)
record or enum parameter decodes normally, subject to the same finite-schema
restriction as an agent output type or cast target — see [Generics](generics.md#the-finite-schema-boundary).

A rendered record or enum value (see
[Uniform rendering rules](strings-and-interpolation.md#uniform-rendering-rules))
parses back through `parse`/`as` to an equal value when every field,
transitively, holds value-syntax data (no function values) and each nested
record's or enum member's declaration sits at top level or directly inside
one scope or enum — so its rendered qualifier is the single name value syntax
accepts.

### Presentation attributes

`@opt-name`, `@opt-short`, `@opt-env`, `@opt-metavar`, `@opt-hidden`, and
`@doc` shape how a value parameter is addressed and described on the command
line; see [Attributes](attributes.md#host-parameter-attributes).

### Help

A host is expected to describe the selected program on request. What it shows
is drawn entirely from the program's own declaration: the `program def`'s
`@doc` prose describes it, each positional slot is named after its parameter,
and each name-addressed parameter is listed with its long flag, its
`@opt-short` spelling if it has one, its value placeholder, and its own `@doc`
prose.

## Host-configurable settings

### Module parameters

A static `let` or `var` marked `@param` is a **module parameter**. Its host
surface belongs to every selected program whose transitive import closure
contains its declaring module. The selected program's own module comes first;
the imported modules follow. The initializer remains the fallback: a supplied
value is bound during static initialization instead of evaluating that
initializer. See [Attributes](attributes.md#module-parameters) for the
declaration form and its restrictions.

The parameter's **external name** is its `@opt-name`, or its declared name
when it has none. That name is used in flags, config leaves, help, and
completion. A root parameter in `A/logging`, such as `verbose`, has a bare
`--verbose` spelling when it resolves, plus dotted qualified spellings such as
`--logging.verbose` and `--A.logging.verbose`. A parameter in `scope debug`
adds the scope path: `--logging.debug.trace` and
`--A.logging.debug.trace`. Qualified spellings are dotted; source declaration
paths use `::` instead. `@opt-short("v")` adds `-v` on the command line.

`bool`, `Option[T]`, and `Optional[T]` parameters also accept their normal `--no-...` form;
other parameter types take a value under the same type-directed rules as
[program arguments](#program-arguments). A bare name is resolved in this
order: host flags and engine-setting leaves, the selected program's own value
parameters, parameters in its own module, then parameters in imported
modules. A later level retains qualified spellings. Several parameters at the
same level make a spelling ambiguous only when it is used; the diagnostic
names the candidate declaration paths. The same resolution applies to a
short spelling.

For an unset module parameter, the value is chosen in this order:

```
CLI flag  >  @opt-env variable  >  program route  >  @config  >  module route  >  initializer
```

The **program route** is the selected program's table. It addresses a parameter
through any spelling that resolves for it: the bare external name, or one of
the dotted qualified spellings, written as a quoted key. A parameter whose bare
name a nearer declaration claims — an engine setting, the signature, a nearer
module — is therefore still configurable for one program through a qualified
leaf. Setting two spellings of one parameter in a single layer is an error, and
a spelling several parameters claim is rejected naming them, exactly as the
matching flag would be. The program route wins over the selected program's own
[`@config`](attributes.md#config) entries, which in turn win over the
**module route**, even when the latter appears in a more-specific TOML layer.
A module route addresses the binding's declaring module and scope, so it
remains available too.

For example, this program imports the `A/logging` module:

<!-- agl-check: fragment -->
```agl
import A/logging

program def main() -> unit =
  print A/logging::verbose
```

Its configuration can use the module tables below. The selected program's
table overrides a resolving module-table leaf.

```toml
[A.logging]
verbose = true

[A.logging.debug]
trace = true

[tool.main]
verbose = false
"A.logging.debug.trace" = false
```

`["A/logging".debug]` is the exact quoted-module spelling of the scoped
module table; use it instead of `[A.logging.debug]` when an exact module route
is needed. Module and scope paths are dotted in TOML and flags, never `::`.
Each route accepts every unambiguous module suffix, shortest first. A config
leaf uses the external name, not the declared binding name.

Help lists the selected program's own options first and then one section for
each closure module with a visible parameter, titled by its module path. Each
visible module parameter is shown under its shortest resolving spelling.
`@opt-hidden` omits the parameter from help and completion without making it
unavailable to config, environment, or an explicit resolving flag.

`agm check` validates module parameter declarations but supplies no host
values and evaluates nothing. The REPL has no program route or parameter
flags: when it first loads a module, its parameters read that module's route
only. An inline `agm exec -c` entry has no module route for its own parameters,
so they resolve as CLI flag > `@opt-env` variable > initializer; its
file-backed imports keep their module routes.

### Engine settings

The standard-library module `std/config` exposes the following fixed engine
settings as mutable bindings ([Bindings and scope](bindings-and-scope.md)). Each
`builtin var` declaration may supply its portable default with a constant
initializer; the host uses that declared value only when it has no seed for the
key:

| Setting | AgL type | Portable default |
| --- | -------- | ---------------- |
| `trace` | `bool` | `false` |
| `strict-json` | `bool` | `false` (lenient recovery) |
| `default-agent` | `Agent` | `AgentClaude("sonnet", "medium")` |
| `trace-file` | `Option[path]` | `None` |
| `timeout` | `Option[text]` | `None` |

Import `std/config` and read or write a setting through a qualified target
(`std/config::strict-json`).
`default-agent` is a typed `Agent` value — its selected member `RecordValue` at runtime — used by `ask` when its `agent` option is omitted. Host CLI and TOML values read the same [host Agent syntax](../../commands/agl.md#host-agent-syntax) as an `Agent`-typed parameter: compact shorthand, then a JSON object, then an `Agent` member constructor call, and otherwise a verbatim command. The optional settings (`trace-file`, `timeout`) take a `Some("…")` or `None` value.

### Precedence

`agm exec` resolves initial values as:

```
setting X:    source (std/config::X := e)  >  CLI --X  >  qualified program table  >  @config  >  [exec].X  >  declared default
argument Y:   CLI token (--Y / positional) >  qualified program table             >  declared default > required error
```

The CLI flag and config-file layers, and the selected program's own
[`@config`](attributes.md#config) entries, supply a setting's **initial**
value; a source write to `std/config::X` overrides them from its program
point onward. A program that never writes a setting keeps the value chosen by
those layers.

`agm repl` resolves engine settings as source writes > CLI > `[exec]` > declared
default. It has no entry program: a `program def` declared at the prompt is an
ordinary function, called with its own arguments like any other declaration
there. Its named-only default zone still applies, so a plain parameter is
supplied as `main(x = 5)`; the positional form `main(5)` is rejected as a
positional argument in a named-only position.

### Config-file schema

`[exec]` holds global engine defaults with kebab field names (`strict-json`,
`trace-file`). A qualified table uses a module suffix (the entry
file's stem, or a package's declared route) and the selected program's own
declaration name — `[prog.main]` for a program named `main` in a file whose
stem or route is `prog`. The same table supplies both that program's engine-key
overrides and its own value parameters. A longer suffix, including an exact
quoted module route, disambiguates same-leaf modules. Inline `-c` value
parameters are CLI-only. The selected program's own
[`@config`](attributes.md#config) entries rank between this qualified table
and `[exec]` for an engine setting, and between this table and a module route
for a module parameter; `@config` is a source declaration, so it never
appears in a config file.

A program a package registers as a CLI command is addressed by that command
path too: `agm dev review`, registered for `review-tools/review::main`, reads
`[dev.review]`, whose segments are the whole table path. It is one more
spelling of the same address, so it applies however the program is run — as
the command, by installed reference, or by file path — and setting one leaf
through two spellings in the same config layer is an error, exactly as two
module suffixes are.

A config-table string value must be valid Unicode, the same requirement a
CLI token or `@opt-env` variable is held to above; a value holding a
surrogate code point is a host invocation error.

### Positional effect

The host applies each effective initial setting before execution. Thus a declared `trace` or `trace-file` default configures the trace service when
no CLI/config seed is supplied. Every setting takes effect
**positionally** thereafter: a write to `std/config::X` governs the statements
that follow it, in program order, and does not affect statements before it. A
completed write remains effective if a later expression fails. Writing `trace`
or `trace-file` updates the trace destination used by subsequent calls.
Assigning `Some(path)` to `trace-file`
enables tracing; a later `trace := false` disables it while retaining the path.
Writing `strict-json` or `timeout` changes subsequent agent-output parsing or
`exec` calls, respectively. A write the engine cannot accept — a `timeout`
whose text is not a duration — raises the catchable `TypeError`
([Exceptions](exceptions.md#typeerror)) and leaves the setting unchanged.
Trace output is best-effort: a filesystem failure disables tracing for the rest
of the run without rolling back the assigned `trace` or `trace-file` value.

### Error surface for `timeout`

- A bad `--timeout`, qualified program-table timeout, or `[exec].timeout` value is caught
  before execution (exit 1 pre-execution error).
- A bad duration in `std/config::timeout := Some("…")` is evaluated at runtime;
  a bad value raises the catchable `TypeError`
  ([Exceptions](exceptions.md#typeerror)), terminating the run (exit 2) when
  uncaught.
- A CLI, qualified program table, or `[exec]` timeout initially seeds both shell execution
  and agent idle timeout. A source write to the `timeout` setting changes only
  the **shell-exec** timeout; agent idle timeout cannot be changed mid-program.
- Reading `timeout` returns the exact `Option[text]` value assigned or supplied
  initially; duration parsing does not normalize its text.

### `--no-trace-file` semantics

`--no-trace-file` clears the initial `trace-file` value. It does **not**
suppress a trace configured elsewhere — a `[exec] trace-file` path or an auto
path from `--trace` still applies. Use `--no-trace` to disable tracing
entirely.

### Other host-configurable defaults

| Setting | Portable default | Used when |
| ------- | ---------------- | --------- |
| Default parse policy | `abort` | call without `on-parse-error` |
| Default JSON parsing mode | lenient recovery | JSON-codec call without `strict-json` |
| Agent idle timeout | host-defined | every agent dispatch |

Source-level call options always override host defaults — in both
directions: `strict-json = false` forces lenient parsing even under a strict
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
- an exception only when it escapes the program uncaught;
- records a companion emits through `runtime.trace` ([FFI](ffi.md#trace-hook)).

Ordinary expression evaluation is not traced. Trace records and exception
values have no host-generated `trace-id`; a user-declared exception may still
have a field with that name.

## Results and termination

A run ends in one of three ways:

1. **Success** — all statements executed; the host can observe the final
   bindings, each scoped one under its full path spelling.
2. **Pre-execution failure** — a static error, program-argument validation error, or
   host configuration error; nothing was executed.
3. **Uncaught exception** — the program started and an exception reached the
   top. The host reports the exception's type name, its fields keyed by their
   effective JSON name, and the source location of the raise site. A field
   holding a value with a reference cycle ([Types](types.md#cycles)), or a
   value of a kind with no JSON representation, is reported as a placeholder
   marker, so reporting a failure never fails.

## Static call inventory

Because contracts are materialized before execution, a host can present a
complete static inventory of a program's agent-call and `exec` sites — for
each: callee, target type, codec, schema presence, parse policy, and source
location — without running anything. This supports dry-run inspection of a
workflow's external interactions.
