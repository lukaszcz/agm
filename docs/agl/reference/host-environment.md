# Host Environment

[← Index](index.md)

An AgL program does not run in a vacuum: a **host** embeds the language,
supplies the agents, provides input values, executes shell commands, and
records the trace. This chapter specifies the contract between a program and
its host — what a program may assume, and which knobs are host-configurable.

## The execution pipeline

A conforming host processes a program in this order:

1. **Parse** — syntax errors abort here.
2. **Name resolution** — scope errors abort here.
3. **Type checking** against the host's *capability catalog* — type errors
   abort here; warnings are collected.
4. **Input validation** — provided inputs are checked against the program's
   `input` declarations.
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

- **The default agent** backs the contextual keyword `prompt`.
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

A `prompt` call requires a default agent to be configured, or it is a static
error. The names `prompt` and `exec` can never be declared as agents.

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

## Codecs and renderers

The built-in codecs are `text` and `json`; the built-in renderers are
`default`, `raw`, `json`, and `bullets`. Hosts may register additional
codecs (selectable per call with `format:`) and additional renderers
(usable as `${x as name}`). Built-in names are reserved; duplicate
registrations are host configuration errors.

Each registration declares which type kinds it supports, and the
type-checker validates every `format` option and every explicit renderer
against this **capability catalog** before execution — an unsupported
codec/type or renderer/type combination is a static error, not a runtime
surprise. A registered renderer that declares no kind restriction accepts
every type, like the built-ins.

## Inputs

Host inputs are declared in the program with `input`
([Bindings and scope](bindings-and-scope.md)) and provided by the host as
named values at run start. Validation happens after type checking and
**before any statement executes**:

- a declared input that was not provided,
- a provided input that was not declared, and
- a value that fails its declared type

are each **host invocation errors** — reported like static failures, not
catchable in-language. `text` inputs are taken verbatim; every other type is
parsed from its JSON representation **strictly** (host-supplied values are
not chatty agent output, so no lenient recovery applies) and validated
against the declared type.

## Host-configurable defaults

| Setting | Portable default | Used when |
| ------- | ---------------- | --------- |
| Default loop bound | `5` | `do` without an explicit `[N]` |
| Default parse policy | `abort` | call without `on_parse_error` |
| Default JSON parsing mode | lenient recovery | JSON-codec call without `strict_json` |
| Shell `exec` timeout | none | every `exec` call |
| Agent idle timeout | host-defined | every agent dispatch |

Source-level call options always override host defaults — in both
directions: `strict_json: false` forces lenient parsing even under a strict
host default.

## Tracing

A conforming host traces execution so runs can be audited and debugged. The
trace records:

- run start and end (with success/failure);
- every agent call attempt (agent, attempt number, rendered prompt) and
  every parse result (raw output, normalized output, error summary);
- every `exec` invocation (command, exit code, duration, stdout, stderr,
  timeout flag);
- every `set` mutation and every `print`;
- every raised exception.

Every exception value carries a `trace_id` field linking it to the
corresponding trace record. The normalized (recovered) JSON of a lenient
parse is traced alongside the raw output.

## Results and termination

A run ends in one of three ways:

1. **Success** — all statements executed; the host can observe the final
   root-scope bindings.
2. **Pre-execution failure** — a static error, input-validation error, or
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
