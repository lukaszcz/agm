# Agent Calls

[← Index](index.md)

An agent call is the heart of AgL: an expression that sends a rendered
prompt to a host-provided agent and yields a **typed** result. Invoke the
built-in `ask` function directly, or select it as a method on the `Agent`
value that should receive the request:

<!-- agl-check: fragment -->
```agl
ask "Summarize %{topic}"
reviewer.ask("Review this artifact:\n%{artifact}")
reviewer.ask("Review %{artifact}", on-parse-error = Retry(n = 2))
reviewer.ask::[Review]("Review %{artifact}", on-parse-error = Retry(n = 2))
```

## `ask` — the agent call function

`ask` is a built-in function with the following effective call signature:

```text
ask(prompt: text, agent: Agent = std/config::default-agent,
    format: text = "", strict-json: bool = false,
    on-parse-error: ParsePolicy = ParsePolicy::Abort) -> T
```

where `T` is the **target type** — determined from the calling context (see
below). All parameters after `prompt` are optional and passed by name.

An `Agent` value also provides the method form:

```text
Agent::ask(self, prompt: text, format: text = "",
           strict-json: bool = false,
           on-parse-error: ParsePolicy = ParsePolicy::Abort) -> T
```

Free `ask` uses the snapshot default `Session` when `agent` is omitted; its
agent is selected from `std/config::default-agent`. Supplying the named `agent`
argument instead selects an explicit agent for that one call, as does the
`reviewer.ask(...)` receiver form. Explicit-agent calls open a short-lived
session for the call and all of its parse retries, then close it.
Both forms support contextual and explicit `::[T]` target types and named parse
options. A method reference captures its receiver and retains only the required
`prompt` parameter:

```agl
program def main() -> unit =
  let reviewer: Agent = AgentCommand("reviewer")
  let query: text -> int = reviewer.ask
  let query-text = reviewer.ask::[text]
  ()
```

Configured variants still use a lambda around a direct call.

`ask` is a **contextual keyword**: it cannot be declared with `let`, `var`,
or as a function parameter name, but it can be referenced as a function value.
The value is an eta-expanded `text -> T` callable using the default session;
its result is fixed by an expected function type or explicit `::[T]`, and
defaults to `text` when unconstrained:

```agl
program def main() -> unit =
  let query: text -> int = ask
  let query-json = ask::[json]
  ()
```

Use an explicit lambda around a direct call to capture a non-default agent or
parse options. `ask` remains legal as a record/enum **field name**.

### `Session::ask`

`Session` also has the method form:

```text
Session::ask[T](self, prompt: text, format: text = "",
                strict-json: bool = false,
                on-parse-error: ParsePolicy = ParsePolicy::Abort) -> T
```

It sends the prompt through that live session, so the session's stored agent
and transport select the backend; it has no `agent` argument. It uses the same
contextual or explicit `::[T]` target, concrete-target restriction, parse
options, and output-contract checking as `ask`. `session.ask$` has the same
raw-tail spelling rules as `reviewer.ask$`. Parse retries remain in this same
conversation. A reference such as `let query: text -> Review = session.ask`
captures the live session.

### Single-argument sugar

With no named arguments, `ask` may be called with the prompt string written
directly, without parentheses:

```agl
program def main() -> unit =
  let s = ask "Hello?"
```

With named arguments, parentheses are required:

<!-- agl-check: fragment -->
```agl
let r: Review = reviewer.ask("Review %{artifact}")
```

## Raw-tail `ask$`

`ask$` writes a prompt directly after the keyword. Inline form consumes the
rest of its line; block form consumes one dedented, newline-joined prompt.
It desugars to the same call as `ask(<template>)`, so explicit type arguments
and target-type inference work exactly as for `ask`. It may also follow an
`Agent` projection: `reviewer.ask$` desugars to `reviewer.ask(<template>)`.
Type arguments must touch the raw name (`ask$::[T]` or
`reviewer.ask$::[T]`); in `ask$ ::[T]`, the spaced `::[T]` is prompt payload:

```agl
record Review
  summary: text

program def main() -> unit =
  let subject = "the release notes"
  let summary: text = ask$ Summarize %{subject}.
  let review: Review = ask$::[Review]
    Review %{subject} and provide a concise summary.
```

Raw-tail prompt text is verbatim except for `%{expr}` interpolation and
trailing spaces and tabs in an inline prompt; `\%{` writes a literal `%{`.
A raw call needs a nonempty inline prompt or a block with at least one nonblank
line. A bare `ask$` uses the default session and `reviewer.ask$` opens the
short-lived session for its receiver. Use `ask(...)` or `reviewer.ask(...)`
when setting `format`, `strict-json`, or `on-parse-error`; the parenthesized
forms are also available outside a raw-tail line-final position.

## Agents as values

`Agent` is a built-in enum whose values describe the backend to invoke:

```agl
def review-with(agent: Agent, artifact: text) -> text =
  agent.ask("Review this artifact:\n%{artifact}")

program def main() -> unit =
  let reviewer: Agent = AgentClaude("sonnet", "medium")
  let second-opinion = AgentCodex("o3", "high")
  let scripted = AgentCommand("claude -p")
  let hosted = AgentPi("openai", "gpt-5", "low")
  let candidates: array[Agent] = [reviewer, second-opinion, scripted, hosted]
  let first-pass: text = review-with(reviewer, "the release notes")
  let second-pass: text = review-with(second-opinion, first-pass)
  print("%{candidates.size()} agents available")
  print(second-pass)
```

Each member record selects its backend invocation. `AgentCommand` accepts a
shell-like command string; the provider members carry their model and thinking
settings. An explicit `agent.ask(...)` uses an ephemeral session for the complete
parse-retry loop, so its initial invocation is:

| Member | Initial invocation for an explicit ask |
| --- | --- |
| `AgentCommand(command)` | the supplied command, with the normal prompt-file handling; retries require `%{SESSION_ID}` |
| `AgentClaude(model, thinking)` | `claude -p --session-id <id> --model <model> --effort <thinking>` |
| `AgentCodex(model, thinking)` | `codex exec --json --model <model> -c model_reasoning_effort=<thinking> -` (prompt on stdin) |
| `AgentPi(provider, model, thinking)` | `pi --mode rpc --provider <provider> --model <model> --thinking <thinking>` |

An empty provider, model, or thinking field omits its flag. `Agent` values are
ordinary enum data: they can be stored, passed to functions, rendered,
inspected, and JSON-encoded like other enum values. At a host boundary, a CLI or TOML
value whose declared type is `Agent` additionally accepts the compact native-agent and
command forms documented under [`agm exec`](../../commands/exec.md#host-agent-syntax).

Because `Agent` is ordinary enum data, it is also decodable: an `ask` whose
target type is `Agent`, or a cast of foreign JSON to `Agent`, produces a value
whose `AgentCommand` member record carries the command the host will spawn on
the next call through it. Data reaching such a decode therefore chooses a
subprocess. Construct `Agent` values in source, or from data you trust, when
that matters.

### The default agent

Free `ask` uses `Session::default`, which lazily opens one session and snapshots
the current `std/config::default-agent` when it is first used. Every later free
`ask` in that run or REPL session uses the same conversation and agent; a later
`default-agent` write does not switch it. The standard library supplies a
default; CLI `--default-agent` and `[exec] default-agent` seeds override it, and a
source write takes effect before that snapshot is created:

```agl
import std/config

program def main() -> unit =
  std/config::default-agent := AgentClaude("sonnet", "medium")
  let answer: text = ask("Summarize")
```

## Sessions

Open an explicit conversation when several calls or lifecycle operations must
share it:

<!-- agl-check: fragment -->
```agl
let session = Session::open(AgentClaude("sonnet", "medium"), name = "review")
let first: text = session.ask("Read the artifact.")
let second: text = session.ask("Now list the risks.")
let branch = session.fork()
session.close()
branch.close()
```

`Session::open(agent, transport = None, name = "")` opens a session;
`Session::default()` returns the same lazy default session used by free `ask`.
`compact(instructions = "")`, `reset()`, `fork()`, `stats()`,
`set-name(name)`, and `close()` are session operations. `reset` keeps the
AgL session value but starts a fresh backend conversation; `fork` returns a
new session whose history begins from the parent; `close` is idempotent, but
later use of that session raises `SessionError`. Backend support for the other
operations is runtime-dependent; an unsupported operation raises
`SessionError`.

When `transport` is omitted or `None`, `AgentPi` uses `Rpc`; every other
agent uses `Cli`. `Rpc` is supported only for `AgentPi`; selecting it for
another agent raises `SessionError` while opening. The CLI backends use their
respective continuation conventions, while Pi RPC keeps one `pi --mode rpc`
process for the session.

| Agent and transport | Native optional operations | Notes |
| --- | --- | --- |
| `AgentCommand`, `Cli` | none | `ask`, `reset`, and `close` work. A continuing session requires an unescaped `%{SESSION_ID}` in the command. |
| `AgentClaude`, `Cli` | `compact`, `fork` | `name` is accepted when opened; later `set-name` and `stats` are unsupported. |
| `AgentCodex`, `Cli` | none | The first ask starts a thread; later asks resume it. |
| `AgentPi`, `Cli` | `fork` | `name` is accepted when opened; later `compact`, `set-name`, and `stats` are unsupported. |
| `AgentPi`, `Rpc` | `compact`, `fork`, `set-name`, `stats` | Persistent Pi RPC process; this is the default for `AgentPi`. |

Every row supports `ask`, `reset`, and `close`. A nonempty `name` passed to
`Session::open` is rejected for command and Codex CLI sessions. An explicit `Agent::ask` has
the same transport default, but its session lasts only for that call and its
retries; use `Session::open` to keep the conversation after the call. A single-attempt
`Agent::ask` sends exactly one prompt, so its session never has to be continued: an
`AgentCommand` does not require `%{SESSION_ID}` and the other CLI backends run their plain
prompt command. Enabling corrective retries makes every attempt share one conversation, so
an `AgentCommand` then requires the placeholder.

## Target types: types as contracts

Every `ask` call has a **target type**, determined from context exactly as
expected types propagate ([Expressions](expressions.md)):

1. the annotation of the enclosing `let`/`var`,
2. the declared type of the binding in a `:=`,
3. an expected type propagated from a larger expression (for example a
   function parameter type),
4. otherwise **`text`**.

<!-- agl-check: fragment -->
```agl
let x = ask "A"                          # target: text
let review: Review = reviewer.ask("…")   # target: Review
var proposal: Turn = researcher.ask("…")
proposal := researcher.ask("Revise.")  # target: Turn
let completed: unit = ask("Perform this task")  # response ignored
```

The target type drives the call's **output contract**:

- the codec used to parse the raw output (`text` or `json`; hosts may
  register more),
- a JSON Schema derived from the type,
- format instructions delivered to the agent alongside the prompt,
- runtime validation, the parse policy on failure, and the typed value bound
  on success.

`unit` is the exception: the call is dispatched once without an output
contract, its response is ignored, and the expression evaluates to `void`.
A bare `ask` in a discarded-value position therefore runs as a fire-and-forget
unit call:

```agl
program def main() -> unit =
  ask "Notify the reviewer."
```

Because nothing is parsed, `format`, `strict-json`, and `on-parse-error` are
invalid for a `unit` target.

You normally never write "Return JSON matching …" yourself — the type
annotation is the source of truth.

When a target is constrained by another part of the same enclosing expression,
AgL first resolves that expression's type and then builds the contract from the
resolved concrete target. This gives a call in a generic call or constructor
the same codec, schema, and validation as writing that concrete target directly:

```agl
def select[T](first: T, second: T) -> T = first

program def main() -> unit =
  let count = select(ask("Choose a count."), 1)
```

### Target types may not be type variables

The target type of `ask` and `exec` must be a **concrete** type. It may not
be — and may not contain — a type variable of an enclosing generic `def`
([Generics](generics.md)). The contract for an
agent call (codec, derived JSON Schema, format instructions, validation) is
built from the type that is statically known at the call site, and a type
variable is opaque at that point: there is nothing to derive a schema from.

<!-- agl-check: error -->
```agl
def fetch[T]() -> T =
  ask::[T]("give me a value")   # static error: target type is a type variable
```

This applies whether the type variable is the whole target (`ask::[T](…)`) or
merely appears inside it (`ask::[array[T]](…)` is equally rejected). It also
applies when the target is inferred from an enclosing annotation typed by a
type variable (`let r: T = ask(…)`).

A **concrete instance** of a generic type is perfectly fine, because it is no
longer a type variable — only its erased instantiation matters to the
contract:

<!-- agl-check: fragment -->
```agl
enum Option[T]
  | None
  | Some(value: T)

let n: Option[int] = picker.ask("Pick a number, or nothing.")
```

### Recursive target types

A [recursive record or enum](types.md#recursive-types) works as a target type
exactly like any other: the response is validated and decoded through the same
output contract, and construction/matching/equality on the resulting value
work normally.

<!-- agl-check: fragment -->
```agl
enum Tree
  | Leaf
  | Node(value: int, left: Tree, right: Tree)

let t: Tree = builder.ask("Build a tree.")
```

The one restriction is on the type's **shape**, not on recursion itself: the
target type must have a finite JSON Schema. A non-generic recursive type
always does. A [polymorphically recursive](generics.md#recursive-generic-types)
generic type whose reachable instantiations never close (its schema would need
infinitely many distinct concrete shapes) is rejected as a target type at the
call site with a static error — the type is otherwise fully usable in every
other position, only its JSON Schema is unbounded. See
[Generics](generics.md#the-finite-schema-boundary) for the exact boundary.

## Named parameters

### `format`

Selects the output codec by name, as a `text` value. Normally unnecessary:
the codec is auto-selected from the target type — `text` targets use the
`text` codec; `json`, records, enums, arrays, dictionaries, and numeric/boolean
types use the `json` codec. An explicit `format` must name a registered codec
that supports the call's target type; both are checked statically.

<!-- agl-check: fragment -->
```agl
let r: Review = reviewer.ask("Review %{a}", format = "json")
```

### `strict-json`

Opts a JSON-codec call into **strict** parsing (a `bool`). With `true`,
the response must be exactly one bare JSON value with nothing but surrounding
whitespace — no Markdown fences, no prose, no repair. With `false`,
explicitly selects lenient parsing, overriding a host default. It is a
static error unless the selected codec is `json`. When omitted, the host
default applies; the portable default is **lenient recovery** (see below).

### `on-parse-error`

The parse policy for invalid structured output. The value is a `ParsePolicy`
— one of two members from the standard-library enum:

<!-- agl-check: fragment -->
```agl
enum ParsePolicy
  | Abort
  | Retry(n: int)
```

- `Abort` — raise `AgentParseError` on the first invalid output (the default).
- `Retry(n: N)` — after the initial call, make up to `N` additional
  corrective calls; raise `AgentParseError` if all fail.

<!-- agl-check: fragment -->
```agl
let r: Review = reviewer.ask(
  "Review %{artifact}",
  on-parse-error = Retry(n = 2)
)
```

A policy on a `text` target produces a static **warning** — text never
fails parsing, so the policy can never fire.

## The prompt

The `prompt` argument is a template rendered using the uniform interpolation
rules ([Strings and interpolation](strings-and-interpolation.md)). The
rendered prompt is delivered to the agent verbatim, together with the
contract's format instructions; the host must not perform further template
or environment-variable expansion over it. The prompt is delivered through its
session. Corrective feedback includes a
category-based validation summary, never validation paths, keys, or other
response-derived details. Retries stay in their existing conversation and send
only that summary plus a JSON-format reminder; they never resend the original
prompt or invalid response.

## The JSON wire format

The canonical wire format for structured outputs is **JSON**, with these
rules:

1. The agent must return exactly one JSON value.
2. The response must not include Markdown fences, prose, or other text
   (this is what the format instructions request; lenient parsing forgives
   violations, strict parsing does not).
3. Records are JSON objects with exactly the declared fields.
4. Enums are JSON objects with a reserved **`"$case"`** tag naming the
   terminal member name, plus that member record's fields. `"$case"` is
   reserved; since AgL field names are ordinary identifiers, user fields can
   never collide with it. A record-typed slot for the same value is a plain
   object with no `"$case"` tag.
5. Unknown fields are rejected.
6. Missing required fields are rejected.

Example — for

```agl
enum Review
  | Pass
  | Fail(issues: array[text])
```

valid responses are:

```json
{ "$case": "Pass" }
```

```json
{ "$case": "Fail", "issues": ["missing tests", "unclear API"] }
```

Numbers on the wire are parsed exactly (never through binary floats). A
number with an integral value (e.g. `1.0`) satisfies an `int` target.

### Derived JSON Schema

The schema sent to schema-aware agents and used for validation is derived
mechanically from the target type:

| AgL type | Schema |
| -------- | ------ |
| `text` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `decimal` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `json` | `{}` (any JSON value) |
| `array[T]` | `{"type": "array", "items": <T>}` |
| `dict[text, V]` | `{"type": "object", "additionalProperties": <V>}` |
| record | object schema: `additionalProperties: false`, all fields `required`, per-field `properties` |
| enum | `oneOf` of per-member-record schemas, each with a `"$case"` `const` plus record fields, `additionalProperties: false` |

A target type's schema uses standard JSON Schema `$defs`/`$ref` for any
record/enum it would otherwise repeat. A reachable type gets one entry under a
top-level `"$defs"` object when it is
[recursive](types.md#recursive-types), or when it occurs in more than one
place, and every occurrence — including the target itself, if it is directly
recursive — is a `{"$ref": "#/$defs/<name>"}` instead of being inlined. A type
reached exactly once and not recursive stays inlined, exactly as the table
above shows.

### Format instructions

Alongside the prompt the agent receives instructions derived from the target
type. For a JSON-typed target the instructions embed the actual JSON Schema
(the same schema used to validate the response), so the agent receives the
precise, authoritative shape rather than a prose paraphrase. They are
equivalent to:

```text
Return exactly one JSON value conforming to the following JSON Schema.
Do not include Markdown, prose, or code fences.

```json
<derived JSON Schema>
```
```

For the permissive `json` type (schema `{}`) only the behavioural preamble is
emitted, since there is no shape to convey.

For `text` targets the format instructions are absent (empty): a text
target imposes no format on the agent's response.

Format instructions always describe the **strict** shape, regardless of the
parsing mode.

## Parsing: lenient and strict

### Lenient recovery (portable default)

Lenient mode recovers **exactly one** JSON value from a possibly chatty
response, then validates it strictly:

1. If the (whitespace-stripped) response is already a single valid JSON
   value, it is used as-is.
2. Otherwise, if the response contains a Markdown code fence, the fenced
   content is parsed, with trivial-malformation repair if needed.
3. Otherwise the whole response undergoes the same repair, which strips
   surrounding prose such as `Here you go: {…}`.
4. As a last resort, a single bare scalar embedded in prose is recovered —
   but only when exactly one such token is present.

If the response contains two or more top-level JSON values, recovery fails
as ambiguous. Schema validation is always strict regardless of lenient mode.

### Strict parsing

With `strict-json = true` (or a host default of strict), the response must be
exactly one bare JSON value with nothing but surrounding whitespace.

## Parse policies and retries

For a call with `on-parse-error = Retry(n = N)`, attempt 1 sends the rendered
prompt plus its output-format instructions. The output is then parsed and
validated. Each failed parse or validation sends at most `N` corrective
follow-ups in the **same session** (`N + 1` attempts total). A follow-up contains
only a category-based validation summary and a reminder to return valid JSON;
it does not repeat the original prompt, output contract, or invalid response.
The summary never exposes response-derived paths, keys, or values. Success
returns the typed value; exhausting the attempts raises **`AgentParseError`**.

With `Abort` (the default), the first parse or validation failure raises
`AgentParseError` directly. This retry rule applies equally to the default,
explicit-agent, and explicit-session forms.

## Transport and session failures

A failed `ask` transport — for example a process spawn failure, nonzero exit,
idle timeout, or a failed Pi RPC prompt — raises **`AgentCallError`**. It is
catchable and is never retried by `on-parse-error`. **`SessionError`** instead
reports a session lifecycle, capability, or non-ask backend failure: opening or
using a closed session, an unsupported operation or transport, and failed
compaction/fork/reset/name/stats operations. `SessionError.operation` names the
operation. `AgentParseError` is only for output that arrived but could not meet
the requested structured contract.

## Text targets

For a `text` target the raw output is bound verbatim — no parsing, no
validation. `on-parse-error` on such a call draws a static warning.

## What the agent receives

Each dispatch delivers to the host agent:

- the selected encoded `Agent` value;
- the fully rendered prompt;
- the output contract: target type, format instructions, and derived JSON
  Schema;
- the 0-based attempt number; retries include only a category-based validation
  summary and a format reminder. The summary excludes response-derived
  validation paths, keys, and values.

See [Host environment](host-environment.md).

## `ask-request` — the request builder

`ask-request` is the side-effect-free twin of `ask`: it builds the
first-attempt `AgentRequest` that the matching `ask` would have dispatched,
**without invoking an agent**. Its direct form mirrors `ask`'s whole call
surface:

```text
ask-request[T](
  prompt: text,
  agent: Agent = std/config::default-agent,
  format: text = "",
  strict-json: bool = false,
  on-parse-error: ParsePolicy = ParsePolicy::Abort,
) -> AgentRequest
```

The optional named `agent` is captured in the request; `reviewer.ask-request(...)`
captures its receiver instead. The type argument and the parse-shaping options
select the output contract exactly as they do for `ask`, and the target type
comes from the type argument alone — never from context, whose expected type
here is the request record rather than the output the request asks for. Without
one, the request describes a `text` output, as `ask` does. The builder never
dispatches, retries, parses, or emits trace events.

<!-- agl-check: fragment -->
```agl
let r = ask-request::[Summary]("Summarize %{topic}")
```

The result is an `AgentRequest` record (see [Types](types.md)) with `attempt`
set to `0`, `previous-error` set to `None`, and the requested output contract
recorded for inspection.
