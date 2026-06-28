# Agent Calls

[← Index](index.md)

An agent call is the heart of AgL: an expression that sends a rendered
prompt to a host-provided agent and yields a **typed** result. All
agent invocations use the built-in `ask` function:

```agl
ask "Summarize ${topic}"
ask("Review this artifact:\n${artifact}", agent = reviewer)
ask("Review ${artifact}", agent = reviewer, on_parse_error = Retry(n = 2))
```

## `ask` — the agent call function

`ask` is a built-in function with the following declared-name signature:

```text
ask(prompt: text, agent: agent = «default»,
    format: text = «auto», strict_json: bool = «host default»,
    on_parse_error: ParsePolicy = Abort) -> T
```

where `T` is the **target type** — determined from the calling context (see
below). All parameters after `prompt` are optional and passed by name.

`ask` is a **contextual keyword**: it cannot be declared with `let`, `var`,
or `param`; it cannot be declared as an agent name; it may not be bound as a
function value (`let f = ask` is a static error, because `ask`'s type is
not a fully expressible monomorphic type). It remains legal as a
record/enum **field name**.

### Single-argument sugar

With no named arguments, `ask` may be called with the prompt string written
directly, without parentheses:

```agl
let s = ask "Hello?"          # equivalent to ask("Hello?")
```

With named arguments, parentheses are required:

```agl
let r: Review = ask("Review ${artifact}", agent = reviewer)
```

## Agents as values

Declared agents are **values** of the opaque type `agent`. An `agent`
declaration introduces a name binding of type `agent` in the top-level scope:

```agl
agent reviewer
agent planner = "claude -p %{PROMPT_FILE}"
```

Agent values may be stored, passed to functions, and held in lists — they
are ordinary value bindings:

```agl
let agents: list[agent] = [reviewer, planner]

def call_first(agents: list[agent], prompt: text) -> text =
  ask(prompt, agent = reviewer)
```

The `agent` type is **opaque**: no field access, no equality, no JSON encoding.
Agent values can be rendered as opaque handles such as `<agent reviewer>`, but
cannot be passed to `ask` except via the `agent` parameter.

### Agent declarations

```ebnf
agent_decl ::= "agent" NAME ("=" STRING)?
```

`agent` declarations are valid **only at the program root**. Each declared
name enters the root scope as an immutable binding of type `agent`. The
optional `= "…"` string attaches a *runner hint* consumed by the host and
has no language effect. The runner string must be a static literal — no
`${…}` interpolation.

Declaring the same agent name twice, or declaring `ask` or `exec` as an
agent name, is a static error.

### The default agent

When `agent` is omitted, `ask` uses the host's configured default agent.
There is no surface name for the default agent — it is implicit. A call
that omits `agent` requires the host to have a default agent configured; if
none is configured the host reports an invocation error before the program
runs.

## Target types: types as contracts

Every `ask` call has a **target type**, determined from context exactly as
expected types propagate ([Expressions](expressions.md)):

1. the annotation of the enclosing `let`/`var`,
2. the declared type of the binding in a `:=`,
3. an expected type propagated from a larger expression (for example a
   function parameter type),
4. otherwise **`text`**.

```agl
let x = ask "A"                          # target: text
let review: Review = ask("…", agent = reviewer)   # target: Review
var proposal: Turn = ask("…", agent = researcher)
proposal := ask("Revise.", agent = researcher)  # target: Turn
let _: unit = ask("Perform this task")          # response ignored
```

The target type drives the call's **output contract**:

- the codec used to parse the raw output (`text` or `json`; hosts may
  register more),
- a JSON Schema derived from the type,
- format instructions delivered to the agent alongside the prompt,
- runtime validation, the parse policy on failure, and the typed value bound
  on success.

`unit` is the exception: the call is dispatched once without an output
contract, its response is ignored, and the expression evaluates to `()`.
Because nothing is parsed, `format`, `strict_json`, and `on_parse_error` are
invalid for a `unit` target.

You normally never write "Return JSON matching …" yourself — the type
annotation is the source of truth.

### Target types may not be type variables

The target type of `ask`, of `exec`, and of the `ask-request` builder must be
a **concrete** type. It may not be — and may not contain — a type variable of
an enclosing generic `def` ([Generics](generics.md)). The contract for an
agent call (codec, derived JSON Schema, format instructions, validation) is
built from the type that is statically known at the call site, and a type
variable is opaque at that point: there is nothing to derive a schema from.

```agl
def fetch[T]() -> T =
  ask::[T]("give me a value")   # static error: target type is a type variable
```

This applies whether the type variable is the whole target (`ask::[T](…)`) or
merely appears inside it (`ask::[list[T]](…)` is equally rejected). It also
applies when the target is inferred from an enclosing annotation typed by a
type variable (`let r: T = ask(…)`).

A **concrete instance** of a generic type is perfectly fine, because it is no
longer a type variable — only its erased instantiation matters to the
contract:

```agl
enum Option[T]
  | none
  | some(value: T)

let n: Option[int] = ask("Pick a number, or nothing.", agent = picker)
```

## Named parameters

### `agent`

Selects the agent. The value must have type `agent`. When omitted, the host
default applies:

```agl
ask("Plan the next step.", agent = planner)
```

### `format`

Selects the output codec by name, as a `text` value. Normally unnecessary:
the codec is auto-selected from the target type — `text` targets use the
`text` codec; `json`, records, enums, lists, dictionaries, and numeric/boolean
types use the `json` codec. An explicit `format` must name a registered codec
that supports the call's target type; both are checked statically.

```agl
let r: Review = ask("Review ${a}", agent = reviewer, format = "json")
```

### `strict_json`

Opts a JSON-codec call into **strict** parsing (a `bool`). With `true`,
the response must be exactly one bare JSON value with nothing but surrounding
whitespace — no Markdown fences, no prose, no repair. With `false`,
explicitly selects lenient parsing, overriding a host default. It is a
static error unless the selected codec is `json`. When omitted, the host
default applies; the portable default is **lenient recovery** (see below).

### `on_parse_error`

The parse policy for invalid structured output. The value is a `ParsePolicy`
— one of two variants from the standard core enum:

```agl
enum ParsePolicy
  | Abort
  | Retry(n: int)
```

- `Abort` — raise `AgentParseError` on the first invalid output (the default).
- `Retry(n: N)` — after the initial call, make up to `N` additional
  corrective calls; raise `AgentParseError` if all fail.

```agl
let r: Review = ask(
  "Review ${artifact}",
  agent = reviewer,
  on_parse_error = Retry(n = 2)
)
```

A policy on a `text` target produces a static **warning** — text never
fails parsing, so the policy can never fire.

## The prompt

The `prompt` argument is a template rendered using the uniform interpolation
rules ([Strings and interpolation](strings-and-interpolation.md)). The
rendered prompt is delivered to the agent verbatim, together with the
contract's format instructions; the host must not perform further template
or environment-variable expansion over it. Retries resend the same rendered
prompt with corrective feedback appended.

## The JSON wire format

The canonical wire format for structured outputs is **JSON**, with these
rules:

1. The agent must return exactly one JSON value.
2. The response must not include Markdown fences, prose, or other text
   (this is what the format instructions request; lenient parsing forgives
   violations, strict parsing does not).
3. Records are JSON objects with exactly the declared fields.
4. Enums are JSON objects with a reserved **`"$case"`** tag naming the
   variant, plus that variant's fields. `"$case"` is reserved; since AgL
   field names are ordinary identifiers, user fields can never collide with
   it.
5. Unknown fields are rejected.
6. Missing required fields are rejected.

Example — for

```agl
enum Review
  | Pass
  | Fail(issues: list[text])
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
| `list[T]` | `{"type": "array", "items": <T>}` |
| `dict[text, V]` | `{"type": "object", "additionalProperties": <V>}` |
| record | object schema: `additionalProperties: false`, all fields `required`, per-field `properties` |
| enum | `oneOf` of per-variant schemas, each with a `"$case"` `const` plus payload fields, `additionalProperties: false` |

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

With `strict_json = true` (or a host default of strict), the response must be
exactly one bare JSON value with nothing but surrounding whitespace.

## Parse policies and retries

For a call with `on_parse_error = Retry(n = N)`:

1. The agent is called with the rendered prompt and the output contract.
2. The raw output is parsed and validated.
3. On success, the typed value is the call's result.
4. On failure, a **corrective retry** is dispatched with the same prompt and
   contract plus the previous invalid output and the structured validation
   errors.
5. At most `N` retries are made (`N + 1` total attempts).
6. If every attempt fails, **`AgentParseError`** is raised.

With `Abort` (the default), the first failure raises `AgentParseError` directly.

## Transport failures

A failure to *run* the agent at all — the agent process cannot be spawned,
exits nonzero, or times out — is distinct from a parse failure. It raises
**`AgentCallError`**, is catchable in-language, and is not eligible for
`on_parse_error` retries.

## Text targets

For a `text` target the raw output is bound verbatim — no parsing, no
validation. `on_parse_error` on such a call draws a static warning.

## What the agent receives

Each dispatch delivers to the host agent:

- the agent name (the name declared in source, or `"ask"` for the default
  agent);
- the fully rendered prompt;
- the output contract: target type, format instructions, and derived JSON
  Schema;
- the 0-based attempt number, and on retries the previous invalid output and
  its validation errors.

See [Host environment](host-environment.md).

## `ask-request` — the request builder

`ask-request` is the side-effect-free twin of `ask`: it builds the
`AgentRequest` that the corresponding `ask` call would dispatch to its agent
on its first attempt, **without invoking the agent**. It never dispatches,
never retries, never parses, and emits no trace events — it only assembles the
request value.

```agl
let r = ask-request("Summarize ${topic}")
let r = ask-request::[Review]("Review ${artifact}", agent = reviewer)
```

### Target type

Unlike `ask`, `ask-request` cannot infer its target type from context (its
result type is fixed to `AgentRequest`), so the target type is given
**explicitly** with the typed-call syntax `::[Type]`:

```agl
let r = ask-request::[Review]("Review ${artifact}")
let n = ask-request::[int]("How many?")
```

Omitting the type argument defaults the target to `text`:

```agl
let r = ask-request("Anything goes")   # target type is text
```

The target type drives the output contract exactly as it would for `ask`:
a `Review` target selects the JSON codec, derives a JSON Schema, and produces
format instructions; a `text` target selects the text codec. The returned
`AgentRequest.target_type` is `Some(value = "...")` for parsed response types
and `None` for `unit`, because a `unit` response is ignored:

```agl
let r = ask-request::[unit]("Perform this task")
```

`format`, `strict_json`, and `on_parse_error` are invalid for a `unit` target
because there is no response to parse.

### Typed-call syntax

`callee::[Type](args)` is a general typed-call form. The typed suffix follows a
postfix callee, so qualified constructors such as `Option.some::[int](value = 1)`
are valid as well. The type argument is delimited by square brackets because
AgL identifiers may contain `<` and `>`, so `Review>` would otherwise scan as
one token.

### Arguments

`ask-request` accepts the same named arguments as `ask` (`agent`, `format`,
`strict_json`, `on_parse_error`). `agent` labels the request's `agent` field
but never dispatches; `on_parse_error` is accepted (it shapes the contract's
parse policy) but has no runtime effect since no call is made.

### Result

The result is an `AgentRequest` record (see [Types](types.md)) with `attempt`
set to `0`, `previous_error` set to `None`, and optional contract details
represented with `Option[T]`. Because no call is made, `ask-request` works even
when no default agent is configured.
