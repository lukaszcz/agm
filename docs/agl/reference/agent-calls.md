# Agent Calls

[← Index](index.md)

An agent call is the heart of AgL: an expression that sends a rendered
prompt to a host-provided agent and yields a **typed** result.

```ebnf
agent_call   ::= agent_name call_options? template
agent_name   ::= VAR_NAME                 (* includes the contextual 'prompt' *)
call_options ::= "[" call_option ("," call_option)* ","? "]"
```

```agl
agent reviewer
agent planner

prompt "Summarize ${input}"
reviewer "Review this artifact:\n${artifact}"
reviewer[on_parse_error: retry[2]] "Review this artifact:\n${artifact}"
planner[format: json, on_parse_error: abort] "Plan the work."
```

## Agent names

The callee is a lowercase name resolved at check time:

- **`prompt`** is a contextual keyword denoting the *default agent*. It cannot
  be declared as a variable, bound by a pattern or `catch`, or declared as an
  agent name, but it remains legal as a field name
  ([Lexical structure](lexical-structure.md)).
- **`exec`** in call position denotes the shell executor — same call shape,
  different semantics ([Shell execution](shell-execution.md)). It too is never
  declared.
- **Any other name must be declared** at program root with an `agent`
  declaration before it may be called
  ([Bindings and scope](bindings-and-scope.md)). Calling a name that the
  program never declares is a static binding error
  (*"Unknown agent 'x'…"*), reported before anything executes. The
  declaration fixes the program's set of named agents; the environment that
  runs the program only supplies a *backing* for each declared name
  ([Host environment](host-environment.md)). A `prompt` call similarly
  requires a default agent to be configured.

```agl
agent reviewer
agent planner

let review: Review = reviewer "Review ${artifact}"
let plan = planner "Plan the work."
```

## Target types: types as contracts

Every call has a **target type**, determined from context exactly as
expected types propagate ([Expressions](expressions.md)):

1. the annotation of the enclosing `let`/`var`,
2. the declared type of the binding in a `set`,
3. an expected type propagated from a larger expression (e.g. a constructor
   field),
4. otherwise **`text`**.

```agl
let x = prompt "A"                       # target: text
let review: Review = reviewer "…"        # target: Review
var proposal: Turn = researcher "…"
set proposal = researcher "Revise."      # target: Turn (from the var)
```

The target type drives the call's **output contract**:

- the codec used to parse the raw output (`text` or `json` for the built-in
  types; hosts may register more),
- a JSON Schema derived from the type,
- format instructions delivered to the agent alongside the prompt,
- runtime validation, the parse policy on failure, and the typed value bound
  on success.

You normally never write "Return JSON matching …" yourself — the type
annotation is the source of truth.

## Call options

```ebnf
call_option ::= "format" ":" format_name
              | "strict_json" ":" ("true" | "false")
              | "on_parse_error" ":" ("abort" | "retry" "[" INT "]")
```

Duplicate option keys are a syntax error. Unknown option keys, malformed
values (`strict_json: maybe`, `on_parse_error: retry` without a bound,
anything other than `retry[N]` in bracket form) are syntax errors with
targeted messages.

### `format`

Selects the output codec by name. Normally unnecessary: the codec is
auto-selected from the target type — `text` targets use the `text` codec;
`json`, records, enums, lists, dictionaries, and the numeric/boolean types
use the `json` codec. An explicit `format` must name a registered codec that
supports the call's target type (both checked statically).

### `strict_json`

Opts a JSON-codec call into **strict** parsing (or, with `false`, explicitly
into lenient parsing, overriding a host default). It is a static error
unless the selected codec is `json`. When omitted, the host default applies;
the portable default is **lenient recovery** (below).

### `on_parse_error`

The parse policy for invalid structured output:

- `abort` — raise `AgentParseError` on the first invalid output.
- `retry[N]` — after the initial call, make up to `N` additional corrective
  calls; raise `AgentParseError` if all fail.

The portable default is `abort` (a single attempt). A policy on a `text`
target produces a static **warning** — text never fails parsing, so the
policy can never fire.

## The prompt

The template argument is rendered with **prompt rendering**: interpolated
values are boundary-marked and type-directed
([Strings and interpolation](strings-and-interpolation.md)). The rendered
prompt is delivered to the agent verbatim, together with the contract's
format instructions; the host must not perform any further template or
environment-variable expansion over it. The template is rendered **once**
per call site — retries resend the same rendered prompt with corrective
feedback appended.

## The JSON wire format

The canonical wire format for structured outputs is **JSON**, with these
rules:

1. The agent must return exactly one JSON value.
2. The response must not include Markdown fences, prose, or any other text
   (this is what the format instructions request; lenient parsing forgives
   violations, strict parsing does not).
3. Records are JSON objects with exactly the declared fields.
4. Enums are JSON objects with a reserved **`"$case"`** tag naming the
   variant, plus that variant's fields. `"$case"` is reserved by the wire
   format; since AgL field names are ordinary identifiers, a user field can
   never collide with it.
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
number with an integral value (e.g. `1.0`) satisfies an `int` target; a
non-integral number does not.

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
| enum | `oneOf` of per-variant object schemas, each with a `"$case"` `const` plus payload fields, `additionalProperties: false` |

### Format instructions

Alongside the prompt the agent receives instructions derived from the target
type. For a record they are equivalent to:

```text
Return exactly one JSON object.
Do not include Markdown, prose, or code fences.
The JSON must have exactly these fields:
- title: string
- severity: integer
- description: string
```

For an enum:

```text
Return exactly one JSON object.
Do not include Markdown, prose, or code fences.
Use "$case" to identify the selected variant.

Valid shapes:
{ "$case": "Pass" }
{ "$case": "Fail", "issues": [...] }
```

For other JSON-codec targets (scalars, lists, dictionaries, `json`):

```text
Return exactly one JSON value.
Do not include Markdown, prose, or code fences.
```

For `text` targets: `Return plain text.`

Format instructions always describe the **strict** shape, regardless of the
parsing mode.

## Parsing: lenient and strict

### Lenient recovery (portable default)

Lenient mode recovers **exactly one** JSON value from a possibly chatty
response, then validates it strictly:

1. If the (whitespace-stripped) response is already a single valid JSON
   value, it is used as-is.
2. Otherwise, if the response contains a Markdown code fence
   (``` ```json … ``` ``` or ``` ``` … ``` ```), the fenced content is
   parsed, with trivial-malformation repair (single-quoted strings, trailing
   commas, and similar) if needed.
3. Otherwise the whole response undergoes the same repair, which strips
   surrounding prose such as `Here you go: {…}`.
4. As a last resort, a single bare scalar (a number, `true`, `false`, or
   `null`) embedded in prose is recovered — but only when exactly one such
   token is present.

If the response contains **two or more** top-level JSON values, recovery
fails as ambiguous rather than guessing. If nothing can be recovered,
parsing fails.

Leniency applies only to *finding and repairing* the JSON value. **Schema
validation is always strict** — wire-format rules 3–6 are never relaxed. The
recovered, normalized JSON text is traced alongside the raw output and is
available as `AgentParseError.normalized_raw` on failure.

### Strict parsing

With `strict_json: true` (or a host default of strict), the response must be
exactly one bare JSON value, with nothing but surrounding whitespace — no
fences, no prose, no repair.

## Parse policies and retries

For a call with `on_parse_error: retry[N]`:

1. The agent is called with the rendered prompt and the output contract.
2. The raw output is parsed and validated.
3. On success, the typed value is the call's result.
4. On failure, a **corrective retry** is dispatched to the same agent with
   the same prompt and contract, plus the previous invalid output and the
   structured validation errors, and an incremented attempt counter.
5. At most `N` retries are made (so `N + 1` attempts in total).
6. If every attempt fails, **`AgentParseError`** is raised.

The corrective feedback delivered on retries is equivalent to:

```text
Your previous response did not match the required output format.

Validation errors:
- <error>
…

Previous response:
<previous raw output>

Return only valid JSON matching the schema.
```

The exact wording is host-configurable; the semantics — same agent, same
prompt, same typed contract, previous output and errors included — are not.

Validation errors are structured records with a `category` (one of
`missing_field`, `unknown_field`, `wrong_type`, `bad_case` for an enum whose
`"$case"` is missing or names no variant, or `invalid_json` when no JSON
value could be recovered at all), a human-readable `message`, a JSON `path`
to the offending value, and the offending `field` where applicable. They
appear in retry feedback and in `AgentParseError.validation_errors`.

With `on_parse_error: abort` (or no policy), the first failure raises
`AgentParseError` directly. Failed attempts are still traced and count as
agent attempts.

`AgentParseError` carries the agent name, target type, expected schema, raw
and normalized output, the validation errors, and the attempt count — see
[Exceptions](exceptions.md).

## Transport failures

A failure to *run* the agent at all — the agent process cannot be spawned,
exits nonzero, or times out — is distinct from a parse failure. It raises
**`AgentCallError`** (with a `cause` of `"spawn_failure"`,
`"nonzero_exit"`, or `"timeout"`), is catchable in-language, and is **not**
eligible for `on_parse_error` retries.

An agent that succeeds with empty output is a valid transport result: a
`text` target binds the empty string; a structured target proceeds to
parsing and normally fails with `AgentParseError`.

## Text targets

For a `text` target the raw output is bound verbatim — no parsing, no
validation, no boundary processing. `on_parse_error` on such a call draws a
static warning, as noted above.

## What the agent receives

Conceptually, each dispatch delivers to the host agent:

- the agent name as written in source (`"prompt"` for the default agent);
- the fully rendered prompt;
- the output contract: target type, format instructions, and derived JSON
  Schema (so schema-capable agent backends can request native structured
  output);
- the 0-based attempt number, and on retries the previous invalid output and
  its validation errors.

The agent returns raw text (optionally with host-level metadata). Everything
else — parsing, validation, retrying, binding, tracing — is the runtime's
job. See [Host environment](host-environment.md).
