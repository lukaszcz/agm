# AgL Language Reference

AgL (Agent Language) is a small, statically checked DSL for writing
**agent workflows**: programs that call AI agents with typed output contracts,
chain prompt results, run review/fix/refine loops, and branch on structured
agent outcomes.

AgL is not a general-purpose programming language. It is a typed orchestration
language whose core ideas are:

- **Agent calls are first-class expressions.** `reviewer "Review ${artifact}"`
  calls a host-provided agent with a rendered prompt template.
- **Types are contracts at the LLM boundary.** Annotating an agent call's
  result type drives the format instructions sent to the agent, the parsing
  and validation of its raw output, and the retry-or-abort behavior on
  malformed output.
- **Structured outputs have one canonical wire format** — JSON, with a
  reserved `"$case"` tag for enum variants — parsed leniently by default and
  validated strictly, always.
- **Mutability is explicit.** `let` binds immutably, `var` binds mutably,
  `set` updates; a single `=` is the equality operator, never assignment.
- **Failures are exceptions.** Parse failures, loop exhaustion, match
  failures, and shell errors are typed, catchable exceptions.
- **Blocks use indentation**, with deterministic inline forms for one-liner
  workflows.

A taste of the language:

```agl
input spec: text

enum Review
  | Pass
  | Fail(issues: list[text])

var artifact: text = impl "Implement ${spec}"

do[5]
  let review: Review = reviewer[on_parse_error: retry[2]] """
  Review the artifact for correctness and completeness.

  Artifact:
  ${artifact}
  """

  case review of
    | Fail(issues) =>
        set artifact = impl "Fix these issues:\n${issues}\n\nCurrent:\n${artifact}"
    | Pass =>
        pass

until review is Pass

print artifact
```

The same workflow as a one-liner (assuming `Review` and the input `spec`
are declared):

```agl
var artifact: text = impl "Implement ${spec}"; do[5] let review: Review = reviewer[on_parse_error: retry[2]] "Review ${artifact}"; case review of | Fail(issues) => set artifact = impl "Fix ${issues} in ${artifact}" | Pass => pass until review is Pass
```

## Chapters

| Chapter | Contents |
| ------- | -------- |
| [Lexical structure](lexical-structure.md) | Source text, comments, indentation and layout, keywords, identifiers, tokens, operator precedence |
| [Program structure](program-structure.md) | Programs, statements, blocks, inline forms, the bar-safe rules |
| [Types](types.md) | Built-in types, `record`/`enum`/`type` declarations, assignability and coercion |
| [Bindings and scope](bindings-and-scope.md) | `let`, `var`, `set`, `input`, lexical scoping, shadowing |
| [Expressions](expressions.md) | Literals, constructors, field access, operators, `case` expressions, type inference |
| [Pattern matching](pattern-matching.md) | Patterns, matching semantics, exhaustiveness |
| [Control flow](control-flow.md) | `if`, `case` statements, `do … until` loops |
| [Strings and interpolation](strings-and-interpolation.md) | Templates, escapes, `${…}` interpolation, renderers, rendering contexts |
| [Agent calls](agent-calls.md) | Calling agents, call options, output contracts, the JSON wire format, parse policies and retries |
| [Shell execution](shell-execution.md) | The `exec` built-in, shell-safe interpolation, `ExecError` |
| [Exceptions](exceptions.md) | The exception model, `try`/`catch`/`raise`, the built-in exception catalog |
| [Host environment](host-environment.md) | Agents, inputs, host defaults, capability checking, tracing |
| [Grammar](grammar.md) | The collected grammar |

## Notation

Code blocks marked `agl` contain AgL source. Grammar fragments use an
EBNF-like notation: `::=` defines a production, `|` separates alternatives,
`?` marks an optional element, `*` and `+` mark repetition, and quoted
strings are literal tokens. Token names in `UPPER_CASE` refer to the lexical
tokens defined in [Lexical structure](lexical-structure.md).

Throughout the reference, "the host" refers to the runtime environment that
embeds AgL: it registers agents, supplies program inputs, executes shell
commands, and records traces. Behavior marked *host-configurable* has a
documented portable default that hosts may override; everything else is fixed
by the language.

## Error model at a glance

AgL distinguishes three failure layers:

1. **Static errors** — syntax, scope, and type errors. A program with a static
   error never executes any statement and never calls any agent. Static
   checking stops at the first error.
2. **Static warnings** — advisory diagnostics (for example, a non-exhaustive
   `case` over an enum). Warnings never prevent execution.
3. **Runtime exceptions** — typed, catchable in-language values such as
   `AgentParseError` or `MaxIterationsExceeded`. Uncaught exceptions terminate
   the program. See [Exceptions](exceptions.md).

Invalid host input (a missing or ill-typed program input) is a **host
invocation error**: it is reported before anything runs and is not catchable
in-language. See [Host environment](host-environment.md).
