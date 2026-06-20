# AgL Language Reference

AgL (Agent Language) is a small, statically checked DSL for writing
**agent workflows**: programs that call AI agents with typed output contracts,
chain prompt results, run review/fix/refine loops, and branch on structured
agent outcomes.

AgL is not a general-purpose programming language. It is a typed, expression-
oriented orchestration language whose core ideas are:

- **Agent calls are first-class expressions.** `ask("Review ${artifact}", agent: reviewer)`
  calls a host-provided agent with a rendered prompt template; the result is
  a typed value usable in any expression position.
- **Types are contracts at the LLM boundary.** Annotating an `ask` call's
  result type drives the format instructions sent to the agent, the parsing
  and validation of its raw output, and the retry-or-abort behavior on
  malformed output.
- **Structured outputs have one canonical wire format** — JSON, with a
  reserved `"$case"` tag for enum variants — parsed leniently by default and
  validated strictly, always.
- **Everything is an expression.** There is no statement category: binders
  (`let`/`var`) scope over a continuation, side-effecting forms yield `unit`,
  and `if`/`case`/`try` with matching branches yield a typed value.
- **Functions are first-class.** User-defined `def` declarations and `fn`
  lambdas produce values of function type `(A, B) -> C`; they may be stored,
  passed, and returned.
- **Mutability is explicit.** `let` binds immutably, `var` binds mutably,
  `:=` updates; a single `=` is the equality operator, never assignment.
- **Failures are exceptions.** Parse failures, cast failures, loop exhaustion,
  match failures, recursion depth, and shell errors are typed, catchable exceptions.
- **Blocks use indentation**, with deterministic inline forms for one-liner
  workflows.

A taste of the language:

```agl
param spec: text

enum Review
  | Pass
  | Fail(issues: list[text])

agent reviewer
agent impl

def review_and_fix(artifact: text, retries: int = 2) -> text =
  let r: Review = ask(
    "Review the artifact for correctness:\n${artifact}",
    agent: reviewer,
    on_parse_error: Retry(n: retries)
  )
  case r of
    | Pass => artifact
    | Fail(issues) => ask(
        "Fix these issues:\n${issues}\n\nCurrent:\n${artifact}",
        agent: impl
      )

var artifact: text = ask("Implement ${spec}", agent: impl)

do[5]
  artifact := review_and_fix(artifact)
  let final: Review = ask("Final review:\n${artifact}", agent: reviewer)
until final is Pass
```

## Chapters

| Chapter | Contents |
| ------- | -------- |
| [Lexical structure](lexical-structure.md) | Source text, comments, indentation and layout, keywords, tokens, operator precedence |
| [Program structure](program-structure.md) | Programs, blocks, items, binders, inline forms |
| [Modules](modules.md) | File-based module system: module identity, import forms, qualified access, visibility, cyclic imports, REPL imports |
| [Types](types.md) | Built-in types (`unit`, `text`, `int`, `decimal`, `bool`, `json`, `agent`, function types), `record`/`enum`/`type` declarations, prelude types (`ExecResult`, `ParsePolicy`), assignability, casts and convertibility (`as`/`as?`) |
| [Bindings and scope](bindings-and-scope.md) | `let`, `var`, `:=`, `param`, `program`, `agent`, `def`, lexical scoping, shadowing |
| [Expressions](expressions.md) | Literals, constructors, calls, operators, `as`/`as?` cast operators, `parse_json`, `case`/`if` expressions, `unit`-typed forms, expected-type propagation |
| [Functions](functions.md) | `def` declarations, `fn` lambdas, optional/named arguments, function types, first-class values, recursion and depth limit |
| [Pattern matching](pattern-matching.md) | Patterns, matching semantics, exhaustiveness |
| [Generics](generics.md) | Type parameters on `def`/`record`/`enum`/`type`, type application, inference and `::[…]` override, generic constructor values, strict parametricity, invariance, erasure |
| [Control flow](control-flow.md) | `if`, `case`, `do … until` loops |
| [Strings and interpolation](strings-and-interpolation.md) | Templates, escapes, `${…}` interpolation, uniform rendering rules |
| [Agent calls](agent-calls.md) | `ask`, agents as values, call options, output contracts, the JSON wire format, parse policies and retries |
| [Shell execution](shell-execution.md) | `exec`, the `ExecResult` structured form vs the parsed form, `ExecError` |
| [Exceptions](exceptions.md) | The exception model, `try`/`catch`/`raise`, the built-in exception catalog |
| [Host environment](host-environment.md) | Agents, params, host defaults, capability checking, tracing |
| [Grammar](grammar.md) | The collected grammar |

## Notation

Code blocks marked `agl` contain AgL source. Grammar fragments use an
EBNF-like notation: `::=` defines a production, `|` separates alternatives,
`?` marks an optional element, `*` and `+` mark repetition, and quoted
strings are literal tokens. Token names in `UPPER_CASE` refer to the lexical
tokens defined in [Lexical structure](lexical-structure.md).

Throughout the reference, "the host" refers to the runtime environment that
embeds AgL: it backs the program's declared agents, supplies program params,
executes shell commands, and records traces. Behavior marked *host-configurable*
has a documented portable default that hosts may override; everything else is
fixed by the language.

## Error model at a glance

AgL distinguishes three failure layers:

1. **Static errors** — syntax, scope, and type errors. A program with a
   static error never executes any expression and never calls any agent.
   Static checking stops at the first error.
2. **Static warnings** — advisory diagnostics (for example, a non-exhaustive
   `case` over an enum, or `on_parse_error` on a `text` target). Warnings
   never prevent execution.
3. **Runtime exceptions** — typed, catchable in-language values such as
   `AgentParseError`, `MaxIterationsExceeded`, or `RecursionError`. Uncaught
   exceptions terminate the program. See [Exceptions](exceptions.md).

Invalid host params (missing or ill-typed required params) are **host
invocation error**: it is reported before anything runs and is not catchable
in-language. See [Host environment](host-environment.md).
