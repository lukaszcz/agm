# AgL Language Reference

AgL (Agent Language) is a small, statically checked DSL for writing
**agent workflows**: programs that call AI agents with typed output contracts,
chain prompt results, run review/fix/refine loops, and branch on structured
agent outcomes.

AgL is not a general-purpose programming language. It is a typed, expression-
oriented orchestration language whose core ideas are:

- **Agent calls are first-class expressions.** `reviewer.ask("Review %{artifact}")`
  calls a host-provided agent with a rendered prompt template; the result is
  a typed value usable in any expression position.
- **Types are contracts at the LLM boundary.** Annotating an `ask` call's
  result type drives the format instructions sent to the agent, the parsing
  and validation of its raw output, and the retry-or-abort behavior on
  malformed output.
- **Structured outputs have one canonical wire format** — JSON, with a
  reserved `"$case"` tag holding a member's effective JSON name (its terminal
  name unless renamed) — parsed leniently by default and validated strictly,
  always.
- **Everything is an expression.** There is no statement category: binders
  (`let`/`var`) scope over a continuation, side-effecting forms have type
  `unit` and return `()`, and `if`/`case`/`try` with matching branches yield
  a typed value.
- **Functions are first-class.** User-defined `def` declarations and `fn`
  lambdas produce values of function type `A -> B` or `(A, B) -> C`; they may be stored,
  passed, and returned.
- **Mutability is explicit.** `let` binds a name immutably and `var` binds it
  mutably; records and enum-member records may also mark individual fields
  `var`. `:=` updates a mutable binding, container element, or `var` field in
  place. Binding never copies, so every alias observes mutable arrays,
  dictionaries, and record fields. Equality is `==`, while a single `=` is a
  binder and named-argument separator, never assignment.
- **Failures are exceptions.** Parse failures, cast failures, loop exhaustion,
  explicitly raised match errors, recursion depth, and shell errors are typed,
  catchable exceptions.
- **Blocks use indentation**, with deterministic inline forms for one-liner
  workflows.

A taste of the language:

```agl
enum Review
  | Pass
  | Fail(issues: array[text])

let reviewer = AgentCommand("reviewer")
let impl = AgentCommand("impl")

def review-and-fix(artifact: text) -> text =
  let r: Review = reviewer.ask(
    "Review the artifact for correctness:\n%{artifact}",
    on-parse-error = Retry(n = 2)
  )
  case r of
    | Pass => artifact
    | Fail(issues) => impl.ask(
        "Fix these issues:\n%{issues}\n\nCurrent:\n%{artifact}"
      )

program def main(spec: text) -> unit =
  var artifact: text = impl.ask("Implement %{spec}")

  do[5]
    artifact := review-and-fix(artifact)
    let final: Review = reviewer.ask("Final review:\n%{artifact}")
  until final is Pass
```

## Chapters

| Chapter | Contents |
| ------- | -------- |
| [Lexical structure](lexical-structure.md) | Source text, comments, indentation and layout, keywords, tokens, declaration attributes, operator precedence |
| [Program structure](program-structure.md) | Modules, `program def` entry points and their parameters, items, binders, inline forms |
| [Types](types.md) | Built-in types (`unit`, `text`, `int`, `decimal`, `bool`, `json`, function types), `record`/`enum`/`type` declarations, the library types the language itself names (`ExecResult`, `ParsePolicy`, `Agent`, `AgentRequest`, `SessionTransport`, `Session`, `SessionStats`), assignability, casts and convertibility (`as`/`as?`), mutable record fields and reference semantics, cycles, copying (`copy`/`shallow-copy`), and parsing (`parse`/`try-parse`) |
| [Bindings and scope](bindings-and-scope.md) | `let`, `var`, `:=`, `builtin var`, `def`, lexical scoping, shadowing |
| [Expressions](expressions.md) | Literals, constructors, calls, operators, `as`/`as?` cast operators, `render`, JSON parsing, `case`/`if` expressions, `unit`-typed forms, expected-type propagation |
| [Strings and interpolation](strings-and-interpolation.md) | Templates, escapes, `%{…}` interpolation, uniform rendering rules |
| [Functions](functions.md) | `def` declarations, `fn` lambdas, optional/named arguments, function types, first-class values, recursion and depth limit |
| [Control flow](control-flow.md) | `if`, `case`, unified loops (`for`/`while`/`do`/`until`/`done`), `break`, `continue` |
| [Pattern matching](pattern-matching.md) | Patterns, source priority, exhaustiveness, redundancy |
| [Generics](generics.md) | Type parameters on `def`/`record`/`enum`/`type`, type application, inference and `::[…]` override, generic constructor values, strict parametricity, invariance, erasure |
| [Agent calls](agent-calls.md) | `ask`, agents as values, call options, output contracts, the JSON wire format, parse policies and retries |
| [Shell execution](shell-execution.md) | `exec`, the `ExecResult` structured form vs the parsed form, `ExecError` |
| [Exceptions](exceptions.md) | The exception model, `try`/`catch`/`raise`, the built-in exception catalog |
| [Attributes](attributes.md) | `@name` declaration attributes: placement, catalog, zones, `@doc`, `@extern-name`, command registration, program parameter options |
| [Modules](modules.md) | File-based module system: module identity, import forms, qualified access, visibility, cyclic imports, REPL imports, the standard-library module inventory |
| [Named scopes](scopes.md) | Nestable declaration namespaces, qualifier paths, visibility, and `use` |
| [Packages](packages.md) | Package module trees and identity, package-qualified paths, import visibility, programs as commands, params and qualified configuration keys, resources and companions |
| [Python FFI](ffi.md) | `extern def`, the companion Python file, value mapping across the boundary, `ExternError`, interpreter-local companion state, the trace hook |
| [Host environment](host-environment.md) | Agents, program arguments, host defaults, capability checking, tracing |
| [Grammar](grammar.md) | The collected grammar |

This reference describes the language. It names a library type or function only
where the language's own semantics depend on it; the `std/*` modules themselves
are documented by their sources.

## Notation

Code blocks marked `agl` contain AgL source. Grammar fragments use an
EBNF-like notation: `::=` defines a production, `|` separates alternatives,
`?` marks an optional element, `*` and `+` mark repetition, and quoted
strings are literal tokens. Token names in `UPPER_CASE` refer to the lexical
tokens defined in [Lexical structure](lexical-structure.md).

Throughout the reference, "the host" refers to the runtime environment that
embeds AgL: it dispatches selected `Agent` values, supplies program arguments,
executes shell commands, and records traces. Behavior marked *host-configurable*
has a documented portable default that hosts may override; everything else is
fixed by the language.

## Error model at a glance

AgL distinguishes three failure layers:

1. **Static errors** — syntax, scope, type, case-exhaustiveness, and
   case-redundancy errors. A program with a static error never executes any
   expression and never calls any agent.
2. **Static warnings** — advisory diagnostics such as `on-parse-error` on a
   `text` target. Warnings never prevent execution.
3. **Runtime exceptions** — typed, catchable in-language values such as
   `AgentParseError`, `MaxIterationsExceeded`, or `RecursionError`. Uncaught
   exceptions terminate the program. See [Exceptions](exceptions.md).

A missing required program argument, or a supplied value that does not
decode to its declared type, is a **host invocation error**: reported before
anything runs and not catchable in-language. See
[Host environment](host-environment.md).
