# Bindings and Scope

[← Index](index.md)

AgL has three binding statements, a param declaration, and an agent
declaration. There is no bare
assignment: `x = e` as a statement is a static error with the guidance to use
`let`, `var`, or `set`. A single `=` in expression position is the equality
operator ([Expressions](expressions.md)).

## `let` — immutable binding

```ebnf
let_decl ::= "let" VAR_NAME (":" type_expr)? "=" expr
```

`let` evaluates the initializer, checks it against the annotation (if any),
and creates an **immutable** binding in the current scope:

```agl
let review: Review = reviewer[on_parse_error: retry[2]] "Review ${artifact}"
let count = 3
```

## `var` — mutable binding

```ebnf
var_decl ::= "var" VAR_NAME (":" type_expr)? "=" expr
```

Identical to `let` except the binding is **mutable** — it may later be
updated with `set`:

```agl
var artifact: text = impl "Implement ${spec}"
```

## `set` — mutation

```ebnf
set_stmt ::= "set" VAR_NAME "=" expr
```

`set` updates the nearest visible **mutable** binding. It never creates a
binding. The expected type of the right-hand side is the declared type of
the binding being updated — so an agent call on the right of `set` inherits
the variable's type as its output contract
([Agent calls](agent-calls.md)):

```agl
var proposal: Turn = researcher "Initial proposal."
set proposal = researcher "Revise proposal."   # target type: Turn
```

Static rules, all checked before execution:

1. Redeclaring a name in the same scope (by any of `let`, `var`, `param`) is
   an error.
2. `set` on an undeclared name is an error.
3. `set` on an immutable binding is an error; the diagnostic names the actual
   binder kind — a `let` binding, a `param` binding, a catch binder, or a
   pattern binding.
4. Reading a name that is not visible in the current scope chain is an error.
5. The contextual keywords `ask` and `exec` cannot be used as binding or
   param names.

## Typing of bindings

- With an annotation, the initializer is checked against the annotated type
  (the single `int → decimal` widening applies; see [Types](types.md)), and
  the binding's declared type is the annotation.
- Without an annotation, the binding's type is inferred from the
  initializer. **Untyped agent-call and `exec` results default to `text`**:

  ```agl
  let x = ask "A"           # x: text
  ```

- Empty list/dictionary literals cannot be inferred and require an
  annotation: `let items: list[Issue] = []`.

## `param` — program parameters

```ebnf
param_decl ::= "param" VAR_NAME (":" type_expr)? ("=" expr)?
```

A program declares its externally-supplied parameters with top-level `param`
declarations. A `param` is best understood as a `let` that is overridable
from outside: when the host supplies an external value for a parameter that
value is used; when no external value is supplied the default expression is
evaluated; when neither is available it is a *required-param error* (a host
invocation error; see below).

### Type rules

The four declaration forms and their type/optionality rules (O5):

| Form | Type | Required? |
| ---- | ---- | --------- |
| `param x` | `text` | yes — must be supplied externally |
| `param x: T` | `T` | yes — must be supplied externally |
| `param x = e` | inferred from `e` | no — defaults to `e` |
| `param x: T = e` | `T` | no — defaults to `e`; `e` must conform to `T` |

```agl
param spec                        # type text, required
param max_severity: int           # type int,  required
param metadata: json              # type json, required
param retries: int = 3            # type int,  optional; defaults to 3
param label = "draft"             # type text, optional; defaults to "draft"
```

### Default expressions

The default expression `e` is an ordinary expression (the full expression
language, including agent calls and arithmetic). It is evaluated **lazily
and in declaration order**, ONLY when no external value is supplied for that
parameter. A default may reference earlier `param` bindings and any earlier
root-scope bindings, but not itself or any later declaration — declaration-order
semantics are enforced statically. Self-referential or forward-referencing
defaults are static errors.

### Rules

1. `param` declarations are valid **only at the program root**; a `param`
   inside a nested block is a static error.
2. Each declared param enters the root scope as an **immutable** binding of
   the declared type, exactly like a `let`.
3. Static checking uses the declarations alone — no external values are
   needed to typecheck the program.
4. At run start, **before any statement executes**, the host validates the
   provided external values against the declarations. A required param for
   which no external value is provided, a provided value that is not declared,
   or a value that fails its declared type is a *host invocation error* — it
   is not an AgL exception and cannot be caught in-language
   ([Host environment](host-environment.md)).
5. A `text` param takes the external value verbatim. A param of any other
   type is parsed from its JSON representation **strictly** (externally
   supplied values are not chatty agent output, so no lenient recovery
   applies) and validated against the declared type.
6. Redeclaring a param name at the root is the standard redeclaration error.

Params are never mutable. To derive mutable workflow state from a param,
copy it into a `var`:

```agl
agent critic
param spec: text
var current_spec: text = spec
set current_spec = critic "Refine this spec:\n${current_spec}"
```

## `program` — program name declaration

```ebnf
program_decl ::= "program" VAR_NAME
```

A program may declare its own name with a top-level `program` declaration:

```agl
program review_workflow
```

Rules:

1. `program` is valid **only at the program root**; a `program` declaration
   inside a nested block is a static error.
2. At most **one** `program` declaration is allowed per source file; a second
   `program` declaration is a static error.
3. A `program` declaration **introduces no binding** — the name is not
   added to the scope and cannot be referenced by expressions.
4. The declared name is used by the host to identify this program (for
   example, when selecting externally-supplied param values from a
   configuration source). Its effect is entirely host-level.
5. The name must be a `VAR_NAME` (lowercase identifier).

## `agent` — declared agents

```ebnf
agent_decl ::= "agent" VAR_NAME ("=" STRING)?
```

A program declares the named agents it may call with top-level `agent`
declarations. A bare declaration names the agent; the optional `= "…"` form
attaches a *runner hint* — a host-consumed string with no effect on the
language ([Host environment](host-environment.md)):

```agl
agent reviewer
agent impl = "claude -p %{PROMPT_FILE}"
```

A declared name may then be called like any agent ([Agent calls](agent-calls.md)):

```agl
let review: Review = reviewer "Review ${artifact}"
```

Rules:

1. `agent` declarations are valid **only at the program root**; anywhere else
   is a static error (like `param`).
2. **Agent names occupy a namespace separate from variables.** Because an
   agent call is syntactically distinct (a name applied to a template), an
   `agent impl` declaration does not collide with a `let impl` or `param impl`
   binding — both may coexist, the call form selecting the agent and a
   reference selecting the value.
3. Declaring the same agent name twice is a static error.
4. The contextual keywords `ask` and `exec` are built in and cannot be
   declared as agents; `agent ask` and `agent exec` are static errors.
5. **Calling an undeclared name is a static binding error.** This is what ties
   a call to a declaration ([Agent calls](agent-calls.md)).
6. A declared agent that is never called is reported as a non-fatal
   **warning** — it never prevents execution.
7. The runner hint, when present, is a **static string literal**: it must
   contain no `${…}` interpolation. Interpolation in a runner hint is a static
   error. The language does not interpret the hint in any way.

## Lexical scoping

All binding is statically scoped. The program root is a scope, and these
constructs each introduce a nested scope:

1. each `do` loop iteration (a fresh scope per iteration),
2. each `if` branch body,
3. each `case` branch body (statement and expression forms),
4. a `try` body,
5. each `catch` body.

`let` and `var` bind in the *current* scope only; bindings never escape
their scope:

```agl
let x = "outer"

if condition =>
  let x = "inner"          # shadows outer x
  ask "Uses ${x}"          # inner
| else =>
  pass

ask "Uses ${x}"            # outer
```

Inner scopes may **shadow** outer bindings with a new `let` or `var`.
Shadowing is not mutation: when the inner scope ends, the outer binding is
unchanged.

### Mutation across scopes

`set` reaches *through* scopes to the nearest visible mutable binding, so
mutating outer state from inside a branch or loop persists:

```agl
var artifact: text = impl "Implement ${spec}"

case review of
  | Fail(issues) =>
      set artifact = impl "Fix ${issues} in ${artifact}"   # updates outer var
  | Pass =>
      pass
```

### Loop scope

A `do` body opens a fresh scope on each iteration, and the `until` condition
is evaluated **in that same iteration scope** — it can see `let` bindings
made by the body during that iteration:

```agl
do[5]
  let review: Review = reviewer "Review ${artifact}"
until review is Pass
```

`review` is visible to `until` but does not exist after the loop. State that
must survive iterations lives in an outer `var` updated with `set`. See
[Control flow](control-flow.md).

### Pattern and catch variables

Pattern variables ([Pattern matching](pattern-matching.md)) and `catch`
binders ([Exceptions](exceptions.md)) are immutable and scoped to their
branch or handler body:

```agl
case review of
  | Fail(issues) => ask "${issues}"
  | Pass => pass

ask "${issues}"   # static error: 'issues' is not defined
```
