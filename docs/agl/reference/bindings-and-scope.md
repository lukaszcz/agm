# Bindings and Scope

[← Index](index.md)

AgL has three binding statements, an input declaration, and an agent
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

1. Redeclaring a name in the same scope (by any of `let`, `var`, `input`) is
   an error.
2. `set` on an undeclared name is an error.
3. `set` on an immutable binding is an error; the diagnostic names the actual
   binder kind — a `let` binding, an `input` binding, a catch binder, or a
   pattern binding.
4. Reading a name that is not visible in the current scope chain is an error.
5. The contextual keywords `ask` and `exec` cannot be used as binding or
   input names.

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

## `input` — declared host inputs

```ebnf
input_decl ::= "input" VAR_NAME (":" type_expr)?
```

A program declares the inputs it expects from the host with top-level
`input` declarations. The annotation defaults to `text`:

```agl
input spec                 # same as: input spec: text
input max_severity: int
input metadata: json
```

Rules:

1. `input` declarations are valid **only at the program root**; anywhere else
   is a static error.
2. Each declared input enters the root scope as an **immutable** binding of
   the declared type.
3. Static checking uses the declarations alone — no host values are needed to
   typecheck the program.
4. At run start, **before any agent call**, the host validates the provided
   inputs against the declarations. A missing declared input, a
   provided-but-undeclared input, or a type-invalid value is a *host
   invocation error* — it is not an AgL exception and cannot be caught
   in-language ([Host environment](host-environment.md)).
5. A `text` input takes the host value verbatim. An input of any other type
   is parsed from JSON **strictly** (the value must be exactly one valid JSON
   value — no lenient recovery) and validated against the declared type.
6. Redeclaring an input name at the root is the standard redeclaration error.

Inputs are never mutable. To derive mutable workflow state from an input,
copy it into a `var`:

```agl
agent critic
input spec: text
var current_spec: text = spec
set current_spec = critic "Refine this spec:\n${current_spec}"
```

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
   is a static error (like `input`).
2. **Agent names occupy a namespace separate from variables.** Because an
   agent call is syntactically distinct (a name applied to a template), an
   `agent impl` declaration does not collide with a `let impl` or `input impl`
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

if
| condition =>
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
