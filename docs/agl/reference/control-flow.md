# Control Flow

[← Index](index.md)

Control flow in AgL is expression-oriented: every construct produces a
value. `if` and `case` with branches unify to a common type; `do … until`
and else-less `if` produce `unit`. Exception control flow — `try`, `catch`,
`raise` — is covered in [Exceptions](exceptions.md).

## `if`

```ebnf
if_expr        ::= "if" "|"? if_cond_branch ("|" if_cond_branch)* if_else_branch?
if_cond_branch ::= or_expr "=>" branch_body
if_else_branch ::= "|"? "else" "=>" branch_body
```

An optional leading `|` after `if` is accepted, making all branches —
including the first condition branch — introduced by `|`. The `else` branch,
if present, must be last; its own `|` is optional:

```agl
if code is Fail or design is Fail =>
  artifact := ask("Fix issues:\n${code}\n${design}", agent = impl)
else =>
  ()
```

Inline:

```agl
if | status is Complete => () | status is Blocked => print status | else => ()
```

Semantics:

1. Conditions are evaluated left to right; each must be `bool`.
2. The first true condition's body executes in a fresh branch scope.
3. With no matching condition, `else` runs if present; otherwise the
   expression yields `unit`.

### `if` with `else`: a value-producing expression

When an `else` branch is present **and** all branches have a common type `T`
(with `int → decimal` widening), the `if` expression has type `T`:

```agl
let label: text = if | score > 90 => "A" | score > 75 => "B" | else => "C"

print(if status is Pass => "passed" else => "failed")
```

### `if` without `else`: type `unit`

When `else` is absent, the `if` expression has type `unit`. All branch bodies
must also have type `unit`:

```agl
if res.exit_code != 0 =>
  print "command failed: ${res.stderr}"
```

Because an `else`-less `if` always yields `unit`, it is for effectful control
flow. A branch body that produces a non-`unit` value is a static error.

Branch bodies are suites (indented blocks) or single expressions at the
`or_expr` level. Because branch bodies are `or_expr` positions, a `case` or
`if` expression used inline must be parenthesized. Mutating an outer `var`
with `:=` inside a branch persists after the `if`.

## `case`

```ebnf
case_expr        ::= "case" expr "of" "|"? case_branch ("|" case_branch)*
case_branch      ::= pattern "=>" branch_body
```

```agl
case result of
  Complete(output) => artifact := output
  | Changed(output) => artifact := output
  | Blocked(reason) => raise Abort(message = reason)
  | _ => ()
```

The scrutinee is evaluated once; patterns are tried in order; the first
match's body runs in a fresh branch scope with the pattern's bindings; if
nothing matches, `MatchError` is raised.

When all branches have a common type `T`, the `case` expression has type `T`.
When used in a position where `|` would be ambiguous, a `case` expression
must be parenthesized.

Pattern forms, static checks, and the non-exhaustiveness warning are
described in [Pattern matching](pattern-matching.md).

### `case` as a value-producing expression

```agl
let next_prompt: text = case action of
  | Stop => "Stop."
  | Continue(prompt) => prompt
  | Escalate(reason) => "Investigate:\n${reason}"
```

## `do … until` loops

```ebnf
do_until   ::= "do" ("[" INT "]")? block "until" or_expr
```

The only loop in AgL is a **bounded, post-test** loop. The bound is written
immediately after `do`:

```agl
do[5]
  let r: Review = ask(
    "Review ${artifact}",
    agent = reviewer,
    on_parse_error = Retry(n = 2)
  )
  case r of
    | Fail(issues) =>
        artifact := ask("Fix ${issues}", agent = impl)
    | Pass => ()
until r is Pass
```

Inline:

```agl
do[5] let r: Review = ask("Review ${a}", agent = reviewer) until r is Pass
```

Semantics:

1. Execute the body in a **fresh iteration scope**.
2. Evaluate the `until` condition *in that same iteration scope* — it sees
   the body's bindings for that iteration. The condition must be `bool`.
3. If the condition is true, the loop ends.
4. Otherwise discard the iteration scope and repeat.
5. If the body has executed `N` times and the condition is still false, raise
   **`MaxIterationsExceeded`**.

The `do … until` loop always has type **`unit`**: it runs for effect and
produces no value.

The bound:

- must be a positive integer literal;
- counts **body executions**, not agent calls — retries inside the body do
  not consume loop iterations;
- when omitted, the host's default bound applies (portable default: **5**).
  `do` without a bound does *not* mean unbounded — AgL has no unbounded loops.

Because the condition is post-tested, the body always executes at least once.

### Loop state

Iteration-local bindings do not survive to the next iteration or past the
loop. Persistent state lives in an outer `var` mutated with `:=`:

```agl
var artifact: text = ask("Implement ${spec}", agent = impl)

do[5]
  let review: Review = ask("Review ${artifact}", agent = reviewer)
  case review of
    | Fail(issues) =>
        artifact := ask("Fix ${issues}", agent = impl)
    | Pass => ()
until review is Pass
```

The `until` keyword may start its own line aligned with `do`, courtesy of the
branch-marker continuation rule
([Lexical structure](lexical-structure.md)).
