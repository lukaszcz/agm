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
  artifact := ask("Fix issues:\n${code}\n${design}", agent: impl)
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
  | Blocked(reason) => raise Abort(message: reason)
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
do_loop    ::= "do" ("[" expr "]")? block ("until" or_expr | "done")
```

AgL's loop is a **post-test** loop. An optional bound is written immediately
after `do`:

```agl
do[5]
  let r: Review = ask(
    "Review ${artifact}",
    agent: reviewer,
    on_parse_error: Retry(n: 2)
  )
  case r of
    | Fail(issues) =>
        artifact := ask("Fix ${issues}", agent: impl)
    | Pass => ()
until r is Pass
```

Inline:

```agl
do[5] let r: Review = ask("Review ${a}", agent: reviewer) until r is Pass
```

Semantics:

1. If a bound `[expr]` is present, evaluate `expr` (an `int`-typed expression)
   **once in the enclosing scope** to obtain the bound `N`. A bound `N ≤ 0`
   runs the body **zero times** and the loop completes normally (yields `unit`).
2. Execute the body in a **fresh iteration scope**.
3. Evaluate the `until` condition *in that same iteration scope* — it sees
   the body's bindings for that iteration. The condition must be `bool`.
4. If the condition is true, the loop ends.
5. Otherwise discard the iteration scope and repeat.
6. If a bound `N ≥ 1` was given and the body has executed `N` times with the
   condition still false, raise **`MaxIterationsExceeded`**. An unbounded loop
   (`do` without `[…]`) never raises `MaxIterationsExceeded`.

The `do … until` loop always has type **`unit`**: it runs for effect and
produces no value.

### `done` terminator

`done` is an alternative terminator equivalent to `until false` — the loop
never exits on its own. This form is only useful with a bound `[n]`: when the
bound is exhausted the loop raises `MaxIterationsExceeded`; with no bound it
runs forever.

```agl
do[max_rounds]
  artifact := ask("Implement ${spec}", agent: impl)
done
```

When a multi-line `do` body is written without an explicit `until` or `done`,
it is implicitly terminated as if `done` were written — so the two forms are
identical:

```agl
# identical to the example above
do[max_rounds]
  artifact := ask("Implement ${spec}", agent: impl)
```

The bound:

- is an arbitrary `int`-typed expression — it may reference params, `let`s,
  `var`s, arithmetic, or calls. It is evaluated **once at loop entry, in the
  enclosing scope**: it cannot see the body's bindings, and mutating a `var`
  the bound references from inside the body does not change the already-fixed
  bound;
- counts **body executions**, not agent calls — retries inside the body do
  not consume loop iterations;
- when omitted, the loop is **unbounded** — it runs until the `until`
  condition holds and never raises `MaxIterationsExceeded`.

A bound ≤ 0 runs the body zero times and completes normally. For a positive
bound or an omitted bound, the body always executes at least once.

### Expression bound example

```agl
param max_rounds: int

var artifact: text = ask("Implement ${spec}", agent: impl)

do[max_rounds]
  let r: Review = ask("Review ${artifact}", agent: reviewer)
  case r of
    | Fail(issues) =>
        artifact := ask("Fix ${issues}", agent: impl)
    | Pass => ()
until r is Pass
```

The bound `max_rounds` is a param, resolved before the loop begins. Arithmetic
is equally valid: `do[rounds + 1]`, `do[base * 2]`.

### Loop state

Iteration-local bindings do not survive to the next iteration or past the
loop. Persistent state lives in an outer `var` mutated with `:=`:

```agl
var artifact: text = ask("Implement ${spec}", agent: impl)

do[5]
  let review: Review = ask("Review ${artifact}", agent: reviewer)
  case review of
    | Fail(issues) =>
        artifact := ask("Fix ${issues}", agent: impl)
    | Pass => ()
until review is Pass
```

The `until` keyword may start its own line aligned with `do`, courtesy of the
branch-marker continuation rule
([Lexical structure](lexical-structure.md)).

## `break` and `continue`

`break` and `continue` provide early loop-exit and iteration-skip control.
Both are expressions with **bottom type** and may appear in any expression
position inside a loop body or `until` condition:

```agl
do
  let line: text = ask("Enter a value (or 'quit'):", agent: user)
  if line = "quit" => break
  print "Got: ${line}"
done
```

`break` exits the innermost enclosing loop immediately. The loop expression
then completes normally and yields `unit`.

`continue` ends the current iteration, skipping the remainder of the loop body
— including the `until` condition — and immediately restarts the body from the
top (re-running the bound check and any header guards).

```agl
do
  let n: int = ask("Enter a positive int:", agent: user)
  if n <= 0 => continue
  process(n)
until done_processing(n)
```

Using `break` or `continue` outside any enclosing loop, or inside a `fn` or
`lambda` definition nested inside a loop, is a static error.
