# Control Flow

[← Index](index.md)

Control flow in AgL is expression-oriented: every construct produces a
value. `if` and `case` with branches unify to a common type; loops and
else-less `if` produce `unit`. Exception control flow — `try`, `catch`,
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

## Loops

```ebnf
loop_expr    ::= for_clause? while_clause? "do" ("[" expr "]")? block loop_end
               | for_clause? while_clause? "do" ("[" expr "]")? inline_body "until" or_expr

for_clause   ::= "for" NAME "in" or_expr range_tail? NEWLINE?
range_tail   ::= ("to" | "downto") or_expr ("by" or_expr)?

while_clause ::= "while" or_expr NEWLINE?

loop_end     ::= "until" or_expr
               | "done"
               | (* omitted — suite body only; equivalent to "done" *)
```

A loop body executes repeatedly. All three header clauses — `for`, `while`,
and the `[bound]` — are optional and may be combined. The terminator (`until
E`, `done`, or omitted) follows the body.

### `for` clause

`for NAME in EXPR` declares a loop variable `NAME` bound to successive elements
of the collection `EXPR`:

- **`list[T]`** — elements in declaration order; `NAME` has type `T`.
- **`dict[text, V]`** — keys in insertion order; `NAME` has type `text`. Values
  are retrieved from the dict with the key: `d[NAME]`.
- **`text`** — UTF-8 characters in order; `NAME` has type `text`.

The loop variable is immutable (`:=` to it is a static error), has the type of
the collection's element, and is visible in the `while` guard, the body, and
the `until` condition. It is not visible outside the loop.

The loop exits after the last element is consumed without raising any exception.

```agl
var total: int = 0
for item in items do
  total := total + item
done

for key in settings do
  print "Key: ${key} = ${settings[key]}"
done
```

### Range `for`

When a `range_tail` follows the start expression, the `for` clause iterates an
`int` counter instead of a collection. There are four forms:

```agl
for i in a to b do … done           # i = a, a+1, …, b        (inclusive)
for i in a to b by k do … done      # i = a, a+k, …  (≤ b)
for i in a downto b do … done       # i = a, a-1, …, b        (inclusive)
for i in a downto b by k do … done  # i = a, a-k, …  (≥ b)
```

Semantics:

- **Direction** comes from the keyword: `to` iterates upward; `downto` iterates
  downward.
- **Step** (`by k`) must be a **positive** `int` (default `1`). A non-positive
  step raises [`RangeError`](exceptions.md#rangeerror) once at loop entry,
  before the body runs. A literal non-positive step is also a static error.
- **Bounds are inclusive**: `1 to 3` yields `1, 2, 3`; `3 downto 1` yields
  `3, 2, 1`. With a step, the last value is the furthest one not past `b`:
  `1 to 6 by 2` yields `1, 3, 5`.
- **Degenerate range**: if `a > b` for `to`, or `a < b` for `downto`, the
  body runs zero times and the loop completes normally.
- **Bounds and step are evaluated once at loop entry**, in the enclosing scope
  and in source order (`a`, then `b`, then `k`). They cannot reference the
  loop variable.
- The loop variable has type **`int`**, is **immutable** (`:=` to it is a
  static error), and is visible in the `while` guard, the body, and the `until`
  condition — identical rules to the collection `for`.

The range `for` composes with `while`, `[n]`, `until`/`done`, `break`/
`continue`, and nesting exactly like the collection `for`. The loop variable
is not visible outside the loop.

```agl
# Sum 1 to n
var total: int = 0
for i in 1 to n do
  total := total + i
done

# Countdown by threes
for i in 9 downto 1 by 3 do
  print "${i}"
done
# prints: 9, 6, 3
```

### `while` clause

`while EXPR` is a pre-body guard: if `EXPR` (which must be `bool`) evaluates
to `false`, the loop exits immediately without running the body. It is
re-evaluated at the start of each iteration, after the `for` variable has been
bound (if a `for` clause is also present).

```agl
var n: int = 0
while n < limit do
  n := n + 1
done
```

### `for`+`while` combination

When both clauses are present, each iteration:
1. Checks whether the collection has a next element; exits if not.
2. Binds the loop variable to that element.
3. Evaluates the `while` guard; exits if false.
4. Runs the body.

```agl
for x in candidates while score(x) > threshold do
  process(x)
done
```

### `do` and the body

The `do` keyword introduces the body. An optional bound `[expr]` may appear
immediately after `do`:

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

Inline body (single expression; no INDENT):

```agl
do[5] let r: Review = ask("Review ${a}", agent: reviewer) until r is Pass
```

### Bound `[expr]`

The bound is an arbitrary `int`-typed expression evaluated **once at loop
entry, in the enclosing scope**:

- **N ≤ 0** — runs the body zero times; the loop completes normally.
- **N ≥ 1** — when the body has executed N times and the exit condition has
  not yet triggered, **`MaxIterationsExceeded`** is raised.
- **Absent** — the loop is unbounded and never raises `MaxIterationsExceeded`.

The bound counts body executions, not agent calls. Retries inside the body do
not consume iterations. Mutating a `var` the bound references from inside the
body does not change the already-fixed bound. Arithmetic is equally valid:
`do[rounds + 1]`, `do[base * 2]`.

### Terminators

**`until EXPR`** — after each body execution, evaluate `EXPR` (must be
`bool`) in the iteration scope; if true, exit the loop normally.

**`done`** — equivalent to `until false`; the loop only exits via `break`,
the bound, or for-collection exhaustion.

**Omitted terminator** — when a multi-line `do` body is written without an
explicit `until` or `done`, the dedent to the enclosing indent level implicitly
closes the loop as if `done` were present.

```agl
# all three forms are equivalent:
do[n]
  artifact := ask("Implement ${spec}", agent: impl)
done

do[n]
  artifact := ask("Implement ${spec}", agent: impl)
until false

do[n]
  artifact := ask("Implement ${spec}", agent: impl)
# (next statement at enclosing indent — implicit done)
```

### Stacked header form

Each header clause may appear on its own line above `do`:

```agl
for x in candidates
while score(x) > threshold
do[max_rounds]
  process(x)
until satisfied(x)
```

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

The `until` and `done` keywords may start their own lines aligned with the
enclosing statement, courtesy of the branch-marker continuation rule
([Lexical structure](lexical-structure.md)).

All loops have type **`unit`** — they run for effect and produce no value.

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
