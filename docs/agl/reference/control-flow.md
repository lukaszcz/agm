# Control Flow

[← Index](index.md)

Control flow in AgL is expression-oriented: every construct produces a
value. `if` and `case` with branches unify to a common type; loops and
else-less `if` have type `unit` and return `void`. Exception control flow —
`try`, `catch`, `raise` — is covered in [Exceptions](exceptions.md).

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
   expression returns `void`.

### `if` with `else`: a value-producing expression

When an `else` branch is present **and** all branches have a common type `T`
(with `int → decimal` widening), the `if` expression has type `T`:

```agl
let label: text = if | score > 90 => "A" | score > 75 => "B" | else => "C"

print(if status is Pass => "passed" else => "failed")
```

### `if` without `else`: type `unit`

When `else` is absent, the `if` expression has type `unit` and returns `void`.
All branch bodies must also have type `unit`:

```agl
if res.exit_code != 0 =>
  print "command failed: ${res.stderr}"
```

Because an `else`-less `if` always has type `unit`, it is for effectful
control flow. A branch body that produces a non-`unit` value is a static error.

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

## Loops

AgL has one uniform loop construct covering every loop shape — `for`,
`while`, `do`, a `[n]` bound, and `until`/`done` termination — plus
`break`/`continue`. The surface form composes a fixed-ordered header with
an optional bound and a body:

```ebnf
loop        ::= for_clause? while_clause? "do" loop_bound? body loop_end
for_clause  ::= "for" name "in" or_expr range_tail? _NL?
range_tail  ::= ("to" | "downto") or_expr ("by" or_expr)?
while_clause::= "while" or_expr _NL?
loop_bound  ::= "[" or_expr "]"
loop_end    ::= "until" or_expr | "done"
body        ::= suite | inline_seq
```

A loop may begin with **at most one `for`** clause and **at most one
`while`** clause, in that fixed order (so a `while` guard can always
reference the `for` variable). `do` is always present and may carry an
optional iteration **bound** `[n]`. The body is an indented suite or an
inline `;`-separated sequence. It ends with `until E`, `done`, or — in the
**multi-line** form only — nothing (the body is delimited by indentation).
`done` and an omitted terminator are equivalent to `until false`.

```agl
# collection for, terminated by `done`
for x in items do
  process(x)
done

# integer range: i = a, a+step, … up/down to and including b
for i in 1 to n do print i done
for i in n downto 1 by 2 do print i done

# while with an explicit bound (safety limit)
var i: int = 0
while i < target do[1000]
  i := i + 1
done

# bare post-test loop
var r: Review = ask("Review ${a}", agent = reviewer)
do
  case r of
    | Fail(issues) => r := ask("Fix ${issues}", agent = impl)
    | Pass => break
until r is Pass
```

Every loop expression has type **`unit`** — it runs for effect and returns
`void`.

### Clause semantics

**`for` — collection iteration.** `for x in COLLECTION` iterates `list[T]`
(elements, in order), `dict[text, V]` (keys, in dict order), or `text`
(each character as a length-1 `text`). The loop variable `x` takes the
element/key/char type respectively.

**`for` — integer range.** `for i in a to b` runs `i = a, a+1, …, b`
(inclusive); `for i in a downto b` runs `i = a, a-1, …, b` (inclusive).
An optional `by k` sets the step (a positive `int`); an omitted step is `1`.
Bounds and step are each evaluated exactly once at loop entry, in source
order (`a`, then `b`, then `k`). A degenerate range (`a > b` for `to`,
`a < b` for `downto`) runs the body zero times and completes normally. A
non-positive step raises **`RangeError`** at loop entry.

**`while E`.** Before each body execution, `E` is evaluated; if false the
loop exits. `E` must be `bool`. (Placed after `for` so it may reference
the `for` variable.)

**`[n]` bound.** An `int` expression evaluated once at loop entry, in the
enclosing scope. It counts **body executions**. Exceeding it raises
**`MaxIterationsExceeded`**. `n ≤ 0` runs the body **zero times** and
completes normally (even though a positive-bound loop is post-test and
always runs at least once). An omitted bound is unbounded from the loop's
own machinery. The bound's counter and raise machinery exist only when a
`[n]` bound is present.

**`until E`.** After each body execution, `E` is evaluated in the iteration
scope (it sees the body's bindings); if true the loop exits. `E` must be
`bool`. `done` and an omitted terminator mean `until false` — so a
`do[n] … done` loop runs the body `n` times and then **raises**
`MaxIterationsExceeded` (the bound is the only real exit).

The loop's own exception-free exits are: `for` exhaustion, a false `while`,
a true `until`, and `break`. A `return` inside the loop exits the enclosing
function instead of producing a loop value.

### Iteration order and scope

The `for` variable is a fresh **immutable** binding each iteration
(assigning it with `:=` is a static error). It is in scope in the `while`
guard, the body, and the `until` condition. It is **not** in scope in the
`for` collection/range expressions or the `[n]` bound (all evaluated once at
entry, in the enclosing scope), nor after the loop. Body bindings are
visible to the `until` condition but do not survive to the next iteration
or past the loop. Persistent state lives in an outer `var` mutated with `:=`.

### `break` and `continue`

`break` exits the innermost enclosing loop immediately; `continue` skips the
remainder of the current iteration's body and starts the next iteration
(re-running the `for` advance, the `while`/`until` guards, and the bound
check — a continued iteration still advances the iterator and still counts
toward the bound). Both are nullary expressions of the **bottom type**:
assignable to any expected type and usable in any branch position (like
`raise`).

```agl
for x in items do
  if x is Skip => continue
  if x is Stop => break
  process(x)
done

# bottom type: `break` in a typed branch
let v: int = if done => break else => count
```

`break`/`continue` are valid only **lexically inside a loop body within the
same function/lambda**: one outside any loop, or one that would cross a
`fn`/`def`/lambda boundary into an outer loop, is a static error. `break`
inside a `try` body exits the loop (it is not caught by `catch`, which
handles only AgL exceptions).

A `return` inside a loop unwinds past the loop and returns from the nearest
enclosing `def` or `fn`. It is not a `break`: the loop does not produce its
usual `unit` result, and enclosing loops are abandoned as well.

### The host `max-iters` safety valve

A `[n]` bound is the loop's own termination machinery. Loops with a `for`
clause are bounded by a finite collection. Both are **self-bounded** and are
never cut short by the host. The host's `max-iters` setting
(`--max-iters` / `[exec] max-iters` / `config max-iters`) is a **safety
valve** that applies **only to unbounded loops** — those with no `[n]` bound
and no `for` clause (a bare `while … do … done` or `do … until E`). It
caps such loops at `max-iters` body executions, raising
`MaxIterationsExceeded`. The valve is **off by default** (unbounded loops run
until they self-terminate); setting `max-iters` turns it on.

The `until` keyword (and `done`) may start its own line aligned with `do`,
courtesy of the branch-marker continuation rule
([Lexical structure](lexical-structure.md)).
