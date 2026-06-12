# Control Flow

[← Index](index.md)

AgL has three control-flow statements: `if`, the `case` statement, and the
bounded post-test loop `do … until`. (Exception control flow — `try`,
`catch`, `raise` — is covered in [Exceptions](exceptions.md).)

## `if`

```ebnf
if_stmt   ::= "if" if_branch ("|" if_branch)*
if_branch ::= bar_safe_expr "=>" branch_body
            | "else" "=>" branch_body
```

Every branch — including subsequent conditions — is introduced by `|`, and
the `else` branch, if present, must be last (checked statically):

```agl
if code is Fail or design is Fail =>
  set artifact = impl "Fix valid issues:\n${code}\n${design}"
| else =>
  pass
```

Inline:

```agl
if status is Complete => pass | status is Blocked => print status | else => pass
```

Semantics:

1. Conditions are evaluated left to right; each must be `bool` (statically
   checked — there is no truthiness).
2. The first true condition's body executes in a fresh branch scope; the
   rest are skipped.
3. With no matching condition, `else` runs if present; otherwise the
   statement does nothing.

Branch bodies are suites, single bar-safe statements, or `try` statements
([Program structure](program-structure.md)). Because conditions are bar-safe
positions, a `case` expression used as a condition must be parenthesized.
Mutating an outer `var` with `set` inside a branch persists after the `if`.

## `case` statement

```ebnf
case_stmt ::= "case" expr "of" ("|" pattern "=>" branch_body)+
```

```agl
case result of
  | Complete(output) => set artifact = output
  | Changed(output) => set artifact = output
  | Blocked(reason) => raise Abort(message: reason)
  | _ => pass
```

The scrutinee is evaluated once; patterns are tried in order; the first
match's body runs in a fresh branch scope with the pattern's bindings; if
nothing matches, `MatchError` is raised. Pattern forms, static checks, and
the non-exhaustiveness warning are described in
[Pattern matching](pattern-matching.md).

A `case` whose branches are single *expressions* can also be used in
expression position — see [Expressions](expressions.md). At statement level
a bare `case` is always the statement form.

## `do … until` loops

```ebnf
do_until   ::= "do" ("[" INT "]")? do_body "until" bar_safe_expr
```

The only loop in AgL is a **bounded, post-test** loop. The bound is part of
`do`, written immediately after it:

```agl
do[5]
  let status: Status = prompt[on_parse_error: retry[2]] "Do X."
until status is Complete
```

Inline:

```agl
do[5] let status: Status = prompt "Do X." until status is Complete
```

Semantics:

1. Execute the body in a **fresh iteration scope**.
2. Evaluate the `until` condition *in that same iteration scope* — it sees
   the body's bindings for that iteration. The condition must be `bool`.
3. If the condition is true, the loop ends.
4. Otherwise discard the iteration scope and repeat.
5. If the body has executed `N` times (the bound) and the condition is still
   false, raise **`MaxIterationsExceeded`**.

The bound:

- must be a positive integer literal (`do[0]` is a syntax error:
  *"Loop bound must be a positive integer"*);
- counts **body executions**, not agent calls — parse retries inside the
  body are additional agent calls but do not consume loop iterations;
- when omitted, the host's default bound applies (portable default: **5**).
  `do` without a bound does *not* mean unbounded — unbounded loops do not
  exist in v1.

Because the condition is post-tested, the body always executes at least once
when control reaches the loop.

The `MaxIterationsExceeded` exception carries the bound (`limit`), the
source text of the `until` condition (`condition`), and the condition's
final value (`last_condition_value`) — see [Exceptions](exceptions.md).

### Loop state

Iteration-local `let`/`var` bindings do not survive to the next iteration or
past the loop. Persistent state lives in an outer `var` mutated with `set`:

```agl
var artifact: text = impl "Implement ${spec}"

do[5]
  let review: Review = reviewer "Review ${artifact}"
  case review of
    | Fail(issues) => set artifact = impl "Fix ${issues}"
    | Pass => pass
until review is Pass
```

The inline body forms (closed statements separated by `;`, optionally ending
in a single open statement sealed by `until`) are specified in
[Program structure](program-structure.md). The `until` keyword may start its
own line aligned with `do`, courtesy of the branch-marker continuation rule
([Lexical structure](lexical-structure.md)).
