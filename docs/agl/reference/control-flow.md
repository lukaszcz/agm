# Control Flow

[← Index](index.md)

AgL has three control-flow statements: `if`, the `case` statement, and the
bounded post-test loop `do … until`. (Exception control flow — `try`,
`catch`, `raise` — is covered in [Exceptions](exceptions.md).)

## `if`

```ebnf
if_stmt   ::= "if" "|"? if_branch ("|" if_branch)*
if_branch ::= bar_safe_expr "=>" branch_body
            | "else" "=>" branch_body   (* must be last, if present *)
```

An optional leading `|` after `if` is accepted, making all branches —
including the first — introduced by `|`. The leading-pipe form is preferred
in docs whenever an `if` has more than one branch; single-branch `if` stays
pipe-less. The `else` branch, if present, must be last (checked statically):

```agl
if
| code is Fail or design is Fail =>
  set artifact = impl "Fix valid issues:\n${code}\n${design}"
| else =>
  pass
```

Inline:

```agl
if | status is Complete => pass | status is Blocked => print status | else => pass
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
positions, a `case` or `if` expression used as a condition must be
parenthesized. Mutating an outer `var` with `set` inside a branch persists
after the `if`.

## `if` expressions

An `if` can also be used where a value is expected:

```ebnf
if_expr        ::= "if" "|"? if_expr_branch ("|" if_expr_branch)*
if_expr_branch ::= bar_safe_expr "=>" bar_safe_expr
                 | "else" "=>" bar_safe_expr   (* must be last *)
```

```agl
let status_text: text = if | code is Fail => "failed" | else => "ok"

print (if | score > 90 => "A" | score > 75 => "B" | else => "C")
```

Rules:

1. **`else` is required** — an `if` expression without an `else` branch is a
   static type error. This ensures every `if` expression always yields a
   value. (The statement form imposes no such requirement.)
2. **Branch types must agree** — all branch result expressions must share one
   type, with `int → decimal` widening as the only coercion, exactly as for
   `case` expressions.
3. **Parenthesization in bar-safe positions** — in bar-safe positions (branch
   bodies, `if`/`until` conditions) an unparenthesized `if` expression is
   rejected; wrap it in parentheses: `(if | a => 1 | else => 2)`.
4. **Bare in general expression positions** — wherever a general expression is
   accepted (`let`/`var`/`set` RHS, `print`/`raise` operand, list and dict
   element, template interpolation, `case` scrutinee) an `if` expression may
   appear unparenthesized.

A bare `if` at statement level is always the statement form, never an
expression. To echo an `if` expression in the REPL, parenthesize it:
`(if | a => 1 | else => 2)`.

The `if` expression design mirrors the `case` expression
([Expressions](expressions.md)). The key difference: `case` permits a runtime
`MatchError` on non-exhaustive patterns, while `if` requires `else` statically
— `if` has no pattern structure to drive exhaustiveness analysis, so a static
`else` requirement is the clean analogue.

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
  let status: Status = ask[on_parse_error: retry[2]] "Do X."
until status is Complete
```

Inline:

```agl
do[5] let status: Status = ask "Do X." until status is Complete
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
