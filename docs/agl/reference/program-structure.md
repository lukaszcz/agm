# Program Structure

[← Index](index.md)

## Programs

An AgL program is a sequence of statements executed top to bottom. Statements
are separated by newlines or by semicolons:

```ebnf
program    ::= stmt_list EOF
stmt_list  ::= stmt ((NEWLINE | ";") stmt)* (NEWLINE | ";")?
```

The program root is itself a lexical scope
([Bindings and scope](bindings-and-scope.md)). Three statement kinds are
restricted to the root:

- **Type declarations** (`record`, `enum`, `type`) — a type declaration inside
  a nested block is a static error. Type declarations are collected
  program-wide before anything else is checked, so their order relative to
  each other and to value statements does not matter; forward references are
  fine ([Types](types.md)).
- **Input declarations** (`input`) — see
  [Bindings and scope](bindings-and-scope.md).
- **Agent declarations** (`agent`) — the names of the agents the program may
  call. Like inputs, they may appear only at the root; a declaration nested in
  a block is a static error. See
  [Bindings and scope](bindings-and-scope.md).

## Statement kinds

```ebnf
stmt ::= record_def | enum_def | type_alias        (root only)
       | input_decl                                (root only)
       | agent_decl                                (root only)
       | let_decl | var_decl | set_stmt
       | do_until
       | if_stmt | case_stmt | try_stmt
       | raise_stmt
       | pass_stmt
       | print_stmt
       | expr_stmt
```

An **expression statement** evaluates an expression and discards its result —
most usefully a bare agent call or `exec` executed for effect:

```agl
exec "make build"
prompt "Log a status update."
```

A bare equality at statement level that looks like an assignment is rejected
with a targeted error: `n = 2` as a statement produces
**"Bare assignment 'n = …' is not valid. Use 'set' to reassign a mutable
variable."**

### `pass`

`pass` is a no-op. It exists because every branch body is a statement block
and empty branches are common:

```agl
case review of
  | Pass => pass
  | Fail(issues) => set artifact = impl "Fix ${issues}"
```

### `print`

`print expr` evaluates its operand — which may have any type — renders it
with *console rendering* (text verbatim; numbers and booleans as scalar
text; everything else as pretty-printed JSON; never any boundary markers),
writes the result plus a trailing newline to the host's standard output, and
records the printed text in the trace. `print` never fails on a well-typed
value and produces no result. See
[Strings and interpolation](strings-and-interpolation.md) for the rendering
rules.

```agl
print "Review round failed; retrying."
print "Final artifact:\n${artifact}"
print review
```

## Blocks: suites

A *suite* is an indented block containing any statements:

```agl
if review is Fail =>
  set artifact = impl "Fix ${review}"
  print "fixed"
| else =>
  pass
```

Suites may appear as the body of a `do` loop, an `if`/`case` branch, a `try`
body, or a `catch` handler.

## Inline forms and the bar-safe rules

AgL is designed so that small workflows fit on one line. Inline (single-line)
forms are restricted by a deterministic taxonomy so that a `|`, `catch`, or
`until` token always belongs to exactly one construct.

Statements are classified by how they end:

- **Closed statements** — self-contained; safe anywhere inline and may be
  followed by `;`: `let`, `var`, `set`, `pass`, `raise`, `print`, expression
  statements, type, input, and agent declarations, and `do … until` (which is
  right-delimited by its `until` clause).
- **Open statements** — extend open-endedly to the right: `if`, `case`, and
  `try`. Inline, an open statement is valid only as the **last** element of a
  `do` body (where the loop's `until` seals it).
- **Bar-safe statements** — the closed statements, with the additional
  restriction that every trailing expression slot (a `let`/`var`/`set`
  initializer, a `raise` operand, a `print` operand) must be a *bar-safe
  expression*.

A **bar-safe expression** is any expression other than an unparenthesized
`case` expression. In a bar-safe position a `case` expression must be wrapped
in parentheses; everywhere else (statement level, suites, inside parentheses)
it stays bare.

The bodies, normatively:

```ebnf
do_body      ::= suite
              |  closed_stmt (";" closed_stmt)* (";" open_stmt)?
              |  open_stmt

try_body     ::= suite
              |  closed_stmt (";" closed_stmt)*

branch_body  ::= suite            (* if/case statement branch *)
              |  bar_safe_stmt
              |  try_stmt

catch_body   ::= suite
              |  bar_safe_stmt
```

**Bar-safe positions** — places whose next token may legally be `|` — are:
inline `if`/`case` branch bodies, inline `catch` bodies, `if` conditions,
`until` conditions, and `case`-expression branch results.

Consequences, stated as rules:

1. An inline `if`/`case` branch body is a single bar-safe statement or a
   `try`; nesting another `if` or `case` on the same line requires a suite.
2. An unparenthesized `case` expression is rejected in every bar-safe
   position — including after every `until`. Parenthesize it or use a suite.
3. An inline `catch` body is a single bar-safe statement; a handler
   containing `if`/`case`/`try` requires a suite.
4. An inline `try` body contains only closed statements; a trailing
   `if`/`case`/`try` requires a suite. (A trailing open statement *is*
   allowed at the end of an inline `do` body, because `until` seals it.)

Violations produce targeted diagnostics, for example
**"`case` is not allowed inline here; parenthesize the case expression,
e.g. `(case x of ...)`."** or
**"`if` is not allowed inline here; write it as an indented block instead."**

Examples:

```agl
# Inline do body: closed statements separated by ';', sealed by 'until'.
do[5] let s: Status = prompt "Do X."; print s until s is Complete

# Inline do body ending in an open statement (legal: 'until' seals it).
do[5] let r: Review = reviewer "Review ${a}"; case r of | Fail(i) => set a = impl "Fix ${i}" | Pass => pass until r is Pass

# A case expression as a loop condition must be parenthesized:
do[3] set n = n + 1 until (case st of | Done => true | _ => false)
```
