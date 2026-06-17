# Program Structure

[← Index](index.md)

## Programs

An AgL program is a **block** — a sequence of items executed top to bottom.
Items are separated by newlines or semicolons. In v2, the syntactic
distinction between *statements* and *expressions* is removed: every former
statement is an expression with a well-defined type, and the program is an
expression-oriented sequence.

```ebnf
program    ::= block EOF
block      ::= item ((NEWLINE | ";") item)* (NEWLINE | ";")?
item       ::= declaration | binder | expr
declaration ::= record_def | enum_def | type_alias | param_decl
              | program_decl | agent_decl | func_def | config_pragma
binder     ::= let_decl | var_decl
```

### Declarations

The following are **root-only**: a static error if nested inside a block.

- **Type declarations** (`record`, `enum`, `type`) — collected program-wide
  before checking begins; forward references are fine.
- **`param` declarations** — the program's host/config/CLI-supplied parameters.
- **`program` declaration** — the program name used for params config lookup.
- **`agent` declarations** — the names of the agents the program may call.
- **`def` declarations** — user-defined functions. Like type declarations,
  all `def`s at the program root are collected before any expression is
  evaluated, enabling mutual recursion (see [Functions](functions.md)).

### The block's value

A block's **value** is the value of its last item. A block that ends in a
`let` or `var` binder (with no continuation) is a static error — a binder
must be followed by at least one more item. This ensures that bindings always
scope over a meaningful continuation.

```agl
let x = ask "A"
let y = ask "B"
y              # the program's value is y
```

Side-effecting forms (`print`, `set`, loops, else-less `if`) yield `unit`
and are commonly followed by another expression.

## Config pragmas

A **config pragma** sets a program-level option:

```ebnf
config_pragma ::= "config" KEY "=" VALUE
VALUE         ::= "true" | "false" | INT | DECIMAL | string_literal
```

Pragmas must appear **before every other item** at the program root (the
*header* position). A pragma after any non-pragma item is a static error.
Pragmas nested inside a block are also static errors.

Each key may appear at most once; duplicate keys are an error.

| Key | Value type | Meaning |
|-----|------------|---------|
| `log` | `bool` | Enable/disable trace logging. |
| `log_file` | non-empty string | Path to the trace log file. |
| `strict_json` | `bool` | Parse agent JSON output strictly. |
| `max_iters` | positive integer | Maximum iterations for `do` loops. |
| `runner` | non-empty string | Default agent runner command. |
| `timeout` | string or positive integer | Shell execution timeout. |

```agl
config log = true
config max_iters = 10
config runner = "claude -p"
param spec
let result = ask "Process ${spec}"
print result
```

**Precedence.** CLI flags override pragma values, which override config-file
settings.

**String values** must be static literals — no interpolation.

## Binders: `let` and `var`

`let` and `var` bind a name and scope it over the **continuation** — the
rest of the block. They are **not self-contained items**: they must be
followed by at least one more item in the same block.

```agl
let x = 3          # x is in scope below
let y = x + 1      # y is in scope below
y                  # block ends here; its value is y
```

A block ending in a bare `let` or `var` is a static error:

```agl
def broken() -> int =
  let x = 1        # static error: 'let' must be followed by an expression
```

## Inline forms

AgL is designed so that small workflows fit on one line. Items are separated
by `;` inline. In v2, the bar-safe stratification that previously governed
statement bodies is replaced by a simpler model: branch bodies, `until`
conditions, and the right-hand sides of binders are **`or_expr`** — the
operator-chain level — and a `case` or `if` expression in those positions
must be parenthesized:

```agl
# Inline block: items separated by ';'
let x = 3; let y = x + 1; y

# Inline do loop: body items, then until condition
do[5] let r: Review = ask("Review ${a}", agent: reviewer); case r of | Fail(i) => set a = ask("Fix ${i} in ${a}", agent: impl) | Pass => () until r is Pass

# A case expression as a loop condition must be parenthesized:
do[3] set n = n + 1 until (case st of | Done => true | _ => false)
```

The `()` unit literal replaces `pass` — it is the idiomatic no-op in a
branch body:

```agl
case review of
  | Pass => ()
  | Fail(issues) => set artifact = ask("Fix ${issues}", agent: impl)
```

### Branch bodies

An `if` or `case` branch body is either a suite (indented block) or a single
expression at the `or_expr` level. A branch body that begins a new `if` or
`case` in the same inline position must be parenthesized or placed in a suite.

### Inline `try`

A `try`/`catch` inline holds a sequence of items up to the first `catch`
keyword; the `catch` body is a single expression at `or_expr` level or a
suite.

### A note on `pass`

`pass` is no longer a keyword in v2. Its role is taken by the unit literal
`()`. Existing code using `pass` should replace it with `()`.

## Expression statements

An expression evaluated at block level for its side effect is simply written
as an item. Its value is either discarded (if not the last item) or becomes
the block's value (if it is the last item):

```agl
exec "make build"
ask "Log a status update."
print "done"
```

A bare equality at block level that looks like an assignment is rejected with
a targeted error: `n = 2` as an item produces
**"Bare assignment 'n = …' is not valid. Use 'set' to reassign a mutable
variable."**
