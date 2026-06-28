# Program Structure

[← Index](index.md)

## Programs

An AgL program is a **block** — a sequence of items executed top to bottom.
Items are separated by newlines or semicolons. There is no syntactic
distinction between *statements* and *expressions*: every item is an
expression with a well-defined type, and the program is an
expression-oriented sequence.

```ebnf
program    ::= block EOF
block      ::= item ((NEWLINE | ";") item)* (NEWLINE | ";")?
item       ::= import_decl                        (* header position only *)
             | config_decl                        (* root only *)
             | "private"? record_def              (* root only *)
             | "private"? enum_def                (* root only *)
             | "private"? type_alias              (* root only *)
             | param_decl                         (* root only *)
             | program_decl                       (* root only *)
             | agent_decl                         (* root only *)
             | "private"? func_def                (* root only *)
             | binder | expr
binder     ::= let_decl | var_decl
```

### Import declarations

`import` declarations are **header-only**: they must appear before any
other declaration or expression. See [Modules](modules.md) for the full
import syntax and semantics.

### Declarations

The following are **root-only**: a static error if nested inside a block.

- **Type declarations** (`record`, `enum`, `type`) — collected program-wide
  before checking begins; forward references are fine. Each may be **generic**,
  declaring type parameters in a bracketed list after the name
  (`record Box[T]`, `enum Option[T]`, `type Pair[A, B] = …`); see
  [Generics](generics.md). Any of these may be prefixed with `private` to
  restrict visibility to the defining module.
- **`param` declarations** — the program's host/config/CLI-supplied parameters.
  Entry-module only.
- **`program` declaration** — the program name used for params config lookup.
  Entry-module only.
- **`agent` declarations** — the names of the agents the program may call.
  Entry-module only (see [Modules](modules.md)).
- **`def` declarations** — user-defined functions. Like type declarations,
  all `def`s at the program root are collected before any expression is
  evaluated, enabling mutual recursion (see [Functions](functions.md)). A `def`
  may also be **generic** (`def id[T](x: T) -> T = x`); see
  [Generics](generics.md). May be prefixed with `private`.

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

Side-effecting forms (`print`, `:=`, loops, else-less `if`) yield `unit`
and are commonly followed by another expression.

## Config declarations

A **config declaration** names a fixed engine-setting key and binds it as an
immutable, runtime-resolved **readable value** — like `param`, but for the
program's own engine options:

```ebnf
config_decl ::= "config" NAME ("=" expr)?
```

Config declarations may appear **anywhere at the program root** — before or
after other items. Nesting inside a block is a static error. Each key may
appear at most once; duplicate keys are an error. Entry-module only.

The value expression, when present, must have the key's declared type. A bare
`config KEY` (no value) resolves from the host's configured default.

| Key | Type | Meaning |
|-----|------|---------|
| `log` | `bool` | Enable/disable trace logging. |
| `log-file` | `Option[text]` | Path to the trace log file. |
| `strict-json` | `bool` | Parse agent JSON output strictly. |
| `max-iters` | `int` | Maximum iterations for `do` loops. |
| `runner` | `text` | Default agent runner command. |
| `timeout` | `Option[text]` | Shell execution timeout. |

For an `Option[T]` key (`log-file`, `timeout`) a bare `T` value is accepted and
projected into `Some(value)`; an `Option[T]` value may also be given directly.

A config key is a normal readable binding: it can be used in any expression.
The binding is immutable — assigning to it is a static error.

```agl
config max-iters = 10
config timeout = "30s"        # projected into Some("30s")
config runner = "claude -p"
let budget = max-iters        # config keys are readable
print budget
```

**Precedence.** The bound value is resolved per key as:
`CLI flag > source value (if given) > [<program>].KEY > [exec].KEY > engine default`.

A bare `config KEY` (no `=` value) contributes no source value and falls through
to the program-section / exec-section / engine-default layers.

**Effect-at-binding.** The three eval-consumed keys (`strict-json`, `max-iters`,
`timeout`) take effect at the point the declaration executes in declaration order;
expressions that follow see the updated setting. The remaining keys (`runner`,
`log`, `log-file`) are start-resolved before the program runs; place them near the
top of the program so the agent factory and trace infrastructure see them.

**Error surface.** A source `config timeout = "…"` value is a runtime-evaluated
expression; a bad value raises a runtime error (exit 2). A bad `--timeout` or
`[exec].timeout` value is caught before execution (exit 1). The source `config
timeout` controls the **shell-exec** timeout only; the agent idle timeout is always
start-resolved from the CLI or `[exec]`.

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
by `;` inline. Branch bodies, `until` conditions, and the right-hand sides of
binders are **`or_expr`** — the operator-chain level — and a `case` or `if`
expression in those positions must be parenthesized:

```agl
# Inline block: items separated by ';'
let x = 3; let y = x + 1; y

# Inline do loop: body items, then until condition
do[5] let r: Review = ask("Review ${a}", agent: reviewer); case r of Fail(i) => a := ask("Fix ${i} in ${a}", agent: impl) | Pass => () until r is Pass

# A case expression as a loop condition must be parenthesized:
do[3] n := n + 1 until (case st of Done => true | _ => false)
```

The `()` unit literal replaces `pass` — it is the idiomatic no-op in a
branch body:

```agl
case review of
  Pass => ()
  | Fail(issues) => artifact := ask("Fix ${issues}", agent: impl)
```

### Branch bodies

An `if` or `case` branch body is either a suite (indented block) or a single
expression at the `or_expr` level. A branch body that begins a new `if` or
`case` in the same inline position must be parenthesized or placed in a suite.

### Inline `try`

A `try`/`catch` inline holds a sequence of items up to the first `catch`
keyword; the `catch` body is a single expression at `or_expr` level or a
suite.

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
**"Bare assignment 'n = …' is not valid. Use ':=' to reassign a mutable
variable."**
