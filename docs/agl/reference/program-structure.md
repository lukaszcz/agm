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
             | "private"? record_def              (* root only *)
             | "private"? enum_def                (* root only *)
             | "private"? type_alias              (* root only *)
             | param_decl                         (* root only *)
             | program_decl                       (* root only *)
             | agent_decl                         (* root only *)
             | builtin_var_def                     (* root only; std/config only *)
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
- **`builtin var` declarations** — body-less engine-backed mutable bindings.
  They are reserved to the canonical standard-library `std/config` module;
  entry programs and ordinary libraries cannot declare them.
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

Side-effecting forms (`print`, `:=`, loops, else-less `if`) have type `unit`,
return `void`, and are commonly followed by another expression.

## Engine settings

The standard-library module `std/config` exposes the program's engine
settings — the knobs that control the default agent runner, trace logging,
JSON strictness, the loop safety valve, and the shell-exec timeout. Each is a
**mutable binding**; import the module and assign it through a qualified target
to change a setting:

```agl
import std/config

std/config::max-iters := 10
std/config::timeout := Some("30s")
std/config::runner := "claude -p"
let budget = std/config::max-iters      # settings are readable
print budget
```

The settings and their types are:

| Setting | Type | Meaning |
|---------|------|---------|
| `log` | `bool` | Enable/disable trace logging. |
| `log-file` | `Option[text]` | Path to the trace log file. |
| `strict-json` | `bool` | Parse agent JSON output strictly. |
| `max-iters` | `int` | Safety-valve cap for unbounded loops. |
| `runner` | `text` | Default agent runner command. |
| `timeout` | `Option[text]` | Shell-exec timeout. |

A write takes effect **positionally**, exactly like any `var` mutation: it
governs the statements that follow it, in program order. An assignment target
names an imported setting the same way a read does ([Modules](modules.md)): a
qualified target always works, and a bare `max-iters := …` works when the
import is open, so the name is in scope unqualified.
The `Option[text]` settings (`log-file`, `timeout`) are set with `Some("…")` or
`None`. A `timeout` read preserves the exact assigned text; its parsed duration
controls shell execution without normalizing the stored value.

See [Host environment](host-environment.md) for the full settings table with
their defaults and for how a source write combines with the host's CLI and
config-file layers.

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
by `;` inline. `until` conditions and the right-hand sides of binders are
**`or_expr`** — the operator-chain level — so a `case` or `if` expression in
those positions must be parenthesized. Loop bodies also admit `case`, `if`,
`try`, and nested loops directly:

```agl
# Inline block: items separated by ';'
let x = 3; let y = x + 1; y

# Inline do loop: body items, then until condition
do[5] r := ask("Review ${a}", agent = reviewer); case r of Fail(issues) => a := ask("Fix ${issues} in ${a}", agent = impl) | Pass => () until r is Pass

# A case expression as a loop condition must be parenthesized:
do[3] n := n + 1 until (case st of Done => true | _ => false)
```

The `()` unit literal replaces `pass` — it is the idiomatic no-op in a
branch body:

```agl
case review of
  Pass => ()
  | Fail(issues) => artifact := ask("Fix ${issues}", agent = impl)
```

### Branch bodies

An `if` or `case` branch body is either a suite (indented block) or an inline
body. An inline body after `=>` is exactly one item: an `or_expr`, an
assignment, `raise`, or `return`. A `;` sequence, a binder, or a body that
begins a new `if`, `case`, `try`, or loop must be parenthesized or placed in a
suite — see [Inline bodies](grammar.md#inline-bodies).

### Inline `try`

A `try`/`catch` inline holds a sequence of items up to the first `catch`
keyword — binders and assignments included, with the last item an expression.
The `catch` body, like any `=>` body, is a single item or a suite.

## Expression statements

An expression evaluated at block level for its side effect is simply written
as an item. Its value is either discarded (if not the last item) or becomes
the block's value (if it is the last item):

```agl
exec "make build"
ask "Log a status update."
print "done"
```

`=` is not an expression operator, so `n = 2` as a block item is a syntax
error. Use `:=` to reassign a mutable binding, `let`/`var` to introduce a new
binding, or `==` to compare for equality.
