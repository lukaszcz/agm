# Program Structure

[← Index](index.md)

## Modules and programs

A source file is a **module**. Its root is static: it holds declarations and
constant `let`/`var` initializers. Executable code — bare expressions,
assignments, loops, agent calls — lives in function bodies, and a run starts
in a [`program def`](#program-definitions). Items are separated by newlines
or semicolons. There is no syntactic distinction between *statements* and
*expressions*: every item is an expression with a well-defined type.

The module block is the only block that may be empty. A source file holding no
items — blank, or nothing but comments — is a legal module that declares and
exports nothing; running it still requires a `program def`.

```ebnf
program      ::= module_block EOF
module_block ::= [ module_item ((NEWLINE | ";") module_item)* (NEWLINE | ";")? ]
module_item   ::= scope_region | item
block         ::= item ((NEWLINE | ";") item)* (NEWLINE | ";")?
item          ::= import_decl                     (* header position only *)
             | use_decl                           (* header position only *)
             | export_decl                        (* header position only *)
             | builtin_modifier? record_def        (* root only *)
             | builtin_modifier? enum_def          (* root only *)
             | type_alias                          (* root only *)
             | builtin_modifier? exception_def     (* root only *)
             | program_func_def                    (* root or scope region *)
             | infix_decl                          (* root only *)
             | builtin_var_def                     (* root or standard-library scope region *)
             | func_def                            (* root only *)
             | builtin_func_def                    (* root only *)
             | extern_func_def                     (* module root only; file-backed modules *)
             | let_decl | var_decl | assign_stmt
             | expr

builtin_modifier ::= "builtin" NEWLINE?
```

## Program definitions

`program def` declares an **entry point**: a function the host can run.

```agl
@doc("Publish one artifact.")
program def main(
  @doc("Artifact to publish.") @arg-pos artifact: text,
  @doc("Target environment.") @opt-short("t") target: text = "staging",
) -> unit =
  print "%{artifact} -> %{target}"
```

Rules:

- The result is `unit`, written or inferred; any other result is a static error.
- It declares no type parameters and cannot be `builtin`, `extern`, or a
  method (no `self` receiver).
- It is legal at the module root or as a member of a named scope region, never
  in a nested block.
- It is addressed by its declaration path: `main`, `review::main`, or, in a
  package, `review-tools/review::main` ([Packages](packages.md#programs-and-commands)).
- Otherwise it is an ordinary function: callable from any code, first-class,
  and subject to the usual [function](functions.md) rules.

### Entry selection

A host runs one **selected** program from the entry module: its only
`program def` implicitly, or one chosen by path when several exist
(`agm exec FILE -p review::main`). Modules reached through imports contribute
no entries; their `program def`s stay ordinary functions. A run initializes
every module's static bindings, then calls the selected program.

Inline `-c` source without a `program def` is wrapped in a synthetic
parameterless `program def main`. The REPL has no entry: a `program def`
declared there is an ordinary function.

### Parameters

The selected program's value parameters are its **external inputs**. The
parameter list defaults to the **named-only** zone: a plain `name: text`
parameter is addressed by `--name`. `@arg-pos` opens a positional slot;
`@arg-std` accepts both. Presentation attributes (`@doc`, `@opt-name`,
`@opt-short`, `@opt-env`, `@opt-metavar`, `@opt-hidden`) shape the flag; see
[Attributes](attributes.md#program-parameter-attributes).

Each parameter resolves as CLI token > `@opt-env` variable > qualified config
table > declared default. A required parameter with no external value is a
host invocation error, reported before anything executes. Parameter types
must be JSON-wire-serializable: `text` crosses verbatim, every other type
reads its external text as strict JSON or an [AgL value syntax
literal](host-environment.md#value-syntax); `unit` and function types are
rejected. A name-addressable parameter cannot spell an
[engine setting](#engine-settings) name, since both share one flag and config
namespace. Full resolution and help rules:
[Host environment](host-environment.md#program-arguments).

## Module items

### Import declarations

`import`, `use`, and `export` declarations are **header-only**: they must
appear before any other declaration or expression, at the module root or
inside a named scope region. See [Modules](modules.md) and
[Grammar](grammar.md#import-and-export-declarations) for their syntax.

### Named scope regions

A named scope region is a module item containing nested regions, header `use`
and `import` declarations, `export` declarations, static declarations,
`program def` declarations, and `let`/`var` bindings. Its matching `scope`/`end`
syntax, declaration and binder paths, and visibility rules are described in
[Named scopes](scopes.md).

### Declarations

The following are **root-only unless noted**: a static error if nested inside
an ordinary block or, for forms without a scope-region exception, a named scope
region.

Every declaration that defines a name — the type, function, and binding forms
below, along with their parameters and fields — may carry an
[attribute](attributes.md) prefix. `import`, `use`, `export`, and
`infix` declarations may not.

- **Type declarations** (`record`, `enum`, `exception`, `type`) — valid at the
  module root and in named scope regions. They may refer to types declared
  later in the program. `record`,
  `enum`, and `type` declarations may be **generic**, declaring type parameters
  in a bracketed list after the name (`record Box[T]`, `enum Option[T]`,
  `type Pair[A, B] = …`); see [Generics](generics.md). `builtin` may prefix
  `record`, `enum`, or `exception` for a host-recognized declaration, at the
  module root or in a named scope region; a type alias does not accept it. A
  scoped `builtin` type carries its declared scope path as part of its
  nominal identity, exactly like an ordinary scoped type — but its complete
  scoped name is shared with the host across the whole program and may be
  declared only once at that path; see [Built-in functions](functions.md#built-in-functions).
- **`import`/`use`/`export` declarations** — module-system declarations;
  root-only or members of a named scope region. A scoped import tail or use
  contributes bare names only to its own region and nested regions; an import's qualifier route
  stays module-wide. A scoped export re-roots its forwarded atoms under the
  region's path. See [Named scopes](scopes.md#import-and-export) and
  [Modules](modules.md#imports-and-use-inside-a-scope-region).
- **`program def` declaration** — an entry point; legal at the module root or
  as a non-method member of a named scope region. See
  [Program definitions](#program-definitions).
- **`builtin var` declarations** — body-less host-backed mutable bindings.
  A module whose path identity lies under `std` may declare one, whether it is
  the entry program or one of its imports; no other module may. A declaration
  may be a member of a named scope region and is read and written through its
  full path like any other scoped member.
  `std/config` exclusively owns engine settings; other standard-library
  modules own their domain-specific ambient bindings.
- **`infix` declarations** — root-only operator-fixity declarations.
- **Function declarations** — ordinary `def`s, `program def` entry functions, and body-less companion-backed
  `extern def`s. They may be declared at the root or in named scope regions.
  Root and same-scope `def`s may refer to declarations that appear later,
  enabling mutual recursion (see [Functions](functions.md)). An ordinary `def`
  may be **generic** (`def id[T](x: T) -> T = x`); see [Generics](generics.md).
  `builtin` introduces only the body-less `builtin def` form, legal at the
  module root or in a named scope region; a scoped `builtin def` still
  dispatches to the same host implementation as a root one. A `builtin def`
  with first parameter `self` in a type scope is a builtin method when it
  matches a supported host contract (`Agent::ask` in
  the standard library); it is selected by ordinary member syntax and can be
  projected as a receiver-capturing function value. Its complete scoped name is
  subject to the same whole-program
  uniqueness as a `builtin` type (see [Built-in functions](functions.md#built-in-functions)).
  Extern functions are file-backed-module only.
- **Scoped `let`/`var` bindings** — a plain `let` or `var` is legal anywhere a
  binder is. The scope-path-prefixed spelling (`let A::x = …`,
  `var A::count = …`) declares a member of that scope instead, and the prefix
  is legal only at the module root or inside a named scope region — the same
  placement `def` and the type forms use. See
  [Named scopes](scopes.md#binder-paths) for the complete spelling and
  visibility rules.

### The block's value

A block's **value** is the value of its last item. A final `let` or `var`
binder has no in-block continuation, so the block value is `unit` (or bottom
when its initializer exits). The binding remains available to an enclosing
continuation that evaluates after the block, such as a loop's `until`
condition.

```agl
program def main() -> unit =
  let x = ask "A"
  let y = ask "B"
  let _ = y
```

Side-effecting forms (`print`, `:=`, loops, else-less `if`) have type `unit`,
return `void`, and are commonly followed by another expression.

## Engine settings

The standard-library module `std/config` exposes the program's engine
settings — the knobs that control the default agent, trace logging,
JSON strictness, and the shell-exec timeout. Each is a
**mutable binding**; import the module and assign it through a qualified target
to change a setting:

```agl
import std/config

program def main() -> unit =
  std/config::strict-json := true
  std/config::timeout := Some("30s")
  std/config::default-agent := AgentClaude("sonnet", "medium")
  let strict = std/config::strict-json    # settings are readable
  print strict
```

The settings and their types are:

| Setting | Type | Meaning |
|---------|------|---------|
| `log` | `bool` | Enable/disable trace logging. |
| `log-file` | `Option[path]` | Path to the trace log file. |
| `strict-json` | `bool` | Parse agent JSON output strictly. |
| `default-agent` | `Agent` | Default value for `ask` calls. |
| `timeout` | `Option[text]` | Shell-exec timeout. |

A write takes effect **positionally**, exactly like any `var` mutation: it
governs the statements that follow it, in program order. An assignment target
names an imported setting the same way a read does ([Modules](modules.md)): a
qualified target always works, and a bare `strict-json := …` works after
`import std/config::*` or an equivalent `use`, so the name is in scope
unqualified.
The optional settings (`log-file`, `timeout`) are set with `Some("…")` or
`None`. A `timeout` read preserves the exact assigned text; its parsed duration
controls shell execution without normalizing the stored value.

See [Host environment](host-environment.md) for the full settings table with
their defaults and for how a source write combines with the host's CLI and
config-file layers.

## Binders: `let` and `var`

`let` carries an immutable pattern; `var` binds a single mutable name. Both
scope their binding over the **continuation** — the rest of the block and any
enclosing continuation that consumes the block. A `let` pattern must be
irrefutable for its complete initializer type, so destructuring has no runtime
match failure. A final binder makes its block `unit`-valued unless its
initializer exits, in which case the block is bottom-valued.

```agl
record Pair
  left: int
  right: int

program def main() -> unit =
  let Pair(left, right) = Pair(left = 3, right = 4)
  let _ = left + right
```

## Inline forms

AgL is designed so that small workflows fit on one line. Items are separated
by `;` inline. `until` conditions and the right-hand sides of binders are
**`or_expr`** — the operator-chain level — so a `case` or `if` expression in
those positions must be parenthesized. Loop bodies also admit `case`, `if`,
`try`, and nested loops directly:

<!-- agl-check: fragment -->
```agl
# Inline block: items separated by ';'
let x = 3; let y = x + 1; y

# Inline do loop: body items, then until condition
do[5] r := reviewer.ask("Review %{a}"); case r of Fail(issues) => a := impl.ask("Fix %{issues} in %{a}") | Pass => () until r is Pass

# A case expression as a loop condition must be parenthesized:
do[3] n := n + 1 until (case st of Done => true | _ => false)
```

The `()` unit literal replaces `pass` — it is the idiomatic no-op in a
branch body:

<!-- agl-check: fragment -->
```agl
case review of
  Pass => ()
  | Fail(issues) => artifact := impl.ask("Fix %{issues}")
```

### Branch bodies

An `if` or `case` branch body is either a suite (indented block) or an inline
body. An inline body after `=>` is exactly one item: an `or_expr`, an inline
assignment (`assign_target := or_expr`), `raise`, or `return`. A `;` sequence, a
binder, or a body that begins a new `if`, `case`, `try`, or loop must be
parenthesized or placed in a suite — see [Inline bodies](grammar.md#inline-bodies).

### Inline `try`

A `try`/`catch` inline holds a sequence of items up to the first `catch`
keyword — binders and `assign_target := or_expr` assignments included. Its final item may be a `let` or
`var`, making the try body `unit`-valued unless its initializer exits. The
`catch` body, like any `=>` body, is a single item or a suite.

## Expression statements

An expression evaluated at block level for its side effect is simply written
as an item. A non-final bare expression is a discarded-value position and
therefore must have type `unit` or `bottom`; a value-producing call can be
made explicit with `let _ = call()`. A final expression becomes the block's
value:

```agl
program def main() -> unit =
  exec "make build"
  ask "Log a status update."
  print "done"
```

`=` is not an expression operator, so `n = 2` as a block item is a syntax
error. Use `:=` to reassign a mutable binding, `let`/`var` to introduce a new
binding, or `==` to compare for equality.
