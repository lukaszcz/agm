# Bindings and Scope

[← Index](index.md)

AgL has two value binders (`let` and `var`), destructive assignment, a param declaration, an
agent declaration, and a function declaration (`def`). There is no bare
assignment: `x = e` as an item is a syntax error — use `let`/`var` to bind or
`:=` to reassign. The equality operator is `==`
([Expressions](expressions.md)).

## `let` — immutable binding

```ebnf
let_decl ::= "let" name (":" type_expr)? "=" expr
```

`let` evaluates the initializer, checks it against the annotation (if any),
and creates an **immutable** binding in the current scope. It scopes over the
**continuation** — the remaining items in the block and any enclosing
continuation that consumes the block. A block ending in a bare `let` has type
`unit` unless its initializer exits, in which case it has bottom type:

<!-- agl-check: fragment -->
```agl
let review: Review = ask(
  "Review ${artifact}",
  agent = reviewer,
  on_parse_error = Retry(n = 2)
)
let count = 3
```

`_` is a wildcard binder. `let _ = e` evaluates `e` and discards its value
without introducing a readable name; it may be repeated in a scope. `var _ =
e` has the same discard behavior. `_` is never readable, even if an outer
scope has a binding with that spelling. An annotation on `_` is deliberately
ignored: annotations normally constrain an initializer and declare its binder's
type, but `_` creates neither a readable binder nor a binding type. Its RHS is
therefore checked without an annotation-derived expected type. Use it when a
non-`unit` value is intentionally discarded.

## `var` — mutable binding

```ebnf
var_decl ::= "var" name (":" type_expr)? "=" expr
```

Identical to `let` except the binding is **mutable** — it may later be
updated with `:=`. A final `var` also makes its block unit-valued (or bottom
when its initializer exits); like a final `let`, it remains visible to an
enclosing continuation:

<!-- agl-check: fragment -->
```agl
var artifact: text = ask("Implement ${spec}", agent = impl)
```

## `:=` — destructive assignment

```ebnf
assign_stmt ::= assign_target ":=" expr
assign_target ::= name ("[" expr "]")*
                | qual_prefix name
```

`:=` updates the nearest visible **mutable** binding, has type `unit`, and
returns `void`. It never creates a binding. The expected type of the
right-hand side is the declared type of the binding being updated:

<!-- agl-check: fragment -->
```agl
var proposal: Turn = ask("Initial proposal.", agent = researcher)
proposal := ask("Revise proposal.", agent = researcher)   # target type: Turn
```

`:=` can also update an element of a mutable list or an existing key of a
mutable dictionary:

```agl
var xs = [1, 2]
xs[0] := 10

var metadata = {"status": "draft"}
metadata["status"] := "ready"
```

Assignment indexes are adjacency-sensitive: the opening `[` must be adjacent
to the target name or preceding index, as in `xs[0]`. A spaced form such as
`xs [0]` is not an indexed assignment target.

Indexed assignment is copy-on-write: the binding is updated with a new list or
dictionary value containing the changed element. The root must be a `var`
binding. Assigning through `let`, `param`, function arguments, function return
temporaries, fields, or a type-qualified constructor reference is a static
error. List assignment uses the same
negative-index and `IndexError` rules as list access. Dictionary assignment
updates existing keys only; assigning to a missing key raises `KeyError`.

Static rules, all checked before execution:

1. Redeclaring a name in the same scope is an error.
2. Assignment to an undeclared name is an error.
3. Assignment to an immutable binding is an error; the diagnostic names the binder
   kind — `let`, `param`, a catch binder, or a pattern binding.
4. Reading a name that is not visible in the current scope chain is an error.
5. The contextual keywords `ask` and `exec` cannot be used as binding or
   param names.

## `def` — function declarations

```ebnf
func_def ::= ["private"] "def" name type_params? "(" param_list? ")" ("->" type_expr)? ("=" func_body | suite)
```

`def` is a root-only declaration. It introduces an immutable binding of
function type in the top-level scope. All `def`s at the program root are
collected in a pre-pass so they are mutually visible — every `def` may call
every other `def` (and itself) without forward declaration.

A `def` may be prefixed with `private`, making it invisible to other modules
(see [Modules](modules.md)). The `private` modifier has no effect on lexical
scoping within the same module.

```agl
def is_even(n: int) -> bool =
  if n == 0 => true else => is_odd(n - 1)

def is_odd(n: int) -> bool =
  if n == 0 => false else => is_even(n - 1)
```

A `def` may be **generic** — it can declare type parameters in a bracketed
list after its name (`def id[T](x: T) -> T = x`); see [Generics](generics.md).

A `def` inside a nested block is a static error. See
[Functions](functions.md) for full details.

## Typing of bindings

- With an annotation, the initializer is checked against the annotated type
  (`int → decimal` widening applies; see [Types](types.md)), and the
  binding's declared type is the annotation.
- Without an annotation, the binding's type is inferred from the initializer.
  **An untyped `ask` defaults to `text`; an untyped `exec` defaults to the
  structured `ExecResult`**:

  ```agl
  let x = ask "A"           # x: text
  let res = exec "ls"       # res: ExecResult (structured default)
  ```

- Empty list/dictionary literals cannot be inferred and require an
  annotation: `let items: list[Issue] = []`.
- A `let` or `var` bound to a function value has a function type:

  <!-- agl-check: fragment -->
  ```agl
  let double = fn(x: int) => x * 2   # double: int -> int
  let f: int -> text = classify     # explicit function type annotation
  ```

## `param` — declared program parameters

```ebnf
param_decl ::= "param" name (":" type_expr)? ("=" expr)?
```

`param` declarations are root-only. Each enters the root scope as an
immutable binding. A param may declare a type, a default expression, both, or
neither. Without an explicit type or default, the param defaults to `text`.

```agl
param spec                 # same as: param spec: text
param max_severity: int
param metadata: json
param limit: int = 10
```

At run start, before any expression evaluates, the host validates provided
param values. Required params without an external value or default, and params
whose external values fail conversion, are *host invocation errors* — not AgL
exceptions and not catchable in-language. Extra external values are ignored by
the runtime after CLI/config resolution.

Every program param must have a JSON-wire-serializable type, even when it has a
default and the host does not supply a value. Supported param types are `text`,
`int`, `decimal`, `bool`, `json`, lists, dictionaries, records, and enums.
Runtime-only types such as `unit`, `agent`, and function types cannot be used as
program param types.

## `builtin var` — engine-setting bindings

```ebnf
builtin_var_def ::= "builtin" NEWLINE? "var" name ":" type_expr
```

A `builtin var` declares a body-less, host-backed, **mutable** binding with a
mandatory type and no initializer. The `builtin` marker may be on the same line
as `var` or on the line directly above it. It may appear only at the root of the
canonical standard-library module `std/config`; declarations in entry programs
or other library modules are static errors. `std/config` uses it to expose the
program's engine settings:

```agl
open import std/config

std/config::max-iters := 10           # write a setting (qualified target)
runner := "claude -p"                 # the open import also allows a bare target
let cap = std/config::max-iters       # read a setting
```

An engine setting is an ordinary mutable binding in another module, so an
assignment target names it exactly as a read does: a qualifier always works, and
a bare name works whenever the import is open, so the name is in scope
unqualified. A write takes effect from its
program point onward, exactly like any `var` mutation. The `Option[text]` settings
are set with `Some("…")` or `None`.

See [Host environment](host-environment.md) for the settings table, their types
and defaults, and how a source write combines with the host's CLI and config-file
layers.

## `agent` — declared agents

```ebnf
agent_decl ::= "agent" name ("=" STRING)?
```

`agent` declarations are root-only and **entry-module only** — they are a
static error inside an imported library module (see [Modules](modules.md)).
Each declared name enters the root scope as an **immutable binding of type
`agent`**. Agent values may be stored in bindings, passed to `def` parameters,
and held in `list[agent]`:

```agl
agent reviewer
agent impl = "claude -p %{PROMPT_FILE}"

let agents: list[agent] = [reviewer, impl]
```

A declared agent is a first-class value. It is passed to `ask` via the
`agent` parameter:

<!-- agl-check: fragment -->
```agl
let r: Review = ask("Review ${artifact}", agent = reviewer)
```

Rules:

1. `agent` declarations are valid **only at the program root**.
2. Agent names and variable names share the same value namespace — both are
   looked up by the same name resolution. An `agent impl` declaration and a
   `let impl` declaration cannot coexist in the same scope (redeclaration
   error).
3. Declaring the same agent name twice is a static error.
4. `ask` and `exec` cannot be declared as agents.
5. An unused declared agent produces a non-fatal **warning**.
6. The runner hint must be a static string literal with no interpolation.

## Names, namespaces, and constructors

Identifier capitalization carries **no** meaning: a name's case never
determines whether it denotes a type, a value, or a constructor
([Lexical structure](lexical-structure.md)). What a name denotes is fixed by
how it is declared and the position it appears in.

AgL keeps **two namespaces**: a *type* namespace and a *value* namespace. A
name may exist in both at once without collision. A `record` or `enum`
declaration introduces a type name *and* a same-spelled value binding for its
constructor:

```agl
record Box[T]
  value: T
# 'Box' the type lives in the type namespace;
# 'Box' the constructor lives in the value namespace.
let b: Box[int] = Box(value = 1)
```

### Constructors are ordinary value bindings

Record constructors and enum variants are normal bindings in the value
namespace. They can be referenced bare, stored, and passed like any value:

<!-- agl-check: fragment -->
```agl
let mk: int -> Box[int] = Box   # the constructor as a first-class value
let one = mk(1)                    # called positionally, in field order
```

Direct construction uses positional-greedy binding — positional arguments fill
positional-capable fields first, then named arguments follow (`Box(value = 1)`,
`Some(value = x)`, or `Ok(42)` for a single-standard-field variant). A
constructor reached **through a variable** is an ordinary function value invoked
**positionally**, in declaration order. Nullary enum variants are ordinary
values (`let e: Option[int] = None`). See [Generics](generics.md) for the full
constructor-value story (including when a generic constructor needs an
expected-type annotation).

### Overload sets, shadowing, and ambiguity

Two enums may declare the **same** unqualified variant name; that name then
resolves to an *overload set*. An unqualified reference in ordinary expression
position is a **static ambiguity error** — regardless of payload, surrounding
context, or explicit type arguments. **Qualify** the reference with the owning
enum to disambiguate. Constructor patterns and `is` tests are different: their
scrutinee's static enum type selects the owner.

```agl
enum Holder[T]
  | empty
  | tagged(by: T)

enum Other
  | tagged(label: text)      # same unqualified name 'tagged'

let h: Holder[int] = Holder::tagged(by = 7)   # qualified — unambiguous
```

A **nearer ordinary binding shadows** a constructor (or an overload set): an
inner `let`, `var`, or function parameter named `tagged` hides the outer
constructor for the rest of its scope, exactly like any other shadowing.

```agl
def shadow(tagged: int) -> int = tagged * 10   # parameter hides the constructor
```

Whether an ordinary declaration may claim a constructor's spelling **in that
constructor's own scope** depends on whether the constructor stays reachable:

- An **enum variant** may be claimed. Its name is `Owner::variant`, and the
  unqualified spelling is a convenience, so the variant remains reachable
  qualified. The claiming declaration owns expression position, while
  case-pattern constructor lookup stays independent.
- A constructor **declared in another module** may be claimed, since module
  qualification still reaches it. This covers the **standard core** names —
  exception types (`Abort`, `AgentParseError`, …), enum variants (`Some`,
  `Retry`, …), and records (`ExecResult`, `AgentRequest`). They are
  conveniences, not reserved words.
- A **record**, **exception**, or **type alias** declared in the *same* module
  may **not** be claimed. Its constructor name is the declaration itself, with
  no qualified spelling to fall back on, so a second declaration of that name
  is a duplicate-declaration error.

<!-- agl-check: error -->
```agl
enum Color
  | Red
  | Blue

let Red = 5                 # allowed — 'Color::Red' still names the variant
let ExecResult = 0          # allowed — declared in another module
def Retry(n: int) -> int = n + 1

record Widget
  x: int

let Widget = 1              # error: 'Widget' is already declared in this scope
```

Use [`::name`](modules.md) to reach the module's own top-level declaration past
any shadowing:

<!-- agl-check: fragment -->
```agl
print(::Red)                # the 'let', not the variant
```

## Lexical scoping

All binding is statically scoped. The program root is a scope, and these
constructs each introduce a nested scope:

1. each `do` loop iteration (a fresh scope per iteration),
2. each `if` branch body,
3. each `case` branch body,
4. a `try` body,
5. each `catch` body,
6. each function call (a fresh call scope with parameters bound).

`let` and `var` bind in the *current* scope only; bindings never escape
their scope. Lambda bodies close over their definition environment but are not
self-recursive (the lambda name is not in scope inside its own body):

```agl
let double = fn(x: int) => x * 2   # 'double' is NOT in scope inside
```

### Shadowing

Inner scopes may **shadow** outer bindings with a new `let` or `var`.
Shadowing is not mutation; the outer binding is unchanged when the inner
scope ends:

<!-- agl-check: fragment -->
```agl
let x = "outer"
if condition =>
  let x = "inner"    # shadows outer x
  ask "Uses ${x}"    # inner
ask "Uses ${x}"      # outer
```

### Mutation across scopes

`:=` reaches *through* scopes to the nearest visible mutable binding:

<!-- agl-check: fragment -->
```agl
var artifact: text = ask("Implement ${spec}", agent = impl)

case review of
  | Fail(issues) =>
      artifact := ask("Fix ${issues} in ${artifact}", agent = impl)
  | Pass => ()
```

### Loop scope

A `do` body opens a fresh scope on each iteration, and the `until` condition
is evaluated **in that same iteration scope** — it can see `let` or `var`
bindings made by the body, including a final binder. A `while` condition runs
before the body and cannot see its bindings:

<!-- agl-check: fragment -->
```agl
do[5]
  let review: Review = ask("Review ${artifact}", agent = reviewer)
until review is Pass
```

`review` is visible to `until` but does not exist after the loop.

When a bound expression `[expr]` is present, it is evaluated **once in the
enclosing scope**, before the first iteration. It cannot see any binding
introduced by the body, and mutating a `var` the bound references from inside
the body does not affect the already-fixed bound.

### Pattern and catch variables

Pattern variables and `catch` binders are immutable and scoped to their
branch or handler body. See [Pattern matching](pattern-matching.md) and
[Exceptions](exceptions.md).

### Function parameter scope

Each call opens a fresh scope with the function's parameters bound. Defaults
are evaluated in the function's **definition** scope (not the call site):

```agl
let default_limit = 3

def summarize(doc: text, limit: int = default_limit) -> text =
  "[${limit}] ${doc}"
```

`default_limit` is resolved at definition time, not call time.
