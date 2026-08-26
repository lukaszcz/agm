# Bindings and Scope

[← Index](index.md)

AgL has two value binders (`let` and `var`), destructive assignment, a param declaration, and a
function declaration (`def`). There is no bare
assignment: `x = e` as an item is a syntax error — use `let`/`var` to bind or
`:=` to reassign. The equality operator is `==`
([Expressions](expressions.md)).

## `let` — immutable binding

```ebnf
let_decl ::= "let" pattern (":" type_expr)? "=" expr
```

The annotation applies to the complete pattern. The initializer is evaluated
before any name introduced by the pattern becomes visible in the continuation.

A bare name at a `let` root always introduces a binding, even when a visible
constructor has the same spelling. Write `Only()` or a qualified constructor
pattern when the pattern must test a constructor instead. Within a constructor
pattern, bare names follow the same field-directed rules as nested `case`
patterns; an `as` name always binds and `_` never binds.

A `let` pattern may use record or enum constructors, literals, wildcards, and
`as` binders. The annotation describes the complete value being matched, not any
individual binder: the initializer is checked once against it, then each
selected binder receives its field or whole-value type. Without an annotation,
the complete matched type is inferred from the initializer. A bottom initializer
needs that annotation to type binders. `let _` is a discard: its annotation does
not constrain the initializer. Every `let` pattern must be irrefutable; a
refutable pattern is a static error. A destructuring `let` evaluates its
initializer once and installs every selected binder from that value.

`let` evaluates the initializer, checks it against the complete annotation (if
any), and creates **immutable** bindings in the current scope. It scopes over the
**continuation** — the remaining items in the block and any enclosing
continuation that consumes the block. A block ending in a bare `let` has type
`unit` unless its initializer exits, in which case it has bottom type:

<!-- agl-check: fragment -->
```agl
let review: Review = reviewer.ask(
  "Review %{artifact}",
  on_parse_error = Retry(n = 2)
)
let count = 3
```

`let` freezes the *name*, never the *data*: a `let`-bound array or dict is
still a mutable reference value and can be updated in place through the name
([Types](types.md)). A program that wants an independent value detaches one
with `copy` ([Copying values](types.md#copying-values)).

`_` is a wildcard binder. `let _ = e` evaluates `e` and discards its value
without introducing a readable name; it may be repeated in a scope. `var _ =
e` has the same discard behavior. `_` is never readable, even if an outer
scope has a binding with that spelling. An annotation on `_` is deliberately
ignored: annotations normally constrain an initializer and declare its binder's
type, but `_` creates neither a readable binder nor a binding type. Its RHS is
therefore checked without an annotation-derived expected type. Use it when a
non-`unit` value is intentionally discarded.

### REPL persistence and echo

At the REPL top level, a completed destructuring `let` persists every selected
binder across later entries. Its echo shows the complete matched value and type,
not a synthetic binder name. If a later initializer fails, completed
pattern initializers (and completed function closures) remain available; the
failing initializer contributes no binders.

### Binder scope paths

`let` and `var` also accept an optional scope-path prefix on a single-name
binder at the module root (`let A::x = 1`, `var A::count = 0`), declaring a
binding at that path rather than in the module root namespace. For `let`,
the prefix is written as an ordinary qualifier chain at the pattern root: a
plain chain spellable as a declaration path (no argument list, no `as`
binder, no module route or type-argument-applied segment, not anchored at
the module root) is read as a scoped binding path, while any other pattern
shape keeps its constructor-pattern meaning. See
[Named scopes](scopes.md#binder-paths) for the complete disambiguation and
for declaring a binder inside a `scope` region.

## `var` — mutable binding

```ebnf
var_decl ::= "var" decl_head (":" type_expr)? "=" expr
```

Identical to `let` except the binding is **mutable** — it may later be
updated with `:=`. A final `var` also makes its block unit-valued (or bottom
when its initializer exits); like a final `let`, it remains visible to an
enclosing continuation:

<!-- agl-check: fragment -->
```agl
var artifact: text = impl.ask("Implement %{spec}")
```

## `:=` — destructive assignment

```ebnf
assign_stmt ::= assign_target ":=" expr
assign_target ::= qualifier_chain? name
                | postfix "[" expr "]"
                | postfix "." field_name
```

A bare `:=` (no index) rebinds the nearest visible **mutable** binding, has
type `unit`, and returns `void`. It never creates a binding. The expected
type of the right-hand side is the declared type of the binding being
updated. `qualifier_chain? name` is the same qualifier syntax a read uses, so
a scoped `var`'s path (`A::count := 1`) is a valid target exactly as a scoped
read is; see [Named scopes](scopes.md#names-and-visibility) for a scoped
`var`'s bare-name and path assignment forms and its immutability rule:

<!-- agl-check: fragment -->
```agl
var proposal: Turn = researcher.ask("Initial proposal.")
proposal := researcher.ask("Revise proposal.")   # target type: Turn
```

`:=` can also update an element of an array or an existing key of a
dictionary:

```agl
program def main() -> unit =
  let xs = [1, 2]
  xs[0] := 10
  var metadata = {"status": "draft"}
  metadata["status"] := "ready"
```

Assignment indexes are adjacency-sensitive: the opening `[` must be adjacent
to the target name or preceding index, as in `xs[0]`. A spaced form such as
`xs [0]` is not an indexed assignment target.

Arrays and dicts are mutable reference values ([Types](types.md)), so an
indexed assignment mutates the array or dictionary **in place**: every other
binding, field, closure capture, or loop cursor that references the same
array or dictionary observes the change.

Indexed assignment is legal on **any** array- or dict-typed expression: a
`let` binding, a `param`, a function argument, a record or exception field, a
function call result, and a qualified read all work as the container of
`target[index] := value`. `let` and `param` guarantee only that the *name*
cannot be rebound; they say nothing about the contents of what the name
refers to, so mutating an array or dict through one is not a rebinding and is
not restricted. A bare `name := value` — an actual rebinding, with no index —
remains a static error unless `name` is a mutable `var` binding. Array
assignment uses the same negative-index and `IndexError` rules as array
access. Dictionary assignment updates existing keys only; assigning to a
missing key raises `KeyError`.

`:=` can also update a field declared with `var` on a record or enum-member
record. The receiver may be any record-typed postfix expression, including a
field, indexed element, parameter, or call result. The field must exist and be
marked `var`; fields of exceptions and unmarked record fields are not
assignable. An enum-typed receiver has no fields, so narrow it with a `case`
pattern or cast it to its member record first. Like indexed assignment, field
assignment mutates the shared value in place, even through a `let` binding.

Either indexed or `var`-field assignment can close a reference cycle — see
[Cycles](types.md#cycles) for the operations that detect one. Because binding
and assignment never copy, a program that wants an independent value asks for
one with the `copy`/`shallow_copy` built-ins — see [`copy` and
`shallow_copy`](types.md#copying-values).

Evaluation order for `target[index] := value` is left to right: the
container, then the index, then `value`, then the checked in-place store. A
nested target such as `m["a"]["b"] := v` evaluates the outer container, reads
its `"a"` entry to reach the inner container, evaluates `"b"` and `v`, then
stores into the inner container — so a `KeyError` or `IndexError` raised
while reaching the target aborts before `value` is evaluated, but a `value`
with a side effect always runs before the store is checked. For
`receiver.field := value`, the receiver is evaluated first, then `value`, then
the in-place store.

Static rules, all checked before execution:

1. Redeclaring a name in the same scope is an error.
2. Assignment to an undeclared name is an error.
3. A bare `name := value` (no index) to an immutable binding is an error; the
   diagnostic names the binder kind — `let`, `param`, a catch binder, or a
   pattern binding.
4. A field-assignment target must be a `var` field of a record or
   enum-member record; an enum-typed receiver must be narrowed first.
5. Reading a name that is not visible in the current scope chain is an error.
6. The contextual keywords `ask` and `exec` cannot be used as binding or
   param names.

## `def` — function declarations

```ebnf
func_def ::= "def" decl_head type_params? "(" param_list? ")" ("->" type_expr)? ("=" func_body | suite)
decl_head ::= [scope_path "::"] name
```

`def` is a static declaration valid at the module root and in
[named scope regions](scopes.md). Its head may include a scope path, such as
`def Text::render(value: text) -> text = value`. It introduces an immutable
function binding in its declaration layer.
A root `def` may call itself or any other root `def`, including one declared
later. `def`s in the same named scope have the same visibility, so they may
call one another recursively without a forward declaration.

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

- With an annotation, the initializer is checked against the annotated complete
  type (`int → decimal` widening applies; see [Types](types.md)); every pattern
  binder receives its selected concrete type.
- Without an annotation, the complete matched type and each binder type are
  inferred from the initializer.
  **An untyped `ask` defaults to `text`; an untyped `exec` defaults to the
  structured `ExecResult`**:

  ```agl
program def main() -> unit =
  let x = ask "A"
  let res = exec "ls"
  ```

- Empty array/dictionary literals cannot be inferred and require an
  annotation: `let items: array[Issue] = []`.
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

`param` declarations are legal at the module root or as a member of a named
scope region in every entry and library module ([Named scopes](scopes.md#parameters)
— no declaration-path shorthand). An engine-setting leaf name cannot be used
for a param, including within a named scope. Each enters its scope as an immutable binding.
A param may declare a type, a default expression, both, or neither. Without an
explicit type or default, the param defaults to `text`.

Each program's parameter inventory includes params declared in its module and
in its transitive imports. Every inventory parameter receives a host value or
default before execution, so imported functions can read their module's params.
Defaults resolve parameter dependencies independently of declaration and module
order, including dependencies reached through function calls. A cycle among
omitted parameter defaults is a host invocation error.

```agl
param spec                 # same as: param spec: text
param max_severity: int
param metadata: json
param limit: int = 10
```

At run start, before any expression evaluates, the host validates provided
values for every parameter in the selected program's inventory. Required params
without an external value or default, and params whose external values fail
conversion, are *host invocation errors* — not AgL exceptions and not catchable
in-language. Extra external values are ignored by the runtime after CLI/config
resolution.

Every discovered parameter must have a JSON-wire-serializable type,
even when it has a default and the host does not supply a value. Supported param
types are `text`,
`int`, `decimal`, `bool`, `json`, arrays, dictionaries, records, and enums.
Runtime-only types such as `unit` and function types cannot be used as program
param types. `Agent` is ordinary enum data and is valid wherever an enum is.

## `builtin var` — engine-setting bindings

```ebnf
builtin_var_def ::= "builtin" NEWLINE? "var" name ":" type_expr ["=" expr]
```

A `builtin var` declares a body-less, host-backed, **mutable** binding with a
mandatory type and an optional declared default. Its initializer must be a
constant expression of the declared type: literals, literal containers,
constructor applications, and a unary operator over any of those (`-1`,
`not true`) are allowed; reads, calls other than constructors, and binary
operators are not. The `builtin` marker may be on the same line as `var` or on
the line directly above it. A declaration may appear only at the root, or in a
named scope region, of the canonical standard-library module
`std/config`; declarations in entry programs or other library modules are
static errors regardless of scoping. `std/config` uses it to expose the
program's engine settings:

```agl
import std/config::*

program def main() -> unit =
  std/config::max-iters := 10           # write a setting (qualified target)
  default-agent := AgentClaude("sonnet", "medium") # import tail allows a bare target
  let cap = std/config::max-iters       # read a setting
```

An engine setting is an ordinary mutable binding in another module, so an
assignment target names it exactly as a read does: a qualifier always works,
and an import tail or `use` declaration can contribute a bare name. See
[Modules](modules.md) for import and `use` contributions. When the host
supplies no initial value, the declared default is
used; a host seed wins over it. A write takes effect from its program point
onward, exactly like any `var` mutation. The `Option[text]` settings
are set with `Some("…")` or `None`.

See [Host environment](host-environment.md) for the settings table, their types
and defaults, and how a source write combines with the host's CLI and config-file
layers.

## Agent values

`Agent` is a standard-library enum. Construct an agent with an enum constructor
and store it in ordinary bindings, arrays, or function parameters. The word
`agent` is an ordinary identifier and may also be used as a record field name.

```agl
program def main() -> unit =
  let reviewer = AgentClaude("sonnet", "medium")
  let impl = AgentCommand("claude -p")
  let agents: array[Agent] = [reviewer, impl]
  let r: text = reviewer.ask("Review the artifact")
```

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

### Constructors in the value namespace

Record constructors and injected enum-member constructors are normal bindings
in the value namespace. A field-bearing constructor can be referenced bare, stored, and
passed as a function value:

<!-- agl-check: fragment -->
```agl
let mk: int -> Box[int] = Box   # the constructor as a first-class value
let one = mk(1)                  # called positionally, in field order
```

Direct construction uses positional-greedy binding — positional arguments fill
positional-capable fields first, then named arguments follow (`Box(value = 1)`,
`Some(value = x)`, or `Ok(42)` for a single-standard-field member). A
field-bearing constructor reached **through a variable** is an ordinary function
value invoked **positionally**, in declaration order. A fieldless constructor
reference constructs its value; use `fn() => R1` where a `() -> R1` function is
required. See [Expressions](expressions.md#fieldless-constructor-references)
and [Generics](generics.md) for constructor typing and inference.

### Overload sets, shadowing, and ambiguity

Several visible constructors may share an unqualified member name. In ordinary
value position, a bare reference must resolve to exactly one constructor
candidate in its lexical scope. If it does not, it is a **static scope
ambiguity error**, even when an expected enum type contains one of the
candidates: scope resolves the name before that type is used to check the
expression. **Qualify** the reference with the member's owning enum or record
to disambiguate. Enum-member patterns and `is` tests are different: their
scrutinee's static enum type selects the member rather than using ordinary
value-position scope selection.

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

- An inline **enum member** may be claimed. Its name is `Owner::member`, and
  the unqualified spelling is an injected convenience, so the member remains
  reachable qualified. The claiming declaration owns expression position,
  while case-pattern constructor lookup stays independent.
- A constructor **declared in another module** may be claimed, since module
  qualification still reaches it. This covers the **standard core** names —
  exception types (`Abort`, `AgentParseError`, …), enum members (`Some`,
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

let Red = 5                 # allowed — 'Color::Red' still names the member
let ExecResult = 0          # allowed — declared in another module
def Retry(n: int) -> int = n + 1

scope Collision
record Widget
  x: int

let Widget = 1              # error: 'Widget' is already declared in this scope
end Collision
```

Use [`::name`](modules.md) to reach the module's own top-level declaration past
any shadowing:

<!-- agl-check: fragment -->
```agl
print(::Red)                # the 'let', not the member
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
program def main() -> unit =
  let double = fn(x: int) => x * 2
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
  ask "Uses %{x}"    # inner
ask "Uses %{x}"      # outer
```

### Mutation across scopes

`:=` reaches *through* scopes to the nearest visible mutable binding:

<!-- agl-check: fragment -->
```agl
var artifact: text = impl.ask("Implement %{spec}")

case review of
  | Fail(issues) =>
      artifact := impl.ask("Fix %{issues} in %{artifact}")
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
  let review: Review = reviewer.ask("Review %{artifact}")
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
param default_limit: int = 3

def summarize(doc: text, limit: int = default_limit) -> text =
  "[%{limit}] %{doc}"
```

`default_limit` is resolved at definition time, not call time.
