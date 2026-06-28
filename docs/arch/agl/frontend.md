# AgL Frontend

The frontend turns source text into a fully resolved, type-checked program. It is four passes — lexer, parser, scope, typecheck — over one shared AST. Everything here is static: no agent calls, no shell execution, no evaluation. See [index.md](index.md) for how the frontend sits in the overall pipeline.

## Lexer and Parser

The lexer is hand-written because AgL is indentation-sensitive: it produces INDENT/DEDENT tokens, handles multiline strings and string interpolation, and emits a single case-neutral `NAME` token for every identifier. The parser is a Lark LALR grammar that consumes those tokens and an AST builder constructs the AST. These two passes are the only Lark-aware code in the system.

## AST

The AST is plain frozen dataclasses with no parser types — the firewall the rest of the implementation depends on. Because AgL is expression-oriented, there is no statement/expression split: a single unified node family covers blocks, bindings, assignment, control flow (`if`/`case`/`loop`/`try`), and a single call node for every kind of invocation (user functions, built-ins, and function values, with the bare-argument call sugar desugaring to the same node). Casts (`as` / `as?`), indexing, lambdas and named function definitions, the unit literal, divergence (`raise`), and loop-control (`break`/`continue`) are all expression nodes. Type-level nodes describe the surface type syntax, including generic type applications.

The `Loop` node carries four optional header fields — `for_var` (the loop variable name), `for_iter` (the collection expression or, for a range `for`, the lower bound `a`), `while_cond` (the per-iteration boolean guard), and `bound` (the iteration-count expression) — plus three integer-range fields (`for_range_to`, the upper bound `b`; `for_range_down`, a boolean flag for `downto`; `for_range_by`, the optional step expression), a `body`, and `until_cond`. A range `for` is distinguished by `for_range_to` being non-`None`; the collection `for` leaves all three range fields as `None`. `Break` and `Continue` are leaf nodes with no children.

Each node carries a stable id assigned at build time. Later passes never mutate nodes; they record their conclusions in side tables keyed by that id (carried in the resolved and checked program objects). This is the universal annotation convention — it is why nodes can be frozen and shared, and why `id()`-based identity is never used.

## Scope and Name Resolution

The scope pass performs full name resolution and records its results in side tables. It runs pre-passes before resolving expression bodies so that declarations are visible regardless of order: agents and top-level functions are collected first, enabling mutual recursion, and constructors (from record and enum declarations) are collected into the ordinary value namespace.

Resolution is namespace- and scope-directed, never spelling-directed — a direct consequence of the case-neutral `NAME` token:

- Built-in calls (`print`, `exec`, `ask`, and friends) are recognized by resolving the callee to a known built-in declaration and recorded as such, rather than by keyword.
- Constructors live in the value namespace; an ambiguous unqualified constructor name is a static error, disambiguated by type qualification.
- A bare name in a `case` pattern is treated as a constructor pattern when it names an in-scope constructor and otherwise as a variable binder — decided by resolution, not capitalization.

Agents must be declared in source; the scope pass binds each declared agent as a first-class value of agent type. The resolver tracks a loop-context flag, reset at function/lambda boundaries, to catch `break`/`continue` used outside any enclosing loop at resolution time. For-loop variables are bound with `BinderKind.loop_var_binding` (immutable, `:=` is a static error) and are visible in the `while` guard, the body, and the `until` condition, but not outside the loop. Graph-mode resolution extends this pass across modules; see [modules.md](modules.md).

## Type System

The semantic type model lives in the `semantics` foundation package and is consumed by the typecheck pass. Alongside the ordinary scalar, container, record, and enum types it carries the types the expression-oriented design needs: a unit type for side-effecting expressions, a positional function type, an opaque agent type, a bottom type for `raise`/`break`/`continue`, and rigid type variables for generics. Records and enums have nominal identity by name (and, in graph mode, owning module), not by structure.

The pass selects concrete behavior that the evaluator later relies on:

- **Built-in typing rules** are dispatched from the resolver's built-in classification — for example `print` accepts anything and yields unit; `exec` chooses between returning a structured result and parsing stdout into a target type; `ask` takes its result type from context.
- **Casts** are validated against a table of permitted source/target pairs, each classified as total or fallible; `as?` always types as a boolean.
- **Generics** use rank-1 parametric polymorphism. A generic definition is checked with its type parameters as rigid, opaque variables (enforcing parametricity), and a small one-sided matching solver infers type arguments for both explicit and inferred calls. Type arguments exist only during checking — they are erased before execution.
- **For-loop iterable typing** branches on the loop kind. For a collection `for` (`for_range_to` is `None`), it derives the element type from the collection type: `list[T]` → `T`, `dict[text, V]` → `text`, `text` → `text`; any other type is a static error. For a range `for` (`for_range_to` is non-`None`), the start expression, upper bound, and step (if present) must each be `int`; the loop variable is typed `int`. A literal non-positive `by` step is also rejected here as a static error. The `to`/`downto`/`by` keywords are admitted as field names (same rule as `agent`), so `tagged(by: value)` remains valid.

Output contracts (the schema and format metadata that agent/`exec` calls need) are computed here while types are available, then compiled into the typeless execution layer described in [execution.md](execution.md).

## Code Entry Points

- `src/agm/agl/lexer/` — the indentation-aware lexer.
- `src/agm/agl/grammar/` and `src/agm/agl/parser/` — the Lark grammar and AST builder.
- `src/agm/agl/syntax/` — the AST dataclasses, type nodes, and source spans.
- `src/agm/agl/scope/` — name resolution and the resolved-program side tables.
- `src/agm/agl/typecheck/` — type checking, built-in typing rules, casts, and generics.
- `src/agm/agl/semantics/` — the shared value model, semantic types, and exceptions.
- Tests: `tests/test_agl_lexer.py`, `tests/test_agl_parser.py`, `tests/test_agl_ast.py`, `tests/test_agl_scope.py`, `tests/test_agl_typecheck.py`, `tests/test_agl_types.py`.
