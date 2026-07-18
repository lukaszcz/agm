# AgL Name Resolution

The scope pass performs full name resolution and records its results in side tables. Pre-passes collect agents, top-level functions, and constructors before any expression body is resolved, so declarations are visible regardless of order and mutual recursion works.

## Namespace-Directed Resolution

Resolution is namespace- and scope-directed, never capitalization-directed — a direct consequence of AgL's case-neutral name model:

- Built-in calls (`print`, `exec`, `ask`, and friends) are recognized by resolving the callee to a known built-in declaration, not by keyword.
- Constructors live in the value namespace; an ambiguous unqualified constructor name is a static error, disambiguated with `Type::Ctor` qualification.
- A bare name in a `case` pattern is a constructor pattern when it names an in-scope constructor and a variable binder otherwise — decided by resolution, not capitalization.

## Static Guarantees

Agents must be declared in source; the pass binds each declared agent as a first-class value of agent type. Register-backed `builtin var` declarations are admitted only in the canonical `std.config` module. The pass enforces lexical control-flow boundaries — `break`/`continue` must stay within a loop in the same function, `return` must appear inside a function body — and the extern (Python FFI) placement rule that externs are only allowed in file-backed modules.

Graph-mode resolution extends this pass across modules; see [../modules.md](../modules.md).

## Code Entry Points

- `src/agm/agl/scope/` — the single-module resolver, graph-aware resolution, and the resolved-program side tables.
- Tests: `tests/test_agl_scope.py`, `tests/test_agl_scope_graph.py`, `tests/test_agl_scope_imports.py`.
