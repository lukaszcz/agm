# AgL Name Resolution

The scope pass performs full name resolution and records its results in side tables. Pre-passes collect agents, top-level functions, and constructors before any expression body is resolved, so declarations are visible regardless of order and mutual recursion works.

## Namespace-Directed Resolution

Resolution is namespace- and scope-directed, never capitalization-directed — a direct consequence of AgL's case-neutral name model:

- Built-in calls (`print`, `exec`, `ask`, and friends) are recognized by resolving the callee to a known built-in declaration, not by keyword.
- Constructors live in the value namespace; an ambiguous unqualified constructor name is a static error, disambiguated with `Type::Ctor` qualification.
- A declaration may claim a constructor's unqualified spelling exactly when the constructor stays reachable some other way: enum variants (reachable as `Owner::variant`) and constructors owned by another module (reachable by module qualification) yield to it, while a same-module record, exception, or alias constructor — whose declaration *is* the bare name — collides as a duplicate declaration.
- Scope records constructor candidates for bare pattern names independently of ordinary value bindings. Top-level names are constructor-only; nested field-directed names resolve to immutable pattern-slot `BindingRef`s, with slot metadata recording candidates and visible alternatives, grouped per case branch so typechecking selects exactly the branch it just checked. Candidate metadata retains whether a spelling can be a bare nullary enum pattern, allowing scope to reject definite duplicate binders eagerly while leaving genuine field-directed cases to typechecking. Typechecking selects each slot's final binder or constructor in checker-owned maps; consumers use the checked artifact's accessors for those meanings. No later pass rewrites scope's resolution tables. An `as`-pattern name is always a variable binder.

Assignment follows the same split. Scope resolves an unqualified `:=` target and rejects an undeclared name, but leaves assignability to typechecking, which alone knows which binding a pattern slot selected. A qualified target is settled in scope, since only `builtin var` is assignable across a module boundary and no qualified name is a pattern slot.

## Import Environments

`scope/imports.py` is the pure import-policy seam. Its contribution environment
merges every declaration for a module into its selected members, bare injection, aliases,
and plain-path routes. The selected set bounds both routes and bare injection: plain
imports are qualified-only, while `using` and `open import` inject bare names. One shared
suffix/anchored resolver serves value reads and writes,
constructors, and type qualification, retaining ambiguity and route identity until the use
site; its diagnostics distinguish private declarations from names outside a contribution.
One shared translator walks those verdicts and raises an error the caller constructs, so
the scope and typecheck passes share the walk while keeping their own exception types and
wording.
Whitespace-separated qualifier near-misses are reported from the lexer's advisories
rather than reconstructed from AST shapes: when a reference fails to resolve at an offset
an advisory covers, the pass offers the tight spelling — but only when re-resolving that
route actually contributes the intended member, preserving valid division and
juxtaposition expressions.

## Static Guarantees

Agents must be declared in source; the pass binds each declared agent as a first-class value of agent type. `let _ = value` and `var _ = value` still resolve their right-hand sides but register no binding, so `_` cannot be read and may be repeated. Register-backed `builtin var` declarations are admitted only in the canonical `std/config` module. The pass enforces lexical control-flow boundaries — `break`/`continue` must stay within a loop in the same function, `return` must appear inside a function body — and the extern (Python FFI) placement rule that externs are only allowed in file-backed modules.

Program resolution extends this pass across modules; see [modules.md](agl/modules.md).

## Code Entry Points

- `src/agm/agl/scope/` — `resolve_module`, `resolve_program`, their resolution side tables, and the pure import-policy models in `imports.py`.
- Tests: `tests/test_agl_scope.py`, `tests/test_agl_scope_program.py`,
  `tests/test_agl_scope_imports.py`, `tests/test_agl_scope_contributions.py`,
  `tests/test_agl_namespace_wiring.py`, and `tests/test_agl_pattern_slots.py`.
