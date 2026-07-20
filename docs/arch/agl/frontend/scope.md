# AgL Name Resolution

The scope pass performs full name resolution and records its results in side tables. Pre-passes collect agents, top-level functions, and constructors before any expression body is resolved, so declarations are visible regardless of order and mutual recursion works.

## Namespace-Directed Resolution

Resolution is namespace- and scope-directed, never capitalization-directed — a direct consequence of AgL's case-neutral name model:

- Built-in calls (`print`, `exec`, `ask`, and friends) are recognized by resolving the callee to a known built-in declaration, not by keyword.
- Constructors live in the value namespace; an ambiguous unqualified constructor name is a static error, disambiguated with `Type::Ctor` qualification.
- A bare name in a `case` pattern is a constructor pattern when it names an in-scope constructor and a variable binder otherwise — decided by resolution, not capitalization.

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

Agents must be declared in source; the pass binds each declared agent as a first-class value of agent type. Register-backed `builtin var` declarations are admitted only in the canonical `std/config` module. The pass enforces lexical control-flow boundaries — `break`/`continue` must stay within a loop in the same function, `return` must appear inside a function body — and the extern (Python FFI) placement rule that externs are only allowed in file-backed modules.

Program resolution extends this pass across modules; see [modules.md](agl/modules.md).

## Code Entry Points

- `src/agm/agl/scope/` — `resolve_module`, `resolve_program`, their resolution side tables, and the pure import-policy models in `imports.py`.
- Tests: `tests/test_agl_scope.py`, `tests/test_agl_scope_program.py`,
  `tests/test_agl_scope_contributions.py`, and `tests/test_agl_namespace_wiring.py`.
