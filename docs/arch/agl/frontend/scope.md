# AgL Name Resolution

The scope pass resolves every name in an AgL program and records the results
in immutable side tables. It runs after parsing and before typecheck, over the
whole module graph when a program imports files.

## Namespaces and Scopes

Collection builds module-root and named-scope layers and pre-populates them
with static function/type declarations and constructors. Bindings and
parameters are installed during the subsequent ordered resolution walk, which
preserves their textual visibility. Declarations in a named scope are addressed
by their complete `::` path; repeated scope regions extend the same namespace.
The resolver applies lexical visibility for bindings and resolves qualified
chains through local scope paths and imported module routes. Ambiguous bare or
qualified routes are static errors.

## Imports and `use`

`scope/imports.py` builds contribution environments for import declarations.
Program resolution tracks named-scope existence separately from declaration exports, so an imported empty
scope remains a nameable `use` target without becoming a value. An import contributes its full public
qualified surface, except paths hidden by that declaration. A positive
import tail or `use` declaration contributes selected bare names without
narrowing qualified access. A wildcard import alias retains its declaration
identity as a shared facade; unrelated imports that reuse an alias remain
ambiguous. `use` selects from an already nameable local scope or imported route,
including a scope route exposed by an earlier `use`; it does not create a
module-loading edge. Region-scoped
bare contributions apply within that region, while imports still make their
qualified routes available to the module.

Scope resolution also classifies declarations, bindings, constructors, and
built-ins for typecheck. It records each `use` target's semantic local path or
imported routes so incremental hosts retain target identity without re-deriving
it from syntax. It publishes resolved program artifacts rather than rewriting
source nodes.

## Code Entry Points

- `src/agm/agl/scope/` — whole-program resolution and resolution side tables.
- `src/agm/agl/scope/imports.py` — import contribution environments and qualified resolution.
- `src/agm/agl/scope/program.py` — export maps, re-exports, and cross-module resolution.
- `src/agm/agl/scope/resolver.py` — declaration collection, `use` selection, and regional bare
  contributions.
- Tests: `tests/test_agl_scope*.py`, `tests/test_agl_namespace_*.py`, and
  `tests/test_agl_qualifier_*.py`.
