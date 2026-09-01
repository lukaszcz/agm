# AgL Name Resolution

The scope pass resolves every name in the program, over the whole module graph, and publishes immutable side tables. Typecheck consumes them and never re-resolves; scope in turn never guesses — any ambiguous bare or qualified route is a static error.

## Namespaces and Scopes

Collection builds module-root and named-scope layers pre-populated with static declarations and constructors; bindings and parameters are then installed by an ordered walk that preserves textual visibility. Named-scope declarations are addressed by their full `::` path, and repeated `scope` regions extend one namespace. Type and value namespaces are looked up independently through lexical layers, so a type-only contribution never hides an outer value.

Inline enum members are nominal record declarations beneath their enum's scope. Every constructor reference resolves to a canonical `ConstructorRef` carrying module, scope path, terminal name, and declaration identity; display spellings are never identity. Where a bare constructor spelling in a pattern or `is` test is ambiguous, scope keeps a candidate set and typecheck selects from the matched nominal type.

## Imports and `use`

An import contributes a module's full public qualified surface minus what its `hiding` clause removes. Import tails and `use` declarations add bare names to the lexical region that declares them without narrowing qualified access; `use` selects from routes that are already nameable and never loads a module. Exports and re-exports are resolved program-wide with the same collision rules as local names; routes to one declaration deduplicate, distinct origins stay ambiguous. Loading-level rules are in [modules.md](../modules.md).

## Classification

Scope classifies what typecheck will type: a built-in call is recognized by resolving its callee to a `builtin def` declaration, never by spelling; a `self`-receiver `def` in a nominal scope or on an applied builtin receiver is a method; `builtin var` bindings are host-backed values, of which only root `std/config` bindings are engine settings. Index and field assignment receivers resolve as ordinary reads; typecheck owns container, field, and mutability rules.

## Code Entry Points

- `src/agm/agl/scope/resolver.py` — declaration collection, `use` selection, regional bare contributions.
- `src/agm/agl/scope/imports.py` — import contribution environments and qualified resolution.
- `src/agm/agl/scope/program.py` — export maps, re-exports, cross-module resolution.
- `src/agm/agl/scope/symbols.py` — builtin call names and symbol kinds.
- Tests: `tests/test_agl_scope*.py`, `test_agl_namespace_*.py`, `test_agl_qualifier_*.py`.
