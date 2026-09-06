# AgL Name Resolution

The scope pass resolves every name in the program, over the whole module graph, and publishes immutable side tables. Typecheck consumes them and never re-resolves; scope in turn never guesses — any ambiguous bare or qualified route is a static error.

## Namespaces and Scopes

Collection builds module-root and named-scope layers pre-populated with static declarations and constructors; bindings and parameters are then installed by an ordered walk that preserves textual visibility. Named-scope declarations are addressed by their full `::` path, and repeated `scope` regions extend one namespace. Type and value namespaces are looked up independently through lexical layers, so a type-only contribution never hides an outer value.

Inline enum members are nominal record declarations beneath their enum's scope. Every constructor reference resolves to a canonical `ConstructorRef` carrying module, scope path, terminal name, and declaration identity; display spellings are never identity. Where a bare constructor spelling in a pattern or `is` test is ambiguous, scope keeps a candidate set and typecheck selects from the matched nominal type.

## Imports and `use`

An import contributes a module's full public qualified surface minus what its `hiding` clause removes. Import tails and `use` declarations add bare names to the lexical region that declares them without narrowing qualified access; `use` selects from routes that are already nameable and never loads a module. Exports and re-exports are resolved program-wide with the same collision rules as local names; routes to one declaration deduplicate, distinct origins stay ambiguous. Loading-level rules are in [modules.md](../modules.md).

## Classification

Scope classifies what typecheck will type: a built-in call is recognized by resolving its callee to a `builtin def` declaration, never by spelling; a `self`-receiver `def` in a nominal scope or on an applied builtin receiver is a method; `builtin var` bindings are host-backed values, of which only root `std/config` bindings are engine settings. Index and field assignment receivers resolve as ordinary reads; typecheck owns container, field, and mutability rules.

## Attribute Recognition

The parser keeps every `@attribute` verbatim; scope is where one acquires meaning. A single walk over the declarations checks each attribute against the catalog in `agl/attributes.py` — the name exists, the declaration is an admitted target, the arguments match the declared schema (including the spelling an option attribute's schema narrows its text to), the attribute is neither repeated nor contradicted — and hands the surviving attributes to small per-attribute fact builders that write typed side-table entries. Recognizing a further attribute adds a fact builder, never another traversal.

The fact built from the `@arg-*` attributes is `param_zones`, the zone of every parameter and field keyed by its node id: an entry's own attribute wins over its declaration's default, which wins over the form's default (standard, but named-only for a `program def`). A `self` receiver is positional-only and admits no zone attribute, and an entry list written out of zone order is an error here rather than in the parser. Typecheck builds every parameter and constructor-field list from this table; the AST itself carries no zone.

The fact built from `@extern-name` is `extern_names`, every `extern def`'s Python companion name keyed by its node id: the attribute's argument, or the declared name verbatim. Scope applies the Python-identifier rule to that effective name, so a name Python could not define is rejected here with the attribute as the remedy, and keys companion-symbol uniqueness on it too — a module's externs share one companion, wherever in the module they are declared, so no two of them may map to the same effective name. Lowering and companion resolution read the table; no name is derived from the AST or mangled.

The facts built from the `@opt-*` attributes and `@doc` are `program_options` and `docs`. `program_options` holds one `ProgramOptionSpec` per `program def` parameter, keyed by node id: the external name a host addresses it by (`@opt-name`, else the declared name) plus its short spelling, environment variable, metavar, hidden flag and help text. Those shapes a host has to spell — a short option is one ASCII letter, an option name a flag word of letters, digits and hyphens that a host can also negate — are the catalog's, checked with every other argument rule; scope additionally rejects the attributes that presuppose a name (`@opt-name`, `@opt-short`, `@opt-env`, `@opt-hidden`) on a positional-only parameter. `docs` holds the `@doc` text of every declaration carrying one, parameters and fields included, keyed by that declaration's node id. Program discovery ([hosting.md](../hosting.md)) hands both to the host.

## Code Entry Points

- `src/agm/agl/scope/attributes.py` — attribute recognition against the catalog and the fact builders it feeds.
- `src/agm/agl/scope/resolver.py` — declaration collection, `use` selection, regional bare contributions.
- `src/agm/agl/scope/imports.py` — import contribution environments and qualified resolution.
- `src/agm/agl/scope/program.py` — export maps, re-exports, cross-module resolution.
- `src/agm/agl/scope/symbols.py` — builtin call names, symbol kinds, and `AttributeFacts`, the recognized-attribute tables the resolution carries.
- Tests: `tests/test_agl_scope*.py`, `test_agl_namespace_*.py`, `test_agl_qualifier_*.py`.
