# AgL Modules

AgL has a file-based module system. A program with `import` declarations is compiled as a *module graph* rather than a single file: loading discovers the transitive set of modules, and graph-aware variants of scope, typecheck, and lowering operate over the whole graph at once. Single-file programs bypass this machinery entirely and use the single-module passes. The two modes share the same AST, value model, and evaluator (see [index.md](index.md)).

## Module Identity and Roots

A module is identified by a dotted path mapping to a relative file path; a sentinel id keys the entry program. Modules are found by searching an unordered, canonicalized set of *roots* — the invocation directory, the installed standard library, a global library root, configured roots, and roots from CLI flags. Resolution searches all roots and requires exactly one match: zero matches is "not found", two distinct files for the same module id is "ambiguous". There is deliberately no first-root-wins shadowing. Wildcard imports glob matching files across all roots under global-uniqueness rules.

## Loading

The loader parses the entry source, then does a breadth-first traversal of transitive imports, parsing each module so that node ids are globally unique (disjoint) across all modules — a prerequisite for the shared side-table convention. Traversal terminates when a module is already loaded, which makes import cycles finite and safe; cycles are allowed. An import that resolves back to the entry file is rejected. The result is a module graph carrying each loaded module's AST, canonical path, source id, and imports, plus strongly-connected-component information for diagnostics. All traversal is deterministic regardless of filesystem or root ordering.

## Graph-Aware Passes

The graph passes generalize the single-module passes to whole-graph operation while preserving cross-module forward references and mutual recursion:

- **Scope.** Export maps are computed per module as `{name → (defining_module, name)}` (QName), capturing where each public name was originally defined. Explicit `ExportDecl` nodes add re-exported names to the current module's export map with their origin QName preserved through chains, resolved by a fixed-point iteration. Each import is then mapped to its target module(s), and a per-module import environment provides unqualified and qualified name lookup. A whole-graph pre-pass collects all public declarations before any body is resolved, enabling cross-module mutual recursion. Unqualified references that clash across modules are an error at the reference site; qualified references and self-references resolve to specific modules. Non-entry modules are restricted to declarations (no top-level statements, agents, params, or config declarations).
- **Typecheck.** Type identity becomes module-qualified, so same-named types in different modules are distinct. A two-phase pre-pass registers all type shells across all modules first (so forward references work even with import cycles), then resolves bodies in structural-dependency topological order. A function-signature pre-pass resolves every module's signatures before any body is checked, so cross-file mutual recursion does not depend on module checking order.
- **Lowering.** The checked graph links into one executable program with all modules, exports, functions, nominals, contracts, and sources resolved to stable ids before evaluation. Library initializers run in dependency order and the entry module runs last; cross-module calls use linked ids directly, never resolver or checker side tables.

## Source Attribution

Every source span carries a source id identifying its origin file, so multi-file diagnostics name the right file. Span positions compare equal across files (the source id is excluded from equality), consistent with how node ids are excluded from AST equality. Diagnostics carry the source label through to formatting.

## Code Entry Points

- `src/agm/agl/modules/` — module identity, root sets, resolver, and the BFS loader.
- `src/agm/agl/scope/` — graph-aware resolution and the import-environment builder (alongside the single-module resolver).
- `src/agm/agl/typecheck/` — graph-aware type checking and module-qualified type identity.
- `src/agm/agl/lower/` — graph linking into the executable program.
- `src/agm/agl/syntax/` — source-id-stamped spans.
- `src/agm/config/module_roots.py` — module-root configuration.
- Tests: `tests/test_agl_modules_*.py`, `tests/test_agl_scope_graph.py`, `tests/test_agl_scope_imports.py`, `tests/test_agl_typecheck_graph.py`, `tests/test_agl_multifile.py` (fixtures under `tests/agl/multi_file/`).
