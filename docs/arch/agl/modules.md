# AgL Modules

AgL has a file-based module system. Every production program is loaded into a `ModuleGraph`: the entry module and its transitive imports. Unless the host disables it, every loaded entry and library module except `std/core` itself receives an automatic `open import std/core` prelude. Program-level scope, typecheck, match compilation, and lowering passes orchestrate per-module workers over that loaded program. The AST, value model, and evaluator are shared throughout (see [index.md](agl/index.md)).

## Module Identity and Roots

A module is identified by a slash-separated path (`foo/bar`) mapping to a relative `.agl` file; a sentinel id keys the entry program. `ModuleId` construction and display use this path form, while its segments remain the shared internal identity. Modules are found by searching an unordered, canonicalized set of *roots* — the invocation directory, the installed standard library, a global library root, configured roots, and roots from CLI flags. Resolution requires exactly one match across all roots: zero is "not found", two distinct files is "ambiguous". There is deliberately no first-root-wins shadowing. Wildcard imports glob matching files across all roots under the same global-uniqueness rules.

## Loading

The loader parses the entry source, then does a breadth-first traversal of transitive imports, parsing each module so that node ids are globally unique across all modules — a prerequisite for the shared side-table convention. Traversal terminates when a module is already loaded, which makes import cycles finite and safe; cycles are allowed, but an import that resolves back to the entry file is rejected. The result is a module graph carrying each module's AST, canonical path, source id, and imports, plus strongly-connected-component information for diagnostics. Traversal is deterministic regardless of filesystem or root ordering.

A module declaring an `extern def` (Python FFI) needs a companion Python file at its own path with a `.py` suffix. The companion path is derived from the module's canonical path, not searched, so root-ambiguity rules never apply to it; the loader records it and verifies it exists during graph loading, while the import itself happens in the execution layer ([execution/evaluator.md](agl/execution/evaluator.md)).

## Program Passes

Program passes coordinate the module workers while preserving cross-module forward references and mutual recursion:

- **Scope.** A whole-program pre-pass computes per-module export maps — each public name mapped to its originating module, with re-export chains resolved to their origin — and collects all public declarations before any body is resolved. Import declarations merge into per-module contributions: each selected set bounds both route access and bare injection; plain imports are route-only, while `using` and `open import` add bare candidates. Aliases replace plain-path routes, and wildcard declarations expand per module. Suffix (`suffix::name`), anchored (`/path::name`), self (`::name`), constructor, and type references all use the same contribution resolver; ambiguities fail at the use site. Non-entry modules are restricted to declarations (no top-level statements, agents, or params).
- **Typecheck.** Type identity becomes module-qualified, so same-named types in different modules are distinct. Because nominal types are shape-free handles ([frontend/types.md](agl/frontend/types.md)), declarations resolve in a fixed order with no dependency ordering — forward references and type cycles spanning modules work exactly like same-module ones, while alias cycles remain illegal. The resolved-program artifact preserves the loader's reverse-topological import SCCs. After collecting explicit, builtin, and extern headers, typecheck traverses those SCCs dependencies first: closed unannotated candidates publish by declaration node id before importers, while an import cycle shares one cross-module function graph. It then authoritatively checks each module body with the completed concrete signature table. The whole-table analyses run once over the shared `TypeTable`.
- **Match compilation.** Every source case in every checked module is compiled, including cases entry code never calls. The program artifact carries a total per-module mapping; any coverage or redundancy issue prevents lowering and is reported with its originating source label.
- **Lowering.** The program links into one executable program; library initializers run in dependency order with the entry module last, and cross-module calls use linked ids directly, never resolver or checker side tables.

## Source Attribution

Every source span carries a source id identifying its origin file, so multi-file diagnostics name the right file. Span positions compare equal across files (the source id is excluded from equality), consistent with how node ids are excluded from AST equality. Diagnostics carry the source label through to formatting, including related notes that can point into other modules.

## Code Entry Points

- `src/agm/agl/modules/` — module identity, root sets, resolver, and the BFS loader.
- `src/agm/agl/scope/`, `src/agm/agl/typecheck/`, `src/agm/agl/matchcompile/`, `src/agm/agl/lower/` — program passes and their per-module workers.
- `src/agm/config/module_roots.py` — module-root configuration.
- Tests: `tests/test_agl_modules_*.py`, `tests/test_agl_scope_program.py`, `tests/test_agl_scope_imports.py`, `tests/test_agl_typecheck_program.py`, `tests/test_agl_multifile.py` (fixtures under `tests/agl/multi_file/`).
