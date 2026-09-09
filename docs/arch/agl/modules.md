# AgL Modules

AgL programs are file-based module graphs. A module is addressed by a slash path such as `tools/format`; the loader resolves the entry module and its transitive import and re-export dependencies into a `ModuleGraph` before the whole-program passes run.

## Roots and Visibility

Module roots come from the invocation directory, the global library in the AGM home, `[modules]` configuration, and CLI module paths; these are loose roots resolving any path beneath themselves. A package is not a root: each selected package mounts its `src/` tree under its own name, and the selected standard-library root mounts `std` the same way. The invocation directory is dropped when the entry file is owned by a mounted package — including the standard library, when the entry lies in the tree selected as the stdlib root. A path must resolve to exactly one eligible `.agl` file across all roots. Package-owned modules may import only their own tree, the standard library, and their declared dependencies ([packages.md](../packages.md)); loose modules are unrestricted.

`import` adds a graph edge and qualified access to a module's public surface; import tails and `use` add bare names to a region without narrowing that surface; `hiding` subtracts paths. Reachable qualified declaration routes also determine method visibility, while `use` never loads a module or makes methods visible. Imports, uses, and exports may appear inside `scope` regions: qualified routes stay module-wide, bare contributions are regional, and a scoped export re-roots forwarded paths beneath the region. Re-exports are part of the same graph.

The entry module is keyed by the module id its owning package declares for its file, so a package file executed or checked directly is still that module: the rest of its package may import it back, and its parameters and configuration route under that path. An entry no package owns — inline source, a REPL entry, a file under a loose root — has no module identity, is keyed by an anonymous sentinel, cannot be imported, and spells `<entry>` wherever a module route is shown.

## Prelude and Standard-Library Surfaces

Every loaded module except `std/prelude` receives the `std/prelude` prelude unless the host disables it or the module imports `std/prelude` explicitly. The prelude is only that auto-imported entry point (`stdlib/src/prelude.agl` holds export lines): it re-exports the scopes that declare builtin-receiver methods, making those methods visible through ordinary import routes wherever the prelude is loaded. The host recognizes the *standard* declaration of a built-in name as the `builtin` declaration of that name in any standard-library module, wherever it lives, and a `builtin` declaration outside the standard library overrides it. Built-in names nothing declares — under `--no-stdlib`, say — fall back to fixed reserved identities that belong to a sentinel module rather than to any file, and spell bare wherever a name is rendered. Standard-library modules may declare host-backed `builtin var` bindings, identified by module, scope path, and name; only root `std/config` bindings are engine settings. Modules with `extern def` declarations have Python companions ([execution/ffi.md](execution/ffi.md)); the companion-backed surface spans collections, text, JSON, TOML, math, paths, filesystem, process, time, random, and regex. Library conventions — every AgL-visible name is kebab-case (Python companion attributes and host-side keys stay snake_case), an exception type is declared beside the operations raising it, a raising operation's `?` twin returns `Option` and its `try-` twin returns `Result` with the specific exception type, `!` marks in-place operations, and Option-valued operations with no raising form carry no suffix — are stated for users in `docs/agl/reference/modules.md`; `try-` twins are pure AgL over the raising operation, never separate externs.

Text construction also has a pure AgL builder in `stdlib/src/text.agl`: a record backed by the existing mutable array operations and joined into immutable text snapshots. It follows ordinary record/collection aliasing and copying semantics without introducing another runtime value kind.

## Infix Resolution

User-operator chains are resolved once the graph is known, using each module's local declarations plus every operator made bare-visible at the chain's lexical scope. Conflicting fixities in one layer are rejected before scope runs. The graph keeps both full import/export adjacency and source-authored adjacency without loader injections, so execution uses the former while `CheckedProgram.runtime_modules` and dry-run call-site inventories use the latter.

## Compilation Caches

Imported modules are precompiled on demand and reused in memory and across CLI processes. The pipeline retains parsed syntax, resolution, closed type/function interfaces, checked bodies, compiled match sites, and independently lowered module IR. Entry modules are compiled afresh; non-entry members of cycles reaching the entry may be retained against the exact entry source.

`modules/parsed_module_cache.py` assigns content-addressed node namespaces so unchanged declarations keep their identity across import orders and processes. `artifact_cache.py` validates each stage against the module and its transitive import/export dependencies; checked stages also depend on host capabilities. Modules sharing a dependency closure share a persisted frontend image. Restored artifacts attach to the current compilation's source objects, preserving stage provenance.

`lower/module.py` persists module bodies and linkable symbol/function/contract tables. Linking assembles them in the current graph's initialization order and rebuilds whole-program metadata. Resource paths are revalidated, and host-materialized contracts distinguish IR variants. Python companions, configuration, module initializers, and mutable runtime state remain invocation-specific.

`modules/disk_cache.py` stores artifacts under `$XDG_CACHE_HOME/agm/agl` (default `~/.cache/agm/agl`). Source contents, compiler contents, Python and compiler-dependency versions validate reuse; timestamps alone are insufficient. Writes are atomic. `artifact_serialization.py` accepts compiler data classes and source anchors, never arbitrary constructors or host callables. Missing, incompatible, damaged, or inaccessible entries trigger ordinary compilation.

## Code Entry Points

- `src/agm/agl/modules/` — module ids, root assembly, path resolution, graph loading, the parsed-module cache.
- `src/agm/agl/artifact_cache.py` — the cross-compilation artifact cache.
- `src/agm/config/module_roots.py`, `src/agm/packages/` — configured and package-mounted roots.
- `stdlib/src/` — the standard library.
- Tests: `tests/test_agl_modules_*.py`, `test_agl_multifile.py`, `test_agl_parsed_module_cache.py`, `test_agl_artifact_cache.py`, `test_agl_precompilation.py`, `test_agl_stdlib*.py`.
