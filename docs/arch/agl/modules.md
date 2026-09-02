# AgL Modules

AgL programs are file-based module graphs. A module is addressed by a slash path such as `tools/format`; the loader resolves the entry module and its transitive import and re-export dependencies into a `ModuleGraph` before the whole-program passes run.

## Roots and Visibility

Module roots come from the invocation directory, the selected standard library, the global library in the AGM home, `[modules]` configuration, CLI module paths, and mounted packages. A path must resolve to exactly one eligible `.agl` file across all roots. Package-owned modules may import only their own tree, the standard library, and their declared dependencies ([packages.md](../packages.md)); loose modules are unrestricted.

`import` adds a graph edge and qualified access to a module's public surface; import tails and `use` add bare names to a region without narrowing that surface; `hiding` subtracts paths. `use` never loads a module. Imports, uses, and exports may appear inside `scope` regions: qualified routes stay module-wide, bare contributions are regional, and a scoped export re-roots forwarded paths beneath the region. Re-exports are part of the same graph.

## Prelude and Standard-Library Surfaces

Every loaded module except `std/prelude` receives the `std/prelude` prelude unless the host disables it or the module imports `std/prelude` explicitly. Standard-library modules may declare host-backed `builtin var` bindings, identified by module, scope path, and name; only root `std/config` bindings are engine settings. `std/builtin-methods` is an optional registry importing the modules that declare methods on structural and scalar builtin receivers; the loader injects it as an *ambient* module — checked, linked, and initialized with every program, but not an import contribution to user modules. Modules with `extern def` declarations have Python companions ([execution/ffi.md](execution/ffi.md)); the companion-backed surface spans collections, text, JSON, TOML, math, paths, filesystem, process, time, random, and regex.

## Infix Resolution

User-operator chains are resolved once the graph is known, using each module's local declarations plus every operator made bare-visible at the chain's lexical scope. Conflicting fixities in one layer are rejected before scope runs. The graph keeps both full import/export adjacency and source-authored adjacency without loader injections, so execution uses the former while parameter discovery and dry-run inventories use the latter.

## Compilation Caches

Two process-global caches make repeated compilation cheap, and neither privileges the standard library:

- The **parsed-module cache** parses each file once per process, validated against the source text itself rather than stat metadata, and draws node ids from a reserved band so cached modules stay disjoint from every graph they are served into.
- The **artifact cache** (`artifact_cache.py`) is a bounded LRU of what the passes derived per module — resolved modules, checked modules, compiled match sites. An artifact is a pure function of the loaded modules it could read (the module, its transitive dependencies, the ambient method modules), so it is served again only while every one of those is the very same parsed object; an edited file reparses and misses by construction. The entry module is never cached, and host capabilities key the artifacts checked under them.

## Code Entry Points

- `src/agm/agl/modules/` — module ids, root assembly, path resolution, graph loading, the parsed-module cache.
- `src/agm/agl/artifact_cache.py` — the cross-compilation artifact cache.
- `src/agm/config/module_roots.py`, `src/agm/packages/` — configured and package-mounted roots.
- `stdlib/std/` — the standard library.
- Tests: `tests/test_agl_modules_*.py`, `test_agl_multifile.py`, `test_agl_parsed_module_cache.py`, `test_agl_artifact_cache.py`, `test_agl_stdlib*.py`.
