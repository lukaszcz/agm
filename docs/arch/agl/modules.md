# AgL Modules

AgL programs are file-based module graphs. A module is addressed by a slash
path such as `tools/format`, and its public declarations are available to
importers through qualified routes. The loader resolves the entry module and
its transitive imports before the program-level scope, typecheck, match
compilation, and lowering passes run.

## Loading and Visibility

Module roots come from the invocation, standard library, configured roots, and
mounted packages. A path must resolve to exactly one eligible `.agl` file;
ambiguous paths are errors. Package roots limit visibility to the package's
declared module tree.

`import` adds a module-graph edge and qualified access to a module's public
surface. Import tails and `use` declarations add bare names without narrowing
that qualified surface; `hiding` removes paths from the declaration that uses
it. `use` resolves a local or already imported route and never loads a module.
Imports, uses, and exports may occur in named scope regions, where their bare
contributions apply to that region. Re-exports form part of the same program
graph.

Every loaded module except `std/core` receives the `std/core` prelude unless
the host disables it. An explicit `import std/core` supplies that module's
core contribution instead.

## Program Passes

The loader produces a `ModuleGraph` for whole-program passes. Scope resolves
declarations and import contributions across the graph; typecheck, match
compilation, and lowering then process the same graph. Module roots are static:
workflow code belongs in a `program def` body.

## Code Entry Points

- `src/agm/agl/modules/` — module identities, roots, resolution, and graph loading.
- `src/agm/agl/scope/` — import contributions, exports, and whole-program name resolution.
- `src/agm/agl/pipeline.py` — orchestration of the program passes.
- `src/agm/config/module_roots.py` and `src/agm/packages/` — configured and package-mounted roots.
- Tests: `tests/test_agl_modules_*.py`, `tests/test_agl_multifile.py`, and
  `tests/test_agl_scope_program.py`.
