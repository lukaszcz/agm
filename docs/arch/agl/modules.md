# AgL Modules

AgL programs are file-based module graphs. A module is addressed by a slash
path such as `tools/format`, and its public declarations are available to
importers through qualified routes. The loader resolves the entry module and its transitive import and re-export
dependencies before the program-level scope, typecheck, match compilation, and
lowering passes run.

## Loading and Visibility

Module roots come from the invocation, selected standard library, global library,
configuration, CLI module paths, and mounted packages. A path must resolve to
exactly one eligible `.agl` file; ambiguous paths are errors. Package-owned
modules can depend only on their own module tree, the standard library, and
mounted dependencies declared by their package; loose modules remain unrestricted.

`import` adds a module-graph edge and qualified access to a module's public
surface. Import tails and `use` declarations add bare names without narrowing
that qualified surface; `hiding` removes paths from the declaration that uses
it. `use` resolves a local or already imported route and never loads a module.
Imports, uses, and exports may occur in named scope regions. An import's
qualified routes remain module-wide, while import-tail and `use` bare
contributions apply only to that region and its descendants. A scoped export
instead re-roots forwarded paths beneath the region and contributes no regional
bare names. Re-exports form part of the same program graph.

Every loaded module except `std/core` receives the `std/core` prelude unless
the host disables it. An explicit import that includes `std/core`, directly or
through wildcard expansion, supplies that module's core contribution instead.

Loading resolves raw infix chains once the graph is known, using each module's
local declarations plus every operator made bare-visible at the chain's lexical
scope — through an import tail, a `use`, a scoped re-export, or the `std/core`
prelude. Nearest bare layers determine visibility; conflicting fixities in one
layer and mixed associativity at one priority are rejected before scope. The
graph carries both full import/export adjacency and explicit-source adjacency
excluding loader injections, so execution uses the former while inventories use
the latter.

Every imported module is parsed once per process and served from a shared cache
afterwards. Nothing privileges the standard library: a module is cacheable
because its source is unchanged, so user and package modules are served on the
same terms. Cached modules draw node ids from a reserved high band ordinary
graph allocation never reaches, so they stay disjoint from every graph they are
served into, and they are cached before infix resolution, which still runs per
graph. An entry is validated against the source text itself rather than the
file's stat metadata, which cannot separate an in-place rewrite of the same
length under a preserved modification time — the case a process that both
writes and compiles modules actually reaches.

Infix resolution runs per graph, but a caller that already resolved some of a
graph's modules — the REPL, entry after entry — can name them, and their
programs are left untouched. Their operator declarations still take part in the
fixity fixed point, because the modules being resolved may import them.

## Standard-Library Surfaces

Standard-library modules may declare host-backed `builtin var` bindings. Their
runtime identity is the defining module, scope path, and binding name, so
same-named scoped declarations stay independent; only root `std/config`
bindings use engine-setting registers, while scoped `std/config` and
domain-module bindings are ordinary ambient values.

A standard library may provide `std/builtin-methods`, an optional registry
importing the modules that declare methods on structural and scalar builtin
receivers. With default-stdlib loading enabled the loader injects that registry
when it exists; `--no-stdlib`, or a library that omits it, leaves it out. A
loader-injected registry is **ambient**: checked, linked, and initialized with
every selected program, but not an import contribution to user modules. Its
receiver methods register globally by builtin constructor, while the owning
modules' free functions keep ordinary explicit-import visibility. A
source-level registry import is an ordinary dependency. Ambient modules are
virtual dependencies only for candidate-inference ordering, so their closed
unannotated method signatures reach consumers without changing the source
import/export graph, its cycles, or scope visibility.

A module declaring an `extern def` needs a companion Python file at its own
canonical path with a `.py` suffix ([execution/evaluator.md](execution/evaluator.md)).
The companion-backed surface covers `std/array` and `std/dict` for collection
operations and callback-based transforms, `std/text` for runtime interpolation
and text methods, `std/json` for parsing and inspection, `std/toml` for
TOML/JSON conversion, `std/math` for decimal numerics and integer power,
`std/path` for lexical path manipulation, `std/fs` for typed-error filesystem
effects, `std/process` for process metadata and controlled host termination,
`std/time` for clocks and timestamp conversion, `std/random` for seedable random
operations and live-array shuffling, and `std/regex` for Python `re` matching,
rewriting, and splitting. `std/fs` delegates to the shared dry-run-aware
filesystem primitive layer, so it follows the same invocation
working-directory behavior as other host filesystem operations.

## Program Passes

The loader produces a `ModuleGraph` for whole-program passes. Scope resolves
declarations and import contributions across the graph; typecheck, match
compilation, and lowering then process the same graph. File-backed module roots
are static: workflow code belongs in a `program def` body. Incremental REPL entries
are the executable-root exception.

Typecheck derives candidate-inference SCCs from the loader's reverse-topological
import SCCs by adding ordering-only ambient-method dependencies, so ambient
method signatures are closed before any consumer is inferred; the original
import SCCs remain the execution graph. Parameter discovery and dry-run
inventory follow source-authored import/export edges, so a directly imported
registry member stays visible while loader-injected modules and transitive
ambient implementations do not.

## The Artifact Cache

Almost every compilation in a process sees the same modules behind a different
entry, so re-deriving their artifacts dominates the cost of a short program.
`artifact_cache.py` is a process-global, bounded LRU store of what the passes
derived per module: resolved modules, checked modules, and compiled match sites.
Scope, typecheck, and match compilation each consult it through the
`cached_modules`/`cached_checked_modules`/`cached_sites` parameters they already
accepted for the REPL, and refresh it with what they produced; a caller-supplied
image (a REPL session's) takes precedence.

An artifact is reusable because it is a pure function of the loaded modules it
was derived from, never of the entry. So each is retained alongside the modules
its derivation could read — the module, its transitive dependencies, and the
ambient method modules — and is served again only while every one of those is
the very same object. The parsed-module cache is what makes that identity hold
across compilations. That identity condition is the whole condition, which is
why nothing is keyed by root set and nothing is confined to the standard
library: a root set naming different files yields different loaded modules and
misses, one that merely adds unrelated roots hits, and a user module edited
between two compilations reparses and so misses by construction. Only the entry
module is excluded, being the thing compiled. Host capabilities are not
derivable from modules, so they key the artifacts checked against them.

## Code Entry Points

- `src/agm/agl/modules/` — module identities, roots, resolution, graph loading, and the
  parsed-module cache (`parsed_module_cache.py`); tests in `tests/test_agl_parsed_module_cache.py`.
- `src/agm/agl/artifact_cache.py` — the cross-compilation artifact cache; tests in
  `tests/test_agl_artifact_cache.py`.
- `src/agm/agl/scope/` — import contributions, exports, and whole-program name resolution.
- `src/agm/agl/pipeline.py` — orchestration of the program passes.
- `src/agm/config/module_roots.py` and `src/agm/packages/` — configured and package-mounted roots.
- Tests: `tests/test_agl_modules_*.py`, `tests/test_agl_multifile.py`, and
  `tests/test_agl_scope_program.py`.
