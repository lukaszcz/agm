# Packages

The package domain defines portable, versioned AgL module collections. A package is a `package.toml` manifest plus a module tree named after the package, so its modules import under a stable package-qualified path. A manifest may declare version-floor dependencies, literal resources, and CLI commands backed by parameterless `program def` entries. The store, installation, and the managed `std` package are covered in [package-store.md](package-store.md).

## Package Model

- **Identity** is the package name plus its complete canonical version, build metadata included. Semantic precedence still governs minimum-version requirements and highest-version selection.
- **Dependencies** are unbounded minimums resolved by minimal version selection: one version per name across the whole closure, with a path-sourced development checkout taking precedence over the store. `std` is special: its floor must fall within the compatible AGM release line.
- **Development packages** are live source checkouts discovered from an execution root or a source file's containing tree. A file inside one gets that package's dependency closure and keeps its package-qualified module path for configuration; immutable store entries are never rediscovered as development source.
- **Import visibility** is enforced by the AgL module loader: package roots mount only each selected package's declared module tree, and a package module may import only its own tree, its declared dependencies, and the standard library. Loose modules remain unrestricted.

## Discipline Validation

`agm pkg check` validates a development package without modifying it. Source-free manifest and module-tree checks run first; dependency-aware validation then loads and name-resolves the package module graph once, with runtime visibility, rejecting unresolved modules, imports outside the dependency closure, command programs that name no valid entry, and missing resource targets. Resource call sites come from the scope pass's own builtin classification. The same validation runs during `pkg create` (against the source tree and the archive contents) and `pkg install` (against the staged tree).

## Registered Commands

A manifest `[commands]` table maps a single- or multi-word CLI path to a package-owned `program def`. Activation rejects command conflicts unless the later installation passes `--shadow`. Each invocation derives its effective command registry from the selected package manifests — so project `[packages]` pins affect dispatch, help, and completion — and CLI dispatch resolves the longest registered path and runs it through the same execution host as `agm exec` after verifying manifest and module ownership ([cli.md](cli.md)).

## Façade

The `agm.packages` public façade resolves its exports lazily. Package model leaves stay available to AgL module loading without pulling in discipline validation, which depends on AgL scope resolution.

## Code Entry Points

- `src/agm/packages/manifest.py` — manifest schema and the distribution-manifest view.
- `src/agm/packages/model.py`, `development.py` — package identity, version selection, `std` compatibility bounds, development-package discovery.
- `src/agm/packages/dependencies.py` — dependency-closure resolution.
- `src/agm/packages/discipline.py` — manifest, tree, and graph validation for directory and archive packages.
- `src/agm/packages/__init__.py` — the lazy façade; `layout.py` — pure store-path constants usable from the config layer.
- `src/agm/agl/modules/roots.py`, `loader.py` — package-root mounting and import visibility.
- `src/agm/commands/pkg/` — the package CLI; `src/agm/cli_dispatch.py`, `src/agm/commands/exec_program.py` — registered-command lookup and execution.
- `tests/test_packages_*.py`, `tests/agl/packages/` — package-domain and package-program coverage.
