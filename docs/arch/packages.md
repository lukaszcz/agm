# Packages

The package domain defines portable, versioned AgL module collections. A package is a `package.toml` manifest plus a module tree in its `src/` directory, mounted under the package name, so its modules import under a stable package-qualified path. A manifest may declare version-floor dependencies, literal resources, and CLI commands backed by `program def` entries with no type parameters; a program may equally register its own command with `@command`. Value parameters project onto the command's CLI surface exactly as they do for `agm exec`. `agm pkg init` scaffolds that minimum for a new development package. The store, installation, and the managed `std` package are covered in [package-store.md](package-store.md).

## Package Model

- **Identity** is the package name plus its complete canonical version, build metadata included. Semantic precedence still governs minimum-version requirements and highest-version selection.
- **Dependencies** are unbounded minimums resolved by minimal version selection: one version per name across the whole closure, with a path-sourced development checkout taking precedence over the store. `std` is special: its floor must fall within the compatible AGM release line.
- **Development packages** are live source checkouts discovered from an execution root or a source file's containing tree. A file inside one gets that package's dependency closure and keeps its package-qualified module path for configuration; immutable store entries are never rediscovered as development source.
- **Import visibility** is enforced by the AgL module loader: a package root is never a search root — each selected package mounts only its `src/` tree, under its package name — and a package module may import only its own tree, its declared dependencies, and the standard library. Loose roots and the modules they own remain unrestricted.

## Discipline Validation

`agm pkg check` validates a development package without modifying it. Source-free manifest and module-tree checks run first; dependency-aware validation then loads and name-resolves the package module graph once, with runtime visibility, rejecting unresolved modules, imports outside the dependency closure, command programs that name no valid entry, and missing resource targets. Resource call sites come from the scope pass's own builtin classification. The same validation runs during `pkg create` (against the source tree and the archive contents) and `pkg install` (against the staged tree).

## Registered Commands

A manifest `[commands]` hierarchy maps nested TOML components to the words of a CLI path (with quoted space-separated keys also supported), backed by a package-owned `program def`, or describes a group when `program` is absent. `[aliases]` maps alternate paths to canonical commands or groups; the manifest layer flattens command tables and expands group aliases to their descendants for activation, dispatch, and qualified config routing. Activation rejects command conflicts unless the later installation passes `--shadow`. Each invocation derives its effective command registry from the selected package manifests — so project `[packages]` pins affect dispatch, help, and completion — and CLI dispatch resolves the longest registered path and runs it through the same execution host as `agm exec` after verifying manifest and module ownership, binding the referenced program's own CLI arguments and qualified config table the same way ([cli.md](cli.md)). A registered command's help is the referenced program's own command help, spelled for the path the reader invokes and listing `--dry-run` beside the program's options; the registration's `description` introduces the program's `@doc` and any additional `help`, from whichever side declared them. Explicit and implicit groups generate descendant listings with optional authored guidance.

A command is equally declared at its `program def`, by `@command` and the `@description`/`@help` prose beside it ([scope.md](agl/frontend/scope.md)). Those registrations reach the rest of AGM as ordinary manifest commands: wherever a *source* tree is loaded — `pkg check`, `pkg create`, a directory `install`, an editable activation — a scan of the module tree merges them into the manifest dataclass before any other consumer sees it. The scan parses each module and reads its AST, with no scope resolution, typecheck, or module graph. Nothing downstream distinguishes the two origins, and an already-merged table merges back into itself unchanged, so a store tree or an archive of one reinstalls identically; two *differing* declarations of one path, from two programs or from a program and the manifest, are an error. Groups and aliases stay manifest-only and may name a command a program registers — so a source tree's manifest is only consistent together with its merge, and loads with its cross-entry command invariants deferred until then.

Whether a failed scan is fatal is the caller's choice, made once at activation: an operation that *persists* a decision about commands — any install — must see the true command set and fails, while a *read* of activation state (dispatch, help, completion, `pkg list`) falls back to the commands the manifest declares alone, and to none at all when those need the missing registrations to make sense. A package being edited therefore costs its own commands rather than every package's.

## Façade

The `agm.packages` public façade resolves its exports lazily. Package model leaves stay available to AgL module loading without pulling in discipline validation, which depends on AgL scope resolution. Source-command discovery depends on the AgL parser, so activation, archiving, and installation import it inside the functions that resolve a source tree; reading package metadata still loads no part of the compiler.

## Code Entry Points

- `src/agm/packages/manifest.py` — manifest schema, canonical alias expansion, and the distribution-manifest view.
- `src/agm/packages/model.py`, `development.py` — package identity, version selection, `std` compatibility bounds, development-package discovery (the containing checkout of a path, and its path-dependency closure).
- `src/agm/packages/dependencies.py` — dependency-closure resolution.
- `src/agm/packages/discipline.py` — manifest, tree, and graph validation for directory and archive packages.
- `src/agm/packages/source_commands.py` — the parse-only scan for `@command` registrations and their merge into a source tree's manifest.
- `src/agm/packages/__init__.py` — the lazy façade; `layout.py` — pure layout constants (store paths and the `src` module-tree name) usable from the config and AgL layers.
- `src/agm/agl/modules/roots.py`, `loader.py` — package-root mounting and import visibility.
- `src/agm/commands/pkg/` — the package CLI; `src/agm/cli_dispatch.py`, `src/agm/commands/exec_program.py` — registered-command lookup and execution.
- `tests/test_packages_*.py`, `tests/agl/packages/` — package-domain and package-program coverage.
