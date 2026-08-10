# Packages

The package domain defines portable, versioned AgL module collections. A package has a
`package.toml` manifest and a module tree named after the package, so its modules import
under a stable package-qualified path. Packages may declare minimum-version dependencies,
literal resources, and CLI commands backed by `program def` entries.

## Package Lifecycle

`agm pkg check` validates a development package without modifying it. `agm pkg create`
produces a deterministic `.agmpkg` archive from that validated package. `agm pkg install`
validates dependencies, records immutable installations in the AGM-home package store, and
activates one version of each package; editable installations instead mount their live source
tree. The store follows the single runtime AGM home: `AGM_HOME`, otherwise a populated
installation-prefix `.agm`, otherwise `~/.agm`. `agm pkg uninstall` removes an active selection
and its registered commands.

The activation index selects global package versions and caches registered commands. Package
installations and removals serialize their store and activation-index changes with a store lock;
only lock acquisition failures are reported as lock errors. Immutable removal first renames the
verified active tree to a hidden tombstone, restores it if activation publication fails, and keeps
interrupted cleanup retryable by a later uninstall. Project `[packages]` pins override global
selections for that invocation, and their selected
manifests derive that invocation's effective command registry. Package-root assembly mounts
only each selected package's declared module tree alongside ordinary loose module roots, while
the module loader enforces that a package module imports only itself, its declared dependencies,
and the selected standard library. A source file inside a development package similarly gives
its package and local path
dependency closure precedence for that invocation and retains its package-qualified module path
for execution configuration. Development discovery rejects distinct local
roots with the same package identity, so mounted imports remain unambiguous.

Installed package contents have a SHA-256 `RECORD`. Archive readers bind metadata-limit
preflight, ZIP parsing, verification, and extraction to one opened file. Immutable directory installs
stage beside the final store path, revalidate the copied manifest, package discipline, and `RECORD`,
then publish with an atomic rename; archive installation follows the same verify-before-publication
boundary. Active immutable selections verify `RECORD` again when resolved for activation or execution.
Editable selections remain live and are exempt. The shipped
`std` package is a managed store package whose
version must exactly match the running AGM version. It is refreshed by `just install`, not by
the ordinary package-install paths, and cannot be uninstalled.

The `agm.packages` public façade resolves domain exports lazily. This keeps package model
leaves available to AgL module loading without pulling in package discipline validation, which
depends on AgL scope resolution.

## Registered Commands

A manifest `[commands]` table maps a single- or multi-word CLI path to a package-owned
`program def`. Activation rejects command conflicts unless the later installation uses
`--shadow`; registry reconciliation applies the same provenance check to live editable manifests.
Each invocation derives its effective registry from its selected package manifests and the
persisted priority of each selected immutable version, while editable and legacy selections retain
their activation-index priority. Project pins therefore affect dispatch, help, and completion. New
registrations advance beyond provenance retained by inactive immutable package versions, keeping
later activation-index rebuilds unambiguous.
When a built-in root command does not match, CLI dispatch resolves the longest registered path
and runs its program through the same execution host as `agm exec`. Dispatch verifies the
selected manifest and module ownership before executing it.

## Code Entry Points

- `src/agm/packages/manifest.py` — manifest schema and distribution-manifest view.
- `src/agm/packages/__init__.py` — lazy public package-domain façade.
- `src/agm/packages/discipline.py` — module-tree, command, and lexical resource-alias validation, including bounded re-export propagation.
- `src/agm/packages/model.py` and `development.py` — package identity, ownership, and
  development-package discovery.
- `src/agm/packages/store.py`, `record.py`, and `archive.py` — store layout, integrity records,
  and portable archives.
- `src/agm/packages/activation.py`, `dependencies.py`, `fetch.py`, and `install.py` — active
  selections, dependency resolution, downloads, and lifecycle operations.
- `src/agm/agl/modules/roots.py` and `loader.py` — package-root mounting and import visibility.
- `src/agm/commands/pkg/` — package CLI commands.
- `src/agm/cli_dispatch.py` and `src/agm/commands/exec_program.py` — registered-command lookup
  and shared program execution.
- `tests/test_packages_*.py` and `tests/agl/packages/` — package-domain and package-program
  coverage.
