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
tree. `agm pkg uninstall` removes an active selection and its registered commands.

The activation index selects global package versions and caches registered commands. Project
`[packages]` pins override those selections for that invocation, and their selected manifests
derive that invocation's effective command registry. Package-root assembly mounts
the selected packages alongside ordinary module roots, while the module loader enforces that a
package module imports only itself, its declared dependencies, and the selected standard
library. A source file inside a development package similarly gives its package and local path
dependency closure precedence for that invocation.

Installed package contents have a SHA-256 `RECORD`; archive creation and installation verify
that content before publication. The shipped `std` package is a managed store package whose
version must exactly match the running AGM version. It is refreshed by `just install`, not by
the ordinary package-install paths, and cannot be uninstalled.

## Registered Commands

A manifest `[commands]` table maps a single- or multi-word CLI path to a package-owned
`program def`. Activation rejects command conflicts unless the later installation uses
`--shadow`. Each invocation derives its effective registry from its selected package manifests,
preserving global registration priority, so project pins affect dispatch, help, and completion.
When a built-in root command does not match, CLI dispatch resolves the longest registered path
and runs its program through the same execution host as `agm exec`. Dispatch verifies the
selected manifest and module ownership before executing it.

## Code Entry Points

- `src/agm/packages/manifest.py` — manifest schema and distribution-manifest view.
- `src/agm/packages/discipline.py` — module-tree, command, and resource validation.
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
