# Packages

The package domain defines portable, versioned AgL module collections. A package has a
`package.toml` manifest and a module tree named after the package, so its modules import
under a stable package-qualified path. Package and dependency names must be AgL identifier
segments and cannot be reserved AgL keywords. Packages may declare version-floor dependencies,
literal resources, and CLI commands backed by parameterless `program def` entries with explicit
`unit` results. Ordinary dependencies are unbounded minimums; `std` is a minimum within the
compatible AGM release line (the same minor before 1.0, the same major thereafter).

## Package Lifecycle

`agm pkg check` validates a development package without modifying it. Manifest and module-tree
checks that read no module source run before dependency resolution; dependency-aware validation
then loads and name-resolves the package module graph with runtime package visibility, rejecting
unresolved modules, imports outside the declared dependency closure, non-converging unrestricted
scoped re-export cycles, command program references that name no valid entry, and literal resource
targets that are absent. Every module is parsed once, by that one graph load, and resource call
sites come from the scope pass's own built-in classification rather than a separate name-resolution
rule.
Dependency checking retains one selected package per name across the closure, mirroring
installation's path-source and minimum-version selection for ordinary diamond dependencies, while
validating `std` directly against the running AGM release line.
`agm pkg create` resolves the dependency closure, validates both the source tree and
selected archive contents with those dependency modules, and produces a deterministic `.agmpkg`
archive. `agm pkg install` validates dependencies, records immutable installations in the AGM-home
package store, and activates one version of each package; editable installations instead mount their
live source tree. The store follows the single runtime AGM home: `AGM_HOME`, otherwise a populated
installation-prefix `.agm`, otherwise `~/.agm`. `agm pkg uninstall` removes an active selection
and its registered commands.

The activation index selects global package versions and caches registered commands. Package
installations and removals serialize their store and activation-index changes with a store lock;
only lock acquisition failures are reported as lock errors. Immutable removal first renames the
verified active tree to a hidden tombstone, restores it if activation publication fails, and keeps
interrupted cleanup retryable by a later uninstall. Immutable package identity uses the package
name and complete canonical version, including build metadata; semantic precedence still governs
minimum requirements and highest-version selection. Project `[packages]` pins override global
selections with an exact identity for that invocation, and their selected
manifests derive that invocation's effective command registry. Package-root assembly mounts
only each selected package's declared module tree alongside ordinary loose module roots, while
the module loader enforces that a package module imports only itself, its declared dependencies,
and the selected standard library. A source file inside a development package similarly gives
its package and local path
dependency closure precedence for that invocation and retains its package-qualified module path
for execution configuration. Immutable store entries remain activation-selected during execution
and are never rediscovered as development source, including when the store root is relocated by a
symlink. Development discovery validates each path source against its named dependency and minimum
version, and rejects distinct local roots with the same package identity, so mounted imports remain
unambiguous.

Install plans finalize command-shadow diagnostics from the validated activation snapshot while the
store lock is held and before activation publication; the CLI only renders this immutable result.
Any diagnostic or activation failure, command registration included, aborts publication and rolls
back newly created package trees. Dry-run installs
retain the same transient plan while leaving the persisted index unchanged.

A package source directory is stored and shipped as its *distribution*: the normalized manifest plus
the source files that survive gitignore rules and the dotfile, VCS, cache, and archive exclusions.
One module owns that selection, so archive creation, immutable directory staging, and the content
hash that identifies an installed version all agree — the same source installs identically from a
directory and from an archive of it. Dependents are validated against what a package actually
publishes: the store tree for an immutable install, the live source for an editable one.

Installed package contents have a SHA-256 `RECORD`, which serves as the manifest of the files an
install created — uninstall reads it to remove exactly those. Removal additionally clears cache and
VCS residue the store can acquire after publication; anything else unrecorded still fails removal
loudly. `RECORD` is unsigned and lives inside the tree it describes, so hashing is spent only where
it is checked against an anchor outside that
tree: reading an archive (entry digests against the archive's own `RECORD`, feeding the declared
dependency hash) and installing over an existing tree (the destination against the hash of the
source being installed). Nothing re-hashes a package to read, activate, or execute it — publication
is an atomic rename, so a partially written tree is never reachable through a store path.
Archive readers enforce bounded metadata, entry sizes, total expansion, and path depth while binding
preflight, ZIP parsing, verification, and extraction to one opened file. A URL dependency fetch is
bounded by transfer inactivity and the archive download size limit rather than by total elapsed
time, so a large healthy download completes while a stalled connection still fails fast. Immutable directory installs
stage the source's distribution beside the final store path, revalidate the staged manifest and package
discipline, then publish with an atomic rename; dry runs validate the filtered distribution and check
directory sources for `RECORD` eligibility without hashing or publishing it. Archive installation selects the canonical destination from verified metadata
without reopening the archive, then extracts to a sibling staging tree. It resolves dependencies and runs
dependency-aware discipline validation against that staging tree before atomic publication, including when
the store root is relocated across filesystems; dry runs repeat in-archive discipline validation with the
resolved dependency closure. The shipped
`std` package is a managed store package whose
version must exactly match the running AGM version. Package `std` requirements accept that version
only when it meets their floor and compatible-line upper bound. Refreshing it after an AGM release-line
upgrade deactivates incompatible packages and their dependents while retaining their installed trees. A shared locator finds its source at the
repository root during development and inside the installed `agm` package in a wheel. It is
refreshed by `just install`, not by the ordinary package-install paths, and cannot be uninstalled.
Its package-domain refresh holds the store lock while staging and validating a complete replacement
beside the active version; rename publication replaces the whole tree rather than exposing an
in-place copy. A wheel can use its bundled copy directly when the managed store has not been
populated.

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
- `src/agm/packages/discipline.py` — source-free manifest and module-tree validation, plus the
  single graph load and scope pass whose resolution decides command program references and which
  call sites are resource built-ins, for both directory and archive packages.
- `src/agm/packages/model.py` and `development.py` — package identity, ordinary MVS version
  selection, `std` compatibility bounds, and development-package discovery.
- `src/agm/packages/layout.py` — pure store-directory and activation-index filename constants,
  taking an already-resolved AGM home so `config.general` can depend on it without a cycle back
  into the package domain.
- `src/agm/packages/store.py`, `record.py`, and `archive.py` — store layout, the shared
  store-scanning and MVS-candidate-selection helpers dependency resolution and installation both
  use, integrity records, and portable archives.
- `src/agm/packages/distribution.py` — the one distribution view of a source tree: file selection,
  manifest normalization, its record entries, and the cache/VCS classification uninstall reuses.
- `src/agm/packages/activation.py`, `dependencies.py`, `fetch.py`, and `install.py` — active
  selections, dependency resolution, downloads, and lifecycle operations.
- `src/agm/packages/stdlib.py` — resolves the active managed standard-library package root
  (activation lookup, canonical identity, store paths); `config.module_roots`
  handles only the `AGM_STDLIB` override and delegates here.
- `src/agm/agl/modules/roots.py` and `loader.py` — package-root mounting and import visibility.
- `src/agm/commands/pkg/` — package CLI commands.
- `src/agm/cli_dispatch.py` and `src/agm/commands/exec_program.py` — registered-command lookup
  and shared program execution.
- `tests/test_packages_*.py` and `tests/agl/packages/` — package-domain and package-program
  coverage.
