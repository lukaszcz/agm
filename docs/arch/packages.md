# Packages

The package domain defines the portable package boundary independently of installation and
CLI dispatch. A package directory has a `package.toml` manifest and a module tree named after
the package; its `PackageInfo` is the source-agnostic input module-root assembly mounts for
development directories now and installed packages later. The versioned store layout helpers
place extracted package trees under `<AGM home>/packages/<name>/<version>/`, rejecting path
components that could leave the AGM home and resolving every install/removal/info path before it can
follow an ancestor link outside the canonical store. Its activation index selects one store version per
package (or records an editable root), is deterministically rebuildable from installed manifests,
and validates every selected package's direct minimum-version requirements after development
roots exclude same-name active or pinned selections before store resolution. Non-editable selections
must canonically remain within the store root, including when a store path traverses a link. Project
package pins override global versions before
the selected packages are passed through the existing root seam.
`RECORD` helpers write and verify the complete relative file set with SHA-256 digests while refusing
linked package trees, providing the integrity and removal manifest for installed trees. Directory
installation resolves direct minimum-version requirements store-first, then through declared local
path sources, copies and records immutable trees, and validates then atomically rewrites the complete
activation selection; dry-run validates the same selection from transient source manifests without
creating a store tree; editable installs instead record a live root. Failed activation leaves copied
trees inactive, while removal validates the remaining selection before it clears activation and
deletes an immutable tree. URL sources stream
through a hash-verifying `requests` fetch seam and await archive installation support. `agm pkg check`
exposes the validation boundary without modifying or installing a package. `agm exec` discovers the package
containing its file (or its current directory for inline source), while `agm repl` discovers
one at its current directory; each gives that explicitly path-sourced dependency closure
precedence over the selected active roots.

## Manifest and Identity

`packages/manifest.py` loads manifests through the shared TOML primitive and validates
package metadata, semantic versions, dependency source declarations, and command metadata.
Package versions are complete SemVer; dependency minimum versions may omit trailing minor and
patch components and are normalized through the `semver` runtime library, preserving SemVer
precedence. The model
in `packages/model.py` pairs a manifest with a canonical package root and identifies which
package owns a canonical path inside a module tree. The AgL loader uses that ownership map at
its dependency edge seam: package modules may import their own modules, declared dependencies,
and the host-selected standard library, while ad-hoc modules retain open visibility.

## Discipline

`packages/discipline.py` validates package naming, module-tree naming, manifest command
paths, and that each registered command names an actual `program def` in the package. It
uses the existing AgL module and identifier rules plus the built-in command catalog, so
package names and command registrations cannot claim AGM's command namespace.

## Code Entry Points

- `src/agm/packages/manifest.py` — manifest schema and SemVer parsing.
- `src/agm/packages/model.py` — package-root identity and canonical module ownership.
- `src/agm/packages/development.py` — containing development-package and path-dependency discovery.
- `src/agm/packages/store.py` — AGM-home-relative versioned store paths.
- `src/agm/packages/activation.py` — activation index, project pins, requirement validation, and store-root selection.
- `src/agm/packages/record.py` — deterministic SHA-256 `RECORD` writing and verification.
- `src/agm/packages/install.py` — directory installation, MVS dependency activation, and verified removal.
- `src/agm/packages/fetch.py` — explicit-timeout streaming URL download and SHA-256 handoff seam.
- `src/agm/agl/modules/roots.py` and `loader.py` — root mounting and ownership-based import visibility.
- `src/agm/packages/discipline.py` — directory and command/program validation.
- `src/agm/commands/pkg/` — CLI-facing validation, installation, inspection, and removal commands.
- `tests/test_packages_manifest.py`, `tests/test_packages_discipline.py`, and
  `tests/agl/packages/` — focused validation tests and reusable package fixtures.
