# Packages

The package domain defines the portable package boundary independently of installation and
CLI dispatch. A package directory has a `package.toml` manifest and a module tree named after
the package; its `PackageInfo` is the source-agnostic input module-root assembly mounts for
development directories now and installed packages later. The versioned store layout helpers
place extracted package trees under `<AGM home>/packages/<name>/<version>/`, rejecting path
components that could leave the AGM home and resolving every install/removal/info path before it can
follow an ancestor link outside the canonical store. Its activation index selects one store version per
package (or records an editable root) and caches manifest-declared command registrations. Command
priority is instead durable package-local provenance: each immutable tree has a sibling store sidecar
that records its activation order and `--shadow` intent, outside the tree and therefore outside its
immutable `RECORD` payload. Rebuild reads those sidecars before reconciling command owners, so an
index loss retains temporal shadow precedence rather than inventing an order from manifests. Missing
legacy provenance is tolerated only when no command collision needs it; malformed or contradictory
provenance, or a collision with missing provenance, stops rebuilding. Editable selections remain
index-only and cannot be recovered from the store. It validates every selected package's direct minimum-version requirements after development
roots exclude same-name active or pinned selections before store resolution. Non-editable selections
must canonically remain within the store root, including when a store path traverses a link. Project
package pins override global versions before
the selected packages are passed through the existing root seam.
`RECORD` helpers write and verify the complete relative file set with SHA-256 digests while refusing
linked package trees, providing the integrity and removal manifest for installed trees. Archive helpers
first derive an in-memory distribution manifest that removes local dependency `path` sources, leaving
the development tree unchanged. They use the same canonical record format for deterministic portable ZIP
contents and stream-verify bounded archive layout, canonical ZIP entry metadata, manifest, and file
digests before a later install path extracts anything. Writers apply the same content limits before
publication and fail closed when any source directory cannot be traversed. Archive installation stages
under the AGM home but outside the scanned package-store root, so temporary trees cannot be mistaken
for installed package versions. Directory installation resolves direct
minimum-version requirements store-first, then through declared local
path sources, copies and records immutable trees, and validates then atomically rewrites the complete
activation selection; dry-run validates the same selection from transient source manifests without
creating a store tree; editable installs instead record a live root. Failed activation leaves copied
trees inactive, while removal validates the remaining selection before it clears activation and
deletes an immutable tree. URL sources stream
through a hash-verifying `requests` fetch seam, whose verified archive handoff installs within the
same activation transaction. Archive extraction verifies canonical ZIP metadata, the normalized
manifest, and every `RECORD` digest through one opened archive stream before writing a private store
sibling; validation and record verification complete before the sibling is atomically published and
activated. Existing archive identities, including dry-run plans, must also match the installed content
hash. Dry-run archive creation and installation report their planned operations without writing; dry-run
archive installation also validates archived module and command discipline directly from the verified
stream. The archive
writer rejects output whose canonical destination belongs to the source tree or aliases one of its
files. Creation retains an opened source-root capability, rejects ordinary source links, renders
into a private temporary sibling, and atomically replaces the requested destination through an opened
parent-directory capability. It compares that parent capability against the source before and after
publication; a detected containment race restores any replaced destination (or removes new output) and
raises an archive error. This protects a quiescent source tree and reports verifiable races, not an
adversarially mutating namespace: POSIX cannot atomically prove that an already-open output parent remains
outside a source tree that another same-user writer can rename, nor prevent a writer from creating a target
after any snapshot. Concurrent source or namespace mutation is therefore unsupported. An existing
destination must be hard-linkable within that parent for detected-race rollback; platforms without the
required directory-relative operations fail before publication rather than using a pathname fallback.
`agm pkg check` exposes the validation boundary without modifying or installing a package: it resolves
requirements store-first, then through local paths, while URL sources remain deferred. `agm pkg create`
checks the portable distribution view instead, so stripped local paths cannot make an archive's requirements
unsatisfiable. `agm exec` discovers the package
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
package names and command registrations cannot claim AGM's command namespace. Activation
merges valid manifest commands in `packages/activation.py`; conflicts with every active manifest,
including an owner displaced by an earlier shadow, refuse an install unless its `--shadow` request
replaces them. Reconciliation restores a remaining active owner when a winner is removed and records
only current owners. Install and list render the resulting shadow relationships explicitly. Command
dispatch remains a CLI concern.

## Code Entry Points

- `src/agm/packages/manifest.py` — manifest schema, SemVer parsing, and the non-mutating distribution-manifest view.
- `src/agm/packages/model.py` — package-root identity and canonical module ownership.
- `src/agm/packages/development.py` — containing development-package and path-dependency discovery.
- `src/agm/packages/store.py` — AGM-home-relative versioned store and provenance-sidecar paths.
- `src/agm/packages/activation.py` — activation index, durable command provenance, project pins, requirement validation, and store-root selection.
- `src/agm/packages/record.py` — deterministic SHA-256 `RECORD` writing, verification, and content identity.
- `src/agm/packages/dependencies.py` — non-mutating store/path/URL dependency satisfiability checks.
- `src/agm/packages/archive.py` — deterministic `.agmpkg` writing, bounded metadata reads, streaming integrity and dry-run discipline verification, and safe private-tree extraction.
- `src/agm/packages/install.py` — directory and archive installation, MVS dependency activation, URL archive handoff, and verified removal.
- `src/agm/packages/fetch.py` — explicit-timeout streaming URL download and SHA-256 handoff seam.
- `src/agm/agl/modules/roots.py` and `loader.py` — root mounting and ownership-based import visibility.
- `src/agm/packages/discipline.py` — directory and command/program validation.
- `src/agm/commands/pkg/` — CLI-facing validation, archive creation, installation, inspection, and removal commands.
- `tests/test_packages_manifest.py`, `tests/test_packages_discipline.py`, and
  `tests/agl/packages/` — focused validation tests and reusable package fixtures.
