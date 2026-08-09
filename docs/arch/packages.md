# Packages

The package domain defines the portable package boundary independently of module-root
assembly, installation, and CLI dispatch. A package directory has a `package.toml` manifest
and a module tree named after the package; the domain validates that shape before later
layers mount or install it.

## Manifest and Identity

`packages/manifest.py` loads manifests through the shared TOML primitive and validates
package metadata, semantic versions, dependency source declarations, and command metadata.
Package versions are complete SemVer; dependency minimum versions may omit trailing minor and
patch components and are normalized through the `semver` runtime library, preserving SemVer
precedence. The model
in `packages/model.py` pairs a manifest with a canonical package root and identifies which
package owns a canonical path inside a module tree.

## Discipline

`packages/discipline.py` validates package naming, module-tree naming, manifest command
paths, and that each registered command names an actual `program def` in the package. It
uses the existing AgL module and identifier rules plus the built-in command catalog, so
package names and command registrations cannot claim AGM's command namespace.

## Code Entry Points

- `src/agm/packages/manifest.py` — manifest schema and SemVer parsing.
- `src/agm/packages/model.py` — package-root identity and canonical module ownership.
- `src/agm/packages/discipline.py` — directory and command/program validation.
- `tests/test_packages_manifest.py`, `tests/test_packages_discipline.py`, and
  `tests/agl/packages/` — focused validation tests and reusable package fixtures.
