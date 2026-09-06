# Package Store and Installation

Installed packages live in a store under the AGM home ([config.md](config.md)). An *immutable* installation is a content-hashed tree copied into the store; an *editable* installation mounts a live source tree. An activation index selects one version of each package globally and caches the registered commands; project `[packages]` pins override that selection per invocation.

## Distribution

A package source directory is stored and shipped as its *distribution*: the normalized manifest plus the files that survive gitignore rules and the dotfile, VCS, cache, and archive exclusions. One module owns that selection, so archive creation, store staging, and the content hash that identifies an installed version all agree — the same source installs identically from a directory and from an archive of it.

## Install and Uninstall

Installation resolves the dependency closure, stages the distribution beside its final store path, runs dependency-aware discipline validation against the staging tree, then publishes with an atomic rename, so a partially written tree is never reachable. Command-shadow diagnostics are finalized while the store lock is held; any diagnostic or activation failure rolls the new tree back. Dry runs perform the same validation without persisting. Store and index changes are serialized by a store lock.

Every installed tree carries a SHA-256 `RECORD` listing the files the install created; uninstall removes exactly those (plus cache and VCS residue) and fails loudly on anything else. Removal renames the tree to a hidden tombstone first, so an interrupted cleanup is retryable. `RECORD` is unsigned and lives inside the tree it describes, so it is only ever checked against an anchor outside that tree: an archive's own record on read, or the source hash when installing over an existing tree. Nothing re-hashes a package to read, activate, or execute it.

Archives (`.agmpkg`) are deterministic ZIP files. Readers bound metadata, entry sizes, total expansion, and path depth, and bind preflight, parsing, verification, and extraction to one opened file. URL dependency fetches are bounded by transfer inactivity and download size rather than total elapsed time.

## The Managed `std` Package

The standard library ships as a managed store package whose version must exactly match the running AGM. `just install` refreshes it (holding the store lock, staging a full replacement, publishing by rename); the ordinary install paths never touch it and it cannot be uninstalled. A refresh after a release-line upgrade deactivates incompatible packages and their dependents while retaining their trees. A wheel falls back to its bundled copy when the store has not been populated. A development `std` checkout whose module tree holds the anchored path outranks the store selection — so editing the library never resolves against an installed copy, and a version difference is not a mismatch; an unrelated package named `std` beside the anchor is inert. Whichever tree is selected is the one standard-library root — its `src/` tree mounts as `std` — and the owning package of an entry file inside it.

## Code Entry Points

- `src/agm/packages/store.py`, `record.py`, `archive.py` — store layout and scanning, integrity records, portable archives.
- `src/agm/packages/distribution.py` — the one distribution view of a source tree.
- `src/agm/packages/activation.py`, `install.py`, `fetch.py` — active selections, lifecycle operations, downloads.
- `src/agm/packages/stdlib.py` — anchor-aware `std` root selection; `src/agm/stdlib_locator.py` — the shipped fallback tree.
- `src/agm/commands/pkg/` — `install`, `uninstall`, `list`, `info`, `create`, `check`.
- `tests/test_packages_store.py`, `test_packages_install.py`, `test_packages_archive.py`, `test_packages_record.py`, `test_packages_distribution.py`.
