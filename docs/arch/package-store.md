# Package Store and Installation

Installed packages live in a store under the AGM home ([config.md](config.md)). An *immutable* installation is a content-hashed tree copied into the store; an *editable* installation mounts a live source tree. An activation index selects one version of each package globally and records the registered commands, though every invocation rebuilds the registry it uses from the selected manifests; dependency installation and validation share the same store-or-editable selection rule. Project `[packages]` pins override that selection per invocation.

## Distribution

A package source directory is stored and shipped as its *distribution*: the normalized manifest plus the files that survive gitignore rules and the dotfile, VCS, cache, and archive exclusions. Symlinked regular files are copied as regular files, keeping stored trees and archives portable while allowing source trees to use links such as documentation aliases. One module owns that selection, so archive creation, store staging, and the content hash that identifies an installed version all agree — the same source installs identically from a directory and from an archive of it. The normalized manifest is rendered from the manifest dataclass, so the commands a directory's own programs register ([packages.md](packages.md)) are merged in before anything is staged, hashed, or archived: an immutable installation carries one complete command table and is never rescanned. An editable installation has a manifest but no baked table to trust: it re-derives its commands from live source on every invocation that needs them, so an edited `@command` applies without reinstalling ([packages.md](packages.md) covers what happens when that source cannot be read).

## Install and Uninstall

Installation resolves the dependency closure, stages the distribution beside its final store path, checks the staging tree against the source resolution it was copied from, then publishes with an atomic rename, so a partially written tree is never reachable. Command-shadow diagnostics are finalized, and the new selection's Python requirements synced into AGM's interpreter, while the store lock is held and before activation is published; any failure or interruption rolls the new tree back. The activation index is the source of truth for the union of those requirements: when any is unsatisfied, the whole union goes to the installer so it resolves them jointly. Uninstall never removes Python distributions. Dry runs perform the same validation without persisting. Store and index changes are serialized by a store lock.

Every installed tree carries a SHA-256 `RECORD` listing the files the install created; uninstall removes exactly those (plus cache and VCS residue) and fails loudly on anything else. Removal renames the tree to a hidden tombstone first, so an interrupted cleanup is retryable. `RECORD` is unsigned and lives inside the tree it describes, so it is only ever checked against an anchor outside that tree: an archive's own record on read, or the source hash when installing over an existing tree. Nothing re-hashes a package to read, activate, or execute it.

Archives (`.agmpkg`) are deterministic ZIP files. Readers bound metadata, entry sizes, total expansion, and path depth, and bind preflight, parsing, verification, and extraction to one opened file. URL dependency fetches are bounded by transfer inactivity and download size rather than total elapsed time.

## The Managed `std` Package

The standard library ships as a managed store package whose version must exactly match the running AGM. `just install` refreshes it (holding the store lock, staging a full replacement, publishing by rename); the ordinary install paths never touch it and it cannot be uninstalled. A refresh after a release-line upgrade deactivates incompatible packages and their dependents while retaining their trees. A wheel falls back to its bundled copy when the store has not been populated. A development `std` checkout whose module tree holds the anchored path outranks the store selection — so editing the library never resolves against an installed copy, and a version difference is not a mismatch; an unrelated package named `std` beside the anchor is inert. Whichever tree is selected is the one standard-library root — its `src/` tree mounts as `std` — and the owning package of an entry file inside it.

## Code Entry Points

- `src/agm/packages/store.py`, `errors.py` — store layout and enumeration, with shared lifecycle errors; metadata reads do not import installation or compiler code. Installation loads validation and download dependencies only when needed.
- `src/agm/packages/record.py`, `archive.py` — integrity records and portable archives.
- `src/agm/packages/distribution.py` — the one distribution view of a source tree.
- `src/agm/packages/activation.py`, `install.py`, `fetch.py` — active selections, lifecycle operations, downloads.
- `src/agm/packages/python_deps.py` — the active selection's Python-requirement union and its sync through `src/agm/core/pyenv.py`.
- `src/agm/packages/stdlib.py` — anchor-aware `std` root selection; `src/agm/stdlib_locator.py` — the shipped fallback tree.
- `src/agm/commands/pkg/` — `install`, `uninstall`, `list`, `info`, `create`, `check`.
- `tests/test_packages_store.py`, `test_packages_install.py`, `test_packages_archive.py`, `test_packages_record.py`, `test_packages_distribution.py`, `test_packages_python_deps.py`.
