# Packages

| Command | Description |
|---|---|
| `agm pkg check [DIR]` | Validate a package directory |
| `agm pkg create [DIR] [-o FILE]` | Validate and create a portable package archive |
| `agm pkg install SRC [--editable] [--shadow]` | Install or activate a package directory or archive |
| `agm pkg uninstall NAME` | Remove an active package |
| `agm pkg list` | List immutable installed versions and active editable packages |
| `agm pkg info NAME` | Show active package details and dependency status |

A package directory contains `package.toml` and a module tree whose directory matches
`[package] name`. The manifest requires a complete semantic `version`; optional
`[dependencies]` entries state minimum versions and may provide a local `path` or a URL with
its SHA-256 hash. `[commands]` maps a one- or multi-word command path to a package-owned
`MODULE::PROGRAM` reference, where `PROGRAM` is a `program def` declaration. For example:

```toml
[package]
name = "review_tools"
version = "1.0.0"
description = "Review workflows"
license = "MIT"
authors = ["Ada <ada@example.test>"]
repository = "https://example.test/review-tools"
keywords = ["review", "workflow"]

[dependencies]
std = "0.1.0"
helpers = { version = "1.2.0", path = "../helpers" }

[dependencies.remote]
version = "2.0.0"
url = "https://example.test/remote.agmpkg"
hash = "sha256=0000000000000000000000000000000000000000000000000000000000000000"

[commands]
review = { program = "review_tools/main::review", description = "Review a change" }
"review batch" = { program = "review_tools/main::batch" }
```

A dependency `path` is relative to the package and cannot be combined with `url`. A URL
requires a 64-hex-digit SHA-256 hash prefixed with `sha256=`, `sha256:`, or `sha256-`. Archive
creation excludes hidden paths, VCS and cache directories, `.agmpkg` files, and files ignored by
root or nested `.gitignore` files.

`agm pkg check` validates the `package.toml` manifest, module-tree naming discipline, program
references used by manifest command registrations, and literal resource targets. It also checks
dependencies without modifying packages: a `std` requirement is checked against the running AGM
version; a matching stored version is used first for other packages, then a declared local `path`;
a URL with its required hash is a deferred satisfiable source and is not fetched.
`DIR` defaults to the current directory.

`agm pkg create` validates the selected portable archive contents, including literal resource
targets, then checks the *distribution* dependencies, including its `std` requirement against the
running AGM version, before writing a deterministic
`<name>-<version>.agmpkg` archive. A resource excluded by archive filtering causes creation to fail
rather than producing a broken package.
`DIR` defaults to the current directory; without `-o`, the archive is written beside `DIR`. The
archived manifest removes local `path` dependency sources while leaving the development manifest
unchanged, so a package that relies only on a local path must have a matching stored version (or a
URL source) before it can be archived.

`agm pkg install` accepts either a package directory or a `.agmpkg` archive and copies its verified
contents to `$AGM_HOME/packages/<name>/<version>/` (or
`$HOME/.agm/packages/` when `AGM_HOME` is unset), writes and verifies its SHA-256 `RECORD`,
and makes that version globally active only after the complete resulting selection validates.
Activation updates are atomically published, so a failed install leaves newly copied trees inactive.
With `--dry-run`, AGM reports the planned archive creation or archive installation and validates its
archive module/command discipline and any resulting selection resolvable from local sources and the
store, without creating archive, package-store, or activation-index files. It never fetches URL
dependencies, so an installation that needs one fails in dry-run mode.
Versions are retained side by side. Dependencies use
minimum-version resolution: a satisfying stored version is selected first, otherwise a declared
local `path` source is installed. URL dependencies are fetched with a required SHA-256 hash, then
the downloaded archive's normalized manifest and `RECORD` are verified before atomic extraction and
activation.

`--editable` activates the source directory directly, so its edits are visible immediately and
no immutable copy or `RECORD` is created. Manifest `[commands]` registrations are merged into the
activation index. Invoke a registered single- or multi-word command directly as `agm COMMAND ...`;
the longest matching path wins and trailing words are passed to its AgL program. Registered commands
accept the same parameter flags and qualified configuration/engine-setting tables as `agm exec`.
They appear in `agm help` and shell completion while active. An editable command re-reads its live
manifest when dispatched, so a changed registered program takes effect without reinstalling. A
command path cannot begin with an AGM built-in command or alias (`wsp`, `wt`,
`cp`, or `copy`). A conflicting registration refuses installation unless `--shadow` is supplied;
the replacing package becomes the active command owner, and successful shadow installs identify the
displaced command owners. Command precedence is persisted in a sidecar beside each immutable store
tree, never in the `RECORD`-covered payload, so rebuilding a lost activation index preserves it.

The built-in `std` package is installed and activated with AGM itself at the same version as the
running binary. Its managed store tree is refreshed only from AGM's shipped `stdlib/` directory
by `just install`; editable and archive installs are rejected, and `agm pkg uninstall std` always
refuses. A package's `std` minimum-version requirement is also its minimum AGM version, so
installation refuses a package that requires a newer AGM binary.

`agm pkg uninstall` verifies the active immutable package's `RECORD`, validates the remaining
activation selection, then clears activation before removing every recorded file. Remaining
active manifests are reconciled so their command owners are restored. It refuses any store path
whose resolved ancestors leave the canonical store root. For an editable package it only clears
activation. `agm pkg list` shows every immutable installed version as `active` or `installed`,
plus active editable packages and commands only beneath their active owner, annotating commands
that shadow another active package; `agm pkg info` shows package design metadata, command registrations, and whether each direct requirement is active,
unsatisfied, or missing; `std` is reported against the running AGM version rather than the active
package store.
