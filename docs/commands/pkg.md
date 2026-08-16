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
`[package] name`. Package and dependency names must each be one AgL identifier segment and cannot
be reserved AgL keywords. The manifest requires a complete semantic `version`; optional
`[dependencies]` entries state minimum versions and may provide a local `path` or a URL with
its SHA-256 hash. `[commands]` maps a one- or multi-word command path to a package-owned
`MODULE::PROGRAM` reference, where `PROGRAM` is a `program def` declaration with no value or type parameters and an explicit `-> unit` result. For example:

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
std = "0.1.1"
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
root or nested `.gitignore` files. Portable archives are limited to 10,000 entries, 16 MiB of ZIP
metadata, 256 path components per entry, 64 MiB expanded per entry, and 512 MiB expanded in total;
ZIP64 archives are not supported.

`agm pkg check` validates the `package.toml` manifest, module-tree naming discipline, program
references used by manifest command registrations, and literal resource targets reached through
imports, opens, or resolved dependency re-exports (rejecting scoped resource re-export cycles that
keep expanding their paths). It also checks
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

`agm pkg install` accepts either a package directory or a `.agmpkg` archive and stores its verified
distribution — the normalized manifest and the same files archive creation selects, so a directory
install omits hidden, VCS, cache, and ignored files and local `path` dependency sources just as
archive creation does — in `<AGM-home>/packages/<name>/<version>/`, where the runtime home is
`$AGM_HOME`, otherwise a populated `<install-prefix>/.agm`, otherwise `$HOME/.agm`. It writes
and verifies the package's SHA-256 `RECORD`,
and makes that version globally active only after the complete resulting selection validates.
Activation updates are atomically published, so a failed install leaves newly copied trees inactive.
With `--dry-run`, AGM reports the planned archive creation or archive installation and validates its
archive module/command discipline and any resulting selection resolvable from local sources and the
store, without creating archive, package-store, or activation-index files. It never fetches URL
dependencies, so an installation that needs one fails in dry-run mode.
Versions are retained side by side. A package identity includes the complete canonical version,
including build metadata, so versions such as `1.0.0+linux` and `1.0.0+macos` are distinct store
entries and activation selections. Dependencies use semantic-version precedence for minimum-version
resolution, where build metadata does not affect whether a version satisfies a range: a satisfying
stored version is selected first, otherwise a declared local `path` source is installed. URL
dependencies are fetched with a required SHA-256 hash, then the downloaded archive's normalized
manifest and `RECORD` are verified before atomic extraction and activation. Downloads have a 128 MiB
size limit and a 30-second inactivity timeout, including blocked connection and body reads; each
nonempty chunk resets the timeout, and partial temporary archives are removed on failure.

## Package version pins

Use `[packages]` in a layered `config.toml` to select an installed package version for the current
invocation:

```toml
[packages]
review_tools = "1.2.3"
platform_tools = "1.0.0+linux"
```

Each value must be a quoted, complete semantic version (`MAJOR.MINOR.PATCH`, with optional
prerelease and build metadata). A pin selects the exact package identity: `1.0.0+linux` does not
match `1.0.0+macos`, even though build metadata is ignored when checking a dependency's minimum
version.

Pins follow the normal general-config precedence: installation-prefix config, AGM-home config,
shared project `config/config.toml`, then the invocation directory's `.agm/config.toml`. The
`[packages]` tables merge by package name, with a later project or workspace value overriding the
same earlier pin. AGM discovers the current project from the invocation directory or `PROJ_DIR`, so
leaving that project restores the less-specific selection. A pin overlays the globally active
version only for package-aware operations; it does not install, fetch, or globally activate that
version, and does not change `agm pkg list` or `agm pkg info`.

The exact pinned version must already be an immutable, valid `RECORD`-verified entry in the selected
AGM home's package store; a global editable activation is not a substitute. Invalid pin syntax, a
missing or corrupt store entry, a manifest identity mismatch, or unsatisfied dependencies fail when
AGM resolves package roots or registered commands. AGM does not fall back to the globally active
version or fetch a dependency. For module-root selection, a discovered development package with the
same name takes precedence over the stored selection.

Pins determine the effective package module roots used by `agm exec`, `agm repl`, installed program
references, and their parameter discovery. They also rebuild the invocation's registered-command
registry from the selected manifests, so command dispatch, `agm help`, registered-command `--help`,
and shell completion all reflect the pinned versions. Built-in commands remain reserved and take
precedence over package registrations.

`--editable` activates the source directory directly, so its edits are visible immediately and
no immutable copy or `RECORD` is created. Manifest `[commands]` registrations are merged into the
activation index. Invoke a registered single- or multi-word command directly as `agm COMMAND ...`;
the longest matching path wins and trailing words are passed to its AgL program. Registered commands
accept the same parameter flags and qualified configuration/engine-setting tables as `agm exec`.
The global `--dry-run` flag can appear before or after a registered command path; it runs the static
pipeline and parameter validation without executing the program. They appear in `agm help` and
shell completion while active. An editable command re-reads its live
manifest when dispatched, so nonconflicting command edits take effect without reinstalling. Live
command additions are rechecked against the activation's recorded `--shadow` intent; an editable
package activated without `--shadow` cannot acquire a conflict. A command path cannot begin with an
AGM built-in command or root alias (`wsp` or `wt`). A conflicting registration refuses installation
unless `--shadow` is supplied;
the replacing package becomes the active command owner, and successful shadow installs identify the
displaced command owners. Command precedence is persisted in a sidecar beside each immutable store
tree, never in the `RECORD`-covered payload, so rebuilding a lost activation index preserves it.

The built-in `std` package is installed and activated with AGM itself at the same version as the
running binary. Its managed store tree is refreshed only from AGM's shipped `stdlib/` directory
by `just install`; editable and archive installs are rejected, and `agm pkg uninstall std` always
refuses. A package's `std` minimum-version requirement is also its minimum AGM version, so
installation refuses a package that requires a newer AGM binary.

`agm pkg uninstall` verifies the active immutable package's `RECORD`, validates the remaining
activation selection, then clears activation before removing every recorded file, along with any
tool-cache or VCS content the store tree acquired after installation. Remaining
active manifests are reconciled so their command owners are restored. It refuses any store path
whose resolved ancestors leave the canonical store root. For an editable package it only clears
activation. `agm pkg list` shows every immutable installed version as `active` or `installed`,
plus active editable packages and commands only beneath their active owner, annotating commands
that shadow another active package; `agm pkg info` shows package design metadata, command registrations, and whether each direct requirement is active,
unsatisfied, or missing; `std` is reported against the running AGM version rather than the active
package store.
