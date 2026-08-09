# Packages

| Command | Description |
|---|---|
| `agm pkg check [DIR]` | Validate a package directory |
| `agm pkg create [DIR] [-o FILE]` | Validate and create a portable package archive |
| `agm pkg install SRC [--editable] [--shadow]` | Install or activate a package directory or archive |
| `agm pkg uninstall NAME` | Remove an active package |
| `agm pkg list` | List immutable installed versions and active editable packages |
| `agm pkg info NAME` | Show active package details and dependency status |

`agm pkg check` validates the `package.toml` manifest, module-tree naming discipline, and
program references used by manifest command registrations. It also checks dependencies without
modifying packages: a matching stored version is used first, then a declared local `path`; a URL
with its required hash is a deferred satisfiable source and is not fetched. `DIR` defaults to the
current directory.

`agm pkg create` runs the same manifest and package-discipline validation, then checks the
*distribution* dependencies before writing a deterministic `<name>-<version>.agmpkg` archive.
`DIR` defaults to the current directory; without `-o`, the archive is written beside `DIR`. The
archived manifest removes local `path` dependency sources while leaving the development manifest
unchanged, so a package that relies only on a local path must have a matching stored version (or a
URL source) before it can be archived.

`agm pkg install` accepts either a package directory or a `.agmpkg` archive and copies its verified
contents to `$AGM_HOME/packages/<name>/<version>/` (or
`$HOME/.agm/packages/` when `AGM_HOME` is unset), writes and verifies its SHA-256 `RECORD`,
and makes that version globally active only after the complete resulting selection validates.
Activation updates are atomically published, so a failed install leaves newly copied trees inactive.
With `--dry-run`, AGM reports the planned archive creation or archive installation and validates the
same resulting activation selection and archive module/command discipline without creating archive,
package-store, or activation-index files, or fetching URL dependencies.
Versions are retained side by side. Dependencies use
minimum-version resolution: a satisfying stored version is selected first, otherwise a declared
local `path` source is installed. URL dependencies are fetched with a required SHA-256 hash, then
the downloaded archive's normalized manifest and `RECORD` are verified before atomic extraction and
activation.

`--editable` activates the source directory directly, so its edits are visible immediately and
no immutable copy or `RECORD` is created. Manifest `[commands]` registrations are merged into the
activation index. Invoke a registered single- or multi-word command directly as `agm COMMAND ...`;
the longest matching path wins and trailing words are passed to its AgL program. A command path cannot begin with an AGM built-in command or alias (`wsp`, `wt`,
`cp`, or `copy`). A conflicting registration refuses installation unless `--shadow` is supplied;
the replacing package becomes the active command owner, and successful shadow installs identify the
displaced command owners. Command precedence is persisted in a sidecar beside each immutable store
tree, never in the `RECORD`-covered payload, so rebuilding a lost activation index preserves it.

`agm pkg uninstall` verifies the active immutable package's `RECORD`, validates the remaining
activation selection, then clears activation before removing every recorded file. Remaining
active manifests are reconciled so their command owners are restored. It refuses any store path
whose resolved ancestors leave the canonical store root. For an editable package it only clears
activation. `agm pkg list` shows every immutable installed version as `active` or `installed`,
plus active editable packages and commands only beneath their active owner, annotating commands
that shadow another active package; `agm pkg info` shows package design metadata, command registrations, and whether each direct requirement is active,
unsatisfied, or missing.
