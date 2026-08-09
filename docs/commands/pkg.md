# Packages

| Command | Description |
|---|---|
| `agm pkg check [DIR]` | Validate a package directory |
| `agm pkg install SRC [--editable] [--shadow]` | Install or activate a package directory |
| `agm pkg uninstall NAME` | Remove an active package |
| `agm pkg list` | List immutable installed versions and active editable packages |
| `agm pkg info NAME` | Show active package details and dependency status |

`agm pkg check` validates the `package.toml` manifest, module-tree naming discipline, and
program references used by manifest command registrations. `DIR` defaults to the current
directory. It does not install, create, or otherwise modify packages.

`agm pkg install` copies a package directory to `$AGM_HOME/packages/<name>/<version>/` (or
`$HOME/.agm/packages/` when `AGM_HOME` is unset), writes and verifies its SHA-256 `RECORD`,
and makes that version globally active only after the complete resulting selection validates.
Activation updates are atomically published, so a failed install leaves newly copied trees inactive.
With `--dry-run`, AGM validates the same resulting activation selection without creating package
store or activation-index files, or fetching URL dependencies.
Versions are retained side by side. Dependencies use
minimum-version resolution: a satisfying stored version is selected first, otherwise a declared
local `path` source is installed. URL dependencies are fetched with a required SHA-256 hash;
archive installation is not available yet.

`--editable` activates the source directory directly, so its edits are visible immediately and
no immutable copy or `RECORD` is created. `--shadow` is accepted now and recorded for future
package-command registration.

`agm pkg uninstall` verifies the active immutable package's `RECORD`, validates the remaining
activation selection, then clears activation before removing every recorded file. It refuses any
store path whose resolved ancestors leave the canonical store root. For an editable package it
only clears activation. `agm pkg list` shows every immutable installed version as `active` or
`installed`, plus active editable packages; `agm pkg info` shows package design metadata, command
registrations, and whether each direct requirement is active, unsatisfied, or missing.
