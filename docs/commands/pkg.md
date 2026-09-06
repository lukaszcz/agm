# Packages

A package is a portable, versioned collection of AgL modules that can register its own `agm`
commands. See also the [AgL package reference](../agl/reference/packages.md) for what a package
means to AgL source.

| Command | Description |
|---|---|
| `agm pkg init [DIR] [--name NAME] [--version VERSION]` | Scaffold a new package |
| `agm pkg check [DIR]` | Validate a package directory |
| `agm pkg create [DIR] [-o FILE]` | Validate and write a `<name>-<version>.agmpkg` archive |
| `agm pkg install SRC [--editable] [--shadow]` | Install and activate a directory or archive |
| `agm pkg uninstall NAME` | Remove an active package |
| `agm pkg list` | List installed versions and active editable packages |
| `agm pkg info NAME` | Show an active package's metadata and dependency status |

`DIR` defaults to the current directory. Every command honors the global `--dry-run` flag.

## Quick start: a package with a command

```sh
agm pkg init review-tools
# creates review-tools/package.toml and review-tools/src/main.agl
```

Replace the starter program in `review-tools/src/main.agl` (any module in the
`review-tools/src/` module tree works) and register it in `review-tools/package.toml`:

```agl
program def review(target: text, strict: bool = false) -> unit =
  let _ = print("reviewing %{target}")
```

```toml
[commands]
pr-review = { program = "review-tools/main::review", description = "Review a change" }
```

Then validate, install, and run it:

```sh
agm pkg check review-tools
agm pkg install --editable review-tools   # source edits take effect immediately
agm pr-review --target src --strict
agm pr-review --help
```

## Layout

```
review-tools/
  package.toml        # manifest
  src/                # module tree: modules import as review-tools/...
    main.agl
    main.py           # optional extern companion beside its module
  prompts/            # resources: anything outside the module tree
```

The package name is one AgL identifier segment and not a reserved keyword. Package modules may
import only their own tree, packages declared in `[dependencies]`, and `std`.

## Manifest

`package.toml` has three tables: `[package]` (required), `[dependencies]`, and `[commands]`.

### `[package]`

```toml
[package]
name = "review-tools"          # required: one AgL identifier segment, not a keyword
version = "1.0.0"              # required: complete semantic version
description = "Review workflows"
license = "MIT"
authors = ["Ada <ada@example.test>"]
repository = "https://example.test/review-tools"
keywords = ["review", "workflow"]
```

Only `name` and `version` are required. A package identity is the complete version, build
metadata included: `1.0.0+linux` and `1.0.0+macos` are distinct packages.

### `[dependencies]`

```toml
[dependencies]
std = "0.1"                                          # minimum version; "0.1" means 0.1.0
helpers = { version = "1.2.0", path = "../helpers" }  # local checkout, relative to the package
remote = { version = "2.0.0", url = "https://example.test/remote.agmpkg", hash = "sha256=<64 hex>" }
```

- A value is a minimum version with no upper bound; build metadata is ignored when matching.
- `path` and `url` are optional development and download sources; they cannot be combined. A
  `url` requires a SHA-256 `hash` with a `sha256=`, `sha256:`, or `sha256-` prefix.
- `std` is an AGM compatibility contract: the running AGM must be at least the declared version
  and in the same release line (same minor for `0.x`, same major from `1.x`).
- Only direct dependencies are importable; a dependency's own dependencies are not.

### `[commands]`

```toml
[commands]
pr-review = { program = "review-tools/main::review", description = "Review a change" }
"pr-review batch" = { program = "review-tools/main::batch" }   # multi-word command path
```

- A key is a one- or multi-word command path. It cannot start with a built-in command or root
  alias (`wsp`, `wt`).
- `program` names the `program def` to run as `<module>::<program>`, e.g., `review-tools/main::review` is the program `review` in module `review-tools/main`, the file `review-tools/src/main.agl`. The program must belong to this package, take no type parameters, and return unit. Its arguments become the command's arguments.
- `description` is optional and shown in `agm help`.

## Registered commands

An active package's commands run as `agm COMMAND ...`; the longest matching path wins. They appear
in `agm help` and shell completion and support `--help`.

- **Arguments.** The program's value parameters project onto the command's CLI exactly as for
  `agm exec`: positional-capable parameters fill trailing words in order, name-addressable ones
  take `--name VALUE` (`--name`/`--no-name` for `bool`). See
  [Program arguments](exec.md#program-arguments).
- **Configuration.** Omitted arguments and engine settings come from the program's qualified table,
  e.g. `[review-tools.main.review]` for `review-tools/main::review`, or from the registered command
  path itself: `[pr-review]`, and `[dev.review]` for a command registered as `dev review`. Both
  address the same program, so either spelling applies however it is run, and setting one key
  through both in one config layer is an error. See [Configuration](exec.md#configuration).
- **`--dry-run`**, before or after the command path, runs the static pipeline and argument
  validation without executing.
- **Conflicts.** Two active packages cannot own the same command path; install the later one with
  `--shadow` to make it the owner. Shadowing is recorded per store tree, so a rebuilt activation
  index preserves it.
- **Editable packages** re-read their manifest on dispatch, so command edits apply without
  reinstalling. A package activated without `--shadow` cannot acquire a conflicting command later.

## Commands

**`init`** creates `DIR` when missing and writes `package.toml` (name from the directory, version
`0.1.0`, no dependencies, commented dependency guidance) plus a starter `src/main.agl` unless one
exists. It refuses a directory that already holds a manifest.

**`check`** validates the manifest, the `src/` module tree, `[commands]` program references, literal
`resource` targets, import visibility, and dependency satisfiability without modifying anything.
The `std` floor is checked against the running AGM; other dependencies resolve from the store,
then a `path`; a `url` counts as satisfiable and is not fetched.

**`create`** runs the same validation on the *distribution*, then writes a deterministic archive
beside `DIR` (or at `-o FILE`). The distribution excludes hidden paths, VCS and cache directories,
`.agmpkg` files, and anything matched by `.gitignore` files; a resource excluded this way fails
creation. The archived manifest drops `path` sources, so every path-only dependency needs a stored
version or `url` first.

**`install`** takes a directory or archive, resolves the dependency closure, validates, and stores
the distribution in `<AGM-home>/packages/<name>/<version>/` with a SHA-256 `RECORD`. Activation is
published atomically only after the resulting selection validates; a failed install leaves nothing
active. Dependencies resolve from the store first, then a declared `path` (installed alongside),
then a `url` (fetched and hash-verified; never in `--dry-run`). Versions are kept side by side;
`1.0.0+linux` and `1.0.0+macos` are distinct identities. `--editable` activates the source directory
in place: no copy, no `RECORD`, edits visible immediately.

**`uninstall`** verifies the `RECORD`, validates the remaining selection, deactivates, and removes
the recorded files (plus cache and VCS residue). An editable package is only deactivated. Command
ownership displaced by the removed package is restored.

**`list`** shows every stored version as `active` or `installed`, active editable packages, and
each package's commands, annotating commands that shadow another package.

**`info`** shows metadata, command registrations, and whether each direct dependency is active,
unsatisfied, or missing, including the inferred `std` upper bound.

## Version pins

Select a stored version per project instead of the globally active one:

```toml
[packages]
review-tools = "1.2.3"
platform-tools = "1.0.0+linux"
```

Pins live in any layered `config.toml` (install prefix, AGM home, project `config/config.toml`,
then `.agm/config.toml`), merge by name with later layers winning, and select an exact identity that
must already be a valid stored version. A pin affects module roots, command dispatch, help, and
completion for the invocation only; it never installs or fetches, and `list`/`info` ignore it. A
development checkout of the same name discovered from the execution root still takes precedence.

## The `std` package

The standard library is a managed package at the running AGM's version, refreshed only by
`just install`; it cannot be installed or uninstalled. A refresh across a release line deactivates
packages requiring the old line and their dependents, keeping their store trees.

## Limits

- Archives: 10,000 entries, 16 MiB ZIP metadata, 256 path components, 64 MiB per entry, 512 MiB
  total; no ZIP64.
- Downloads: 128 MiB, 30-second inactivity timeout, partial files removed on failure.
