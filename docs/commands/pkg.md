# Packages

A package is a portable, versioned collection of AgL modules that can register `agm` commands.
See the [AgL package reference](../agl/reference/packages.md) for what a package means to AgL
source.

| Command | Description |
|---|---|
| `agm pkg init [DIR] [--name NAME] [--version VERSION]` | Scaffold a new package |
| `agm pkg check [DIR]` | Validate a package directory |
| `agm pkg create [DIR] [-o FILE]` | Validate and write a `<name>-<version>.agmpkg` archive |
| `agm pkg install SRC [--editable] [--shadow]` | Install and activate a directory or archive |
| `agm pkg uninstall NAME` | Remove an active package |
| `agm pkg list` | List installed versions and active editable packages |
| `agm pkg info NAME` | Show an active package's metadata and dependency status |
| `agm pkg sync` | Install the active packages' unsatisfied Python requirements |

`DIR` defaults to the current directory. Every command honors the global `--dry-run` flag.

## Quick start: a package with a command

```sh
agm pkg init review-tools
# creates review-tools/package.toml and review-tools/src/main.agl
```

Replace the starter program in `review-tools/src/main.agl` (any module under `review-tools/src/`
works) and register it in `review-tools/package.toml`:

```agl
@doc("Review a change")
program def review(target: text, strict: bool = false) -> unit =
  print("reviewing %{target}")
```

```toml
[commands]
pr-review = { program = "review-tools/main::review" }
```

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

`package.toml` supports `[package]` (required), `[dependencies]`, `[python]`, `[commands]`, and
`[aliases]`.
`pkg check`, `pkg create`, and `pkg install` reject a key the schema does not define, so a
misspelled field is an error rather than silently ignored.

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

A package identity is the complete version, build metadata included: `1.0.0+linux` and
`1.0.0+macos` are distinct packages.

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

### `[python]`

```toml
[python]
dependencies = ["typesafe-sdk>=0.7,<1", "tomli; python_version < '3.11'"]
```

`dependencies` lists the third-party Python distributions the package's
[companions](../agl/reference/ffi.md) import, as [PEP 508](https://peps.python.org/pep-0508/)
requirements. Companions run in AGM's own interpreter, so requirements are checked against its
environment: one whose marker does not apply is always satisfied, and a requested extra holds when
every requirement the installed distribution declares for that extra is itself satisfied.
An invalid requirement, a direct URL reference (`name @ url`), and a marker that does not evaluate
as a requirement's — one referring to `extra`, `extras`, or `dependency_groups`, or comparing
values incomparably (such as `os_name ~= "posix"`) — are manifest errors. A distribution may be
listed more than once, for example with markers selecting per-environment variants.
Requirements are stored verbatim, as declared, and are part of the package's content hash.
[`install`](#commands) and [`sync`](#commands) install unsatisfied requirements into that
environment (`just install` ends with `agm pkg sync`); [`check`](#commands) and
[`info`](#commands) only report them. The activation index is the source of truth: `sync` installs
only the requirements of active packages. A program whose run imports one of the package's
companions while a requirement is unsatisfied fails before running, with an error naming the
package and the requirement. It suggests `agm pkg sync` when the package is the active one (same
name and root), otherwise installing it with `agm pkg install [--editable] <root>`, as for a
development checkout or a project-pinned version that is not active.

### `[commands]`

```toml
[commands]
pr-review = { program = "review-tools/main::review" }

[commands.pr-review.batch] # multi-level command: agm pr-review batch
program = "review-tools/main::batch"
```

- Nested table components become command words: `[commands.devel.review]` registers `devel
  review`. The equivalent quoted flat form, `[commands."devel review"]`, is also accepted. A path
  cannot start with a built-in command or root alias (`wsp`, `wt`).
- `program` names the `program def` to run as `<module>::<program>`: `review-tools/main::review` is
  program `review` in module `review-tools/main`, file `review-tools/src/main.agl`. Must belong to
  this package, take no type parameters, return unit; its signature arguments and closure module
  parameters become the command's host surface.
- `doc` is the command's prose. A command naming a program never states it: that program's source
  `@doc` supplies it, and an installed manifest records the result. Command help shows it, then
  signature options followed by one section for each closure module with visible module
  parameters. Parameter `@doc` attributes describe their options.
- Omit `program` for a command group (must have descendant commands), whose only prose is its own
  `doc` (TOML multiline strings work). Undeclared parent groups work automatically, with generated
  help listing their descendants. A listing shows the opening paragraph of each command's `doc`,
  falling back to a generated summary when there is none.

```toml
[commands.devel]
doc = "Development workflows"

[commands.devel.review]
program = "review-tools/main::review"

[aliases]
dev = "devel"
rev = "devel review"
```

`agm devel`, `agm devel --help`, and `agm help devel` show the group's guidance and a generated
subcommand listing. Leaf commands generate usage and option help from their program signatures and
closure module parameters, so authored help is optional.

A program may register its own command instead, via
[`@command`](../agl/reference/attributes.md#command) on the `program def`
([Programs and commands](../agl/reference/packages.md#programs-and-commands)). The following
replaces the `[commands.devel.review]` entry above; the manifest needs entries only for commands
no program claims. Declaring one path in both places is an error whenever the two name different
programs, as is two programs claiming the same path.

```agl
@command("devel review")
@doc("Review changes")
program def review(target: text) -> unit =
  print "reviewing %{target}"
```

### `[aliases]`

`[aliases]` and command groups are always manifest-declared, never via program attributes. Each
key is an alternate command path (same restrictions as commands) naming a canonical command or
group in this package, never another alias; an alias cannot overwrite another command or alias,
but may target a command a program registers via `@command`. Group aliases expose all canonical
descendants: the example supports both `agm dev review` and `agm rev`. Aliases participate in
activation conflicts, project pins, help, and completion like commands.

Config tables use dots between path words: `[rev]`, `[dev.review]`, and `[devel.review]` all
address the same program route (signature arguments, engine settings, and resolving module
parameters, bare or dotted-qualified), even via `agm exec`. Different keys in these tables
combine; setting the same key
through multiple spellings in one layer is an ambiguity error, and a later config layer overrides
an earlier one. CLI flags still take precedence. Group tables do not supply inherited defaults.

## Registered commands

An active package's commands run as `agm COMMAND ...` (longest matching path wins), appear in
`agm help` and shell completion, and support `--help`. A command comes from the manifest's
`[commands]` table, a program's own `@command` attribute, or both merged.

- **Host parameters.** Signature value parameters project onto the command's CLI as for `agm exec`:
  positional-capable parameters fill trailing words in order, name-addressable ones take
  `--name VALUE` (`--name`/`--no-name` for `bool`). Every `@param` binding in the selected
  program's transitive import closure is a module parameter too: its external name supplies a
  resolving bare flag and dotted qualified flags, including `--no-...` where its type permits.
  Help groups visible module parameters by declaring module, and completion offers their resolving
  spellings. See [Program arguments](agl.md#program-arguments) and
  [Module parameters](agl.md#module-parameters). A registered command reserves its run-time
  options, `--dry-run`, and `-h`/`--help`; its parameters may reuse other spellings `agm exec`
  reserves, such as `--module-path` or `-p`, which `agm exec` still rejects.
- **Run-time options.** `agm exec`'s engine-setting flags (`--strict-json`/`--no-strict-json`,
  `--default-agent`, `--default-sandbox`, `--timeout`/`--no-timeout`, `--trace`/`--no-trace`,
  `--trace-file`/`--no-trace-file`) and `--max-call-depth` work as for
  [`agm exec`](agl.md#agm-exec), anywhere after the command path.
- **Configuration.** Omitted signature arguments and engine settings use the program route:
  the program's qualified table (e.g. `[review-tools.main.review]` for
  `review-tools/main::review`) or a registered command path (`[pr-review]`, or `[dev.review]`
  for command `dev review`). Both name the same program regardless of how it runs, so setting
  one key through both in one config layer is an error. The program's own
  [`@config`](../agl/reference/attributes.md#config) entries rank below the program route: an
  engine setting falls through to `@config` before `[exec]`, and a module parameter falls
  through to `@config` before its declaring module's module route. A resolving program-route
  leaf — its bare external name, or a dotted qualified spelling as a quoted key — still overrides
  both. CLI values win over all of them; `@opt-env` reaches a module parameter the same way, but
  not an engine setting, which only a source `std/config` write outranks. See
  [Configuration](agl.md#configuration).
- **`--dry-run`**, before or after the command path, runs the static pipeline and host-input
  validation without executing.
- **Conflicts.** Two active packages cannot own the same command path; install the later one with
  `--shadow` to make it the owner. Shadowing is recorded per store tree, so a rebuilt activation
  index preserves it.
- **Editable packages** re-scan the module tree and re-read the manifest on every dispatch, so
  command edits — including a program's `@command` — apply without reinstalling. The scan is
  best-effort: a module that cannot be parsed, or a registration that conflicts with another in
  the same package, registers nothing and leaves the manifest's own commands standing, so a
  mid-edit module does not break the rest of the CLI. `check` reports those errors. A command path
  that collides with another active package's is still a conflict, resolved as below. An installed store package or an archive instead carries
  one complete command table, merged at `install` or `create` time; nothing rescans it afterwards.
  A package activated without `--shadow` cannot acquire a conflicting command later.

## Commands

**`init`** creates `DIR` when missing and writes `package.toml` (name from the directory, version
`0.1.0`, and no dependencies) plus a starter `src/main.agl` unless one
exists; refuses a directory that already holds a manifest.

**`check`** scans the module tree for `@command`-registered programs, merges them with
`[commands]`, then validates the manifest, the `src/` module tree, command program references,
literal `resource` targets, import visibility, and dependency satisfiability without modifying
anything — the same diagnostics `install` reports. The `std` floor is checked against the running
AGM; other dependencies resolve from the store, then a `path`; a `url` counts as satisfiable and
is not fetched. Each `[python]` requirement AGM's interpreter environment does not satisfy is
reported with its status (as `info` shows it) and fails the check; `check` never installs it —
`sync` does, once the package is active.

**`create`** runs the same validation on the *distribution*, after merging the module tree's
`@command` registrations into the manifest, then writes a deterministic archive beside `DIR` (or
at `-o FILE`) whose manifest already carries that merged command table. The distribution excludes
hidden paths, VCS and cache directories, `.agmpkg` files, and anything matched by `.gitignore`
files; a resource excluded this way fails creation. The archived manifest drops `path` sources, so
every path-only dependency needs a stored version or `url` first.

**`install`** takes a directory or archive, resolves the dependency closure, validates, and stores
the distribution in `<AGM-home>/packages/<name>/<version>/` with a SHA-256 `RECORD`. From a
directory source it first merges the module tree's `@command` registrations into the manifest, so
the stored package carries one complete command table. Activation is published atomically only
after the resulting selection validates; a failed install leaves nothing active. Dependencies
resolve from the store first, then a declared `path` (installed alongside), then a `url` (fetched
and hash-verified; never in `--dry-run`). Versions are kept side by side, one per identity (build
metadata included, as above). `--editable` activates the source directory in place: no copy, no
`RECORD`, edits visible immediately, and its command table is re-derived from source on each
dispatch. Before activation is published, if AGM's interpreter environment does not satisfy some
`[python]` requirement of a package in the new selection, the union of every selected package's
requirements is installed into it with `uv pip install --python <interpreter>`
(`<interpreter> -m pip install` when `uv` is not on `PATH`). The installer resolves them jointly:
already satisfied distributions are left alone, and conflicting requirements across packages fail
the install rather than downgrading one. An installer failure or interruption fails the install and
leaves the store as it was. `--dry-run` prints the installer command instead of running it.

**`uninstall`** verifies the `RECORD`, validates the remaining selection, deactivates, and removes
the recorded files (plus cache and VCS residue). An editable package is only deactivated. Command
ownership displaced by the removed package is restored. Python distributions are never removed.

**`list`** shows every stored version as `active` or `installed`, and every active editable
package.

**`info`** shows metadata, command registrations with their prose, whether each direct
dependency is active, unsatisfied, or missing, including the inferred `std` upper bound, and, as
`requires python package SPEC: STATUS`, each `[python]` requirement's status in AGM's interpreter
environment: `installed VERSION`, `installed VERSION (unsatisfied)`, `missing`, or
`not applicable` (its marker does not hold).

**`sync`** repairs AGM's interpreter environment, for example after an AGM upgrade replaced it:
when some `[python]` requirement of an active package (store or editable) is unsatisfied, the
union of every active package's requirements is installed exactly as `install` does, and each
previously unsatisfied requirement is listed; otherwise it reports that all are satisfied and runs
no installer. An installer failure exits non-zero. `--dry-run` prints the installer command
instead of running it. `just install` ends by running the freshly installed `agm pkg sync`.

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
