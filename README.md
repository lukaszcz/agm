# AGM

Agent Management Framework is a CLI for setting up agent-oriented project directories, managing
AGM workspaces, opening tmux sessions, running setup scripts, executing
commands with sandbox settings, and running AgL agent workflows — as whole programs (`agm exec`)
or in an interactive REPL (`agm repl`).

## Requirements

- `git`
- `bash`
- `tmux` for `agm open` and `agm tmux ...`
- Python 3.14
- [`uv`](https://docs.astral.sh/uv/)
- [`just`](https://github.com/casey/just)
- [`srt`](https://github.com/anthropic-experimental/sandbox-runtime) for sandboxed `agm run`
- `systemd-run` for memory limits on `agm run` when enabled

## Install

Set up the development environment:

```bash
just setup
```

Install the CLI into an isolated `uv tool` environment and copy AGM config files,
prompts, and sandbox templates into the selected AGM home (`$AGM_HOME`, or
`$HOME/.agm/` by default). It also installs and activates the lockstep immutable
`std` package at `<AGM-home>/packages/std/<version>/` and installs the AgL editor
support: the Micro syntax file into `$HOME/.config/micro/syntax/`, and — when an
`emacs` binary is available — the AgL Emacs mode (skipped with a notice
otherwise):

```bash
just install
```

Install the Emacs mode on its own with `just setup-emacs`. It gives `.agl` files
syntax highlighting, indentation, declaration navigation, on-save diagnostics
through `agm check`, and an inferior `agm repl` — see
[config/emacs/README.md](config/emacs/README.md).

Built wheels also contain a bundled standard-library fallback, so a direct wheel
installation can execute AgL before an AGM home or managed `std` store is populated.

Pass arguments through to the config installer when needed:

```bash
just install --force
just install /usr/local
just install /usr/local --force
```

An explicit prefix installs the executable link into `<prefix>/bin` and the complete
runtime tree into `<prefix>/.agm`; add `<prefix>/bin` to `PATH`. That prefix takes
precedence over `AGM_HOME` while installing. At runtime, `AGM_HOME` explicitly overrides
an installation-local tree; otherwise AGM uses the populated `.agm` beside its executable,
then falls back to `$HOME/.agm`.

## Project layouts

`agm init` supports two layouts.

Split layout:

```text
myproject/
├── repo/
├── deps/
├── notes/
├── worktrees/
└── config/
```

Embedded layout:

```text
myproject/
├── .agm/
│   ├── config/
│   ├── deps/
│   ├── notes/
│   └── worktrees/
└── <main workspace files>
```

Without `--embedded` or `--split`, AGM chooses:

- embedded when the target project directory already exists and is a git repo
- split otherwise

`agm init` also creates `config/env.sh` and an executable `config/setup.sh` if they do not
already exist. When initializing an embedded Git repository with no commits, it commits the
generated `.gitignore` as the repository's initial commit.

For split layouts without a repository URL, AGM initializes `repo/` as an empty git repository.
Use `--no-repo-git` to skip that repository, or `--no-git-init` to skip all git repositories
created by `agm init`.

## Usage

```bash
agm <command> [options] [args]
```

Use `agm help` for the command list and `agm help <command>` for detailed help. Global options:

- `--dry-run`
- `--install-completion`
- `--show-completion`

## Commands

### `agm pkg`

Validate, archive, install, inspect, and remove AgL packages. Installations use a versioned
store; editable packages mount their working directory directly. A package can register
single- or multi-word commands that run its `program def` entries directly.

```bash
agm pkg check [DIR]
agm pkg create [DIR] [-o package.agmpkg]
agm pkg install path/to/package
agm pkg install package.agmpkg
agm pkg install --editable path/to/package
agm pkg list
agm pkg info package-name
agm pkg uninstall package-name
```

`pkg check` validates package discipline and dependency satisfiability without writing. `pkg create`
checks that its distribution dependencies are portable, then writes a deterministic archive; it removes
local dependency paths from the archived manifest without changing the source manifest. Archive installs
verify the archive contents before atomically activating them. `--dry-run` reports package archive creation
or installation without writing an archive, store tree, or activation index.

### `agm open`

Shortcut for `agm workspace open`. Open a tmux session for the main workspace or a branch
workspace, creating or checking out the branch when needed. `--parent` bases only new
branches; AGM warns if the target branch already exists and errors if its workspace already
exists.

```bash
agm open repo
agm open main
agm open --num-panes 4 feat/login
agm open --parent develop feat/search
agm open --detach feat/search
```

### `agm close`

Shortcut for `agm workspace close`. Remove a branch workspace and close its tmux session.

```bash
agm close feat/search
agm close --force feat/search
agm close --keep-branch feat/search
agm close --keep-workspace feat/search
```

`repo` and the branch currently checked out in the main workspace resolve to the main workspace and
cannot be closed. Use `--keep-branch` to remove only the worktree, or `--keep-workspace` to close the
session while keeping both the workspace and branch.

### `agm init`

Initialize a project directory, optionally cloning a repo.

```bash
agm init myproject
agm init https://github.com/org/repo.git
agm init myproject https://github.com/org/repo.git
agm init --split -b develop myproject https://github.com/org/repo.git
agm init --embedded myproject
```

When only `REPO_URL` is provided, AGM derives the project name from the repository URL.

### `agm workspace`

Manage AGM workspaces. A workspace may be the main repo or a linked Git worktree, interpreted
with AGM project config, workspace config, dependency environment, setup scripts, and tmux session
lifecycle.

```bash
agm workspace open repo
agm wsp open feat/login
agm workspace close feat/login
agm workspace list
agm wsp list -v
agm workspace setup
```

`agm workspace setup` runs configured setup scripts for the current workspace, in this order:

1. `config/setup.sh`
2. `<workspace>/.config/setup.sh`
3. `<workspace>/.setup.sh`

### `agm sync fetch`

Prune stale Git worktree registrations, fetch the main repo and all checked-out dependencies, then
create local tracking branches for remote branches that are not yet merged into the default origin
branch.

```bash
agm sync fetch
```

### `agm sync pull`

Run `agm sync fetch`, then run `git merge` in every Git worktree: dependency worktrees, the main
workspace, and branch workspaces.

```bash
agm sync pull
```

### `agm exec`

Execute an AgL (Agent Language) workflow program. AgL is a statically-typed, expression-oriented
DSL for composable agent workflows: it supports typed params and outputs, user-defined functions
(`def`/`fn`), functions implemented by a co-located Python file (`extern def`), structured JSON
targets, do-loops with retry/abort policies, control flow (if/case/try), shell execution (`exec`),
and typed `Agent` values. Free `ask` lazily opens a persistent default agent
session from `std/config::default-agent`; use `agent.ask(...)` or `Session::open(...)`
to select an explicit agent or conversation. The selected value determines the invoked command.

A file workflow declares one or more zero-argument `program def` entries: `agm exec`
invokes the sole one after initialization, or selects one of several with
`-p`/`--program PATH` (for example, `review::main`). Inline `-c` source is wrapped
in a synthetic entry when needed. Programs can span multiple `.agl` files via the
module system (`import utils/math`). Every loaded entry and library module,
except `std/core` itself, receives `import std/core::*` by default. An explicit import
whose expansion includes `std/core` supplies that module's core contribution instead,
so plain `import std/core` leaves its names qualified-only. `--no-stdlib` disables the
automatic prelude throughout the loaded program.
Other imports are qualified by default; an import tail or `use` declaration makes selected names
bare. `agm exec` searches the entry file's directory, the selected standard library
(the active `<AGM-home>/packages/std/<AGM_VERSION>/` package, then the wheel-bundled or
source-checkout fallback),
the selected AGM home's global `lib` directory, and any configured
`[modules] roots` for imported modules.

```bash
agm exec workflow.agl
agm exec workflow.agl --name Alice   # --<param> per declared param
agm exec -c 'print "hello"'       # run inline program text instead of a file
agm exec --dry-run workflow.agl   # static check only — no agent calls
```

See `agm help exec` for options, exit codes, and config. The AgL language itself is
documented in the [AgL language reference](docs/agl/reference/index.md).

### `agm check`

Statically check one or more AgL files: run the full static pipeline (parse, module
loading, scope resolution, type checking, match compilation, and lowering) and report
GNU-style diagnostics, without evaluating anything or running an agent. Unlike `agm exec`,
no file needs to declare a `program def`, so plain library modules can be checked too.
Each file is checked independently, in argument order; a clean file produces no output.

```bash
agm check workflow.agl
agm check lib/*.agl              # library modules, no `program def` required
```

See `agm help check` and [docs/commands/check.md](docs/commands/check.md) for options and
exit codes. This is distinct from [`agm pkg check`](docs/commands/pkg.md), which validates
a package directory's manifest and module-tree discipline.

### `agm repl`

Start an interactive read-eval-print loop for AgL. The REPL keeps a persistent session:
each entry is parsed, type-checked, and evaluated once against an environment that
accumulates bindings, types, and declarations, so earlier results stay available and agent
calls fire exactly once. By default it fires agent calls immediately; `--confirm-agents`
asks before each one. Multiline editing, syntax highlighting, tab-completion, and history are
built in, and `:` meta-commands (`:help`, `:type`, `:bindings`, …) inspect the session.
Each loaded program receives the `std/core` standard-library prelude, as in
`agm exec`, unless an explicit import includes `std/core`; plain `import std/core`
therefore leaves its names qualified-only. Pass `--no-stdlib` to disable the
prelude for the entry and its library modules. Imported-module params can be supplied by their qualified
config tables; params declared directly at the prompt use source defaults (or are required).

```bash
agm repl                        # launch; type :help for commands, :quit to exit
agm repl --confirm-agents       # confirm calls; prompt params use source defaults
agl> let n = 21 * 2             # bindings persist across entries → "n : int = 42"
```

See `agm help repl` and [docs/commands/index.md](docs/commands/index.md) for the full reference, and the
[AgL language reference](docs/agl/reference/index.md) for the language.

### `agm review`

Run the review prompt. Review output is saved to a timestamped file by default.

```bash
agm review
agm review --scope "full codebase" implement_feature
```

### `agm revise`

Run the revision prompt against a review file.

```bash
agm revise .agent-files/review-20260101-120000-000000.md
agm revise implement_feature .agent-files/review-20260101-120000-000000.md
```

### `agm refine`

Run review/revise cycles until the revise response is `COMPLETE` or the step limit is reached.

```bash
agm refine
agm refine --max-steps 10 implement_feature
```

### `agm loop`

Run an iterative prompt loop using a configured runner, with optional selector-based task
selection.

```bash
agm loop implement_feature
agm loop run review --runner "claude -p"
agm loop step fix_tests --log-file loop.log
agm loop select review --selector "codex exec"
```

**Config** — Loop configuration is loaded from merged `config.toml` `[loop]` and `[loop.<command>]`
sections. CLI flags override config values; `RUNNER_ARGS` are appended to the final runner command.

**Subcommands**:

- `agm loop step` — single loop iteration
- `agm loop select` — run `select.md` once; requires selector mode

See `agm help loop` for selector/no-selector mode, prompt options, timeout, and logging details.

### `agm run`

Run a command directly or inside an Anthropic Sandbox Runtime container.

```bash
agm run pytest -q
agm run --file .sandbox/ci.json make test
agm run --no-patch python script.py
agm run --no-sandbox --memory 8G make lint
```

`agm run` loads config from:

1. `<install-prefix>/.agm/config.toml` when present
2. the selected AGM home's `config.toml` when distinct (`$AGM_HOME`, otherwise
   `<install-prefix>/.agm` when populated, otherwise `$HOME/.agm`)
3. `<project-config-dir>/config.toml`
4. `./.agm/config.toml`

Sandbox settings are resolved from the global sandbox directory, the project sandbox config
directory, and `./.sandbox/`, with later files merged over earlier ones.

### `agm config copy`

Copy known project config files from the shared project config directory into an existing target
directory.

```bash
agm config copy target-dir
agm config cp target-dir
```

Known files currently include `.setup.sh`, `.env`, `.env.local`, `.config`, `.agents`,
`.opencode`, `.codex`, `.claude`, `.pi`, and `.mcp.json`.

### `agm config env`

Print shell statements that refresh the current workspace environment from project and workspace
`.env`, `.env.local`, and `env.sh` files. Apply them to the current shell with:

```bash
eval "$(agm config env)"
```

### `agm config update`

Create missing project and workspace `config.toml` files and commit generated changes.

```bash
agm config update
```

### `agm dep`

Manage dependency checkouts under the project dependency directory.

```bash
agm dep list
agm dep new https://github.com/org/lib.git
agm dep new --branch develop https://github.com/org/lib.git
agm dep switch mylib feat/update
agm dep switch --branch mylib feat/new-work
agm dep rm mylib/feat/update
agm dep rm --all mylib
```

### `agm worktree`

Low-level worktree operations for the main project repo.

```bash
agm worktree new feat/search
agm wt new --dir /tmp/worktrees feat/search
agm worktree remove --force old-branch
agm wt rm old-branch
```

### `agm tmux`

Manage tmux sessions and apply AGM pane layouts directly.

```bash
agm tmux open
agm tmux open --detach --num-panes 4 my-session
agm tmux close my-session
agm tmux layout 4 --window @1
```

## Aliases

- `agm wsp` → `agm workspace`
- `agm dep remove` → `agm dep rm`
- `agm config cp` → `agm config copy`
- `agm wt` → `agm worktree`
- `agm wt rm` / `agm worktree rm` → `agm worktree remove`

## Help

```bash
agm help
agm help run
agm help worktree new
agm open --help
```
