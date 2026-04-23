# AGM

Agent Management Framework is a CLI for setting up agent-oriented project workspaces, managing
repo and dependency worktrees, opening tmux sessions, running setup scripts, and executing
commands with sandbox settings.

## Requirements

- `git`
- `bash`
- `tmux` for `agm open` and `agm tmux ...`
- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- [`just`](https://github.com/casey/just)
- [`srt`](https://github.com/anthropic-experimental/sandbox-runtime) for sandboxed `agm run`
- `systemd-run` for memory limits on `agm run` when enabled

## Install

Set up the development environment:

```bash
just setup
```

Install the CLI into an isolated `uv tool` environment and copy AGM config files into
`$HOME/.agm/`:

```bash
just install
```

Pass arguments through to the config installer when needed:

```bash
just install --force
just install /usr/local
just install /usr/local --force
```

## Project layouts

`agm init` supports two layouts.

Workspace layout:

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
└── <main checkout>
```

Without `--embedded` or `--workspace`, AGM chooses:

- embedded when the target project directory already exists and is a git repo
- workspace otherwise

`agm init` also creates `config/env.sh` and an executable `config/setup.sh` if they do not
already exist.

## Usage

```bash
agm <command> [options] [args]
```

Use `agm help` for the command list and `agm help <command>` for detailed help. Global options:

- `--dry-run`
- `--install-completion`
- `--show-completion`

## Commands

### `agm open`

Open a tmux session for the main checkout or a branch worktree, creating or checking out the
branch when needed.

```bash
agm open repo
agm open main
agm open --num-panes 4 feat/login
agm open --parent develop feat/search
agm open --detach feat/search
```

### `agm close`

Remove a branch worktree and close its tmux session.

```bash
agm close feat/search
```

`repo` and the branch currently checked out in the main checkout resolve to the main checkout and
cannot be closed.

### `agm init`

Initialize a project directory, optionally cloning a repo.

```bash
agm init myproject
agm init https://github.com/org/repo.git
agm init myproject https://github.com/org/repo.git
agm init --workspace -b develop myproject https://github.com/org/repo.git
agm init --embedded myproject
```

When only `REPO_URL` is provided, AGM derives the project name from the repository URL.

### `agm fetch`

Fetch the main repo and all checked-out dependencies, then create local tracking branches for
remote branches that are not yet merged into `origin/main`.

```bash
agm fetch
```

### `agm loop`

Run an iterative prompt loop using a configured runner, with optional selector-based task
selection.

```bash
agm loop implement_feature
agm loop run review --runner "claude -p"
agm loop step fix_tests --log-file loop.log
agm loop next review --selector "codex exec"
```

Loop configuration is loaded from merged `config.toml` files via `[loop]` and optional
`[loop.<command>]` overrides.

### `agm run`

Run a command directly or inside an Anthropic Sandbox Runtime container.

```bash
agm run pytest -q
agm run --file .sandbox/ci.json make test
agm run --no-patch python script.py
agm run --no-sandbox --memory 8G make lint
```

`agm run` loads config from:

1. `<install-prefix>/.agm/config.toml` when present, otherwise `$HOME/.agm/config.toml`
2. `<project-config-dir>/config.toml`
3. `./.agm/config.toml`

Sandbox settings are resolved from the global sandbox directory, the project sandbox config
directory, and `./.sandbox/`, with later files overriding earlier ones.

### `agm config copy`

Copy known project config files from the shared project config directory into an existing target
directory.

```bash
agm config copy target-dir
agm config cp target-dir
```

Known files currently include `.setup.sh`, `.env`, `.env.local`, `.config`, `.agents`,
`.opencode`, `.codex`, `.claude`, `.pi`, and `.mcp.json`.

### `agm worktree`

Low-level worktree operations for the main project repo.

```bash
agm worktree new feat/search
agm wt new --dir /tmp/worktrees feat/search
agm worktree setup
agm worktree remove --force old-branch
agm wt rm old-branch
```

`agm worktree setup` runs executable setup scripts, in order, from:

1. `config/setup.sh`
2. `<checkout>/.config/setup.sh`
3. `<checkout>/.setup.sh`

### `agm dep`

Manage dependency checkouts under the project dependency directory.

```bash
agm dep new https://github.com/org/lib.git
agm dep new --branch develop https://github.com/org/lib.git
agm dep switch mylib feat/update
agm dep switch --branch mylib feat/new-work
agm dep rm mylib/feat/update
agm dep rm --all mylib
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

- `agm wt` → `agm worktree`
- `agm config cp` → `agm config copy`
- `agm wt rm` → `agm worktree remove`

## Help

```bash
agm help
agm help run
agm help worktree setup
agm open --help
```
