# agm

Agent Management Framework — a unified CLI for managing worktrees, project dependencies, configuration, sandbox execution, and tmux sessions.

## Requirements

- `bash`
- `git`
- Python ≥ 3.12
- [`just`](https://github.com/casey/just) (for installation)
- [`srt`](https://github.com/anthropic-experimental/sandbox-runtime) (required by `agm run`)

## Install

Install the `agm` CLI, all supporting scripts, and sandbox configuration to `prefix/bin` (default: `$HOME/.local/bin`):

```bash
just install
```

This also installs all files from `sandbox/` to `$HOME/.sandbox/`.

Install to a custom prefix:

```bash
just install /usr/local
```

## Usage

```
agm <command> [options] [args]
```

Run `agm help` to list all commands, or `agm help <command>` for detailed help on a specific command. Every subcommand also supports `--help`.

## Commands

### `agm open` — Open a project session

Open a tmux session for a project branch. If no branch is given, the current branch is used.

```bash
agm open                      # open session for current branch
agm open feat/login           # open session for feat/login
agm open -n 4 feat/login      # open with 4 panes
```

### `agm new` — Create a branch and open a session

Create a new branch worktree and immediately open a tmux session for it.

```bash
agm new feat/search
agm new -p develop feat/search         # fork from 'develop'
agm new -n 3 -p main feat/search      # 3 panes, fork from 'main'
```

### `agm checkout` — Check out a branch and open a session

Check out an existing branch into a worktree (creating it if needed) and open a tmux session. Alias: `agm co`.

```bash
agm co feat/login
agm checkout -n 4 feat/login
agm co -p main feat/new-thing
```

### `agm init` — Initialize a new project

Clone a repository and set up the project structure. The project name is derived from the URL if omitted.

```bash
agm init https://github.com/org/repo.git
agm init myproject https://github.com/org/repo.git
agm init -b develop myproject https://github.com/org/repo.git
```

### `agm fetch` — Fetch repo and dependencies

Fetch the latest changes for the main repository and for every dependency that has a checked-out worktree under `deps/`.

```bash
agm fetch
```

### `agm branch sync` — Sync remote tracking branches

Fetch and prune `origin`, then create local tracking branches for every remote branch not yet merged into `origin/main`. Alias: `agm br sync`.

```bash
agm br sync
agm branch sync
```

### `agm config copy` — Copy configuration files

Copy project configuration files into a target directory. Alias: `agm config cp`.

```bash
agm config cp mydir
agm config copy -d /path/to/project target
```

### `agm worktree` — Git worktree management

Low-level worktree operations. Alias: `agm wt`.

```bash
# check out a branch into a worktree
agm wt co feat/login
agm worktree checkout -d /custom/dir feat/login

# create a new branch and worktree
agm wt new feat/search
agm wt co -b feat/search          # equivalent

# remove a worktree and its local branch
agm wt rm old-branch
agm wt rm -f old-branch           # force removal
```

### `agm dep` — Manage dependencies

Manage dependency checkouts under `deps/`.

```bash
# clone a new dependency
agm dep new https://github.com/org/lib.git
agm dep new -b v2 https://github.com/org/lib.git

# switch a dependency to a different branch
agm dep switch mylib feat/update
agm dep switch -b mylib feat/new-thing    # create the branch
```

### `agm run` — Run a command in a sandbox

Run a command inside an [Anthropic Sandbox Runtime](https://github.com/anthropic-experimental/sandbox-runtime) container. Settings are loaded from `~/.sandbox/default.json` and/or `./.sandbox/default.json` (local overrides global with section-aware merging).

```bash
agm run npm test
agm run -f .sandbox/ci.json make build
agm run --no-patch python3 script.py
```

### `agm tmux` — Tmux session management

Create tmux sessions and apply pane layouts.

```bash
agm tmux new
agm tmux new -d -n 4 my-session       # detached, 4 panes
agm tmux layout 4 @1 200 50           # apply layout (internal use)
```

## Getting help

```bash
agm help               # list all commands
agm help open          # detailed help for 'open'
agm help worktree      # detailed help for 'worktree'
agm open --help        # argparse-style option summary
```
