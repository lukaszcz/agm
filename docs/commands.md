# AGM commands reference

## Global usage

```text
agm <command> [options] [args]
```

Global options:

- `--dry-run`
- `--install-completion`
- `--show-completion`

## Project session and lifecycle commands

| Command | Description |
|---|---|
| `agm open [-d\|--detach] [-n\|--num-panes PANES] [-p\|--parent PARENT] TARGET` | Open the main checkout or a branch worktree, creating or checking it out when needed |
| `agm close BRANCH` | Remove a branch worktree and close its tmux session |
| `agm init [--embedded \| --workspace] [-b\|--branch BRANCH] PROJECT_NAME` | Initialize a project directory without cloning a repo |
| `agm init [--embedded \| --workspace] [-b\|--branch BRANCH] [PROJECT_NAME] REPO_URL` | Initialize a project directory and clone a repo |
| `agm fetch` | Fetch the main repo and checked-out dependencies, then create missing tracking branches |

`agm open` behavior:

- `repo` opens the main checkout session
- the branch currently checked out in the main checkout also opens the main checkout session
- an existing worktree target opens its tmux session
- an existing branch without a worktree is checked out into a worktree and then opened
- a missing branch is created from `--parent` or the main checkout's current branch and then opened

`agm open` options:

- `-d`, `--detach`: create the tmux session without attaching
- `-n`, `--num-panes PANES`: create the session with `PANES` panes
- `-p`, `--parent PARENT`: base a newly created branch worktree on `PARENT`

`agm close` notes:

- closes only branch worktrees
- `repo` and the main checkout branch cannot be removed with `agm close`

`agm init` options:

- `--embedded`: force the embedded layout with AGM data under `.agm/`
- `--workspace`: force the workspace layout with `repo/`, `deps/`, `notes/`, `worktrees/`, and `config/`
- `-b`, `--branch BRANCH`: clone this branch when `REPO_URL` is provided

`agm init` layout selection:

- with `REPO_URL`, the default is the workspace layout unless `--embedded` is provided
- without `REPO_URL`, AGM chooses the embedded layout only when the target project directory already exists and is a git repo
- otherwise it chooses the workspace layout

## Loop automation

| Command | Description |
|---|---|
| `agm loop [--runner COMMAND] [--selector COMMAND] [--tasks-dir DIR] [--no-log\|--log-file PATH] CMD [RUNNER_ARGS...]` | Shorthand for `agm loop run` when `CMD` is not a built-in subcommand |
| `agm loop run [--runner COMMAND] [--selector COMMAND] [--tasks-dir DIR] [--no-log\|--log-file PATH] [CMD [RUNNER_ARGS...]]` | Run the loop until completion |
| `agm loop step [--runner COMMAND] [--selector COMMAND] [--tasks-dir DIR] [--no-log\|--log-file PATH] CMD [RUNNER_ARGS...]` | Perform a single loop iteration |
| `agm loop next [--runner COMMAND] [--selector COMMAND] [--tasks-dir DIR] [CMD [RUNNER_ARGS...]]` | Run the progress-update prompt once |

Loop config is loaded from merged `config.toml` files:

- `[loop]` defines default `runner`, `selector`, and `tasks_dir`
- `[loop.<command>]` overrides the base loop config for a specific prompt command
- CLI flags override config values

Loop behavior:

- without a selector, AGM appends `@<resolved-loop-prompt>` to the runner command and stops when the response is `COMPLETE` after whitespace is stripped
- with a selector, AGM first runs the selector with `@update_progress.md`
- if the selector returns `COMPLETE`, the loop stops
- otherwise the selector output is treated as the next task path and passed to the runner
- by default AGM writes `loop-YYYYMMDD-HHMMSS.log` in the current directory
- `--log-file PATH` writes to a specific file
- `--no-log` disables file logging

## Worktrees and dependencies

| Command | Description |
|---|---|
| `agm worktree new [-d\|--dir DIR] BRANCH` | Create a new branch worktree or check out an existing branch |
| `agm worktree setup` | Run configured setup scripts for the current checkout |
| `agm worktree remove [-f\|--force] BRANCH` | Remove a worktree and delete its local branch |
| `agm wt new [-d\|--dir DIR] BRANCH` | Alias form of `agm worktree new` |
| `agm wt setup` | Alias form of `agm worktree setup` |
| `agm wt rm [-f\|--force] BRANCH` | Alias form of `agm worktree remove` |
| `agm dep new [-b\|--branch BRANCH] REPO_URL` | Clone a new dependency |
| `agm dep switch [-b\|--branch] DEP BRANCH` | Switch a dependency to another branch |
| `agm dep rm [--all] DEP \| DEP/BRANCH \| DEP/repo \| DEP/MAIN_BRANCH` | Remove a dependency checkout or worktree |

`agm worktree new` options:

- `-d`, `--dir DIR`: create the worktree under `DIR` instead of the project's default worktrees directory

`agm worktree setup` runs executable setup scripts, in this order:

1. project-level `config/setup.sh`
2. checkout-local `.config/setup.sh`
3. checkout-local `.setup.sh`

`agm worktree remove` options:

- `-f`, `--force`: force removal even when git reports uncommitted or locked state

`agm dep new` options:

- `-b`, `--branch BRANCH`: clone `BRANCH` instead of the dependency's default branch

`agm dep switch` options:

- `-b`, `--branch`: create `BRANCH` from the dependency's default branch before adding the new worktree

`agm dep rm` targets:

- `DEP/BRANCH`: remove a dependency worktree for `BRANCH`
- `DEP/repo`: remove the main dependency checkout
- `DEP/MAIN_BRANCH`: remove the main dependency checkout by branch name

`agm dep rm` options:

- `--all`: remove the entire dependency directory, including the main checkout and any linked worktrees

## Configuration, sandboxing, and tmux

| Command | Description |
|---|---|
| `agm config copy DIRNAME` | Copy known project config files into an existing target directory |
| `agm config cp DIRNAME` | Alias form of `agm config copy` |
| `agm run [--no-sandbox] [--no-patch] [--memory LIMIT] [-f\|--file SETTINGS] COMMAND [ARGS...]` | Run a command directly or in an Anthropic Sandbox Runtime container |
| `agm tmux open [-d\|--detach] [-n\|--num-panes PANES] [SESSION]` | Open a tmux session |
| `agm tmux close SESSION` | Close a tmux session |
| `agm tmux layout PANES [-w\|--window WINDOW_ID]` | Apply AGM's tmux pane layout to a window |

`agm config copy` copies known files from the shared project config directory, including:

- `.setup.sh`
- `.env`
- `.env.local`
- `.config`
- `.agents`
- `.opencode`
- `.codex`
- `.claude`
- `.pi`
- `.mcp.json`

`agm run` config lookup:

1. `<install-prefix>/.agm/config.toml` when present, otherwise `$HOME/.agm/config.toml`
2. `<project-config-dir>/config.toml`
3. `./.agm/config.toml`

`agm run` options:

- `--no-sandbox`: run the command directly without `srt`; skips sandbox settings discovery and patching
- `-f`, `--file SETTINGS`: use one settings file directly instead of discovered settings
- `--memory LIMIT`: wrap the command in `systemd-run --user --scope -p MemoryMax=LIMIT`; defaults to `20G`; values `<= 0` disable memory limiting
- `--no-patch`: do not append project notes, deps, and repo `.git` paths to `filesystem.allowWrite`

Sandbox settings resolution:

- for each config directory, AGM prefers `<command>.json`
- if that file does not exist there, AGM tries the aliased command name's settings file
- if neither exists, AGM falls back to `default.json`
- AGM merges matching files in this order:
  1. `$HOME/.agm/sandbox/`
  2. the project sandbox config directory
  3. `./.sandbox/`
- later files override earlier ones
- `network` and `filesystem` are merged by key
- `ignoreViolations` replaces the earlier value
- `enabled` and `enableWeakerNestedSandbox` override when set

`agm tmux open` options:

- `-d`, `--detach`: create the session without attaching
- `-n`, `--num-panes PANES`: create the session with `PANES` panes

`agm tmux layout` options:

- `-w`, `--window WINDOW_ID`: apply the layout to a specific tmux window ID

## Help and aliases

| Alias | Canonical form |
|---|---|
| `agm wt` | `agm worktree` |
| `agm config cp` | `agm config copy` |
| `agm wt rm` | `agm worktree remove` |

Use `agm help` to show the command overview and `agm help <command>` for detailed command help.
