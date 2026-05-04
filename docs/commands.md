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
| `agm init [--embedded \| --workspace]` | Initialize the current directory without cloning a repo |
| `agm init [--embedded \| --workspace] PROJECT_NAME` | Initialize a child project directory without cloning a repo |
| `agm init [--embedded \| --workspace] [-b\|--branch BRANCH] [PROJECT_NAME] REPO_URL` | Initialize the current directory or named child directory and clone a repo |
| `agm init --clone [--embedded \| --workspace] [-b\|--branch BRANCH] REPO_URL` | Initialize a URL-derived child project directory and clone a repo |
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
- `--clone`: initialize a child directory derived from `REPO_URL` when no `PROJECT_NAME` is provided
- `-b`, `--branch BRANCH`: clone this branch when `REPO_URL` is provided

`agm init` layout selection:

- with `REPO_URL`, the default is the workspace layout unless `--embedded` is provided
- without `REPO_URL`, AGM chooses the embedded layout only when the target project directory is a git repo
- otherwise it chooses the workspace layout
- without `PROJECT_NAME`, AGM initializes the current directory
- with `PROJECT_NAME`, AGM initializes a child directory with that name
- with `--clone REPO_URL`, AGM initializes a child directory derived from the URL

## Loop automation

| Command | Description |
|---|---|
| `agm loop [--runner COMMAND] [--selector COMMAND\|--no-selector] [--tasks-dir DIR] [--no-log\|--log-file PATH] CMD [RUNNER_ARGS...]` | Shorthand for `agm loop run` when `CMD` is not a built-in subcommand |
| `agm loop run [--runner COMMAND] [--selector COMMAND\|--no-selector] [--tasks-dir DIR] [--no-log\|--log-file PATH] [CMD [RUNNER_ARGS...]]` | Run the loop until completion |
| `agm loop step [--runner COMMAND] [--selector COMMAND\|--no-selector] [--tasks-dir DIR] [--no-log\|--log-file PATH] CMD [RUNNER_ARGS...]` | Perform a single loop iteration |
| `agm loop next [--runner COMMAND] [--selector COMMAND\|--no-selector] [--tasks-dir DIR] [CMD [RUNNER_ARGS...]]` | Run the progress-update prompt once |

Loop config is loaded from merged `config.toml` files:

- `[loop]` defines default `runner`, `selector`, `no_selector`, and `tasks_dir`
- `[loop.<command>]` overrides the base loop config for a specific prompt command
- `agm loop CMD` is shorthand for `agm loop run CMD` when `CMD` is not a built-in subcommand, and selects `[loop.CMD]` overrides
- CLI flags (`--runner`, `--selector`, `--no-selector`, `--tasks-dir`) override config values
- `RUNNER_ARGS` are appended to the final runner command after AGM resolves `--runner`, config, or the built-in default
- bare `agm loop` prints help text

Prompt file path:

- AGM passes the resolved prompt file path to the runner/selector command
- by default it is appended as `@<path>` to the command
- use `%%` or `%{PROMPT_FILE}` in the command to insert the path at a specific position — when either placeholder is present, it is replaced with the path and no `@<path>` suffix is appended

Selector mode (default):

- AGM runs the selector with `@update_progress.md`
- if the selector returns `COMPLETE` after whitespace is removed, AGM stops
- otherwise the selector output is treated as the next task path and AGM runs the runner with that task file
- when no explicit selector command is configured, the runner command is used for the progress update

No-selector mode (`--no-selector` / `no_selector = true`):

- AGM appends the loop prompt to the runner command
- stops when the runner response is `COMPLETE` after whitespace is removed

Subcommands:

- `agm loop step` performs a single loop iteration using the same runner, selector, and logging behavior as `agm loop run`
- `agm loop next` runs `update_progress.md` once using the resolved selector, or the resolved runner when no selector is configured — it requires selector mode; `--no-selector` is an error for `loop next`

Logging:

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
| `agm dep new [-b\|--branch BRANCH] REPO_URL` | Clone a new dependency checkout |
| `agm dep switch [-b\|--branch] DEP BRANCH` | Select or add a dependency checkout |
| `agm dep rm --all DEP` | Remove an entire dependency directory |
| `agm dep rm DEP/NAME_OR_BRANCH \| DEP/repo \| DEP/MAIN_CHECKOUT` | Remove a dependency checkout or worktree |

`agm worktree new` options:

- `-d`, `--dir DIR`: use `agm worktree new --dir DIR BRANCH` to create the worktree under `DIR` instead of the project's default worktrees directory

`agm worktree setup` runs executable setup scripts, in this order:

1. project-level `config/setup.sh`
2. checkout-local `.config/setup.sh`
3. checkout-local `.setup.sh`

`agm worktree remove` options:

- `-f`, `--force`: use `agm worktree remove --force BRANCH` to force removal even when git reports uncommitted or locked state

`agm dep new` options:

- `-b`, `--branch BRANCH`: use `agm dep new --branch BRANCH REPO_URL` to clone the dependency's initial checkout from `BRANCH` instead of the dependency's default branch

`agm dep switch` options:

- `-b`, `--branch`: use `agm dep switch --branch DEP BRANCH` to create `DEP`'s `BRANCH` from the dependency's default branch before adding the new worktree; without this flag, `BRANCH` must already exist

Dependency commands track selected dependency checkout names in config `config.toml` `[deps]` tables. Environment loading turns those entries into dependency path variables, so `[deps].vyper-automation = "feat/app"` provides `VYPER_AUTOMATION=/path/to/proj/deps/vyper-automation/feat/app` before `.env` and `env.sh` are loaded.

`agm dep rm` targets:

- `DEP/NAME_OR_BRANCH`: remove a dependency checkout by directory name under `deps/DEP/` or by checked-out branch name
- `DEP/repo`: remove the main dependency checkout
- `DEP/MAIN_CHECKOUT`: remove the main dependency checkout by directory name

`agm dep rm` options:

- `--all DEP`: use `agm dep rm --all DEP` to remove the entire dependency directory, including the main checkout and any linked worktrees

## Configuration, sandboxing, and tmux

| Command | Description |
|---|---|
| `agm config copy DIRNAME` | Copy known project config files into an existing target directory |
| `agm config cp DIRNAME` | Alias form of `agm config copy` |
| `agm config env` | Print shell statements for refreshing the current checkout environment |
| `agm run [--no-sandbox] [--no-patch] [--memory LIMIT] [--swap LIMIT] [--no-memory-limit] [--no-swap-limit] [-f\|--file SETTINGS] COMMAND [ARGS...]` | Run a command directly or in an Anthropic Sandbox Runtime container |
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

`agm config env` uses the same environment resolution as `agm open`: project and branch
`config.toml` `[deps]` tables first, then project `.env`, project `.env.local`, project
`env.sh`, and matching branch config files when the current checkout is a branch worktree.
Apply the printed shell statements with:

```bash
eval "$(agm config env)"
```

`agm run` config lookup:

1. `<install-prefix>/.agm/config.toml` when present, otherwise `$HOME/.agm/config.toml`
2. `<project-config-dir>/config.toml`
3. `./.agm/config.toml`

`agm run` config keys:

- `[run].memory`: default `MemoryMax` for sandboxed runs
- `[run].swap`: default `MemorySwapMax` for sandboxed runs
- `[run.<command>].memory`: per-command `MemoryMax` override
- `[run.<command>].swap`: per-command `MemorySwapMax` override
- `[run.<command>].alias`: replace the invoked command name before execution

`agm run` options:

- `--no-sandbox`: run the command directly without `srt`; skips sandbox settings discovery and patching
- `-f`, `--file SETTINGS`: use one settings file directly instead of discovered settings
- `--memory LIMIT`: set `MemoryMax=LIMIT` in the delegated `systemd-run --user --scope`; the bootstrap exports `SANDBOX_CGROUP` and enables the memory controller for descendant cgroups; defaults to `20G` in sandbox mode; `0` means a zero memory limit; `unlimited` means no memory cap
- `--swap LIMIT`: set `MemorySwapMax=LIMIT` in the delegated scope; defaults to `0` in sandbox mode; `unlimited` means no swap cap
- `--no-memory-limit`: do not set `MemoryMax`
- `--no-swap-limit`: do not set `MemorySwapMax`
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
