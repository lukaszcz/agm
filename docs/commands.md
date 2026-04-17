# AGM commands reference

## Project session management

| Command | Description |
|---|---|
| `agm open [-d] [-n PANES] [-p PARENT] TARGET` | Open `repo/`, an existing branch worktree, or create/check out the branch as needed |
| `agm init [-b BRANCH] PROJECT_NAME` | Initialize a new project directory without cloning a repository |
| `agm init [-b BRANCH] [PROJECT_NAME] REPO_URL` | Initialize a new project directory and clone the repository into `repo/` |
| `agm fetch` | Fetch latest changes for the repo and all dependencies |

`agm open` options:
- `-d` creates the tmux session without attaching to it.
- `-n PANES` creates the session with `PANES` panes.
- `-p PARENT` bases a newly created branch worktree on `PARENT` instead of the current branch in `repo/`.

`agm init` options:
- `-b BRANCH` clones `BRANCH` when `REPO_URL` is provided.

## Branch and worktree management

| Command | Description |
|---|---|
| `agm branch sync` | Fetch/prune origin and create local tracking branches |
| `agm worktree new [-d DIR] BRANCH` | Create a new branch worktree or check out an existing branch |
| `agm worktree setup` | Run setup scripts for the current repo or worktree checkout |
| `agm worktree remove [-f] BRANCH` | Remove a worktree and delete the local branch |

`agm worktree new` options:
- `-d DIR` creates the worktree under `DIR` instead of the default `worktrees/` or `.worktrees/` directory.

`agm worktree remove` options:
- `-f` forces worktree removal even when git reports uncommitted or locked state.

## Dependency management

| Command | Description |
|---|---|
| `agm dep new [-b BRANCH] REPO_URL` | Clone a new dependency into deps/ |
| `agm dep rm [--all] DEP | DEP/BRANCH | DEP/repo | DEP/MAIN_BRANCH` | Remove a dependency worktree or the main dependency checkout |
| `agm dep switch [-b] DEP BRANCH` | Switch a dependency to a different branch |

`agm dep new` options:
- `-b BRANCH` clones `BRANCH` instead of the dependency's default branch.

`agm dep rm` options:
- `--all` removes the entire dependency directory; the target must be `DEP`.

`agm dep rm` targets:
- `DEP/BRANCH` removes a dependency worktree for `BRANCH`.
- `DEP/repo` removes the main dependency checkout.
- `DEP/MAIN_BRANCH` removes the main dependency checkout by branch name.

`agm dep switch` options:
- `-b` creates `BRANCH` from the dependency's default branch before adding the new worktree.

## Configuration and sandbox

By default, `agm run` derives the sandbox settings filename from the command
basename. In each of `$HOME/.agm/sandbox`, `$PROJ_DIR/config/sandbox`, and
`./.sandbox`, it prefers `<command>.json` and falls back to `default.json` only
when `<command>.json` does not exist in that directory. Existing files are then
merged in that order, with more local files taking precedence.

`-f SETTINGS` skips that discovery and uses the given settings file directly.
Unless `--no-patch` is set, `agm run` also adds `$PROJ_DIR/notes` and
`$PROJ_DIR/deps` to `filesystem.allowWrite` after loading the selected
settings.

| Command | Description |
|---|---|
| `agm config copy [-d PROJECT_DIR] DIRNAME` | Copy project configuration files into an existing target directory |
| `agm run [--no-patch] [-f SETTINGS] COMMAND [ARGS...]` | Run a command in an Anthropic Sandbox Runtime |

`agm config copy` options:
- `-d PROJECT_DIR` reads shared config from `PROJECT_DIR` instead of auto-detecting the current project.

`agm run` options:
- `-f SETTINGS` skips default sandbox settings discovery and uses `SETTINGS` directly.
- `--no-patch` skips adding `$PROJ_DIR/notes` and `$PROJ_DIR/deps` to `filesystem.allowWrite`.

## Tmux

| Command | Description |
|---|---|
| `agm tmux open [-d] [-n PANES] [SESSION]` | Open a tmux session |
| `agm tmux close SESSION` | Kill a tmux session by name |
| `agm tmux layout PANES [--window WINDOW_ID]` | Apply a tiled pane layout to the current tmux window |

`agm tmux open` options:
- `-d` creates the session without attaching to it.
- `-n PANES` creates the session with `PANES` panes.

## Aliases

| Alias | Canonical form |
|---|---|
| `agm br` | `agm branch` |
| `agm wt` | `agm worktree` |
| `agm config cp` | `agm config copy` |
| `agm wt rm` | `agm worktree remove` |
