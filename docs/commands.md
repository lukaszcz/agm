# AGM commands reference

## Project session management

| Command | Description |
|---|---|
| `agm open [-d] [-n PANES] [-p PARENT] TARGET` | Open `repo/`, an existing branch worktree, or create/check out the branch as needed |
| `agm init [-b BRANCH] [PROJECT_NAME] REPO_URL` | Initialize a new project by cloning a repository |
| `agm fetch` | Fetch latest changes for the repo and all dependencies |

## Branch and worktree management

| Command | Description |
|---|---|
| `agm branch sync` | Fetch/prune origin and create local tracking branches |
| `agm worktree checkout [-b BRANCH] [-d DIR] [BRANCH]` | Check out a branch into a worktree |
| `agm worktree new [-d DIR] BRANCH` | Create a new branch and its worktree |
| `agm worktree setup` | Run setup scripts for the current repo or worktree checkout |
| `agm worktree remove [-f] BRANCH` | Remove a worktree and delete the local branch |

## Dependency management

| Command | Description |
|---|---|
| `agm dep new [-b BRANCH] REPO_URL` | Clone a new dependency into deps/ |
| `agm dep switch [-b] DEP BRANCH` | Switch a dependency to a different branch |

## Configuration and sandbox

By default, `agm run` loads sandbox settings from `$HOME/.agm/sandbox/default.json`
and merges them with `$PROJ_DIR/config/sandbox/default.json` and `./.sandbox/default.json`
when present, with more local files taking precedence.

| Command | Description |
|---|---|
| `agm config copy [-d PROJECT_DIR] DIRNAME` | Copy project configuration files |
| `agm run [--no-patch] [-f SETTINGS] COMMAND [ARGS...]` | Run a command in an Anthropic Sandbox Runtime |

## Tmux

| Command | Description |
|---|---|
| `agm tmux new [-d] [-n PANES] [SESSION]` | Create a new tmux session |
| `agm tmux layout PANES WINDOW_ID WIDTH HEIGHT` | Apply a tiled pane layout to a tmux window |

## Aliases

| Alias | Canonical form |
|---|---|
| `agm br` | `agm branch` |
| `agm wt` | `agm worktree` |
| `agm config cp` | `agm config copy` |
| `agm wt co` | `agm worktree checkout` |
| `agm wt rm` | `agm worktree remove` |
