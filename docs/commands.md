# AGM commands reference

## Project session management

| Command | Description |
|---|---|
| `agm open [-n PANES] [BRANCH]` | Open a tmux session for a project branch |
| `agm new [-n PANES] [-p PARENT] BRANCH` | Create a new branch worktree and open a tmux session |
| `agm checkout [-n PANES] [-p PARENT] BRANCH` | Check out a branch into a worktree and open a tmux session |
| `agm init [-b BRANCH] [PROJECT_NAME] REPO_URL` | Initialize a new project by cloning a repository |
| `agm fetch` | Fetch latest changes for the repo and all dependencies |

## Branch and worktree management

| Command | Description |
|---|---|
| `agm branch sync` | Fetch/prune origin and create local tracking branches |
| `agm worktree checkout [-b BRANCH] [-d DIR] [BRANCH]` | Check out a branch into a worktree |
| `agm worktree new [-d DIR] BRANCH` | Create a new branch and its worktree |
| `agm worktree remove [-f] BRANCH` | Remove a worktree and delete the local branch |

## Dependency management

| Command | Description |
|---|---|
| `agm dep new [-b BRANCH] REPO_URL` | Clone a new dependency into deps/ |
| `agm dep switch [-b] DEP BRANCH` | Switch a dependency to a different branch |

## Configuration and sandbox

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
| `agm co` | `agm checkout` |
| `agm config cp` | `agm config copy` |
| `agm wt co` | `agm worktree checkout` |
| `agm wt rm` | `agm worktree remove` |
