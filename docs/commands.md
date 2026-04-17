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
| `agm worktree new [-d DIR] BRANCH` | Create a new branch worktree or check out an existing branch |
| `agm worktree setup` | Run setup scripts for the current repo or worktree checkout |
| `agm worktree remove [-f] BRANCH` | Remove a worktree and delete the local branch |

## Dependency management

| Command | Description |
|---|---|
| `agm dep new [-b BRANCH] REPO_URL` | Clone a new dependency into deps/ |
| `agm dep switch [-b] DEP BRANCH` | Switch a dependency to a different branch |

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
| `agm wt rm` | `agm worktree remove` |
