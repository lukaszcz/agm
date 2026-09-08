# Projects and Workspaces

The project-management half of AGM turns a directory tree into an agent-oriented project: a main repository, parallel git worktrees for branches, dependency checkouts, per-workspace environment and setup, and tmux sessions. State is read from disk rather than tracked separately, so commands detect the current layout and act on it.

## Project Layout

A project has a fixed set of roles — repository, branch worktrees, config directory, dependencies, notes — arranged in one of two layouts: **embedded** (the AGM directories live under `.agm/` inside the project root) or **split** (the roles are sibling directories under the project root). The layout module detects the layout from the directory structure and resolves each role's path. `agm init` creates a project in either layout, optionally cloning a repository.

## Workspaces and Worktrees

A *workspace* is the main repository or a linked git worktree for a branch, interpreted with AGM's project config, dependency environment, setup scripts, and tmux lifecycle. Opening a branch without a worktree checks it out into one at a path derived from the branch name; opening a missing branch creates it first. Both paths share the preparation flow in `commands/workspace/open.py`, which assembles config and environment, creates the worktree, commits generated config, and starts the session. Worktree orchestration coordinates git with dependency setup: creating the worktree, ensuring tracking branches, resolving a branch across remotes the way git does (exactly one remote may carry it). Only branch workspaces can be closed; closing can retain the branch or the worktree while still ending the session.

## Dependencies

Dependencies are sibling repositories under the project's deps directory, managed by the `dep` command group. Checkout discovery accepts only actual Git roots, excluding ordinary directories that happen to sit inside an enclosing repository, and distinguishes the main checkout from linked worktrees. Each dependency contributes `_DIR` environment variables for its checked-out branch, assembled from the project's dependency TOML tables, so a workspace's environment reflects which branch of each dependency is active. Branch dependency configs inherit from the main config.

## Sync

`sync fetch` prunes stale worktree registrations, fetches the main repo and checked-out dependencies, and creates missing tracking branches. `sync pull` runs that fetch and then merges every dependency, the main workspace, and each branch workspace from its configured upstream.

## Workspace Environment and Shell

When a workspace opens, its environment chains the dependency environment, the project and branch config directories' dotenv files, and shell env files. A per-workspace shell wrapper is generated so interactive sessions start with that environment; the wrapper also serves as the workspace's `$SHELL`, so invoked with arguments it runs them through the real shell instead of opening one. Configured setup scripts run to prepare the workspace.

## Git and Tmux

All git work goes through one VCS module wrapping git as subprocess calls; every helper accepts an explicit environment so it composes with workspace environments. Workspace sessions are tmux sessions created with a filtered environment and a tiled pane layout; tmux object IDs keep targeting independent of user index settings, session names are unique per workspace, exact-name targeting protects session operations from tmux prefix matching, and an occupied name stops `open` before any git work. All tmux invocations go through the module's wrappers, so a missing binary is a plain error.

## Code Entry Points

- `src/agm/project/layout.py` — layout detection, role-path resolution, current-workspace detection.
- `src/agm/project/worktree.py` — worktree creation and branch/remote synchronization.
- `src/agm/project/workspace_env.py`, `workspace_shell.py`, `workspace_setup.py` — environment assembly, shell wrapper, setup scripts.
- `src/agm/project/dependency_env.py`, `dependency_checkout.py`, `config_git.py` — dependency env vars, checkout discovery, config-directory git operations.
- `src/agm/vcs/git.py` — the git integration surface.
- `src/agm/tmux/session.py`, `src/agm/tmux/layout.py` — tmux session creation and pane layout.
- `src/agm/commands/init.py`, `workspace/`, `worktree/`, `dep/`, `sync/`, `tmux/` — the management commands.
