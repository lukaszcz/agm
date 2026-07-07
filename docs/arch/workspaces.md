# Projects and Workspaces

The project-management half of AGM turns a directory tree into an agent-oriented project: a main repository, parallel git worktrees for branches, dependency checkouts, per-workspace environment and setup, and tmux sessions. State is read from disk rather than tracked separately, so commands detect the current layout and act on it.

## Project Layout

A project has a fixed set of roles — the repository, branch worktrees, the config directory, dependencies, and notes — arranged in one of two layouts:

- **Embedded** — the AGM directories live under an `.agm/` directory inside the project root.
- **Split** — the roles are parallel sibling directories under the project root.

The layout module detects which layout is in use from the directory structure and resolves each role's path accordingly. `agm init` creates a project in either layout, optionally cloning a repository and initializing git in the relevant roles.

## Workspaces and Worktrees

A *workspace* is either the main repository or a linked git worktree for a branch, interpreted with AGM's project config, dependency environment, setup scripts, and tmux lifecycle. Opening a branch that has no worktree checks it out into one; opening a missing branch creates it first. Branch workspaces map to a predictable worktree path derived from the branch name.

Worktree orchestration coordinates git operations with dependency setup: creating a worktree, ensuring tracking branches exist, and syncing remote-tracking branches. The `worktree` and `workspace` command groups expose creation, removal, listing, and opening; only branch workspaces can be closed (the main workspace and its branch are protected), and close can optionally retain the branch or the whole workspace while still closing the session.

## Dependencies

Dependencies are sibling repositories checked out under the project's deps directory, managed by the `dep` command group (list, new, switch, remove). Each dependency contributes environment variables describing its checked-out branch, assembled from the project's dependency TOML tables, so that a workspace's environment reflects which branch of each dependency is active. Branch dependency configs are inherited from the parent or main config; checkout directories under `deps/` are resolved only for dependencies that the inherited config already declares.

## Sync

The `sync` command group keeps repositories current. `sync fetch` prunes stale worktree registrations, fetches the main repo and checked-out dependencies, and creates any missing tracking branches. `sync pull` runs that fetch and then merges in every dependency, the main workspace, and each branch workspace, relying on each worktree's configured upstream.

## Workspace Environment and Shell

When a workspace opens, its environment is assembled by chaining dependency environment, the project and branch-specific config directories' dotenv files, and shell env files. A per-workspace shell wrapper is generated so interactive sessions start with that environment, and configured setup scripts run to prepare the workspace.

## Git Integration

All git work goes through one VCS module that wraps git as subprocess calls: repository and root detection, branch queries, worktree listing/creation/deletion, tracking-branch creation, remote fetch/prune, and unmerged-branch discovery. Every helper accepts an explicit environment so it composes with workspace environments. This is the single place git semantics live.

## Tmux

Workspace sessions are realized as tmux sessions. The tmux module creates a session with a filtered environment (dropping terminal- and SSH-specific variables and unsafe names), handles attached vs. detached creation and nested-tmux detection, and applies a tiled pane layout. The `tmux` command group exposes session open/close and layout directly.

## Code Entry Points

- `src/agm/project/layout.py` — layout detection, role-path resolution, current-workspace detection, config copying.
- `src/agm/project/worktree.py` — worktree creation and branch/remote synchronization.
- `src/agm/project/workspace_env.py`, `workspace_shell.py`, `workspace_setup.py` — environment assembly, shell-wrapper generation, setup-script execution.
- `src/agm/project/dependency_env.py`, `dependency_checkout.py`, `config_git.py` — dependency env vars, checkout discovery, config-directory git operations.
- `src/agm/vcs/git.py` — the git integration surface.
- `src/agm/tmux/session.py` and `src/agm/tmux/layout.py` — tmux session creation and pane layout.
- `src/agm/commands/init.py`, `workspace/`, `worktree/`, `dep/`, `sync/`, `tmux/` — the management commands.
