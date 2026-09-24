# Workspace and project lifecycle commands

| Command | Description |
|---|---|
| `agm open [-d\|--detach] [-n\|--num-panes PANES] [-p\|--parent PARENT] [--no-fetch] TARGET` | Shortcut for `agm workspace open` |
| `agm close [-f\|--force] [-D] [--keep-branch] [--keep-workspace] BRANCH` | Shortcut for `agm workspace close` |
| `agm workspace open [-d\|--detach] [-n\|--num-panes PANES] [-p\|--parent PARENT] [--no-fetch] TARGET` | Open the main workspace or a branch workspace, creating or checking it out when needed |
| `agm workspace close [-f\|--force] [-D] [--keep-branch] [--keep-workspace] BRANCH` | Remove a branch workspace and close its tmux session |
| `agm workspace list [-v\|--verbose]` | List all open AGM workspaces |
| `agm workspace setup` | Run setup scripts for the current workspace |
| `agm workspace shell-regen SHELL_DIR` | Regenerate the per-session shell wrapper and rc files in `SHELL_DIR` |
| `agm wsp open [-d\|--detach] [-n\|--num-panes PANES] [-p\|--parent PARENT] [--no-fetch] TARGET` | Alias form of `agm workspace open` |
| `agm wsp close [-f\|--force] [-D] [--keep-branch] [--keep-workspace] BRANCH` | Alias form of `agm workspace close` |
| `agm wsp list [-v\|--verbose]` | Alias form of `agm workspace list` |
| `agm wsp setup` | Alias form of `agm workspace setup` |
| `agm wsp shell-regen SHELL_DIR` | Alias form of `agm workspace shell-regen` |
| `agm init [--embedded \| --split] [--no-git-init \| --no-repo-git \| --no-config-git \| --no-notes-git]` | Initialize the current directory without cloning a repo |
| `agm init [--embedded \| --split] [--no-git-init \| --no-repo-git \| --no-config-git \| --no-notes-git] PROJECT_NAME` | Initialize a child project directory without cloning a repo |
| `agm init [--embedded \| --split] [-b\|--branch BRANCH] [--no-git-init \| --no-repo-git \| --no-config-git \| --no-notes-git] [PROJECT_NAME] REPO_URL` | Initialize the current directory or named child directory and clone a repo |
| `agm init --clone [--embedded \| --split] [-b\|--branch BRANCH] [--no-git-init \| --no-repo-git \| --no-config-git \| --no-notes-git] REPO_URL` | Initialize a URL-derived child project directory and clone a repo |
| `agm sync fetch` | Prune stale worktrees, fetch the main repo and checked-out dependencies, then create missing tracking branches |
| `agm sync pull` | Run `agm sync fetch`, then run `git merge` in the main workspace, every branch workspace, and every dependency |

An AGM workspace is the main repo or a linked Git worktree in the project's worktrees directory,
combined with AGM project config, workspace config, dependency environment, setup scripts, and tmux
session lifecycle. A branch workspace is named by its path under the worktrees directory (the
branch it was opened for); other Git worktrees of the repo are not workspaces.

`agm workspace open` behavior:

- `repo` opens the main workspace
- the branch currently checked out in the main workspace also opens the main workspace
- an existing branch workspace opens its tmux session
- an existing branch without a workspace is checked out into a Git worktree and then opened
- a branch that exists only on a remote is checked out as a tracking branch of whichever remote carries it; one carried by several remotes is ambiguous and rejected
- a missing branch is created from `--parent` or the main workspace's current branch and then opened
- with `--parent`, an existing target branch warns (`--parent` only bases new branches); an existing target workspace errors
- a workspace whose tmux session is already running errors instead of reopening, and nothing is created; attach to the running session instead
- `--no-fetch` skips Git fetches; remote branches are resolved using the local refs already available

`agm workspace open` options:

- `-d`, `--detach`: create the tmux session without attaching
- `-n`, `--num-panes PANES`: create the session with `PANES` panes
- `-p`, `--parent PARENT`: base a newly created branch workspace on `PARENT`
- `--no-fetch`: skip Git fetches, avoiding network access to remote repositories

`agm workspace close` options:

- `-f`, `--force`: force remove the branch workspace's Git worktree (even with untracked or uncommitted changes) and force delete the branch (`git branch -D`). Implies `-D`.
- `-D`: force delete the branch (`git branch -D`) instead of safe delete (`git branch -d`). The worktree is only removed if the branch deletion would succeed.
- `--keep-branch`: remove the worktree but keep the local branch.
- `--keep-workspace`: keep the branch workspace and local branch, remove no workspace config, and only close the workspace session. Implies `--keep-branch`.

`agm workspace close` notes:

- closes only branch workspaces: `repo` and the main workspace branch cannot be removed
- `BRANCH` names the workspace, so it is closed even when its checkout has switched branches or detached HEAD

`agm sync fetch` notes:

- prunes stale Git worktree registrations (those whose directories no longer exist) in each repo before fetching

`agm sync pull` notes:

- runs `agm sync fetch`'s prune, fetch, and tracking-branch sync first, then `git merge` in the main workspace, each branch workspace, and each dependency checkout/worktree
- relies on each Git worktree's current branch upstream, matching plain `git merge`

`agm workspace list` options:

- `-v`, `--verbose`: show the workspace directory path after each workspace

`agm workspace list` notes:

- the main workspace is listed first, by its checked-out branch
- branch workspaces follow, sorted by name; `(detached)` or `(on OTHER)` marks a checkout whose HEAD is detached or on another branch
- the current workspace is indicated with a leading `*`

`agm workspace setup` runs executable setup scripts for the current workspace, in this order:

1. project-level `config/setup.sh`
2. workspace-local `.config/setup.sh`
3. workspace-local `.setup.sh`

`agm workspace open` session shell:

- each session runs the user's real interactive shell (`zsh`/`bash`/`sh`) through a wrapper: it sources `~/.zshrc`/`~/.bashrc`/`~/.shrc` (preserving keybindings, prompts, completions, aliases), restores the project and workspace paths selected by `agm workspace open`, then runs `eval "$(agm config env)"` so the workspace environment wins over the user's rc and inherited tmux state
- the wrapper and its rc files live under `$XDG_CACHE_HOME/agm/shell/<key>/` (defaulting to `~/.cache/agm/shell/<key>/`), keyed by session name; nothing is written under the project's `.agent-files/`
- `agm workspace open` recreates the per-session dir fresh (cleaning any stale files); `agm workspace close` removes it
- `agm workspace shell-regen SHELL_DIR` rewrites the wrapper and rc files into an existing per-session dir (used for manual recovery)

`agm init` options:

- `--embedded`: force the embedded layout with AGM data under `.agm/`
- `--split`: force the split layout with `repo/`, `deps/`, `notes/`, `worktrees/`, and `config/`
- `--clone`: initialize a child directory derived from `REPO_URL` when no `PROJECT_NAME` is provided
- `-b`, `--branch BRANCH`: clone this branch when `REPO_URL` is provided
- `--no-git-init`: do not create git repositories in `repo/`, `config/`, and `notes/`
- `--no-repo-git`: do not create a git repository in `repo/`
- `--no-config-git`: do not create a git repository in `config/`
- `--no-notes-git`: do not create a git repository in `notes/`

`agm init` layout selection:

- with `REPO_URL`, split layout by default unless `--embedded`; without `REPO_URL`, embedded layout only when the target project directory is a git repo, otherwise split
- initializes the current directory by default, a child directory named `PROJECT_NAME` when given, or a `REPO_URL`-derived child directory with `--clone`
- when an embedded repository has no commits, AGM creates its initial commit containing the
  generated `.gitignore`

`agm init` split layout notes:

- without `REPO_URL`, AGM initializes `repo/` as an empty git repository
- AGM writes a `.agent-files` entry into `repo/.gitignore` and the repository's `info/exclude`,
  leaving the tracked tree untouched; every branch worktree inherits the `info/exclude` copy, so
  agent artifacts never make a workspace look dirty
