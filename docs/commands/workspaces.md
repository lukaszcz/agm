# Workspace and project lifecycle commands

| Command | Description |
|---|---|
| `agm open [-d\|--detach] [-n\|--num-panes PANES] [-p\|--parent PARENT] TARGET` | Shortcut for `agm workspace open` |
| `agm close [-f\|--force] [-D] BRANCH` | Shortcut for `agm workspace close` |
| `agm workspace open [-d\|--detach] [-n\|--num-panes PANES] [-p\|--parent PARENT] TARGET` | Open the main workspace or a branch workspace, creating or checking it out when needed |
| `agm workspace close [-f\|--force] [-D] BRANCH` | Remove a branch workspace and close its tmux session |
| `agm workspace list [-v\|--verbose]` | List all open AGM workspaces |
| `agm workspace setup` | Run setup scripts for the current workspace |
| `agm workspace shell-regen SHELL_DIR` | Regenerate the per-session shell wrapper and rc files in `SHELL_DIR` |
| `agm wsp open [-d\|--detach] [-n\|--num-panes PANES] [-p\|--parent PARENT] TARGET` | Alias form of `agm workspace open` |
| `agm wsp close [-f\|--force] [-D] BRANCH` | Alias form of `agm workspace close` |
| `agm wsp list [-v\|--verbose]` | Alias form of `agm workspace list` |
| `agm wsp setup` | Alias form of `agm workspace setup` |
| `agm wsp shell-regen SHELL_DIR` | Alias form of `agm workspace shell-regen` |
| `agm init [--embedded \| --split] [--no-git-init \| --no-repo-git \| --no-config-git \| --no-notes-git]` | Initialize the current directory without cloning a repo |
| `agm init [--embedded \| --split] [--no-git-init \| --no-repo-git \| --no-config-git \| --no-notes-git] PROJECT_NAME` | Initialize a child project directory without cloning a repo |
| `agm init [--embedded \| --split] [-b\|--branch BRANCH] [--no-git-init \| --no-repo-git \| --no-config-git \| --no-notes-git] [PROJECT_NAME] REPO_URL` | Initialize the current directory or named child directory and clone a repo |
| `agm init --clone [--embedded \| --split] [-b\|--branch BRANCH] [--no-git-init \| --no-repo-git \| --no-config-git \| --no-notes-git] REPO_URL` | Initialize a URL-derived child project directory and clone a repo |
| `agm sync fetch` | Prune stale worktrees, fetch the main repo and checked-out dependencies, then create missing tracking branches |
| `agm sync pull` | Run `agm sync fetch`, then run `git merge` in every dependency, main workspace, and branch workspace |

An AGM workspace may be the main repo or a linked Git worktree, interpreted with AGM project
config, workspace config, dependency environment, setup scripts, and tmux session lifecycle.

`agm workspace open` behavior:

- `repo` opens the main workspace
- the branch currently checked out in the main workspace also opens the main workspace
- an existing branch workspace opens its tmux session
- an existing branch without a workspace is checked out into a Git worktree and then opened
- a missing branch is created from `--parent` or the main workspace's current branch and then opened

`agm workspace open` options:

- `-d`, `--detach`: create the tmux session without attaching
- `-n`, `--num-panes PANES`: create the session with `PANES` panes
- `-p`, `--parent PARENT`: base a newly created branch workspace on `PARENT`

`agm workspace close` options:

- `-f`, `--force`: force remove the branch workspace's Git worktree (even with untracked or uncommitted changes) and force delete the branch (`git branch -D`). Implies `-D`.
- `-D`: force delete the branch (`git branch -D`) instead of safe delete (`git branch -d`). The worktree is only removed if the branch deletion would succeed.

`agm workspace close` notes:

- closes only branch workspaces
- `repo` and the main workspace branch cannot be removed with `agm workspace close`

`agm sync fetch` notes:

- prunes stale Git worktree registrations (those whose directories no longer exist) in each repo before fetching

`agm sync pull` notes:

- runs the same prune, fetch, and tracking-branch sync as `agm sync fetch` first
- runs `git merge` in each dependency checkout/worktree, the main workspace, and each branch workspace
- relies on each Git worktree's current branch upstream, matching plain `git merge`

`agm workspace list` options:

- `-v`, `--verbose`: show the workspace directory path after each branch name

`agm workspace list` notes:

- the main workspace is listed first
- the current workspace is indicated with a leading `*`

`agm workspace setup` runs executable setup scripts for the current workspace, in this order:

1. project-level `config/setup.sh`
2. workspace-local `.config/setup.sh`
3. workspace-local `.setup.sh`

`agm workspace open` session shell:

- each workspace tmux session runs the user's real interactive shell (`zsh`/`bash`/`sh`) through a small wrapper that first sources `~/.zshrc`/`~/.bashrc`/`~/.shrc` (so keybindings, prompts, completions and aliases are preserved) and then appends `eval "$(agm config env)"` so the workspace environment wins over the user's rc
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

- with `REPO_URL`, the default is the split layout unless `--embedded` is provided
- without `REPO_URL`, AGM chooses the embedded layout only when the target project directory is a git repo
- otherwise it chooses the split layout
- without `PROJECT_NAME`, AGM initializes the current directory
- with `PROJECT_NAME`, AGM initializes a child directory with that name
- with `--clone REPO_URL`, AGM initializes a child directory derived from the URL

`agm init` split layout notes:

- without `REPO_URL`, AGM initializes `repo/` as an empty git repository
- `--no-repo-git` skips the empty `repo/` git repository initialization
- `--no-git-init` includes `--no-repo-git`, `--no-config-git`, and `--no-notes-git`
