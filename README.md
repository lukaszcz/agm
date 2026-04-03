# devtools

Small shell utilities for local Git/GitHub workflows: worktrees, PR notes sync, and branch cleanup.

## Requirements

- `bash`
- `git`
- [`just`](https://github.com/casey/just) (for installation)
- [`gh`](https://cli.github.com/) (required by `prlog.sh` and `prsync.sh`)
- [`srt`](https://github.com/anthropic-experimental/sandbox-runtime) (required by `sandbox.sh`)

## Install

Install all scripts to `prefix/bin` (default: `$HOME/.local/bin`):

```bash
just install
```

This also installs all files from `sandbox/` to `$HOME/.sandbox/`.

Install to a custom prefix:

```bash
just install /usr/local
```

Install as symlinks instead of copying files:

```bash
just install "$HOME/.local" true
```

Top-level overrides (named style):

```bash
just prefix="$HOME/.local" symlink=true install
```

## Scripts

### `mkwt.sh`

Create a git worktree and optionally create a new branch.

```bash
# existing branch
mkwt.sh feature/my-branch

# create new branch
mkwt.sh -b feature/new-branch

# custom worktrees dir
mkwt.sh -d .worktrees feature/my-branch
```

### `pm-dep.sh`

Manage dependency checkouts under `deps/DEP/BRANCH`.

```bash
# clone a new dependency into deps/repo-name/main
pm-dep.sh new https://github.com/org/repo-name.git

# clone a specific branch into deps/repo-name/feature/foo
pm-dep.sh new -b feature/foo https://github.com/org/repo-name.git

# add a worktree for an existing branch
pm-dep.sh switch repo-name feature/foo

# create a new branch worktree from the dependency's default branch
pm-dep.sh switch repo-name -b feature/bar
```

### `pm-fetch.sh`

Fetch `repo/` and one checked out worktree for each dependency under `deps/DEP/`.

```bash
pm-fetch.sh
```

### `rmwt.sh`

Remove a worktree by branch name, then delete the branch locally.

```bash
rmwt.sh feature/my-branch
rmwt.sh -f feature/my-branch
```

### `brsync.sh`

Fetch/prune `origin` and create local tracking branches for remote branches that are not merged into `origin/main`.

```bash
brsync.sh
```

### `git-rm-branches.sh`

Delete all local branches except `main` and the currently checked out branch.

```bash
git-rm-branches.sh
```

Use with care: this force-deletes branches (`git branch -D`).

### `prlog.sh`

Create a draft PR from current branch, move PR notes into `docs/history/pr_<N>_<slug>.md`, commit, and push.

```bash
prlog.sh -t "feat: add useful thing" -F PR.md
```

Notes:
- Requires a clean working tree/index before running.
- `-F` defaults to `PR.md`.

### `prsync.sh`

Sync current branch PR title/body into `docs/history/pr_<N>_<slug>.md`, rename file if title subject changed, then commit.

```bash
prsync.sh
```

Notes:
- Requires a clean working tree/index before running.
- Expects PR title format `type: subject`.

### `sandbox.sh`

Run a command inside [Anthropic Sandbox Runtime](https://github.com/anthropic-experimental/sandbox-runtime).

```bash
# use ~/.sandbox/default.json and/or ./.sandbox/default.json
sandbox.sh npm test

# use an explicit settings file
sandbox.sh -f .sandbox/ci.json npm test

# disable the temporary PROJ_DIR-based patching
sandbox.sh --no-patch npm test
```

Notes:
- If both default settings files exist, `./.sandbox/default.json` overrides `~/.sandbox/default.json` with section-aware merging.
- If neither default settings file exists, the script exits with an error.
- If `PROJ_DIR` is set, the selected settings file is patched temporarily to add `$PROJ_DIR/notes` and `$PROJ_DIR/issues` to `filesystem.allowWrite`.
- Use `--no-patch` to disable the temporary patching.
