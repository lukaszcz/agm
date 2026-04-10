# Plan: Migrate Shell Scripts to Python

## Overview

Replace all 10 bash/zsh scripts in `scripts/` with Python modules in `src/agm/`.
The current `commands.py` dispatches to scripts via `os.execvp`; we replace that
with native Python implementations. The CLI layer (`cli.py`) and argparse
structure remain unchanged -- only the command handler bodies change.

## Current Architecture

```
cli.py (argparse) --> commands.py (_run / os.execvp) --> scripts/*.sh
```

## Target Architecture

```
cli.py (argparse) --> commands.py (calls into internal modules)
                          |
                          v
                src/agm/
                  ├── shell.py          # shell/env helpers
                  ├── project.py        # project detection & config loading
                  ├── git.py            # git operations (plumbum wrappers)
                  ├── worktree.py       # mkwt, rmwt, brsync
                  ├── dep.py            # pm-dep (dependency management)
                  ├── init.py           # pm-init (project initialization)
                  ├── fetch.py          # pm-fetch (fetch all repos)
                  ├── session.py        # pm.sh (open/new/checkout with env loading)
                  ├── sandbox.py        # sandbox.sh (srt wrapper)
                  ├── tmux_layout.py    # tmux-apply-layout (pure computation)
                  └── tmux_session.py   # tmux.sh (session creation)
```

## Dependencies

**Keep:**
- `plumbum>=1.9.0` -- shell command orchestration (already a dependency)

**Add: none**

We do not add `libtmux`, `GitPython`, `pygit2`, or `deepmerge`.

- **libtmux**: The tmux scripts use compound commands with `;` separators,
  `run-shell`, and custom layout strings. libtmux's API does not map well to
  these patterns and would add a heavy dependency for little gain. We use
  plumbum to call `tmux` directly, exactly as the scripts do.

- **GitPython / pygit2**: Git operations are simple command invocations (fetch,
  worktree add/remove, branch). Plumbum wrapping `git` is simpler, has fewer
  edge cases, and avoids the C library dependency of pygit2.

- **deepmerge / jsonmerge**: The sandbox settings merge is ~20 lines of custom
  logic with specific field-level semantics (not a generic deep merge). A helper
  function is clearer than pulling in a library.

- **Click / Typer**: argparse is already working and fully tested. No benefit
  to migrating the CLI framework.

## Library Usage: Plumbum

We already depend on plumbum. Key patterns we use:

```python
from plumbum import local
from plumbum.cmd import git

# Run a command and capture output
output = git["fetch", "--prune", "origin"]()

# Run in a specific directory
with local.cwd("/some/path"):
    git["status"]()

# Check if command exists
from plumbum import CommandNotFound
try:
    local["srt"]
except CommandNotFound:
    ...

# Foreground execution (inherits stdio)
from plumbum import FG
git["clone", url, path] & FG

# Piping
(git["for-each-ref", ...] | local["awk"][...])()
```

## Module Design

### 1. `shell.py` -- Shell & Environment Helpers

Provides utilities for running bash config scripts and capturing the resulting
environment. This is the key bridge between bash configuration and Python.

```python
def capture_shell_env(script_path: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Source a bash script and return the resulting environment.

    Runs: bash -c 'source <script> && env -0'
    Parses the NUL-delimited output to capture all exported variables.
    Returns a new dict with the full post-source environment.
    """

def source_env_file(path: Path, env: dict[str, str]) -> dict[str, str]:
    """Source an env file if it exists, return updated env. No-op if missing."""

def run_foreground(cmd: list[str], *, cwd: Path | None = None,
                   env: dict[str, str] | None = None) -> int:
    """Run a command in foreground (inheriting stdio), return exit code."""

def run_capture(cmd: list[str], *, cwd: Path | None = None,
                env: dict[str, str] | None = None) -> tuple[int, str, str]:
    """Run a command capturing stdout/stderr, return (returncode, stdout, stderr)."""
```

**Implementation detail for `capture_shell_env`:**
```bash
bash -c 'set -a; source "$1" && env -0' _ /path/to/env.sh
```
Parse NUL-separated output. This handles multi-line values, special characters,
and any side effects the script might have.

### 2. `project.py` -- Project Detection & Config

Replaces `project_dir()` and `default_worktrees_dir()` functions shared by
`cpconfig.sh`, `mkwt.sh`, and others. Also handles config file copying.

```python
def detect_project_dir(cwd: Path | None = None) -> Path:
    """Detect the project root directory.

    Detection logic (matches cpconfig.sh/mkwt.sh):
    1. If cwd basename is 'repo' and sibling 'worktrees/' exists -> parent
    2. If parent basename is 'worktrees' and grandparent has 'repo/' -> grandparent
    3. If cwd contains 'repo/' subdir -> cwd
    4. Fallback: cwd itself (simple project)
    """

def default_worktrees_dir(project_dir: Path) -> Path:
    """Return worktrees/ if it exists, else .worktrees/."""

CONFIG_FILES: list[str]  # .setup.sh, .env, .env.local, .config, .agents, ...

def copy_config(*, project_dir: Path | None = None,
                target: Path, cwd: Path | None = None) -> None:
    """Copy config files from cwd and project_dir/config/ to target.
    Silently ignores missing files (matches cpconfig.sh behavior).
    """
```

### 3. `git.py` -- Git Operations

Thin plumbum wrappers for git commands used across modules.

```python
def git_cmd(repo_dir: Path | None = None) -> BoundCommand:
    """Return a plumbum git command, optionally with -C repo_dir."""

def git_setup(cwd: Path | None = None) -> BoundCommand:
    """Detect git repo (direct or repo/ subdir), return bound git command.
    Matches git_setup() from mkwt.sh/rmwt.sh.
    """

def is_git_repo(path: Path) -> bool:
    """Check if path is inside a git work tree."""

def fetch(git: BoundCommand) -> None:
    """Run git fetch."""

def current_branch(git: BoundCommand) -> str:
    """Get current branch name via rev-parse --abbrev-ref HEAD."""

def worktree_add(git: BoundCommand, path: Path, branch: str,
                 create: bool = False, start_point: str | None = None) -> None:
    """Add a worktree. If create=True, use -b to create branch."""

def worktree_remove(git: BoundCommand, path: Path, force: bool = False) -> None:
    """Remove a worktree."""

def worktree_list(git: BoundCommand) -> list[WorktreeInfo]:
    """Parse git worktree list --porcelain output."""

def branch_delete(git: BoundCommand, branch: str) -> None:
    """Delete a local branch with -d."""
```

### 4. `worktree.py` -- Worktree Management (mkwt, rmwt, brsync)

Replaces `mkwt.sh`, `rmwt.sh`, `brsync.sh`.

```python
def cmd_worktree_checkout(*, new_branch: str | None, worktrees_dir: str | None,
                          branch: str | None) -> NoReturn:
    """Create/checkout a worktree (replaces mkwt.sh).

    1. git_setup() to find repo
    2. Fetch refs
    3. Resolve worktrees path (default_worktrees_dir or -d flag)
    4. git worktree add (with or without -b)
    5. copy_config() to new worktree
    6. Run setup scripts in precedence order:
       - $PROJ_DIR/config/setup.sh
       - .config/setup.sh
       - .setup.sh
    """

def cmd_worktree_remove(*, force: bool, branch: str) -> NoReturn:
    """Remove a worktree and delete its branch (replaces rmwt.sh).

    1. git_setup() to find repo
    2. Parse worktree list to find path for branch
    3. git worktree remove [--force]
    4. git branch -d
    """

def cmd_branch_sync() -> NoReturn:
    """Sync remote tracking branches (replaces brsync.sh).

    1. git fetch --prune origin
    2. For each unmerged remote branch (not origin/HEAD):
       - If local branch doesn't exist, create tracking branch
    """
```

### 5. `dep.py` -- Dependency Management

Replaces `pm-dep.sh`.

```python
def cmd_dep_new(*, branch: str | None, repo_url: str) -> NoReturn:
    """Clone a new dependency (replaces pm-dep.sh new).

    1. Derive dep name from URL
    2. Check deps/$dep doesn't exist
    3. Resolve default branch from remote if not specified
    4. Clone into deps/$dep/$branch/
    5. On failure, clean up deps/$dep/
    """

def cmd_dep_switch(*, dep: str, branch: str, create_branch: bool) -> NoReturn:
    """Switch dep to different branch (replaces pm-dep.sh switch).

    1. Check deps/$dep exists
    2. Find first git repo in deps/$dep/
    3. Check target dir doesn't exist
    4. Fetch refs
    5. If create_branch: get default branch, worktree add -b
       Else: worktree add with existing branch
    """

# Helpers (private):
def _derive_dep_name(url: str) -> str
def _default_branch_from_remote(url: str) -> str
def _default_branch_from_repo(repo_path: Path) -> str
def _first_dep_repo(dep_dir: Path) -> Path
```

### 6. `init.py` -- Project Initialization

Replaces `pm-init.sh`.

```python
def cmd_init(*, branch: str | None, positional: list[str]) -> NoReturn:
    """Initialize a new project (replaces pm-init.sh).

    1. Parse positional args (1 or 2: [project-name] repo-url)
    2. Detect URL vs project name using heuristics
    3. Derive project name from URL if not given
    4. Create directory structure: repo/, deps/, worktrees/, notes/, config/
    5. Write config/env.sh and config/setup.sh templates (if missing)
    6. Make config/setup.sh executable
    7. Clone repo if URL given and repo/ is empty
    """

def _looks_like_repo_url(value: str) -> bool
def _derive_project_name(url: str) -> str
```

### 7. `fetch.py` -- Fetch All Repos

Replaces `pm-fetch.sh`.

```python
def cmd_fetch() -> NoReturn:
    """Fetch main repo and all dependencies (replaces pm-fetch.sh).

    1. Verify $PWD/repo exists
    2. Fetch main repo
    3. If deps/ exists, for each dep subdir:
       - Find first git repo
       - Fetch it
    """
```

### 8. `session.py` -- Project Session Management

Replaces `pm.sh`. This is the module that handles environment variable loading
from bash config scripts.

```python
def cmd_open(*, pane_count: str | None, branch: str | None) -> NoReturn:
    """Open a tmux session for a branch (replaces pm.sh open).

    1. Set PROJ_DIR = CWD, derive PROJ name
    2. Determine session name and repo path
    3. Source config/env.sh (project-level)
    4. Source config/$BRANCH/env.sh (branch-level)
    5. Call tmux_session.create_session() with captured env
    """

def cmd_new(*, pane_count: str | None, parent: str | None,
            branch: str) -> NoReturn:
    """Create new branch and open session (replaces pm.sh new).

    1-4. Same env loading as open
    5. Determine parent branch
    6. cd to parent worktree
    7. Call mkwt.sh equivalent (worktree.cmd_worktree_checkout with -b)
    8. cd to new worktree
    9. Call tmux_session.create_session() with -d flag
    """

def cmd_checkout(*, pane_count: str | None, parent: str | None,
                 branch: str) -> NoReturn:
    """Checkout existing branch and open session (replaces pm.sh co).

    Same as cmd_new but calls mkwt without -b.
    """
```

**Key design for env loading:**
```python
# Source bash env files and capture the resulting environment
env = os.environ.copy()
env["PROJ_DIR"] = str(proj_dir)

env = source_env_file(proj_dir / "config" / "env.sh", env)
if branch:
    env = source_env_file(proj_dir / "config" / branch / "env.sh", env)

# Pass env to all subsequent commands
```

### 9. `sandbox.py` -- Sandbox Runtime Wrapper

Replaces `sandbox.sh`. The most complex script (320 lines).

```python
def cmd_run(*, no_patch: bool, settings_file: str | None,
            run_command: list[str]) -> NoReturn:
    """Run command in sandbox (replaces sandbox.sh).

    1. Validate srt is available
    2. Resolve settings file:
       - Explicit -f file
       - Or discover ~/.sandbox/default.json and ./.sandbox/default.json
       - Merge if both found
    3. If PROJ_DIR set and --no-patch not given: patch settings
    4. Track bwrap artifacts for cleanup
    5. Run srt --settings FILE -- COMMAND
    6. Clean up temp files and empty artifacts on exit
    """

def _merge_settings(home: dict, local: dict) -> dict:
    """Section-aware merge matching sandbox.sh merge logic.

    - 'enabled': take from local if present
    - 'network': shallow merge (local overrides home)
    - 'filesystem': shallow merge (local overrides home)
    - 'ignoreViolations': take from local if present
    - 'enableWeakerNestedSandbox': take from local if present
    """

def _patch_for_proj_dir(settings: dict, proj_dir: str) -> dict:
    """Add PROJ_DIR to filesystem.allowWrite."""

def _track_bwrap_artifacts(settings_path: Path, cwd: Path) -> list[Path]:
    """Identify filesystem artifacts that bwrap may create."""
```

### 10. `tmux_layout.py` -- Layout Calculation

Replaces `tmux-apply-layout.sh`. This is pure computation + one tmux call.

```python
def cmd_tmux_layout(*, pane_count: int, window_id: str,
                    width: int, height: int) -> NoReturn:
    """Calculate and apply tmux layout (replaces tmux-apply-layout.sh).

    1. Calculate grid: rows, cols
    2. Build layout string recursively
    3. Compute checksum
    4. exec tmux select-layout -t window_id layout
    """

def layout_checksum(layout: str) -> str:
    """16-bit rolling checksum matching tmux's format."""

def build_row_layout(width: int, height: int, x: int, y: int,
                     start_index: int, pane_total: int) -> str:
    """Build layout for a single row of panes."""

def build_window_layout(width: int, height: int, pane_total: int,
                        cols: int, rows: int) -> str:
    """Build layout for the full window (recursive)."""
```

### 11. `tmux_session.py` -- Session Creation

Replaces `tmux.sh`. Handles environment filtering, session creation with
pane splitting, and layout application.

```python
def cmd_tmux_new(*, detach: bool, pane_count: int,
                 session_name: str | None,
                 env: dict[str, str] | None = None) -> NoReturn:
    """Create tmux session with panes (replaces tmux.sh).

    1. Filter environment (skip TMUX, TERM, DISPLAY, etc.)
    2. Build tmux env args (-e NAME=VALUE for each kept var)
    3. Determine execution path:
       - If inside TMUX and not detached: create detached, then switch
       - If -d: create detached
       - Otherwise: create attached with compound command
    4. Split panes, apply layout, select pane 0
    """

SKIP_NAMES: set[str]       # TMUX, TERM, DISPLAY, PWD, ...
SKIP_PREFIXES: tuple[str, ...]  # TMUX_, TERM_, SSH_, DBUS_, XDG_

def _filter_env(env: dict[str, str]) -> list[tuple[str, str]]:
    """Filter environment variables for tmux, matching tmux.sh logic."""
```

## Changes to Existing Files

### `commands.py`

Complete rewrite. Instead of `_run()` / `os.execvp()`, each handler calls
directly into the new modules. The functions keep the same signatures so
`cli.py` dispatch remains unchanged.

```python
# Before:
def cmd_branch_sync() -> NoReturn:
    _run("brsync.sh", [])

# After:
def cmd_branch_sync() -> NoReturn:
    worktree.branch_sync()
```

Most handlers become thin wrappers that call into the new modules.

### `cli.py`

**No changes to argument parsing.** The dispatch function calls the same
`commands.*` functions with the same signatures.

One potential change: some functions currently typed as `NoReturn` may need
to become `-> int` or `-> None` if we stop using `os.execvp`. However, we
can keep the `NoReturn` + `sys.exit()` pattern for top-level commands.

### `pyproject.toml`

No new dependencies. Remove the `install` justfile target that copies scripts.

### `justfile`

- Remove `install` target (scripts no longer exist)
- Keep `install-agm` (installs via `uv tool install`)
- Keep `setup`, `test`, `typecheck`, `check`

## Migration Order

The modules have dependencies on each other. Migrate in this order:

### Phase 1: Foundation (no script dependencies)
1. **`shell.py`** -- env capture utilities (new, no deps)
2. **`project.py`** -- project detection, config copy (new, no deps)
3. **`git.py`** -- git command wrappers (new, no deps)
4. **`tmux_layout.py`** -- pure computation (self-contained)

### Phase 2: Independent commands
5. **`worktree.py`** -- depends on git.py, project.py
6. **`dep.py`** -- depends on git.py
7. **`init.py`** -- self-contained
8. **`fetch.py`** -- depends on git.py

### Phase 3: Composite commands
9. **`tmux_session.py`** -- depends on tmux_layout.py
10. **`session.py`** -- depends on shell.py, project.py, worktree.py, tmux_session.py
11. **`sandbox.py`** -- depends on shell.py (for python3 check, but now native)

### Phase 4: Integration
12. **Rewrite `commands.py`** to dispatch to new modules
13. **Update tests** -- e2e tests no longer need scripts on PATH
14. **Remove `scripts/` directory**
15. **Update `justfile`** -- remove script installation targets

## Test Strategy

### Existing tests

- **`test_cli_parsing.py`** (390 lines): No changes needed. Tests argparse
  parsing only, which is unchanged.

- **`test_e2e.py`** (~1000 lines): These tests invoke `agm` as a subprocess
  and check git/filesystem state. They should pass without changes once the
  Python implementations replace the script dispatching. The key difference:
  tests will no longer need scripts on PATH. The fake tmux mock script will
  still be needed (tmux is an external binary we invoke).

### New unit tests

Add unit tests for the new modules, especially:

- `test_shell.py`: Test `capture_shell_env()` with real bash scripts
- `test_project.py`: Test `detect_project_dir()` with various directory layouts
- `test_tmux_layout.py`: Test layout computation (pure functions, easy to test)
- `test_sandbox.py`: Test settings merge/patch logic

### Test execution

Tests call `run_agm()` which uses `subprocess.run([sys.executable, "-m", "agm.cli", ...])`.
This already doesn't depend on scripts being on PATH for the Python-implemented
commands -- only for the `os.execvp` fallthrough. Once all commands are Python-native,
no scripts need to be on PATH.

**Exception**: The fake `tmux` binary in tests must still be on PATH because
our Python code will invoke `tmux` via plumbum.

## Risk Areas & Mitigations

### 1. Environment variable propagation from bash config scripts

**Risk**: `pm.sh` sources `config/env.sh` which can run arbitrary bash.
Our `capture_shell_env()` must faithfully capture the resulting environment.

**Mitigation**: Use `bash -c 'source "$1" && env -0'` with NUL-separated output.
This handles multi-line values and special characters. Test with edge cases.

### 2. Working directory semantics

**Risk**: Several scripts `cd` into directories and the cwd affects subsequent
operations (e.g., `mkwt.sh` cds into the new worktree to run setup scripts).

**Mitigation**: Use `plumbum.local.cwd()` context manager or pass explicit
`cwd` to subprocess calls. Never change the process's actual cwd.

Actually, `plumbum.local.cwd()` does change the process cwd. For isolation
we should use the `cwd` parameter on command execution instead:
```python
git_cmd = local["git"]
git_cmd["-C", str(path), "fetch"]()
```

### 3. Tmux compound commands

**Risk**: `tmux.sh` builds a single tmux invocation with `;` separators for
the non-detached path. This is zsh-specific behavior.

**Mitigation**: Use plumbum to call tmux with the same argument structure.
The `;` separators are tmux's command separator, not shell's -- they're passed
as literal arguments. This works identically from Python.

### 4. Error handling and exit codes

**Risk**: Scripts use `set -euo pipefail` for fail-fast. Python doesn't have
this, and plumbum raises exceptions on non-zero exits by default.

**Mitigation**: Let plumbum's `ProcessExecutionError` propagate for unexpected
failures. For expected non-zero exits (e.g., `git show-ref --verify` checking
if a branch exists), use `retcode=None` or try/except.

### 5. The sandbox.sh bwrap artifact tracking

**Risk**: Complex embedded Python in sandbox.sh for tracking filesystem
artifacts. Moving this to native Python is straightforward but must match
the exact semantics.

**Mitigation**: The Python code is already written inline in the bash script.
We literally extract it into a proper Python function.

## File Size Estimates

| Module | Est. Lines | Replaces |
|---|---|---|
| `shell.py` | ~60 | (new) |
| `project.py` | ~80 | cpconfig.sh (49 lines) |
| `git.py` | ~100 | shared git helpers |
| `tmux_layout.py` | ~90 | tmux-apply-layout.sh (122 lines) |
| `worktree.py` | ~120 | mkwt.sh (106) + rmwt.sh (57) + brsync.sh (20) |
| `dep.py` | ~130 | pm-dep.sh (174 lines) |
| `init.py` | ~80 | pm-init.sh (102 lines) |
| `fetch.py` | ~60 | pm-fetch.sh (51 lines) |
| `tmux_session.py` | ~120 | tmux.sh (185 lines) |
| `session.py` | ~100 | pm.sh (116 lines) |
| `sandbox.py` | ~200 | sandbox.sh (320 lines) |
| `commands.py` (rewrite) | ~120 | commands.py (265 lines) |
| **Total new Python** | **~1260** | **Replaces ~1500 lines bash/zsh** |
