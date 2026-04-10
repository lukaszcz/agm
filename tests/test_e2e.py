"""End-to-end tests that exercise commands through the ``agm`` CLI.

These tests create real git repos, invoke the ``agm`` command-line interface,
and verify the resulting filesystem and git state — going beyond the dispatch
tests that only check argument passing.

Test setup uses only raw git/filesystem operations, never other scripts from
this repository.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

HAS_ZSH = shutil.which("zsh") is not None
needs_zsh = pytest.mark.skipif(not HAS_ZSH, reason="zsh is required")

# A fake ``tmux`` binary used by tests for pm.sh, tmux.sh and
# tmux-apply-layout.sh.  It logs every invocation and returns canned
# responses so that the calling zsh scripts can complete without a real
# tmux server.
_FAKE_TMUX_SCRIPT = r"""#!/bin/bash
log="${TMUX_LOG:?TMUX_LOG must be set}"
echo "CMD: $*" >> "$log"

case "$1" in
  new-session)
    # If -P flag is present (detached-print mode) output session name.
    has_P=false
    session_name="0"
    prev=""
    for arg in "$@"; do
      case "$arg" in -dP|-Pd|-P) has_P=true ;; esac
      [[ "$prev" == "-s" ]] && session_name="$arg"
      prev="$arg"
    done
    if $has_P; then echo "$session_name"; fi
    ;;
  display-message)
    for arg in "$@"; do
      case "$arg" in
        *window_id*)     echo "@0"; break ;;
        *window_width*)  echo "200"; break ;;
        *window_height*) echo "50"; break ;;
      esac
    done
    ;;
esac
exit 0
"""


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def env(tmp_path: Path) -> dict[str, str]:
    """Environment dict with scripts on PATH, git identity, and isolated HOME."""
    e = os.environ.copy()
    e["PATH"] = str(SCRIPTS_DIR) + ":" + e.get("PATH", "")
    e["GIT_AUTHOR_NAME"] = "Test"
    e["GIT_AUTHOR_EMAIL"] = "test@test.com"
    e["GIT_COMMITTER_NAME"] = "Test"
    e["GIT_COMMITTER_EMAIL"] = "test@test.com"
    e["GIT_CONFIG_NOSYSTEM"] = "1"
    e.pop("PROJ_DIR", None)
    e.pop("TMUX", None)
    e.pop("TMUX_PANE", None)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    e["HOME"] = str(fake_home)
    return e


def run_agm(
    args: list[str],
    *,
    env: dict[str, str],
    cwd: str | Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run an ``agm`` CLI command as a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "agm.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        check=check,
    )


def _git(
    *args: str,
    cwd: str | Path,
    env: dict[str, str],
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a git command."""
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        check=check,
    )


def make_bare_repo(path: Path, env: dict[str, str]) -> Path:
    """Create a bare git repo with an initial commit on *main*."""
    tmp = path.parent / f"_tmpinit_{path.name}"
    tmp.mkdir(parents=True)
    _git("init", "-b", "main", cwd=str(tmp), env=env)
    (tmp / "README.md").write_text("initial\n")
    _git("add", "README.md", cwd=str(tmp), env=env)
    _git("commit", "-m", "initial commit", cwd=str(tmp), env=env)
    path.mkdir(parents=True, exist_ok=True)
    _git("clone", "--bare", str(tmp), str(path), cwd=str(path.parent), env=env)
    shutil.rmtree(tmp)
    return path


def make_working_repo(path: Path, bare: Path, env: dict[str, str]) -> Path:
    """Clone *bare* into *path*."""
    _git("clone", str(bare), str(path), cwd=str(path.parent), env=env)
    return path


def _push_branch(
    work: Path,
    bare: Path,
    branch: str,
    filename: str,
    env: dict[str, str],
) -> None:
    """Create a branch with a file, push it, then go back to main."""
    _git("checkout", "-b", branch, cwd=str(work), env=env)
    (work / filename).write_text(filename)
    _git("add", ".", cwd=str(work), env=env)
    _git("commit", "-m", f"add {filename}", cwd=str(work), env=env)
    _git("push", "-u", "origin", branch, cwd=str(work), env=env)
    _git("checkout", "main", cwd=str(work), env=env)


def _install_fake_tmux(bin_dir: Path, log_path: Path, env: dict[str, str]) -> None:
    """Put a fake ``tmux`` on *PATH* that logs to *log_path*."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake = bin_dir / "tmux"
    fake.write_text(_FAKE_TMUX_SCRIPT)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env["PATH"] = str(bin_dir) + ":" + env["PATH"]
    env["TMUX_LOG"] = str(log_path)


def _make_project(
    tmp_path: Path,
    bare: Path,
    env: dict[str, str],
    name: str = "proj",
) -> Path:
    """Create a standard agm project layout with repo/ cloned from *bare*."""
    project = tmp_path / name
    project.mkdir()
    for d in ("worktrees", "deps", "config", "notes"):
        (project / d).mkdir()
    make_working_repo(project / "repo", bare, env)
    return project


# ── agm br sync ─────────────────────────────────────────────────────────────


class TestBranchSync:
    """agm br sync: sync remote tracking branches."""

    def test_creates_tracking_branches_for_unmerged_remote(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        _push_branch(work, bare, "feat/a", "a.txt", env)
        _git("branch", "-D", "feat/a", cwd=str(work), env=env)

        run_agm(["br", "sync"], env=env, cwd=str(work))

        branches = _git("branch", cwd=str(work), env=env).stdout
        assert "feat/a" in branches

    def test_skips_already_existing_local_branches(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        _push_branch(work, bare, "feat/b", "b.txt", env)
        # Local branch still exists — brsync must not fail.
        run_agm(["br", "sync"], env=env, cwd=str(work))

    def test_ignores_merged_branches(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        _git("checkout", "-b", "merged-feat", cwd=str(work), env=env)
        (work / "m.txt").write_text("m")
        _git("add", ".", cwd=str(work), env=env)
        _git("commit", "-m", "merged feat", cwd=str(work), env=env)
        _git("checkout", "main", cwd=str(work), env=env)
        _git("merge", "merged-feat", cwd=str(work), env=env)
        _git("push", cwd=str(work), env=env)
        _git("push", "origin", "merged-feat", cwd=str(work), env=env)
        _git("branch", "-d", "merged-feat", cwd=str(work), env=env)

        run_agm(["br", "sync"], env=env, cwd=str(work))

        branches = _git("branch", cwd=str(work), env=env).stdout
        assert "merged-feat" not in branches

    def test_creates_multiple_tracking_branches(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        for name in ("feat/x", "feat/y"):
            _push_branch(work, bare, name, f"{name.split('/')[-1]}.txt", env)
            _git("branch", "-D", name, cwd=str(work), env=env)

        run_agm(["br", "sync"], env=env, cwd=str(work))

        branches = _git("branch", cwd=str(work), env=env).stdout
        assert "feat/x" in branches
        assert "feat/y" in branches


# ── agm config cp ───────────────────────────────────────────────────────────


class TestCpConfig:
    """agm config cp: copy configuration files."""

    def test_copies_files_from_current_dir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"
        dest.mkdir()

        (src / ".env").write_text("KEY=val")
        (src / ".claude").mkdir()
        (src / ".claude" / "settings.json").write_text("{}")

        run_agm(["config", "cp", str(dest)], env=env, cwd=str(src))

        assert (dest / ".env").read_text() == "KEY=val"
        assert (dest / ".claude" / "settings.json").read_text() == "{}"

    def test_copies_files_from_project_config_dir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "repo").mkdir()
        (project / "worktrees").mkdir()
        config = project / "config"
        config.mkdir()
        (config / ".env").write_text("FROM_CONFIG=1")

        dest = tmp_path / "dest"
        dest.mkdir()

        # Run from project/repo → project_dir() returns project/
        run_agm(["config", "cp", str(dest)], env=env, cwd=str(project / "repo"))

        assert (dest / ".env").read_text() == "FROM_CONFIG=1"

    def test_d_option_overrides_project_dir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        custom = tmp_path / "custom"
        custom.mkdir()
        (custom / "config").mkdir()
        (custom / "config" / ".env").write_text("CUSTOM=1")

        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"
        dest.mkdir()

        run_agm(
            ["config", "cp", "-d", str(custom), str(dest)], env=env, cwd=str(src)
        )

        assert (dest / ".env").read_text() == "CUSTOM=1"

    def test_auto_detects_project_from_worktrees_subdir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "repo").mkdir()
        (project / "worktrees").mkdir()
        config = project / "config"
        config.mkdir()
        (config / ".env").write_text("FROM_WT=1")

        wt = project / "worktrees" / "feat-x"
        wt.mkdir()
        dest = tmp_path / "dest"
        dest.mkdir()

        run_agm(["config", "cp", str(dest)], env=env, cwd=str(wt))

        assert (dest / ".env").read_text() == "FROM_WT=1"

    def test_auto_detects_project_from_project_root(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "repo").mkdir()
        config = project / "config"
        config.mkdir()
        (config / ".env").write_text("FROM_ROOT=1")

        dest = tmp_path / "dest"
        dest.mkdir()

        run_agm(["config", "cp", str(dest)], env=env, cwd=str(project))

        assert (dest / ".env").read_text() == "FROM_ROOT=1"

    def test_missing_files_are_silently_ignored(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"
        dest.mkdir()

        result = run_agm(["config", "cp", str(dest)], env=env, cwd=str(src))
        assert result.returncode == 0

    def test_requires_dirname_argument(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()

        result = run_agm(["config", "cp"], env=env, cwd=str(src), check=False)
        assert result.returncode != 0
        assert "usage" in result.stderr.lower()

    def test_relative_dirname_resolved_to_cwd(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "target").mkdir()
        (src / ".env").write_text("REL=1")

        run_agm(["config", "cp", "target"], env=env, cwd=str(src))

        assert (src / "target" / ".env").read_text() == "REL=1"

    def test_copies_all_recognized_file_types(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"
        dest.mkdir()

        files = [".setup.sh", ".env", ".env.local", ".mcp.json",
                 ".agents", ".opencode", ".codex", ".pi"]
        for f in files:
            (src / f).write_text(f"content of {f}")

        run_agm(["config", "cp", str(dest)], env=env, cwd=str(src))

        for f in files:
            assert (dest / f).read_text() == f"content of {f}"


# ── agm wt co / wt new ──────────────────────────────────────────────────────


class TestMkWt:
    """agm wt co / wt new: create / checkout git worktrees."""

    def test_checkout_existing_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        _push_branch(work, bare, "feat/x", "x.txt", env)
        _git("branch", "-D", "feat/x", cwd=str(work), env=env)

        wt_dir = tmp_path / "worktrees"
        wt_dir.mkdir()

        run_agm(["wt", "co", "-d", str(wt_dir), "feat/x"], env=env, cwd=str(work))

        assert (wt_dir / "feat/x").is_dir()
        assert (wt_dir / "feat/x" / "x.txt").read_text() == "x.txt"
        # Verify the worktree is on the correct branch.
        head = _git(
            "rev-parse", "--abbrev-ref", "HEAD",
            cwd=str(wt_dir / "feat/x"), env=env,
        ).stdout.strip()
        assert head == "feat/x"

    def test_create_new_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        wt_dir = tmp_path / "worktrees"
        wt_dir.mkdir()

        run_agm(
            ["wt", "new", "-d", str(wt_dir), "new-branch"],
            env=env, cwd=str(work),
        )

        assert (wt_dir / "new-branch").is_dir()
        branches = _git("branch", cwd=str(work), env=env).stdout
        assert "new-branch" in branches
        # Verify the worktree is on the new branch.
        head = _git(
            "rev-parse", "--abbrev-ref", "HEAD",
            cwd=str(wt_dir / "new-branch"), env=env,
        ).stdout.strip()
        assert head == "new-branch"

    def test_new_branch_contains_same_files(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        wt_dir = tmp_path / "worktrees"
        wt_dir.mkdir()

        run_agm(
            ["wt", "new", "-d", str(wt_dir), "copy-branch"],
            env=env, cwd=str(work),
        )

        assert (wt_dir / "copy-branch" / "README.md").read_text() == "initial\n"

    def test_custom_worktrees_dir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        custom = tmp_path / "my-worktrees"
        custom.mkdir()

        run_agm(
            ["wt", "new", "-d", str(custom), "test-branch"],
            env=env, cwd=str(work),
        )

        assert (custom / "test-branch").is_dir()

    def test_copies_config_files_to_worktree(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)
        (project / "config" / ".env").write_text("PROJ=1")

        run_agm(["wt", "new", "with-config"], env=env, cwd=str(project / "repo"))

        wt_path = project / "worktrees" / "with-config"
        assert wt_path.is_dir()
        assert (wt_path / ".env").read_text() == "PROJ=1"

    def test_runs_project_setup_script(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        setup = project / "config" / "setup.sh"
        setup.write_text('#!/bin/bash\ntouch "$PWD/.setup-ran"\n')
        setup.chmod(setup.stat().st_mode | stat.S_IEXEC)

        run_agm(["wt", "new", "setup-test"], env=env, cwd=str(project / "repo"))

        assert (project / "worktrees" / "setup-test" / ".setup-ran").exists()

    def test_runs_dot_setup_sh(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """A .setup.sh copied into the worktree should be executed."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        dot_setup = project / "config" / ".setup.sh"
        dot_setup.write_text('#!/bin/bash\ntouch "$PWD/.dot-setup-ran"\n')
        dot_setup.chmod(dot_setup.stat().st_mode | stat.S_IEXEC)

        run_agm(["wt", "new", "dotsetup"], env=env, cwd=str(project / "repo"))

        assert (project / "worktrees" / "dotsetup" / ".dot-setup-ran").exists()

    def test_runs_dotconfig_setup_sh(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """A .config/setup.sh copied into the worktree should be executed."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        # cpconfig copies .config/ directory, then the setup script is run.
        config_dir = project / "config" / ".config"
        config_dir.mkdir(parents=True)
        setup = config_dir / "setup.sh"
        setup.write_text('#!/bin/bash\ntouch "$PWD/.dotconfig-setup-ran"\n')
        setup.chmod(setup.stat().st_mode | stat.S_IEXEC)

        run_agm(["wt", "new", "dc-setup"], env=env, cwd=str(project / "repo"))

        assert (project / "worktrees" / "dc-setup" / ".dotconfig-setup-ran").exists()

    def test_relative_worktrees_dir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """A relative -d path should be resolved against CWD."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)
        (work / "relwt").mkdir()

        run_agm(
            ["wt", "new", "-d", "relwt", "rel-branch"],
            env=env, cwd=str(work),
        )

        assert (work / "relwt" / "rel-branch").is_dir()

    def test_error_when_not_git_repo(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        not_git = tmp_path / "not-git"
        not_git.mkdir()

        result = run_agm(
            ["wt", "new", "test"], env=env, cwd=str(not_git), check=False,
        )
        assert result.returncode != 0

    def test_from_project_root_with_repo_subdir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """git_setup() should find repo/ subdirectory automatically."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        wt_dir = tmp_path / "custom-wt"
        wt_dir.mkdir()

        run_agm(
            ["wt", "new", "-d", str(wt_dir), "from-root"],
            env=env, cwd=str(project),
        )

        assert (wt_dir / "from-root").is_dir()

    def test_default_worktrees_dir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """Without -d, uses <project>/worktrees/ if it exists."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        run_agm(["wt", "new", "auto-dir"], env=env, cwd=str(project / "repo"))

        assert (project / "worktrees" / "auto-dir").is_dir()

    def test_checkout_without_b_or_branch_fails(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        result = run_agm(["wt", "co"], env=env, cwd=str(work), check=False)
        assert result.returncode != 0
        assert "usage" in result.stdout.lower()


# ── agm wt rm ───────────────────────────────────────────────────────────────


class TestRmWt:
    """agm wt rm: remove git worktrees."""

    def test_removes_worktree_and_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        wt_dir = tmp_path / "worktrees"
        wt_dir.mkdir()
        _git(
            "worktree", "add", "-b", "to-remove",
            str(wt_dir / "to-remove"),
            cwd=str(work), env=env,
        )
        assert (wt_dir / "to-remove").is_dir()

        run_agm(["wt", "rm", "to-remove"], env=env, cwd=str(work))

        assert not (wt_dir / "to-remove").exists()
        branches = _git("branch", cwd=str(work), env=env).stdout
        assert "to-remove" not in branches

    def test_force_removes_dirty_worktree(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        wt_dir = tmp_path / "worktrees"
        wt_dir.mkdir()
        _git(
            "worktree", "add", "-b", "dirty-branch",
            str(wt_dir / "dirty-branch"),
            cwd=str(work), env=env,
        )

        # Make worktree dirty with staged changes.
        (wt_dir / "dirty-branch" / "dirty.txt").write_text("uncommitted")
        _git("add", "dirty.txt", cwd=str(wt_dir / "dirty-branch"), env=env)

        # Normal removal should fail.
        result = run_agm(
            ["wt", "rm", "dirty-branch"], env=env, cwd=str(work), check=False,
        )
        assert result.returncode != 0

        # Force removal should succeed and delete the branch.
        run_agm(["wt", "rm", "-f", "dirty-branch"], env=env, cwd=str(work))
        assert not (wt_dir / "dirty-branch").exists()
        r = _git(
            "rev-parse", "--verify", "dirty-branch",
            cwd=str(work), env=env, check=False,
        )
        assert r.returncode != 0, "branch should be deleted"

    def test_error_for_nonexistent_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        result = run_agm(
            ["wt", "rm", "no-such-branch"], env=env, cwd=str(work), check=False,
        )
        assert result.returncode != 0

    def test_from_project_root_with_repo_subdir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """agm wt rm should find repo/ when run from the project root."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        wt_dir = tmp_path / "wts"
        wt_dir.mkdir()
        _git(
            "worktree", "add", "-b", "removable",
            str(wt_dir / "removable"),
            cwd=str(project / "repo"), env=env,
        )

        run_agm(["wt", "rm", "removable"], env=env, cwd=str(project))

        assert not (wt_dir / "removable").exists()
        branches = _git("branch", cwd=str(project / "repo"), env=env).stdout
        assert "removable" not in branches

    def test_requires_branch_argument(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        result = run_agm(["wt", "rm"], env=env, cwd=str(work), check=False)
        assert result.returncode != 0
        assert "usage" in result.stderr.lower()


# ── agm dep new ─────────────────────────────────────────────────────────────


class TestDepNew:
    """agm dep new: clone dependencies."""

    def test_clones_dependency(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "mylib.git", env)
        project = tmp_path / "proj"
        project.mkdir()
        (project / "deps").mkdir()

        run_agm(["dep", "new", str(bare)], env=env, cwd=str(project))

        dep = project / "deps" / "mylib"
        assert dep.is_dir()
        assert (dep / "main").is_dir()
        assert (dep / "main" / "README.md").exists()

    def test_clones_with_specific_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "lib.git", env)

        clone = tmp_path / "tmp-clone"
        _git("clone", str(bare), str(clone), cwd=str(tmp_path), env=env)
        _push_branch(clone, bare, "v2", "v2.txt", env)

        project = tmp_path / "proj"
        project.mkdir()
        (project / "deps").mkdir()

        run_agm(
            ["dep", "new", "-b", "v2", str(bare)],
            env=env, cwd=str(project),
        )

        assert (project / "deps" / "lib" / "v2" / "v2.txt").exists()

    def test_error_when_dep_already_exists(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "mylib.git", env)
        project = tmp_path / "proj"
        project.mkdir()
        deps = project / "deps"
        deps.mkdir()
        (deps / "mylib").mkdir()

        result = run_agm(
            ["dep", "new", str(bare)],
            env=env, cwd=str(project), check=False,
        )
        assert result.returncode != 0
        assert "already exists" in result.stderr

    def test_derives_name_from_url(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "some-lib.git", env)
        project = tmp_path / "proj"
        project.mkdir()
        (project / "deps").mkdir()

        run_agm(["dep", "new", str(bare)], env=env, cwd=str(project))

        assert (project / "deps" / "some-lib").is_dir()

    def test_no_subcommand_shows_usage(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        result = run_agm(
            ["dep"], env=env, cwd=str(tmp_path), check=False,
        )
        assert result.returncode != 0
        assert "usage" in result.stderr.lower()


# ── agm dep switch ──────────────────────────────────────────────────────────


class TestDepSwitch:
    """agm dep switch: switch dependency branches."""

    @staticmethod
    def _setup_dep(
        tmp_path: Path, bare: Path, env: dict[str, str],
    ) -> Path:
        """Create deps/mylib/main/ by cloning *bare* directly (no agm)."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "deps").mkdir()
        dep_main = project / "deps" / "mylib" / "main"
        dep_main.mkdir(parents=True)
        _git(
            "clone", "--branch", "main", str(bare), str(dep_main),
            cwd=str(tmp_path), env=env,
        )
        return project

    def test_switch_to_existing_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "mylib.git", env)

        # Push a feature branch.
        clone = tmp_path / "tmp-clone"
        _git("clone", str(bare), str(clone), cwd=str(tmp_path), env=env)
        _push_branch(clone, bare, "feat/new", "new.txt", env)

        project = self._setup_dep(tmp_path, bare, env)

        run_agm(
            ["dep", "switch", "mylib", "feat/new"],
            env=env, cwd=str(project),
        )

        switched_wt = project / "deps" / "mylib" / "feat/new"
        assert (switched_wt / "new.txt").exists()
        # Verify the worktree is on the correct branch.
        head = _git(
            "rev-parse", "--abbrev-ref", "HEAD",
            cwd=str(switched_wt), env=env,
        ).stdout.strip()
        assert head == "feat/new"
        # Verify it's linked as a worktree of the original clone.
        wt_list = _git(
            "worktree", "list", "--porcelain",
            cwd=str(project / "deps" / "mylib" / "main"), env=env,
        ).stdout
        assert str(switched_wt) in wt_list

    def test_switch_create_new_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "mylib.git", env)
        project = self._setup_dep(tmp_path, bare, env)

        run_agm(
            ["dep", "switch", "-b", "mylib", "my-new-branch"],
            env=env, cwd=str(project),
        )

        new_wt = project / "deps" / "mylib" / "my-new-branch"
        assert new_wt.is_dir()
        assert (new_wt / "README.md").exists()
        # Verify the new branch exists and is checked out.
        head = _git(
            "rev-parse", "--abbrev-ref", "HEAD",
            cwd=str(new_wt), env=env,
        ).stdout.strip()
        assert head == "my-new-branch"

    def test_switch_error_when_dep_missing(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "deps").mkdir()

        result = run_agm(
            ["dep", "switch", "nonexistent", "main"],
            env=env, cwd=str(project), check=False,
        )
        assert result.returncode != 0
        assert "does not exist" in result.stderr

    def test_switch_error_when_target_exists(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "mylib.git", env)
        project = self._setup_dep(tmp_path, bare, env)

        # Switching to "main" fails because deps/mylib/main/ already exists.
        result = run_agm(
            ["dep", "switch", "mylib", "main"],
            env=env, cwd=str(project), check=False,
        )
        assert result.returncode != 0
        assert "already exists" in result.stderr


# ── agm fetch ───────────────────────────────────────────────────────────────


class TestFetch:
    """agm fetch: fetch repo and dependencies."""

    def test_fetches_main_repo(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        result = run_agm(["fetch"], env=env, cwd=str(project))
        assert result.returncode == 0
        assert "Fetching repo" in result.stdout

    def test_fetches_dependencies(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare_main = make_bare_repo(tmp_path / "main.git", env)
        bare_dep = make_bare_repo(tmp_path / "dep.git", env)

        project = _make_project(tmp_path, bare_main, env)

        dep_wt = project / "deps" / "mylib" / "main"
        dep_wt.mkdir(parents=True)
        make_working_repo(dep_wt, bare_dep, env)

        result = run_agm(["fetch"], env=env, cwd=str(project))
        assert result.returncode == 0
        assert "Fetching repo" in result.stdout
        assert "Fetching deps/" in result.stdout

    def test_no_deps_dir_ok(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = tmp_path / "proj"
        project.mkdir()
        make_working_repo(project / "repo", bare, env)

        result = run_agm(["fetch"], env=env, cwd=str(project))
        assert result.returncode == 0

    def test_error_when_repo_missing(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = tmp_path / "proj"
        project.mkdir()

        result = run_agm(
            ["fetch"], env=env, cwd=str(project), check=False,
        )
        assert result.returncode != 0
        assert "error" in result.stderr.lower()

    def test_picks_up_new_remote_commits(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        # Push a new commit from a separate clone.
        other = make_working_repo(tmp_path / "other", bare, env)
        (other / "new.txt").write_text("new")
        _git("add", ".", cwd=str(other), env=env)
        _git("commit", "-m", "new commit", cwd=str(other), env=env)
        _git("push", cwd=str(other), env=env)

        run_agm(["fetch"], env=env, cwd=str(project))

        log = _git(
            "log", "--oneline", "origin/main", cwd=str(project / "repo"), env=env,
        ).stdout
        assert "new commit" in log


# ── agm init ────────────────────────────────────────────────────────────────


class TestInit:
    """agm init: initialise new projects."""

    def test_init_with_url_only(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "myproject.git", env)

        run_agm(["init", str(bare)], env=env, cwd=str(tmp_path))

        proj = tmp_path / "myproject"
        assert proj.is_dir()
        assert (proj / "repo" / "README.md").exists()

    def test_init_with_name_and_url(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "repo.git", env)

        run_agm(
            ["init", "custom-name", str(bare)],
            env=env, cwd=str(tmp_path),
        )

        assert (tmp_path / "custom-name" / "repo" / "README.md").exists()

    def test_creates_directory_structure(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "proj.git", env)

        run_agm(["init", str(bare)], env=env, cwd=str(tmp_path))

        proj = tmp_path / "proj"
        for d in ("repo", "deps", "worktrees", "notes", "config"):
            assert (proj / d).is_dir(), f"{d}/ should exist"

    def test_creates_config_templates(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "proj.git", env)

        run_agm(["init", str(bare)], env=env, cwd=str(tmp_path))

        proj = tmp_path / "proj"
        assert (proj / "config" / "env.sh").exists()
        assert (proj / "config" / "setup.sh").exists()
        assert os.access(proj / "config" / "setup.sh", os.X_OK)

    def test_init_with_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "proj.git", env)

        clone = tmp_path / "tmp-clone"
        _git("clone", str(bare), str(clone), cwd=str(tmp_path), env=env)
        _push_branch(clone, bare, "dev", "dev.txt", env)

        run_agm(
            ["init", "-b", "dev", str(bare)],
            env=env, cwd=str(tmp_path),
        )

        proj = tmp_path / "proj"
        assert (proj / "repo" / "dev.txt").exists()
        head = _git(
            "rev-parse", "--abbrev-ref", "HEAD",
            cwd=str(proj / "repo"), env=env,
        ).stdout.strip()
        assert head == "dev"

    def test_error_when_repo_not_empty(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "proj.git", env)

        proj = tmp_path / "proj"
        (proj / "repo").mkdir(parents=True)
        (proj / "repo" / "existing.txt").write_text("exists")

        result = run_agm(
            ["init", str(bare)], env=env, cwd=str(tmp_path), check=False,
        )
        assert result.returncode != 0
        assert "already exists" in result.stderr

    def test_init_with_nonexistent_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "proj.git", env)

        result = run_agm(
            ["init", "-b", "no-such-branch", str(bare)],
            env=env, cwd=str(tmp_path), check=False,
        )
        assert result.returncode != 0

    def test_init_without_url_creates_structure_only(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        run_agm(["init", "myproj"], env=env, cwd=str(tmp_path))

        proj = tmp_path / "myproj"
        for d in ("repo", "deps", "worktrees", "notes", "config"):
            assert (proj / d).is_dir()
        assert not list((proj / "repo").iterdir())

    def test_error_no_args(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        result = run_agm(
            ["init"], env=env, cwd=str(tmp_path), check=False,
        )
        assert result.returncode != 0
        assert "usage" in result.stderr.lower()

    def test_config_templates_are_not_overwritten(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        run_agm(["init", "myproj"], env=env, cwd=str(tmp_path))

        proj = tmp_path / "myproj"
        (proj / "config" / "env.sh").write_text("CUSTOM=1")

        run_agm(["init", "myproj"], env=env, cwd=str(tmp_path))

        assert (proj / "config" / "env.sh").read_text() == "CUSTOM=1"


# ── agm run ─────────────────────────────────────────────────────────────────


class TestSandbox:
    """agm run: sandbox execution."""

    @staticmethod
    def _make_fake_srt(directory: Path, env: dict[str, str]) -> None:
        """Create a fake ``srt`` binary that dumps the settings file contents."""
        directory.mkdir(parents=True, exist_ok=True)
        srt = directory / "srt"
        srt.write_text(
            "#!/bin/bash\n"
            'while [[ $# -gt 0 ]]; do\n'
            '  case "$1" in\n'
            "    --settings)\n"
            "      shift\n"
            '      if [[ -f "$1" ]]; then\n'
            '        echo "SETTINGS:$(cat "$1")"\n'
            "      fi\n"
            "      shift\n"
            "      ;;\n"
            "    --)\n"
            "      shift\n"
            "      break\n"
            "      ;;\n"
            "    *) shift ;;\n"
            "  esac\n"
            "done\n"
            "exit 0\n"
        )
        srt.chmod(srt.stat().st_mode | stat.S_IEXEC)
        env["PATH"] = str(directory) + ":" + env["PATH"]

    def test_error_when_srt_missing(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        # Build a restricted PATH that has bash and scripts but not srt.
        bin_dir = tmp_path / "nosrt"
        bin_dir.mkdir()
        bash_path = shutil.which("bash")
        assert bash_path, "bash is required"
        os.symlink(bash_path, bin_dir / "bash")
        env["PATH"] = str(SCRIPTS_DIR) + ":" + str(bin_dir)

        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(
            ["run", "echo", "hi"], env=env, cwd=str(work), check=False,
        )
        assert result.returncode != 0
        assert "srt is not installed" in result.stderr

    def test_error_when_command_missing(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        result = run_agm(
            ["run"], env=env, cwd=str(tmp_path), check=False,
        )
        assert result.returncode != 0
        assert "command is required" in result.stderr.lower()

    def test_error_when_no_settings_file(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)
        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(
            ["run", "echo", "hi"], env=env, cwd=str(work), check=False,
        )
        assert result.returncode != 0
        assert "no sandbox settings file found" in result.stderr.lower()

    def test_uses_home_settings(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        home = Path(env["HOME"])
        sandbox_dir = home / ".sandbox"
        sandbox_dir.mkdir(parents=True)
        settings = {"network": {"allowedDomains": ["example.com"]}}
        (sandbox_dir / "default.json").write_text(json.dumps(settings))

        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        assert result.returncode == 0

    def test_uses_local_settings(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        work = tmp_path / "work"
        work.mkdir()
        sandbox_dir = work / ".sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "default.json").write_text(
            json.dumps({"filesystem": {"allowWrite": ["."]}})
        )

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        assert result.returncode == 0

    def test_merges_home_and_local_settings(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        home = Path(env["HOME"])
        home_sandbox = home / ".sandbox"
        home_sandbox.mkdir(parents=True)
        (home_sandbox / "default.json").write_text(
            json.dumps({
                "network": {"allowedDomains": ["home.com"]},
                "filesystem": {"allowWrite": ["/home"]},
            })
        )

        work = tmp_path / "work"
        work.mkdir()
        local_sandbox = work / ".sandbox"
        local_sandbox.mkdir()
        (local_sandbox / "default.json").write_text(
            json.dumps({"network": {"allowedDomains": ["local.com"]}})
        )

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        assert result.returncode == 0

        merged = json.loads(result.stdout.split("SETTINGS:")[1].strip())
        assert merged["network"]["allowedDomains"] == ["local.com"]
        assert merged["filesystem"]["allowWrite"] == ["/home"]

    def test_merge_overrides_ignore_violations(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """ignoreViolations in local should completely replace home's."""
        self._make_fake_srt(tmp_path / "bin", env)

        home = Path(env["HOME"])
        home_sandbox = home / ".sandbox"
        home_sandbox.mkdir(parents=True)
        (home_sandbox / "default.json").write_text(json.dumps({
            "ignoreViolations": {"ruleA": True},
            "network": {"allowedDomains": []},
        }))

        work = tmp_path / "work"
        work.mkdir()
        local_sandbox = work / ".sandbox"
        local_sandbox.mkdir()
        (local_sandbox / "default.json").write_text(json.dumps({
            "ignoreViolations": {"ruleB": True},
        }))

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        merged = json.loads(result.stdout.split("SETTINGS:")[1].strip())
        # Local ignoreViolations replaces (not merges with) home.
        assert merged["ignoreViolations"] == {"ruleB": True}

    def test_merge_overrides_enabled_and_weaker_sandbox(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """enabled and enableWeakerNestedSandbox can be overridden by local."""
        self._make_fake_srt(tmp_path / "bin", env)

        home = Path(env["HOME"])
        home_sandbox = home / ".sandbox"
        home_sandbox.mkdir(parents=True)
        (home_sandbox / "default.json").write_text(json.dumps({
            "enabled": True,
            "enableWeakerNestedSandbox": False,
            "network": {"allowedDomains": []},
        }))

        work = tmp_path / "work"
        work.mkdir()
        local_sandbox = work / ".sandbox"
        local_sandbox.mkdir()
        (local_sandbox / "default.json").write_text(json.dumps({
            "enabled": False,
            "enableWeakerNestedSandbox": True,
        }))

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        merged = json.loads(result.stdout.split("SETTINGS:")[1].strip())
        assert merged["enabled"] is False
        assert merged["enableWeakerNestedSandbox"] is True

    def test_explicit_settings_file(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        work = tmp_path / "work"
        work.mkdir()
        sf = work / "custom.json"
        sf.write_text(json.dumps({"network": {"allowedDomains": ["custom.com"]}}))

        result = run_agm(
            ["run", "-f", str(sf), "echo", "hi"], env=env, cwd=str(work),
        )
        assert result.returncode == 0
        merged = json.loads(result.stdout.split("SETTINGS:")[1].strip())
        assert merged["network"]["allowedDomains"] == ["custom.com"]

    def test_explicit_settings_file_not_found(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        result = run_agm(
            ["run", "-f", "/nonexistent.json", "echo"],
            env=env, cwd=str(tmp_path), check=False,
        )
        assert result.returncode != 0
        assert "not found" in result.stderr.lower()

    def test_proj_dir_patching(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        work = tmp_path / "work"
        work.mkdir()
        sandbox_dir = work / ".sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "default.json").write_text(
            json.dumps({"filesystem": {"allowWrite": ["."]}})
        )

        env["PROJ_DIR"] = "/some/project/dir"

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        assert result.returncode == 0
        parsed = json.loads(result.stdout.split("SETTINGS:")[1].strip())
        assert "/some/project/dir" in parsed["filesystem"]["allowWrite"]

    def test_no_patch_flag(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        work = tmp_path / "work"
        work.mkdir()
        sandbox_dir = work / ".sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "default.json").write_text(
            json.dumps({"filesystem": {"allowWrite": ["."]}})
        )

        env["PROJ_DIR"] = str(work)

        result = run_agm(
            ["run", "--no-patch", "echo", "hi"], env=env, cwd=str(work),
        )
        assert result.returncode == 0
        parsed = json.loads(result.stdout.split("SETTINGS:")[1].strip())
        assert str(work) not in parsed["filesystem"]["allowWrite"]

    def test_invalid_option(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        result = run_agm(
            ["run", "--bad-opt", "echo"],
            env=env, cwd=str(tmp_path), check=False,
        )
        assert result.returncode != 0

    def test_f_and_no_patch_together(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        work = tmp_path / "work"
        work.mkdir()
        sf = work / "custom.json"
        sf.write_text(json.dumps({"filesystem": {"allowWrite": ["/x"]}}))

        env["PROJ_DIR"] = "/proj"

        result = run_agm(
            ["run", "--no-patch", "-f", str(sf), "echo"],
            env=env, cwd=str(work),
        )
        assert result.returncode == 0
        parsed = json.loads(result.stdout.split("SETTINGS:")[1].strip())
        assert parsed["filesystem"]["allowWrite"] == ["/x"]
        assert "/proj" not in parsed["filesystem"]["allowWrite"]


# ── agm open / new / co ─────────────────────────────────────────────────────


@needs_zsh
class TestPm:
    """agm open / new / co: project session management."""

    def test_open_default_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """open with no branch opens a session in repo/."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env, name="myproj")
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(["open"], env=env, cwd=str(project))

        log = tmux_log.read_text()
        assert "new-session" in log
        assert "-s myproj" in log

    def test_open_specific_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """open BRANCH opens a session in worktrees/BRANCH/."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env, name="proj")
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(["open", "feat/x"], env=env, cwd=str(project))

        log = tmux_log.read_text()
        assert "new-session" in log
        assert "-s proj/feat/x" in log
        # The worktrees/feat/x dir should have been mkdir -p'd.
        assert (project / "worktrees" / "feat/x").is_dir()

    def test_open_with_pane_count(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(["open", "-n", "6"], env=env, cwd=str(project))

        log = tmux_log.read_text()
        assert log.count("split-window") == 5  # 6 panes = 5 splits

    def test_new_creates_worktree_and_detached_session(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env, name="proj")
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(["new", "feat/test"], env=env, cwd=str(project))

        # Worktree should exist.
        assert (project / "worktrees" / "feat/test").is_dir()
        assert (project / "worktrees" / "feat/test" / "README.md").exists()
        # Session should be detached (new always passes -d).
        log = tmux_log.read_text()
        assert "-dP" in log
        assert "-s proj/feat/test" in log

    def test_new_with_pane_count(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(["new", "-n", "2", "feat/n"], env=env, cwd=str(project))

        log = tmux_log.read_text()
        assert log.count("split-window") == 1  # 2 panes = 1 split

    def test_new_with_parent(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """new -p PARENT creates the branch from the specified parent."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        # Create a parent worktree using raw git.
        _git(
            "worktree", "add", "-b", "develop",
            str(project / "worktrees" / "develop"),
            cwd=str(project / "repo"), env=env,
        )
        # Add a file on develop.
        (project / "worktrees" / "develop" / "dev.txt").write_text("dev")
        _git("add", ".", cwd=str(project / "worktrees" / "develop"), env=env)
        _git("commit", "-m", "dev", cwd=str(project / "worktrees" / "develop"), env=env)

        run_agm(
            ["new", "-p", "develop", "feat/from-dev"],
            env=env, cwd=str(project),
        )

        wt = project / "worktrees" / "feat/from-dev"
        assert wt.is_dir()
        # The new branch should have dev.txt from develop.
        assert (wt / "dev.txt").exists()

    def test_co_checks_out_existing_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)

        # Push a branch to origin.
        clone = tmp_path / "tmp-clone"
        _git("clone", str(bare), str(clone), cwd=str(tmp_path), env=env)
        _push_branch(clone, bare, "feat/z", "z.txt", env)

        project = _make_project(tmp_path, bare, env, name="proj")
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(["co", "feat/z"], env=env, cwd=str(project))

        wt = project / "worktrees" / "feat/z"
        assert wt.is_dir()
        assert (wt / "z.txt").exists()
        log = tmux_log.read_text()
        assert "-s proj/feat/z" in log

    def test_co_with_pane_count(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        clone = tmp_path / "tmp-clone"
        _git("clone", str(bare), str(clone), cwd=str(tmp_path), env=env)
        _push_branch(clone, bare, "feat/p", "p.txt", env)

        project = _make_project(tmp_path, bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(
            ["co", "-n", "3", "feat/p"], env=env, cwd=str(project),
        )

        log = tmux_log.read_text()
        assert log.count("split-window") == 2  # 3 panes

    def test_co_with_parent(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        clone = tmp_path / "tmp-clone"
        _git("clone", str(bare), str(clone), cwd=str(tmp_path), env=env)
        _push_branch(clone, bare, "feat/q", "q.txt", env)

        project = _make_project(tmp_path, bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(
            ["co", "-p", "main", "feat/q"], env=env, cwd=str(project),
        )

        assert (project / "worktrees" / "feat/q" / "q.txt").exists()

    def test_error_new_without_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        result = run_agm(
            ["new"], env=env, cwd=str(project), check=False,
        )
        assert result.returncode != 0
        assert "branch" in result.stderr.lower()

    def test_error_co_without_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        result = run_agm(
            ["co"], env=env, cwd=str(project), check=False,
        )
        assert result.returncode != 0
        assert "branch" in result.stderr.lower()

    def test_invalid_pane_count(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        result = run_agm(
            ["open", "-n", "abc"], env=env, cwd=str(project), check=False,
        )
        assert result.returncode != 0
        assert "pane count must be a positive integer" in result.stderr

    def test_no_subcommand_shows_usage(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        result = run_agm(
            [], env=env, cwd=str(tmp_path), check=False,
        )
        assert result.returncode != 0
        assert "usage" in result.stderr.lower()

    def test_sources_project_env_file(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """config/env.sh should be sourced before opening the session."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env, name="proj")
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        # env.sh creates a marker file to prove it was sourced.
        (project / "config" / "env.sh").write_text(
            f'touch "{project}/env-sourced"\n'
        )

        run_agm(["open"], env=env, cwd=str(project))

        assert (project / "env-sourced").exists()

    def test_sources_branch_env_file(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """config/<branch>/env.sh should be sourced for branch sessions."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env, name="proj")
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        branch_cfg = project / "config" / "mybranch"
        branch_cfg.mkdir(parents=True)
        (branch_cfg / "env.sh").write_text(
            f'touch "{project}/branch-env-sourced"\n'
        )

        run_agm(["open", "mybranch"], env=env, cwd=str(project))

        assert (project / "branch-env-sourced").exists()

    def test_checkout_alias(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """agm checkout should work identically to agm co."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        clone = tmp_path / "tmp-clone"
        _git("clone", str(bare), str(clone), cwd=str(tmp_path), env=env)
        _push_branch(clone, bare, "feat/alias", "alias.txt", env)

        project = _make_project(tmp_path, bare, env, name="proj")
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(
            ["checkout", "feat/alias"], env=env, cwd=str(project),
        )

        wt = project / "worktrees" / "feat/alias"
        assert wt.is_dir()
        assert (wt / "alias.txt").exists()
        log = tmux_log.read_text()
        assert "-s proj/feat/alias" in log

    def test_init_then_new_lifecycle(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """Full workflow: agm init creates a project, agm new creates a branch."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        # Step 1: init the project.
        run_agm(["init", "myproj", str(bare)], env=env, cwd=str(tmp_path))
        project = tmp_path / "myproj"

        # Step 2: create a new branch session.
        run_agm(["new", "feat/lifecycle"], env=env, cwd=str(project))

        # Verify: worktree exists with correct branch and content.
        wt = project / "worktrees" / "feat/lifecycle"
        assert wt.is_dir()
        assert (wt / "README.md").exists()
        head = _git(
            "rev-parse", "--abbrev-ref", "HEAD", cwd=str(wt), env=env,
        ).stdout.strip()
        assert head == "feat/lifecycle"
        # Verify: tmux session was created detached.
        log = tmux_log.read_text()
        assert "-dP" in log
        assert "myproj/feat/lifecycle" in log


# ── agm tmux new ────────────────────────────────────────────────────────────


@needs_zsh
class TestTmuxSession:
    """agm tmux new: create tmux sessions with tiled pane layout."""

    def test_creates_session_default_panes(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        work = tmp_path / "work"
        work.mkdir()

        run_agm(["tmux", "new"], env=env, cwd=str(work))

        log = tmux_log.read_text()
        assert "new-session" in log
        # Default is 4 panes → 3 splits.
        assert log.count("split-window") == 3

    def test_custom_pane_count(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)
        work = tmp_path / "work"
        work.mkdir()

        run_agm(["tmux", "new", "-n", "6"], env=env, cwd=str(work))

        log = tmux_log.read_text()
        assert log.count("split-window") == 5

    def test_single_pane(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)
        work = tmp_path / "work"
        work.mkdir()

        run_agm(["tmux", "new", "-n", "1"], env=env, cwd=str(work))

        log = tmux_log.read_text()
        assert "new-session" in log
        assert "split-window" not in log

    def test_detached_session(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)
        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(["tmux", "new", "-d", "my-session"], env=env, cwd=str(work))

        log = tmux_log.read_text()
        assert "-dP" in log
        assert "-s my-session" in log
        assert "Detached tmux session" in result.stdout

    def test_long_detach_flag(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)
        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(["tmux", "new", "--detach"], env=env, cwd=str(work))

        log = tmux_log.read_text()
        assert "-dP" in log
        assert "Detached" in result.stdout

    def test_session_name_from_argument(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)
        work = tmp_path / "work"
        work.mkdir()

        run_agm(["tmux", "new", "my-custom-name"], env=env, cwd=str(work))

        log = tmux_log.read_text()
        assert "-s my-custom-name" in log

    def test_detach_with_custom_pane_count_and_name(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)
        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(
            ["tmux", "new", "-d", "-n", "3", "sess"], env=env, cwd=str(work),
        )

        log = tmux_log.read_text()
        assert "-dP" in log
        assert "-s sess" in log
        assert log.count("split-window") == 2
        assert "Detached" in result.stdout

    def test_invalid_pane_count(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)
        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(
            ["tmux", "new", "-n", "abc"], env=env, cwd=str(work), check=False,
        )
        assert result.returncode != 0
        assert "invalid pane count" in result.stderr.lower()

    def test_too_many_positional_args(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)
        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(
            ["tmux", "new", "a", "b"], env=env, cwd=str(work), check=False,
        )
        assert result.returncode != 0


# ── agm tmux layout ────────────────────────────────────────────────────────


@needs_zsh
class TestTmuxLayout:
    """agm tmux layout: apply tiled pane layout."""

    def test_single_pane_layout(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(
            ["tmux", "layout", "1", "@0", "200", "50"],
            env=env, cwd=str(tmp_path),
        )

        log = tmux_log.read_text()
        assert "select-layout" in log
        assert "@0" in log
        # Single-pane layout body: WxH,0,0,0
        assert "200x50,0,0,0" in log

    def test_four_pane_layout(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(
            ["tmux", "layout", "4", "@1", "200", "50"],
            env=env, cwd=str(tmp_path),
        )

        log = tmux_log.read_text()
        assert "select-layout" in log
        assert "@1" in log
        # 4 panes → 2x2 grid; all 4 pane indices must appear.
        for idx in range(4):
            assert f",{idx}" in log

    def test_nine_pane_layout(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(
            ["tmux", "layout", "9", "@0", "300", "90"],
            env=env, cwd=str(tmp_path),
        )

        log = tmux_log.read_text()
        assert "select-layout" in log
        for idx in range(9):
            assert f",{idx}" in log

    def test_two_pane_layout(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(
            ["tmux", "layout", "2", "@0", "200", "50"],
            env=env, cwd=str(tmp_path),
        )

        log = tmux_log.read_text()
        assert "select-layout" in log
        # 2 panes → 1 row, 2 columns.
        assert ",0" in log
        assert ",1" in log

    def test_layout_has_checksum(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """Layout string must start with a 4-hex-digit checksum."""
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(
            ["tmux", "layout", "4", "@0", "200", "50"],
            env=env, cwd=str(tmp_path),
        )

        log = tmux_log.read_text()
        # Extract layout argument (last arg after @0).
        match = re.search(r"select-layout -t @0 ([0-9a-f]{4},\S+)", log)
        assert match, f"expected checksum,layout in: {log}"

    def test_missing_args(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        result = run_agm(
            ["tmux", "layout", "4"], env=env, cwd=str(tmp_path), check=False,
        )
        assert result.returncode != 0


# ── full CLI integration ────────────────────────────────────────────────────


class TestAgmCli:
    """Integration tests that invoke the agm CLI entry point."""

    def test_init_through_cli(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "myrepo.git", env)

        result = subprocess.run(
            [sys.executable, "-m", "agm.cli", "init", str(bare)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0
        proj = tmp_path / "myrepo"
        assert (proj / "repo" / "README.md").exists()
        for d in ("deps", "worktrees", "notes", "config"):
            assert (proj / d).is_dir()

    def test_fetch_through_cli(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        result = subprocess.run(
            [sys.executable, "-m", "agm.cli", "fetch"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(project),
        )
        assert result.returncode == 0
        assert "Fetching repo" in result.stdout

    def test_help_overview_through_cli(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "agm.cli", "help"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0
        assert "agm - Agent Management Framework" in result.stdout
        assert "Commands:" in result.stdout

    def test_help_specific_command_through_cli(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "agm.cli", "help", "init"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0
        assert "agm init" in result.stdout
        assert "REPO_URL" in result.stdout

    def test_branch_sync_through_cli(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        _push_branch(work, bare, "feat/cli", "cli.txt", env)
        _git("branch", "-D", "feat/cli", cwd=str(work), env=env)

        result = subprocess.run(
            [sys.executable, "-m", "agm.cli", "br", "sync"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(work),
        )
        assert result.returncode == 0

        branches = _git("branch", cwd=str(work), env=env).stdout
        assert "feat/cli" in branches

    def test_help_alias_through_cli(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "agm.cli", "help", "co"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0
        assert "agm checkout" in result.stdout

    def test_help_unknown_command_through_cli(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "agm.cli", "help", "bogus"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(tmp_path),
        )
        assert result.returncode == 1
        assert "unknown command" in result.stderr
