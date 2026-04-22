"""End-to-end tests that exercise commands through the ``agm`` CLI.

These tests create real git repos, invoke the ``agm`` command-line interface,
and verify the resulting filesystem and git state.

Test setup uses only raw git/filesystem operations, never other scripts from
this repository.  The shell scripts must be installed on PATH (via
``just install``) before running these tests.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict, cast

import pytest

HAS_ZSH = shutil.which("zsh") is not None
needs_zsh = pytest.mark.skipif(not HAS_ZSH, reason="zsh is required")

# A fake ``tmux`` binary used by tests for agm open, tmux.sh and
# tmux-apply-layout.sh.  It logs every invocation and returns canned
# responses so that the calling zsh scripts can complete without a real
# tmux server.
#
# Real tmux subcommands emulated and their actual behaviour:
#
#   new-session [-d] [-P] [-F FMT] [-c DIR] [-s NAME] [-e K=V …]
#     Creates a new session.  With -dP the real tmux creates the session
#     detached and prints the session info formatted by -F.  Our mock
#     prints the NAME supplied via -s (or "0" as default) — matching the
#     format string '#{session_name}' used by tmux.sh.
#
#   display-message -p [-t TARGET] FORMAT
#     Real tmux expands format variables and prints the result.  Our mock
#     returns hard-coded values for the three variables the scripts query:
#       #{window_id}     → @0   (tmux window-id format: @ + integer)
#       #{window_width}  → 200  (columns)
#       #{window_height} → 50   (rows)
#
#   split-window [-d] [-h|-v] [-t TARGET] [-c DIR]
#     Splits a pane.  The mock logs and exits 0.
#
#   select-layout [-t WINDOW] LAYOUT_STRING
#     Applies a custom layout string.  The mock logs and exits 0.
#
#   select-pane [-t TARGET]
#     Selects a pane.  The mock logs and exits 0.
#
#   switch-client [-t TARGET]
#     Switches the current client to another session.  The mock logs
#     and exits 0.
#
#   attach-session [-t TARGET]
#     Attaches the current client to another session. The mock logs and
#     exits 0.
#
#   send-keys [-t TARGET] KEYS... [C-m]
#     Sends keystrokes to a pane. The mock executes the command
#     asynchronously when Enter is present so tests can verify that setup
#     work is queued into the new session rather than run inline.
#
#   run-shell COMMAND
#     Real tmux executes COMMAND in a shell.  The mock also executes the
#     command so that tmux-apply-layout.sh (invoked via run-shell by
#     tmux.sh in non-detached mode) issues its own select-layout call
#     which is then captured in the log.
#
# Limitation: when tmux.sh passes a compound command with ';' separators
# (non-detached path), the mock receives them as a single invocation.  It
# handles the first subcommand (new-session) and logs everything, but
# individual embedded subcommands are not executed separately — only the
# argument list is captured.  This is acceptable because the detached path
# (the primary test path) makes separate tmux calls.
_FAKE_TMUX_SCRIPT = r"""#!/bin/bash
log="${TMUX_LOG:?TMUX_LOG must be set}"
echo "CMD: $*" >> "$log"

case "$1" in
  new-session)
    # With -dP (detach + print), real tmux prints the session name
    # formatted by -F '#{session_name}'.  We return the -s argument.
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
    # Real tmux expands #{var} in the format argument.
    # We return the hard-coded value for the requested variable.
    for arg in "$@"; do
      case "$arg" in
        *window_id*)     echo "@0"; break ;;
        *window_width*)  echo "200"; break ;;
        *window_height*) echo "50"; break ;;
      esac
    done
    ;;
  run-shell)
    # Real tmux executes the argument in a shell.  We do the same so
    # that tmux-apply-layout.sh can issue its own tmux select-layout.
    shift
    eval "$*"
    ;;
  send-keys)
    shift
    if [[ "${1:-}" == "-t" ]]; then
      shift 2
    fi
    command=""
    for arg in "$@"; do
      if [[ "$arg" == "C-m" ]]; then
        break
      fi
      command="$arg"
    done
    if [[ -n "$command" ]]; then
      ( eval "$command" ) &
    fi
    ;;
esac
exit 0
"""


class _SrtNetworkSettings(TypedDict):
    allowedDomains: list[str]


class _SrtFilesystemSettings(TypedDict):
    allowWrite: list[str]


class _SrtSettings(TypedDict, total=False):
    network: _SrtNetworkSettings
    filesystem: _SrtFilesystemSettings
    ignoreViolations: dict[str, bool]
    enabled: bool
    enableWeakerNestedSandbox: bool


def _network_settings(*allowed_domains: str) -> _SrtNetworkSettings:
    return {"allowedDomains": list(allowed_domains)}


def _filesystem_settings(*allow_write: str) -> _SrtFilesystemSettings:
    return {"allowWrite": list(allow_write)}


def _settings(
    *,
    network: _SrtNetworkSettings | None = None,
    filesystem: _SrtFilesystemSettings | None = None,
    ignore_violations: dict[str, bool] | None = None,
    enabled: bool | None = None,
    enable_weaker_nested_sandbox: bool | None = None,
) -> _SrtSettings:
    settings: _SrtSettings = {}
    if network is not None:
        settings["network"] = network
    if filesystem is not None:
        settings["filesystem"] = filesystem
    if ignore_violations is not None:
        settings["ignoreViolations"] = ignore_violations
    if enabled is not None:
        settings["enabled"] = enabled
    if enable_weaker_nested_sandbox is not None:
        settings["enableWeakerNestedSandbox"] = enable_weaker_nested_sandbox
    return settings


def _install_fake_claude(directory: Path, env: dict[str, str]) -> Path:
    """Create a fake ``claude`` binary that returns scripted outputs."""

    directory.mkdir(parents=True, exist_ok=True)
    claude = directory / "claude"
    claude.write_text(
        "#!/bin/bash\n"
        'state_file="${FAKE_CLAUDE_STATE:?FAKE_CLAUDE_STATE must be set}"\n'
        'log_file="${FAKE_CLAUDE_LOG:?FAKE_CLAUDE_LOG must be set}"\n'
        'count=0\n'
        'if [[ -f "$state_file" ]]; then\n'
        '  count="$(cat "$state_file")"\n'
        "fi\n"
        'count=$((count + 1))\n'
        'printf "%s" "$count" > "$state_file"\n'
        'echo "$*" >> "$log_file"\n'
        'case "$count" in\n'
        '  1) printf "keep going\\n" ;;\n'
        '  2) printf "  COMPLETE  \\n" ;;\n'
        '  *) printf "COMPLETE\\n" ;;\n'
        "esac\n"
    )
    claude.chmod(claude.stat().st_mode | stat.S_IEXEC)
    env["PATH"] = str(directory) + ":" + env["PATH"]
    return claude


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


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


def _wait_for_path(path: Path, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


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


def _srt_settings(result: subprocess.CompletedProcess[str]) -> _SrtSettings:
    """Extract the JSON settings dict from fake srt stdout."""
    for line in result.stdout.splitlines():
        if line.startswith("SETTINGS:"):
            raw: object = json.loads(line[len("SETTINGS:"):])
            if not isinstance(raw, dict):
                raise ValueError(f"Expected JSON object in srt output:\n{result.stdout}")
            return cast(_SrtSettings, raw)
    raise ValueError(f"No SETTINGS line in srt output:\n{result.stdout}")


def _srt_command(result: subprocess.CompletedProcess[str]) -> str:
    """Extract the forwarded command string from fake srt stdout."""
    for line in result.stdout.splitlines():
        if line.startswith("COMMAND:"):
            return line[len("COMMAND:"):]
    return ""


def _systemd_run_command(result: subprocess.CompletedProcess[str]) -> str:
    """Extract the forwarded systemd-run argument string from stdout."""
    for line in result.stdout.splitlines():
        if line.startswith("SYSTEMD_RUN:"):
            return line[len("SYSTEMD_RUN:"):]
    return ""


# ── agm config cp ───────────────────────────────────────────────────────────


class TestCpConfig:
    """agm config cp: copy configuration files."""

    def test_ignores_files_from_current_dir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "repo").mkdir()
        (project / "worktrees").mkdir()
        config = project / "config"
        config.mkdir()
        (config / ".env").write_text("FROM_CONFIG=1")

        src = project / "repo"
        (src / ".env").write_text("FROM_CWD=1")
        (src / ".claude").mkdir()
        (src / ".claude" / "settings.json").write_text("{\"cwd\":true}")

        dest = tmp_path / "dest"
        dest.mkdir()

        run_agm(["config", "cp", str(dest)], env=env, cwd=str(src))

        assert (dest / ".env").read_text() == "FROM_CONFIG=1"
        assert not (dest / ".claude").exists()

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

    def test_d_option_is_rejected(self, tmp_path: Path, env: dict[str, str]) -> None:
        custom = tmp_path / "custom"
        custom.mkdir()
        (custom / "config").mkdir()
        (custom / "config" / ".env").write_text("CUSTOM=1")

        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"
        dest.mkdir()

        result = run_agm(
            ["config", "cp", "-d", str(custom), str(dest)],
            env=env,
            cwd=str(src),
            check=False,
        )
        assert result.returncode != 0

        assert not (dest / ".env").exists()

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

    def test_auto_detects_project_from_custom_git_worktree_path(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)
        (project / "config" / ".env").write_text("FROM_CUSTOM_WT=1")

        custom_worktrees = tmp_path / "custom-worktrees"
        custom_worktrees.mkdir()
        run_agm(
            ["wt", "new", "-d", str(custom_worktrees), "feat/custom-copy"],
            env=env,
            cwd=str(project / "repo"),
        )

        wt = custom_worktrees / "feat/custom-copy"
        dest = tmp_path / "dest"
        dest.mkdir()

        run_agm(["config", "cp", str(dest)], env=env, cwd=str(wt))

        assert (dest / ".env").read_text() == "FROM_CUSTOM_WT=1"

    def test_auto_detects_embedded_project_from_dot_worktrees_subdir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = make_working_repo(tmp_path / "proj", bare, env)
        (project / ".agm" / "config").mkdir(parents=True)
        (project / ".agm" / "config" / ".env").write_text("FROM_EMBEDDED=1")

        wt = project / ".agm" / "worktrees" / "feat-x"
        wt.mkdir(parents=True)
        dest = tmp_path / "dest"
        dest.mkdir()

        run_agm(["config", "cp", str(dest)], env=env, cwd=str(wt))

        assert (dest / ".env").read_text() == "FROM_EMBEDDED=1"

    def test_auto_detects_embedded_project_from_repo_subdir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = make_working_repo(tmp_path / "proj", bare, env)
        src = project / "src"
        src.mkdir()
        (project / ".agm" / "config").mkdir(parents=True)
        (project / ".agm" / "config" / ".env").write_text("FROM_EMBEDDED_SUBDIR=1")

        dest = tmp_path / "dest"
        dest.mkdir()

        run_agm(["config", "cp", str(dest)], env=env, cwd=str(src))

        assert (dest / ".env").read_text() == "FROM_EMBEDDED_SUBDIR=1"

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
        project = tmp_path / "proj"
        project.mkdir()
        (project / "repo").mkdir()
        (project / "worktrees").mkdir()
        config = project / "config"
        config.mkdir()
        (config / ".env").write_text("REL=1")

        src = project / "repo"
        (src / "target").mkdir()

        run_agm(["config", "cp", "target"], env=env, cwd=str(src))

        assert (src / "target" / ".env").read_text() == "REL=1"

    def test_copies_all_recognized_file_types(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "repo").mkdir()
        (project / "worktrees").mkdir()
        config = project / "config"
        config.mkdir()

        src = project / "repo"
        dest = tmp_path / "dest"
        dest.mkdir()

        files = [".setup.sh", ".env", ".env.local", ".mcp.json",
                 ".agents", ".opencode", ".codex", ".pi"]
        for f in files:
            (config / f).write_text(f"content of {f}")

        run_agm(["config", "cp", str(dest)], env=env, cwd=str(src))

        for f in files:
            assert (dest / f).read_text() == f"content of {f}"

    def test_config_copy_long_alias(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """The 'config copy' alias works identically to 'config cp'."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "repo").mkdir()
        (project / "config").mkdir()
        (project / "config" / ".env").write_text("ALIAS=1")

        src = project / "repo"
        dest = tmp_path / "dest"
        dest.mkdir()

        run_agm(["config", "copy", str(dest)], env=env, cwd=str(src))

        assert (dest / ".env").read_text() == "ALIAS=1"


# ── agm wt new ───────────────────────────────────────────────────────────────


class TestMkWt:
    """agm wt new: create or check out git worktrees."""

    def test_new_checks_out_existing_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        _push_branch(work, bare, "feat/x", "x.txt", env)
        _git("branch", "-D", "feat/x", cwd=str(work), env=env)

        wt_dir = tmp_path / "worktrees"
        wt_dir.mkdir()

        run_agm(["wt", "new", "-d", str(wt_dir), "feat/x"], env=env, cwd=str(work))

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

    def test_existing_local_branch_checks_out_worktree(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        _git("checkout", "-b", "existing-local", cwd=str(work), env=env)
        (work / "local.txt").write_text("local\n")
        _git("add", "local.txt", cwd=str(work), env=env)
        _git("commit", "-m", "add local branch file", cwd=str(work), env=env)
        _git("checkout", "main", cwd=str(work), env=env)

        wt_dir = tmp_path / "worktrees"
        wt_dir.mkdir()

        run_agm(
            ["wt", "new", "-d", str(wt_dir), "existing-local"],
            env=env,
            cwd=str(work),
        )

        assert (wt_dir / "existing-local" / "local.txt").read_text() == "local\n"
        head = _git(
            "rev-parse", "--abbrev-ref", "HEAD",
            cwd=str(wt_dir / "existing-local"),
            env=env,
        ).stdout.strip()
        assert head == "existing-local"

    def test_existing_remote_branch_checks_out_worktree(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        _push_branch(work, bare, "existing-remote", "remote.txt", env)
        _git("branch", "-D", "existing-remote", cwd=str(work), env=env)

        wt_dir = tmp_path / "worktrees"
        wt_dir.mkdir()

        run_agm(
            ["wt", "new", "-d", str(wt_dir), "existing-remote"],
            env=env,
            cwd=str(work),
        )

        assert (wt_dir / "existing-remote" / "remote.txt").read_text() == "remote.txt"
        head = _git(
            "rev-parse", "--abbrev-ref", "HEAD",
            cwd=str(wt_dir / "existing-remote"),
            env=env,
        ).stdout.strip()
        assert head == "existing-remote"

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

    def test_wt_new_does_not_run_project_setup_script(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        setup = project / "config" / "setup.sh"
        setup.write_text('#!/bin/bash\ntouch "$PWD/.setup-ran"\n')
        setup.chmod(setup.stat().st_mode | stat.S_IEXEC)

        run_agm(["wt", "new", "setup-test"], env=env, cwd=str(project / "repo"))

        assert not (project / "worktrees" / "setup-test" / ".setup-ran").exists()

    def test_wt_setup_runs_project_setup_script(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        setup = project / "config" / "setup.sh"
        setup.write_text('#!/bin/bash\ntouch "$PWD/.setup-ran"\n')
        setup.chmod(setup.stat().st_mode | stat.S_IEXEC)

        run_agm(["wt", "new", "setup-test"], env=env, cwd=str(project / "repo"))
        result = run_agm(["wt", "setup"], env=env, cwd=str(project / "worktrees" / "setup-test"))

        assert (project / "worktrees" / "setup-test" / ".setup-ran").exists()
        assert "Running setup for" in result.stdout
        assert "Setup complete" in result.stdout
        assert "true" not in result.stdout

    def test_wt_setup_runs_dot_setup_sh(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """A .setup.sh copied into the worktree should be executed."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        dot_setup = project / "config" / ".setup.sh"
        dot_setup.write_text('#!/bin/bash\ntouch "$PWD/.dot-setup-ran"\n')
        dot_setup.chmod(dot_setup.stat().st_mode | stat.S_IEXEC)

        run_agm(["wt", "new", "dotsetup"], env=env, cwd=str(project / "repo"))
        run_agm(["wt", "setup"], env=env, cwd=str(project / "worktrees" / "dotsetup"))

        assert (project / "worktrees" / "dotsetup" / ".dot-setup-ran").exists()

    def test_wt_setup_runs_dotconfig_setup_sh(
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
        run_agm(["wt", "setup"], env=env, cwd=str(project / "worktrees" / "dc-setup"))

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
        assert "not a git repository" in result.stderr.lower()

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

    def test_default_worktrees_dir_for_embedded_project(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = make_working_repo(tmp_path / "proj", bare, env)
        (project / ".agm").mkdir()

        run_agm(["wt", "new", "embedded-auto"], env=env, cwd=str(project))

        assert (project / ".agm" / "worktrees" / "embedded-auto").is_dir()

    def test_default_worktrees_dir_for_embedded_project_from_repo_subdir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = make_working_repo(tmp_path / "proj", bare, env)
        (project / ".agm").mkdir()
        src = project / "src"
        src.mkdir()

        run_agm(["wt", "new", "embedded-subdir"], env=env, cwd=str(src))

        assert (project / ".agm" / "worktrees" / "embedded-subdir").is_dir()

    def test_rejects_repo_alias_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        result = run_agm(
            ["wt", "new", "repo"], env=env, cwd=str(project / "repo"), check=False,
        )

        assert result.returncode != 0
        assert "repo checkout" in (result.stdout + result.stderr).lower()

    def test_rejects_main_repo_branch_name(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        result = run_agm(
            ["wt", "new", "main"], env=env, cwd=str(project / "repo"), check=False,
        )

        assert result.returncode != 0
        assert "repo checkout" in (result.stdout + result.stderr).lower()

    def test_wt_new_checks_out_existing_remote_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """``wt new`` checks out an existing remote branch into a worktree."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        _push_branch(work, bare, "feat/long", "long.txt", env)
        _git("branch", "-D", "feat/long", cwd=str(work), env=env)

        wt_dir = tmp_path / "worktrees"
        wt_dir.mkdir()

        run_agm(
            ["wt", "new", "-d", str(wt_dir), "feat/long"],
            env=env, cwd=str(work),
        )

        assert (wt_dir / "feat/long" / "long.txt").exists()
        head = _git(
            "rev-parse", "--abbrev-ref", "HEAD",
            cwd=str(wt_dir / "feat/long"), env=env,
        ).stdout.strip()
        assert head == "feat/long"


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

        # Normal removal should fail because worktree is dirty.
        result = run_agm(
            ["wt", "rm", "dirty-branch"], env=env, cwd=str(work), check=False,
        )
        assert result.returncode != 0
        assert "dirty" in result.stderr.lower() or "changes" in result.stderr.lower()
        # Worktree must still exist after the failed removal.
        assert (wt_dir / "dirty-branch").is_dir()

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
        output = (result.stdout + result.stderr).lower()
        assert "no-such-branch" in output

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

    def test_worktree_remove_long_alias(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """'worktree remove' works identically to 'wt rm'."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        work = make_working_repo(tmp_path / "work", bare, env)

        wt_dir = tmp_path / "worktrees"
        wt_dir.mkdir()
        _git(
            "worktree", "add", "-b", "long-rm",
            str(wt_dir / "long-rm"),
            cwd=str(work), env=env,
        )

        run_agm(["worktree", "remove", "long-rm"], env=env, cwd=str(work))

        assert not (wt_dir / "long-rm").exists()
        branches = _git("branch", cwd=str(work), env=env).stdout
        assert "long-rm" not in branches

    def test_rejects_removing_repo_alias_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        result = run_agm(["wt", "rm", "repo"], env=env, cwd=str(project), check=False)

        assert result.returncode != 0
        assert "repo checkout" in (result.stdout + result.stderr).lower()

    def test_rejects_removing_main_repo_branch_name(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        result = run_agm(["wt", "rm", "main"], env=env, cwd=str(project), check=False)

        assert result.returncode != 0
        assert "repo checkout" in (result.stdout + result.stderr).lower()


# ── agm close ────────────────────────────────────────────────────────────────


class TestClose:
    """agm close: remove worktrees and stop matching tmux sessions."""

    def test_removes_worktree_and_kills_tmux_session(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env, name="proj")
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(["wt", "new", "feat/close-me"], env=env, cwd=str(project / "repo"))
        worktree = project / "worktrees" / "feat/close-me"
        assert worktree.is_dir()

        result = run_agm(["close", "feat/close-me"], env=env, cwd=str(project))

        assert not worktree.exists()
        branches = _git("branch", cwd=str(project / "repo"), env=env).stdout
        assert "feat/close-me" not in branches
        assert "Closed session proj/feat/close-me" in result.stdout
        log = tmux_log.read_text()
        assert "kill-session -t proj/feat/close-me" in log

    def test_requires_branch_argument(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        result = run_agm(["close"], env=env, cwd=str(tmp_path), check=False)
        assert result.returncode != 0
        assert "usage" in result.stderr.lower()

    def test_removes_embedded_project_worktree_from_repo_subdir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = make_working_repo(tmp_path / "proj", bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)
        (project / ".agm").mkdir()

        run_agm(["wt", "new", "feat/close-embedded"], env=env, cwd=str(project))
        worktree = project / ".agm" / "worktrees" / "feat/close-embedded"
        src = project / "src"
        src.mkdir()

        result = run_agm(["close", "feat/close-embedded"], env=env, cwd=str(src))

        assert not worktree.exists()
        branches = _git("branch", cwd=str(project), env=env).stdout
        assert "feat/close-embedded" not in branches
        assert "Closed session proj/feat/close-embedded" in result.stdout


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

    def test_clones_dependency_from_repo_dir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare_main = make_bare_repo(tmp_path / "main.git", env)
        bare_dep = make_bare_repo(tmp_path / "mylib.git", env)
        project = _make_project(tmp_path, bare_main, env)

        run_agm(["dep", "new", str(bare_dep)], env=env, cwd=str(project / "repo"))

        dep = project / "deps" / "mylib" / "main"
        assert dep.is_dir()
        assert (dep / "README.md").exists()

    def test_no_subcommand_shows_help(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        result = run_agm(
            ["dep"], env=env, cwd=str(tmp_path), check=False,
        )
        expected = run_agm(["help", "dep"], env=env, cwd=str(tmp_path))
        assert result.returncode == 0
        assert result.stderr == ""
        assert result.stdout == expected.stdout


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

    def test_switch_with_slashed_branch_name(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """dep switch with slashed branch names creates nested dirs."""
        bare = make_bare_repo(tmp_path / "mylib.git", env)

        clone = tmp_path / "tmp-clone"
        _git("clone", str(bare), str(clone), cwd=str(tmp_path), env=env)
        _push_branch(clone, bare, "feat/deep/name", "deep.txt", env)

        project = self._setup_dep(tmp_path, bare, env)

        run_agm(
            ["dep", "switch", "mylib", "feat/deep/name"],
            env=env, cwd=str(project),
        )

        switched = project / "deps" / "mylib" / "feat/deep/name"
        assert (switched / "deep.txt").exists()
        head = _git(
            "rev-parse", "--abbrev-ref", "HEAD",
            cwd=str(switched), env=env,
        ).stdout.strip()
        assert head == "feat/deep/name"

    def test_switch_nonexistent_remote_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """dep switch to a branch that doesn't exist fails."""
        bare = make_bare_repo(tmp_path / "mylib.git", env)
        project = self._setup_dep(tmp_path, bare, env)

        result = run_agm(
            ["dep", "switch", "mylib", "no-such-branch"],
            env=env, cwd=str(project), check=False,
        )
        assert result.returncode != 0


class TestDepRemove:
    """agm dep rm: remove dependency worktrees and repositories."""

    def test_removes_dependency_worktree(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "mylib.git", env)
        clone = tmp_path / "tmp-clone"
        _git("clone", str(bare), str(clone), cwd=str(tmp_path), env=env)
        _push_branch(clone, bare, "feat/remove", "remove.txt", env)
        project = TestDepSwitch._setup_dep(tmp_path, bare, env)
        run_agm(["dep", "switch", "mylib", "feat/remove"], env=env, cwd=str(project))

        run_agm(["dep", "rm", "mylib/feat/remove"], env=env, cwd=str(project))

        dep_dir = project / "deps" / "mylib"
        assert not (dep_dir / "feat/remove").exists()
        assert (dep_dir / "main").is_dir()
        branches = _git("branch", cwd=str(dep_dir / "main"), env=env).stdout
        assert "feat/remove" not in branches

    def test_removes_all_dependency_worktrees_and_repo(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "mylib.git", env)
        clone = tmp_path / "tmp-clone"
        _git("clone", str(bare), str(clone), cwd=str(tmp_path), env=env)
        _push_branch(clone, bare, "feat/remove", "remove.txt", env)
        project = TestDepSwitch._setup_dep(tmp_path, bare, env)
        run_agm(["dep", "switch", "mylib", "feat/remove"], env=env, cwd=str(project))

        run_agm(["dep", "rm", "--all", "mylib"], env=env, cwd=str(project))

        assert not (project / "deps" / "mylib").exists()

    def test_removes_dependency_repo_via_repo_alias(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "mylib.git", env)
        project = TestDepSwitch._setup_dep(tmp_path, bare, env)

        run_agm(["dep", "rm", "mylib/repo"], env=env, cwd=str(project))

        assert not (project / "deps" / "mylib").exists()

    def test_removes_dependency_repo_via_main_branch_target(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "mylib.git", env)
        project = TestDepSwitch._setup_dep(tmp_path, bare, env)

        run_agm(["dep", "rm", "mylib/main"], env=env, cwd=str(project))

        assert not (project / "deps" / "mylib").exists()

    def test_repo_removal_fails_when_other_worktrees_exist(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "mylib.git", env)
        clone = tmp_path / "tmp-clone"
        _git("clone", str(bare), str(clone), cwd=str(tmp_path), env=env)
        _push_branch(clone, bare, "feat/remove", "remove.txt", env)
        project = TestDepSwitch._setup_dep(tmp_path, bare, env)
        run_agm(["dep", "switch", "mylib", "feat/remove"], env=env, cwd=str(project))

        result = run_agm(["dep", "rm", "mylib/repo"], env=env, cwd=str(project), check=False)

        assert result.returncode != 0
        assert (project / "deps" / "mylib").is_dir()
        output = (result.stdout + result.stderr).lower()
        assert "other worktrees" in output


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

    def test_fetch_from_repo_dir_uses_project_root(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare_main = make_bare_repo(tmp_path / "main.git", env)
        bare_dep = make_bare_repo(tmp_path / "dep.git", env)

        project = _make_project(tmp_path, bare_main, env)
        dep_wt = project / "deps" / "mylib" / "main"
        dep_wt.mkdir(parents=True)
        make_working_repo(dep_wt, bare_dep, env)

        result = run_agm(["fetch"], env=env, cwd=str(project / "repo"))
        assert result.returncode == 0
        assert "Fetching repo" in result.stdout
        assert "Fetching deps/mylib/main" in result.stdout

    def test_no_deps_dir_ok(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = tmp_path / "proj"
        project.mkdir()
        make_working_repo(project / "repo", bare, env)

        result = run_agm(["fetch"], env=env, cwd=str(project))
        assert result.returncode == 0
        assert "Fetching repo" in result.stdout

    def test_fetches_embedded_project_main_repo(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = make_working_repo(tmp_path / "proj", bare, env)

        result = run_agm(["fetch"], env=env, cwd=str(project))

        assert result.returncode == 0
        assert "Fetching ." in result.stdout

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

    def test_creates_tracking_branches_for_main_repo(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        other = make_working_repo(tmp_path / "other", bare, env)
        _push_branch(other, bare, "feat/main-sync", "main-sync.txt", env)

        branches_before = _git("branch", cwd=str(project / "repo"), env=env).stdout
        assert "feat/main-sync" not in branches_before

        run_agm(["fetch"], env=env, cwd=str(project))

        branches = _git("branch", cwd=str(project / "repo"), env=env).stdout
        assert "feat/main-sync" in branches

    def test_creates_tracking_branches_for_dependency_repos(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare_main = make_bare_repo(tmp_path / "main.git", env)
        bare_dep = make_bare_repo(tmp_path / "dep.git", env)

        project = _make_project(tmp_path, bare_main, env)
        dep_wt = project / "deps" / "mylib" / "main"
        dep_wt.mkdir(parents=True)
        make_working_repo(dep_wt, bare_dep, env)

        other = make_working_repo(tmp_path / "dep-other", bare_dep, env)
        _push_branch(other, bare_dep, "feat/dep-sync", "dep-sync.txt", env)

        branches_before = _git("branch", cwd=str(dep_wt), env=env).stdout
        assert "feat/dep-sync" not in branches_before

        run_agm(["fetch"], env=env, cwd=str(project))

        branches = _git("branch", cwd=str(dep_wt), env=env).stdout
        assert "feat/dep-sync" in branches

    def test_fetches_multiple_dependencies(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """fetch picks up all dependencies under deps/."""
        bare_main = make_bare_repo(tmp_path / "main.git", env)
        bare_dep1 = make_bare_repo(tmp_path / "dep1.git", env)
        bare_dep2 = make_bare_repo(tmp_path / "dep2.git", env)

        project = _make_project(tmp_path, bare_main, env)

        dep1_wt = project / "deps" / "dep1" / "main"
        dep1_wt.mkdir(parents=True)
        make_working_repo(dep1_wt, bare_dep1, env)

        dep2_wt = project / "deps" / "dep2" / "main"
        dep2_wt.mkdir(parents=True)
        make_working_repo(dep2_wt, bare_dep2, env)

        result = run_agm(["fetch"], env=env, cwd=str(project))
        assert result.returncode == 0
        assert "Fetching repo" in result.stdout
        assert "Fetching deps/dep1" in result.stdout
        assert "Fetching deps/dep2" in result.stdout

    def test_prunes_non_origin_remote_tracking_refs(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        bare_mirror = make_bare_repo(tmp_path / "mirror.git", env)
        project = _make_project(tmp_path, bare, env)
        repo_dir = project / "repo"

        _git("remote", "add", "mirror", str(bare_mirror), cwd=str(repo_dir), env=env)

        mirror_work = make_working_repo(tmp_path / "mirror-work", bare_mirror, env)
        _push_branch(mirror_work, bare_mirror, "feat/mirror-prune", "mirror.txt", env)

        _git("fetch", "mirror", cwd=str(repo_dir), env=env)
        assert (
            _git(
                "show-ref",
                "--verify",
                "--quiet",
                "refs/remotes/mirror/feat/mirror-prune",
                cwd=str(repo_dir),
                env=env,
                check=False,
            ).returncode
            == 0
        )

        _git(
            "push",
            "origin",
            "--delete",
            "feat/mirror-prune",
            cwd=str(mirror_work),
            env=env,
        )

        run_agm(["fetch"], env=env, cwd=str(project))

        assert (
            _git(
                "show-ref",
                "--verify",
                "--quiet",
                "refs/remotes/mirror/feat/mirror-prune",
                cwd=str(repo_dir),
                env=env,
                check=False,
            ).returncode
            != 0
        )


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
        assert "no-such-branch" in result.stderr.lower() or result.returncode == 128

    def test_init_without_url_creates_structure_only(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        run_agm(["init", "myproj"], env=env, cwd=str(tmp_path))

        proj = tmp_path / "myproj"
        for d in ("repo", "deps", "worktrees", "notes", "config"):
            assert (proj / d).is_dir()
        assert not list((proj / "repo").iterdir())

    def test_init_auto_detects_embedded_layout_for_existing_git_repo(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "proj.git", env)
        project = make_working_repo(tmp_path / "proj", bare, env)

        result = run_agm(["init", "proj"], env=env, cwd=str(tmp_path))

        assert "git repo detected, choosing embedded layout" in result.stdout.lower()
        assert (project / ".agm" / "config").is_dir()
        assert (project / ".agm" / "deps").is_dir()
        assert (project / ".agm" / "notes").is_dir()
        assert (project / ".agm" / "worktrees").is_dir()
        assert not (project / "config").exists()
        assert not (project / "deps").exists()

    def test_init_workspace_overrides_existing_git_repo(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "proj.git", env)
        project = make_working_repo(tmp_path / "proj", bare, env)

        result = run_agm(["init", "--workspace", "proj"], env=env, cwd=str(tmp_path))

        assert "git repo detected" not in result.stdout.lower()
        assert (project / "repo").is_dir()
        assert (project / "config").is_dir()

    def test_init_embedded_with_url_only(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "embedded.git", env)

        run_agm(["init", "--embedded", str(bare)], env=env, cwd=str(tmp_path))

        proj = tmp_path / "embedded"
        assert (proj / ".git").is_dir()
        assert (proj / ".agm").is_dir()
        assert (proj / ".agm" / "config").is_dir()
        assert (proj / ".agm" / "deps").is_dir()
        assert (proj / ".agm" / "notes").is_dir()
        assert (proj / ".agm" / "worktrees").is_dir()
        assert ".agm" in (proj / ".gitignore").read_text().splitlines()
        assert (proj / "README.md").exists()
        assert (proj / "config").exists() is False
        assert (proj / "deps").exists() is False

    def test_init_embedded_without_url_creates_structure_only(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        run_agm(["init", "--embedded", "myproj"], env=env, cwd=str(tmp_path))

        proj = tmp_path / "myproj"
        assert proj.is_dir()
        assert (proj / ".agm").is_dir()
        assert (proj / ".agm" / "deps").is_dir()
        assert (proj / ".agm" / "notes").is_dir()
        assert (proj / ".agm" / "config").is_dir()
        assert (proj / ".agm" / "worktrees").is_dir()
        assert ".agm" in (proj / ".gitignore").read_text().splitlines()
        assert not (proj / "repo").exists()
        assert not (proj / "worktrees").exists()
        assert not (proj / "config").exists()

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
        """Create a fake ``srt`` (Anthropic Sandbox Runtime) binary.

        Real srt interface::

            srt --settings FILE -- COMMAND [ARGS...]

        The real ``srt``:
          1. Reads the JSON settings file specified by ``--settings``.
          2. Creates a sandboxed environment per those settings (bwrap).
          3. Executes ``COMMAND [ARGS...]`` inside the sandbox.
          4. Exits with the command's exit code.

        This mock:
          1. Captures the ``--settings`` file path.
          2. Prints ``SETTINGS:<contents>`` so tests can verify settings
             merging/patching via :func:`_srt_settings`.
          3. Captures the command after the ``--`` separator.
          4. Prints ``COMMAND:<args>`` so tests can verify command
             forwarding via :func:`_srt_command`.
          5. Exits 0.
        """
        directory.mkdir(parents=True, exist_ok=True)
        TestSandbox._make_fake_systemd_run(directory, env)
        srt = directory / "srt"
        srt.write_text(
            "#!/bin/bash\n"
            "# Mock srt — see _make_fake_srt docstring for correspondence\n"
            "# to real srt behaviour.\n"
            'settings_file=""\n'
            'while [[ $# -gt 0 ]]; do\n'
            '  case "$1" in\n'
            "    --settings)\n"
            "      shift\n"
            '      settings_file="$1"\n'
            "      shift\n"
            "      ;;\n"
            "    --)\n"
            "      shift\n"
            "      break\n"
            "      ;;\n"
            "    *) shift ;;\n"
            "  esac\n"
            "done\n"
            'if [[ -n "$settings_file" && -f "$settings_file" ]]; then\n'
            '  echo "SETTINGS:$(cat "$settings_file")"\n'
            "fi\n"
            'if [[ $# -gt 0 ]]; then\n'
            '  echo "COMMAND:$*"\n'
            "fi\n"
            "exit 0\n"
        )
        srt.chmod(srt.stat().st_mode | stat.S_IEXEC)
        env["PATH"] = str(directory) + ":" + env["PATH"]

    @staticmethod
    def _make_fake_systemd_run(directory: Path, env: dict[str, str]) -> None:
        """Create a fake ``systemd-run`` that logs and forwards to the command."""
        directory.mkdir(parents=True, exist_ok=True)
        systemd_run = directory / "systemd-run"
        systemd_run.write_text(
            "#!/bin/bash\n"
            'echo "SYSTEMD_RUN:$*"\n'
            'while [[ $# -gt 0 ]]; do\n'
            '  case "$1" in\n'
            "    --user|--scope)\n"
            "      shift\n"
            "      ;;\n"
            "    -p)\n"
            "      shift 2\n"
            "      ;;\n"
            "    *)\n"
            '      exec "$@"\n'
            "      ;;\n"
            "  esac\n"
            "done\n"
            "exit 0\n"
        )
        systemd_run.chmod(systemd_run.stat().st_mode | stat.S_IEXEC)
        env["PATH"] = str(directory) + ":" + env["PATH"]

    def test_error_when_srt_missing(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        # Build a PATH that has everything except ``srt``.  For each PATH
        # directory that contains ``srt``, symlink all *other* executables
        # into a replacement directory so bash/zsh/scripts are still found.
        dirs = env.get("PATH", "").split(":")
        clean: list[str] = []
        for d in dirs:
            dp = Path(d)
            if (dp / "srt").exists():
                alt = tmp_path / f"nosrt-{dp.name}"
                alt.mkdir(parents=True, exist_ok=True)
                for item in dp.iterdir():
                    if item.name != "srt" and item.is_file() and not (alt / item.name).exists():
                        os.symlink(item, alt / item.name)
                clean.append(str(alt))
            else:
                clean.append(d)
        env["PATH"] = ":".join(clean)

        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(
            ["run", "echo", "hi"], env=env, cwd=str(work), check=False,
        )
        assert result.returncode != 0
        assert "srt is not installed" in result.stderr

    def test_help_when_command_missing(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        result = run_agm(
            ["run"], env=env, cwd=str(tmp_path), check=False,
        )
        assert result.returncode == 0
        assert result.stderr == ""
        assert (
            "agm run [--no-sandbox] [--no-patch] [--memory LIMIT] [-f|--file SETTINGS] "
            "COMMAND [ARGS...]"
            in result.stdout
        )

    def test_no_sandbox_runs_command_without_srt_or_settings(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(
            ["run", "--no-sandbox", "python3", "-c", 'print("hello")'],
            env=env,
            cwd=str(work),
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "hello"
        assert result.stderr == ""

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
        sandbox_dir = home / ".agm" / "sandbox"
        sandbox_dir.mkdir(parents=True)
        settings = {"network": {"allowedDomains": ["example.com"]}}
        (sandbox_dir / "echo.json").write_text(json.dumps(settings))

        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        assert result.returncode == 0
        parsed = _srt_settings(result)
        assert parsed["network"]["allowedDomains"] == ["example.com"]

    def test_uses_local_settings(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        work = tmp_path / "work"
        work.mkdir()
        sandbox_dir = work / ".sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "echo.json").write_text(
            json.dumps(_settings(filesystem=_filesystem_settings(".")))
        )

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        assert result.returncode == 0
        parsed = _srt_settings(result)
        assert parsed["filesystem"]["allowWrite"] == ["."]

    def test_falls_back_to_default_when_command_settings_missing(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        work = tmp_path / "work"
        work.mkdir()
        sandbox_dir = work / ".sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "default.json").write_text(
            json.dumps(_settings(filesystem=_filesystem_settings(".")))
        )

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        assert result.returncode == 0
        parsed = _srt_settings(result)
        assert parsed["filesystem"]["allowWrite"] == ["."]

    def test_uses_proj_dir_sandbox_settings(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        proj_dir = tmp_path / "project"
        (proj_dir / "config" / "sandbox").mkdir(parents=True)
        (proj_dir / "config" / "sandbox" / "echo.json").write_text(
            json.dumps(_settings(network=_network_settings("proj.com")))
        )

        work = tmp_path / "work"
        work.mkdir()
        env["PROJ_DIR"] = str(proj_dir)

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        assert result.returncode == 0
        parsed = _srt_settings(result)
        assert parsed["network"]["allowedDomains"] == ["proj.com"]

    def test_merges_home_and_local_settings(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        home = Path(env["HOME"])
        home_sandbox = home / ".agm" / "sandbox"
        home_sandbox.mkdir(parents=True)
        (home_sandbox / "echo.json").write_text(
            json.dumps(
                _settings(
                    network=_network_settings("home.com"),
                    filesystem=_filesystem_settings("/home"),
                )
            )
        )

        work = tmp_path / "work"
        work.mkdir()
        local_sandbox = work / ".sandbox"
        local_sandbox.mkdir()
        (local_sandbox / "echo.json").write_text(
            json.dumps(_settings(network=_network_settings("local.com")))
        )

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        assert result.returncode == 0

        merged = _srt_settings(result)
        assert merged["network"]["allowedDomains"] == ["local.com"]
        assert merged["filesystem"]["allowWrite"] == ["/home"]

    def test_merges_proj_dir_and_cwd_settings(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        proj_dir = tmp_path / "project"
        (proj_dir / "config" / "sandbox").mkdir(parents=True)
        (proj_dir / "config" / "sandbox" / "echo.json").write_text(
            json.dumps(
                _settings(
                    network=_network_settings("proj.com"),
                    filesystem=_filesystem_settings("/proj"),
                )
            )
        )

        work = tmp_path / "work"
        work.mkdir()
        local_sandbox = work / ".sandbox"
        local_sandbox.mkdir()
        (local_sandbox / "echo.json").write_text(
            json.dumps(_settings(network=_network_settings("local.com")))
        )
        env["PROJ_DIR"] = str(proj_dir)

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        assert result.returncode == 0

        merged = _srt_settings(result)
        assert merged["network"]["allowedDomains"] == ["local.com"]
        assert merged["filesystem"]["allowWrite"] == [
            "/proj",
            str(proj_dir / "notes"),
            str(proj_dir / "deps"),
        ]

    def test_merge_overrides_ignore_violations(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """ignoreViolations in local should completely replace home's."""
        self._make_fake_srt(tmp_path / "bin", env)

        home = Path(env["HOME"])
        home_sandbox = home / ".agm" / "sandbox"
        home_sandbox.mkdir(parents=True)
        (home_sandbox / "echo.json").write_text(
            json.dumps(
                _settings(
                    ignore_violations={"ruleA": True},
                    network=_network_settings(),
                )
            )
        )

        work = tmp_path / "work"
        work.mkdir()
        local_sandbox = work / ".sandbox"
        local_sandbox.mkdir()
        (local_sandbox / "echo.json").write_text(
            json.dumps(_settings(ignore_violations={"ruleB": True}))
        )

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        merged = _srt_settings(result)
        # Local ignoreViolations replaces (not merges with) home.
        assert merged["ignoreViolations"] == {"ruleB": True}

    def test_merge_overrides_enabled_and_weaker_sandbox(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """enabled and enableWeakerNestedSandbox can be overridden by local."""
        self._make_fake_srt(tmp_path / "bin", env)

        home = Path(env["HOME"])
        home_sandbox = home / ".agm" / "sandbox"
        home_sandbox.mkdir(parents=True)
        (home_sandbox / "echo.json").write_text(
            json.dumps(
                _settings(
                    enabled=True,
                    enable_weaker_nested_sandbox=False,
                    network=_network_settings(),
                )
            )
        )

        work = tmp_path / "work"
        work.mkdir()
        local_sandbox = work / ".sandbox"
        local_sandbox.mkdir()
        (local_sandbox / "echo.json").write_text(
            json.dumps(
                _settings(
                    enabled=False,
                    enable_weaker_nested_sandbox=True,
                )
            )
        )

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        merged = _srt_settings(result)
        assert merged["enabled"] is False
        assert merged["enableWeakerNestedSandbox"] is True

    def test_explicit_settings_file(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        work = tmp_path / "work"
        work.mkdir()
        sf = work / "custom.json"
        sf.write_text(json.dumps(_settings(network=_network_settings("custom.com"))))

        result = run_agm(
            ["run", "-f", str(sf), "echo", "hi"], env=env, cwd=str(work),
        )
        assert result.returncode == 0
        merged = _srt_settings(result)
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
        (sandbox_dir / "echo.json").write_text(
            json.dumps(_settings(filesystem=_filesystem_settings(".")))
        )

        env["PROJ_DIR"] = "/some/project/dir"

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        assert result.returncode == 0
        parsed = _srt_settings(result)
        assert "/some/project/dir/notes" in parsed["filesystem"]["allowWrite"]
        assert "/some/project/dir/deps" in parsed["filesystem"]["allowWrite"]
        assert "/some/project/dir" not in parsed["filesystem"]["allowWrite"]

    def test_no_patch_flag(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        work = tmp_path / "work"
        work.mkdir()
        sandbox_dir = work / ".sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "echo.json").write_text(
            json.dumps(_settings(filesystem=_filesystem_settings(".")))
        )

        env["PROJ_DIR"] = str(work)

        result = run_agm(
            ["run", "--no-patch", "echo", "hi"], env=env, cwd=str(work),
        )
        assert result.returncode == 0
        parsed = _srt_settings(result)
        assert str(work) not in parsed["filesystem"]["allowWrite"]
        assert str(work / "notes") not in parsed["filesystem"]["allowWrite"]
        assert str(work / "deps") not in parsed["filesystem"]["allowWrite"]

    def test_invalid_option(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        result = run_agm(
            ["run", "--bad-opt", "echo"],
            env=env, cwd=str(tmp_path), check=False,
        )
        assert result.returncode != 0
        assert "unrecognized" in result.stderr.lower() or "bad-opt" in result.stderr

    def test_f_and_no_patch_together(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        work = tmp_path / "work"
        work.mkdir()
        sf = work / "custom.json"
        sf.write_text(json.dumps(_settings(filesystem=_filesystem_settings("/x"))))

        env["PROJ_DIR"] = "/proj"

        result = run_agm(
            ["run", "--no-patch", "-f", str(sf), "echo"],
            env=env, cwd=str(work),
        )
        assert result.returncode == 0
        parsed = _srt_settings(result)
        assert parsed["filesystem"]["allowWrite"] == ["/x"]
        assert "/proj/notes" not in parsed["filesystem"]["allowWrite"]
        assert "/proj/deps" not in parsed["filesystem"]["allowWrite"]

    def test_run_forwards_command_to_srt(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """The command after agm run options is forwarded to srt."""
        self._make_fake_systemd_run(tmp_path / "bin", env)
        self._make_fake_srt(tmp_path / "bin", env)

        work = tmp_path / "work"
        work.mkdir()
        sandbox_dir = work / ".sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "npm.json").write_text(json.dumps(_settings(enabled=True)))

        result = run_agm(
            ["run", "npm", "test", "--coverage"],
            env=env, cwd=str(work),
        )
        assert result.returncode == 0
        assert _systemd_run_command(result) == (
            f"--user --scope -p MemoryMax=20G srt --settings {work / '.sandbox' / 'npm.json'} "
            "-- npm test --coverage"
        )
        assert _srt_command(result) == "npm test --coverage"

    def test_run_memory_flag_overrides_config(self, tmp_path: Path, env: dict[str, str]) -> None:
        self._make_fake_systemd_run(tmp_path / "bin", env)
        self._make_fake_srt(tmp_path / "bin", env)

        home = Path(env["HOME"])
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[run]\nmemory = "10G"\n[run.echo]\nmemory = "5G"\n'
        )

        work = tmp_path / "work"
        work.mkdir()
        sandbox_dir = work / ".sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "echo.json").write_text(json.dumps(_settings(enabled=True)))

        result = run_agm(["run", "--memory", "2G", "echo", "hi"], env=env, cwd=str(work))
        assert result.returncode == 0
        assert _systemd_run_command(result) == (
            f"--user --scope -p MemoryMax=2G srt --settings {work / '.sandbox' / 'echo.json'} "
            "-- echo hi"
        )

    def test_run_command_memory_overrides_run_default(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_systemd_run(tmp_path / "bin", env)
        self._make_fake_srt(tmp_path / "bin", env)

        home = Path(env["HOME"])
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[run]\nmemory = "10G"\n[run.echo]\nmemory = "5G"\n'
        )

        work = tmp_path / "work"
        work.mkdir()
        sandbox_dir = work / ".sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "echo.json").write_text(json.dumps(_settings(enabled=True)))

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        assert result.returncode == 0
        assert _systemd_run_command(result) == (
            f"--user --scope -p MemoryMax=5G srt --settings {work / '.sandbox' / 'echo.json'} "
            "-- echo hi"
        )

    def test_run_does_not_wrap_when_memory_limit_is_zero_or_less(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_systemd_run(tmp_path / "bin", env)
        self._make_fake_srt(tmp_path / "bin", env)

        home = Path(env["HOME"])
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[run]\nmemory = "0"\n')

        work = tmp_path / "work"
        work.mkdir()
        sandbox_dir = work / ".sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "echo.json").write_text(json.dumps(_settings(enabled=True)))

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        assert result.returncode == 0
        assert _systemd_run_command(result) == ""
        assert _srt_command(result) == "echo hi"

    def test_run_with_double_dash_separator(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """agm run -- command works with an explicit separator."""
        self._make_fake_srt(tmp_path / "bin", env)

        work = tmp_path / "work"
        work.mkdir()
        sandbox_dir = work / ".sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "echo.json").write_text(json.dumps(_settings(enabled=True)))

        result = run_agm(
            ["run", "--", "echo", "hello"],
            env=env, cwd=str(work),
        )
        assert result.returncode == 0
        assert _srt_command(result) == "echo hello"

    def test_prefers_command_settings_over_default_at_same_level(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        work = tmp_path / "work"
        work.mkdir()
        sandbox_dir = work / ".sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "default.json").write_text(
            json.dumps(_settings(network=_network_settings("default.com")))
        )
        (sandbox_dir / "echo.json").write_text(
            json.dumps(_settings(network=_network_settings("echo.com")))
        )

        result = run_agm(["run", "echo", "hi"], env=env, cwd=str(work))
        assert result.returncode == 0
        parsed = _srt_settings(result)
        assert parsed["network"]["allowedDomains"] == ["echo.com"]

    def test_uses_command_basename_for_settings_lookup(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        work = tmp_path / "work"
        work.mkdir()
        sandbox_dir = work / ".sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "echo.json").write_text(json.dumps(_settings(enabled=True)))

        result = run_agm(
            ["run", "/bin/echo", "hello"], env=env, cwd=str(work),
        )
        assert result.returncode == 0
        assert _srt_command(result) == "/bin/echo hello"

    def test_run_uses_global_alias_for_command_execution(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        home = Path(env["HOME"])
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[run.echo]\nalias = "printf"\n')
        (home / ".agm" / "sandbox").mkdir(parents=True)
        (home / ".agm" / "sandbox" / "printf.json").write_text(
            json.dumps(_settings(network=_network_settings("printf.com")))
        )

        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(["run", "echo", "hello"], env=env, cwd=str(work))
        assert result.returncode == 0
        assert _srt_command(result) == "printf hello"
        assert _srt_settings(result)["network"]["allowedDomains"] == ["printf.com"]

    def test_local_run_config_overrides_global_alias(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        home = Path(env["HOME"])
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[run.echo]\nalias = "printf"\n')
        (home / ".agm" / "sandbox").mkdir(parents=True)
        (home / ".agm" / "sandbox" / "printf.json").write_text(
            json.dumps(_settings(network=_network_settings("printf.com")))
        )

        proj_dir = tmp_path / "project"
        (proj_dir / "config").mkdir(parents=True)
        (proj_dir / "config" / "config.toml").write_text('[run.echo]\nalias = "cat"\n')
        (proj_dir / "config" / "sandbox").mkdir(parents=True)
        (proj_dir / "config" / "sandbox" / "cat.json").write_text(
            json.dumps(_settings(network=_network_settings("cat.com")))
        )

        work = tmp_path / "work"
        work.mkdir()
        env["PROJ_DIR"] = str(proj_dir)

        result = run_agm(["run", "echo", "hello"], env=env, cwd=str(work))
        assert result.returncode == 0
        assert _srt_command(result) == "cat hello"
        assert _srt_settings(result)["network"]["allowedDomains"] == ["cat.com"]

    def test_prefers_original_command_settings_before_alias_settings(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        self._make_fake_srt(tmp_path / "bin", env)

        home = Path(env["HOME"])
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[run.echo]\nalias = "printf"\n')

        work = tmp_path / "work"
        work.mkdir()
        sandbox_dir = work / ".sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "echo.json").write_text(
            json.dumps(_settings(network=_network_settings("echo.com")))
        )
        (sandbox_dir / "printf.json").write_text(
            json.dumps(_settings(network=_network_settings("printf.com")))
        )

        result = run_agm(["run", "echo", "hello"], env=env, cwd=str(work))
        assert result.returncode == 0
        assert _srt_command(result) == "printf hello"
        assert _srt_settings(result)["network"]["allowedDomains"] == ["echo.com"]


class TestLoop:
    """agm loop: repeated Claude execution until COMPLETE."""

    def test_runs_loop_prompt_until_complete(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        _install_fake_claude(tmp_path / "bin", env)
        env["FAKE_CLAUDE_STATE"] = str(tmp_path / "claude-count")
        env["FAKE_CLAUDE_LOG"] = str(tmp_path / "claude.log")

        home = Path(env["HOME"])
        prompt_dir = home / ".agm" / "prompts"
        prompt_dir.mkdir(parents=True)
        prompt_file = prompt_dir / "loop.md"
        prompt_file.write_text("loop prompt\n")

        work = tmp_path / "work"
        work.mkdir()
        (work / ".agent-files" / "tasks").mkdir(parents=True)
        (work / ".agent-files" / "tasks" / "PROGRESS.md").write_text("started\n")

        result = run_agm(["loop"], env=env, cwd=str(work))

        assert result.returncode == 0
        assert "Logging to loop-" in result.stdout
        assert "Step 1" in result.stdout
        assert "Step 2" in result.stdout
        assert "Completed." in result.stdout

        log_file = next(work.glob("loop-*.log"))
        assert log_file.read_text() == (
            "\n"
            "-------------------------------------------------------------\n"
            "                        Step 1\n"
            "-------------------------------------------------------------\n"
            "\n"
            "keep going\n"
            "\n"
            "-------------------------------------------------------------\n"
            "                        Step 2\n"
            "-------------------------------------------------------------\n"
            "\n"
            "  COMPLETE  \n"
        )
        assert Path(env["FAKE_CLAUDE_LOG"]).read_text().splitlines() == [f"-p @{prompt_file}"] * 2

    def test_uses_prompt_from_agm_install_prefix(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        _install_fake_claude(tmp_path / "bin", env)
        env["FAKE_CLAUDE_STATE"] = str(tmp_path / "claude-count")
        env["FAKE_CLAUDE_LOG"] = str(tmp_path / "claude.log")

        prefix = tmp_path / "prefix"
        (prefix / "bin").mkdir(parents=True)
        agm = prefix / "bin" / "agm"
        agm.write_text("#!/bin/bash\n")
        agm.chmod(agm.stat().st_mode | stat.S_IEXEC)
        env["PATH"] = f"{prefix / 'bin'}:{env['PATH']}"

        prompt_dir = prefix / ".agm" / "prompts"
        prompt_dir.mkdir(parents=True)
        prompt_file = prompt_dir / "loop.md"
        prompt_file.write_text("loop prompt\n")

        work = tmp_path / "work"
        work.mkdir()
        (work / ".agent-files" / "tasks").mkdir(parents=True)
        (work / ".agent-files" / "tasks" / "PROGRESS.md").write_text("started\n")

        result = run_agm(["loop"], env=env, cwd=str(work))

        assert result.returncode == 0
        assert Path(env["FAKE_CLAUDE_LOG"]).read_text().splitlines() == [f"-p @{prompt_file}"] * 2

    def test_bootstraps_progress_prompt_when_progress_file_is_missing(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        _install_fake_claude(tmp_path / "bin", env)
        env["FAKE_CLAUDE_STATE"] = str(tmp_path / "claude-count")
        env["FAKE_CLAUDE_LOG"] = str(tmp_path / "claude.log")

        prefix = tmp_path / "prefix"
        (prefix / "bin").mkdir(parents=True)
        agm = prefix / "bin" / "agm"
        agm.write_text("#!/bin/bash\n")
        agm.chmod(agm.stat().st_mode | stat.S_IEXEC)
        env["PATH"] = f"{prefix / 'bin'}:{env['PATH']}"

        prompt_dir = prefix / ".agm" / "prompts"
        prompt_dir.mkdir(parents=True)
        bootstrap_prompt = prompt_dir / "update_progress.md"
        bootstrap_prompt.write_text("update progress\n")
        prompt_file = prompt_dir / "loop.md"
        prompt_file.write_text("loop prompt\n")

        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(["loop"], env=env, cwd=str(work))

        assert result.returncode == 0
        assert "Step 1" in result.stdout
        assert "Step 2" not in result.stdout
        log_file = next(work.glob("loop-*.log"))
        assert log_file.read_text() == (
            "\n"
            "-------------------------------------------------------------\n"
            "                        Step 1\n"
            "-------------------------------------------------------------\n"
            "\n"
            "  COMPLETE  \n"
        )
        assert Path(env["FAKE_CLAUDE_LOG"]).read_text().splitlines() == [
            f"-p @{bootstrap_prompt}",
            f"-p @{prompt_file}",
        ]

    def test_uses_configured_loop_command(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        _install_fake_claude(tmp_path / "bin", env)
        env["FAKE_CLAUDE_STATE"] = str(tmp_path / "claude-count")
        env["FAKE_CLAUDE_LOG"] = str(tmp_path / "claude.log")

        home = Path(env["HOME"])
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[loop]\ncommand = "claude --print"\n')
        prompt_dir = home / ".agm" / "prompts"
        prompt_dir.mkdir(parents=True)
        prompt_file = prompt_dir / "loop.md"
        prompt_file.write_text("loop prompt\n")

        work = tmp_path / "work"
        work.mkdir()
        (work / ".agent-files" / "tasks").mkdir(parents=True)
        (work / ".agent-files" / "tasks" / "PROGRESS.md").write_text("started\n")

        result = run_agm(["loop"], env=env, cwd=str(work))

        assert result.returncode == 0
        assert Path(env["FAKE_CLAUDE_LOG"]).read_text().splitlines() == [
            f"--print @{prompt_file}"
        ] * 2

    def test_cli_loop_command_overrides_configured_loop_command(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        _install_fake_claude(tmp_path / "bin", env)
        env["FAKE_CLAUDE_STATE"] = str(tmp_path / "claude-count")
        env["FAKE_CLAUDE_LOG"] = str(tmp_path / "claude.log")

        home = Path(env["HOME"])
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[loop]\ncommand = "claude --print"\n')
        prompt_dir = home / ".agm" / "prompts"
        prompt_dir.mkdir(parents=True)
        prompt_file = prompt_dir / "loop.md"
        prompt_file.write_text("loop prompt\n")

        work = tmp_path / "work"
        work.mkdir()
        (work / ".agent-files" / "tasks").mkdir(parents=True)
        (work / ".agent-files" / "tasks" / "PROGRESS.md").write_text("started\n")

        result = run_agm(["loop", "-c", "claude --stream"], env=env, cwd=str(work))

        assert result.returncode == 0
        assert Path(env["FAKE_CLAUDE_LOG"]).read_text().splitlines() == [
            f"--stream @{prompt_file}"
        ] * 2

    def test_uses_configured_loop_tasks_dir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        _install_fake_claude(tmp_path / "bin", env)
        env["FAKE_CLAUDE_STATE"] = str(tmp_path / "claude-count")
        env["FAKE_CLAUDE_LOG"] = str(tmp_path / "claude.log")

        home = Path(env["HOME"])
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[loop]\ntasks_dir = "custom/tasks"\n')
        prompt_dir = home / ".agm" / "prompts"
        prompt_dir.mkdir(parents=True)
        bootstrap_prompt = prompt_dir / "update_progress.md"
        bootstrap_prompt.write_text("update progress\n")
        prompt_file = prompt_dir / "loop.md"
        prompt_file.write_text("loop prompt\n")

        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(["loop"], env=env, cwd=str(work))

        assert result.returncode == 0
        assert Path(env["FAKE_CLAUDE_LOG"]).read_text().splitlines() == [
            f"-p @{bootstrap_prompt}",
            f"-p @{prompt_file}",
        ]

    def test_cli_loop_tasks_dir_overrides_configured_tasks_dir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        _install_fake_claude(tmp_path / "bin", env)
        env["FAKE_CLAUDE_STATE"] = str(tmp_path / "claude-count")
        env["FAKE_CLAUDE_LOG"] = str(tmp_path / "claude.log")

        home = Path(env["HOME"])
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[loop]\ntasks_dir = "custom/tasks"\n')
        prompt_dir = home / ".agm" / "prompts"
        prompt_dir.mkdir(parents=True)
        prompt_file = prompt_dir / "loop.md"
        prompt_file.write_text("loop prompt\n")

        work = tmp_path / "work"
        work.mkdir()
        (work / "cli" / "tasks").mkdir(parents=True)
        (work / "cli" / "tasks" / "PROGRESS.md").write_text("started\n")

        result = run_agm(["loop", "--tasks-dir", "cli/tasks"], env=env, cwd=str(work))

        assert result.returncode == 0
        assert Path(env["FAKE_CLAUDE_LOG"]).read_text().splitlines() == [
            f"-p @{prompt_file}"
        ] * 2


# ── agm open ────────────────────────────────────────────────────────────────


@needs_zsh
class TestOpen:
    """agm open: project session management."""

    def test_open_repo_target(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env, name="myproj")
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(["open", "repo"], env=env, cwd=str(project))

        log = tmux_log.read_text()
        assert "new-session" in log
        assert "-s myproj" in log

    def test_open_default_branch_opens_repo_session(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env, name="myproj")
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(["open", "main"], env=env, cwd=str(project))

        log = tmux_log.read_text()
        assert "new-session" in log
        assert "-s myproj" in log
        assert not (project / "worktrees" / "main").exists()

    def test_open_existing_branch_checks_out_worktree(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env, name="proj")
        setup = project / "config" / "setup.sh"
        setup.write_text('#!/bin/bash\ntouch "$PWD/.setup-ran"\n')
        setup.chmod(setup.stat().st_mode | stat.S_IEXEC)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        clone = tmp_path / "tmp-clone"
        _git("clone", str(bare), str(clone), cwd=str(tmp_path), env=env)
        _push_branch(clone, bare, "feat/x", "x.txt", env)

        result = run_agm(["open", "feat/x"], env=env, cwd=str(project))

        log = tmux_log.read_text()
        assert "new-session" in log
        assert "attach-session" in log
        assert "send-keys" in log
        assert "-s proj/feat/x" in log
        assert (project / "worktrees" / "feat/x" / "x.txt").exists()
        assert (project / "worktrees" / "feat/x").is_dir()
        _wait_for_path(project / "worktrees" / "feat/x" / ".setup-ran")
        assert "Detached tmux session proj/feat/x created" in result.stdout
        assert "true" not in result.stdout

    def test_open_existing_worktree_in_embedded_project(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = make_working_repo(tmp_path / "proj", bare, env)
        (project / ".agm").mkdir()
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        wt = project / ".agm" / "worktrees" / "feat/x"
        wt.parent.mkdir(parents=True, exist_ok=True)
        _git("worktree", "add", "-b", "feat/x", str(wt), cwd=str(project), env=env)

        run_agm(["open", "feat/x"], env=env, cwd=str(project))

        log = tmux_log.read_text()
        assert "new-session" in log
        assert "-s proj/feat/x" in log
        assert wt.is_dir()

    def test_open_with_pane_count(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(["open", "-n", "6", "repo"], env=env, cwd=str(project))

        log = tmux_log.read_text()
        assert log.count("split-window") == 5  # 6 panes = 5 splits

    def test_open_missing_branch_creates_worktree_and_detached_session(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env, name="proj")
        setup = project / "config" / "setup.sh"
        setup.write_text('#!/bin/bash\ntouch "$PWD/.setup-ran"\n')
        setup.chmod(setup.stat().st_mode | stat.S_IEXEC)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        result = run_agm(["open", "feat/test"], env=env, cwd=str(project))

        assert (project / "worktrees" / "feat/test").is_dir()
        assert (project / "worktrees" / "feat/test" / "README.md").exists()
        log = tmux_log.read_text()
        assert "-dP" in log
        assert "attach-session" in log
        assert "send-keys" in log
        assert "-s proj/feat/test" in log
        _wait_for_path(project / "worktrees" / "feat/test" / ".setup-ran")
        assert "Detached tmux session proj/feat/test created" in result.stdout
        assert "true" not in result.stdout
        assert "cannot stat" not in result.stderr

    def test_open_missing_branch_detached_keeps_current_session(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env, name="proj")
        setup = project / "config" / "setup.sh"
        setup.write_text('#!/bin/bash\ntouch "$PWD/.setup-ran"\n')
        setup.chmod(setup.stat().st_mode | stat.S_IEXEC)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        result = run_agm(["open", "-d", "feat/detached"], env=env, cwd=str(project))

        assert (project / "worktrees" / "feat/detached").is_dir()
        _wait_for_path(project / "worktrees" / "feat/detached" / ".setup-ran")
        log = tmux_log.read_text()
        assert "-dP" in log
        assert "send-keys" in log
        assert "attach-session" not in log
        assert "switch-client" not in log
        assert "Detached tmux session proj/feat/detached created" in result.stdout

    def test_open_missing_branch_with_pane_count(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(["open", "-n", "2", "feat/n"], env=env, cwd=str(project))

        log = tmux_log.read_text()
        assert log.count("split-window") == 1  # 2 panes = 1 split

    def test_open_missing_branch_with_parent(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        _git(
            "worktree", "add", "-b", "develop",
            str(project / "worktrees" / "develop"),
            cwd=str(project / "repo"), env=env,
        )
        (project / "worktrees" / "develop" / "dev.txt").write_text("dev")
        _git("add", ".", cwd=str(project / "worktrees" / "develop"), env=env)
        _git("commit", "-m", "dev", cwd=str(project / "worktrees" / "develop"), env=env)

        run_agm(
            ["open", "-p", "develop", "feat/from-dev"],
            env=env, cwd=str(project),
        )

        wt = project / "worktrees" / "feat/from-dev"
        assert wt.is_dir()
        assert (wt / "dev.txt").exists()

    def test_open_existing_branch_with_pane_count(
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
            ["open", "-n", "3", "feat/p"], env=env, cwd=str(project),
        )

        log = tmux_log.read_text()
        assert log.count("split-window") == 2  # 3 panes

    def test_open_existing_branch_with_parent(
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
            ["open", "-p", "main", "feat/q"], env=env, cwd=str(project),
        )

        assert (project / "worktrees" / "feat/q" / "q.txt").exists()

    def test_error_open_without_target(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        result = run_agm(
            ["open"], env=env, cwd=str(project), check=False,
        )
        assert result.returncode != 0
        assert "target" in result.stderr.lower() or "branch" in result.stderr.lower()

    def test_invalid_pane_count(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        result = run_agm(
            ["open", "-n", "abc", "repo"], env=env, cwd=str(project), check=False,
        )
        assert result.returncode != 0
        assert "pane count must be a positive integer" in result.stderr

    def test_no_subcommand_shows_help(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        result = run_agm(
            [], env=env, cwd=str(tmp_path), check=False,
        )
        expected = run_agm(["help"], env=env, cwd=str(tmp_path))
        assert result.returncode == 0
        assert result.stderr == ""
        assert result.stdout == expected.stdout

    def test_sources_project_env_file(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env, name="proj")
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        (project / "config" / "env.sh").write_text(
            f'touch "{project}/env-sourced"\n'
        )

        run_agm(["open", "repo"], env=env, cwd=str(project))

        assert (project / "env-sourced").exists()

    def test_sources_branch_env_file(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env, name="proj")
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        branch_cfg = project / "config" / "mybranch"
        branch_cfg.mkdir(parents=True)
        (branch_cfg / "env.sh").write_text(
            f'touch "{project}/branch-env-sourced"\n'
        )

        clone = tmp_path / "tmp-clone"
        _git("clone", str(bare), str(clone), cwd=str(tmp_path), env=env)
        _push_branch(clone, bare, "mybranch", "branch.txt", env)

        run_agm(["open", "mybranch"], env=env, cwd=str(project))

        assert (project / "branch-env-sourced").exists()

    def test_init_then_open_lifecycle(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        bare = make_bare_repo(tmp_path / "origin.git", env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(["init", "myproj", str(bare)], env=env, cwd=str(tmp_path))
        project = tmp_path / "myproj"

        run_agm(["open", "feat/lifecycle"], env=env, cwd=str(project))

        wt = project / "worktrees" / "feat/lifecycle"
        assert wt.is_dir()
        assert (wt / "README.md").exists()
        head = _git(
            "rev-parse", "--abbrev-ref", "HEAD", cwd=str(wt), env=env,
        ).stdout.strip()
        assert head == "feat/lifecycle"
        log = tmux_log.read_text()
        assert "-dP" in log
        assert "myproj/feat/lifecycle" in log


# ── agm tmux open/close ─────────────────────────────────────────────────────


@needs_zsh
class TestTmuxOpenSession:
    """agm tmux open: create tmux sessions with tiled pane layout."""

    def test_creates_session_default_panes(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        work = tmp_path / "work"
        work.mkdir()

        run_agm(["tmux", "open"], env=env, cwd=str(work))

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

        run_agm(["tmux", "open", "-n", "6"], env=env, cwd=str(work))

        log = tmux_log.read_text()
        assert log.count("split-window") == 5

    def test_single_pane(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)
        work = tmp_path / "work"
        work.mkdir()

        run_agm(["tmux", "open", "-n", "1"], env=env, cwd=str(work))

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

        result = run_agm(["tmux", "open", "-d", "my-session"], env=env, cwd=str(work))

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

        result = run_agm(["tmux", "open", "--detach"], env=env, cwd=str(work))

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

        run_agm(["tmux", "open", "my-custom-name"], env=env, cwd=str(work))

        log = tmux_log.read_text()
        assert "-s my-custom-name" in log

    def test_non_detached_open_uses_user_facing_layout_interface(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)
        work = tmp_path / "work"
        work.mkdir()

        run_agm(["tmux", "open", "-n", "4"], env=env, cwd=str(work))

        log = tmux_log.read_text()
        assert "run-shell" in log
        assert "agm.cli tmux layout 4" in log
        assert "#{window_id}" not in log
        assert "#{window_width}" not in log
        assert "#{window_height}" not in log

    def test_detach_with_custom_pane_count_and_name(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)
        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(
            ["tmux", "open", "-d", "-n", "3", "sess"], env=env, cwd=str(work),
        )

        log = tmux_log.read_text()
        assert "-dP" in log
        assert "-s sess" in log
        assert log.count("split-window") == 2
        assert "Detached" in result.stdout


@needs_zsh
class TestTmuxCloseSession:
    """agm tmux close: kill tmux sessions by name."""

    def test_kills_named_session(self, tmp_path: Path, env: dict[str, str]) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)
        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(["tmux", "close", "my-session"], env=env, cwd=str(work))

        assert "Closed session my-session" in result.stdout
        log = tmux_log.read_text()
        assert "kill-session -t my-session" in log


@needs_zsh
class TestTmuxOpenErrors:
    """agm tmux open: argument validation."""

    def test_invalid_pane_count(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)
        work = tmp_path / "work"
        work.mkdir()

        result = run_agm(
            ["tmux", "open", "-n", "abc"], env=env, cwd=str(work), check=False,
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
            ["tmux", "open", "a", "b"], env=env, cwd=str(work), check=False,
        )
        assert result.returncode != 0
        assert "unexpected extra argument" in result.stderr.lower()


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
            ["tmux", "layout", "1"],
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
            ["tmux", "layout", "4", "--window", "@1"],
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
            ["tmux", "layout", "9"],
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
            ["tmux", "layout", "2"],
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
            ["tmux", "layout", "4"],
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
            ["tmux", "layout"], env=env, cwd=str(tmp_path), check=False,
        )
        assert result.returncode != 0
        assert "usage" in result.stderr.lower() or "argument" in result.stderr.lower()

    def test_five_pane_layout(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """Non-square pane count: 5 panes → 2 rows (3+2)."""
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(
            ["tmux", "layout", "5"],
            env=env, cwd=str(tmp_path),
        )

        log = tmux_log.read_text()
        assert "select-layout" in log
        for idx in range(5):
            assert f",{idx}" in log

    def test_seven_pane_layout(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """Non-square pane count: 7 panes → 2 rows (4+3)."""
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(
            ["tmux", "layout", "7"],
            env=env, cwd=str(tmp_path),
        )

        log = tmux_log.read_text()
        assert "select-layout" in log
        for idx in range(7):
            assert f",{idx}" in log

    def test_queries_current_window_dimensions(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(
            ["tmux", "layout", "4"],
            env=env, cwd=str(tmp_path),
        )

        log = tmux_log.read_text()
        assert "display-message -p #{window_id}" in log
        assert "display-message -p -t @0 #{window_width}" in log
        assert "display-message -p -t @0 #{window_height}" in log


# ── help system ────────────────────────────────────────────────────────────


class TestHelp:
    """agm help: overview and per-command help."""

    def test_overview_lists_all_commands(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        result = run_agm(["help"], env=env, cwd=str(tmp_path))
        assert result.returncode == 0
        assert "agm - Agent Management Framework" in result.stdout
        assert "Commands:" in result.stdout
        assert "--install-completion" in result.stdout
        assert "--show-completion" in result.stdout
        for cmd in ("open", "init", "fetch",
                     "close", "config", "worktree", "dep", "run",
                     "tmux", "help"):
            assert cmd in result.stdout, f"'{cmd}' missing from overview"

    def test_help_for_each_canonical_command(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """Every canonical command has a detailed help entry."""
        for cmd in ("open", "init", "fetch",
                     "close", "config", "worktree", "dep", "run",
                     "tmux", "help"):
            result = run_agm(["help", cmd], env=env, cwd=str(tmp_path))
            assert result.returncode == 0, f"help {cmd} failed"
            assert f"agm {cmd}" in result.stdout, f"help {cmd} missing header"

    def test_help_help_mentions_completion_options(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        result = run_agm(["help", "help"], env=env, cwd=str(tmp_path))
        assert result.returncode == 0
        assert "--install-completion" in result.stdout
        assert "--show-completion" in result.stdout

    def test_run_help_describes_options_and_settings_merge(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        result = run_agm(["help", "run"], env=env, cwd=str(tmp_path))
        assert result.returncode == 0
        assert "<install-prefix>/.agm/config.toml" in result.stdout
        assert "otherwise $HOME/.agm/config.toml is used" in result.stdout
        assert "project config.toml and ./.agm/config.toml" in result.stdout
        assert '[run.<command>] alias = "<other-command>"' in result.stdout
        assert '[run] memory = "20G"' in result.stdout or "The default is 20G" in result.stdout
        assert "--memory LIMIT" in result.stdout
        assert "MemoryMax=LIMIT" in result.stdout
        assert "-f, --file SETTINGS" in result.stdout
        assert "Use this settings file directly" in result.stdout
        assert "--no-patch" in result.stdout
        assert "Do not append the project notes and deps directories" in result.stdout
        assert "$HOME/.agm/sandbox/<command>.json" in result.stdout
        assert "$HOME/.agm/sandbox/default.json" in result.stdout
        assert "the project sandbox config directory" in result.stdout
        assert "./.sandbox/<command>.json" in result.stdout
        assert "./.sandbox/default.json" in result.stdout
        assert "try the aliased command's" in result.stdout
        assert "Later files override earlier ones." in result.stdout
        assert "agm adds the project notes and deps" in result.stdout

    def test_help_aliases_resolve(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """Aliases show help for the canonical command."""
        alias_map = {"wt": "worktree"}
        for alias, canonical in alias_map.items():
            result = run_agm(["help", alias], env=env, cwd=str(tmp_path))
            assert result.returncode == 0
            assert f"agm {canonical}" in result.stdout

    @pytest.mark.parametrize(
        ("argv", "help_argv", "expected_text"),
        [
            (["help", "wt", "new"], ["wt", "new", "-h"], "agm wt new"),
            (["help", "worktree", "remove"], ["worktree", "remove", "-h"], "agm worktree remove"),
            (["help", "config", "cp"], ["config", "cp", "-h"], "agm config cp"),
            (["help", "dep", "switch"], ["dep", "switch", "-h"], "agm dep switch"),
            (["help", "tmux", "layout"], ["tmux", "layout", "-h"], "agm tmux layout"),
        ],
    )
    def test_help_for_subcommands(
        self,
        argv: list[str],
        help_argv: list[str],
        expected_text: str,
        tmp_path: Path,
        env: dict[str, str],
    ) -> None:
        result = run_agm(argv, env=env, cwd=str(tmp_path))
        expected = run_agm(help_argv, env=env, cwd=str(tmp_path))

        assert result.returncode == 0
        assert expected_text in result.stdout
        assert result.stdout == expected.stdout
        assert result.stderr == expected.stderr

    @pytest.mark.parametrize(
        ("argv", "help_argv"),
        [
            (["-h"], ["help"]),
            (["help", "-h"], ["help", "help"]),
            (["open", "-h"], ["help", "open"]),
            (["close", "-h"], ["help", "close"]),
            (["init", "-h"], ["help", "init"]),
            (["fetch", "-h"], ["help", "fetch"]),
            (["config", "-h"], ["help", "config"]),
            (["config", "cp", "-h"], ["help", "config", "cp"]),
            (["wt", "-h"], ["help", "wt"]),
            (["wt", "new", "-h"], ["help", "wt", "new"]),
            (["worktree", "-h"], ["help", "worktree"]),
            (["worktree", "remove", "-h"], ["help", "worktree", "remove"]),
            (["dep", "-h"], ["help", "dep"]),
            (["dep", "switch", "-h"], ["help", "dep", "switch"]),
            (["run", "-h"], ["help", "run"]),
            (["tmux", "-h"], ["help", "tmux"]),
            (["tmux", "layout", "-h"], ["help", "tmux", "layout"]),
        ],
    )
    def test_help_flag_matches_help_command(
        self, argv: list[str], help_argv: list[str], tmp_path: Path, env: dict[str, str]
    ) -> None:
        result = run_agm(argv, env=env, cwd=str(tmp_path))
        expected = run_agm(help_argv, env=env, cwd=str(tmp_path))
        assert result.returncode == 0
        assert result.stdout == expected.stdout
        assert result.stderr == expected.stderr

    def test_help_unknown_command(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        result = run_agm(
            ["help", "bogus"], env=env, cwd=str(tmp_path), check=False,
        )
        assert result.returncode == 1
        assert "unknown command" in result.stderr
        assert "bogus" in result.stderr

    @pytest.mark.parametrize(
        ("argv", "help_argv"),
        [
            (["config"], ["help", "config"]),
            (["run"], ["help", "run"]),
            (["run", "--"], ["help", "run"]),
            (["wt"], ["help", "wt"]),
            (["worktree"], ["help", "worktree"]),
            (["tmux"], ["help", "tmux"]),
        ],
    )
    def test_missing_subcommand_matches_help(
        self, argv: list[str], help_argv: list[str], tmp_path: Path, env: dict[str, str]
    ) -> None:
        result = run_agm(argv, env=env, cwd=str(tmp_path), check=False)
        expected = run_agm(help_argv, env=env, cwd=str(tmp_path))
        assert result.returncode == 0
        assert result.stderr == ""
        assert result.stdout == expected.stdout

    @pytest.mark.parametrize(
        ("argv", "help_argv", "error_text"),
        [
            (["init"], ["init", "-h"], "error: the following arguments are required"),
            (
                ["config", "cp"],
                ["config", "cp", "-h"],
                "error: the following arguments are required",
            ),
            (
                ["open", "-n", "abc", "repo"],
                ["open", "-h"],
                "error: pane count must be a positive integer",
            ),
            (["tmux", "open", "-n", "abc"], ["tmux", "open", "-h"], "invalid pane count"),
        ],
    )
    def test_incorrect_usage_includes_full_help(
        self,
        argv: list[str],
        help_argv: list[str],
        error_text: str,
        tmp_path: Path,
        env: dict[str, str],
    ) -> None:
        result = run_agm(argv, env=env, cwd=str(tmp_path), check=False)
        expected = run_agm(help_argv, env=env, cwd=str(tmp_path))

        assert result.returncode != 0
        assert error_text in result.stderr.lower()
        assert expected.stdout in result.stderr


# ── edge cases ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases across various commands."""

    def test_branch_name_with_slashes(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """Branch names with nested slashes (feat/auth/login) work."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        run_agm(
            ["wt", "new", "feat/auth/login"],
            env=env, cwd=str(project / "repo"),
        )

        wt = project / "worktrees" / "feat/auth/login"
        assert wt.is_dir()
        assert (wt / "README.md").exists()
        head = _git(
            "rev-parse", "--abbrev-ref", "HEAD", cwd=str(wt), env=env,
        ).stdout.strip()
        assert head == "feat/auth/login"

    def test_branch_name_with_dots(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """Branch names with dots (release/v2.1.0) work."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        run_agm(
            ["wt", "new", "release/v2.1.0"],
            env=env, cwd=str(project / "repo"),
        )

        wt = project / "worktrees" / "release/v2.1.0"
        assert wt.is_dir()
        head = _git(
            "rev-parse", "--abbrev-ref", "HEAD", cwd=str(wt), env=env,
        ).stdout.strip()
        assert head == "release/v2.1.0"

    @needs_zsh
    def test_pane_count_zero(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """Pane count of 0 should be rejected."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        result = run_agm(
            ["open", "-n", "0", "repo"], env=env, cwd=str(project), check=False,
        )
        assert result.returncode != 0
        assert "pane count must be a positive integer" in result.stderr

    @needs_zsh
    def test_negative_pane_count(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """Negative pane count should be rejected."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        result = run_agm(
            ["open", "-n", "-1", "repo"], env=env, cwd=str(project), check=False,
        )
        assert result.returncode != 0

    def test_malformed_json_settings(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """Malformed JSON in sandbox settings causes a JSON parse error."""
        TestSandbox._make_fake_srt(tmp_path / "bin", env)

        work = tmp_path / "work"
        work.mkdir()
        sandbox_dir = work / ".sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "echo.json").write_text("{invalid json")

        result = run_agm(
            ["run", "echo", "hi"], env=env, cwd=str(work), check=False,
        )
        # The merge step emits a JSON decode traceback on stderr.
        assert "json" in result.stderr.lower()

    def test_worktree_already_exists(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """An existing branch worktree is reused by ``wt new``."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        run_agm(["wt", "new", "dupe-branch"], env=env, cwd=str(project / "repo"))
        assert (project / "worktrees" / "dupe-branch").is_dir()

        result = run_agm(
            ["wt", "new", "dupe-branch"],
            env=env, cwd=str(project / "repo"), check=False,
        )
        assert result.returncode == 0

    def test_config_copy_destination_does_not_exist(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """Copying config to a non-existent destination: files are not copied."""
        src = tmp_path / "src"
        src.mkdir()
        (src / ".env").write_text("KEY=val")

        run_agm(
            ["config", "cp", str(tmp_path / "nonexistent")],
            env=env, cwd=str(src),
        )
        # cpconfig.sh does not create the target directory — nothing is copied.
        assert not (tmp_path / "nonexistent").exists()

    def test_init_derives_name_from_bare_url(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """Name derivation strips .git suffix from URLs."""
        bare = make_bare_repo(tmp_path / "my-cool-project.git", env)

        run_agm(["init", str(bare)], env=env, cwd=str(tmp_path))

        assert (tmp_path / "my-cool-project" / "repo").is_dir()

    @needs_zsh
    def test_large_pane_count(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """A larger pane count (16) should work without error."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        run_agm(["open", "-n", "16", "repo"], env=env, cwd=str(project))

        log = tmux_log.read_text()
        assert log.count("split-window") == 15


# ── multi-step workflows ───────────────────────────────────────────────────


class TestWorkflows:
    """Multi-step user workflows that combine several agm commands."""

    def test_full_branch_lifecycle(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """init → wt new → verify → wt rm → verify cleanup."""
        bare = make_bare_repo(tmp_path / "origin.git", env)

        # Step 1: init
        run_agm(["init", "proj", str(bare)], env=env, cwd=str(tmp_path))
        project = tmp_path / "proj"
        assert (project / "repo").is_dir()

        # Step 2: create a branch worktree
        run_agm(
            ["wt", "new", "feat/lifecycle"],
            env=env, cwd=str(project / "repo"),
        )
        wt = project / "worktrees" / "feat/lifecycle"
        assert wt.is_dir()
        assert (wt / "README.md").exists()

        # Step 3: remove the worktree
        run_agm(
            ["wt", "rm", "feat/lifecycle"],
            env=env, cwd=str(project / "repo"),
        )
        assert not wt.exists()
        branches = _git("branch", cwd=str(project / "repo"), env=env).stdout
        assert "feat/lifecycle" not in branches

    def test_dep_workflow(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """init → dep new → dep switch → fetch verifies dependency management."""
        bare_main = make_bare_repo(tmp_path / "main.git", env)
        bare_dep = make_bare_repo(tmp_path / "mylib.git", env)

        # Push a feature branch to the dep.
        dep_clone = tmp_path / "dep-clone"
        _git("clone", str(bare_dep), str(dep_clone), cwd=str(tmp_path), env=env)
        _push_branch(dep_clone, bare_dep, "feat/dep-feature", "dep.txt", env)

        # Step 1: init
        run_agm(["init", "proj", str(bare_main)], env=env, cwd=str(tmp_path))
        project = tmp_path / "proj"

        # Step 2: add dependency
        run_agm(["dep", "new", str(bare_dep)], env=env, cwd=str(project))
        assert (project / "deps" / "mylib" / "main" / "README.md").exists()

        # Step 3: switch dep to feature branch
        run_agm(
            ["dep", "switch", "mylib", "feat/dep-feature"],
            env=env, cwd=str(project),
        )
        assert (project / "deps" / "mylib" / "feat/dep-feature" / "dep.txt").exists()

        # Step 4: fetch should pick up both main repo and deps
        result = run_agm(["fetch"], env=env, cwd=str(project))
        assert "Fetching repo" in result.stdout
        assert "Fetching deps/" in result.stdout

    def test_config_copied_to_new_worktree(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """init → add config → wt new: config files are propagated."""
        bare = make_bare_repo(tmp_path / "origin.git", env)

        run_agm(["init", "proj", str(bare)], env=env, cwd=str(tmp_path))
        project = tmp_path / "proj"

        # Add project config files.
        (project / "config" / ".env").write_text("DB_HOST=localhost")
        (project / "config" / ".mcp.json").write_text('{"key": "value"}')

        # Create a new branch — config should be copied automatically.
        run_agm(
            ["wt", "new", "feat/with-config"],
            env=env, cwd=str(project / "repo"),
        )

        wt = project / "worktrees" / "feat/with-config"
        assert (wt / ".env").read_text() == "DB_HOST=localhost"
        assert (wt / ".mcp.json").read_text() == '{"key": "value"}'

    @needs_zsh
    def test_init_then_open_then_new_sessions(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """init → open repo → open feature branch → verify both."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        tmux_log = tmp_path / "tmux.log"
        _install_fake_tmux(tmp_path / "bin", tmux_log, env)

        # Step 1: init
        run_agm(["init", "proj", str(bare)], env=env, cwd=str(tmp_path))
        project = tmp_path / "proj"

        # Step 2: open main session
        run_agm(["open", "repo"], env=env, cwd=str(project))
        log = tmux_log.read_text()
        assert "-s proj" in log

        # Step 3: create a feature branch session
        run_agm(["open", "feat/work"], env=env, cwd=str(project))
        log = tmux_log.read_text()
        assert "-s proj/feat/work" in log
        assert (project / "worktrees" / "feat/work" / "README.md").exists()

    def test_multiple_worktrees_isolated(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """Two worktrees from the same repo are independent."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        run_agm(["wt", "new", "branch-a"], env=env, cwd=str(project / "repo"))
        run_agm(["wt", "new", "branch-b"], env=env, cwd=str(project / "repo"))

        wt_a = project / "worktrees" / "branch-a"
        wt_b = project / "worktrees" / "branch-b"

        # Write a file in branch-a — it must not appear in branch-b.
        (wt_a / "only-in-a.txt").write_text("a")
        _git("add", ".", cwd=str(wt_a), env=env)
        _git("commit", "-m", "add to a", cwd=str(wt_a), env=env)

        assert (wt_a / "only-in-a.txt").exists()
        assert not (wt_b / "only-in-a.txt").exists()

        # Verify each is on its own branch.
        head_a = _git(
            "rev-parse", "--abbrev-ref", "HEAD", cwd=str(wt_a), env=env,
        ).stdout.strip()
        head_b = _git(
            "rev-parse", "--abbrev-ref", "HEAD", cwd=str(wt_b), env=env,
        ).stdout.strip()
        assert head_a == "branch-a"
        assert head_b == "branch-b"

    def test_fetch_then_checkout(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """fetch discovers remote branches, then wt new checks one out."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        # Push a branch from a separate clone.
        other = make_working_repo(tmp_path / "other", bare, env)
        _push_branch(other, bare, "feat/synced", "synced.txt", env)

        # Fetch to discover the remote branch.
        run_agm(["fetch"], env=env, cwd=str(project / "repo"))
        branches = _git("branch", cwd=str(project / "repo"), env=env).stdout
        assert "feat/synced" in branches

        # Check out the synced branch into a worktree.
        run_agm(
            ["wt", "new", "feat/synced"],
            env=env, cwd=str(project / "repo"),
        )
        wt = project / "worktrees" / "feat/synced"
        assert (wt / "synced.txt").exists()
        head = _git(
            "rev-parse", "--abbrev-ref", "HEAD", cwd=str(wt), env=env,
        ).stdout.strip()
        assert head == "feat/synced"

    def test_parallel_worktrees_remove_one(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """Create two worktrees, remove one, verify the other is unaffected."""
        bare = make_bare_repo(tmp_path / "origin.git", env)
        project = _make_project(tmp_path, bare, env)

        run_agm(["wt", "new", "branch-keep"], env=env, cwd=str(project / "repo"))
        run_agm(["wt", "new", "branch-rm"], env=env, cwd=str(project / "repo"))

        # Both exist.
        assert (project / "worktrees" / "branch-keep").is_dir()
        assert (project / "worktrees" / "branch-rm").is_dir()

        # Remove one.
        run_agm(["wt", "rm", "branch-rm"], env=env, cwd=str(project / "repo"))

        # Removed worktree is gone.
        assert not (project / "worktrees" / "branch-rm").exists()
        branches = _git("branch", cwd=str(project / "repo"), env=env).stdout
        assert "branch-rm" not in branches

        # Surviving worktree is still healthy.
        assert (project / "worktrees" / "branch-keep" / "README.md").exists()
        head = _git(
            "rev-parse", "--abbrev-ref", "HEAD",
            cwd=str(project / "worktrees" / "branch-keep"), env=env,
        ).stdout.strip()
        assert head == "branch-keep"
