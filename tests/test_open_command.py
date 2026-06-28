"""Tests for agm.commands.workspace.open."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

import agm.commands.workspace.open as open_module
from agm.cli_support.args import OpenArgs
from agm.commands.workspace.open import (
    checkout_workspace,
    create_workspace,
    open_or_create_workspace,
    open_workspace,
    queue_setup_and_focus_workspace_session,
    validate_pane_count,
)
from agm.core import dry_run
from agm.project import workspace_shell
from agm.project.workspace_shell import ensure_workspace_shell


def _make_git_project(tmp_path: Path, env: dict[str, str]) -> Path:
    project = tmp_path / "proj"
    repo = project / "repo"
    (project / "config").mkdir(parents=True)
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, env=env, check=True)
    (repo / "README.md").write_text("main\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, env=env, check=True)
    return project

# ===========================================================================
# validate_pane_count
# ===========================================================================


class TestValidatePaneCount:
    def test_valid_count_does_not_raise(self) -> None:
        validate_pane_count("4")  # should not raise

    def test_none_does_not_raise(self) -> None:
        validate_pane_count(None)

    def test_invalid_count_exits(self) -> None:
        with pytest.raises(SystemExit):
            validate_pane_count("bad")

    def test_zero_exits(self) -> None:
        with pytest.raises(SystemExit):
            validate_pane_count("0")

    def test_re_raises_when_exit_code_is_not_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cover the re-raise when exc.code != 1."""

        def _raise_exit_2(cmd: list[str], pane_count: str | None) -> int:
            raise SystemExit(2)

        # Patch validate_tmux_pane_count to raise SystemExit(2) — code != 1
        monkeypatch.setattr(open_module, "validate_tmux_pane_count", _raise_exit_2)
        with pytest.raises(SystemExit) as exc_info:
            validate_pane_count("whatever")
        assert exc_info.value.code == 2


# ===========================================================================
# queue_setup_and_focus_workspace_session
# ===========================================================================


class TestQueueSetupAndFocusSession:
    def test_detached_returns_without_focusing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cache = tmp_path / "cache"
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
        dry_run.set_enabled(True)

        queue_setup_and_focus_workspace_session(
            detached=True,
            pane_count=None,
            session_name="s",
            repo_path=tmp_path,
        )

        out = capsys.readouterr().out
        wrapper = str(workspace_shell.workspace_shell_dir("s") / "shell")
        assert "tmux new-session -dP" in out
        assert wrapper in out
        assert ".agent-files" not in out
        assert "tmux send-keys -t s:0.0 'agm workspace setup' C-m" in out
        assert out.index("tmux new-session") < out.index("agm workspace setup")
        assert "tmux attach-session" not in out
        assert "tmux switch-client" not in out

    def test_not_detached_raises_system_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cache = tmp_path / "cache"
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
        dry_run.set_enabled(True)
        monkeypatch.delenv("TMUX", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            queue_setup_and_focus_workspace_session(
                detached=False,
                pane_count=None,
                session_name="s",
                repo_path=tmp_path,
            )
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        wrapper = str(workspace_shell.workspace_shell_dir("s") / "shell")
        assert "tmux new-session -dP" in out
        assert wrapper in out
        assert ".agent-files" not in out
        assert "tmux send-keys -t s:0.0 'agm workspace setup' C-m" in out
        assert out.index("tmux new-session") < out.index("agm workspace setup")
        assert "tmux attach-session -t s" in out



class TestEnsureWorkspaceShell:
    def _setup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, session_name: str = "s"
    ) -> tuple[Path, Path, Path, Path]:
        cache = tmp_path / "cache"
        home = tmp_path / "home"
        bin_dir = tmp_path / "bin"
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
        home.mkdir()
        bin_dir.mkdir()
        return cache, home, bin_dir, workspace_shell.workspace_shell_dir(session_name)

    def _fake_agm(self, bin_dir: Path) -> Path:
        agm = bin_dir / "agm"
        agm.write_text(
            "\n".join(
                [
                    "#!/bin/sh",
                    'if [ "$1" = config ] && [ "$2" = env ]; then',
                    '  printf "export HOLDIR=%s/hold\\n" "$PWD"',
                    "  exit 0",
                    "fi",
                    "exit 64",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        agm.chmod(0o755)
        return agm

    def test_writes_wrapper_under_cache_dir_not_agent_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache, _home, _bin_dir, shell_dir = self._setup(tmp_path, monkeypatch)
        wrapper = ensure_workspace_shell("s")

        assert wrapper == shell_dir / "shell"
        assert shell_dir.is_relative_to(cache)
        assert wrapper.stat().st_mode & 0o111 != 0
        # Nothing is written under any .agent-files directory.
        assert ".agent-files" not in str(wrapper)

    def test_rc_files_source_user_dotfiles_then_apply_agm_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _cache, _home, _bin_dir, shell_dir = self._setup(tmp_path, monkeypatch)
        ensure_workspace_shell("s")

        zshrc = (shell_dir / "zsh" / ".zshrc").read_text(encoding="utf-8")
        bashrc = (shell_dir / "bash" / "bashrc").read_text(encoding="utf-8")
        shrc = (shell_dir / "sh" / "shrc").read_text(encoding="utf-8")

        assert 'eval "$(agm config env)"' in zshrc
        assert 'eval "$(agm config env)"' in bashrc
        assert 'eval "$(agm config env)"' in shrc
        # zshrc restores ZDOTDIR to $HOME and sources the user's .zshrc from there.
        assert '. "$ZDOTDIR/.zshrc"' in zshrc
        assert '"$HOME/.bashrc"' in bashrc
        assert '"$HOME/.shrc"' in shrc

    def test_wrapper_execs_real_shell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _cache, _home, _bin_dir, shell_dir = self._setup(tmp_path, monkeypatch)
        wrapper = ensure_workspace_shell("s", env={"SHELL": "/bin/bash"})
        text = wrapper.read_text(encoding="utf-8")
        assert 'exec "$AGM_REAL_SHELL" --rcfile' in text
        assert 'ZDOTDIR="$AGM_WORKSPACE_SHELL_DIR/zsh"' in text
        assert 'export ENV="$AGM_WORKSPACE_SHELL_DIR/sh/shrc"' in text

    def test_recreate_cleans_stale_files_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _cache, _home, _bin_dir, shell_dir = self._setup(tmp_path, monkeypatch)
        shell_dir.mkdir(parents=True)
        (shell_dir / "stale-marker").write_text("gone\n", encoding="utf-8")

        ensure_workspace_shell("s")

        assert not (shell_dir / "stale-marker").exists()
        assert (shell_dir / "shell").exists()

    def test_bash_runs_user_bashrc_then_agm_env_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash is required")

        _cache, home, bin_dir, _shell_dir = self._setup(tmp_path, monkeypatch)
        self._fake_agm(bin_dir)
        (home / ".bashrc").write_text(
            'export USERRC_RAN=1\nexport HOLDIR=user\n', encoding="utf-8"
        )
        wrapper = ensure_workspace_shell("s", env={"SHELL": bash})

        result = subprocess.run(
            [str(wrapper)],
            input="\n".join(
                [
                    'printf "userrc:%s holdir:%s\n" "$USERRC_RAN" "$HOLDIR"',
                    "exit",
                    "",
                ]
            ),
            cwd=tmp_path,
            env={
                **os.environ,
                "HOME": str(home),
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "SHELL": bash,
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        assert result.returncode == 0
        assert "userrc:1" in result.stdout
        # agm env (HOLDIR=<cwd>/hold) overrides the user rc (HOLDIR=user).
        assert f"holdir:{tmp_path}/hold" in result.stdout

    def test_bash_restart_refreshes_workspace_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash is required")

        _cache, home, bin_dir, _shell_dir = self._setup(tmp_path, monkeypatch)
        self._fake_agm(bin_dir)
        (home / ".bashrc").write_text('export USERRC_RAN=1\n', encoding="utf-8")
        wrapper = ensure_workspace_shell("s", env={"SHELL": bash})

        result = subprocess.run(
            [str(wrapper)],
            input="\n".join(
                [
                    'printf "first:%s\n" "$HOLDIR"',
                    "export HOLDIR=broken",
                    'if [ "${AGM_RESTARTED:-}" != 1 ]; then',
                    "  export AGM_RESTARTED=1",
                    '  exec "$SHELL"',
                    "fi",
                    'printf "second:%s\n" "$HOLDIR"',
                    'printf "shell:%s\n" "$SHELL"',
                    "exit",
                    "",
                ]
            ),
            cwd=tmp_path,
            env={
                **os.environ,
                "HOME": str(home),
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "SHELL": bash,
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        assert result.returncode == 0
        assert f"first:{tmp_path}/hold" in result.stdout
        assert f"second:{tmp_path}/hold" in result.stdout
        assert f"shell:{wrapper}" in result.stdout

    def test_self_heals_after_cache_dir_deletion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash is required")

        _cache, home, bin_dir, shell_dir = self._setup(tmp_path, monkeypatch)
        self._fake_agm(bin_dir)
        (home / ".bashrc").write_text('export USERRC_RAN=1\n', encoding="utf-8")
        wrapper = ensure_workspace_shell("s", env={"SHELL": bash})

        # Simulate partial deletion from inside a running pane: the rc
        # subdirectories are removed, then `exec $SHELL` re-runs the wrapper.
        # The wrapper self-heals via `agm workspace shell-regen` before exec'ing.
        result = subprocess.run(
            [str(wrapper)],
            input="\n".join(
                [
                    # Delete only the rc subdirs, leaving the wrapper intact.
                    f"rm -rf {shell_dir / 'zsh'} {shell_dir / 'bash'} {shell_dir / 'sh'}",
                    'if [ "${AGM_HEALED:-}" != 1 ]; then',
                    "  export AGM_HEALED=1",
                    '  exec "$SHELL"',
                    "fi",
                    'printf "healed:%s\n" "$HOLDIR"',
                    "exit",
                    "",
                ]
            ),
            cwd=tmp_path,
            env={
                **os.environ,
                "HOME": str(home),
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "SHELL": bash,
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        assert result.returncode == 0
        assert f"healed:{tmp_path}/hold" in result.stdout
        assert wrapper.exists()


# ===========================================================================
# open_workspace
# ===========================================================================


class TestOpenSession:
    """Tests for open_workspace.

    Note: the happy-path behaviors are also covered end-to-end by TestOpen in
    test_e2e.py; these unit tests add dry-run output assertions and additional
    scenario coverage, all driven by real on-disk project/worktree state.
    """

    def test_opens_main_repo_session_when_branch_is_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        project = _make_git_project(tmp_path, env)

        open_workspace(detached=True, pane_count=None, branch=None, cwd=project)

        out = capsys.readouterr().out
        repo_dir = project / "repo"
        assert "tmux new-session -dP" in out
        assert f"-c {repo_dir}" in out
        assert "Detached tmux session proj created" in out

    def test_exits_when_worktree_not_at_expected_path(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        # "feature" worktree is absent — has_expected_worktree returns False for real.
        project = _make_git_project(tmp_path, env)
        with pytest.raises(SystemExit) as exc_info:
            open_workspace(detached=True, pane_count=None, branch="feature", cwd=project)
        assert exc_info.value.code == 1

    def test_opens_branch_session_when_worktree_exists(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        project = _make_git_project(tmp_path, env)
        feat_path = project / "worktrees" / "feature"
        subprocess.run(
            ["git", "worktree", "add", "-b", "feature", str(feat_path)],
            cwd=project / "repo",
            env=env,
            check=True,
        )

        open_workspace(detached=True, pane_count=None, branch="feature", cwd=project)

        out = capsys.readouterr().out
        assert "tmux new-session -dP" in out
        assert f"-c {feat_path}" in out
        assert "Detached tmux session proj/feature created" in out

    def test_commits_config_with_worktree_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
    ) -> None:
        project = _make_git_project(tmp_path, env)
        config_dir = project / "config"
        subprocess.run(["git", "init", "-b", "main"], cwd=config_dir, env=env, check=True)
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "init"], cwd=config_dir, env=env, check=True
        )
        feature_config = config_dir / "feature"
        feature_config.mkdir()
        (feature_config / "config.toml").write_text("[settings]\n", encoding="utf-8")
        # The per-branch env.sh sets a DIFFERENT git author than the process env, so the
        # commit author proves load_workspace_env's env was actually plumbed through to
        # commit_config_dir_changes. If the env were dropped, the git subprocess would
        # inherit the process env ("Process Env") and the assertion below would fail.
        (feature_config / "env.sh").write_text(
            'export GIT_AUTHOR_NAME="Worktree Env"\n'
            'export GIT_COMMITTER_NAME="Worktree Env"\n',
            encoding="utf-8",
        )
        feat_path = project / "worktrees" / "feature"
        subprocess.run(
            ["git", "worktree", "add", "-b", "feature", str(feat_path)],
            cwd=project / "repo",
            env=env,
            check=True,
        )
        # Process env carries a DIFFERENT author; only the worktree env.sh sets "Worktree Env".
        monkeypatch.setenv("GIT_AUTHOR_NAME", "Process Env")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "Process Env")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "worktree@test.com")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "worktree@test.com")
        monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
        # create_configured_workspace_session is the tmux boundary — suppress it.
        def _noop_session(
            *,
            detached: bool,
            pane_count: str | None,
            session_name: str,
            repo_path: Path,
            run_setup: bool,
        ) -> None:
            pass

        monkeypatch.setattr(open_module, "create_configured_workspace_session", _noop_session)

        open_workspace(detached=True, pane_count=None, branch="feature", cwd=project)

        result = subprocess.run(
            ["git", "log", "-1", "--format=%an"],
            cwd=config_dir,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "Worktree Env"


# ===========================================================================
# create_workspace
# ===========================================================================


class TestNewSession:
    def test_plans_new_worktree_and_setup_queue(
        self,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        project = _make_git_project(tmp_path, env)

        create_workspace(
            detached=True, pane_count=None, parent=None, branch="feature", cwd=project
        )

        out = capsys.readouterr().out
        assert "dry-run: agm mkdir" in out
        assert "git -C" in out
        assert "worktree add -b feature" in out
        assert "tmux send-keys" in out

    def test_plans_new_worktree_from_parent_start_point(
        self,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        project = _make_git_project(tmp_path, env)

        create_workspace(
            detached=True, pane_count=None, parent="shallow-parent",
            branch="feature", cwd=project,
        )

        out = capsys.readouterr().out
        assert "worktree add -b feature" in out
        assert "shallow-parent" in out


# ===========================================================================
# checkout_workspace
# ===========================================================================


class TestCheckoutSession:
    def test_plans_checkout_of_existing_branch(
        self,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        project = _make_git_project(tmp_path, env)

        checkout_workspace(
            detached=True, pane_count=None, parent=None, branch="feature", cwd=project
        )

        out = capsys.readouterr().out
        assert "worktree add" in out
        assert "-b feature" not in out
        assert "feature" in out
        assert "tmux send-keys" in out


# ===========================================================================
# open_or_create_workspace
# ===========================================================================


class TestSmartOpenSession:
    def test_opens_main_session_when_main_branch(
        self,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        project = _make_git_project(tmp_path, env)

        open_or_create_workspace(
            detached=True, pane_count=None, parent=None, branch="main", cwd=project
        )

        out = capsys.readouterr().out
        assert "tmux new-session -dP" in out
        assert f"-c {project / 'repo'}" in out

    def test_opens_existing_worktree_session(
        self,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        project = _make_git_project(tmp_path, env)
        subprocess.run(
            ["git", "worktree", "add", "-b", "feature", str(project / "worktrees" / "feature")],
            cwd=project / "repo",
            env=env,
            check=True,
        )

        open_or_create_workspace(
            detached=True, pane_count=None, parent=None, branch="feature", cwd=project
        )

        out = capsys.readouterr().out
        assert "tmux new-session -dP" in out
        assert f"-c {project / 'worktrees' / 'feature'}" in out

    def test_checks_out_existing_remote_branch(
        self,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        project = _make_git_project(tmp_path, env)
        subprocess.run(["git", "branch", "remote-feat"], cwd=project / "repo", env=env, check=True)

        open_or_create_workspace(
            detached=True, pane_count=None, parent=None, branch="remote-feat", cwd=project
        )

        out = capsys.readouterr().out
        assert "worktree add" in out
        assert "-b remote-feat" not in out
        assert "remote-feat" in out

    def test_creates_create_workspace_when_branch_doesnt_exist(
        self,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        project = _make_git_project(tmp_path, env)

        open_or_create_workspace(
            detached=True, pane_count=None, parent=None, branch="new-branch", cwd=project
        )

        out = capsys.readouterr().out
        assert "worktree add -b new-branch" in out


# ===========================================================================
# run (entry point)
# ===========================================================================


class TestOpenRun:
    def test_run_opens_main_branch_in_dry_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        project = _make_git_project(tmp_path, env)
        monkeypatch.chdir(project)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

        open_module.run(
            OpenArgs(detached=True, pane_count=None, parent=None, branch="main")
        )

        out = capsys.readouterr().out
        repo_dir = project / "repo"
        assert "tmux new-session -dP" in out
        assert f"-c {repo_dir}" in out
        assert "Detached tmux session proj created" in out
