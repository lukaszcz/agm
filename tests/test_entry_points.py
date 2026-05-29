"""Tests for simple run() entry-point wrappers in commands subpackages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import agm.commands.config.copy as config_copy_cmd
import agm.commands.config.env as config_env_cmd
import agm.commands.dep.new as dep_new_cmd
import agm.commands.setup as setup_cmd
import agm.commands.tmux.close as tmux_close_cmd
import agm.commands.tmux.layout as tmux_layout_cmd
import agm.commands.tmux.open as tmux_open_cmd
import agm.commands.worktree.new as worktree_new_cmd
import agm.commands.worktree.remove as worktree_remove_cmd
from agm.commands.args import (
    ConfigCopyArgs,
    ConfigEnvArgs,
    DepNewArgs,
    TmuxCloseArgs,
    TmuxLayoutArgs,
    TmuxOpenArgs,
    WorktreeNewArgs,
    WorktreeRemoveArgs,
)
from agm.core import dry_run as dry_run_module

# ---------------------------------------------------------------------------
# agm.commands.tmux.close
# ---------------------------------------------------------------------------


class TestTmuxCloseRun:
    def test_delegates_to_close_tmux_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            tmux_close_cmd,
            "close_tmux_session",
            lambda **kw: calls.append(kw),
        )
        tmux_close_cmd.run(TmuxCloseArgs(session_name="mysession"))
        assert calls == [{"session_name": "mysession"}]


# ---------------------------------------------------------------------------
# agm.commands.tmux.open
# ---------------------------------------------------------------------------


class TestTmuxOpenRun:
    def test_delegates_to_create_tmux_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            tmux_open_cmd,
            "create_tmux_session",
            lambda **kw: calls.append(kw),
        )
        tmux_open_cmd.run(TmuxOpenArgs(detach=True, pane_count="4", session_name="s"))
        assert len(calls) == 1
        assert calls[0]["detach"] is True
        assert calls[0]["pane_count"] == "4"
        assert calls[0]["session_name"] == "s"


# ---------------------------------------------------------------------------
# agm.commands.tmux.layout
# ---------------------------------------------------------------------------


class TestTmuxLayoutRun:
    def test_delegates_to_apply_layout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resolve_calls: list[Any] = []
        apply_calls: list[dict[str, Any]] = []

        monkeypatch.setattr(
            tmux_layout_cmd,
            "resolve_window_layout_target",
            lambda window_id: (resolve_calls.append(window_id), ("@1", 200, 50))[1],
        )
        monkeypatch.setattr(
            tmux_layout_cmd,
            "apply_layout",
            lambda **kw: apply_calls.append(kw),
        )
        tmux_layout_cmd.run(TmuxLayoutArgs(pane_count="4", window_id="@1"))
        assert len(apply_calls) == 1
        assert apply_calls[0]["pane_count"] == 4
        assert apply_calls[0]["window_id"] == "@1"
        assert apply_calls[0]["width"] == 200
        assert apply_calls[0]["height"] == 50


# ---------------------------------------------------------------------------
# agm.commands.worktree.new
# ---------------------------------------------------------------------------


class TestWorktreeNewRun:
    def test_delegates_to_ensure_worktree(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[dict[str, Any]] = []
        fake_worktree_path = tmp_path / "worktrees" / "feature"
        monkeypatch.setattr(
            worktree_new_cmd,
            "ensure_worktree",
            lambda **kw: (calls.append(kw), fake_worktree_path)[1],
        )
        monkeypatch.setattr(
            worktree_new_cmd, "discover_current_project_dir", lambda *a, **kw: None
        )
        monkeypatch.setattr(
            worktree_new_cmd, "commit_config_dir_changes", lambda *a, **kw: None
        )
        worktree_new_cmd.run(WorktreeNewArgs(branch="feature", worktrees_dir=None))
        assert len(calls) == 1
        assert calls[0]["new_branch"] == "feature"
        assert calls[0]["branch"] is None
        assert calls[0]["existing_ok"] is False
        assert calls[0]["reuse_existing_branch"] is True

    def test_commits_config_when_project_dir_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_worktree_path = tmp_path / "worktrees" / "feature"
        fake_project_dir = tmp_path / "project"
        monkeypatch.setattr(
            worktree_new_cmd,
            "ensure_worktree",
            lambda **kw: fake_worktree_path,
        )
        monkeypatch.setattr(
            worktree_new_cmd,
            "discover_current_project_dir",
            lambda *a, **kw: fake_project_dir,
        )
        commit_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            worktree_new_cmd,
            "commit_config_dir_changes",
            lambda *a, **kw: commit_calls.append({"args": a, "kwargs": kw}),
        )
        monkeypatch.setattr(
            worktree_new_cmd,
            "project_config_dir",
            lambda pd: pd / "config",
        )
        worktree_new_cmd.run(WorktreeNewArgs(branch="feature", worktrees_dir=None))
        assert len(commit_calls) == 1


# ---------------------------------------------------------------------------
# agm.commands.worktree.remove
# ---------------------------------------------------------------------------


class TestWorktreeRemoveRun:
    def test_delegates_to_remove_worktree(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            worktree_remove_cmd,
            "remove_worktree",
            lambda **kw: calls.append(kw),
        )
        monkeypatch.setattr(
            worktree_remove_cmd.git_helpers,
            "checkout_root",
            lambda cwd=None: Path("/tmp/repo"),
        )
        worktree_remove_cmd.run(WorktreeRemoveArgs(force=True, branch="feature"))
        assert calls == [{"repo_dir": Path("/tmp/repo"), "force": True, "branch": "feature"}]


# ---------------------------------------------------------------------------
# agm.commands.setup
# ---------------------------------------------------------------------------


class TestSetupRun:
    def test_delegates_to_run_setup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []
        monkeypatch.setattr(
            setup_cmd,
            "run_setup",
            lambda: calls.append(1),
        )
        setup_cmd.run()
        assert calls == [1]


# ---------------------------------------------------------------------------
# agm.commands.config.copy
# ---------------------------------------------------------------------------


class TestConfigCopyRun:
    def test_delegates_to_copy_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[Path] = []
        monkeypatch.setattr(
            config_copy_cmd,
            "copy_config",
            lambda target: calls.append(target),
        )
        config_copy_cmd.run(ConfigCopyArgs(config_command=None, dirname="/some/dir"))
        assert calls == [Path("/some/dir")]


# ---------------------------------------------------------------------------
# agm.commands.config.env
# ---------------------------------------------------------------------------


class TestConfigEnvRun:
    def test_run_prints_shell_statements(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        before = {"PATH": "/usr/bin", "OLD_VAR": "x"}
        after = {"PATH": "/usr/local/bin", "NEW_VAR": "y"}
        monkeypatch.setattr(config_env_cmd.os, "environ", dict(before))
        monkeypatch.setattr(
            config_env_cmd,
            "load_current_config_env",
            lambda env: after,
        )
        config_env_cmd.run(ConfigEnvArgs())
        out = capsys.readouterr().out
        assert "unset OLD_VAR" in out
        assert "export NEW_VAR=y" in out

    def test_run_empty_delta_produces_no_output(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = {"PATH": "/bin"}
        monkeypatch.setattr(config_env_cmd.os, "environ", dict(env))
        monkeypatch.setattr(
            config_env_cmd,
            "load_current_config_env",
            lambda env: dict(env),
        )
        config_env_cmd.run(ConfigEnvArgs())
        out = capsys.readouterr().out
        assert out == ""


# ---------------------------------------------------------------------------
# agm.commands.dep.new — dry_run branch (cleanup skipped)
# ---------------------------------------------------------------------------


class TestDepNewDryRun:
    @pytest.fixture(autouse=True)
    def reset_dry_run(self) -> None:
        original = dry_run_module.enabled()
        yield  # type: ignore[misc]
        dry_run_module.set_enabled(original)

    def test_dry_run_propagates_system_exit_without_cleanup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dry_run_module.set_enabled(True)
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        monkeypatch.setattr(dep_new_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_new_cmd, "derive_dep_name", lambda url: "mylib")
        monkeypatch.setattr(dep_new_cmd, "exists", lambda p: False)
        monkeypatch.setattr(dep_new_cmd, "mkdir", lambda p, **kw: None)
        monkeypatch.setattr(
            dep_new_cmd.git_helpers, "default_branch_from_remote", lambda url: "main"
        )

        def fail_require_success(cmd: list[str]) -> None:
            raise SystemExit(1)

        monkeypatch.setattr(dep_new_cmd, "require_success", fail_require_success)

        rmdir_calls: list[Path] = []
        monkeypatch.setattr(dep_new_cmd, "rmdir", lambda p: rmdir_calls.append(p))

        with pytest.raises(SystemExit):
            dep_new_cmd.run(DepNewArgs(branch=None, repo_url="https://github.com/org/mylib"))

        # In dry_run mode, cleanup (rmdir) should NOT be called
        assert rmdir_calls == []

    def test_dep_new_run_with_branch_and_dep_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cover resolved_branch = args.branch with branch provided."""
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        monkeypatch.setattr(dep_new_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_new_cmd, "derive_dep_name", lambda url: "mylib")
        monkeypatch.setattr(dep_new_cmd, "exists", lambda p: False)
        monkeypatch.setattr(dep_new_cmd, "mkdir", lambda p, **kw: None)

        require_success_calls: list[list[str]] = []
        monkeypatch.setattr(
            dep_new_cmd, "require_success", lambda cmd: require_success_calls.append(cmd)
        )
        monkeypatch.setattr(dep_new_cmd, "update_dependency_config", lambda **kw: None)
        monkeypatch.setattr(dep_new_cmd, "current_config_branch", lambda pd: None)

        dep_new_cmd.run(DepNewArgs(branch="develop", repo_url="https://github.com/org/mylib"))

        assert len(require_success_calls) == 1
        assert "develop" in require_success_calls[0]

    def test_cleanup_handles_os_error_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cover lines 42-43: OSError during rmdir cleanup is silenced."""
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        monkeypatch.setattr(dep_new_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_new_cmd, "derive_dep_name", lambda url: "mylib")
        monkeypatch.setattr(dep_new_cmd, "exists", lambda p: False)
        monkeypatch.setattr(dep_new_cmd, "mkdir", lambda p, **kw: None)
        monkeypatch.setattr(
            dep_new_cmd.git_helpers, "default_branch_from_remote", lambda url: "main"
        )

        def fail_require_success(cmd: list[str]) -> None:
            raise SystemExit(1)

        monkeypatch.setattr(dep_new_cmd, "require_success", fail_require_success)

        def raise_os_error(p: Path) -> None:
            raise OSError("permission denied")

        monkeypatch.setattr(dep_new_cmd, "rmdir", raise_os_error)

        # Should still raise SystemExit, just silences the OSError
        with pytest.raises(SystemExit):
            dep_new_cmd.run(DepNewArgs(branch=None, repo_url="https://github.com/org/mylib"))
