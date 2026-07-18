"""Tests for AGM workspace setup and environment helpers."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

import agm.project.workspace_env as workspace_env
import agm.project.workspace_setup as project_setup
from agm.project.layout import CurrentWorkspace
from agm.project.workspace_env import load_current_workspace_env


class TestRunSetup:
    """Tests for project.workspace_setup.run_setup."""

    def _make_project(self, tmp_path: Path) -> tuple[Path, Path]:
        """Return (project_dir, repo_dir) with minimal split layout."""
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)
        (project_dir / "config").mkdir()
        return project_dir, repo_dir

    def test_prints_message_when_no_setup_scripts_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir, repo_dir = self._make_project(tmp_path)

        monkeypatch.setattr(
            project_setup, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(project_setup, "current_workspace", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(project_setup.git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            project_setup,
            "load_workspace_env",
            lambda pd, branch, workspace_dir, env=None: dict(os.environ),
        )

        project_setup.run_setup(cwd=project_dir)

        captured = capsys.readouterr()
        assert "No setup scripts found" in captured.out

    def test_runs_executable_setup_sh_in_config_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir, repo_dir = self._make_project(tmp_path)
        config_dir = project_dir / "config"
        setup_script = config_dir / "setup.sh"
        setup_script.write_text("#!/bin/sh\n", encoding="utf-8")
        setup_script.chmod(setup_script.stat().st_mode | stat.S_IEXEC)

        monkeypatch.setattr(
            project_setup, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(project_setup, "current_workspace", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(project_setup.git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            project_setup,
            "load_workspace_env",
            lambda pd, branch, workspace_dir, env=None: dict(os.environ),
        )

        run_calls: list[list[str]] = []
        monkeypatch.setattr(
            project_setup, "require_success", lambda cmd, cwd=None, env=None: run_calls.append(cmd)
        )

        project_setup.run_setup(cwd=project_dir)

        assert len(run_calls) == 1
        assert "bash" in run_calls[0]
        assert str(setup_script) in run_calls[0]
        captured = capsys.readouterr()
        assert "Running setup for" in captured.out
        assert "Setup complete for" in captured.out

    def test_skips_non_executable_setup_sh(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir, repo_dir = self._make_project(tmp_path)
        config_dir = project_dir / "config"
        setup_script = config_dir / "setup.sh"
        # Write the file but do NOT make it executable
        setup_script.write_text("#!/bin/sh\n", encoding="utf-8")
        # Remove executable bit explicitly
        setup_script.chmod(0o644)

        monkeypatch.setattr(
            project_setup, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(project_setup, "current_workspace", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(project_setup.git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            project_setup,
            "load_workspace_env",
            lambda pd, branch, workspace_dir, env=None: dict(os.environ),
        )

        project_setup.run_setup(cwd=project_dir)

        captured = capsys.readouterr()
        assert "No setup scripts found" in captured.out

    def test_runs_all_found_setup_scripts_in_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, repo_dir = self._make_project(tmp_path)
        config_dir = project_dir / "config"

        # Create two setup scripts: one in config_dir, one in workspace_dir
        config_script = config_dir / "setup.sh"
        config_script.write_text("#!/bin/sh\n", encoding="utf-8")
        config_script.chmod(config_script.stat().st_mode | stat.S_IEXEC)

        checkout_script = repo_dir / ".setup.sh"
        checkout_script.write_text("#!/bin/sh\n", encoding="utf-8")
        checkout_script.chmod(checkout_script.stat().st_mode | stat.S_IEXEC)

        monkeypatch.setattr(
            project_setup, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(project_setup, "current_workspace", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(project_setup.git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            project_setup,
            "load_workspace_env",
            lambda pd, branch, workspace_dir, env=None: dict(os.environ),
        )

        run_calls: list[list[str]] = []
        monkeypatch.setattr(
            project_setup, "require_success", lambda cmd, cwd=None, env=None: run_calls.append(cmd)
        )

        project_setup.run_setup(cwd=project_dir)

        assert len(run_calls) == 2

    def test_dry_run_prints_operation_instead_of_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir, repo_dir = self._make_project(tmp_path)
        config_dir = project_dir / "config"
        setup_script = config_dir / "setup.sh"
        setup_script.write_text("#!/bin/sh\n", encoding="utf-8")
        setup_script.chmod(setup_script.stat().st_mode | stat.S_IEXEC)

        monkeypatch.setattr(
            project_setup, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(project_setup, "current_workspace", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(project_setup.git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            project_setup,
            "load_workspace_env",
            lambda pd, branch, workspace_dir, env=None: dict(os.environ),
        )
        monkeypatch.setattr(project_setup.dry_run, "enabled", lambda: True)

        dry_run_calls: list[tuple[str, str]] = []
        monkeypatch.setattr(
            project_setup.dry_run,
            "print_operation",
            lambda name, detail: dry_run_calls.append((name, detail)),
        )

        run_calls: list[list[str]] = []
        monkeypatch.setattr(
            project_setup, "require_success", lambda cmd, cwd=None, env=None: run_calls.append(cmd)
        )

        project_setup.run_setup(cwd=project_dir)

        assert len(dry_run_calls) == 1
        assert dry_run_calls[0][0] == "run-setup"
        # require_success is still called in dry_run mode (dry_run just prints first)
        assert len(run_calls) == 1


class TestLoadCurrentConfigEnvWithNoResult:
    def test_falls_back_when_current_workspace_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import agm.project.workspace_env as setup_module

        project = tmp_path / "proj"
        repo = project / "repo"
        repo.mkdir(parents=True)
        monkeypatch.setattr(setup_module, "require_current_project_dir", lambda cwd=None: project)
        monkeypatch.setattr(setup_module, "current_workspace", lambda pd, cwd=None, env=None: None)

        env_captured: list[dict[str, Any]] = []

        def fake_load_config_env(
            project_dir: Path,
            branch: Any,
            *,
            workspace_dir: Path,
            env: Any = None,
        ) -> dict[str, str]:
            env_captured.append(
                {"project_dir": project_dir, "branch": branch, "workspace_dir": workspace_dir}
            )
            return {}

        monkeypatch.setattr(setup_module, "load_config_env", fake_load_config_env)

        load_current_workspace_env(cwd=project)
        assert len(env_captured) == 1
        assert env_captured[0]["branch"] is None
        # workspace_dir should be repo (since it exists)
        assert env_captured[0]["workspace_dir"] == repo

    def test_falls_back_to_current_when_repo_not_a_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import agm.project.workspace_env as setup_module

        project = tmp_path / "proj"
        project.mkdir()
        (project / ".agm").mkdir()
        plain = tmp_path / "plain"
        plain.mkdir()
        monkeypatch.setattr(setup_module, "require_current_project_dir", lambda cwd=None: plain)
        monkeypatch.setattr(setup_module, "current_workspace", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(
            setup_module, "load_config_env", lambda pd, br, *, workspace_dir, env=None: {}
        )
        # Should not crash
        load_current_workspace_env(cwd=plain)


class TestRunSetupLabelFromProjectDir:
    def test_setup_label_falls_back_to_project_dir_relative(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When setup_path is not relative to workspace_dir, try project_dir."""
        import agm.project.workspace_setup as setup_module

        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)
        config_dir = project_dir / "config"
        config_dir.mkdir()

        # Put a setup script in config_dir (outside workspace_dir=repo_dir)
        setup_script = config_dir / "setup.sh"
        setup_script.write_text("#!/bin/sh\n", encoding="utf-8")
        setup_script.chmod(setup_script.stat().st_mode | stat.S_IEXEC)

        monkeypatch.setattr(
            setup_module, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(setup_module, "current_workspace", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(setup_module.git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            setup_module,
            "load_workspace_env",
            lambda pd, branch, workspace_dir, env=None: dict(os.environ),
        )

        run_calls: list[list[str]] = []
        monkeypatch.setattr(
            setup_module,
            "require_success",
            lambda cmd, cwd=None, env=None: run_calls.append(cmd),
        )

        setup_module.run_setup(cwd=project_dir)

        assert len(run_calls) == 1
        captured = capsys.readouterr()
        assert "setup.sh" in captured.out


class TestLoadCurrentConfigEnvRepoDirFallback:
    def test_falls_back_to_cwd_when_repo_not_dir_and_current_workspace_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import agm.project.workspace_env as setup_module

        project = tmp_path / "proj"
        project.mkdir()
        (project / ".agm").mkdir()
        # No repo/ dir
        cwd = tmp_path / "cwd"
        cwd.mkdir()

        monkeypatch.setattr(setup_module, "require_current_project_dir", lambda cwd=None: project)
        monkeypatch.setattr(setup_module, "current_workspace", lambda pd, cwd=None, env=None: None)

        captured_env: dict[str, Any] = {}

        def fake_load_config_env(
            project_dir: Path, branch: Any, *, workspace_dir: Path, env: Any = None
        ) -> dict[str, str]:
            captured_env["workspace_dir"] = workspace_dir
            return {}

        monkeypatch.setattr(setup_module, "load_config_env", fake_load_config_env)

        load_current_workspace_env(cwd=cwd)
        # For embedded project without repo/, project_repo_dir returns project_dir itself
        # which is a dir, so workspace_dir = project_dir (repo_dir)
        assert captured_env["workspace_dir"] == project


class TestRunSetupNoScripts:
    def test_prints_message_when_no_setup_scripts_found(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import agm.project.workspace_setup as setup_module

        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)
        config_dir = project_dir / "config"
        config_dir.mkdir()

        monkeypatch.setattr(
            setup_module, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(setup_module, "current_workspace", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(setup_module.git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            setup_module,
            "load_workspace_env",
            lambda pd, branch, workspace_dir, env=None: dict(os.environ),
        )

        setup_module.run_setup(cwd=project_dir)

        captured = capsys.readouterr()
        assert "No setup scripts found" in captured.out


class TestLoadCurrentConfigEnvWhenResultNoneNoRepoDir:
    def test_falls_back_to_current_when_repo_dir_not_a_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import agm.project.workspace_env as setup_module

        project = tmp_path / "proj"
        project.mkdir()
        (project / ".agm").mkdir()

        monkeypatch.setattr(setup_module, "require_current_project_dir", lambda cwd=None: project)
        monkeypatch.setattr(setup_module, "current_workspace", lambda pd, cwd=None, env=None: None)

        env_captured: list[dict[str, Any]] = []

        def fake_load_config_env(
            project_dir: Path,
            branch: Any,
            *,
            workspace_dir: Path,
            env: Any = None,
        ) -> dict[str, str]:
            env_captured.append(
                {"project_dir": project_dir, "branch": branch, "workspace_dir": workspace_dir}
            )
            return {}

        monkeypatch.setattr(setup_module, "load_config_env", fake_load_config_env)
        load_current_workspace_env(cwd=project)
        assert len(env_captured) == 1
        assert env_captured[0]["branch"] is None
        # For embedded project, project_repo_dir returns project itself which is a dir
        assert env_captured[0]["workspace_dir"] == project


class TestRunSetupLabelValueErrorFallback:
    def test_setup_label_uses_absolute_path_when_not_relative_to_either_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When setup_path is not relative to workspace_dir or project_dir, use absolute path."""
        import agm.project.workspace_setup as setup_module

        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)
        config_dir = project_dir / "config"
        config_dir.mkdir()

        config_setup = config_dir / "setup.sh"
        config_setup.write_text("#!/bin/sh\n", encoding="utf-8")
        config_setup.chmod(config_setup.stat().st_mode | stat.S_IEXEC)

        monkeypatch.setattr(
            setup_module, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(setup_module, "current_workspace", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(setup_module.git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            setup_module,
            "load_workspace_env",
            lambda pd, branch, workspace_dir, env=None: dict(os.environ),
        )

        external_setup = tmp_path / "external" / "setup.sh"
        external_setup.parent.mkdir(parents=True)
        external_setup.write_text("#!/bin/sh\n", encoding="utf-8")
        external_setup.chmod(external_setup.stat().st_mode | stat.S_IEXEC)

        monkeypatch.setattr(setup_module, "require_success", lambda cmd, cwd=None, env=None: None)
        monkeypatch.setattr(
            setup_module,
            "project_config_dir",
            lambda pd: external_setup.parent,
        )

        setup_module.run_setup(cwd=project_dir)

        captured = capsys.readouterr()
        assert str(external_setup) in captured.out


class TestLoadCurrentConfigEnvFallbackNoResult:
    def test_uses_current_when_repo_dir_not_a_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """load_current_workspace_env uses current when result is None and
        repo_dir is not a directory."""
        import agm.project.workspace_env as setup_module

        project2 = tmp_path / "proj2"
        project2.mkdir(parents=True)
        (project2 / "worktrees").mkdir()

        monkeypatch.setattr(setup_module, "require_current_project_dir", lambda cwd=None: project2)
        monkeypatch.setattr(setup_module, "current_workspace", lambda pd, cwd=None, env=None: None)

        env_captured: list[dict[str, Any]] = []

        def fake_load_config_env(
            project_dir: Path,
            branch: Any,
            *,
            workspace_dir: Path,
            env: Any = None,
        ) -> dict[str, str]:
            env_captured.append({"branch": branch, "workspace_dir": workspace_dir})
            return {}

        monkeypatch.setattr(setup_module, "load_config_env", fake_load_config_env)

        load_current_workspace_env(cwd=project2)
        assert len(env_captured) == 1
        assert env_captured[0]["branch"] is None
        # Since repo_dir (project2 / "repo") is not a dir, workspace_dir = current = project2
        assert env_captured[0]["workspace_dir"] == project2


class TestRunSetupWithCurrentWorkspaceResult:
    def test_run_setup_with_workspace_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """run_setup uses workspace result when current_workspace returns non-None."""
        import agm.project.workspace_setup as setup_module

        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)
        (project_dir / "config").mkdir()

        workspace = CurrentWorkspace(
            workspace_dir=repo_dir,
            branch="feat",
            is_main=False,
        )
        monkeypatch.setattr(
            setup_module, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(
            setup_module, "current_workspace", lambda pd, cwd=None, env=None: workspace
        )
        monkeypatch.setattr(setup_module.git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            setup_module,
            "load_workspace_env",
            lambda pd, branch, workspace_dir, env=None: dict(os.environ),
        )

        setup_module.run_setup(cwd=project_dir)

        captured = capsys.readouterr()
        assert "No setup scripts found" in captured.out

    def test_run_setup_branch_none_uses_repo_branch_for_target_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """run_setup uses repo_branch for target_name when branch is None."""
        import agm.project.workspace_setup as setup_module

        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)
        (project_dir / "config").mkdir()

        monkeypatch.setattr(
            setup_module, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(setup_module, "current_workspace", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(setup_module.git_helpers, "current_branch", lambda p, env=None: "dev")
        monkeypatch.setattr(
            setup_module,
            "load_workspace_env",
            lambda pd, branch, workspace_dir, env=None: dict(os.environ),
        )

        setup_module.run_setup(cwd=project_dir)
        captured = capsys.readouterr()
        assert "proj" in captured.out

    def test_run_setup_value_error_fallback_for_label(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When setup_path is not relative to workspace_dir or project_dir,
        the absolute path is used as the label."""
        import agm.project.workspace_setup as setup_module

        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)

        external_config = tmp_path / "external_config"
        external_config.mkdir()
        setup_script = external_config / "setup.sh"
        setup_script.write_text("#!/bin/sh\n", encoding="utf-8")
        setup_script.chmod(setup_script.stat().st_mode | stat.S_IEXEC)

        monkeypatch.setattr(
            setup_module, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(setup_module, "current_workspace", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(setup_module.git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            setup_module,
            "load_workspace_env",
            lambda pd, branch, workspace_dir, env=None: dict(os.environ),
        )
        monkeypatch.setattr(setup_module, "require_success", lambda cmd, cwd=None, env=None: None)
        monkeypatch.setattr(
            setup_module,
            "project_config_dir",
            lambda pd: external_config,
        )

        setup_module.run_setup(cwd=project_dir)

        captured = capsys.readouterr()
        assert str(setup_script) in captured.out


class TestLoadConfigEnvProjDir:
    """Tests for PROJ_DIR and REPO_DIR env vars set by load_config_env."""

    def test_embedded_layout_proj_dir_points_to_agm_dir(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """For embedded layout, PROJ_DIR should point to .agm inside the repo."""
        project = tmp_path / "proj"
        project.mkdir()
        agm_dir = project / ".agm"
        agm_dir.mkdir()
        (agm_dir / "config").mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=project, env=env, check=True)
        workspace_dir = project

        result_env = workspace_env.load_config_env(
            project_dir=agm_dir,
            branch=None,
            workspace_dir=workspace_dir,
            env={},
        )

        assert result_env["PROJ_DIR"] == str(agm_dir)
        assert result_env["REPO_DIR"] == str(project)

    def test_workspace_layout_proj_dir_points_to_project_root(self, tmp_path: Path) -> None:
        """For split layout, PROJ_DIR should point to the project root."""
        project = tmp_path / "proj"
        repo_dir = project / "repo"
        repo_dir.mkdir(parents=True)
        (project / "config").mkdir()

        env = workspace_env.load_config_env(
            project_dir=project,
            branch=None,
            workspace_dir=repo_dir,
            env={},
        )

        assert env["PROJ_DIR"] == str(project)
        assert env["REPO_DIR"] == str(repo_dir)

    def test_embedded_layout_worktree_proj_dir_still_points_to_agm(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        """For embedded layout with a branch workspace, PROJ_DIR still points to .agm."""
        project = tmp_path / "proj"
        project.mkdir()
        agm_dir = project / ".agm"
        agm_dir.mkdir()
        (agm_dir / "config").mkdir()
        worktree_dir = agm_dir / "worktrees" / "feat"
        worktree_dir.mkdir(parents=True)
        subprocess.run(["git", "init", "-b", "main"], cwd=project, env=env, check=True)

        result_env = workspace_env.load_config_env(
            project_dir=agm_dir,
            branch="feat",
            workspace_dir=worktree_dir,
            env={},
        )

        assert result_env["PROJ_DIR"] == str(agm_dir)
        assert result_env["REPO_DIR"] == str(worktree_dir)
