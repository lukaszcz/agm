"""Tests for agm.project.setup – run_setup."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import agm.project.setup as project_setup


class TestRunSetup:
    """Tests for project.setup.run_setup."""

    def _make_project(self, tmp_path: Path) -> tuple[Path, Path]:
        """Return (project_dir, repo_dir) with minimal workspace layout."""
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
        monkeypatch.setattr(project_setup, "current_checkout", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(
            project_setup.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            project_setup,
            "load_worktree_env",
            lambda pd, branch, checkout_dir, env=None: dict(os.environ),
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
        monkeypatch.setattr(project_setup, "current_checkout", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(
            project_setup.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            project_setup,
            "load_worktree_env",
            lambda pd, branch, checkout_dir, env=None: dict(os.environ),
        )

        run_calls: list[list[str]] = []
        monkeypatch.setattr(
            project_setup, "require_success", lambda cmd, cwd=None, env=None: run_calls.append(cmd)
        )

        project_setup.run_setup(cwd=project_dir)

        assert len(run_calls) == 1
        assert run_calls[0] == ["bash", str(setup_script)]
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
        monkeypatch.setattr(project_setup, "current_checkout", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(
            project_setup.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            project_setup,
            "load_worktree_env",
            lambda pd, branch, checkout_dir, env=None: dict(os.environ),
        )

        project_setup.run_setup(cwd=project_dir)

        captured = capsys.readouterr()
        assert "No setup scripts found" in captured.out

    def test_runs_all_found_setup_scripts_in_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, repo_dir = self._make_project(tmp_path)
        config_dir = project_dir / "config"

        # Create two setup scripts: one in config_dir, one in checkout_dir
        config_script = config_dir / "setup.sh"
        config_script.write_text("#!/bin/sh\n", encoding="utf-8")
        config_script.chmod(config_script.stat().st_mode | stat.S_IEXEC)

        checkout_script = repo_dir / ".setup.sh"
        checkout_script.write_text("#!/bin/sh\n", encoding="utf-8")
        checkout_script.chmod(checkout_script.stat().st_mode | stat.S_IEXEC)

        monkeypatch.setattr(
            project_setup, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(project_setup, "current_checkout", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(
            project_setup.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            project_setup,
            "load_worktree_env",
            lambda pd, branch, checkout_dir, env=None: dict(os.environ),
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
        monkeypatch.setattr(project_setup, "current_checkout", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(
            project_setup.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            project_setup,
            "load_worktree_env",
            lambda pd, branch, checkout_dir, env=None: dict(os.environ),
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
