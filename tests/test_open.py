"""Tests for agm.commands.open helper functions."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import agm.commands.open as open_module
import agm.project.layout as layout_module


class TestResolveParentConfigBranch:
    def _workspace_project(self, tmp_path: Path, env: dict[str, str]) -> Path:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        repo_dir = project_dir / "repo"
        repo_dir.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)
        return project_dir

    def test_explicit_parent_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
    ) -> None:
        project_dir = self._workspace_project(tmp_path, env)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "current_branch",
            lambda _repo_dir, **_kw: "main",
        )
        result = layout_module.parent_config_branch(project_dir, "feature-a")
        assert result == "feature-a"

    def test_command_wrapper_delegates_to_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        monkeypatch.setattr(open_module, "parent_config_branch", lambda _project, parent: parent)

        assert open_module.resolve_parent_config_branch(project_dir, "feature-a") == "feature-a"

    def test_none_parent_returns_repo_branch_if_branch_checkout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
    ) -> None:
        project_dir = self._workspace_project(tmp_path, env)
        # When parent=None and repo branch is "main", resolve uses "main"
        # but since main IS the main checkout, it should return None
        monkeypatch.setattr(
            layout_module.git_helpers,
            "current_branch",
            lambda _repo_dir, **_kw: "main",
        )
        result = layout_module.parent_config_branch(project_dir, None)
        assert result is None

    def test_none_parent_with_non_main_repo_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
    ) -> None:
        project_dir = self._workspace_project(tmp_path, env)
        # When repo branch is "develop" (not "main"), and parent is None,
        # resolve uses "develop" which is the repo branch → main checkout → None
        monkeypatch.setattr(
            layout_module.git_helpers,
            "current_branch",
            lambda _repo_dir, **_kw: "develop",
        )
        # The repo branch IS the main checkout branch, so None
        result = layout_module.parent_config_branch(project_dir, None)
        assert result is None

    def test_explicit_parent_same_as_repo_branch_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
    ) -> None:
        project_dir = self._workspace_project(tmp_path, env)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "current_branch",
            lambda _repo_dir, **_kw: "main",
        )
        # Parent "main" == repo branch "main" → this is the main checkout → None
        result = layout_module.parent_config_branch(project_dir, "main")
        assert result is None

    def test_explicit_parent_different_from_repo_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
    ) -> None:
        project_dir = self._workspace_project(tmp_path, env)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "current_branch",
            lambda _repo_dir, **_kw: "develop",
        )
        # Parent "feature-x" != repo branch "develop" → not main checkout
        result = layout_module.parent_config_branch(project_dir, "feature-x")
        assert result == "feature-x"
