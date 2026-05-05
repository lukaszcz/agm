"""Tests for agm.commands.dep.new."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import agm.commands.dep.new as dep_new
from agm.commands.args import DepNewArgs


class TestDepNewRun:
    """Tests for dep new run()."""

    def test_exits_when_dep_already_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        deps_dir = project_dir / "deps"
        dep_dir = deps_dir / "mylib"
        dep_dir.mkdir(parents=True)

        monkeypatch.setattr(dep_new, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_new, "derive_dep_name", lambda url: "mylib")
        monkeypatch.setattr(dep_new, "exists", lambda p: p == dep_dir)

        with pytest.raises(SystemExit):
            dep_new.run(DepNewArgs(branch=None, repo_url="https://github.com/org/mylib"))

    def test_clones_with_provided_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"

        monkeypatch.setattr(dep_new, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_new, "derive_dep_name", lambda url: "mylib")
        monkeypatch.setattr(dep_new, "exists", lambda p: False)
        monkeypatch.setattr(dep_new, "mkdir", lambda p, parents=False, exist_ok=False: None)

        require_success_calls: list[list[str]] = []
        monkeypatch.setattr(
            dep_new, "require_success", lambda cmd: require_success_calls.append(cmd)
        )

        update_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            dep_new,
            "update_dependency_config",
            lambda *, project_dir, dep_name, dep_branch, config_branch: update_calls.append(
                {"dep_name": dep_name, "dep_branch": dep_branch, "config_branch": config_branch}
            ),
        )
        monkeypatch.setattr(dep_new, "current_config_branch", lambda pd: None)

        dep_new.run(DepNewArgs(branch="main", repo_url="https://github.com/org/mylib"))

        assert len(require_success_calls) == 1
        clone_cmd = require_success_calls[0]
        assert "clone" in clone_cmd
        assert "--branch" in clone_cmd
        assert "main" in clone_cmd
        assert len(update_calls) == 1
        assert update_calls[0]["dep_branch"] == "main"

    def test_resolves_default_branch_when_none_provided(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"

        monkeypatch.setattr(dep_new, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_new, "derive_dep_name", lambda url: "mylib")
        monkeypatch.setattr(dep_new, "exists", lambda p: False)
        monkeypatch.setattr(dep_new, "mkdir", lambda p, parents=False, exist_ok=False: None)
        monkeypatch.setattr(dep_new, "default_branch_from_remote", lambda url: "develop")
        monkeypatch.setattr(dep_new, "require_success", lambda cmd: None)
        monkeypatch.setattr(
            dep_new,
            "update_dependency_config",
            lambda *, project_dir, dep_name, dep_branch, config_branch: None,
        )
        monkeypatch.setattr(dep_new, "current_config_branch", lambda pd: None)

        # No branch provided — should call default_branch_from_remote
        dep_new.run(DepNewArgs(branch=None, repo_url="https://github.com/org/mylib"))

    def test_cleans_up_dep_dir_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"

        monkeypatch.setattr(dep_new, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_new, "derive_dep_name", lambda url: "mylib")
        monkeypatch.setattr(dep_new, "exists", lambda p: False)
        monkeypatch.setattr(dep_new, "mkdir", lambda p, parents=False, exist_ok=False: None)
        monkeypatch.setattr(dep_new, "default_branch_from_remote", lambda url: "main")

        def fail_require_success(cmd: list[str]) -> None:
            raise SystemExit(1)

        monkeypatch.setattr(dep_new, "require_success", fail_require_success)

        rmdir_calls: list[Path] = []
        monkeypatch.setattr(dep_new, "rmdir", lambda p: rmdir_calls.append(p))

        with pytest.raises(SystemExit):
            dep_new.run(DepNewArgs(branch=None, repo_url="https://github.com/org/mylib"))

        assert len(rmdir_calls) == 1

    def test_clones_to_correct_target_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        deps_dir = project_dir / "deps"

        monkeypatch.setattr(dep_new, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_new, "derive_dep_name", lambda url: "mylib")
        monkeypatch.setattr(dep_new, "exists", lambda p: False)

        mkdir_calls: list[Path] = []
        monkeypatch.setattr(
            dep_new, "mkdir", lambda p, parents=False, exist_ok=False: mkdir_calls.append(p)
        )

        require_success_calls: list[list[str]] = []
        monkeypatch.setattr(
            dep_new, "require_success", lambda cmd: require_success_calls.append(cmd)
        )
        monkeypatch.setattr(
            dep_new,
            "update_dependency_config",
            lambda *, project_dir, dep_name, dep_branch, config_branch: None,
        )
        monkeypatch.setattr(dep_new, "current_config_branch", lambda pd: None)

        dep_new.run(DepNewArgs(branch="feat", repo_url="https://github.com/org/mylib"))

        # mkdir should have been called for the dep parent directory
        assert any(p == deps_dir / "mylib" for p in mkdir_calls)
        # Clone target is deps/mylib/feat
        clone_cmd = require_success_calls[0]
        assert str(deps_dir / "mylib" / "feat") in clone_cmd
