"""Real-git repository discovery tests for sync commands."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import agm.commands.sync.fetch as fetch_cmd


def _init_git_repo(path: Path) -> None:
    """Initialize a minimal git repository at *path* (no commits needed)."""
    path.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True, env=env)


class TestFetchProjectRepos:
    """Repository discovery against the filesystem and git, without mocks."""

    def test_returns_main_repo_only_without_deps_dir(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        _init_git_repo(repo_dir)

        assert fetch_cmd.project_git_repos(project_dir) == [repo_dir]

    def test_returns_main_repo_and_dependency_repos(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        dep1_repo = project_dir / "deps" / "libfoo" / "main"
        _init_git_repo(repo_dir)
        _init_git_repo(dep1_repo)

        assert fetch_cmd.project_git_repos(project_dir) == [repo_dir, dep1_repo]

    def test_embedded_project_ignores_main_repo_directories_during_dependency_discovery(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "proj"
        project_dir = repo_dir / ".agm"
        dep_repo = project_dir / "deps" / "lib" / "release" / "v1"
        _init_git_repo(repo_dir)
        _init_git_repo(dep_repo)

        assert fetch_cmd.project_git_repos(project_dir) == [repo_dir, dep_repo]
