"""Behavior tests for agm.commands.dep.new over real git repositories."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path
from typing import cast

import pytest

import agm.commands.dep.new as dep_new
from agm.cli_support.args import DepNewArgs
from tests._git_helpers import init_repo

_GIT_ENV_NAMES = (
    "HOME",
    "GIT_AUTHOR_NAME",
    "GIT_AUTHOR_EMAIL",
    "GIT_COMMITTER_NAME",
    "GIT_COMMITTER_EMAIL",
    "GIT_CONFIG_NOSYSTEM",
)


def _isolate_git_environment(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    """Give in-process git calls the isolated identity/home of the ``env`` fixture.

    ``dep new`` shells out without an explicit environment, so the process
    environment is what its ``git`` invocations see.
    """
    for name in _GIT_ENV_NAMES:
        monkeypatch.setenv(name, env[name])


def _bare_repo(
    parent: Path, name: str, env: dict[str, str], *, extra_branch: str | None = None
) -> Path:
    """Create a bare repo named *name* whose default branch is ``main``."""
    source = init_repo(parent / f"{name}-source", env)
    if extra_branch is not None:
        subprocess.run(
            ["git", "checkout", "-b", extra_branch, "-q"], cwd=source, env=env, check=True
        )
        (source / f"{extra_branch}.txt").write_text("side branch\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=source, env=env, check=True)
        subprocess.run(["git", "commit", "-m", "side", "-q"], cwd=source, env=env, check=True)
        subprocess.run(["git", "checkout", "main", "-q"], cwd=source, env=env, check=True)
    bare = parent / f"{name}.git"
    subprocess.run(["git", "clone", "--bare", "-q", str(source), str(bare)], env=env, check=True)
    return bare


def _make_project(tmp_path: Path, env: dict[str, str]) -> Path:
    """Create a split-layout AGM project whose ``repo/`` is a real checkout."""
    origin = _bare_repo(tmp_path, "proj-origin", env)
    project = tmp_path / "proj"
    for name in ("worktrees", "deps", "config", "notes"):
        (project / name).mkdir(parents=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(project / "repo")], env=env, check=True)
    return project


def _deps_config(project: Path) -> dict[str, object]:
    """Return the ``[deps]`` table recorded in the project's main config file."""
    config_file = project / "config" / "config.toml"
    if not config_file.is_file():
        return {}
    with config_file.open("rb") as handle:
        parsed = cast(dict[str, object], tomllib.load(handle))
    deps = parsed.get("deps")
    return cast(dict[str, object], deps) if isinstance(deps, dict) else {}


class TestDepNew:
    """``agm dep new`` clones a dependency and records it in the project config."""

    def test_clones_the_remote_default_branch_when_no_branch_is_given(
        self, tmp_path: Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_git_environment(monkeypatch, env)
        project = _make_project(tmp_path, env)
        dep_origin = _bare_repo(tmp_path, "mylib", env)
        monkeypatch.chdir(project / "repo")

        dep_new.run(DepNewArgs(branch=None, repo_url=str(dep_origin)))

        checkout = project / "deps" / "mylib" / "main"
        assert (checkout / "README.md").is_file()
        assert _deps_config(project) == {"mylib": "main"}

    def test_clones_the_requested_branch(
        self, tmp_path: Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_git_environment(monkeypatch, env)
        project = _make_project(tmp_path, env)
        dep_origin = _bare_repo(tmp_path, "mylib", env, extra_branch="v2")
        monkeypatch.chdir(project / "repo")

        dep_new.run(DepNewArgs(branch="v2", repo_url=str(dep_origin)))

        checkout = project / "deps" / "mylib" / "v2"
        assert (checkout / "v2.txt").is_file()
        assert not (project / "deps" / "mylib" / "main").exists()
        assert _deps_config(project) == {"mylib": "v2"}

    def test_derives_the_dependency_name_from_the_url(
        self, tmp_path: Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_git_environment(monkeypatch, env)
        project = _make_project(tmp_path, env)
        dep_origin = _bare_repo(tmp_path, "some-lib", env)
        monkeypatch.chdir(project / "repo")

        dep_new.run(DepNewArgs(branch=None, repo_url=str(dep_origin)))

        assert (project / "deps" / "some-lib" / "main").is_dir()

    def test_keeps_unrelated_config_entries(
        self, tmp_path: Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_git_environment(monkeypatch, env)
        project = _make_project(tmp_path, env)
        (project / "config" / "config.toml").write_text(
            '[run]\nrunner = "existing"\n', encoding="utf-8"
        )
        dep_origin = _bare_repo(tmp_path, "mylib", env)
        monkeypatch.chdir(project / "repo")

        dep_new.run(DepNewArgs(branch=None, repo_url=str(dep_origin)))

        with (project / "config" / "config.toml").open("rb") as handle:
            parsed = cast(dict[str, object], tomllib.load(handle))
        assert parsed.get("run") == {"runner": "existing"}
        assert parsed.get("deps") == {"mylib": "main"}

    def test_exits_without_cloning_when_the_dependency_already_exists(
        self, tmp_path: Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_git_environment(monkeypatch, env)
        project = _make_project(tmp_path, env)
        dep_origin = _bare_repo(tmp_path, "mylib", env)
        existing = project / "deps" / "mylib"
        existing.mkdir()
        monkeypatch.chdir(project / "repo")

        with pytest.raises(SystemExit):
            dep_new.run(DepNewArgs(branch=None, repo_url=str(dep_origin)))

        assert list(existing.iterdir()) == []
        assert _deps_config(project) == {}

    def test_removes_the_dependency_directory_when_the_clone_fails(
        self, tmp_path: Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolate_git_environment(monkeypatch, env)
        project = _make_project(tmp_path, env)
        dep_origin = _bare_repo(tmp_path, "mylib", env)
        monkeypatch.chdir(project / "repo")

        with pytest.raises(SystemExit):
            dep_new.run(DepNewArgs(branch="no-such-branch", repo_url=str(dep_origin)))

        assert not (project / "deps" / "mylib").exists()
        assert _deps_config(project) == {}

    def test_records_the_dependency_against_the_current_workspace_branch(
        self, tmp_path: Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run from a branch workspace: the branch's config file gets the entry."""
        _isolate_git_environment(monkeypatch, env)
        project = _make_project(tmp_path, env)
        workspace = project / "worktrees" / "feat"
        subprocess.run(
            ["git", "worktree", "add", "-b", "feat", str(workspace), "-q"],
            cwd=project / "repo",
            env=env,
            check=True,
        )
        dep_origin = _bare_repo(tmp_path, "mylib", env)
        monkeypatch.chdir(workspace)

        dep_new.run(DepNewArgs(branch=None, repo_url=str(dep_origin)))

        assert (project / "deps" / "mylib" / "main" / "README.md").is_file()
        with (project / "config" / "feat" / "config.toml").open("rb") as handle:
            parsed = cast(dict[str, object], tomllib.load(handle))
        assert parsed.get("deps") == {"mylib": "main"}
        assert _deps_config(project) == {}
