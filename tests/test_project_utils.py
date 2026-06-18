"""Tests for shared project path resolution helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Never

import pytest

import agm.project.layout as project_helpers
import agm.vcs.git as git_helpers
from agm.project.dependency_env import current_config_branch
from agm.project.layout import (
    branch_session_name,
    branch_worktree_path,
    current_workspace,
    discover_current_project_dir,
    is_main_workspace_branch,
    main_repo_dir,
)
from agm.project.workspace_env import load_current_workspace_env, load_workspace_env


def test_current_project_dir_from_project_root(tmp_path: Path, env: dict[str, str]) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    (project / "worktrees").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=project / "repo", env=env, check=True)

    assert discover_current_project_dir(project) == project


def test_current_project_dir_from_repo_dir(tmp_path: Path, env: dict[str, str]) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    repo_dir = project / "repo"
    repo_dir.mkdir()
    (project / "worktrees").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)

    assert discover_current_project_dir(repo_dir) == project


def test_current_project_dir_from_repo_subdir(tmp_path: Path, env: dict[str, str]) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    repo_subdir = project / "repo" / "src"
    repo_subdir.mkdir(parents=True)
    (project / "worktrees").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=project / "repo", env=env, check=True)

    assert discover_current_project_dir(repo_subdir) == project


def test_current_project_dir_from_worktree_dir(tmp_path: Path, env: dict[str, str]) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    branch_dir = project / "worktrees" / "feat" / "branch"
    branch_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=project / "repo", env=env, check=True)

    assert discover_current_project_dir(branch_dir) == project


def test_current_project_dir_from_embedded_project_subdir(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    agm_dir = project / ".agm"
    agm_dir.mkdir()
    subdir = project / "src"
    subdir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=project, env=env, check=True)

    assert discover_current_project_dir(subdir) == agm_dir


def test_main_repo_dir_for_embedded_project(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".agm").mkdir()

    assert main_repo_dir(project) == project


def test_main_workspace_branch_helpers_for_repo_name(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    (project / "worktrees").mkdir()

    assert is_main_workspace_branch(project, "repo", repo_branch="main") is True
    assert branch_worktree_path(project, "repo", repo_branch="main") == project / "repo"
    assert branch_session_name(project, "repo") == "proj"


def test_main_workspace_branch_helpers_for_repo_current_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    (project / "worktrees").mkdir()

    def fake_current_branch(_repo_dir: Path, *, env: dict[str, str] | None = None) -> str:
        del env
        return "main"

    monkeypatch.setattr(project_helpers.git_helpers, "current_branch", fake_current_branch)

    assert is_main_workspace_branch(project, "main", repo_branch="main") is True
    assert branch_worktree_path(project, "main", repo_branch="main") == project / "repo"
    assert branch_session_name(project, "main") == "proj"


def test_branch_helpers_for_worktree_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    (project / "worktrees").mkdir()

    def fake_current_branch(_repo_dir: Path, *, env: dict[str, str] | None = None) -> str:
        del env
        return "main"

    monkeypatch.setattr(project_helpers.git_helpers, "current_branch", fake_current_branch)

    assert is_main_workspace_branch(project, "feat/x", repo_branch="main") is False
    assert (
        branch_worktree_path(
            project,
            "feat/x",
            repo_branch="main",
        )
        == project / "worktrees" / "feat/x"
    )
    assert branch_session_name(project, "feat/x") == "proj/feat/x"


def test_branch_session_name_for_repo_name_does_not_need_repo_branch_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    (project / "worktrees").mkdir()

    def fail_current_branch(_repo_dir: Path, *, env: dict[str, str] | None = None) -> Never:
        del env
        raise AssertionError("current_branch should not be called for repo alias")

    monkeypatch.setattr(project_helpers.git_helpers, "current_branch", fail_current_branch)

    assert branch_session_name(project, "repo") == "proj"


def test_load_workspace_env_exposes_repo_dir_to_sourced_scripts(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    workspace_dir = project / "worktrees" / "feat"
    workspace_dir.mkdir(parents=True)
    (config_dir / "env.sh").write_text(
        'export CAPTURE_PROJ_DIR="$PROJ_DIR"\nexport CAPTURE_REPO_DIR="$REPO_DIR"\n',
        encoding="utf-8",
    )

    loaded_env = load_workspace_env(project, "feat", workspace_dir=workspace_dir, env=env)

    assert loaded_env["PROJ_DIR"] == str(project)
    assert loaded_env["REPO_DIR"] == str(workspace_dir)
    assert loaded_env["CAPTURE_PROJ_DIR"] == str(project)
    assert loaded_env["CAPTURE_REPO_DIR"] == str(workspace_dir)


def test_load_workspace_env_overrides_existing_env_from_env_sh(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    workspace_dir = project / "repo"
    workspace_dir.mkdir()
    env["HOLDIR"] = "/before"
    (config_dir / "env.sh").write_text(
        'export HOLDIR="$PROJ_DIR/hold"\n',
        encoding="utf-8",
    )

    loaded_env = load_workspace_env(project, None, workspace_dir=workspace_dir, env=env)

    assert loaded_env["HOLDIR"] == f"{project}/hold"


def test_current_config_branch_ignores_cwd_from_other_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    other_project = tmp_path / "other"
    current = other_project / "repo"
    current.mkdir(parents=True)

    monkeypatch.setattr(
        project_helpers,
        "discover_current_project_dir",
        lambda _cwd, env=None: other_project,
    )

    def fail_checkout_root(_cwd: Path | None = None) -> Never:
        raise AssertionError("checkout_root should not be called for another project")

    monkeypatch.setattr(git_helpers, "checkout_root", fail_checkout_root)

    assert current_config_branch(project, cwd=current) is None


def test_load_workspace_env_applies_dotenv_precedence_before_env_sh(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    config_dir = project / "config"
    branch_config_dir = config_dir / "feat"
    branch_config_dir.mkdir(parents=True)
    workspace_dir = project / "worktrees" / "feat"
    workspace_dir.mkdir(parents=True)

    (config_dir / ".env").write_text("SHARED=project-dotenv\nPROJECT_ONLY=1\n", encoding="utf-8")
    (config_dir / ".env.local").write_text(
        "SHARED=project-local\nPROJECT_LOCAL_ONLY=1\n",
        encoding="utf-8",
    )
    (config_dir / "env.sh").write_text(
        'export PROJECT_ENV_SH="$SHARED"\nexport SHARED="project-env-sh"\n',
        encoding="utf-8",
    )

    (branch_config_dir / ".env").write_text(
        "SHARED=branch-dotenv\nBRANCH_ONLY=1\n",
        encoding="utf-8",
    )
    (branch_config_dir / ".env.local").write_text(
        "SHARED=branch-local\nBRANCH_LOCAL_ONLY=1\n",
        encoding="utf-8",
    )
    (branch_config_dir / "env.sh").write_text(
        'export BRANCH_ENV_SH="$SHARED"\nexport SHARED="branch-env-sh"\n',
        encoding="utf-8",
    )

    loaded_env = load_workspace_env(project, "feat", workspace_dir=workspace_dir, env=env)

    assert loaded_env["PROJECT_ONLY"] == "1"
    assert loaded_env["PROJECT_LOCAL_ONLY"] == "1"
    assert loaded_env["BRANCH_ONLY"] == "1"
    assert loaded_env["BRANCH_LOCAL_ONLY"] == "1"
    assert loaded_env["PROJECT_ENV_SH"] == "project-local"
    assert loaded_env["BRANCH_ENV_SH"] == "branch-local"
    assert loaded_env["SHARED"] == "branch-env-sh"


def test_load_current_workspace_env_uses_current_project_workspace(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    repo_dir = project / "repo"
    config_dir = project / "config"
    repo_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)
    config_dir.mkdir(parents=True)
    (project / "worktrees").mkdir()
    (config_dir / ".env").write_text("FROM_DOTENV=1\n", encoding="utf-8")
    (config_dir / "env.sh").write_text(
        'export FROM_ENV_SH="$FROM_DOTENV:$REPO_DIR"\n',
        encoding="utf-8",
    )

    loaded_env = load_current_workspace_env(cwd=project, env=env)

    assert loaded_env["PROJ_DIR"] == str(project)
    assert loaded_env["REPO_DIR"] == str(repo_dir)
    assert loaded_env["FROM_DOTENV"] == "1"
    assert loaded_env["FROM_ENV_SH"] == f"1:{repo_dir}"


# --- current_workspace ---


def test_current_workspace_returns_none_for_cwd_outside_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    other_project = tmp_path / "other"
    current = other_project / "repo"
    current.mkdir(parents=True)

    monkeypatch.setattr(
        project_helpers,
        "discover_current_project_dir",
        lambda _cwd, env=None: other_project,
    )

    def fail_checkout_root(_cwd: Path | None = None) -> Never:
        raise AssertionError("checkout_root should not be called for another project")

    monkeypatch.setattr(git_helpers, "checkout_root", fail_checkout_root)

    assert current_workspace(project, cwd=current) is None


def test_current_workspace_returns_main_for_workspace_project_root(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    repo_dir = project / "repo"
    repo_dir.mkdir(parents=True)
    (project / "worktrees").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)

    result = current_workspace(project, cwd=project, env=env)

    assert result is not None
    assert result.is_main is True
    assert result.branch is None
    assert result.workspace_dir == repo_dir


def test_current_workspace_returns_main_for_repo_dir(tmp_path: Path, env: dict[str, str]) -> None:
    project = tmp_path / "proj"
    repo_dir = project / "repo"
    repo_dir.mkdir(parents=True)
    (project / "worktrees").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)

    result = current_workspace(project, cwd=repo_dir, env=env)

    assert result is not None
    assert result.is_main is True
    assert result.branch is None
    assert result.workspace_dir == repo_dir


def test_current_workspace_returns_branch_for_worktree(tmp_path: Path, env: dict[str, str]) -> None:
    project = tmp_path / "proj"
    repo_dir = project / "repo"
    repo_dir.mkdir(parents=True)
    (project / "worktrees").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)
    # Need an initial commit before creating worktrees
    subprocess.run(
        ["git", "-C", str(repo_dir), "commit", "--allow-empty", "-m", "init"],
        env=env,
        check=True,
    )
    # Create a worktree
    worktree_dir = project / "worktrees" / "feat"
    subprocess.run(
        ["git", "-C", str(repo_dir), "worktree", "add", "-b", "feat", str(worktree_dir)],
        env=env,
        check=True,
    )

    result = current_workspace(project, cwd=worktree_dir, env=env)

    assert result is not None
    assert result.is_main is False
    assert result.branch == "feat"
    assert result.workspace_dir == worktree_dir


def test_current_workspace_uses_repo_dir_env_var_for_main_workspace(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    repo_dir = project / "repo"
    repo_dir.mkdir(parents=True)
    (project / "worktrees").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)
    # CWD is somewhere else entirely, but REPO_DIR points to the main repo
    other_dir = tmp_path / "somewhere"
    other_dir.mkdir()
    env_with_repo = {**env, "REPO_DIR": str(repo_dir)}

    result = current_workspace(project, cwd=other_dir, env=env_with_repo)

    assert result is not None
    assert result.is_main is True
    assert result.branch is None
    assert result.workspace_dir == repo_dir


def test_current_workspace_uses_repo_dir_env_var_for_worktree(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    repo_dir = project / "repo"
    repo_dir.mkdir(parents=True)
    (project / "worktrees").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)
    subprocess.run(
        ["git", "-C", str(repo_dir), "commit", "--allow-empty", "-m", "init"],
        env=env,
        check=True,
    )
    worktree_dir = project / "worktrees" / "feat"
    subprocess.run(
        ["git", "-C", str(repo_dir), "worktree", "add", "-b", "feat", str(worktree_dir)],
        env=env,
        check=True,
    )
    # CWD is somewhere else, but REPO_DIR points to the worktree
    other_dir = tmp_path / "somewhere"
    other_dir.mkdir()
    env_with_repo = {**env, "REPO_DIR": str(worktree_dir)}

    result = current_workspace(project, cwd=other_dir, env=env_with_repo)

    assert result is not None
    assert result.is_main is False
    assert result.branch == "feat"
    assert result.workspace_dir == worktree_dir


def test_current_workspace_ignores_repo_dir_outside_project(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    repo_dir = project / "repo"
    repo_dir.mkdir(parents=True)
    (project / "worktrees").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)
    # REPO_DIR points outside the project — should be ignored
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=outside_dir, env=env, check=True)
    env_with_outside = {**env, "REPO_DIR": str(outside_dir)}

    result = current_workspace(project, cwd=repo_dir, env=env_with_outside)

    assert result is not None
    assert result.is_main is True
    assert result.branch is None
    # Falls back to cwd-based detection instead of REPO_DIR
    assert result.workspace_dir == repo_dir


def test_current_workspace_ignores_missing_repo_dir(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    repo_dir = project / "repo"
    repo_dir.mkdir(parents=True)
    (project / "worktrees").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)
    env_with_bad = {**env, "REPO_DIR": "/nonexistent/path"}

    result = current_workspace(project, cwd=repo_dir, env=env_with_bad)

    assert result is not None
    assert result.is_main is True
    assert result.branch is None
    assert result.workspace_dir == repo_dir


def test_current_workspace_ignores_non_git_repo_dir(tmp_path: Path, env: dict[str, str]) -> None:
    project = tmp_path / "proj"
    repo_dir = project / "repo"
    repo_dir.mkdir(parents=True)
    (project / "worktrees").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)
    # REPO_DIR points to a non-git dir inside the project
    non_git = project / "some-dir"
    non_git.mkdir()
    env_with_non_git = {**env, "REPO_DIR": str(non_git)}

    result = current_workspace(project, cwd=repo_dir, env=env_with_non_git)

    assert result is not None
    assert result.is_main is True
    assert result.branch is None
    assert result.workspace_dir == repo_dir


def test_current_workspace_subdir_under_repo_dir_is_main(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    repo_dir = project / "repo"
    repo_dir.mkdir(parents=True)
    (project / "worktrees").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)
    # A subdirectory of repo_dir — still main workspace
    sub_dir = repo_dir / "src"
    sub_dir.mkdir()

    result = current_workspace(project, cwd=sub_dir, env=env)

    assert result is not None
    assert result.is_main is True
    assert result.branch is None
