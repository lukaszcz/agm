"""Workspace setup runs real scripts against project and branch configuration."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import agm.project.workspace_env as workspace_env
from agm.core import dry_run
from agm.project.workspace_env import load_current_workspace_env
from agm.project.workspace_setup import run_setup
from tests._git_helpers import add_linked_worktree, init_repo


def _project(tmp_path: Path, env: dict[str, str]) -> tuple[Path, Path]:
    project = tmp_path / "project space λ"
    repo = init_repo(project / "repo", env)
    (project / "config").mkdir()
    return project, repo


def _script(path: Path, body: str, *, executable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\nset -eu\n" + body)
    path.chmod(0o755 if executable else 0o644)


@pytest.mark.parametrize("relative", ["config/setup.sh", "repo/.config/setup.sh", "repo/.setup.sh"])
def test_setup_scripts_receive_the_workspace_environment(
    tmp_path: Path, env: dict[str, str], relative: str
) -> None:
    project, repo = _project(tmp_path, env)
    (project / "config" / ".env").write_text("SETTING=project-value\n")
    _script(
        project / relative,
        'printf "%s\\n" "$PROJ_DIR" "$REPO_DIR" "$PWD" "$SETTING" > observed.txt\n',
    )

    run_setup(cwd=repo, env=env)

    assert (repo / "observed.txt").read_text().splitlines() == [
        str(project),
        str(repo),
        str(repo),
        "project-value",
    ]


def test_setup_uses_branch_configuration_in_the_branch_worktree(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project, repo = _project(tmp_path, env)
    branch = add_linked_worktree(repo, project / "worktrees" / "feature", env, branch="feature")
    (project / "config" / ".env").write_text("SETTING=project-value\n")
    (project / "config" / "feature").mkdir()
    (project / "config" / "feature" / ".env").write_text("SETTING=branch-value\n")
    _script(project / "config" / "setup.sh", 'printf "%s" "$SETTING" > observed.txt\n')

    run_setup(cwd=branch, env=env)

    assert (branch / "observed.txt").read_text() == "branch-value"
    assert not (repo / "observed.txt").exists()


@pytest.mark.parametrize("failure", [False, True], ids=["all-succeed", "second-fails"])
def test_setup_runs_scripts_in_order_and_stops_after_failure(
    tmp_path: Path, env: dict[str, str], failure: bool
) -> None:
    project, repo = _project(tmp_path, env)
    _script(project / "config" / "setup.sh", 'printf "first\\n" > steps.txt\n')
    _script(
        repo / ".config" / "setup.sh",
        "exit 17\n" if failure else 'printf "second\\n" >> steps.txt\n',
    )
    _script(repo / ".setup.sh", 'printf "third\\n" >> steps.txt\n')

    if failure:
        with pytest.raises(SystemExit) as raised:
            run_setup(cwd=repo, env=env)
        assert raised.value.code == 17
    else:
        run_setup(cwd=repo, env=env)

    assert (repo / "steps.txt").read_text().splitlines() == (
        ["first"] if failure else ["first", "second", "third"]
    )


@pytest.mark.parametrize("mode", ["missing", "not-executable", "dry-run"])
def test_setup_does_not_execute_an_unrunnable_or_dry_run_script(
    tmp_path: Path, env: dict[str, str], mode: str, capsys: pytest.CaptureFixture[str]
) -> None:
    project, repo = _project(tmp_path, env)
    if mode != "missing":
        _script(project / "config" / "setup.sh", "touch executed\n", executable=mode == "dry-run")
    dry_run.set_enabled(mode == "dry-run")

    run_setup(cwd=repo, env=env)

    assert not (repo / "executed").exists()
    assert capsys.readouterr().out


def test_setup_still_runs_when_the_display_path_cannot_be_made_relative(
    tmp_path: Path,
    env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project, repo = _project(tmp_path, env)
    script = project / "config" / "setup.sh"
    _script(script, "touch executed\n")
    original = Path.relative_to

    def relative_to(path: Path, *other: str | Path, walk_up: bool = False) -> Path:
        if path == script:
            raise ValueError("unavailable relative path")
        return original(path, *other, walk_up=walk_up)

    monkeypatch.setattr(Path, "relative_to", relative_to)
    run_setup(cwd=repo, env=env)

    assert (repo / "executed").is_file()
    assert str(script) in capsys.readouterr().out


class TestLoadCurrentConfigEnvWithNoResult:
    def test_loads_config_from_outside_the_workspace(
        self, tmp_path: Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "project space λ"
        config = project / "config"
        config.mkdir(parents=True)
        (project / "worktrees").mkdir()
        workspace = init_repo(project / "repo", env)
        monkeypatch.setenv("PROJ_DIR", str(project))
        (config / ".env").write_text("SETTING=project-value\n", encoding="utf-8")
        (config / "env.sh").write_text('export SETUP_CWD="$PWD"\n', encoding="utf-8")

        result = load_current_workspace_env(cwd=tmp_path, env=env)

        assert result["PROJ_DIR"] == str(project)
        assert result["REPO_DIR"] == str(workspace)
        assert result["SETUP_CWD"] == str(workspace)
        assert result["SETTING"] == "project-value"
        assert "SETTING" not in env
        _script(project / "config" / "setup.sh", 'printf "%s" "$SETTING" > observed.txt\n')
        run_setup(cwd=tmp_path, env=env)
        assert (workspace / "observed.txt").read_text() == "project-value"


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
