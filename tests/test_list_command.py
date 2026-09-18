"""Tests for agm.commands.workspace.list."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from typer.main import get_command

import agm.cli as cli
import agm.commands.workspace.list as list_cmd
from tests._git_helpers import add_linked_worktree, git_run, init_repo


def _invoke(runner: CliRunner, argv: list[str]) -> Any:
    return runner.invoke(get_command(cli.app), argv, prog_name="agm")


@pytest.fixture
def project(tmp_path: Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Path:
    """Split project with `fix` and `feat` workspaces and git's environment applied."""

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    project_dir = tmp_path / "proj"
    repo_dir = init_repo(project_dir / "repo", env)
    add_linked_worktree(repo_dir, project_dir / "worktrees" / "fix", env, branch="fix")
    add_linked_worktree(repo_dir, project_dir / "worktrees" / "feat", env, branch="feat")
    return project_dir


def _lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return [line for line in capsys.readouterr().out.splitlines() if line]


def test_lists_main_workspace_first_then_branch_workspaces(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    list_cmd.list_workspaces(cwd=project)

    assert _lines(capsys) == ["* main", "  feat", "  fix"]


def test_verbose_shows_workspace_directories(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    list_cmd.list_workspaces(cwd=project, verbose=True)

    lines = _lines(capsys)
    assert [line[2:].split()[0] for line in lines] == ["main", "feat", "fix"]
    assert lines[0].endswith("repo")
    assert lines[1].endswith(str(Path("worktrees") / "feat"))


@pytest.mark.parametrize(
    ("cwd_parts", "expected"),
    [
        (("worktrees",), ["  main", "  feat", "  fix"]),
        (("worktrees", "feat"), ["  main", "* feat", "  fix"]),
    ],
)
def test_marks_current_workspace(
    project: Path,
    capsys: pytest.CaptureFixture[str],
    cwd_parts: tuple[str, ...],
    expected: list[str],
) -> None:
    list_cmd.list_workspaces(cwd=project.joinpath(*cwd_parts))

    assert _lines(capsys) == expected


def test_ignores_git_worktrees_that_are_not_workspaces(
    project: Path, tmp_path: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    repo_dir = project / "repo"
    add_linked_worktree(repo_dir, tmp_path / "elsewhere", env, branch="outside")
    git_run(repo_dir, ["worktree", "add", "-q", "--detach", str(tmp_path / "detached")], env)

    list_cmd.list_workspaces(cwd=project)

    assert _lines(capsys) == ["* main", "  feat", "  fix"]


def test_shows_detached_and_switched_workspace_state(
    project: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    git_run(project / "repo", ["checkout", "-q", "--detach"], env)
    git_run(project / "worktrees" / "feat", ["checkout", "-q", "--detach"], env)
    git_run(project / "worktrees" / "fix", ["checkout", "-q", "-b", "other"], env)

    list_cmd.list_workspaces(cwd=project)

    assert _lines(capsys) == ["* (detached)", "  feat (detached)", "  fix (on other)"]


def test_embedded_layout_lists_agm_worktrees(
    tmp_path: Path,
    env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    repo_dir = init_repo(tmp_path / "proj", env)
    (repo_dir / ".agm" / "worktrees").mkdir(parents=True)
    add_linked_worktree(repo_dir, repo_dir / ".agm" / "worktrees" / "feat", env, branch="feat")

    list_cmd.list_workspaces(cwd=repo_dir)

    assert _lines(capsys) == ["* main", "  feat"]


def test_run_lists_workspaces_of_the_current_project(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(project / "worktrees" / "fix")

    list_cmd.run()

    assert _lines(capsys) == ["  main", "  feat", "* fix"]


def test_list_cmd_via_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def record(*, verbose: bool = False) -> None:
        calls.append(verbose)

    monkeypatch.setattr(list_cmd, "run", record)
    result = _invoke(CliRunner(), ["workspace", "list", "-v"])
    assert result.exit_code == 0
    assert calls == [True]
