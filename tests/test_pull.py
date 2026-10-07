"""Real-git sync behavior through the in-process CLI coverage boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner
from typer.main import get_command

from agm.cli import app
from tests._git_helpers import (
    add_linked_worktree,
    clone_local_remote,
    commit_file,
    git_output,
    git_run,
)


@pytest.mark.parametrize("command", ["fetch", "pull"])
def test_sync_updates_tracking_refs_and_only_pull_merges_checkouts(
    tmp_path: Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "worktrees").mkdir()
    (project / "deps").mkdir()
    (project / "deps" / "not-a-checkout").write_text("ignore this file\n")
    checkouts: list[Path] = []
    original_heads: list[str] = []
    remote_heads: list[str] = []
    for name, clone_name, worktree in [
        ("main", "project/repo", project / "worktrees" / "feature"),
        ("dep", "project/deps/mylib/main", project / "deps" / "mylib" / "feature"),
    ]:
        source, repo = clone_local_remote(
            tmp_path, env, source_name=f"{name}-source", clone_name=clone_name, branches=["feature"]
        )
        add_linked_worktree(repo, worktree, env, branch="feature")
        git_run(repo, ["branch", "--set-upstream-to=origin/feature", "feature"], env)
        for checkout, branch in [(repo, "main"), (worktree, "feature")]:
            checkouts.append(checkout)
            original_heads.append(git_output(checkout, ["rev-parse", "HEAD"], env))
            git_run(source, ["checkout", branch], env)
            commit_file(source, env, name="remote.txt", content=f"{name}/{branch}\n")
            remote_heads.append(git_output(source, ["rev-parse", "HEAD"], env))

    monkeypatch.chdir(project)
    result = CliRunner().invoke(get_command(app), ["sync", command], catch_exceptions=False)

    assert result.exit_code == 0
    for index, checkout in enumerate(checkouts):
        assert git_output(checkout, ["rev-parse", "@{upstream}"], env) == remote_heads[index]
        expected_head = remote_heads[index] if command == "pull" else original_heads[index]
        assert git_output(checkout, ["rev-parse", "HEAD"], env) == expected_head
        assert (checkout / "remote.txt").exists() == (command == "pull")


@pytest.mark.parametrize("command", ["fetch", "pull"])
def test_sync_rejects_non_git_main_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    (tmp_path / "repo").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROJ_DIR", str(tmp_path))

    result = CliRunner().invoke(get_command(app), ["sync", command], catch_exceptions=False)

    assert result.exit_code != 0
    assert not (tmp_path / "repo" / ".git").exists()
