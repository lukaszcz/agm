"""CLI state and history tests for ``agm config update``."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agm.cli import app
from tests._git_helpers import add_linked_worktree, git_output, git_run, init_repo


@pytest.mark.parametrize("versioned", [False, True], ids=["unversioned", "versioned"])
def test_config_update_refreshes_checkouts_and_preserves_unrelated_files(
    tmp_path: Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch, versioned: bool
) -> None:
    project = tmp_path / "project"
    repo = init_repo(project / "repo", env)
    add_linked_worktree(repo, project / "worktrees" / "feat" / "app", env, branch="feat/app")
    git_run(repo, ["branch", "not-checked-out"], env)
    dep = init_repo(project / "deps" / "mylib" / "main", env)
    add_linked_worktree(dep, dep.parent / "feat" / "app", env, branch="feat/app")
    config = project / "config"
    if versioned:
        init_repo(config, env)
    else:
        config.mkdir()
    local = config / "local.txt"
    local.write_text("keep unstaged\n", encoding="utf-8")
    main_config = config / "config.toml"
    main_config.write_text('# keep this comment\n[deps]\nmylib = "stale"\n', encoding="utf-8")
    before = git_output(config, ["rev-parse", "HEAD"], env) if versioned else None
    monkeypatch.chdir(repo)

    result = CliRunner().invoke(app, ["config", "update"], env=env, catch_exceptions=False)

    assert result.exit_code == 0, result.output
    workspace_config = config / "feat" / "app" / "config.toml"
    assert tomllib.loads(main_config.read_text())["deps"] == {"mylib": "main"}
    assert tomllib.loads(workspace_config.read_text())["deps"] == {"mylib": "feat/app"}
    assert main_config.read_text().startswith("# keep this comment\n")
    assert not (config / "not-checked-out" / "config.toml").exists()
    assert local.read_text() == "keep unstaged\n"
    if versioned:
        after = git_output(config, ["rev-parse", "HEAD"], env)
        assert after != before
        assert set(
            git_output(config, ["show", "--pretty=", "--name-only", "HEAD"], env).splitlines()
        ) == {
            "config.toml",
            "feat/app/config.toml",
        }
        assert git_output(config, ["status", "--short"], env) == "?? local.txt"
        assert (
            git_output(config, ["show", "HEAD:feat/app/config.toml"], env)
            == workspace_config.read_text().strip()
        )

        repeated = CliRunner().invoke(app, ["config", "update"], env=env, catch_exceptions=False)

        assert repeated.exit_code == 0, repeated.output
        assert git_output(config, ["rev-parse", "HEAD"], env) == after
