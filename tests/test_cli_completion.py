"""Tests for Typer completion helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import agm.completion as completion
import agm.vcs.git as git_helpers


def test_complete_open_target_includes_repo_and_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.setattr(completion, "_resolve_project_repo_dir", lambda: repo_dir)
    monkeypatch.setattr(git_helpers, "current_branch", lambda repo_dir: "main")
    monkeypatch.setattr(
        git_helpers,
        "worktree_list",
        lambda repo_dir: [
            git_helpers.WorktreeInfo(path=tmp_path / "worktrees" / "feat/a", branch="feat/a"),
            git_helpers.WorktreeInfo(path=tmp_path / "worktrees" / "feat/b", branch="feat/b"),
        ],
    )

    def fake_fetch_output(
        args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        del cwd, env
        ref = args[-1]
        if ref == "refs/heads":
            return 0, "main\nfeature/local\n", ""
        if ref == "refs/remotes/origin":
            return 0, "origin/HEAD\norigin/feature/remote\n", ""
        raise AssertionError(f"unexpected args: {args}")

    monkeypatch.setattr(git_helpers, "fetch_output", fake_fetch_output)

    suggestions = completion.complete_open_target("feat")

    assert suggestions == ["feat/a", "feat/b", "feature/local", "feature/remote"]


def test_complete_open_target_swallows_helper_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.setattr(completion, "_resolve_project_repo_dir", lambda: repo_dir)
    monkeypatch.setattr(git_helpers, "current_branch", lambda repo_dir: "main")
    monkeypatch.setattr(git_helpers, "worktree_list", lambda repo_dir: [])
    monkeypatch.setattr(
        git_helpers,
        "fetch_output",
        lambda args, cwd=None, env=None: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert completion.complete_open_target("f") == []


def test_complete_close_branch_only_returns_worktree_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.setattr(completion, "_resolve_project_repo_dir", lambda: repo_dir)
    monkeypatch.setattr(git_helpers, "current_branch", lambda repo_dir: "main")
    monkeypatch.setattr(
        git_helpers,
        "worktree_list",
        lambda repo_dir: [
            git_helpers.WorktreeInfo(path=tmp_path / "repo", branch="main"),
            git_helpers.WorktreeInfo(path=tmp_path / "worktrees" / "feat/a", branch="feat/a"),
            git_helpers.WorktreeInfo(path=tmp_path / "worktrees" / "feat/b", branch="feat/b"),
        ],
    )

    suggestions = completion.complete_close_branch("feat/")

    assert suggestions == ["feat/a", "feat/b"]


def test_complete_close_branch_infers_branch_name_from_checkout_worktree_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_dir = tmp_path / "project"
    repo_dir = project_dir / "repo"
    worktrees_dir = project_dir / "worktrees"
    repo_dir.mkdir(parents=True)
    worktrees_dir.mkdir()
    monkeypatch.setattr(completion, "_resolve_project_repo_dir", lambda: repo_dir)
    monkeypatch.setattr(git_helpers, "current_branch", lambda repo_dir: "main")
    monkeypatch.setattr(
        git_helpers,
        "worktree_list",
        lambda repo_dir: [
            git_helpers.WorktreeInfo(path=repo_dir, branch="main"),
            git_helpers.WorktreeInfo(path=worktrees_dir / "feat" / "detached", branch=None),
        ],
    )

    assert completion.complete_close_branch("feat/") == ["feat/detached"]


def test_complete_help_path_suggests_subcommands() -> None:
    assert completion.complete_help_path(["wt"], "n") == ["new"]
    assert completion.complete_help_path(["loop"], "n") == ["next"]
    assert completion.complete_help_path([], "o") == ["open"]


def test_complete_dep_name_lists_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deps_dir = tmp_path / "deps"
    (deps_dir / "alpha").mkdir(parents=True)
    (deps_dir / "beta").mkdir()
    monkeypatch.setattr(completion, "_resolve_project_deps_dir", lambda: deps_dir)

    assert completion.complete_dep_name("a") == ["alpha"]


def test_complete_dep_branch_uses_selected_dependency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "deps" / "alpha" / "repo"
    repo_dir.mkdir(parents=True)
    monkeypatch.setattr(
        completion,
        "_resolve_dep_repo",
        lambda dep_name: repo_dir if dep_name == "alpha" else None,
    )

    def fake_fetch_output(
        args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        del cwd, env
        ref = args[-1]
        if ref == "refs/heads":
            return 0, "main\nfeature/local\n", ""
        if ref == "refs/remotes/origin":
            return 0, "origin/feature/remote\n", ""
        raise AssertionError(f"unexpected args: {args}")

    monkeypatch.setattr(git_helpers, "fetch_output", fake_fetch_output)
    monkeypatch.setattr(git_helpers, "worktree_list", lambda repo_dir: [])
    monkeypatch.setattr(git_helpers, "current_branch", lambda repo_dir: "main")

    assert completion.complete_dep_branch(["alpha"], "feature/") == [
        "feature/local",
        "feature/remote",
    ]


def test_complete_dep_target_includes_repo_and_worktrees(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deps_dir = tmp_path / "deps"
    dep_dir = deps_dir / "alpha"
    dep_dir.mkdir(parents=True)
    repo_dir = dep_dir / "repo"
    repo_dir.mkdir()

    monkeypatch.setattr(completion, "_resolve_project_deps_dir", lambda: deps_dir)
    monkeypatch.setattr(
        completion,
        "_resolve_dep_repo",
        lambda dep_name: repo_dir if dep_name == "alpha" else None,
    )
    monkeypatch.setattr(
        git_helpers,
        "worktree_list",
        lambda repo_dir: [
            git_helpers.WorktreeInfo(path=repo_dir, branch="main"),
            git_helpers.WorktreeInfo(path=dep_dir / "feature/x", branch="feature/x"),
        ],
    )
    monkeypatch.setattr(git_helpers, "current_branch", lambda repo_dir: "main")

    assert completion.complete_dep_target("alpha/") == ["alpha/feature/x", "alpha/repo"]


def test_complete_run_command_lists_executables_and_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tool = bin_dir / "npm"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))

    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "package.json").write_text("{}")
    monkeypatch.chdir(cwd)

    assert completion.complete_run_command([], "np") == ["npm"]
    assert completion.complete_run_command(["npm"], "pack") == ["package.json"]


def test_complete_run_command_includes_config_aliases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "")

    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "config" / "config.toml").write_text(
        '[run.ai-review]\nalias = "python"\n[run.publish]\nalias = "uv"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(completion, "current_project_dir", lambda cwd=None: project)

    assert completion.complete_run_command([], "ai") == ["ai-review"]


def test_complete_run_command_swallows_config_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        completion,
        "load_run_config",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("bad")),
    )

    assert completion.complete_run_command([], "x") == []


def test_complete_tmux_session_reads_tmux_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess(
            args=["tmux"],
            returncode=0,
            stdout="alpha\nbeta\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert completion.complete_tmux_session("a") == ["alpha"]


def test_complete_tmux_session_ignores_missing_tmux(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        raise FileNotFoundError("tmux")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert completion.complete_tmux_session("a") == []


def test_complete_tmux_window_reads_tmux_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess(
            args=["tmux"],
            returncode=0,
            stdout="@1\n@2\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert completion.complete_tmux_window("@") == ["@1", "@2"]


def test_complete_tmux_window_ignores_missing_tmux(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        raise FileNotFoundError("tmux")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert completion.complete_tmux_window("@") == []


def test_complete_pane_count_returns_common_values() -> None:
    assert completion.complete_pane_count("1") == ["1", "12", "16"]
