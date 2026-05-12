"""Tests for Typer completion helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import click
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
    assert completion.complete_help_path(["config"], "e") == ["env"]
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


class TestResolveProjectRepoDir:
    def test_returns_none_on_system_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            completion,
            "current_project_dir",
            lambda cwd=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        assert completion._resolve_project_repo_dir() is None

    def test_returns_path_on_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        project = tmp_path / "proj"
        repo = project / "repo"
        repo.mkdir(parents=True)
        monkeypatch.setattr(completion, "current_project_dir", lambda cwd=None: project)
        result = completion._resolve_project_repo_dir()
        assert result == repo


class TestResolveProjectDepsDir:
    def test_returns_none_on_system_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            completion,
            "current_project_dir",
            lambda cwd=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        assert completion._resolve_project_deps_dir() is None

    def test_returns_path_on_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        project = tmp_path / "proj"
        (project / "repo").mkdir(parents=True)
        monkeypatch.setattr(completion, "current_project_dir", lambda cwd=None: project)
        result = completion._resolve_project_deps_dir()
        assert result == project / "deps"


class TestBranchCandidates:
    def test_skips_current_branch_on_system_exit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        monkeypatch.setattr(
            git_helpers,
            "current_branch",
            lambda p, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        monkeypatch.setattr(git_helpers, "worktree_list", lambda p, env=None: [])
        monkeypatch.setattr(
            git_helpers,
            "fetch_output",
            lambda args, cwd=None, env=None: (0, "main\n", ""),
        )
        result = completion._branch_candidates(repo_dir)
        assert "main" in result

    def test_skips_worktree_list_on_system_exit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        monkeypatch.setattr(git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            git_helpers,
            "worktree_list",
            lambda p, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        monkeypatch.setattr(
            git_helpers, "fetch_output", lambda args, cwd=None, env=None: (1, "", "error")
        )
        result = completion._branch_candidates(repo_dir)
        assert "main" in result

    def test_skips_fetch_output_on_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        monkeypatch.setattr(git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(git_helpers, "worktree_list", lambda p, env=None: [])
        monkeypatch.setattr(
            git_helpers, "fetch_output", lambda args, cwd=None, env=None: (1, "", "error")
        )
        result = completion._branch_candidates(repo_dir)
        assert result == {"main"}

    def test_strips_origin_prefix_from_remotes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        monkeypatch.setattr(git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(git_helpers, "worktree_list", lambda p, env=None: [])

        def fake_fetch(args: list[str], cwd: Any = None, env: Any = None) -> tuple[int, str, str]:
            ref = args[-1]
            if ref == "refs/heads":
                return 0, "", ""
            if ref == "refs/remotes/origin":
                return 0, "origin/HEAD\norigin/feature\n", ""
            return 0, "", ""

        monkeypatch.setattr(git_helpers, "fetch_output", fake_fetch)
        result = completion._branch_candidates(repo_dir)
        assert "feature" in result
        assert "origin/HEAD" not in result

    def test_skips_empty_lines_in_refs_heads(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        monkeypatch.setattr(git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(git_helpers, "worktree_list", lambda p, env=None: [])

        def fake_fetch(args: list[str], cwd: Any = None, env: Any = None) -> tuple[int, str, str]:
            ref = args[-1]
            if ref == "refs/heads":
                return 0, "main\n\n", ""  # empty line at end
            return 0, "", ""

        monkeypatch.setattr(git_helpers, "fetch_output", fake_fetch)
        result = completion._branch_candidates(repo_dir)
        # Empty line should not be in candidates
        assert "" not in result


class TestWorktreeBranchCandidates:
    def test_returns_empty_set_when_current_branch_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        monkeypatch.setattr(
            git_helpers,
            "current_branch",
            lambda p, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        result = completion._worktree_branch_candidates(repo_dir)
        assert result == set()

    def test_returns_empty_set_when_worktree_list_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        project = tmp_path / "proj"
        repo_dir = project / "repo"
        repo_dir.mkdir(parents=True)
        monkeypatch.setattr(git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            git_helpers,
            "worktree_list",
            lambda p, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        result = completion._worktree_branch_candidates(repo_dir)
        assert result == set()

    def test_skips_worktree_not_under_worktrees_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        project = tmp_path / "proj"
        repo_dir = project / "repo"
        other_dir = tmp_path / "other"
        repo_dir.mkdir(parents=True)
        other_dir.mkdir()
        monkeypatch.setattr(git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            git_helpers,
            "worktree_list",
            lambda p, env=None: [
                git_helpers.WorktreeInfo(path=other_dir, branch=None),
            ],
        )
        result = completion._worktree_branch_candidates(repo_dir)
        assert result == set()

    def test_excludes_main_branch_from_candidates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        project = tmp_path / "proj"
        repo_dir = project / "repo"
        worktrees_dir = project / "worktrees"
        repo_dir.mkdir(parents=True)
        worktrees_dir.mkdir()
        monkeypatch.setattr(git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            git_helpers,
            "worktree_list",
            lambda p, env=None: [
                git_helpers.WorktreeInfo(path=repo_dir, branch="main"),
                git_helpers.WorktreeInfo(path=worktrees_dir / "feat", branch="feat"),
            ],
        )
        result = completion._worktree_branch_candidates(repo_dir)
        assert "main" not in result
        assert "feat" in result


class TestResolveDepRepo:
    def test_returns_none_when_deps_dir_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(completion, "_resolve_project_deps_dir", lambda: None)
        assert completion._resolve_dep_repo("mylib") is None

    def test_returns_none_when_dep_dir_not_a_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        deps_dir = tmp_path / "deps"
        deps_dir.mkdir()
        monkeypatch.setattr(completion, "_resolve_project_deps_dir", lambda: deps_dir)
        # "mylib" dir doesn't exist
        assert completion._resolve_dep_repo("mylib") is None

    def test_returns_none_when_main_dep_repo_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        deps_dir = tmp_path / "deps"
        dep_dir = deps_dir / "mylib"
        dep_dir.mkdir(parents=True)
        monkeypatch.setattr(completion, "_resolve_project_deps_dir", lambda: deps_dir)
        monkeypatch.setattr(
            completion,
            "main_dep_repo",
            lambda p: (_ for _ in ()).throw(SystemExit(1)),
        )
        assert completion._resolve_dep_repo("mylib") is None


class TestPathCandidates:
    def test_returns_empty_when_base_dir_not_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = completion._path_candidates("nonexistent_dir/")
        assert result == []

    def test_returns_files_with_trailing_slash_for_dirs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        (work / "mydir").mkdir()
        (work / "myfile.txt").write_text("x")
        monkeypatch.chdir(work)
        result = completion._path_candidates("")
        assert "mydir/" in result
        assert "myfile.txt" in result

    def test_filters_by_prefix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        (work / "alpha").mkdir()
        (work / "beta").mkdir()
        monkeypatch.chdir(work)
        result = completion._path_candidates("a")
        assert "alpha/" in result
        assert "beta/" not in result

    def test_handles_path_outside_cwd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        sub = work / "sub"
        sub.mkdir()
        (sub / "file.txt").write_text("x")
        monkeypatch.chdir(work)
        result = completion._path_candidates("sub/f")
        assert "sub/file.txt" in result


class TestCompleteOpenTarget:
    def test_returns_empty_when_resolve_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(completion, "_resolve_project_repo_dir", lambda: None)
        assert completion.complete_open_target("") == []

    def test_returns_empty_on_system_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            completion,
            "_resolve_project_repo_dir",
            lambda: (_ for _ in ()).throw(SystemExit(1)),
        )
        assert completion.complete_open_target("") == []


class TestCompleteCloseBranch:
    def test_returns_empty_when_resolve_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(completion, "_resolve_project_repo_dir", lambda: None)
        assert completion.complete_close_branch("") == []

    def test_returns_empty_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            completion,
            "_resolve_project_repo_dir",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert completion.complete_close_branch("") == []


class TestCompleteWorktreeBranch:
    def test_returns_empty_when_resolve_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(completion, "_resolve_project_repo_dir", lambda: None)
        assert completion.complete_worktree_branch("") == []

    def test_returns_branches_when_resolve_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        monkeypatch.setattr(completion, "_resolve_project_repo_dir", lambda: repo_dir)
        monkeypatch.setattr(git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(git_helpers, "worktree_list", lambda p, env=None: [])
        monkeypatch.setattr(
            git_helpers, "fetch_output", lambda args, cwd=None, env=None: (0, "main\n", "")
        )
        result = completion.complete_worktree_branch("m")
        assert "main" in result


class TestCompleteDepName:
    def test_returns_empty_when_deps_dir_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(completion, "_resolve_project_deps_dir", lambda: None)
        assert completion.complete_dep_name("") == []

    def test_returns_empty_when_deps_dir_not_a_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # exists but is a file, not a directory
        deps_file = tmp_path / "deps"
        deps_file.write_text("not a dir")
        monkeypatch.setattr(completion, "_resolve_project_deps_dir", lambda: deps_file)
        assert completion.complete_dep_name("") == []

    def test_returns_empty_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            completion,
            "_resolve_project_deps_dir",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert completion.complete_dep_name("") == []


class TestCompleteDepBranch:
    def test_returns_empty_when_no_dep_name_in_args(self) -> None:
        # All args start with '-', so no dep_name is found
        assert completion.complete_dep_branch(["-b"], "") == []

    def test_returns_empty_when_dep_repo_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(completion, "_resolve_dep_repo", lambda name: None)
        assert completion.complete_dep_branch(["mylib"], "") == []

    def test_returns_empty_on_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            completion,
            "_resolve_dep_repo",
            lambda name: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert completion.complete_dep_branch(["mylib"], "") == []


class TestCompleteDepTarget:
    def test_returns_empty_when_deps_dir_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(completion, "_resolve_project_deps_dir", lambda: None)
        assert completion.complete_dep_target("") == []

    def test_returns_empty_when_deps_dir_not_a_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        deps_file = tmp_path / "deps"
        deps_file.write_text("not a dir")
        monkeypatch.setattr(completion, "_resolve_project_deps_dir", lambda: deps_file)
        assert completion.complete_dep_target("") == []

    def test_skips_dep_when_repo_dir_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        deps_dir = tmp_path / "deps"
        (deps_dir / "mylib").mkdir(parents=True)
        monkeypatch.setattr(completion, "_resolve_project_deps_dir", lambda: deps_dir)
        monkeypatch.setattr(completion, "_resolve_dep_repo", lambda name: None)
        monkeypatch.setattr(
            git_helpers, "worktree_list", lambda p, env=None: []
        )
        monkeypatch.setattr(git_helpers, "current_branch", lambda p, env=None: "main")
        result = completion.complete_dep_target("")
        # dep itself is still a candidate
        assert "mylib" in result
        # but mylib/repo should NOT be there since repo is None
        assert "mylib/repo" not in result

    def test_returns_empty_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            completion,
            "_resolve_project_deps_dir",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert completion.complete_dep_target("") == []


class TestCompleteRunCommand:
    def test_handles_system_exit_in_proj_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """complete_run_command gracefully handles OSError/SystemExit from current_project_dir."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PATH", "")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(
            completion,
            "current_project_dir",
            lambda cwd=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        # Should still return [] without crashing
        result = completion.complete_run_command([], "")
        assert result == []

    def test_handles_oserror_in_load_run_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """complete_run_command handles OSError from load_run_config."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PATH", "")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(completion, "current_project_dir", lambda cwd=None: tmp_path)
        monkeypatch.setattr(
            completion,
            "load_run_config",
            lambda **kwargs: (_ for _ in ()).throw(OSError("io")),
        )
        result = completion.complete_run_command([], "")
        assert result == []


class TestCompleteTmuxSession:
    def test_returns_empty_when_returncode_nonzero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["tmux"], returncode=1, stdout="", stderr="error"
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert completion.complete_tmux_session("") == []


class TestCompleteTmuxWindow:
    def test_returns_empty_when_returncode_nonzero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["tmux"], returncode=1, stdout="", stderr="error"
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert completion.complete_tmux_window("") == []


class TestCompletePathArgument:
    def test_calls_path_candidates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        (work / "mydir").mkdir()
        monkeypatch.chdir(work)
        ctx = click.Context(click.Command("test"))
        result = completion.complete_path_argument(ctx, [], "my")
        assert "mydir/" in result

    def test_returns_empty_on_exception(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            completion,
            "_path_candidates",
            lambda incomplete: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        ctx = click.Context(click.Command("test"))
        result = completion.complete_path_argument(ctx, [], "x")
        assert result == []


class TestCompleteReviseCommandOrReviewFile:
    def test_prefers_config_command_names(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        monkeypatch.setattr(completion, "current_project_dir", lambda cwd=None: None)
        monkeypatch.setattr(
            completion,
            "load_merged_config",
            lambda **kwargs: {
                "revise": {
                    "runner": "codex exec",
                    "frontend": {"prompt": "fix ui"},
                    "backend": {"prompt": "fix api"},
                }
            },
        )
        ctx = click.Context(click.Command("test"))

        result = completion.complete_revise_command_or_review_file(ctx, [], "fr")

        assert result == ["frontend"]

    def test_falls_back_to_paths_when_no_command_matches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        work = tmp_path / "work"
        work.mkdir()
        (work / "review.md").write_text("review\n", encoding="utf-8")
        monkeypatch.chdir(work)
        monkeypatch.setattr(completion, "current_project_dir", lambda cwd=None: None)
        monkeypatch.setattr(
            completion,
            "load_merged_config",
            lambda **kwargs: {"revise": {"frontend": {"prompt": "fix ui"}}},
        )
        ctx = click.Context(click.Command("test"))

        result = completion.complete_revise_command_or_review_file(ctx, [], "rev")

        assert result == ["review.md"]


class TestBranchCandidatesExceptionHandling:
    def test_skips_current_branch_on_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        monkeypatch.setattr(
            git_helpers,
            "current_branch",
            lambda p, env=None: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr(
            completion,
            "_resolve_project_repo_dir",
            lambda: repo_dir,
        )
        result = completion.complete_open_target("")
        # RuntimeError caught by complete_open_target's (Exception, SystemExit)
        assert result == []


class TestWorktreeBranchCandidatesCoverage:
    def test_worktree_with_null_branch_and_relative_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """worktree with branch=None resolves branch from relative path."""
        project = tmp_path / "proj"
        repo_dir = project / "repo"
        worktrees_dir = project / "worktrees"
        feat_dir = worktrees_dir / "feat"
        repo_dir.mkdir(parents=True)
        worktrees_dir.mkdir()
        feat_dir.mkdir()
        monkeypatch.setattr(git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            git_helpers,
            "worktree_list",
            lambda p, env=None: [
                git_helpers.WorktreeInfo(path=feat_dir, branch=None),
            ],
        )
        result = completion._worktree_branch_candidates(repo_dir)
        assert "feat" in result


class TestCompleteDepTargetExceptionHandling:
    def test_returns_dep_name_and_branch_when_repo_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        deps_dir = tmp_path / "deps"
        (deps_dir / "mylib").mkdir(parents=True)
        monkeypatch.setattr(completion, "_resolve_project_deps_dir", lambda: deps_dir)
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        monkeypatch.setattr(
            completion,
            "_resolve_dep_repo",
            lambda name: repo_dir,
        )
        monkeypatch.setattr(git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(git_helpers, "worktree_list", lambda p, env=None: [])
        monkeypatch.setattr(
            git_helpers, "fetch_output", lambda args, cwd=None, env=None: (0, "main\n", "")
        )
        result = completion.complete_dep_target("")
        assert "mylib" in result
        assert "mylib/repo" in result


class TestCompletePaneCountExceptionHandling:
    def test_returns_empty_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            completion,
            "_match",
            lambda candidates, incomplete: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert completion.complete_pane_count("") == []


class TestCompleteDepTargetRepoSuffix:
    def test_includes_repo_suffix_when_dep_repo_resolved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When _resolve_dep_repo returns a path, dep_name/repo is added."""
        deps_dir = tmp_path / "deps"
        dep_dir = deps_dir / "mylib"
        dep_dir.mkdir(parents=True)
        repo_dir = dep_dir / "repo"
        repo_dir.mkdir()

        monkeypatch.setattr(completion, "_resolve_project_deps_dir", lambda: deps_dir)
        monkeypatch.setattr(
            completion,
            "_resolve_dep_repo",
            lambda dep_name: repo_dir if dep_name == "mylib" else None,
        )
        monkeypatch.setattr(git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(git_helpers, "worktree_list", lambda p, env=None: [])

        result = completion.complete_dep_target("mylib/repo")
        assert "mylib/repo" in result


class TestCompletePaneCountException:
    def test_returns_empty_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """complete_pane_count returns [] when _match raises."""
        def bad_match(
            candidates: set[str] | list[str], incomplete: str
        ) -> list[str]:
            raise RuntimeError("boom")

        monkeypatch.setattr(completion, "_match", bad_match)
        assert completion.complete_pane_count("1") == []


class TestPathCandidatesWithParentComponent:
    def test_incomplete_path_with_parent_sets_base_dir_and_prefix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When incomplete has a parent component (e.g. 'sub/prefix'),
        base_dir is set to the parent and prefix to the filename."""
        work = tmp_path / "work"
        work.mkdir()
        sub = work / "sub"
        sub.mkdir()
        (sub / "alpha.txt").write_text("x")
        (sub / "beta.txt").write_text("x")
        monkeypatch.chdir(work)

        # Pass incomplete="sub/a" which has parent="sub" != "."
        result = completion._path_candidates("sub/a")
        assert any("alpha.txt" in r for r in result)
        assert not any("beta.txt" in r for r in result)

    def test_nonexistent_base_dir_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When the resolved base_dir doesn't exist as a dir, return []."""
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)

        # "nonexistent_dir/prefix" => base_dir = work / "nonexistent_dir" which doesn't exist
        result = completion._path_candidates("nonexistent_dir/prefix")
        assert result == []

    def test_symlink_outside_cwd_triggers_value_error_display(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When iterdir finds a path whose resolve is outside cwd,
        ValueError in relative_to causes absolute path display."""
        work = tmp_path / "work"
        work.mkdir()
        # Create a symlink inside work that points outside work
        target_dir = tmp_path / "outside"
        target_dir.mkdir()
        (target_dir / "linked_item").write_text("x")

        link = work / "mylink"
        link.symlink_to(target_dir)
        monkeypatch.chdir(work)

        result = completion._path_candidates("")
        # The symlink should appear as "mylink/"
        assert "mylink/" in result


class TestCompleteHelpPathExceptionHandler:
    def test_returns_empty_on_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """complete_help_path returns [] when _match raises RuntimeError."""
        monkeypatch.setattr(
            completion,
            "_match",
            lambda candidates, incomplete: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert completion.complete_help_path(["wt"], "n") == []


class TestCompleteDepTargetExceptionHandler:
    def test_returns_empty_on_runtime_error_in_branch_candidates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """complete_dep_target returns [] when internal call raises."""
        deps_dir = tmp_path / "deps"
        (deps_dir / "alpha").mkdir(parents=True)
        (deps_dir / "alpha" / "repo").mkdir(parents=True)

        monkeypatch.setattr(completion, "_resolve_project_deps_dir", lambda: deps_dir)
        monkeypatch.setattr(
            completion,
            "_resolve_dep_repo",
            lambda dep_name: deps_dir / dep_name / "repo",
        )
        # Make _worktree_branch_candidates raise RuntimeError
        monkeypatch.setattr(
            completion,
            "_worktree_branch_candidates",
            lambda repo_dir: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        result = completion.complete_dep_target("")
        assert result == []


class TestCompleteWorktreeBranchExceptionHandler:
    def test_returns_empty_when_branch_candidates_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """complete_worktree_branch returns [] when _branch_candidates raises."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        monkeypatch.setattr(completion, "_resolve_project_repo_dir", lambda: repo_dir)
        # Make _branch_candidates raise
        monkeypatch.setattr(
            completion,
            "_branch_candidates",
            lambda repo_dir: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert completion.complete_worktree_branch("m") == []


class TestPathCandidatesValueError:
    def test_symlink_base_dir_outside_cwd_triggers_value_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When the resolved base_dir is outside cwd, paths found by iterdir
        raise ValueError in relative_to, causing absolute path display."""
        work = tmp_path / "work"
        work.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "target.txt").write_text("x")

        # Create a symlink inside work pointing outside
        link = work / "ext"
        link.symlink_to(outside)
        monkeypatch.chdir(work)

        # incomplete="ext/t" => parent="ext" => base_dir = resolve(work/ext) = outside
        # Iterating outside/ finds target.txt which is not relative to work
        result = completion._path_candidates("ext/t")
        # Should use absolute path for display
        assert len(result) == 1
        assert result[0] == str(outside / "target.txt")
