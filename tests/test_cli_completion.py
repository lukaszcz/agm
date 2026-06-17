"""Tests for Typer completion helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast

import click
import pytest
import typer
from click.shell_completion import ShellComplete

import agm.completion as completion
import agm.vcs.git as git_helpers


def _make_ctx(**params: Any) -> click.Context:
    """Create a Click Context with the given params for completion testing."""
    ctx = click.Context(click.Command("test"))
    ctx.params.update(params)
    return ctx


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
    assert completion.complete_help_path(_make_ctx(help_command=["wt"]), "n") == ["new"]
    assert completion.complete_help_path(_make_ctx(help_command=["loop"]), "s") == [
        "select",
        "step",
    ]
    assert completion.complete_help_path(_make_ctx(help_command=["config"]), "e") == ["env"]
    assert completion.complete_help_path(_make_ctx(help_command=[]), "o") == ["open"]
    assert completion.complete_help_path(_make_ctx(), "o") == ["open"]
    assert completion.complete_help_path(_make_ctx(), "p") == ["pull"]


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

    assert completion.complete_dep_branch(_make_ctx(dep="alpha"), "feature/") == [
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

    assert completion.complete_run_command(_make_ctx(), "np") == ["npm"]
    assert completion.complete_run_command(
        _make_ctx(run_command_args=["npm"]), "pack"
    ) == ["package.json"]


def test_complete_run_command_lists_executables_without_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tool = bin_dir / "npm"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("PROJ_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    assert completion.complete_run_command(_make_ctx(), "np") == ["npm"]


def test_complete_run_command_includes_config_aliases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("PROJ_DIR", raising=False)

    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    project = tmp_path / "project"
    (project / "repo").mkdir(parents=True)
    (project / "config").mkdir(parents=True)
    (project / "config" / "config.toml").write_text(
        '[run.ai-review]\nalias = "python"\n[run.publish]\nalias = "uv"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "agm.config.context.discover_current_project_dir", lambda cwd=None, env=None: project
    )

    assert completion.complete_run_command(_make_ctx(), "ai") == ["ai-review"]


def test_complete_run_command_swallows_config_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(
        completion,
        "load_run_config",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("bad")),
    )

    assert completion.complete_run_command(_make_ctx(), "x") == []


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
            "discover_current_project_dir",
            lambda cwd=None, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        assert completion._resolve_project_repo_dir() is None

    def test_returns_none_when_project_is_not_discovered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            completion, "discover_current_project_dir", lambda cwd=None, env=None: None
        )
        assert completion._resolve_project_repo_dir() is None

    def test_returns_path_on_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        repo = project / "repo"
        repo.mkdir(parents=True)
        monkeypatch.setattr(
            completion, "discover_current_project_dir", lambda cwd=None, env=None: project
        )
        result = completion._resolve_project_repo_dir()
        assert result == repo


class TestResolveProjectDepsDir:
    def test_returns_none_on_system_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            completion,
            "discover_current_project_dir",
            lambda cwd=None, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        assert completion._resolve_project_deps_dir() is None

    def test_returns_none_when_project_is_not_discovered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            completion, "discover_current_project_dir", lambda cwd=None, env=None: None
        )
        assert completion._resolve_project_deps_dir() is None

    def test_returns_path_on_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        (project / "repo").mkdir(parents=True)
        monkeypatch.setattr(
            completion, "discover_current_project_dir", lambda cwd=None, env=None: project
        )
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

    def test_dedupes_local_and_remote_with_same_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        monkeypatch.setattr(git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(git_helpers, "worktree_list", lambda p, env=None: [])

        def fake_fetch(args: list[str], cwd: Any = None, env: Any = None) -> tuple[int, str, str]:
            ref = args[-1]
            if ref == "refs/heads":
                return 0, "main\nfeature\n", ""
            if ref == "refs/remotes/origin":
                return 0, "origin/HEAD\norigin/feature\n", ""
            return 0, "", ""

        monkeypatch.setattr(git_helpers, "fetch_output", fake_fetch)
        result = completion._branch_candidates(repo_dir)
        # "feature" appears in both local and remote, but set dedupes
        assert result == {"main", "feature"}


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

    def test_filters_by_prefix(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    def test_returns_empty_when_resolve_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
    def test_returns_empty_when_resolve_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
    def test_returns_empty_when_no_dep_name_in_ctx(self) -> None:
        # dep param not set in context
        assert completion.complete_dep_branch(_make_ctx(dep=""), "") == []
        assert completion.complete_dep_branch(_make_ctx(), "") == []

    def test_returns_empty_when_dep_repo_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(completion, "_resolve_dep_repo", lambda name: None)
        assert completion.complete_dep_branch(_make_ctx(dep="mylib"), "") == []

    def test_returns_empty_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            completion,
            "_resolve_dep_repo",
            lambda name: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert completion.complete_dep_branch(_make_ctx(dep="mylib"), "") == []

    def test_returns_branches_when_dep_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "deps" / "mylib" / "repo"
        repo_dir.mkdir(parents=True)
        monkeypatch.setattr(
            completion,
            "_resolve_dep_repo",
            lambda name: repo_dir if name == "mylib" else None,
        )
        monkeypatch.setattr(git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(git_helpers, "worktree_list", lambda p, env=None: [])
        monkeypatch.setattr(
            git_helpers,
            "fetch_output",
            lambda args, cwd=None, env=None: (0, "main\nfeature/x\n", ""),
        )
        result = completion.complete_dep_branch(_make_ctx(dep="mylib"), "feature/")
        assert result == ["feature/x"]


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
        monkeypatch.setattr(git_helpers, "worktree_list", lambda p, env=None: [])
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


class TestBranchCandidatesDetachedHead:
    def test_worktree_with_none_branch_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """WorktreeInfo with branch=None (detached HEAD) is not added to candidates."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        monkeypatch.setattr(git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            git_helpers,
            "worktree_list",
            lambda p, env=None: [
                git_helpers.WorktreeInfo(path=tmp_path / "worktrees" / "detached", branch=None),
            ],
        )
        monkeypatch.setattr(
            git_helpers, "fetch_output", lambda args, cwd=None, env=None: (1, "", "")
        )
        result = completion._branch_candidates(repo_dir)
        # Only "main" from current_branch; the detached worktree contributes nothing
        assert result == {"main"}


class TestCompleteRunCommandNonExecutable:
    def test_skips_non_executable_file_in_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Files that are not executable are not suggested as run commands."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        # Create a non-executable file
        non_exec = bin_dir / "notool"
        non_exec.write_text("#!/bin/sh\n")
        non_exec.chmod(0o644)  # readable but not executable
        # Create an executable file as a control
        exec_file = bin_dir / "mytool"
        exec_file.write_text("#!/bin/sh\n")
        exec_file.chmod(0o755)
        monkeypatch.setenv("PATH", str(bin_dir))
        monkeypatch.chdir(tmp_path)

        result = completion.complete_run_command(_make_ctx(), "")

        assert "mytool" in result
        assert "notool" not in result


class TestCompleteRunCommand:
    def test_handles_system_exit_in_proj_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """complete_run_command handles project discovery errors."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PATH", "")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(
            completion,
            "discover_current_project_dir",
            lambda cwd=None, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        # Should still return [] without crashing
        result = completion.complete_run_command(_make_ctx(), "")
        assert result == []

    def test_handles_oserror_in_load_run_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """complete_run_command handles OSError from load_run_config."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PATH", "")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(
            "agm.config.context.discover_current_project_dir", lambda cwd=None, env=None: tmp_path
        )
        monkeypatch.setattr(
            completion,
            "load_run_config",
            lambda **kwargs: (_ for _ in ()).throw(OSError("io")),
        )
        result = completion.complete_run_command(_make_ctx(), "")
        assert result == []

    def test_returns_empty_on_path_iteration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PATH", "/broken")
        monkeypatch.setattr(
            completion,
            "Path",
            lambda _path: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        assert completion.complete_run_command(_make_ctx(), "") == []

    def test_skips_non_directory_path_entries(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path / "missing-bin"))
        monkeypatch.setenv("HOME", str(tmp_path))

        result = completion.complete_run_command(_make_ctx(), "")

        assert result == []


class TestCompleteTmuxSession:
    def test_returns_empty_when_returncode_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["tmux"], returncode=1, stdout="", stderr="error"
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert completion.complete_tmux_session("") == []


class TestCompleteTmuxWindow:
    def test_returns_empty_when_returncode_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["tmux"], returncode=1, stdout="", stderr="error"
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert completion.complete_tmux_window("") == []


class TestCompletePathArgument:
    def test_calls_path_candidates(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    def test_configured_command_names_ignores_missing_section(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(completion, "load_merged_config", lambda **kwargs: {})

        result = completion._configured_command_names(
            "revise",
            home=tmp_path / "home",
            proj_dir=None,
            cwd=tmp_path,
        )

        assert result == set()

    def test_prefers_config_command_names(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        monkeypatch.setattr(
            "agm.config.context.discover_current_project_dir", lambda cwd=None, env=None: None
        )
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
        monkeypatch.setattr(
            "agm.config.context.discover_current_project_dir", lambda cwd=None, env=None: None
        )
        monkeypatch.setattr(
            completion,
            "load_merged_config",
            lambda **kwargs: {"revise": {"frontend": {"prompt": "fix ui"}}},
        )
        ctx = click.Context(click.Command("test"))

        result = completion.complete_revise_command_or_review_file(ctx, [], "rev")

        assert result == ["review.md"]

    def test_handles_project_lookup_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            completion,
            "discover_current_project_dir",
            lambda cwd=None, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        monkeypatch.setattr(
            completion,
            "load_merged_config",
            lambda **kwargs: {"revise": {"frontend": {"prompt": "fix ui"}}},
        )
        ctx = click.Context(click.Command("test"))

        result = completion.complete_revise_command_or_review_file(ctx, [], "fr")

        assert result == ["frontend"]

    def test_returns_empty_on_config_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            completion,
            "load_merged_config",
            lambda **kwargs: (_ for _ in ()).throw(ValueError("bad config")),
        )
        ctx = click.Context(click.Command("test"))

        result = completion.complete_revise_command_or_review_file(ctx, [], "fr")

        assert result == []

    def test_falls_back_to_review_files_when_config_context_exits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        review_file = tmp_path / "frontend-review.md"
        review_file.write_text("review\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            completion,
            "current_config_context",
            lambda: (_ for _ in ()).throw(SystemExit(1)),
        )
        ctx = click.Context(click.Command("test"))

        result = completion.complete_revise_command_or_review_file(ctx, [], "front")

        assert result == ["frontend-review.md"]


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

        def bad_match(candidates: set[str] | list[str], incomplete: str) -> list[str]:
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
    def test_returns_empty_on_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """complete_help_path returns [] when _match raises RuntimeError."""
        monkeypatch.setattr(
            completion,
            "_match",
            lambda candidates, incomplete: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert completion.complete_help_path(_make_ctx(help_command=["wt"]), "n") == []


class TestCompleteHelpPathCtx:
    def test_suggests_top_level_when_no_command(self) -> None:
        assert completion.complete_help_path(_make_ctx(), "o") == ["open"]
        assert completion.complete_help_path(_make_ctx(help_command=[]), "o") == ["open"]

    def test_suggests_subcommands_after_first_command(self) -> None:
        result = completion.complete_help_path(_make_ctx(help_command=["dep"]), "")
        assert result == ["list", "new", "remove", "rm", "switch"]

    def test_no_suggestions_for_leaf_command(self) -> None:
        result = completion.complete_help_path(_make_ctx(help_command=["pull"]), "")
        assert result == []


class TestCompleteRunCommandCtx:
    def test_shows_executables_when_no_command_args(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        tool = bin_dir / "mytool"
        tool.write_text("#!/bin/sh\n")
        tool.chmod(0o755)
        monkeypatch.setenv("PATH", str(bin_dir))
        monkeypatch.chdir(tmp_path)

        assert completion.complete_run_command(_make_ctx(), "my") == ["mytool"]

    def test_shows_path_completions_when_command_args_given(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        (work / "file.txt").write_text("x")
        monkeypatch.chdir(work)

        result = completion.complete_run_command(
            _make_ctx(run_command_args=["python"]), "f"
        )
        assert result == ["file.txt"]


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


class TestCompleteAglFile:
    def test_returns_agl_files(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        work = tmp_path / "work"
        work.mkdir()
        (work / "flow.agl").write_text("")
        (work / "other.py").write_text("")
        (work / "subdir").mkdir()
        monkeypatch.chdir(work)
        ctx = click.Context(click.Command("test"))
        result = completion.complete_agl_file(ctx, [], "")
        assert "flow.agl" in result
        assert "other.py" not in result
        assert "subdir/" in result

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
        result = completion.complete_agl_file(ctx, [], "")
        assert result == []


class TestExecCommandShellComplete:
    """Integration tests: ``ExecCommand.shell_complete`` wired through the real CLI.

    These tests drive completion through the Click/Typer shell_complete API
    (not just the bare helper function) to prove ``--<param>`` options are
    offered end-to-end in a real shell tab-completion session.
    """

    def _get_cli(self) -> click.BaseCommand:
        from agm.cli import app

        return typer.main.get_command(app)

    def _complete(self, args: list[str], incomplete: str) -> list[str]:
        sc = ShellComplete(self._get_cli(), {}, "agm", "_TYPER_COMPLETE_ARGS")
        return [c.value for c in sc.get_completions(args, incomplete)]

    def test_file_param_offers_param_options(self, tmp_path: Path) -> None:
        """``agm exec FILE --<TAB>`` offers ``--<param>`` from the file."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("param msg: text\n")

        result = self._complete(["exec", str(agl_file)], "--")
        assert "--msg" in result
        # Built-in exec options are still offered alongside param options.
        assert "--runner" in result

    def test_file_with_ask_offers_param_options(self, tmp_path: Path) -> None:
        """Completion discovers params for normal exec programs using ``ask``."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('param topic: text\nlet answer = ask "About ${topic as raw}"\n')

        result = self._complete(["exec", str(agl_file)], "--")
        assert "--topic" in result

    def test_bool_param_offers_no_prefix_via_shell_complete(self, tmp_path: Path) -> None:
        """Bool params offer both ``--name`` and ``--no-name`` through shell_complete."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("param verbose: bool\n")

        result = self._complete(["exec", str(agl_file)], "--")
        assert "--verbose" in result
        assert "--no-verbose" in result

    def test_incomplete_prefix_filters_param_options(self, tmp_path: Path) -> None:
        """Only ``--<param>`` options whose name starts with *incomplete* are returned."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("param msg: text\nparam count: int\n")

        result = self._complete(["exec", str(agl_file)], "--m")
        assert "--msg" in result
        assert "--count" not in result

    def test_command_flag_source_offers_param_options(self) -> None:
        """``agm exec -c 'param ...' --<TAB>`` discovers params from inline source."""
        result = self._complete(["exec", "-c", "param count: int"], "--")
        assert "--count" in result

    def test_nonexistent_file_degrades_to_base_completion(self) -> None:
        """Unreadable file degrades to standard exec option completion (no crash)."""
        result = self._complete(["exec", "/nonexistent/prog.agl"], "--")
        # Built-in options should still appear.
        assert "--runner" in result
        # No param options (nothing to discover).
        assert "--count" not in result

    def test_no_file_no_command_returns_base_completion(self) -> None:
        """Without FILE or -c, only built-in exec options are offered."""
        result = self._complete(["exec"], "--")
        assert "--runner" in result


class TestExecParamCompletionItems:
    """Unit tests for ``_exec_param_completion_items``."""

    def test_text_param_returns_completion_item(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("param msg: text\n")
        items = completion._exec_param_completion_items(agl_file.read_text(), "--")
        assert any(item.value == "--msg" for item in items)

    def test_bool_param_returns_both_flags(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("param flag: bool\n")
        items = completion._exec_param_completion_items(agl_file.read_text(), "--")
        values = [item.value for item in items]
        assert "--flag" in values
        assert "--no-flag" in values

    def test_bool_param_no_prefix_excluded_when_outside_filter(self) -> None:
        """Bool --no-flag is excluded when the incomplete prefix does not match it."""
        # prefix "--fl" matches "--flag" but not "--no-flag"
        items = completion._exec_param_completion_items("param flag: bool\n", "--fl")
        values = [item.value for item in items]
        assert "--flag" in values
        assert "--no-flag" not in values

    def test_filters_by_incomplete(self) -> None:
        source = "param apple: text\nparam banana: text\n"
        items = completion._exec_param_completion_items(source, "--a")
        values = [item.value for item in items]
        assert "--apple" in values
        assert "--banana" not in values

    def test_syntax_error_returns_empty(self) -> None:
        items = completion._exec_param_completion_items("@@@ bad syntax", "--")
        assert items == []

    def test_exception_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agm.agl import WorkflowRuntime

        monkeypatch.setattr(
            WorkflowRuntime,
            "prepare",
            staticmethod(lambda source: (_ for _ in ()).throw(RuntimeError("boom"))),
        )
        items = completion._exec_param_completion_items("param x: text\n", "--")
        assert items == []


class TestExecCommandShellCompleteEdgeCases:
    """Coverage for edge branches in ``ExecCommand.shell_complete``."""

    def _get_cli(self) -> click.BaseCommand:
        from agm.cli import app

        return typer.main.get_command(app)

    def _get_exec_cmd(self) -> completion.ExecCommand:
        from agm.cli import app

        cli = typer.main.get_command(app)
        # cli is a TyperGroup (click.Group); .commands is dict[str, click.Command]
        cmd = cast(click.Group, cli).commands["exec"]
        assert isinstance(cmd, completion.ExecCommand)
        return cmd

    def _complete(self, args: list[str], incomplete: str) -> list[str]:
        sc = ShellComplete(self._get_cli(), {}, "agm", "_TYPER_COMPLETE_ARGS")
        return [c.value for c in sc.get_completions(args, incomplete)]

    def test_non_option_incomplete_returns_base_only(self, tmp_path: Path) -> None:
        """When incomplete does not start with '-', shell_complete returns base result only."""
        from click.shell_completion import _resolve_context

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("param msg: text\n")

        cli = self._get_cli()
        exec_cmd = self._get_exec_cmd()
        ctx = _resolve_context(cli, {}, "agm", ["exec", str(agl_file)])
        # Call shell_complete directly with a non-option incomplete
        result = exec_cmd.shell_complete(ctx, "foo")
        # "--msg" must not appear since we short-circuit on non-option prefix
        assert not any(item.value == "--msg" for item in result)

    def test_exception_in_param_discovery_degrades_to_base(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exception inside the extra-items block returns base completion (no crash)."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("param msg: text\n")

        # Patch _exec_param_completion_items to raise
        monkeypatch.setattr(
            completion,
            "_exec_param_completion_items",
            lambda source, incomplete: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        result = self._complete(["exec", str(agl_file)], "--")
        # Built-in options still returned via base completion.
        assert "--runner" in result
        # No param-option items.
        assert "--msg" not in result
