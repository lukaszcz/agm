"""Tests for agm.commands.dep.list."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner, Result
from typer.main import get_command

import agm.cli as cli
import agm.commands.dep.list as dep_list_cmd
from agm.project.dependency_env import read_deps_table


def _invoke(runner: CliRunner, argv: list[str]) -> Result:
    return runner.invoke(get_command(cli.app), argv, prog_name="agm")


def subprocess_run(args: list[str]) -> None:
    subprocess.run(args, check=True, capture_output=True)


def _init_git_repo(path: Path) -> None:
    """Create a minimal git repo at *path* for testing."""
    subprocess_run(["git", "init", str(path)])
    subprocess_run(["git", "-C", str(path), "commit", "--allow-empty", "-m", "init"])


# ---------------------------------------------------------------------------
# read_deps_table (imported from dependency_env)
# ---------------------------------------------------------------------------


class TestReadDepsFromToml:
    def test_reads_valid_toml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "[deps]" + "\n" + 'mylib = "main"' + "\n" + 'other = "develop"' + "\n",
            encoding="utf-8",
        )
        result = read_deps_table(config_file)
        assert result == {"mylib": "main", "other": "develop"}

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "missing.toml"
        result = read_deps_table(config_file)
        assert result == {}

    def test_returns_empty_when_no_deps_table(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "[run]" + "\n" + 'runner = "echo"' + "\n",
            encoding="utf-8",
        )
        result = read_deps_table(config_file)
        assert result == {}

    def test_skips_non_string_values(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "[deps]" + "\n" + 'mylib = "main"' + "\n" + "count = 42" + "\n",
            encoding="utf-8",
        )
        result = read_deps_table(config_file)
        assert result == {"mylib": "main"}

    def test_skips_empty_string_values(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "[deps]"
            + "\n"
            + 'mylib = ""'
            + "\n"
            + 'other = "dev"'
            + "\n",
            encoding="utf-8",
        )
        result = read_deps_table(config_file)
        assert result == {"other": "dev"}

    def test_raises_on_corrupt_toml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text("this is not valid toml {{{", encoding="utf-8")
        import tomlkit.exceptions

        with pytest.raises(tomlkit.exceptions.ParseError):
            read_deps_table(config_file)

    def test_raises_on_unreadable_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text("[deps]\nmylib = \"main\"\n", encoding="utf-8")
        config_file.chmod(0)
        try:
            with pytest.raises(OSError):
                read_deps_table(config_file)
        finally:
            config_file.chmod(0o644)


# ---------------------------------------------------------------------------
# _deps_for_branch
# ---------------------------------------------------------------------------


class TestDepsForBranch:
    def test_main_branch_reads_config_toml(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text(
            "[deps]" + "\n" + 'mylib = "main"' + "\n",
            encoding="utf-8",
        )

        result = dep_list_cmd._deps_for_branch(tmp_path, None)
        assert result == {"mylib": "main"}

    def test_feature_branch_reads_branch_config(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        branch_dir = config_dir / "feature"
        branch_dir.mkdir(parents=True)
        config_file = branch_dir / "config.toml"
        config_file.write_text(
            "[deps]" + "\n" + 'mylib = "feature"' + "\n",
            encoding="utf-8",
        )

        result = dep_list_cmd._deps_for_branch(tmp_path, "feature")
        assert result == {"mylib": "feature"}

    def test_returns_empty_when_no_config_dir(self, tmp_path: Path) -> None:
        result = dep_list_cmd._deps_for_branch(tmp_path, None)
        assert result == {}


# ---------------------------------------------------------------------------
# _list_all_dep_checkouts
# ---------------------------------------------------------------------------


class TestListAllDepCheckouts:
    def test_returns_empty_when_no_deps_dir(self, tmp_path: Path) -> None:
        result = dep_list_cmd._list_all_dep_checkouts(tmp_path / "nonexistent")
        assert result == {}

    def test_returns_empty_when_deps_dir_empty(self, tmp_path: Path) -> None:
        deps_dir = tmp_path / "deps"
        deps_dir.mkdir()
        result = dep_list_cmd._list_all_dep_checkouts(deps_dir)
        assert result == {}

    def test_lists_git_repo_checkouts_per_dep(self, tmp_path: Path) -> None:
        deps_dir = tmp_path / "deps"
        mylib = deps_dir / "mylib"
        (mylib / "main").mkdir(parents=True)
        _init_git_repo(mylib / "main")
        (mylib / "feature").mkdir(parents=True)
        _init_git_repo(mylib / "feature")
        other = deps_dir / "other"
        (other / "develop").mkdir(parents=True)
        _init_git_repo(other / "develop")

        result = dep_list_cmd._list_all_dep_checkouts(deps_dir)
        assert result == {
            "mylib": [("feature", mylib / "feature"), ("main", mylib / "main")],
            "other": [("develop", other / "develop")],
        }

    def test_skips_non_dir_entries(self, tmp_path: Path) -> None:
        deps_dir = tmp_path / "deps"
        deps_dir.mkdir()
        (deps_dir / "readme.txt").write_text("not a dep", encoding="utf-8")
        mylib = deps_dir / "mylib"
        (mylib / "main").mkdir(parents=True)
        _init_git_repo(mylib / "main")

        result = dep_list_cmd._list_all_dep_checkouts(deps_dir)
        assert result == {"mylib": [("main", mylib / "main")]}

    def test_skips_file_entries_in_dep_dir(self, tmp_path: Path) -> None:
        deps_dir = tmp_path / "deps"
        mylib = deps_dir / "mylib"
        (mylib / "main").mkdir(parents=True)
        _init_git_repo(mylib / "main")
        (mylib / "notes.txt").write_text("not a checkout", encoding="utf-8")

        result = dep_list_cmd._list_all_dep_checkouts(deps_dir)
        assert result == {"mylib": [("main", mylib / "main")]}

    def test_skips_non_git_directories(self, tmp_path: Path) -> None:
        deps_dir = tmp_path / "deps"
        mylib = deps_dir / "mylib"
        (mylib / "main").mkdir(parents=True)
        _init_git_repo(mylib / "main")
        (mylib / "notes").mkdir(parents=True)  # not a git repo

        result = dep_list_cmd._list_all_dep_checkouts(deps_dir)
        assert result == {"mylib": [("main", mylib / "main")]}

    def test_finds_nested_dep_checkouts(self, tmp_path: Path) -> None:
        deps_dir = tmp_path / "deps"
        mylib = deps_dir / "mylib"
        nested = mylib / "sub" / "deep"
        nested.mkdir(parents=True)
        _init_git_repo(nested)

        result = dep_list_cmd._list_all_dep_checkouts(deps_dir)
        assert result == {"mylib": [("sub/deep", nested)]}

    def test_dep_dir_with_no_git_checkouts_is_excluded(self, tmp_path: Path) -> None:
        deps_dir = tmp_path / "deps"
        mylib = deps_dir / "mylib"
        (mylib / "notgit").mkdir(parents=True)  # directory but no git repo inside

        result = dep_list_cmd._list_all_dep_checkouts(deps_dir)
        assert result == {}


# ---------------------------------------------------------------------------
# list_deps — current checkout mode (no --all)
# ---------------------------------------------------------------------------


class TestListDepsCurrentCheckout:
    def test_lists_deps_from_branch_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_dir = project_dir / "config"
        branch_dir = config_dir / "feature"
        branch_dir.mkdir(parents=True)
        config_file = branch_dir / "config.toml"
        config_file.write_text(
            "[deps]" + "\n" + 'mylib = "feature"' + "\n" + 'other = "main"' + "\n",
            encoding="utf-8",
        )
        deps_dir = project_dir / "deps"
        (deps_dir / "mylib" / "feature").mkdir(parents=True)
        (deps_dir / "other" / "main").mkdir(parents=True)

        monkeypatch.setattr(
            dep_list_cmd, "require_current_project_dir", lambda: project_dir
        )
        monkeypatch.setattr(
            dep_list_cmd, "project_deps_dir", lambda pd: project_dir / "deps"
        )
        monkeypatch.setattr(
            dep_list_cmd,
            "config_toml_file",
            lambda pd, branch=None: (
                project_dir / "config" / branch / "config.toml"
                if branch
                else project_dir / "config" / "config.toml"
            ),
        )
        monkeypatch.setattr(
            dep_list_cmd, "current_config_branch", lambda pd: "feature"
        )

        dep_list_cmd.list_deps()

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line]
        assert len(lines) == 2
        assert "mylib/feature" in lines[0]
        assert "other/main" in lines[1]

    def test_lists_deps_from_main_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_dir = project_dir / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text(
            "[deps]" + "\n" + 'mylib = "main"' + "\n",
            encoding="utf-8",
        )
        deps_dir = project_dir / "deps"
        (deps_dir / "mylib" / "main").mkdir(parents=True)

        monkeypatch.setattr(
            dep_list_cmd, "require_current_project_dir", lambda: project_dir
        )
        monkeypatch.setattr(
            dep_list_cmd, "project_deps_dir", lambda pd: project_dir / "deps"
        )
        monkeypatch.setattr(
            dep_list_cmd,
            "config_toml_file",
            lambda pd, branch=None: (
                project_dir / "config" / branch / "config.toml"
                if branch
                else project_dir / "config" / "config.toml"
            ),
        )
        monkeypatch.setattr(
            dep_list_cmd, "current_config_branch", lambda pd: None
        )

        dep_list_cmd.list_deps()

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line]
        assert len(lines) == 1
        assert "mylib/main" in lines[0]

    def test_verbose_shows_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_dir = project_dir / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text(
            "[deps]" + "\n" + 'mylib = "main"' + "\n",
            encoding="utf-8",
        )
        deps_dir = project_dir / "deps"
        (deps_dir / "mylib" / "main").mkdir(parents=True)

        monkeypatch.setattr(
            dep_list_cmd, "require_current_project_dir", lambda: project_dir
        )
        monkeypatch.setattr(
            dep_list_cmd, "project_deps_dir", lambda pd: project_dir / "deps"
        )
        monkeypatch.setattr(
            dep_list_cmd,
            "config_toml_file",
            lambda pd, branch=None: (
                project_dir / "config" / branch / "config.toml"
                if branch
                else project_dir / "config" / "config.toml"
            ),
        )
        monkeypatch.setattr(
            dep_list_cmd, "current_config_branch", lambda pd: None
        )

        dep_list_cmd.list_deps(verbose=True)

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line]
        assert len(lines) == 1
        assert "mylib/main" in lines[0]
        assert str(deps_dir / "mylib" / "main") in lines[0]

    def test_no_deps_produces_no_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_dir = project_dir / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("", encoding="utf-8")

        monkeypatch.setattr(
            dep_list_cmd, "require_current_project_dir", lambda: project_dir
        )
        monkeypatch.setattr(
            dep_list_cmd, "project_deps_dir", lambda pd: project_dir / "deps"
        )
        monkeypatch.setattr(
            dep_list_cmd,
            "config_toml_file",
            lambda pd, branch=None: (
                project_dir / "config" / branch / "config.toml"
                if branch
                else project_dir / "config" / "config.toml"
            ),
        )
        monkeypatch.setattr(
            dep_list_cmd, "current_config_branch", lambda pd: None
        )

        dep_list_cmd.list_deps()

        captured = capsys.readouterr()
        assert captured.out.strip() == ""


# ---------------------------------------------------------------------------
# list_deps — --all mode
# ---------------------------------------------------------------------------


class TestListDepsAll:
    def test_lists_all_checkouts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        deps_dir = project_dir / "deps"
        (deps_dir / "mylib" / "main").mkdir(parents=True)
        _init_git_repo(deps_dir / "mylib" / "main")
        (deps_dir / "mylib" / "feature").mkdir(parents=True)
        _init_git_repo(deps_dir / "mylib" / "feature")
        (deps_dir / "other" / "develop").mkdir(parents=True)
        _init_git_repo(deps_dir / "other" / "develop")

        monkeypatch.setattr(
            dep_list_cmd, "require_current_project_dir", lambda: project_dir
        )
        monkeypatch.setattr(
            dep_list_cmd, "project_deps_dir", lambda pd: project_dir / "deps"
        )

        dep_list_cmd.list_deps(all_checkouts=True)

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line]
        assert len(lines) == 3
        assert "mylib/feature" in lines[0]
        assert "mylib/main" in lines[1]
        assert "other/develop" in lines[2]

    def test_all_verbose_shows_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        deps_dir = project_dir / "deps"
        (deps_dir / "mylib" / "main").mkdir(parents=True)
        _init_git_repo(deps_dir / "mylib" / "main")

        monkeypatch.setattr(
            dep_list_cmd, "require_current_project_dir", lambda: project_dir
        )
        monkeypatch.setattr(
            dep_list_cmd, "project_deps_dir", lambda pd: project_dir / "deps"
        )

        dep_list_cmd.list_deps(verbose=True, all_checkouts=True)

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line]
        assert len(lines) == 1
        assert "mylib/main" in lines[0]
        assert str(deps_dir / "mylib" / "main") in lines[0]

    def test_all_no_deps_produces_no_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()

        monkeypatch.setattr(
            dep_list_cmd, "require_current_project_dir", lambda: project_dir
        )
        monkeypatch.setattr(
            dep_list_cmd, "project_deps_dir", lambda pd: project_dir / "deps"
        )

        dep_list_cmd.list_deps(all_checkouts=True)

        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_all_no_deps_dir_produces_no_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        monkeypatch.setattr(
            dep_list_cmd, "require_current_project_dir", lambda: project_dir
        )
        monkeypatch.setattr(
            dep_list_cmd, "project_deps_dir", lambda pd: project_dir / "deps"
        )

        dep_list_cmd.list_deps(all_checkouts=True)

        captured = capsys.readouterr()
        assert captured.out.strip() == ""


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class TestRun:
    def test_delegates_to_list_deps(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_dir = project_dir / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text(
            "[deps]" + "\n" + 'mylib = "main"' + "\n",
            encoding="utf-8",
        )
        deps_dir = project_dir / "deps"
        (deps_dir / "mylib" / "main").mkdir(parents=True)

        monkeypatch.setattr(
            dep_list_cmd, "require_current_project_dir", lambda: project_dir
        )
        monkeypatch.setattr(
            dep_list_cmd, "project_deps_dir", lambda pd: project_dir / "deps"
        )
        monkeypatch.setattr(
            dep_list_cmd,
            "config_toml_file",
            lambda pd, branch=None: (
                project_dir / "config" / branch / "config.toml"
                if branch
                else project_dir / "config" / "config.toml"
            ),
        )
        monkeypatch.setattr(
            dep_list_cmd, "current_config_branch", lambda pd: None
        )

        dep_list_cmd.run()

        captured = capsys.readouterr()
        assert "mylib/main" in captured.out


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestDepListViaCli:
    @pytest.mark.parametrize(
        "argv, verbose, all_checkouts",
        [
            pytest.param(["dep", "list"], False, False, id="default"),
            pytest.param(["dep", "list", "-v"], True, False, id="verbose"),
            pytest.param(["dep", "list", "--all"], False, True, id="all"),
            pytest.param(["dep", "list", "-v", "--all"], True, True, id="verbose_and_all"),
        ],
    )
    def test_dep_list_via_cli(
        self,
        monkeypatch: pytest.MonkeyPatch,
        argv: list[str],
        verbose: bool,
        all_checkouts: bool,
    ) -> None:
        runner = CliRunner()
        calls: list[dict[str, bool]] = []

        def record(*, verbose: bool = False, all_checkouts: bool = False) -> None:
            calls.append({"verbose": verbose, "all_checkouts": all_checkouts})

        monkeypatch.setattr(cli.dep_list_command, "run", record)
        result = _invoke(runner, argv)
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0]["verbose"] is verbose
        assert calls[0]["all_checkouts"] is all_checkouts
