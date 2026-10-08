"""Tests for CLI argument handling through the Typer application."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

import click
import pytest
from click.testing import CliRunner, Result
from typer import Exit as TyperExit
from typer._click import Context as TyperContext
from typer.core import TyperCommand, TyperGroup
from typer.main import get_command

import agm.cli as cli
import agm.commands.config.copy as config_copy_command
import agm.commands.config.env as config_env_command
import agm.commands.config.update as config_update_command
import agm.commands.dep.new as dep_new_command
import agm.commands.dep.remove as dep_remove_command
import agm.commands.dep.switch as dep_switch_command
import agm.commands.init as init_command
import agm.commands.run as run_command
import agm.commands.sync.fetch as sync_fetch_command
import agm.commands.sync.pull as sync_pull_command
import agm.commands.tmux.close as tmux_close_command
import agm.commands.tmux.layout as tmux_layout_command
import agm.commands.tmux.open as tmux_open_command
import agm.commands.workspace.close as workspace_close_command
import agm.commands.workspace.list as workspace_list_command
import agm.commands.workspace.open as workspace_open_command
import agm.commands.workspace.setup as workspace_setup_command
import agm.commands.workspace.shell_regen as workspace_shell_regen_command
import agm.commands.worktree.new as worktree_new_command
import agm.commands.worktree.remove as worktree_remove_command
import agm.parser as parser_helpers
from agm.core import dry_run as dry_run_state
from agm.packages.layout import MODULE_TREE_DIRNAME
from tests._help_helpers import assert_lists_execution_options


class RecordedArgs(Protocol):
    def __getattr__(self, name: str) -> object: ...


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def invoke(runner: CliRunner, argv: list[str]) -> Result:
    return runner.invoke(get_command(cli.app), argv, prog_name="agm")


def make_recorder(
    monkeypatch: pytest.MonkeyPatch,
    target: object,
    attribute: str = "run",
) -> list[RecordedArgs]:
    calls: list[RecordedArgs] = []

    def record(args: RecordedArgs) -> None:
        calls.append(args)

    monkeypatch.setattr(target, attribute, record)
    return calls


class TestConfigCopy:
    def test_commands_without_dry_run_behavior_reject_the_option(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.commands.check as check_command
        import agm.commands.config.env as config_env_command
        import agm.commands.dep.list as dep_list_command
        import agm.commands.pkg.check as pkg_check_command
        import agm.commands.pkg.info as pkg_info_command
        import agm.commands.pkg.list as pkg_list_command
        import agm.commands.workspace.list as workspace_list_command

        calls: list[bool] = []

        def record(*args: object, **kwargs: object) -> None:
            calls.append(True)

        for command in (
            check_command,
            config_env_command,
            dep_list_command,
            pkg_check_command,
            pkg_info_command,
            pkg_list_command,
            workspace_list_command,
        ):
            monkeypatch.setattr(command, "run", record)

        for argv in (
            ["check", "source.agl", "--dry-run"],
            ["config", "env", "--dry-run"],
            ["dep", "list", "--dry-run"],
            ["pkg", "check", "--dry-run"],
            ["pkg", "info", "alpha", "--dry-run"],
            ["pkg", "list", "--dry-run"],
            ["workspace", "list", "--dry-run"],
        ):
            result = invoke(runner, argv)
            assert result.exit_code != 0, argv

        assert calls == []

    def test_read_only_groups_and_root_reject_dry_run(self, runner: CliRunner) -> None:
        for argv in (
            ["help", "--dry-run"],
            ["--dry-run", "check", "source.agl"],
            ["config", "--dry-run", "env"],
            ["dep", "--dry-run", "list"],
            ["pkg", "--dry-run", "list"],
            ["sync", "--dry-run", "fetch"],
            ["tmux", "--dry-run", "open"],
            ["workspace", "--dry-run", "list"],
            ["worktree", "--dry-run", "new", "branch"],
        ):
            result = invoke(runner, argv)
            assert result.exit_code != 0, argv

    def test_config_cp(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, config_copy_command)
        result = invoke(runner, ["config", "cp", "mydir"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].dirname == "mydir"

    def test_config_copy_rejects_d_option(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, config_copy_command)
        result = invoke(runner, ["config", "copy", "-d", "/some/dir", "target"])
        assert result.exit_code != 0
        assert "No such option" in result.output
        assert len(calls) == 0

    def test_config_copy_rejects_dir_long_option(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, config_copy_command)
        result = invoke(runner, ["config", "copy", "--dir", "/some/dir", "target"])
        assert result.exit_code != 0
        assert "No such option" in result.output
        assert len(calls) == 0

    def test_config_cp_missing_dirname(self, runner: CliRunner) -> None:
        result = invoke(runner, ["config", "cp"])
        assert result.exit_code != 0
        assert "required" in result.output

    def test_config_env(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, config_env_command)
        result = invoke(runner, ["config", "env"])
        assert result.exit_code == 0
        assert len(calls) == 1

    def test_config_update(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, config_update_command)
        result = invoke(runner, ["config", "update"])
        assert result.exit_code == 0
        assert len(calls) == 1


class TestPackageLifecycle:
    def test_pkg_lifecycle_options_map_to_typed_arguments(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.commands.pkg.create as create_command
        import agm.commands.pkg.info as info_command
        import agm.commands.pkg.install as install_command
        import agm.commands.pkg.list as list_command
        import agm.commands.pkg.uninstall as uninstall_command

        create_calls = make_recorder(monkeypatch, create_command, "run")
        install_calls = make_recorder(monkeypatch, install_command, "run")
        uninstall_calls = make_recorder(monkeypatch, uninstall_command, "run")
        list_calls = make_recorder(monkeypatch, list_command, "run")
        info_calls = make_recorder(monkeypatch, info_command, "run")

        assert invoke(runner, ["pkg", "create", "package", "-o", "out.agmpkg"]).exit_code == 0
        assert invoke(runner, ["pkg", "install", "--editable", "--shadow", "source"]).exit_code == 0
        assert invoke(runner, ["pkg", "install", "--reinstall", "source"]).exit_code == 0
        assert invoke(runner, ["pkg", "uninstall", "alpha"]).exit_code == 0
        assert invoke(runner, ["pkg", "list"]).exit_code == 0
        assert invoke(runner, ["pkg", "info", "alpha"]).exit_code == 0

        assert create_calls[0].directory == "package"
        assert create_calls[0].output == "out.agmpkg"
        assert install_calls[0].source == "source"
        assert install_calls[0].editable is True
        assert install_calls[0].shadow is True
        assert install_calls[1].source == "source"
        assert install_calls[1].reinstall is True
        assert uninstall_calls[0].name == "alpha"
        assert len(list_calls) == 1
        assert info_calls[0].name == "alpha"


class TestPackageCheck:
    def test_pkg_check_passes_optional_directory_to_command(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.commands.pkg.check as check_command

        calls = make_recorder(monkeypatch, check_command, "run")

        result = invoke(runner, ["pkg", "check", "package-dir"])

        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].directory == "package-dir"

    def test_pkg_check_uses_current_directory_without_argument(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.commands.pkg.check as check_command

        calls = make_recorder(monkeypatch, check_command, "run")

        result = invoke(runner, ["pkg", "check"])

        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].directory is None

    def test_pkg_check_validates_current_and_explicit_directories(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        package = tmp_path / "demo"
        module_root = package / MODULE_TREE_DIRNAME
        module_root.mkdir(parents=True)
        (package / "package.toml").write_text('[package]\nname = "demo"\nversion = "1.0.0"\n')
        monkeypatch.chdir(package)

        implicit = invoke(runner, ["pkg", "check"])
        explicit = invoke(runner, ["pkg", "check", str(package)])
        missing = invoke(runner, ["pkg", "check", str(tmp_path / "missing")])

        assert implicit.exit_code == 0
        assert explicit.exit_code == 0
        assert missing.exit_code == 1
        assert missing.stderr

    @pytest.mark.parametrize("command_path", ("exec launch", "wsp launch"))
    def test_pkg_check_refuses_builtin_and_alias_command_prefixes(
        self, runner: CliRunner, tmp_path: Path, command_path: str
    ) -> None:
        package = tmp_path / "demo"
        module_root = package / MODULE_TREE_DIRNAME
        module_root.mkdir(parents=True)
        (module_root / "main.agl").write_text("program def main() -> unit = ()\n")
        (package / "package.toml").write_text(
            '[package]\nname = "demo"\nversion = "1.0.0"\n\n'
            f'[commands]\n"{command_path}" = {{ program = "demo/main::main" }}\n'
        )

        assert invoke(runner, ["pkg", "check", str(package)]).exit_code == 1

    def test_pkg_create_refuses_a_package_that_fails_check(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        package = tmp_path / "alpha"
        (package / MODULE_TREE_DIRNAME).mkdir(parents=True)
        (package / MODULE_TREE_DIRNAME / "main.agl").write_text(
            "program def main() -> unit = ()\n", encoding="utf-8"
        )
        (package / "package.toml").write_text(
            '[package]\nname = "alpha"\nversion = "1.0.0"\n'
            '\n[commands]\nlaunch = { program = "alpha/main::missing" }\n',
            encoding="utf-8",
        )
        archive = tmp_path / "alpha.agmpkg"

        checked = invoke(runner, ["pkg", "check", str(package)])
        created = invoke(runner, ["pkg", "create", str(package), "-o", str(archive)])

        assert checked.exit_code == 1
        assert created.exit_code == 1
        assert not archive.exists()

    def test_pkg_create_refuses_a_dependency_without_a_version(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        package = tmp_path / "alpha"
        (package / MODULE_TREE_DIRNAME).mkdir(parents=True)
        (package / MODULE_TREE_DIRNAME / "main.agl").write_text(
            "program def main() -> unit = ()\n", encoding="utf-8"
        )
        (package / "package.toml").write_text(
            '[package]\nname = "alpha"\nversion = "1.0.0"\n'
            '\n[dependencies]\nbravo = { path = "../bravo" }\n',
            encoding="utf-8",
        )
        archive = tmp_path / "alpha.agmpkg"

        result = invoke(runner, ["pkg", "create", str(package), "-o", str(archive)])

        assert result.exit_code == 1
        assert not archive.exists()

    def test_pkg_group_without_subcommand_shows_help(self, runner: CliRunner) -> None:
        result = invoke(runner, ["pkg"])

        assert result.exit_code == 0
        assert "check" in result.output


class TestWorktreeNew:
    def test_wt_new(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, worktree_new_command)
        result = invoke(runner, ["wt", "new", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/y"
        assert calls[0].worktrees_dir is None

    def test_wt_new_with_d(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, worktree_new_command)
        result = invoke(runner, ["wt", "new", "-d", "/custom", "feat/z"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].worktrees_dir == "/custom"
        assert calls[0].branch == "feat/z"

    def test_wt_new_with_dir_long(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, worktree_new_command)
        result = invoke(runner, ["wt", "new", "--dir", "/custom", "feat/z"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].worktrees_dir == "/custom"
        assert calls[0].branch == "feat/z"

    @pytest.mark.parametrize("command", [["worktree", "new"], ["wt", "new"]])
    @pytest.mark.parametrize(
        "flags, no_fetch", [([], True), (["--no-fetch"], True), (["--fetch"], False)]
    )
    def test_wt_new_fetch_policy(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        command: list[str],
        flags: list[str],
        no_fetch: bool,
    ) -> None:
        calls = make_recorder(monkeypatch, worktree_new_command)
        result = invoke(runner, [*command, *flags, "feat/z"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].no_fetch is no_fetch

    def test_wt_new_missing_branch(self, runner: CliRunner) -> None:
        result = invoke(runner, ["wt", "new"])
        assert result.exit_code != 0
        assert "required" in result.output

    def test_wt_co_is_rejected(self, runner: CliRunner) -> None:
        result = invoke(runner, ["wt", "co", "feat/x"])
        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_worktree_checkout_is_rejected(self, runner: CliRunner) -> None:
        result = invoke(runner, ["worktree", "checkout", "feat/x"])
        assert result.exit_code != 0
        assert "No such command" in result.output


class TestWorkspace:
    def test_workspace_setup(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[None] = []

        def record() -> None:
            calls.append(None)

        monkeypatch.setattr(workspace_setup_command, "run", record)
        result = invoke(runner, ["workspace", "setup"])
        assert result.exit_code == 0
        assert len(calls) == 1

    def test_wsp_setup(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[None] = []

        def record() -> None:
            calls.append(None)

        monkeypatch.setattr(workspace_setup_command, "run", record)
        result = invoke(runner, ["wsp", "setup"])
        assert result.exit_code == 0
        assert len(calls) == 1

    def test_top_level_setup_is_rejected(self, runner: CliRunner) -> None:
        result = invoke(runner, ["setup"])
        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_workspace_list(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[bool] = []

        def record(*, verbose: bool = False) -> None:
            calls.append(verbose)

        monkeypatch.setattr(workspace_list_command, "run", record)
        result = invoke(runner, ["workspace", "list"])

        assert result.exit_code == 0
        assert calls == [False]

    def test_workspace_shell_regen(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def record(*, shell_dir: str) -> None:
            calls.append(shell_dir)

        monkeypatch.setattr(workspace_shell_regen_command, "run", record)
        result = invoke(runner, ["workspace", "shell-regen", "/tmp/agm-shell-dir"])

        assert result.exit_code == 0
        assert calls == ["/tmp/agm-shell-dir"]

    def test_wsp_shell_regen(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        def record(*, shell_dir: str) -> None:
            calls.append(shell_dir)

        monkeypatch.setattr(workspace_shell_regen_command, "run", record)
        result = invoke(runner, ["wsp", "shell-regen", "/tmp/agm-shell-dir"])

        assert result.exit_code == 0
        assert calls == ["/tmp/agm-shell-dir"]

    def test_wsp_list_verbose(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[bool] = []

        def record(*, verbose: bool = False) -> None:
            calls.append(verbose)

        monkeypatch.setattr(workspace_list_command, "run", record)
        result = invoke(runner, ["wsp", "list", "--verbose"])

        assert result.exit_code == 0
        assert calls == [True]

    def test_top_level_list_is_rejected(self, runner: CliRunner) -> None:
        result = invoke(runner, ["list"])
        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_workspace_open(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, workspace_open_command)
        result = invoke(runner, ["workspace", "open", "-n", "6", "repo"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "repo"
        assert calls[0].pane_count == "6"

    def test_wsp_close(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, workspace_close_command)
        result = invoke(runner, ["wsp", "close", "-D", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/x"
        assert calls[0].force_delete is True

    def test_wsp_close_keep_workspace(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, workspace_close_command)
        result = invoke(runner, ["wsp", "close", "--keep-workspace", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].keep_workspace is True


class TestWorktreeRemove:
    def test_wt_rm(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, worktree_remove_command)
        result = invoke(runner, ["wt", "rm", "old-branch"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "old-branch"
        assert calls[0].force is False

    def test_worktree_remove_force(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, worktree_remove_command)
        result = invoke(runner, ["worktree", "remove", "-f", "old-branch"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].force is True
        assert calls[0].branch == "old-branch"

    def test_worktree_remove_force_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, worktree_remove_command)
        result = invoke(runner, ["worktree", "remove", "--force", "old-branch"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].force is True
        assert calls[0].branch == "old-branch"

    def test_wt_rm_missing_branch(self, runner: CliRunner) -> None:
        result = invoke(runner, ["wt", "rm"])
        assert result.exit_code != 0
        assert "required" in result.output


class TestDep:
    def test_dep_new(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, dep_new_command)
        result = invoke(runner, ["dep", "new", "https://github.com/org/repo.git"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].repo_url == "https://github.com/org/repo.git"
        assert calls[0].branch is None

    def test_dep_new_with_branch(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, dep_new_command)
        result = invoke(runner, ["dep", "new", "-b", "main", "https://github.com/org/repo.git"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "main"

    def test_dep_new_with_branch_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, dep_new_command)
        result = invoke(
            runner,
            ["dep", "new", "--branch", "main", "https://github.com/org/repo.git"],
        )
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "main"

    def test_dep_switch(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, dep_switch_command)
        result = invoke(runner, ["dep", "switch", "mylib", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].dep == "mylib"
        assert calls[0].branch == "feat/x"
        assert calls[0].create_branch is False

    def test_dep_switch_create(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, dep_switch_command)
        result = invoke(runner, ["dep", "switch", "-b", "mylib", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].create_branch is True

    def test_dep_switch_create_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, dep_switch_command)
        result = invoke(runner, ["dep", "switch", "--branch", "mylib", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].create_branch is True

    @pytest.mark.parametrize(
        "flags, no_fetch", [([], True), (["--no-fetch"], True), (["--fetch"], False)]
    )
    def test_dep_switch_fetch_policy(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        flags: list[str],
        no_fetch: bool,
    ) -> None:
        calls = make_recorder(monkeypatch, dep_switch_command)
        result = invoke(runner, ["dep", "switch", *flags, "mylib", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].no_fetch is no_fetch

    def test_dep_rm(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, dep_remove_command)
        result = invoke(runner, ["dep", "rm", "mylib/feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].target == "mylib/feat/x"
        assert calls[0].all is False

    def test_dep_rm_all(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, dep_remove_command)
        result = invoke(runner, ["dep", "rm", "--all", "mylib"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].target == "mylib"
        assert calls[0].all is True

    def test_dep_remove(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, dep_remove_command)
        result = invoke(runner, ["dep", "remove", "mylib/feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].target == "mylib/feat/x"
        assert calls[0].all is False

    def test_dep_rm_missing_target(self, runner: CliRunner) -> None:
        result = invoke(runner, ["dep", "rm"])
        assert result.exit_code != 0
        assert "required" in result.output

    def test_dep_missing_subcommand_shows_help(self, runner: CliRunner) -> None:
        result = invoke(runner, ["dep"])
        assert result.exit_code == 0
        assert "agm dep" in result.stdout


class TestSync:
    def test_sync_fetch(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, sync_fetch_command)
        result = invoke(runner, ["sync", "fetch"])
        assert result.exit_code == 0
        assert len(calls) == 1

    def test_sync_fetch_accepts_dry_run(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        observed: list[bool] = []

        def record(args: object) -> None:
            del args
            observed.append(dry_run_state.enabled())

        monkeypatch.setattr(sync_fetch_command, "run", record)
        result = invoke(runner, ["sync", "fetch", "--dry-run"])
        assert result.exit_code == 0
        assert observed == [True]

    def test_sync_fetch_rejects_args(self, runner: CliRunner) -> None:
        result = invoke(runner, ["sync", "fetch", "extra"])
        assert result.exit_code != 0
        assert "unexpected extra argument" in result.output

    def test_top_level_fetch_is_rejected(self, runner: CliRunner) -> None:
        result = invoke(runner, ["fetch"])
        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_sync_pull(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, sync_pull_command)
        result = invoke(runner, ["sync", "pull"])
        assert result.exit_code == 0
        assert len(calls) == 1

    def test_sync_pull_accepts_dry_run(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        observed: list[bool] = []

        def record(args: object) -> None:
            del args
            observed.append(dry_run_state.enabled())

        monkeypatch.setattr(sync_pull_command, "run", record)
        result = invoke(runner, ["sync", "pull", "--dry-run"])
        assert result.exit_code == 0
        assert observed == [True]

    def test_sync_pull_rejects_args(self, runner: CliRunner) -> None:
        result = invoke(runner, ["sync", "pull", "extra"])
        assert result.exit_code != 0
        assert "unexpected extra argument" in result.output

    def test_top_level_pull_is_rejected(self, runner: CliRunner) -> None:
        result = invoke(runner, ["pull"])
        assert result.exit_code != 0
        assert "No such command" in result.output


class TestDryRun:
    def test_open_accepts_command_level_dry_run(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        observed: list[bool] = []

        def record(args: RecordedArgs) -> None:
            observed.append(dry_run_state.enabled())
            assert args.branch == "repo"

        monkeypatch.setattr(workspace_open_command, "run", record)
        result = invoke(runner, ["open", "--dry-run", "repo"])
        assert result.exit_code == 0
        assert observed == [True]


class TestInit:
    def test_init_project_and_url(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, init_command)
        result = invoke(runner, ["init", "myproj", "https://github.com/org/repo.git"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].positional == ["myproj", "https://github.com/org/repo.git"]
        assert calls[0].branch is None
        assert calls[0].embedded is False
        assert calls[0].split is False

    def test_init_url_only(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, init_command)
        result = invoke(runner, ["init", "https://github.com/org/repo.git"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].positional == ["https://github.com/org/repo.git"]
        assert calls[0].clone is False

    def test_init_with_clone(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, init_command)
        result = invoke(runner, ["init", "--clone", "https://github.com/org/repo.git"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].positional == ["https://github.com/org/repo.git"]
        assert calls[0].clone is True

    def test_init_with_branch(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, init_command)
        result = invoke(runner, ["init", "-b", "dev", "myproj"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "dev"
        assert calls[0].positional == ["myproj"]

    def test_init_with_branch_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, init_command)
        result = invoke(runner, ["init", "--branch", "dev", "myproj"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "dev"
        assert calls[0].positional == ["myproj"]

    def test_init_with_embedded(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, init_command)
        result = invoke(runner, ["init", "--embedded", "myproj"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].embedded is True
        assert calls[0].positional == ["myproj"]

    def test_init_with_split(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, init_command)
        result = invoke(runner, ["init", "--split", "myproj"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].split is True
        assert calls[0].positional == ["myproj"]

    def test_init_with_no_repo_git(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, init_command)
        result = invoke(runner, ["init", "--no-repo-git", "myproj"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].no_repo_git is True
        assert calls[0].positional == ["myproj"]

    def test_init_with_workspace_is_rejected(self, runner: CliRunner) -> None:
        result = invoke(runner, ["init", "--workspace", "myproj"])
        assert result.exit_code != 0
        assert "No such option" in result.output

    def test_init_without_args(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, init_command)
        result = invoke(runner, ["init"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].positional == []


class TestOpen:
    def test_open_missing_target(self, runner: CliRunner) -> None:
        result = invoke(runner, ["open"])
        assert result.exit_code != 0
        assert "required" in result.output

    def test_open_repo(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, workspace_open_command)
        result = invoke(runner, ["open", "repo"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].detached is False
        assert calls[0].pane_count is None
        assert calls[0].parent is None
        assert calls[0].branch == "repo"

    def test_open_with_branch(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, workspace_open_command)
        result = invoke(runner, ["open", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/x"

    @pytest.mark.parametrize("command", [["open"], ["workspace", "open"], ["wsp", "open"]])
    @pytest.mark.parametrize(
        "flags, no_fetch", [([], True), (["--no-fetch"], True), (["--fetch"], False)]
    )
    def test_open_fetch_policy(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        command: list[str],
        flags: list[str],
        no_fetch: bool,
    ) -> None:
        calls = make_recorder(monkeypatch, workspace_open_command)
        result = invoke(runner, [*command, *flags, "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].no_fetch is no_fetch

    def test_open_with_pane_count(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, workspace_open_command)
        result = invoke(runner, ["open", "-n", "6", "repo"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "6"

    def test_open_with_num_panes_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, workspace_open_command)
        result = invoke(runner, ["open", "--num-panes", "6", "repo"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "6"

    def test_open_with_parent_and_branch(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, workspace_open_command)
        result = invoke(runner, ["open", "-p", "main", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/y"
        assert calls[0].parent == "main"

    def test_open_with_parent_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, workspace_open_command)
        result = invoke(runner, ["open", "--parent", "main", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/y"
        assert calls[0].parent == "main"

    def test_open_with_all(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, workspace_open_command)
        result = invoke(runner, ["open", "-n", "2", "-p", "main", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "2"
        assert calls[0].parent == "main"
        assert calls[0].branch == "feat/y"

    def test_open_detached(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, workspace_open_command)
        result = invoke(runner, ["open", "-d", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].detached is True
        assert calls[0].branch == "feat/y"

    def test_open_detach_long(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, workspace_open_command)
        result = invoke(runner, ["open", "--detach", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].detached is True
        assert calls[0].branch == "feat/y"


class TestClose:
    def test_close_branch(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, workspace_close_command)
        result = invoke(runner, ["close", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/x"

    def test_close_missing_branch(self, runner: CliRunner) -> None:
        result = invoke(runner, ["close"])
        assert result.exit_code != 0
        assert "required" in result.output

    def test_close_D_flag(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, workspace_close_command)
        result = invoke(runner, ["close", "-D", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/x"
        assert calls[0].force_delete is True

    def test_close_default_no_D(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, workspace_close_command)
        result = invoke(runner, ["close", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].force_delete is False

    def test_close_keep_branch_flag(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, workspace_close_command)
        result = invoke(runner, ["close", "--keep-branch", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].keep_branch is True
        assert calls[0].keep_workspace is False

    def test_close_keep_workspace_implies_keep_branch_in_args(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, workspace_close_command)
        result = invoke(runner, ["close", "--keep-workspace", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].keep_branch is True
        assert calls[0].keep_workspace is True


class TestRun:
    def test_run_simple(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, run_command)
        result = invoke(runner, ["run", "npm", "test"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].run_command == ["npm", "test"]
        assert calls[0].no_patch is False
        assert calls[0].pty is None
        assert calls[0].memory is None
        assert calls[0].settings_file is None

    def test_run_with_f(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, run_command)
        result = invoke(runner, ["run", "-f", "ci.json", "make"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].settings_file == "ci.json"
        assert calls[0].run_command == ["make"]

    def test_run_with_file_long(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, run_command)
        result = invoke(runner, ["run", "--file", "ci.json", "make"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].settings_file == "ci.json"
        assert calls[0].run_command == ["make"]

    def test_run_no_patch(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, run_command)
        result = invoke(runner, ["run", "--no-patch", "echo", "hi"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].no_patch is True
        assert calls[0].run_command == ["echo", "hi"]

    def test_run_no_sandbox(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, run_command)
        result = invoke(runner, ["run", "--no-sandbox", "echo", "hi"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].no_sandbox is True
        assert calls[0].run_command == ["echo", "hi"]

    @pytest.mark.parametrize(("option", "enabled"), [("--pty", True), ("--no-pty", False)])
    def test_run_pty_option(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        option: str,
        enabled: bool,
    ) -> None:
        calls = make_recorder(monkeypatch, run_command)
        result = invoke(runner, ["run", option, "echo", "hi"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pty is enabled
        assert calls[0].run_command == ["echo", "hi"]

    def test_run_with_memory(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, run_command)
        result = invoke(runner, ["run", "--memory", "8G", "echo", "hi"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].memory == "8G"
        assert calls[0].swap is None
        assert calls[0].no_memory_limit is False
        assert calls[0].no_swap_limit is False
        assert calls[0].run_command == ["echo", "hi"]

    def test_run_with_unlimited_memory(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, run_command)
        result = invoke(runner, ["run", "--memory", "unlimited", "echo", "hi"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].memory == "unlimited"
        assert calls[0].swap is None
        assert calls[0].no_memory_limit is False
        assert calls[0].no_swap_limit is False
        assert calls[0].run_command == ["echo", "hi"]

    def test_run_with_no_memory_limit(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, run_command)
        result = invoke(runner, ["run", "--no-memory-limit", "echo", "hi"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].memory is None
        assert calls[0].swap is None
        assert calls[0].no_memory_limit is True
        assert calls[0].no_swap_limit is False
        assert calls[0].run_command == ["echo", "hi"]

    def test_run_with_swap(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, run_command)
        result = invoke(runner, ["run", "--swap", "4G", "echo", "hi"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].memory is None
        assert calls[0].swap == "4G"
        assert calls[0].no_memory_limit is False
        assert calls[0].no_swap_limit is False
        assert calls[0].run_command == ["echo", "hi"]

    def test_run_with_unlimited_swap(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, run_command)
        result = invoke(runner, ["run", "--swap", "unlimited", "echo", "hi"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].memory is None
        assert calls[0].swap == "unlimited"
        assert calls[0].no_memory_limit is False
        assert calls[0].no_swap_limit is False
        assert calls[0].run_command == ["echo", "hi"]

    def test_run_with_no_swap_limit(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, run_command)
        result = invoke(runner, ["run", "--no-swap-limit", "echo", "hi"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].memory is None
        assert calls[0].swap is None
        assert calls[0].no_memory_limit is False
        assert calls[0].no_swap_limit is True
        assert calls[0].run_command == ["echo", "hi"]

    def test_run_no_command_shows_help(self, runner: CliRunner) -> None:
        result = invoke(runner, ["run"])
        assert result.exit_code == 0
        assert "agm run" in result.stdout


class TestTmuxOpen:
    def test_tmux_open_bare(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, tmux_open_command)
        result = invoke(runner, ["tmux", "open"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].detach is False
        assert calls[0].pane_count is None
        assert calls[0].session_name is None

    def test_tmux_open_with_all(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, tmux_open_command)
        result = invoke(runner, ["tmux", "open", "-d", "-n", "8", "mysession"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].detach is True
        assert calls[0].pane_count == "8"
        assert calls[0].session_name == "mysession"

    def test_tmux_open_detach_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, tmux_open_command)
        result = invoke(runner, ["tmux", "open", "--detach"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].detach is True

    def test_tmux_open_num_panes_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, tmux_open_command)
        result = invoke(runner, ["tmux", "open", "--num-panes", "8", "mysession"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "8"
        assert calls[0].session_name == "mysession"


class TestTmuxClose:
    def test_tmux_close(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, tmux_close_command)
        result = invoke(runner, ["tmux", "close", "mysession"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].session_name == "mysession"

    def test_tmux_close_missing_name(self, runner: CliRunner) -> None:
        result = invoke(runner, ["tmux", "close"])
        assert result.exit_code != 0
        assert "required" in result.output


class TestTmuxLayout:
    def test_tmux_layout(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, tmux_layout_command)
        result = invoke(runner, ["tmux", "layout", "4"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "4"
        assert calls[0].window_id is None

    def test_tmux_layout_with_explicit_window(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, tmux_layout_command)
        result = invoke(runner, ["tmux", "layout", "4", "--window", "@1"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "4"
        assert calls[0].window_id == "@1"

    def test_tmux_layout_with_window_short(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, tmux_layout_command)
        result = invoke(runner, ["tmux", "layout", "4", "-w", "@1"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "4"
        assert calls[0].window_id == "@1"

    def test_tmux_layout_missing_args(self, runner: CliRunner) -> None:
        result = invoke(runner, ["tmux", "layout"])
        assert result.exit_code != 0
        assert "required" in result.output


class TestHelp:
    def test_help_bare(self, runner: CliRunner) -> None:
        result = invoke(runner, ["help"])
        assert result.exit_code == 0
        assert "agm - Agent Management Framework" in result.stdout

    def test_help_with_command(self, runner: CliRunner) -> None:
        result = invoke(runner, ["help", "open"])
        assert result.exit_code == 0
        assert "agm open" in result.stdout

    def test_help_with_subcommand_path(self, runner: CliRunner) -> None:
        result = invoke(runner, ["help", "wt", "new"])
        assert result.exit_code == 0
        assert "agm wt new" in result.stdout

    def test_help_with_unknown(self, runner: CliRunner) -> None:
        result = invoke(runner, ["help", "bogus"])
        assert result.exit_code != 0
        assert "unknown command" in result.output

    def test_help_with_unknown_multi_segment_path(self, runner: CliRunner) -> None:
        result = invoke(runner, ["help", "nonexistent", "sub"])
        assert result.exit_code == 1
        assert result.stdout == ""
        assert result.stderr != ""

    def test_run_help_mentions_agm_home_relocation(self, runner: CliRunner) -> None:
        result = invoke(runner, ["help", "run"])
        assert result.exit_code == 0
        assert "AGM_HOME" in result.stdout


class TestTopLevel:
    def test_no_command(self, runner: CliRunner) -> None:
        result = invoke(runner, [])
        assert result.exit_code == 0
        assert "agm - Agent Management Framework" in result.stdout

    def test_unknown_command(self, runner: CliRunner) -> None:
        result = invoke(runner, ["bogus"])
        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_show_completion_is_available(self, runner: CliRunner) -> None:
        import unittest.mock

        import shellingham

        # shellingham.detect_shell() fails when no known shell is in the
        # process tree (e.g. running under pi or in certain CI environments),
        # making --show-completion flaky. Mock it so the test is deterministic.
        with unittest.mock.patch.object(
            shellingham, "detect_shell", return_value=("bash", "/bin/bash")
        ):
            result = invoke(runner, ["--show-completion"])
            assert result.exit_code == 0


class TestEveryCommandHasHelpText:
    def test_every_canonical_command_has_help_text(self) -> None:
        from agm.cli import _HELP_TEXTS

        canonical_commands = {
            "open",
            "close",
            "init",
            "workspace",
            "sync",
            "config",
            "worktree",
            "dep",
            "run",
            "tmux",
            "help",
        }
        for cmd in canonical_commands:
            assert cmd in _HELP_TEXTS, f"missing help text for '{cmd}'"

    def test_every_overview_command_has_help_text(self) -> None:
        from agm.cli import _COMMAND_OVERVIEW

        for name, _ in _COMMAND_OVERVIEW:
            text = parser_helpers.help_text_for(name)
            assert text is not None, f"overview lists '{name}' without help text"
            assert text.startswith(f"agm {name}")

    def test_every_command_path_has_help_listing_dry_run_exactly_when_accepted(self) -> None:
        def walk(command: TyperCommand | TyperGroup, path: tuple[str, ...]) -> None:
            text = parser_helpers._help_text_for_path(path)
            accepts = any("--dry-run" in param.opts for param in command.params)
            assert ("--dry-run" in text) == accepts, " ".join(path)
            if isinstance(command, TyperGroup):
                for name in command.list_commands(TyperContext(command)):
                    subcommand = command.get_command(TyperContext(command), name)
                    assert subcommand is not None
                    walk(subcommand, (*path, name))

        root = get_command(cli.app)
        assert isinstance(root, TyperGroup)
        for name in root.list_commands(TyperContext(root)):
            command = root.get_command(TyperContext(root), name)
            assert command is not None
            walk(command, (name,))

    def test_aliases_point_to_valid_commands(self) -> None:
        from agm.cli import _HELP_ALIASES, _HELP_TEXTS

        for alias, target in _HELP_ALIASES.items():
            assert target in _HELP_TEXTS, (
                f"alias '{alias}' -> '{target}' but '{target}' not in _HELP_TEXTS"
            )

    def test_no_empty_help_texts(self) -> None:
        from agm.cli import _HELP_TEXTS

        for cmd, text in _HELP_TEXTS.items():
            assert text.strip(), f"help text for '{cmd}' is empty"

    def test_help_texts_contain_command_name(self) -> None:
        from agm.cli import _HELP_TEXTS

        for cmd, text in _HELP_TEXTS.items():
            assert f"agm {cmd}" in text, f"help text for '{cmd}' doesn't mention 'agm {cmd}'"

    def test_command_reference_mentions_all_cli_commands(self) -> None:
        import re
        from pathlib import Path

        from typer.main import get_command

        from agm.cli import app

        cli_commands = set(get_command(app).list_commands(None))
        docs_dir = Path(__file__).resolve().parents[1] / "docs" / "commands"
        doc_text = "\n".join(
            path.read_text(encoding="utf-8") for path in sorted(docs_dir.glob("*.md"))
        )
        documented = set(re.findall(r"`agm (\w+)", doc_text))
        missing = cli_commands - documented
        assert not missing, f"docs/commands/ is missing entries for CLI commands: {sorted(missing)}"


class TestCommandPathFromContext:
    def test_returns_empty_for_root_context(self) -> None:
        ctx = click.Context(click.Command("agm"))
        ctx.parent = None
        result = cli._command_path_from_context(ctx)
        assert result == []

    def test_returns_command_path(self) -> None:
        root = click.Context(click.Command("agm"))
        root.parent = None
        sub = click.Context(click.Command("worktree"), parent=root, info_name="worktree")
        leaf = click.Context(click.Command("new"), parent=sub, info_name="new")
        result = cli._command_path_from_context(leaf)
        assert result == ["worktree", "new"]

    def test_skips_context_with_none_info_name(self) -> None:
        root = click.Context(click.Command("agm"))
        root.parent = None
        sub = click.Context(click.Command("worktree"), parent=root, info_name=None)
        leaf = click.Context(click.Command("new"), parent=sub, info_name="new")
        result = cli._command_path_from_context(leaf)
        assert result == ["new"]


class TestPrintContextHelp:
    def test_does_nothing_when_value_false(self) -> None:
        ctx = click.Context(click.Command("agm"))
        # Should not raise
        cli._print_context_help(ctx, None, False)

    def test_does_nothing_in_resilient_parsing(self) -> None:
        ctx = click.Context(click.Command("agm"))
        ctx.resilient_parsing = True
        cli._print_context_help(ctx, None, True)  # Should not raise or exit

    def test_prints_overview_for_root_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        root = click.Context(click.Command("agm"))
        root.parent = None
        with pytest.raises((SystemExit, click.exceptions.Exit, TyperExit)):
            cli._print_context_help(root, None, True)
        captured = capsys.readouterr()
        assert "agm - Agent Management Framework" in captured.out

    def test_prints_command_help_for_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        root = click.Context(click.Command("agm"))
        root.parent = None
        sub = click.Context(click.Command("open"), parent=root, info_name="open")
        with pytest.raises((SystemExit, click.exceptions.Exit, TyperExit)):
            cli._print_context_help(sub, None, True)
        captured = capsys.readouterr()
        assert "agm open" in captured.out


class TestCliCallbacks:
    """Test command group callbacks when invoked without subcommand."""

    def test_config_callback_shows_help(self) -> None:
        runner = CliRunner()
        result = invoke(runner, ["config"])
        assert result.exit_code == 0
        assert "agm config" in result.stdout

    def test_worktree_callback_shows_help(self) -> None:
        runner = CliRunner()
        result = invoke(runner, ["worktree"])
        assert result.exit_code == 0
        assert "agm worktree" in result.stdout

    def test_workspace_callback_shows_help(self) -> None:
        runner = CliRunner()
        result = invoke(runner, ["workspace"])
        assert result.exit_code == 0
        assert "agm workspace" in result.stdout

    def test_wsp_callback_shows_help(self) -> None:
        runner = CliRunner()
        result = invoke(runner, ["wsp"])
        assert result.exit_code == 0
        assert "agm wsp" in result.stdout or "agm workspace" in result.stdout

    def test_sync_callback_shows_help(self) -> None:
        runner = CliRunner()
        result = invoke(runner, ["sync"])
        assert result.exit_code == 0
        assert "agm sync" in result.stdout

    def test_dep_callback_shows_help(self) -> None:
        runner = CliRunner()
        result = invoke(runner, ["dep"])
        assert result.exit_code == 0
        assert "agm dep" in result.stdout

    def test_tmux_callback_shows_help(self) -> None:
        runner = CliRunner()
        result = invoke(runner, ["tmux"])
        assert result.exit_code == 0
        assert "agm tmux" in result.stdout

    def test_wt_callback_shows_help(self) -> None:
        runner = CliRunner()
        result = invoke(runner, ["wt"])
        assert result.exit_code == 0
        assert "agm wt" in result.stdout or "agm worktree" in result.stdout


class TestDepSwitchMissingArgs:
    def test_dep_switch_missing_both_args(self) -> None:
        runner = CliRunner()
        result = invoke(runner, ["dep", "switch"])
        assert result.exit_code != 0
        assert "required" in result.output

    def test_dep_switch_missing_branch(self) -> None:
        runner = CliRunner()
        result = invoke(runner, ["dep", "switch", "mylib"])
        assert result.exit_code != 0
        assert "required" in result.output


class TestInitEmbeddedAndWorkspaceMutualExclusion:
    def test_init_embedded_and_split_are_mutually_exclusive(self) -> None:
        runner = CliRunner()
        result = invoke(runner, ["init", "--embedded", "--split"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output


class TestRunWithUnrecognizedFlags:
    def test_run_with_flag_before_command_exits(self) -> None:
        runner = CliRunner()
        result = invoke(runner, ["run", "--unknown-flag"])
        assert result.exit_code != 0
        assert "unrecognized arguments" in result.output


class TestParserHelpers:
    def test_print_command_help_unknown_command_writes_to_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            parser_helpers.print_command_help("totally-unknown")
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "unknown command" in captured.err

    def test_print_help_for_command_path_with_file_param(self) -> None:
        output = io.StringIO()
        parser_helpers.print_help_for_command_path(["open"], file=output)
        result = output.getvalue()
        assert "agm open" in result

    def test_exec_help_lists_current_engine_options(self) -> None:
        output = io.StringIO()
        parser_helpers.print_help_for_command_path(["exec"], file=output)
        result = output.getvalue()
        for option in (
            "--default-agent",
            "--timeout",
            "--no-timeout",
        ):
            assert option in result
        assert "--runner" not in result

    @pytest.mark.parametrize("command", ["exec", "repl"])
    def test_execution_options_follow_the_command_options(self, command: str) -> None:
        output = io.StringIO()
        parser_helpers.print_help_for_command_path([command], file=output)
        result = output.getvalue()

        assert_lists_execution_options(result)
        assert result.index("Options:") < result.index("Execution options:")
        assert result.index("Execution options:") < result.index("Exit codes:")

    def test_exec_help_lists_installed_program_reference(self) -> None:
        output = io.StringIO()
        parser_helpers.print_help_for_command_path(["exec"], file=output)

        assert "PACKAGE/MODULE::PROGRAM" in output.getvalue()

    def test_repl_help_lists_engine_options(self) -> None:
        output = io.StringIO()
        parser_helpers.print_help_for_command_path(["repl"], file=output)
        result = output.getvalue()
        assert "--strict-json" in result
        assert "--default-agent" in result
        assert "--runner" not in result

    def test_print_command_help_with_file_param(self) -> None:
        output = io.StringIO()
        parser_helpers.print_command_help("open", file=output)
        result = output.getvalue()
        assert "agm open" in result


class TestConfigCopyCommand:
    """The 'config copy' CLI command function dispatches correctly."""

    def test_config_copy_via_cli(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[object] = []

        def record(args: object) -> None:
            calls.append(args)

        monkeypatch.setattr(config_copy_command, "run", record)
        result = invoke(runner, ["config", "copy", "mydir"])
        assert result.exit_code == 0
        assert len(calls) == 1


class TestMainEntryPoint:
    """main() entry point delegates to app()."""

    def test_main_calls_app_and_shows_help(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() calls app() which shows the overview help."""
        # Monkeypatch sys.argv so app() sees ["agm", "--help"] and exits
        monkeypatch.setattr("sys.argv", ["agm", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        # --help exits with code 0
        assert exc_info.value.code == 0

    def test_main_module_entry_point(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The `if __name__ == '__main__': main()` block delegates to main()."""
        import runpy
        import warnings

        monkeypatch.setattr("sys.argv", ["agm", "--help"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_module("agm.cli", run_name="__main__")
        assert exc_info.value.code == 0


class TestExecModulePathOption:
    """Parser-contract tests for the -I/--module-path repeatable option."""

    def test_single_module_path_passed_to_exec_args(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.commands.exec as exec_mod

        calls: list[object] = []
        monkeypatch.setattr(exec_mod, "run", lambda a: calls.append(a))

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        extra_root = tmp_path / "myroot"
        extra_root.mkdir()

        result = invoke(runner, ["exec", str(agl_file), "-I", str(extra_root)])
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert getattr(calls[0], "module_paths") == [str(extra_root)]

    def test_multiple_module_paths_accumulate(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.commands.exec as exec_mod

        calls: list[object] = []
        monkeypatch.setattr(exec_mod, "run", lambda a: calls.append(a))

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        root_a = tmp_path / "rootA"
        root_a.mkdir()
        root_b = tmp_path / "rootB"
        root_b.mkdir()

        result = invoke(
            runner,
            ["exec", str(agl_file), "-I", str(root_a), "--module-path", str(root_b)],
        )
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert getattr(calls[0], "module_paths") == [str(root_a), str(root_b)]

    def test_no_module_paths_defaults_to_empty_list(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.commands.exec as exec_mod

        calls: list[object] = []
        monkeypatch.setattr(exec_mod, "run", lambda a: calls.append(a))

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(runner, ["exec", str(agl_file)])
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert getattr(calls[0], "module_paths") == []


class TestExecEngineFlagExclusivity:
    """Parser-contract tests for exec's mutually exclusive engine flags."""

    def test_exec_rejects_trace_file_with_no_trace(self, runner: CliRunner, tmp_path: Path) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        result = invoke(runner, ["exec", str(agl_file), "--trace-file", "out.log", "--no-trace"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_exec_rejects_timeout_with_no_timeout(self, runner: CliRunner, tmp_path: Path) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        result = invoke(runner, ["exec", str(agl_file), "--timeout", "30s", "--no-timeout"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_exec_rejects_removed_no_trace_file_flag(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        assert invoke(runner, ["exec", str(agl_file), "--no-trace-file"]).exit_code != 0

    def test_every_negatable_engine_key_excludes_its_own_negative(self) -> None:
        """The rule is derived from the projection, so a new key brings its own."""
        from agm.cli_support.program_options import engine_key_option_flags
        from agm.cli_support.run_options import _exclusive_flag_groups
        from agm.config.engine_keys import ENGINE_KEYS

        groups = _exclusive_flag_groups()
        for spec in ENGINE_KEYS:
            flag, negative_flag = engine_key_option_flags(spec)
            if negative_flag is None:
                continue
            both = {flag, negative_flag}
            assert any(both <= set(group) for group in groups), spec.name

    def test_register_members_exclude_the_registers_switch(self) -> None:
        """Keys sharing a register name one destination, so only one may be given."""
        from agm.cli_support.run_options import _exclusive_flag_groups
        from agm.config.engine_keys import TRACE_ENGINE_KEYS

        flags = {f"--{name}" for name in TRACE_ENGINE_KEYS}
        assert any(
            flags <= set(group) and "--no-trace" in group for group in _exclusive_flag_groups()
        )


class TestReplStdlibOption:
    def test_no_stdlib_passed_to_repl_args(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.commands.repl as repl_mod

        calls: list[object] = []
        monkeypatch.setattr(repl_mod, "run", lambda a: calls.append(a))

        result = invoke(runner, ["repl", "--no-stdlib"])

        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert getattr(calls[0], "no_stdlib") is True


class TestMaxCallDepthOption:
    """Parser-contract tests for the --max-call-depth option on exec/repl."""

    def test_exec_max_call_depth_passed_to_args(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.commands.exec as exec_mod

        calls: list[object] = []
        monkeypatch.setattr(exec_mod, "run", lambda a: calls.append(a))

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(runner, ["exec", str(agl_file), "--max-call-depth", "42"])
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert getattr(calls[0], "max_call_depth") == 42

    def test_exec_max_call_depth_defaults_to_none(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.commands.exec as exec_mod

        calls: list[object] = []
        monkeypatch.setattr(exec_mod, "run", lambda a: calls.append(a))

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(runner, ["exec", str(agl_file)])
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert getattr(calls[0], "max_call_depth") is None

    def test_repl_max_call_depth_passed_to_args(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.commands.repl as repl_mod

        calls: list[object] = []
        monkeypatch.setattr(repl_mod, "run", lambda a: calls.append(a))

        result = invoke(runner, ["repl", "--max-call-depth", "7"])
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert getattr(calls[0], "max_call_depth") == 7


class TestParseErrorRetriesOption:
    """``--parse-error-retries`` reaches exec and repl argument containers."""

    @pytest.mark.parametrize(
        ("command", "flags", "expected"),
        [
            ("exec", ["--parse-error-retries", "3"], 3),
            ("exec", [], None),
            ("repl", ["--parse-error-retries", "0"], 0),
            ("repl", [], None),
        ],
    )
    def test_flag_passed_to_args(
        self,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        command: str,
        flags: list[str],
        expected: int | None,
    ) -> None:
        import agm.commands.exec as exec_mod
        import agm.commands.repl as repl_mod

        calls: list[object] = []
        monkeypatch.setattr(exec_mod if command == "exec" else repl_mod, "run", calls.append)
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        target = [str(agl_file)] if command == "exec" else []

        result = invoke(runner, [command, *flags, *target])

        assert result.exit_code == 0, result.output
        assert getattr(calls[0], "parse_error_retries") == expected
