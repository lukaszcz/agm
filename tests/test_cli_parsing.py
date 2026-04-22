"""Tests for CLI argument handling through the Typer application."""

from __future__ import annotations

from typing import Protocol

import pytest
from click.testing import CliRunner, Result
from typer.main import get_command

import agm.cli as cli


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
    def test_config_cp(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.config_copy_command)
        result = invoke(runner, ["config", "cp", "mydir"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].dirname == "mydir"

    def test_config_copy_rejects_d_option(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.config_copy_command)
        result = invoke(runner, ["config", "copy", "-d", "/some/dir", "target"])
        assert result.exit_code != 0
        assert len(calls) == 0

    def test_config_copy_rejects_dir_long_option(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.config_copy_command)
        result = invoke(runner, ["config", "copy", "--dir", "/some/dir", "target"])
        assert result.exit_code != 0
        assert len(calls) == 0

    def test_config_cp_missing_dirname(self, runner: CliRunner) -> None:
        result = invoke(runner, ["config", "cp"])
        assert result.exit_code != 0


class TestWorktreeNew:
    def test_wt_new(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.worktree_new_command)
        result = invoke(runner, ["wt", "new", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/y"
        assert calls[0].worktrees_dir is None

    def test_wt_new_with_d(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.worktree_new_command)
        result = invoke(runner, ["wt", "new", "-d", "/custom", "feat/z"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].worktrees_dir == "/custom"
        assert calls[0].branch == "feat/z"

    def test_wt_new_with_dir_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.worktree_new_command)
        result = invoke(runner, ["wt", "new", "--dir", "/custom", "feat/z"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].worktrees_dir == "/custom"
        assert calls[0].branch == "feat/z"

    def test_wt_new_missing_branch(self, runner: CliRunner) -> None:
        result = invoke(runner, ["wt", "new"])
        assert result.exit_code != 0

    def test_wt_co_is_rejected(self, runner: CliRunner) -> None:
        result = invoke(runner, ["wt", "co", "feat/x"])
        assert result.exit_code != 0

    def test_worktree_checkout_is_rejected(self, runner: CliRunner) -> None:
        result = invoke(runner, ["worktree", "checkout", "feat/x"])
        assert result.exit_code != 0


class TestWorktreeSetup:
    def test_wt_setup(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.worktree_setup_command)
        result = invoke(runner, ["wt", "setup"])
        assert result.exit_code == 0
        assert len(calls) == 1


class TestWorktreeRemove:
    def test_wt_rm(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.worktree_remove_command)
        result = invoke(runner, ["wt", "rm", "old-branch"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "old-branch"
        assert calls[0].force is False

    def test_worktree_remove_force(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.worktree_remove_command)
        result = invoke(runner, ["worktree", "remove", "-f", "old-branch"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].force is True
        assert calls[0].branch == "old-branch"

    def test_worktree_remove_force_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.worktree_remove_command)
        result = invoke(runner, ["worktree", "remove", "--force", "old-branch"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].force is True
        assert calls[0].branch == "old-branch"

    def test_wt_rm_missing_branch(self, runner: CliRunner) -> None:
        result = invoke(runner, ["wt", "rm"])
        assert result.exit_code != 0


class TestDep:
    def test_dep_new(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.dep_new_command)
        result = invoke(runner, ["dep", "new", "https://github.com/org/repo.git"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].repo_url == "https://github.com/org/repo.git"
        assert calls[0].branch is None

    def test_dep_new_with_branch(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.dep_new_command)
        result = invoke(runner, ["dep", "new", "-b", "main", "https://github.com/org/repo.git"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "main"

    def test_dep_new_with_branch_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.dep_new_command)
        result = invoke(
            runner,
            ["dep", "new", "--branch", "main", "https://github.com/org/repo.git"],
        )
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "main"

    def test_dep_switch(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.dep_switch_command)
        result = invoke(runner, ["dep", "switch", "mylib", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].dep == "mylib"
        assert calls[0].branch == "feat/x"
        assert calls[0].create_branch is False

    def test_dep_switch_create(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.dep_switch_command)
        result = invoke(runner, ["dep", "switch", "-b", "mylib", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].create_branch is True

    def test_dep_switch_create_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.dep_switch_command)
        result = invoke(runner, ["dep", "switch", "--branch", "mylib", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].create_branch is True

    def test_dep_rm(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.dep_remove_command)
        result = invoke(runner, ["dep", "rm", "mylib/feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].target == "mylib/feat/x"
        assert calls[0].all is False

    def test_dep_rm_all(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.dep_remove_command)
        result = invoke(runner, ["dep", "rm", "--all", "mylib"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].target == "mylib"
        assert calls[0].all is True

    def test_dep_rm_missing_target(self, runner: CliRunner) -> None:
        result = invoke(runner, ["dep", "rm"])
        assert result.exit_code != 0

    def test_dep_missing_subcommand_shows_help(self, runner: CliRunner) -> None:
        result = invoke(runner, ["dep"])
        assert result.exit_code == 0
        assert "agm dep" in result.stdout


class TestFetch:
    def test_fetch(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.fetch_command)
        result = invoke(runner, ["fetch"])
        assert result.exit_code == 0
        assert len(calls) == 1

    def test_fetch_rejects_args(self, runner: CliRunner) -> None:
        result = invoke(runner, ["fetch", "extra"])
        assert result.exit_code != 0


class TestInit:
    def test_init_project_and_url(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.init_command)
        result = invoke(runner, ["init", "myproj", "https://github.com/org/repo.git"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].positional == ["myproj", "https://github.com/org/repo.git"]
        assert calls[0].branch is None
        assert calls[0].embedded is False
        assert calls[0].workspace is False

    def test_init_url_only(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.init_command)
        result = invoke(runner, ["init", "https://github.com/org/repo.git"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].positional == ["https://github.com/org/repo.git"]

    def test_init_with_branch(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.init_command)
        result = invoke(runner, ["init", "-b", "dev", "myproj"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "dev"
        assert calls[0].positional == ["myproj"]

    def test_init_with_branch_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.init_command)
        result = invoke(runner, ["init", "--branch", "dev", "myproj"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "dev"
        assert calls[0].positional == ["myproj"]

    def test_init_with_embedded(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.init_command)
        result = invoke(runner, ["init", "--embedded", "myproj"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].embedded is True
        assert calls[0].positional == ["myproj"]

    def test_init_with_workspace(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.init_command)
        result = invoke(runner, ["init", "--workspace", "myproj"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].workspace is True
        assert calls[0].positional == ["myproj"]

    def test_init_missing_args(self, runner: CliRunner) -> None:
        result = invoke(runner, ["init"])
        assert result.exit_code != 0


class TestOpen:
    def test_open_missing_target(self, runner: CliRunner) -> None:
        result = invoke(runner, ["open"])
        assert result.exit_code != 0

    def test_open_repo(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.open_command)
        result = invoke(runner, ["open", "repo"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].detached is False
        assert calls[0].pane_count is None
        assert calls[0].parent is None
        assert calls[0].branch == "repo"

    def test_open_with_branch(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.open_command)
        result = invoke(runner, ["open", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/x"

    def test_open_with_pane_count(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.open_command)
        result = invoke(runner, ["open", "-n", "6", "repo"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "6"

    def test_open_with_num_panes_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.open_command)
        result = invoke(runner, ["open", "--num-panes", "6", "repo"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "6"

    def test_open_with_parent_and_branch(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.open_command)
        result = invoke(runner, ["open", "-p", "main", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/y"
        assert calls[0].parent == "main"

    def test_open_with_parent_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.open_command)
        result = invoke(runner, ["open", "--parent", "main", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/y"
        assert calls[0].parent == "main"

    def test_open_with_all(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.open_command)
        result = invoke(runner, ["open", "-n", "2", "-p", "main", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "2"
        assert calls[0].parent == "main"
        assert calls[0].branch == "feat/y"

    def test_open_detached(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.open_command)
        result = invoke(runner, ["open", "-d", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].detached is True
        assert calls[0].branch == "feat/y"

    def test_open_detach_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.open_command)
        result = invoke(runner, ["open", "--detach", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].detached is True
        assert calls[0].branch == "feat/y"


class TestClose:
    def test_close_branch(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.close_command)
        result = invoke(runner, ["close", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/x"

    def test_close_missing_branch(self, runner: CliRunner) -> None:
        result = invoke(runner, ["close"])
        assert result.exit_code != 0


class TestRun:
    def test_run_simple(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.run_command)
        result = invoke(runner, ["run", "npm", "test"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].run_command == ["npm", "test"]
        assert calls[0].no_patch is False
        assert calls[0].memory is None
        assert calls[0].settings_file is None

    def test_run_with_f(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.run_command)
        result = invoke(runner, ["run", "-f", "ci.json", "make"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].settings_file == "ci.json"
        assert calls[0].run_command == ["make"]

    def test_run_with_file_long(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.run_command)
        result = invoke(runner, ["run", "--file", "ci.json", "make"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].settings_file == "ci.json"
        assert calls[0].run_command == ["make"]

    def test_run_no_patch(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.run_command)
        result = invoke(runner, ["run", "--no-patch", "echo", "hi"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].no_patch is True
        assert calls[0].run_command == ["echo", "hi"]

    def test_run_no_sandbox(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.run_command)
        result = invoke(runner, ["run", "--no-sandbox", "echo", "hi"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].no_sandbox is True
        assert calls[0].run_command == ["echo", "hi"]

    def test_run_with_memory(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.run_command)
        result = invoke(runner, ["run", "--memory", "8G", "echo", "hi"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].memory == "8G"
        assert calls[0].run_command == ["echo", "hi"]

    def test_run_no_command_shows_help(self, runner: CliRunner) -> None:
        result = invoke(runner, ["run"])
        assert result.exit_code == 0
        assert "agm run" in result.stdout


class TestLoop:
    def test_loop(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name is None
        assert calls[0].runner is None
        assert calls[0].runner_args == []
        assert calls[0].selector is None
        assert calls[0].tasks_dir is None
        assert calls[0].no_log is False
        assert calls[0].log_file is None

    def test_loop_with_positional_command(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "codex"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "codex"
        assert calls[0].runner is None
        assert calls[0].runner_args == []
        assert calls[0].selector is None

    def test_loop_with_runner(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "--runner", "opencode prompt"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].runner == "opencode prompt"

    def test_loop_with_runner_args_after_separator(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "claude", "--", "-p", "--model", "sonnet"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].runner_args == ["-p", "--model", "sonnet"]

    def test_loop_with_selector(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "--selector", "claude -p"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].selector == "claude -p"

    def test_loop_with_tasks_dir(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "--tasks-dir", "custom/tasks"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].tasks_dir == "custom/tasks"

    def test_loop_with_no_log(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "--no-log"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].no_log is True

    def test_loop_with_log_file(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "--log-file", "custom/loop.log"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].log_file == "custom/loop.log"


class TestTmuxOpen:
    def test_tmux_open_bare(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.tmux_open_command)
        result = invoke(runner, ["tmux", "open"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].detach is False
        assert calls[0].pane_count is None
        assert calls[0].session_name is None

    def test_tmux_open_with_all(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.tmux_open_command)
        result = invoke(runner, ["tmux", "open", "-d", "-n", "8", "mysession"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].detach is True
        assert calls[0].pane_count == "8"
        assert calls[0].session_name == "mysession"

    def test_tmux_open_detach_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.tmux_open_command)
        result = invoke(runner, ["tmux", "open", "--detach"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].detach is True

    def test_tmux_open_num_panes_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.tmux_open_command)
        result = invoke(runner, ["tmux", "open", "--num-panes", "8", "mysession"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "8"
        assert calls[0].session_name == "mysession"


class TestTmuxClose:
    def test_tmux_close(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.tmux_close_command)
        result = invoke(runner, ["tmux", "close", "mysession"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].session_name == "mysession"

    def test_tmux_close_missing_name(self, runner: CliRunner) -> None:
        result = invoke(runner, ["tmux", "close"])
        assert result.exit_code != 0


class TestTmuxLayout:
    def test_tmux_layout(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.tmux_layout_command)
        result = invoke(runner, ["tmux", "layout", "4"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "4"
        assert calls[0].window_id is None

    def test_tmux_layout_with_explicit_window(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.tmux_layout_command)
        result = invoke(runner, ["tmux", "layout", "4", "--window", "@1"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "4"
        assert calls[0].window_id == "@1"

    def test_tmux_layout_with_window_short(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.tmux_layout_command)
        result = invoke(runner, ["tmux", "layout", "4", "-w", "@1"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "4"
        assert calls[0].window_id == "@1"

    def test_tmux_layout_missing_args(self, runner: CliRunner) -> None:
        result = invoke(runner, ["tmux", "layout"])
        assert result.exit_code != 0


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


class TestTopLevel:
    def test_no_command(self, runner: CliRunner) -> None:
        result = invoke(runner, [])
        assert result.exit_code == 0
        assert "agm - Agent Management Framework" in result.stdout

    def test_unknown_command(self, runner: CliRunner) -> None:
        result = invoke(runner, ["bogus"])
        assert result.exit_code != 0

    def test_show_completion_is_available(self, runner: CliRunner) -> None:
        result = invoke(runner, ["--show-completion"])
        assert result.exit_code == 0


class TestHelpTextCoverage:
    def test_every_canonical_command_has_help_text(self) -> None:
        from agm.cli import _HELP_TEXTS

        canonical_commands = {
            "open",
            "close",
            "init",
            "fetch",
            "loop",
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
        from agm.cli import _COMMAND_OVERVIEW, _HELP_TEXTS

        for name, _ in _COMMAND_OVERVIEW:
            canonical = name.split(" (")[0]
            assert canonical in _HELP_TEXTS, (
                f"overview lists '{canonical}' but _HELP_TEXTS has no entry"
            )

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
