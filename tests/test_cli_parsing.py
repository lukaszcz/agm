"""Tests for CLI argument handling through the Typer application."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

import click
import pytest
from click.testing import CliRunner, Result
from typer.main import get_command

import agm.cli as cli
import agm.parser as parser_helpers
from agm.core import dry_run as dry_run_state


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
        assert "No such option" in result.output
        assert len(calls) == 0

    def test_config_copy_rejects_dir_long_option(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.config_copy_command)
        result = invoke(runner, ["config", "copy", "--dir", "/some/dir", "target"])
        assert result.exit_code != 0
        assert "No such option" in result.output
        assert len(calls) == 0

    def test_config_cp_missing_dirname(self, runner: CliRunner) -> None:
        result = invoke(runner, ["config", "cp"])
        assert result.exit_code != 0
        assert "required" in result.output

    def test_config_env(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.config_env_command)
        result = invoke(runner, ["config", "env"])
        assert result.exit_code == 0
        assert len(calls) == 1

    def test_config_update(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.config_update_command)
        result = invoke(runner, ["config", "update"])
        assert result.exit_code == 0
        assert len(calls) == 1


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
    def test_workspace_setup(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[None] = []

        def record() -> None:
            calls.append(None)

        monkeypatch.setattr(cli.workspace_setup_command, "run", record)
        result = invoke(runner, ["workspace", "setup"])
        assert result.exit_code == 0
        assert len(calls) == 1

    def test_wsp_setup(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[None] = []

        def record() -> None:
            calls.append(None)

        monkeypatch.setattr(cli.workspace_setup_command, "run", record)
        result = invoke(runner, ["wsp", "setup"])
        assert result.exit_code == 0
        assert len(calls) == 1

    def test_top_level_setup_is_rejected(self, runner: CliRunner) -> None:
        result = invoke(runner, ["setup"])
        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_workspace_list(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[bool] = []

        def record(*, verbose: bool = False) -> None:
            calls.append(verbose)

        monkeypatch.setattr(cli.workspace_list_command, "run", record)
        result = invoke(runner, ["workspace", "list"])

        assert result.exit_code == 0
        assert calls == [False]

    def test_workspace_shell_regen(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def record(*, shell_dir: str) -> None:
            calls.append(shell_dir)

        monkeypatch.setattr(cli.workspace_shell_regen_command, "run", record)
        result = invoke(runner, ["workspace", "shell-regen", "/tmp/agm-shell-dir"])

        assert result.exit_code == 0
        assert calls == ["/tmp/agm-shell-dir"]

    def test_wsp_shell_regen(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def record(*, shell_dir: str) -> None:
            calls.append(shell_dir)

        monkeypatch.setattr(cli.workspace_shell_regen_command, "run", record)
        result = invoke(runner, ["wsp", "shell-regen", "/tmp/agm-shell-dir"])

        assert result.exit_code == 0
        assert calls == ["/tmp/agm-shell-dir"]

    def test_wsp_list_verbose(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[bool] = []

        def record(*, verbose: bool = False) -> None:
            calls.append(verbose)

        monkeypatch.setattr(cli.workspace_list_command, "run", record)
        result = invoke(runner, ["wsp", "list", "--verbose"])

        assert result.exit_code == 0
        assert calls == [True]

    def test_top_level_list_is_rejected(self, runner: CliRunner) -> None:
        result = invoke(runner, ["list"])
        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_workspace_open(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.workspace_open_command)
        result = invoke(runner, ["workspace", "open", "-n", "6", "repo"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "repo"
        assert calls[0].pane_count == "6"

    def test_wsp_close(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.workspace_close_command)
        result = invoke(runner, ["wsp", "close", "-D", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/x"
        assert calls[0].force_delete is True

    def test_wsp_close_keep_workspace(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.workspace_close_command)
        result = invoke(runner, ["wsp", "close", "--keep-workspace", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].keep_workspace is True


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
        assert "required" in result.output


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

    def test_dep_remove(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.dep_remove_command)
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
        calls = make_recorder(monkeypatch, cli.sync_fetch_command)
        result = invoke(runner, ["sync", "fetch"])
        assert result.exit_code == 0
        assert len(calls) == 1

    def test_sync_fetch_accepts_global_dry_run(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        observed: list[bool] = []

        def record(args: object) -> None:
            del args
            observed.append(dry_run_state.enabled())

        monkeypatch.setattr(cli.sync_fetch_command, "run", record)
        result = invoke(runner, ["--dry-run", "sync", "fetch"])
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
        calls = make_recorder(monkeypatch, cli.sync_pull_command)
        result = invoke(runner, ["sync", "pull"])
        assert result.exit_code == 0
        assert len(calls) == 1

    def test_sync_pull_accepts_global_dry_run(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        observed: list[bool] = []

        def record(args: object) -> None:
            del args
            observed.append(dry_run_state.enabled())

        monkeypatch.setattr(cli.sync_pull_command, "run", record)
        result = invoke(runner, ["--dry-run", "sync", "pull"])
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


class TestReviewReviseRefine:
    def test_review_options(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.review_command, "run")
        result = invoke(
            runner,
            [
                "review",
                "--runner",
                "codex exec",
                "--scope",
                "branch",
                "--aspects",
                "correctness",
                "--extra-aspects",
                "tests",
                "--prompt",
                "review this",
                "--extra-prompt",
                "extra",
                "--review-file",
                "review.md",
            ],
        )

        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].runner == "codex exec"
        assert calls[0].scope == "branch"
        assert calls[0].aspects == "correctness"
        assert calls[0].extra_aspects == "tests"
        assert calls[0].prompt == "review this"
        assert calls[0].extra_prompt == "extra"
        assert calls[0].review_file == "review.md"
        assert calls[0].no_review_file is False

    def test_review_rejects_no_review_file_with_review_file(
        self, runner: CliRunner
    ) -> None:
        result = invoke(
            runner,
            ["review", "--no-review-file", "--review-file", "review.md"],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_review_accepts_config_command(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.review_command, "run")
        result = invoke(runner, ["review", "frontend", "--scope", "branch"])

        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "frontend"
        assert calls[0].scope == "branch"

    def test_review_rejects_prompt_with_prompt_file(self, runner: CliRunner) -> None:
        result = invoke(runner, ["review", "--prompt", "text", "--prompt-file", "file.md"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_review_rejects_extra_prompt_with_extra_prompt_file(
        self, runner: CliRunner
    ) -> None:
        result = invoke(
            runner,
            ["review", "--extra-prompt", "text", "--extra-prompt-file", "file.md"],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_revise_requires_review_file(self, runner: CliRunner) -> None:
        result = invoke(runner, ["revise"])
        assert result.exit_code != 0
        assert "required" in result.output

    def test_revise_options(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.revise_command, "run")
        result = invoke(
            runner,
            [
                "revise",
                "review.md",
                "--runner",
                "codex exec",
                "--prompt-file",
                "revise.md",
                "--extra-prompt-file",
                "extra.md",
            ],
        )

        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].review_file == "review.md"
        assert calls[0].runner == "codex exec"
        assert calls[0].prompt_file == "revise.md"
        assert calls[0].extra_prompt_file == "extra.md"

    def test_revise_accepts_config_command(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.revise_command, "run")
        result = invoke(runner, ["revise", "frontend", "review.md"])

        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "frontend"
        assert calls[0].review_file == "review.md"

    def test_revise_single_argument_remains_review_file(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.revise_command, "run")
        result = invoke(runner, ["revise", "review.md"])

        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name is None
        assert calls[0].review_file == "review.md"

    def test_revise_unknown_option_usage_has_distinct_positionals(
        self, runner: CliRunner
    ) -> None:
        result = invoke(runner, ["revise", "--unknown"])

        assert result.exit_code != 0
        assert "No such option" in result.output
        assert "[COMMAND] REVIEW_FILE [REVIEW_FILE]" not in result.output
        assert "COMMAND_OR_REVIEW_FILE" in result.output

    def test_refine_options(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.refine_command, "run")
        result = invoke(
            runner,
            [
                "refine",
                "--max-steps",
                "4",
                "--runner",
                "both",
                "--reviewer",
                "reviewer",
                "--reviser",
                "reviser",
                "--scope",
                "branch",
                "--aspects",
                "correctness",
                "--review-prompt-file",
                "review.md",
                "--extra-review-prompt",
                "extra review",
                "--revise-prompt",
                "revise",
                "--extra-revise-prompt-file",
                "extra-revise.md",
                "--log-file",
                "refine.log",
                "--save-review",
                "--review-file",
                "reviews/last.md",
            ],
        )

        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].max_steps == 4
        assert calls[0].runner == "both"
        assert calls[0].reviewer == "reviewer"
        assert calls[0].reviser == "reviser"
        assert calls[0].scope == "branch"
        assert calls[0].aspects == "correctness"
        assert calls[0].review_prompt_file == "review.md"
        assert calls[0].extra_review_prompt == "extra review"
        assert calls[0].revise_prompt == "revise"
        assert calls[0].extra_revise_prompt_file == "extra-revise.md"
        assert calls[0].log_file == "refine.log"
        assert calls[0].no_log is False
        assert calls[0].save_review is True
        assert calls[0].review_file == "reviews/last.md"

    def test_refine_accepts_config_command(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.refine_command, "run")
        result = invoke(runner, ["refine", "frontend", "--max-steps", "2"])

        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "frontend"
        assert calls[0].max_steps == 2

    def test_refine_rejects_non_positive_max_steps(self, runner: CliRunner) -> None:
        result = invoke(runner, ["refine", "--max-steps", "0"])
        assert result.exit_code != 0
        assert "must be positive" in result.output

    def test_refine_accepts_max_steps_unlimited(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.refine_command, "run")
        result = invoke(runner, ["refine", "--max-steps", "unlimited"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].max_steps is None
        assert calls[0].no_max_steps is True

    def test_refine_accepts_no_max_steps_flag(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.refine_command, "run")
        result = invoke(runner, ["refine", "--no-max-steps"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].max_steps is None
        assert calls[0].no_max_steps is True

    def test_refine_rejects_no_max_steps_with_max_steps(self, runner: CliRunner) -> None:
        result = invoke(runner, ["refine", "--no-max-steps", "--max-steps", "5"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_refine_rejects_invalid_max_steps_value(self, runner: CliRunner) -> None:
        result = invoke(runner, ["refine", "--max-steps", "abc"])
        assert result.exit_code != 0
        assert "must be a positive integer" in result.output

    def test_refine_rejects_no_log_with_log_file(self, runner: CliRunner) -> None:
        result = invoke(runner, ["refine", "--no-log", "--log-file", "out.log"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_refine_rejects_review_prompt_with_review_prompt_file(
        self, runner: CliRunner
    ) -> None:
        result = invoke(
            runner,
            ["refine", "--review-prompt", "text", "--review-prompt-file", "file.md"],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_refine_rejects_extra_revise_prompt_with_extra_revise_prompt_file(
        self, runner: CliRunner
    ) -> None:
        result = invoke(
            runner,
            [
                "refine",
                "--extra-revise-prompt",
                "text",
                "--extra-revise-prompt-file",
                "file.md",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_refine_save_review_defaults_to_none(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.refine_command, "run")
        result = invoke(runner, ["refine"])

        assert result.exit_code == 0
        assert calls[0].save_review is None

    def test_refine_no_save_review_disables_save(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.refine_command, "run")
        result = invoke(runner, ["refine", "--no-save-review"])

        assert result.exit_code == 0
        assert calls[0].save_review is False

    def test_refine_rejects_extra_review_prompt_with_extra_review_prompt_file(
        self, runner: CliRunner
    ) -> None:
        result = invoke(
            runner,
            [
                "refine",
                "--extra-review-prompt",
                "text",
                "--extra-review-prompt-file",
                "file.md",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output


class TestDryRun:
    def test_open_accepts_command_level_dry_run(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        observed: list[bool] = []

        def record(args: RecordedArgs) -> None:
            observed.append(dry_run_state.enabled())
            assert args.branch == "repo"

        monkeypatch.setattr(cli.workspace_open_command, "run", record)
        result = invoke(runner, ["open", "--dry-run", "repo"])
        assert result.exit_code == 0
        assert observed == [True]


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
        assert calls[0].split is False

    def test_init_url_only(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.init_command)
        result = invoke(runner, ["init", "https://github.com/org/repo.git"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].positional == ["https://github.com/org/repo.git"]
        assert calls[0].clone is False

    def test_init_with_clone(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.init_command)
        result = invoke(runner, ["init", "--clone", "https://github.com/org/repo.git"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].positional == ["https://github.com/org/repo.git"]
        assert calls[0].clone is True

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

    def test_init_with_split(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.init_command)
        result = invoke(runner, ["init", "--split", "myproj"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].split is True
        assert calls[0].positional == ["myproj"]

    def test_init_with_no_repo_git(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.init_command)
        result = invoke(runner, ["init", "--no-repo-git", "myproj"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].no_repo_git is True
        assert calls[0].positional == ["myproj"]

    def test_init_with_workspace_is_rejected(self, runner: CliRunner) -> None:
        result = invoke(runner, ["init", "--workspace", "myproj"])
        assert result.exit_code != 0
        assert "No such option" in result.output

    def test_init_without_args(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.init_command)
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
        calls = make_recorder(monkeypatch, cli.workspace_open_command)
        result = invoke(runner, ["open", "repo"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].detached is False
        assert calls[0].pane_count is None
        assert calls[0].parent is None
        assert calls[0].branch == "repo"

    def test_open_with_branch(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.workspace_open_command)
        result = invoke(runner, ["open", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/x"

    def test_open_with_pane_count(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.workspace_open_command)
        result = invoke(runner, ["open", "-n", "6", "repo"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "6"

    def test_open_with_num_panes_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.workspace_open_command)
        result = invoke(runner, ["open", "--num-panes", "6", "repo"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "6"

    def test_open_with_parent_and_branch(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.workspace_open_command)
        result = invoke(runner, ["open", "-p", "main", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/y"
        assert calls[0].parent == "main"

    def test_open_with_parent_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.workspace_open_command)
        result = invoke(runner, ["open", "--parent", "main", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/y"
        assert calls[0].parent == "main"

    def test_open_with_all(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.workspace_open_command)
        result = invoke(runner, ["open", "-n", "2", "-p", "main", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].pane_count == "2"
        assert calls[0].parent == "main"
        assert calls[0].branch == "feat/y"

    def test_open_detached(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.workspace_open_command)
        result = invoke(runner, ["open", "-d", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].detached is True
        assert calls[0].branch == "feat/y"

    def test_open_detach_long(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.workspace_open_command)
        result = invoke(runner, ["open", "--detach", "feat/y"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].detached is True
        assert calls[0].branch == "feat/y"


class TestClose:
    def test_close_branch(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.workspace_close_command)
        result = invoke(runner, ["close", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/x"

    def test_close_missing_branch(self, runner: CliRunner) -> None:
        result = invoke(runner, ["close"])
        assert result.exit_code != 0
        assert "required" in result.output

    def test_close_D_flag(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.workspace_close_command)
        result = invoke(runner, ["close", "-D", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].branch == "feat/x"
        assert calls[0].force_delete is True

    def test_close_default_no_D(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.workspace_close_command)
        result = invoke(runner, ["close", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].force_delete is False

    def test_close_keep_branch_flag(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.workspace_close_command)
        result = invoke(runner, ["close", "--keep-branch", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].keep_branch is True
        assert calls[0].keep_workspace is False

    def test_close_keep_workspace_implies_keep_branch_in_args(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.workspace_close_command)
        result = invoke(runner, ["close", "--keep-workspace", "feat/x"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].keep_branch is True
        assert calls[0].keep_workspace is True


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
        assert calls[0].swap is None
        assert calls[0].no_memory_limit is False
        assert calls[0].no_swap_limit is False
        assert calls[0].run_command == ["echo", "hi"]

    def test_run_with_unlimited_memory(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.run_command)
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
        calls = make_recorder(monkeypatch, cli.run_command)
        result = invoke(runner, ["run", "--no-memory-limit", "echo", "hi"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].memory is None
        assert calls[0].swap is None
        assert calls[0].no_memory_limit is True
        assert calls[0].no_swap_limit is False
        assert calls[0].run_command == ["echo", "hi"]

    def test_run_with_swap(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.run_command)
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
        calls = make_recorder(monkeypatch, cli.run_command)
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
        calls = make_recorder(monkeypatch, cli.run_command)
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


class TestLoop:
    def test_loop(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop"])
        assert result.exit_code == 0
        assert len(calls) == 0
        assert "agm loop" in result.stdout

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
        assert calls[0].no_selector is False

    def test_loop_with_runner(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "--runner", "opencode prompt", "claude"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].runner == "opencode prompt"

    def test_loop_with_runner_args_after_positional_command(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "claude", "-p", "--model", "sonnet"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].runner_args == ["-p", "--model", "sonnet"]

    def test_loop_with_selector(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "--selector", "claude -p", "codex"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "codex"
        assert calls[0].selector == "claude -p"
        assert calls[0].no_selector is False

    def test_loop_with_no_selector(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "--no-selector", "claude"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].selector is None
        assert calls[0].no_selector is True

    def test_loop_rejects_selector_with_no_selector(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "--selector", "cmd", "--no-selector", "claude"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output
        assert len(calls) == 0

    def test_loop_with_tasks_dir(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "--tasks-dir", "custom/tasks", "codex"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "codex"
        assert calls[0].tasks_dir == "custom/tasks"

    def test_loop_with_no_log(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "--no-log"])
        assert result.exit_code == 0
        assert len(calls) == 0
        assert "agm loop" in result.stdout

    def test_loop_with_no_log_before_positional_command(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "--no-log", "claude"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].runner_args == []
        assert calls[0].no_log is True

    def test_loop_treats_loop_flags_after_command_as_runner_args(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "claude", "--no-log"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].runner_args == ["--no-log"]
        assert calls[0].no_log is False

    def test_loop_with_log_file(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "--log-file", "custom/loop.log", "claude"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].log_file == "custom/loop.log"

    def test_loop_run_with_positional_command(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_run_command)
        result = invoke(runner, ["loop", "run", "claude"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].runner is None
        assert calls[0].runner_args == []
        assert calls[0].selector is None
        assert calls[0].no_selector is False
        assert calls[0].tasks_dir is None

    def test_loop_run_without_positional_command(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_run_command)
        result = invoke(runner, ["loop", "run"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name is None
        assert calls[0].runner is None
        assert calls[0].runner_args == []
        assert calls[0].selector is None
        assert calls[0].no_selector is False
        assert calls[0].tasks_dir is None

    def test_loop_run_with_no_selector(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_run_command)
        result = invoke(runner, ["loop", "run", "--no-selector", "claude"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].no_selector is True

    def test_loop_shorthand_dispatches_to_run(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_run_command)
        result = invoke(runner, ["loop", "claude"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].runner_args == []

    def test_loop_select(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = make_recorder(monkeypatch, cli.loop_select_command)
        result = invoke(runner, ["loop", "select"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name is None
        assert calls[0].runner is None
        assert calls[0].runner_args == []
        assert calls[0].selector is None
        assert calls[0].no_selector is False
        assert calls[0].tasks_dir is None

    def test_loop_select_with_positional_command_and_runner_args(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_select_command)
        result = invoke(runner, ["loop", "select", "claude", "-p", "--model", "sonnet"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].runner_args == ["-p", "--model", "sonnet"]

    def test_loop_select_with_selector_override(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_select_command)
        result = invoke(runner, ["loop", "select", "--selector", "codex exec", "claude"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].selector == "codex exec"
        assert calls[0].no_selector is False

    def test_loop_select_with_no_selector(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_select_command)
        result = invoke(runner, ["loop", "select", "--no-selector", "claude"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].no_selector is True

    def test_loop_select_rejects_selector_with_no_selector(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_select_command)
        result = invoke(runner, ["loop", "select", "--selector", "cmd", "--no-selector"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output
        assert len(calls) == 0

    def test_loop_step_with_positional_command_and_runner_args(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_step_command)
        result = invoke(runner, ["loop", "step", "claude", "-p", "--model", "sonnet"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].runner_args == ["-p", "--model", "sonnet"]

    def test_loop_step_without_positional_command(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_step_command)
        result = invoke(runner, ["loop", "step"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name is None
        assert calls[0].runner is None
        assert calls[0].runner_args == []
        assert calls[0].selector is None
        assert calls[0].no_selector is False
        assert calls[0].tasks_dir is None

    def test_loop_step_with_no_selector(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_step_command)
        result = invoke(runner, ["loop", "step", "--no-selector", "claude"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].no_selector is True

    def test_loop_with_prompt(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "--prompt", "fix the bug", "claude"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].prompt == "fix the bug"
        assert calls[0].prompt_file is None

    def test_loop_with_prompt_file(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "--prompt-file", "/tmp/task.md", "claude"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].prompt is None
        assert calls[0].prompt_file == "/tmp/task.md"

    def test_loop_rejects_prompt_with_prompt_file(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(
            runner, ["loop", "--prompt", "text", "--prompt-file", "file.md", "claude"]
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output
        assert len(calls) == 0

    def test_loop_run_with_prompt(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_run_command)
        result = invoke(runner, ["loop", "run", "--prompt", "hello", "claude"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].prompt == "hello"
        assert calls[0].prompt_file is None

    def test_loop_step_with_prompt_file(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_step_command)
        result = invoke(runner, ["loop", "step", "--prompt-file", "task.md", "claude"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].prompt is None
        assert calls[0].prompt_file == "task.md"

    def test_loop_select_with_prompt(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_select_command)
        result = invoke(runner, ["loop", "select", "--prompt", "do it"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].prompt == "do it"
        assert calls[0].prompt_file is None

    def test_loop_select_with_prompt_file(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_select_command)
        result = invoke(runner, ["loop", "select", "--prompt-file", "prompt.md"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].prompt is None
        assert calls[0].prompt_file == "prompt.md"

    def test_loop_select_rejects_prompt_with_prompt_file(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_select_command)
        result = invoke(
            runner, ["loop", "select", "--prompt", "text", "--prompt-file", "file.md"]
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output
        assert len(calls) == 0

    def test_loop_with_selector_prompt(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(runner, ["loop", "--selector-prompt", "pick next task", "claude"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].selector_prompt == "pick next task"
        assert calls[0].selector_prompt_file is None

    def test_loop_with_selector_prompt_file(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(
            runner, ["loop", "--selector-prompt-file", "/tmp/selector.md", "claude"]
        )
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].command_name == "claude"
        assert calls[0].selector_prompt is None
        assert calls[0].selector_prompt_file == "/tmp/selector.md"

    def test_loop_rejects_selector_prompt_with_selector_prompt_file(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_command)
        result = invoke(
            runner,
            [
                "loop",
                "--selector-prompt",
                "text",
                "--selector-prompt-file",
                "file.md",
                "claude",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output
        assert len(calls) == 0

    def test_loop_run_with_selector_prompt(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_run_command)
        result = invoke(
            runner, ["loop", "run", "--selector-prompt", "select task", "claude"]
        )
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].selector_prompt == "select task"
        assert calls[0].selector_prompt_file is None

    def test_loop_step_with_selector_prompt_file(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_step_command)
        result = invoke(
            runner, ["loop", "step", "--selector-prompt-file", "sel.md", "claude"]
        )
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].selector_prompt is None
        assert calls[0].selector_prompt_file == "sel.md"

    def test_loop_select_with_selector_prompt(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_select_command)
        result = invoke(runner, ["loop", "select", "--selector-prompt", "review tasks"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].selector_prompt == "review tasks"
        assert calls[0].selector_prompt_file is None

    def test_loop_select_with_selector_prompt_file(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_select_command)
        result = invoke(
            runner, ["loop", "select", "--selector-prompt-file", "selector.md"]
        )
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0].selector_prompt is None
        assert calls[0].selector_prompt_file == "selector.md"

    def test_loop_select_rejects_selector_prompt_with_selector_prompt_file(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = make_recorder(monkeypatch, cli.loop_select_command)
        result = invoke(
            runner,
            [
                "loop",
                "select",
                "--selector-prompt",
                "text",
                "--selector-prompt-file",
                "file.md",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output
        assert len(calls) == 0


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
        assert "required" in result.output


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


class TestHelpTextCoverage:
    def test_every_canonical_command_has_help_text(self) -> None:
        from agm.cli import _HELP_TEXTS

        canonical_commands = {
            "open",
            "close",
            "init",
            "workspace",
            "sync",
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

    def test_command_reference_mentions_all_cli_commands(self) -> None:
        import re
        from pathlib import Path

        from typer.main import get_command

        from agm.cli import app

        cli_commands = set(get_command(app).list_commands(None))
        doc_text = "\n".join(
            path.read_text(encoding="utf-8") for path in sorted(Path("docs/commands").glob("*.md"))
        )
        documented = set(re.findall(r"`agm (\w+)", doc_text))
        missing = cli_commands - documented
        assert not missing, (
            f"docs/commands/ is missing entries for CLI commands: {sorted(missing)}"
        )


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
        with pytest.raises((SystemExit, click.exceptions.Exit)):
            cli._print_context_help(root, None, True)
        captured = capsys.readouterr()
        assert "agm - Agent Management Framework" in captured.out

    def test_prints_command_help_for_subcommand(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = click.Context(click.Command("agm"))
        root.parent = None
        sub = click.Context(click.Command("open"), parent=root, info_name="open")
        with pytest.raises((SystemExit, click.exceptions.Exit)):
            cli._print_context_help(sub, None, True)
        captured = capsys.readouterr()
        assert "agm open" in captured.out


class TestParseLoopArgs:
    def test_runner_flag(self) -> None:
        args = cli._parse_loop_args(
            ["--runner", "my-runner", "cmd"], command_path=["loop"]
        )
        assert args.runner == "my-runner"
        assert args.command_name == "cmd"

    def test_selector_flag(self) -> None:
        args = cli._parse_loop_args(
            ["--selector", "my-selector", "cmd"], command_path=["loop"]
        )
        assert args.selector == "my-selector"

    def test_tasks_dir_flag(self) -> None:
        args = cli._parse_loop_args(
            ["--tasks-dir", "custom/tasks", "cmd"], command_path=["loop"]
        )
        assert args.tasks_dir == "custom/tasks"

    def test_timeout_flag(self) -> None:
        args = cli._parse_loop_args(
            ["--timeout", "2m", "cmd"], command_path=["loop"]
        )
        assert args.timeout == 120

    def test_invalid_timeout_exits(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_loop_args(["--timeout", "bad", "cmd"], command_path=["loop"])

    def test_prompt_flag(self) -> None:
        args = cli._parse_loop_args(
            ["--prompt", "do the thing", "cmd"], command_path=["loop"]
        )
        assert args.prompt == "do the thing"
        assert args.prompt_file is None

    def test_prompt_file_flag(self) -> None:
        args = cli._parse_loop_args(
            ["--prompt-file", "/tmp/task.md", "cmd"], command_path=["loop"]
        )
        assert args.prompt_file == "/tmp/task.md"
        assert args.prompt is None

    def test_selector_prompt_flag(self) -> None:
        args = cli._parse_loop_args(
            ["--selector-prompt", "select task", "cmd"], command_path=["loop"]
        )
        assert args.selector_prompt == "select task"
        assert args.selector_prompt_file is None

    def test_selector_prompt_file_flag(self) -> None:
        args = cli._parse_loop_args(
            ["--selector-prompt-file", "/tmp/sel.md", "cmd"], command_path=["loop"]
        )
        assert args.selector_prompt_file == "/tmp/sel.md"
        assert args.selector_prompt is None

    def test_selector_and_no_selector_are_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_loop_args(
                ["--selector", "cmd", "--no-selector", "cmd2"],
                command_path=["loop"],
            )

    def test_prompt_and_prompt_file_are_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_loop_args(
                ["--prompt", "text", "--prompt-file", "file.md", "cmd"],
                command_path=["loop"],
            )

    def test_selector_prompt_and_selector_prompt_file_are_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_loop_args(
                [
                    "--selector-prompt",
                    "text",
                    "--selector-prompt-file",
                    "file.md",
                    "cmd",
                ],
                command_path=["loop"],
            )

    def test_empty_args_with_command_optional_true_returns_none_command(self) -> None:
        args = cli._parse_loop_args([], command_path=["loop", "run"], command_optional=True)
        assert args.command_name is None

    def test_empty_args_with_command_optional_false_exits(self) -> None:
        with pytest.raises((SystemExit, click.exceptions.Exit)):
            cli._parse_loop_args([], command_path=["loop"], command_optional=False)

    def test_no_log_and_log_file_mutually_exclusive_with_command(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_loop_args(
                ["--no-log", "--log-file", "out.log", "cmd"],
                command_path=["loop"],
            )

    def test_no_log_and_log_file_mutually_exclusive_without_command(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_loop_args(
                ["--no-log", "--log-file", "out.log"],
                command_path=["loop", "run"],
                command_optional=True,
            )

    def test_double_dash_stops_parsing(self) -> None:
        args = cli._parse_loop_args(
            ["--", "--runner", "val", "cmd"],
            command_path=["loop"],
        )
        # After --, the remaining args are treated as command + runner_args
        assert args.runner is None

    def test_runner_without_value_exits(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_loop_args(["--runner"], command_path=["loop"])


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
        for option in ("--max-iters", "--timeout", "--no-timeout", "--no-log-file"):
            assert option in result

    def test_repl_help_lists_max_iters(self) -> None:
        output = io.StringIO()
        parser_helpers.print_help_for_command_path(["repl"], file=output)
        assert "--max-iters" in output.getvalue()

    def test_print_command_help_with_file_param(self) -> None:
        output = io.StringIO()
        parser_helpers.print_command_help("open", file=output)
        result = output.getvalue()
        assert "agm open" in result


class TestHelpTextForPathValueError:
    def test_unknown_multi_segment_command_path_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown command path"):
            parser_helpers._help_text_for_path(["unknown", "sub"])


class TestHelpTextForPathUnknownMultiElement:
    def test_raises_value_error_for_unknown_multi_element_path(self) -> None:
        """_help_text_for_path raises ValueError for unknown multi-element paths."""
        with pytest.raises(ValueError, match="unknown command path"):
            parser_helpers._help_text_for_path(["nonexistent", "subcommand"])


class TestParseLoopSelectArgsMisc:
    """Miscellaneous _parse_loop_select_args parsing and validation."""

    def test_double_dash_stops_parsing(self) -> None:
        """Double dash separator stops option parsing."""
        args = cli._parse_loop_select_args(
            ["--", "--runner", "val", "cmd"],
            command_path=["loop", "select"],
        )
        # After --, the -- itself becomes command_name and the rest become runner_args
        assert args.runner is None
        assert args.command_name == "--"
        assert args.runner_args == ["--runner", "val", "cmd"]

    def test_runner_flag(self) -> None:
        """--runner flag sets the runner on the parsed args."""
        args = cli._parse_loop_select_args(
            ["--runner", "my-runner", "cmd"],
            command_path=["loop", "select"],
        )
        assert args.runner == "my-runner"
        assert args.command_name == "cmd"

    def test_tasks_dir_flag(self) -> None:
        """--tasks-dir flag sets the tasks directory on the parsed args."""
        args = cli._parse_loop_select_args(
            ["--tasks-dir", "custom/tasks", "cmd"],
            command_path=["loop", "select"],
        )
        assert args.tasks_dir == "custom/tasks"
        assert args.command_name == "cmd"

    def test_timeout_invalid_format_exits(self) -> None:
        """--timeout with an invalid format causes SystemExit."""
        with pytest.raises(SystemExit):
            cli._parse_loop_select_args(
                ["--timeout", "abc", "cmd"],
                command_path=["loop", "select"],
            )

    def test_timeout_flag(self) -> None:
        args = cli._parse_loop_select_args(
            ["--timeout", "1h", "cmd"],
            command_path=["loop", "select"],
        )
        assert args.timeout == 3600
        assert args.command_name == "cmd"


class TestConfigCopyCommand:
    """The 'config copy' CLI command function dispatches correctly."""

    def test_config_copy_via_cli(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[object] = []

        def record(args: object) -> None:
            calls.append(args)

        monkeypatch.setattr(cli.config_copy_command, "run", record)
        result = invoke(runner, ["config", "copy", "mydir"])
        assert result.exit_code == 0
        assert len(calls) == 1


class TestMainEntryPoint:
    """main() entry point delegates to app()."""

    def test_main_calls_app_and_shows_help(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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


class TestParseLoopArgsExtraPromptFlags:
    """Cover --extra-prompt, --extra-prompt-file, --extra-selector-prompt,
    --extra-selector-prompt-file flags in _parse_loop_args."""

    def test_extra_prompt_flag(self) -> None:
        args = cli._parse_loop_args(
            ["--extra-prompt", "extra context", "cmd"], command_path=["loop"]
        )
        assert args.extra_prompt == "extra context"
        assert args.extra_prompt_file is None

    def test_extra_prompt_file_flag(self) -> None:
        args = cli._parse_loop_args(
            ["--extra-prompt-file", "/tmp/extra.md", "cmd"], command_path=["loop"]
        )
        assert args.extra_prompt_file == "/tmp/extra.md"
        assert args.extra_prompt is None

    def test_extra_selector_prompt_flag(self) -> None:
        args = cli._parse_loop_args(
            ["--extra-selector-prompt", "extra select", "cmd"], command_path=["loop"]
        )
        assert args.extra_selector_prompt == "extra select"
        assert args.extra_selector_prompt_file is None

    def test_extra_selector_prompt_file_flag(self) -> None:
        args = cli._parse_loop_args(
            ["--extra-selector-prompt-file", "/tmp/sel.md", "cmd"], command_path=["loop"]
        )
        assert args.extra_selector_prompt_file == "/tmp/sel.md"
        assert args.extra_selector_prompt is None

    def test_extra_prompt_and_extra_prompt_file_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_loop_args(
                ["--extra-prompt", "text", "--extra-prompt-file", "file.md", "cmd"],
                command_path=["loop"],
            )

    def test_extra_selector_prompt_and_file_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_loop_args(
                [
                    "--extra-selector-prompt",
                    "text",
                    "--extra-selector-prompt-file",
                    "file.md",
                    "cmd",
                ],
                command_path=["loop"],
            )


class TestParseLoopSelectArgsExtraPromptFlags:
    """Cover --extra-prompt* flags in _parse_loop_select_args."""

    def test_extra_prompt_flag(self) -> None:
        args = cli._parse_loop_select_args(
            ["--extra-prompt", "extra context", "cmd"], command_path=["loop", "select"]
        )
        assert args.extra_prompt == "extra context"
        assert args.extra_prompt_file is None

    def test_extra_prompt_file_flag(self) -> None:
        args = cli._parse_loop_select_args(
            ["--extra-prompt-file", "/tmp/extra.md", "cmd"], command_path=["loop", "select"]
        )
        assert args.extra_prompt_file == "/tmp/extra.md"

    def test_extra_selector_prompt_flag(self) -> None:
        args = cli._parse_loop_select_args(
            ["--extra-selector-prompt", "extra sel", "cmd"], command_path=["loop", "select"]
        )
        assert args.extra_selector_prompt == "extra sel"

    def test_extra_selector_prompt_file_flag(self) -> None:
        args = cli._parse_loop_select_args(
            ["--extra-selector-prompt-file", "/tmp/sel.md", "cmd"],
            command_path=["loop", "select"],
        )
        assert args.extra_selector_prompt_file == "/tmp/sel.md"



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


class TestParseLoopSelectArgsExtraPromptMutualExclusion:
    def test_extra_prompt_and_extra_prompt_file_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_loop_select_args(
                ["--extra-prompt", "text", "--extra-prompt-file", "file.md", "cmd"],
                command_path=["loop", "select"],
            )

    def test_extra_selector_prompt_and_file_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_loop_select_args(
                [
                    "--extra-selector-prompt",
                    "text",
                    "--extra-selector-prompt-file",
                    "file.md",
                    "cmd",
                ],
                command_path=["loop", "select"],
            )
