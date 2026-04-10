"""Integration tests for the agm CLI.

These tests verify the full Python pipeline — argument parsing, dispatch
routing, and argument construction — without executing the underlying
shell scripts.  ``commands._run`` is mocked so that tests depend only on
the Python CLI code, not on any scripts from this repository.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import pytest

from agm.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RunCalled(SystemExit):
    """Raised by the ``_run`` mock to capture the dispatched script call."""

    def __init__(self, script: str, script_args: list[str]) -> None:
        super().__init__(0)
        self.script = script
        self.script_args = list(script_args)


def _mock_run(script: str, args: list[str]) -> None:
    raise _RunCalled(script, args)


def dispatch(argv: list[str]) -> tuple[str, list[str]]:
    """Run the CLI with *argv* and return ``(script, args)`` that would
    have been passed to the underlying shell script."""
    with patch("agm.commands._run", _mock_run):
        try:
            main(argv)
        except _RunCalled as e:
            return e.script, e.script_args
    raise AssertionError("_run was not called")


def run_main(argv: list[str]) -> tuple[int, str, str]:
    """Run the CLI expecting a help or error exit (no script dispatch).

    Returns ``(exit_code, stdout, stderr)``.
    """
    out = io.StringIO()
    err = io.StringIO()
    code = 0
    with redirect_stdout(out), redirect_stderr(err):
        try:
            main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
    return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# Help system
# ---------------------------------------------------------------------------


class TestHelp:
    """agm help: overview and per-command help."""

    def test_overview_lists_all_commands(self) -> None:
        code, out, _ = run_main(["help"])
        assert code == 0
        assert "agm - Agent Management Framework" in out
        assert "Commands:" in out
        for cmd in ("open", "new", "checkout", "init", "fetch",
                     "branch", "config", "worktree", "dep", "run",
                     "tmux", "help"):
            assert cmd in out, f"'{cmd}' missing from overview"

    def test_help_for_each_canonical_command(self) -> None:
        """Every canonical command has a detailed help entry."""
        for cmd in ("open", "new", "checkout", "init", "fetch",
                     "branch", "config", "worktree", "dep", "run",
                     "tmux", "help"):
            code, out, _ = run_main(["help", cmd])
            assert code == 0, f"help {cmd} failed"
            assert f"agm {cmd}" in out, f"help {cmd} missing header"

    def test_help_aliases_resolve(self) -> None:
        """Aliases (br, wt, co) show help for the canonical command."""
        alias_map = {"br": "branch", "wt": "worktree", "co": "checkout"}
        for alias, canonical in alias_map.items():
            code, out, _ = run_main(["help", alias])
            assert code == 0
            assert f"agm {canonical}" in out

    def test_help_unknown_command(self) -> None:
        code, _, err = run_main(["help", "bogus"])
        assert code == 1
        assert "unknown command" in err
        assert "bogus" in err


# ---------------------------------------------------------------------------
# Dispatch: agm br sync → brsync.sh
# ---------------------------------------------------------------------------


class TestBranchSync:
    def test_br_sync(self) -> None:
        script, args = dispatch(["br", "sync"])
        assert script == "brsync.sh"
        assert args == []

    def test_branch_sync_alias(self) -> None:
        script, args = dispatch(["branch", "sync"])
        assert script == "brsync.sh"
        assert args == []


# ---------------------------------------------------------------------------
# Dispatch: agm config cp → cpconfig.sh
# ---------------------------------------------------------------------------


class TestConfigCopy:
    def test_basic(self) -> None:
        script, args = dispatch(["config", "cp", "mydir"])
        assert script == "cpconfig.sh"
        assert args == ["mydir"]

    def test_copy_alias(self) -> None:
        script, args = dispatch(["config", "copy", "dest"])
        assert script == "cpconfig.sh"
        assert args == ["dest"]

    def test_with_project_dir(self) -> None:
        script, args = dispatch(["config", "cp", "-d", "/proj", "dest"])
        assert script == "cpconfig.sh"
        assert args == ["-d", "/proj", "dest"]


# ---------------------------------------------------------------------------
# Dispatch: agm wt co → mkwt.sh
# ---------------------------------------------------------------------------


class TestWorktreeCheckout:
    def test_checkout_branch(self) -> None:
        script, args = dispatch(["wt", "co", "feat/x"])
        assert script == "mkwt.sh"
        assert args == ["feat/x"]

    def test_worktree_checkout_alias(self) -> None:
        script, args = dispatch(["worktree", "checkout", "feat/x"])
        assert script == "mkwt.sh"
        assert args == ["feat/x"]

    def test_checkout_with_new_branch(self) -> None:
        script, args = dispatch(["wt", "co", "-b", "new-br"])
        assert script == "mkwt.sh"
        assert args == ["-b", "new-br"]

    def test_checkout_with_dir(self) -> None:
        script, args = dispatch(["wt", "co", "-d", "/wts", "br"])
        assert script == "mkwt.sh"
        assert args == ["-d", "/wts", "br"]

    def test_checkout_with_b_and_d(self) -> None:
        script, args = dispatch(["wt", "co", "-b", "new-br", "-d", "/wts"])
        assert script == "mkwt.sh"
        assert args == ["-b", "new-br", "-d", "/wts"]


# ---------------------------------------------------------------------------
# Dispatch: agm wt new → mkwt.sh -b
# ---------------------------------------------------------------------------


class TestWorktreeNew:
    def test_new(self) -> None:
        script, args = dispatch(["wt", "new", "feat/y"])
        assert script == "mkwt.sh"
        assert args == ["-b", "feat/y"]

    def test_new_with_dir(self) -> None:
        script, args = dispatch(["wt", "new", "-d", "/custom", "feat/z"])
        assert script == "mkwt.sh"
        assert args == ["-b", "feat/z", "-d", "/custom"]

    def test_worktree_new_alias(self) -> None:
        script, args = dispatch(["worktree", "new", "feat/w"])
        assert script == "mkwt.sh"
        assert args == ["-b", "feat/w"]


# ---------------------------------------------------------------------------
# Dispatch: agm wt rm → rmwt.sh
# ---------------------------------------------------------------------------


class TestWorktreeRemove:
    def test_remove(self) -> None:
        script, args = dispatch(["wt", "rm", "old-br"])
        assert script == "rmwt.sh"
        assert args == ["old-br"]

    def test_remove_force(self) -> None:
        script, args = dispatch(["worktree", "remove", "-f", "old-br"])
        assert script == "rmwt.sh"
        assert args == ["-f", "old-br"]

    def test_rm_without_force(self) -> None:
        script, args = dispatch(["wt", "rm", "br"])
        assert script == "rmwt.sh"
        assert "-f" not in args


# ---------------------------------------------------------------------------
# Dispatch: agm dep → pm-dep.sh
# ---------------------------------------------------------------------------


class TestDep:
    def test_dep_new(self) -> None:
        script, args = dispatch(["dep", "new", "https://example.com/repo.git"])
        assert script == "pm-dep.sh"
        assert args == ["new", "https://example.com/repo.git"]

    def test_dep_new_with_branch(self) -> None:
        script, args = dispatch(
            ["dep", "new", "-b", "v2", "https://example.com/repo.git"],
        )
        assert script == "pm-dep.sh"
        assert args == ["new", "-b", "v2", "https://example.com/repo.git"]

    def test_dep_switch(self) -> None:
        script, args = dispatch(["dep", "switch", "mylib", "feat/x"])
        assert script == "pm-dep.sh"
        assert args == ["switch", "mylib", "feat/x"]

    def test_dep_switch_create_branch(self) -> None:
        script, args = dispatch(["dep", "switch", "-b", "mylib", "feat/x"])
        assert script == "pm-dep.sh"
        assert args == ["switch", "mylib", "-b", "feat/x"]


# ---------------------------------------------------------------------------
# Dispatch: agm fetch → pm-fetch.sh
# ---------------------------------------------------------------------------


class TestFetch:
    def test_fetch(self) -> None:
        script, args = dispatch(["fetch"])
        assert script == "pm-fetch.sh"
        assert args == []


# ---------------------------------------------------------------------------
# Dispatch: agm init → pm-init.sh
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_url_only(self) -> None:
        script, args = dispatch(["init", "https://example.com/repo.git"])
        assert script == "pm-init.sh"
        assert args == ["https://example.com/repo.git"]

    def test_init_name_and_url(self) -> None:
        script, args = dispatch(
            ["init", "myproj", "https://example.com/repo.git"],
        )
        assert script == "pm-init.sh"
        assert args == ["myproj", "https://example.com/repo.git"]

    def test_init_with_branch(self) -> None:
        script, args = dispatch(["init", "-b", "dev", "myproj"])
        assert script == "pm-init.sh"
        assert args == ["-b", "dev", "myproj"]


# ---------------------------------------------------------------------------
# Dispatch: agm open → pm.sh open
# ---------------------------------------------------------------------------


class TestOpen:
    def test_open_bare(self) -> None:
        script, args = dispatch(["open"])
        assert script == "pm.sh"
        assert args == ["open"]

    def test_open_branch(self) -> None:
        script, args = dispatch(["open", "feat/x"])
        assert script == "pm.sh"
        assert args == ["open", "feat/x"]

    def test_open_with_pane_count(self) -> None:
        script, args = dispatch(["open", "-n", "6"])
        assert script == "pm.sh"
        assert args == ["open", "-n", "6"]

    def test_open_with_all_options(self) -> None:
        script, args = dispatch(["open", "-n", "4", "main"])
        assert script == "pm.sh"
        assert args == ["open", "-n", "4", "main"]


# ---------------------------------------------------------------------------
# Dispatch: agm new → pm.sh new
# ---------------------------------------------------------------------------


class TestNew:
    def test_new_branch(self) -> None:
        script, args = dispatch(["new", "feat/y"])
        assert script == "pm.sh"
        assert args == ["new", "feat/y"]

    def test_new_with_parent(self) -> None:
        script, args = dispatch(["new", "-p", "main", "feat/y"])
        assert script == "pm.sh"
        assert args == ["new", "-p", "main", "feat/y"]

    def test_new_with_all_options(self) -> None:
        script, args = dispatch(["new", "-n", "2", "-p", "main", "feat/y"])
        assert script == "pm.sh"
        assert args == ["new", "-n", "2", "-p", "main", "feat/y"]


# ---------------------------------------------------------------------------
# Dispatch: agm co / checkout → pm.sh co
# ---------------------------------------------------------------------------


class TestCheckout:
    def test_co(self) -> None:
        script, args = dispatch(["co", "feat/z"])
        assert script == "pm.sh"
        assert args == ["co", "feat/z"]

    def test_checkout_long_form(self) -> None:
        script, args = dispatch(["checkout", "feat/z"])
        assert script == "pm.sh"
        assert args == ["co", "feat/z"]

    def test_co_with_all_options(self) -> None:
        script, args = dispatch(["co", "-n", "4", "-p", "dev", "feat/z"])
        assert script == "pm.sh"
        assert args == ["co", "-n", "4", "-p", "dev", "feat/z"]

    def test_co_with_parent_only(self) -> None:
        script, args = dispatch(["co", "-p", "main", "feat/a"])
        assert script == "pm.sh"
        assert args == ["co", "-p", "main", "feat/a"]


# ---------------------------------------------------------------------------
# Dispatch: agm run → sandbox.sh
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_simple(self) -> None:
        script, args = dispatch(["run", "npm", "test"])
        assert script == "sandbox.sh"
        assert args == ["npm", "test"]

    def test_run_with_settings_file(self) -> None:
        script, args = dispatch(["run", "-f", "ci.json", "make"])
        assert script == "sandbox.sh"
        assert args == ["-f", "ci.json", "make"]

    def test_run_no_patch(self) -> None:
        script, args = dispatch(["run", "--no-patch", "echo", "hi"])
        assert script == "sandbox.sh"
        assert args == ["--no-patch", "echo", "hi"]

    def test_run_all_options(self) -> None:
        script, args = dispatch(
            ["run", "--no-patch", "-f", "s.json", "cmd", "arg"],
        )
        assert script == "sandbox.sh"
        assert args == ["--no-patch", "-f", "s.json", "cmd", "arg"]

    def test_run_no_command(self) -> None:
        """run with no command still dispatches — sandbox.sh handles the error."""
        script, args = dispatch(["run"])
        assert script == "sandbox.sh"
        assert args == []


# ---------------------------------------------------------------------------
# Dispatch: agm tmux new → tmux.sh
# ---------------------------------------------------------------------------


class TestTmuxNew:
    def test_bare(self) -> None:
        script, args = dispatch(["tmux", "new"])
        assert script == "tmux.sh"
        assert args == []

    def test_detach_short(self) -> None:
        script, args = dispatch(["tmux", "new", "-d"])
        assert script == "tmux.sh"
        assert args == ["-d"]

    def test_detach_long(self) -> None:
        """--detach is translated to -d for the script."""
        script, args = dispatch(["tmux", "new", "--detach"])
        assert script == "tmux.sh"
        assert args == ["-d"]

    def test_with_all_options(self) -> None:
        script, args = dispatch(["tmux", "new", "-d", "-n", "8", "mysession"])
        assert script == "tmux.sh"
        assert args == ["-d", "-n", "8", "mysession"]

    def test_session_name_only(self) -> None:
        script, args = dispatch(["tmux", "new", "my-session"])
        assert script == "tmux.sh"
        assert args == ["my-session"]


# ---------------------------------------------------------------------------
# Dispatch: agm tmux layout → tmux-apply-layout.sh
# ---------------------------------------------------------------------------


class TestTmuxLayout:
    def test_layout(self) -> None:
        script, args = dispatch(["tmux", "layout", "4", "@1", "200", "50"])
        assert script == "tmux-apply-layout.sh"
        assert args == ["4", "@1", "200", "50"]


# ---------------------------------------------------------------------------
# Error handling (Python-level, no scripts involved)
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Errors caught by argparse before any script is invoked."""

    def test_no_command(self) -> None:
        code, _, _ = run_main([])
        assert code == 2

    def test_unknown_command(self) -> None:
        code, _, _ = run_main(["bogus"])
        assert code == 2

    def test_missing_required_args(self) -> None:
        cases = [
            ["wt", "new"],        # missing branch
            ["wt", "rm"],         # missing branch
            ["config", "cp"],     # missing dirname
            ["dep"],              # missing subcommand
            ["br"],               # missing subcommand
            ["init"],             # missing positional
            ["new"],              # missing branch
            ["co"],               # missing branch
            ["checkout"],         # missing branch
            ["tmux", "layout"],   # missing all 4 args
        ]
        for argv in cases:
            code, _, _ = run_main(argv)
            assert code == 2, f"{argv} should fail with exit code 2"

    def test_unrecognized_option(self) -> None:
        code, _, err = run_main(["open", "--bogus"])
        assert code == 2
        assert "unrecognized" in err.lower()

    def test_extra_positional_on_fetch(self) -> None:
        code, _, _ = run_main(["fetch", "extra"])
        assert code == 2
