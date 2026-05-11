"""Additional tests targeting remaining coverage gaps.

Covers:
- cli.py lines 330, 332-335, 346-349, 376-377: _parse_loop_next_args branches
- cli.py lines 563-565: config_copy command via CLI
- cli.py lines 1017, 1021: main() and __name__ entry point
- process.py line 183: raise queue.Empty when remaining <= 0
- process.py line 207: continue when decoded text is empty
- process.py lines 220-224: final decoder flush with capture_output and callback
- layout.py line 98: try: block for git_common_dir in current_project_dir
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from typer.main import get_command

import agm.cli as cli
import agm.project.layout as layout_module
from agm.core.process import run_subprocess

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def invoke(runner: CliRunner, argv: list[str]) -> Any:
    return runner.invoke(get_command(cli.app), argv, prog_name="agm")


# ---------------------------------------------------------------------------
# cli.py – _parse_loop_next_args missing branches
# ---------------------------------------------------------------------------


class TestParseLoopNextArgsMissingBranches:
    """Cover lines 330, 332-335, 346-349, 376-377 in _parse_loop_next_args."""

    def test_double_dash_stops_parsing(self) -> None:
        """Line 330: 'break' after '--' separator."""
        args = cli._parse_loop_next_args(
            ["--", "--runner", "val", "cmd"],
            command_path=["loop", "next"],
        )
        # After --, the -- itself becomes command_name and the rest become runner_args
        assert args.runner is None
        assert args.command_name == "--"
        assert args.runner_args == ["--runner", "val", "cmd"]

    def test_runner_flag(self) -> None:
        """Lines 332-335: --runner parsing in _parse_loop_next_args."""
        args = cli._parse_loop_next_args(
            ["--runner", "my-runner", "cmd"],
            command_path=["loop", "next"],
        )
        assert args.runner == "my-runner"
        assert args.command_name == "cmd"

    def test_tasks_dir_flag(self) -> None:
        """Lines 346-349: --tasks-dir parsing in _parse_loop_next_args."""
        args = cli._parse_loop_next_args(
            ["--tasks-dir", "custom/tasks", "cmd"],
            command_path=["loop", "next"],
        )
        assert args.tasks_dir == "custom/tasks"
        assert args.command_name == "cmd"

    def test_timeout_invalid_format_exits(self) -> None:
        """Lines 376-377: --timeout with invalid format in _parse_loop_next_args."""
        with pytest.raises(SystemExit):
            cli._parse_loop_next_args(
                ["--timeout", "abc", "cmd"],
                command_path=["loop", "next"],
            )


# ---------------------------------------------------------------------------
# cli.py – config_copy command (lines 563-565)
# ---------------------------------------------------------------------------


class TestConfigCopyCommand:
    """Cover lines 563-565: the 'config copy' CLI command function."""

    @pytest.fixture()
    def runner(self) -> CliRunner:
        return CliRunner()

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


# ---------------------------------------------------------------------------
# cli.py – main() and __name__ entry point (lines 1017, 1021)
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    """Cover lines 1017, 1021: app() in main() and __name__ block."""

    def test_main_calls_app_and_shows_help(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Line 1017: main() calls app() which shows the overview help."""
        # Monkeypatch sys.argv so app() sees ["agm", "--help"] and exits
        monkeypatch.setattr("sys.argv", ["agm", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        # --help exits with code 0
        assert exc_info.value.code == 0

    def test_main_module_entry_point(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Line 1021: the `if __name__ == '__main__': main()` block."""
        import runpy
        import warnings

        monkeypatch.setattr("sys.argv", ["agm", "--help"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_module("agm.cli", run_name="__main__")
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# process.py – line 183: raise queue.Empty when remaining <= 0
# ---------------------------------------------------------------------------


class TestIdleTimeoutRemainingZero:
    """Cover line 183: the explicit 'raise queue.Empty' when remaining <= 0."""

    def test_idle_timeout_zero_seconds_triggers_immediately(self) -> None:
        """With idle_timeout=0, the remaining check is <= 0 immediately."""
        with pytest.raises(SystemExit) as exc_info:
            run_subprocess(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                capture_output=True,
                idle_timeout=0,
                isolate_process_group=True,
            )
        assert exc_info.value.code == 124


# ---------------------------------------------------------------------------
# process.py – line 207: continue when decoded text is empty
# ---------------------------------------------------------------------------


class TestEmptyDecodedTextContinue:
    """Cover line 207: 'continue' when decoder.decode returns empty string."""

    def test_multi_byte_char_split_across_chunks(self) -> None:
        """When a UTF-8 multi-byte char is split across pipe chunks,
        the first partial decode returns empty string and is skipped."""
        # 'é' is encoded as b'\xc3\xa9' in UTF-8.
        # If we feed b'\xc3' alone, incremental decoder returns '' (incomplete).
        # Then feeding b'\xa9' completes it.
        # We simulate this by creating a script that writes partial bytes.
        script = (
            'import sys, time\n'
            'sys.stdout.buffer.write(b"\\xc3")\n'
            'sys.stdout.buffer.flush()\n'
            'time.sleep(0.1)\n'
            'sys.stdout.buffer.write(b"\\xa9")\n'
            'sys.stdout.buffer.flush()\n'
        )
        result = run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
        )
        assert result.stdout == "é"
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# process.py – lines 220-224: final decoder flush
# ---------------------------------------------------------------------------


class TestFinalDecoderFlush:
    """Cover lines 220-224: final decoder.decode(b'', final=True) with
    capture_output and callbacks.

    To make the final flush produce non-empty text, we need the decoder to
    have buffered incomplete bytes. This happens when a process writes a
    partial multi-byte UTF-8 sequence (e.g., just b'\xc3') and then exits.
    The incremental decoder buffers b'\xc3' and returns '' in the while loop
    (hitting the 'continue' on line 207). When the pipe closes, the while
    loop exits. Then decoder.decode(b'', final=True) with errors='replace'
    emits a replacement character, which exercises lines 220-224.
    """

    def test_final_flush_captured_with_callback(self) -> None:
        """When capture_output=True and a callback is set, the final flush
        output is both appended to stream_data and sent to the callback."""
        chunks: list[str] = []

        # Write just the first byte of 'é' (b'\xc3'), then exit.
        # The decoder buffers it, and final flush emits '\ufffd'.
        script = (
            'import sys\n'
            'sys.stdout.buffer.write(b"\\xc3")\n'
            'sys.stdout.buffer.flush()\n'
        )
        result = run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
            stdout_callback=lambda text: chunks.append(text),
        )
        # The replacement character from final flush should be in stdout
        assert "\ufffd" in result.stdout
        # Callback should have been called with the replacement character
        assert any("\ufffd" in c for c in chunks)

    def test_final_flush_stderr_callback(self) -> None:
        """Final decoder flush on stderr with capture_output and callback."""
        chunks: list[str] = []

        # Write just the first byte of a multi-byte char on stderr
        script = (
            'import sys\n'
            'sys.stderr.buffer.write(b"\\xc3")\n'
            'sys.stderr.buffer.flush()\n'
        )
        result = run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
            stderr_callback=lambda text: chunks.append(text),
        )
        # The replacement character from final flush should be in stderr
        assert "\ufffd" in result.stderr
        assert any("\ufffd" in c for c in chunks)

    def test_final_flush_with_no_pending_data(self) -> None:
        """When decoder has no buffered data at flush, no extra text is emitted."""
        # Simple ASCII output — decoder won't have buffered partial bytes.
        # Final flush returns '' so lines 222-224 won't be hit, but the
        # for loop itself (lines 220-221) will still execute.
        script = (
            "import sys\n"
            "sys.stdout.write('hello\\n')\n"
            "sys.stdout.flush()\n"
        )
        result = run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
            stdout_callback=lambda text: None,
        )
        assert result.stdout == "hello\n"


# ---------------------------------------------------------------------------
# layout.py – line 98: try: block for git_common_dir in current_project_dir
# ---------------------------------------------------------------------------


class TestCurrentProjectDirGitCommonDirTry:
    """Cover line 98: the try: block for git_helpers.git_common_dir(current).

    The scenario requires:
    1. cwd is a git repo (is_git_repo returns True)
    2. checkout_root succeeds but doesn't yield a project marker
    3. git_common_dir is called
    4. The common_dir's parent has a project marker
    """

    def test_git_common_dir_search_after_checkout_root_finds_no_project(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When checkout_root returns a dir without project markers,
        git_common_dir is used to search further."""
        from agm.project.layout import current_project_dir

        project = tmp_path / "proj"
        repo = project / "repo"
        repo.mkdir(parents=True)
        (project / "worktrees").mkdir()

        cwd = tmp_path / "cwd"
        cwd.mkdir()

        checkout = tmp_path / "checkout"
        checkout.mkdir()

        git_dir = repo / ".git"

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None: checkout,
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "git_common_dir",
            lambda cwd=None: git_dir,
        )

        assert current_project_dir(cwd) == project

    def test_git_common_dir_finds_workspace_project(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When checkout_root finds nothing, git_common_dir locates a workspace project."""
        from agm.project.layout import current_project_dir

        project = tmp_path / "proj"
        repo = project / "repo"
        repo.mkdir(parents=True)
        (project / "worktrees").mkdir()  # workspace project marker

        cwd = tmp_path / "cwd"
        cwd.mkdir()

        checkout = tmp_path / "checkout"
        checkout.mkdir()

        git_dir = repo / ".git"

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None: checkout,
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "git_common_dir",
            lambda cwd=None: git_dir,
        )

        result = current_project_dir(cwd)
        # git_common_dir returns repo/.git → parent = repo/
        # _project_dir_from_checkout(repo):
        #   repo.name == "repo" ✓
        #   (repo.parent / "worktrees").is_dir() ✓ → returns repo.parent = project
        assert result == project

    def test_git_common_dir_finds_embedded_project_via_parent_walk(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When common_dir.parent doesn't have markers directly but
        a grandparent has .agm, the parent walk finds it."""
        from agm.project.layout import current_project_dir

        project = tmp_path / "proj"
        project.mkdir()
        (project / ".agm").mkdir()  # embedded project marker

        # The .git is deep inside project
        git_subdir = project / "sub" / ".git"
        git_subdir.mkdir(parents=True)

        cwd = tmp_path / "cwd"
        cwd.mkdir()

        checkout = tmp_path / "checkout"
        checkout.mkdir()

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None: checkout,
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "git_common_dir",
            lambda cwd=None: git_subdir,
        )

        result = current_project_dir(cwd)
        # common_dir = git_subdir = project/sub/.git
        # common_checkout = common_dir.parent = project/sub
        # Walk: project/sub → no .agm, no repo/ → None
        # Walk: project → .agm exists → returns project
        assert result == project


# ---------------------------------------------------------------------------
# cli.py – list command via CLI (lines 1032-1034)
# ---------------------------------------------------------------------------


class TestListCommandViaCli:
    """Cover lines 1032-1034: list_cmd function body."""

    @pytest.fixture()
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_list_cmd_via_cli(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[object] = []

        def record() -> None:
            calls.append(True)

        monkeypatch.setattr(cli.list_command, "run", record)
        result = invoke(runner, ["list"])
        assert result.exit_code == 0
        assert len(calls) == 1
