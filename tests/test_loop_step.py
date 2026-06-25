"""Comprehensive tests for agm.commands.loop.step, loop.select, and loop.run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agm.agent.loop import PreparedSelectInvocation
from agm.agent.loop import dry_run_prompt_text as next_dry_run_prompt_text
from agm.agent.runner import ResolvedPrompt
from agm.cli_support.args import LoopArgs, LoopSelectArgs
from agm.commands.loop.run import run as loop_run
from agm.commands.loop.select import _print_dry_run_prompt as next_print_dry_run_prompt
from agm.commands.loop.select import run as next_run
from agm.commands.loop.step import (
    LoopStepRuntime,
    PreparedPrompt,
    _prepare_prompt,
    _print_dry_run_command,
    _print_dry_run_prompt,
    _write_stream,
    cleanup_runtime,
    execute_single_step,
    prepare_runtime,
    print_startup,
    run,
)
from agm.core.log import (
    append_log,
    prepare_log_file,
    resolve_log_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loop_args(
    *,
    no_log: bool = True,
    log_file: str | None = None,
    runner: str | None = "myrunner",
    runner_args: list[str] | None = None,
    selector: str | None = None,
    no_selector: bool = True,
    tasks_dir: str | None = None,
    prompt: str | None = None,
    prompt_file: str | None = None,
    selector_prompt: str | None = None,
    selector_prompt_file: str | None = None,
    extra_prompt: str | None = None,
    extra_prompt_file: str | None = None,
    extra_selector_prompt: str | None = None,
    extra_selector_prompt_file: str | None = None,
    command_name: str | None = None,
    timeout: float | None = None,
) -> LoopArgs:
    return LoopArgs(
        command_name=command_name,
        runner=runner,
        runner_args=runner_args if runner_args is not None else [],
        selector=selector,
        no_selector=no_selector,
        tasks_dir=tasks_dir,
        no_log=no_log,
        log_file=log_file,
        prompt=prompt,
        prompt_file=prompt_file,
        selector_prompt=selector_prompt,
        selector_prompt_file=selector_prompt_file,
        extra_prompt=extra_prompt,
        extra_prompt_file=extra_prompt_file,
        extra_selector_prompt=extra_selector_prompt,
        extra_selector_prompt_file=extra_selector_prompt_file,
        timeout=timeout,
    )


def _make_loop_select_args(
    *,
    runner: str | None = "myrunner",
    runner_args: list[str] | None = None,
    selector: str | None = None,
    no_selector: bool = False,
    tasks_dir: str | None = None,
    prompt: str | None = None,
    prompt_file: str | None = None,
    selector_prompt: str | None = None,
    selector_prompt_file: str | None = None,
    extra_prompt: str | None = None,
    extra_prompt_file: str | None = None,
    extra_selector_prompt: str | None = None,
    extra_selector_prompt_file: str | None = None,
    command_name: str | None = None,
    timeout: float | None = None,
) -> LoopSelectArgs:
    return LoopSelectArgs(
        command_name=command_name,
        runner=runner,
        runner_args=runner_args if runner_args is not None else [],
        selector=selector,
        no_selector=no_selector,
        tasks_dir=tasks_dir,
        prompt=prompt,
        prompt_file=prompt_file,
        selector_prompt=selector_prompt,
        selector_prompt_file=selector_prompt_file,
        extra_prompt=extra_prompt,
        extra_prompt_file=extra_prompt_file,
        extra_selector_prompt=extra_selector_prompt,
        extra_selector_prompt_file=extra_selector_prompt_file,
        timeout=timeout,
    )


def _make_runtime(
    tmp_path: Path,
    *,
    select_invocation: PreparedSelectInvocation | None = None,
    implement_prompt_file: Path | None = None,
    loop_prompt: PreparedPrompt | None = None,
    resolved_prompt: ResolvedPrompt | None = None,
    bootstrap_prompt: PreparedPrompt | None = None,
    log_file: Path | None = None,
    runner_command: list[str] | None = None,
) -> LoopStepRuntime:
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    progress = tasks_dir / "PROGRESS.md"
    if loop_prompt is None and select_invocation is None:
        prompt_file = tmp_path / "loop.md"
        prompt_file.write_text("do stuff\n", encoding="utf-8")
        loop_prompt = PreparedPrompt(
            label="loop", source_file=prompt_file, effective_file=prompt_file
        )
    return LoopStepRuntime(
        temp_files=[],
        resolved_tasks_dir=tasks_dir,
        resolved_progress_file=progress,
        env={"TASKS_DIR": str(tasks_dir)},
        resolved_runner_command=runner_command if runner_command is not None else ["myrunner"],
        select_invocation=select_invocation,
        implement_prompt_file=implement_prompt_file,
        loop_prompt=loop_prompt,
        resolved_prompt=resolved_prompt,
        bootstrap_prompt=bootstrap_prompt,
        extra_prompt_source=None,
        log_file=log_file,
        idle_timeout=None,
    )


# ===========================================================================
# resolve_log_file
# ===========================================================================


class TestLogFile:
    def test_returns_none_when_no_log(self, tmp_path: Path) -> None:
        args = _make_loop_args(no_log=True, log_file=None)
        assert (
            resolve_log_file(
                command_name="loop",
                enabled=not args.no_log,
                log_file=args.log_file,
            )
            is None
        )

    def test_returns_explicit_log_file_when_given(self, tmp_path: Path) -> None:
        explicit = str(tmp_path / "my.log")
        args = _make_loop_args(no_log=False, log_file=explicit)
        result = resolve_log_file(
            command_name="loop",
            enabled=not args.no_log,
            log_file=args.log_file,
        )
        assert result == Path(explicit)

    def test_generates_timestamped_log_file_in_agent_files_when_no_log_file_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agm.core.log.git_helpers.containing_root", lambda _path: None)
        args = _make_loop_args(no_log=False, log_file=None)
        result = resolve_log_file(
            command_name="loop",
            enabled=not args.no_log,
            log_file=args.log_file,
        )
        assert result is not None
        assert result.parent == tmp_path / ".agent-files"
        assert result.name.startswith("loop-")
        assert result.suffix == ".log"

    def test_generates_timestamped_log_file_in_checkout_agent_files_when_under_git(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkout = tmp_path / "checkout"
        nested = checkout / "nested"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)
        monkeypatch.setattr("agm.core.log.git_helpers.containing_root", lambda _path: checkout)
        args = _make_loop_args(no_log=False, log_file=None)
        result = resolve_log_file(
            command_name="refine",
            enabled=not args.no_log,
            log_file=args.log_file,
        )
        assert result is not None
        assert result.parent == checkout / ".agent-files"
        assert result.name.startswith("refine-")
        assert result.suffix == ".log"

    def test_relative_log_file_resolves_against_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        args = _make_loop_args(no_log=False, log_file="logs/run.log")
        result = resolve_log_file(
            command_name="loop",
            enabled=not args.no_log,
            log_file=args.log_file,
        )
        assert result == work / "logs" / "run.log"

    def test_no_log_overrides_explicit_log_file(self, tmp_path: Path) -> None:
        args = _make_loop_args(no_log=True, log_file=str(tmp_path / "ignored.log"))
        assert (
            resolve_log_file(
                command_name="loop",
                enabled=not args.no_log,
                log_file=args.log_file,
            )
            is None
        )

    def test_prepare_log_file_prints_full_default_log_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_file = tmp_path / ".agent-files" / "loop-20260513-120000.log"

        prepare_log_file(log_file)

        out = capsys.readouterr().out
        assert out == f"Logging to {log_file}\n"
        assert log_file.parent.is_dir()


# ===========================================================================
# append_log
# ===========================================================================


class TestAppendLog:
    def test_no_op_when_log_file_is_none(self, tmp_path: Path) -> None:
        # Should not raise; no file should be written
        append_log(None, "some content")

    def test_no_op_when_content_is_empty(self, tmp_path: Path) -> None:
        log = tmp_path / "out.log"
        append_log(log, "")
        assert not log.exists()

    def test_appends_content_to_file(self, tmp_path: Path) -> None:
        log = tmp_path / "out.log"
        append_log(log, "first line\n")
        append_log(log, "second line\n")
        assert log.read_text(encoding="utf-8") == "first line\nsecond line\n"

    def test_creates_file_if_missing(self, tmp_path: Path) -> None:
        log = tmp_path / "new.log"
        append_log(log, "hello")
        assert log.read_text(encoding="utf-8") == "hello"


# ===========================================================================
# _write_stream
# ===========================================================================


class TestWriteStream:
    def test_no_op_when_chunk_is_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        _write_stream("")
        out, err = capsys.readouterr()
        assert out == ""
        assert err == ""

    def test_writes_to_stdout_by_default(self, capsys: pytest.CaptureFixture[str]) -> None:
        _write_stream("hello stdout")
        out, _ = capsys.readouterr()
        assert out == "hello stdout"

    def test_writes_to_stderr_when_flag_set(self, capsys: pytest.CaptureFixture[str]) -> None:
        _write_stream("hello stderr", stderr=True)
        _, err = capsys.readouterr()
        assert err == "hello stderr"

    def test_empty_string_does_not_write_to_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_stream("", stderr=True)
        _, err = capsys.readouterr()
        assert err == ""


# ===========================================================================
# _print_dry_run_command (step)
# ===========================================================================


class TestPrintDryRunCommand:
    def test_delegates_to_dry_run_print_labeled_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_calls: list[tuple[str, list[str]]] = []

        def fake_print_labeled_command(label: str, command: list[str]) -> None:
            captured_calls.append((label, command))

        monkeypatch.setattr(
            "agm.commands.loop.step.dry_run.print_labeled_command",
            fake_print_labeled_command,
        )
        _print_dry_run_command("runner", ["myrunner", "--verbose"])
        assert captured_calls == [("runner", ["myrunner", "--verbose"])]


# ===========================================================================
# _print_dry_run_prompt (step)
# ===========================================================================


class TestPrintDryRunPrompt:
    def test_prints_formatted_prompt_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _print_dry_run_prompt("loop", "/path/to/loop.md")
        out, _ = capsys.readouterr()
        assert out.strip() == "dry-run: prompt [loop]: /path/to/loop.md"

    def test_includes_label_in_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _print_dry_run_prompt("bootstrap", "some text")
        out, _ = capsys.readouterr()
        assert "[bootstrap]" in out
        assert "some text" in out


# ===========================================================================
# _prepare_prompt (step)
# ===========================================================================


class TestPreparePrompt:
    def test_returns_prepared_prompt_with_label_and_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / "loop.md"
        effective = tmp_path / "loop.md.tmp"
        monkeypatch.setattr(
            "agm.commands.loop.step.preprocess_prompt_file",
            lambda path, temp_files, env: effective,
        )
        temp_files: list[Path] = []
        result = _prepare_prompt("loop", source, temp_files=temp_files, env={})
        assert result.label == "loop"
        assert result.source_file == source
        assert result.effective_file == effective

    def test_passes_temp_files_and_env_to_preprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / "loop.md"
        effective = tmp_path / "loop.md.tmp"
        captured: dict[str, object] = {}

        def fake_preprocess(
            path: Path, *, temp_files: list[Path], env: dict[str, str]
        ) -> Path:
            captured["path"] = path
            captured["temp_files"] = temp_files
            captured["env"] = env
            return effective

        monkeypatch.setattr(
            "agm.commands.loop.step.preprocess_prompt_file", fake_preprocess
        )
        env = {"MY_VAR": "value"}
        _prepare_prompt("bootstrap", source, temp_files=[], env=env)
        assert captured["path"] == source
        assert captured["env"] == env

    def test_effective_equals_source_when_no_preprocessing_needed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / "loop.md"
        monkeypatch.setattr(
            "agm.commands.loop.step.preprocess_prompt_file",
            lambda path, temp_files, env: path,
        )
        result = _prepare_prompt("loop", source, temp_files=[], env={})
        assert result.source_file == result.effective_file == source


# ===========================================================================
# prepare_runtime
# ===========================================================================


class TestPrepareRuntime:
    def _setup_home_with_prompts(
        self, tmp_path: Path, prompts: list[str] | None = None
    ) -> Path:
        if prompts is None:
            prompts = ["loop.md", "select.md", "implement.md"]
        home = tmp_path / "home"
        prompt_dir = home / ".agm" / "prompts"
        prompt_dir.mkdir(parents=True)
        for name in prompts:
            (prompt_dir / name).write_text(f"# {name}\n", encoding="utf-8")
        return home

    def test_no_selector_mode_with_loop_prompt_and_progress_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = self._setup_home_with_prompts(tmp_path, ["loop.md", "select.md"])
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        # Create a progress file so bootstrap is skipped
        tasks_dir_path = tmp_path / ".agent-files" / "tasks"
        tasks_dir_path.mkdir(parents=True)
        (tasks_dir_path / "PROGRESS.md").write_text("done\n", encoding="utf-8")

        args = _make_loop_args(no_log=True, no_selector=True, runner="fake-runner")
        runtime = prepare_runtime(args)

        assert runtime.select_invocation is None
        assert runtime.loop_prompt is not None
        assert runtime.loop_prompt.label == "loop"
        assert runtime.bootstrap_prompt is None
        assert runtime.log_file is None
        cleanup_runtime(runtime)

    def test_no_selector_mode_creates_bootstrap_when_no_progress_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = self._setup_home_with_prompts(tmp_path, ["loop.md", "select.md"])
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        run_calls: list[list[str]] = []

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            idle_timeout: float | None = None,
        ) -> str:
            run_calls.append(command)
            return ""

        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)

        args = _make_loop_args(no_log=True, no_selector=True, runner="fake-runner")
        runtime = prepare_runtime(args)

        assert runtime.bootstrap_prompt is not None
        assert runtime.bootstrap_prompt.label == "bootstrap"
        # bootstrap runner was invoked
        assert run_calls == [["fake-runner"]]
        cleanup_runtime(runtime)

    def test_prepare_runtime_uses_explicit_prompt_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = self._setup_home_with_prompts(tmp_path, ["loop.md", "select.md"])
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        # Create progress file so no bootstrap
        tasks_dir_path = tmp_path / ".agent-files" / "tasks"
        tasks_dir_path.mkdir(parents=True)
        (tasks_dir_path / "PROGRESS.md").write_text("done\n", encoding="utf-8")

        custom_prompt = tmp_path / "custom.md"
        custom_prompt.write_text("custom instructions\n", encoding="utf-8")

        args = _make_loop_args(
            no_log=True,
            no_selector=True,
            runner="fake-runner",
            prompt_file=str(custom_prompt),
        )
        runtime = prepare_runtime(args)

        assert runtime.resolved_prompt is not None
        assert runtime.loop_prompt is not None
        cleanup_runtime(runtime)

    def test_prepare_runtime_exits_when_loop_prompt_file_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        (home / ".agm" / "prompts").mkdir(parents=True)
        # No loop.md, no select.md
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        args = _make_loop_args(no_log=True, no_selector=True, runner="fake-runner")
        with pytest.raises(SystemExit) as exc_info:
            prepare_runtime(args)
        assert exc_info.value.code == 1

    def test_prepare_runtime_sets_log_file_from_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = self._setup_home_with_prompts(tmp_path, ["loop.md", "select.md"])
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        tasks_dir_path = tmp_path / ".agent-files" / "tasks"
        tasks_dir_path.mkdir(parents=True)
        (tasks_dir_path / "PROGRESS.md").write_text("done\n", encoding="utf-8")

        log_path = str(tmp_path / "test.log")
        args = _make_loop_args(
            no_log=False, log_file=log_path, no_selector=True, runner="fake-runner"
        )
        runtime = prepare_runtime(args)
        assert runtime.log_file == Path(log_path)
        cleanup_runtime(runtime)

    def test_prepare_runtime_in_selector_mode_creates_invocation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = self._setup_home_with_prompts(tmp_path, ["loop.md", "select.md", "implement.md"])
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        args = _make_loop_args(
            no_log=True, no_selector=False, runner="fake-runner", selector="fake-selector"
        )
        runtime = prepare_runtime(args)

        assert runtime.select_invocation is not None
        assert runtime.loop_prompt is None
        assert runtime.implement_prompt_file is not None
        assert runtime.implement_prompt_file.name == "implement.md"
        cleanup_runtime(runtime)

    def test_prepare_runtime_in_selector_mode_no_implement_prompt_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = self._setup_home_with_prompts(tmp_path, ["loop.md", "select.md"])
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        args = _make_loop_args(
            no_log=True, no_selector=False, runner="fake-runner", selector="fake-selector"
        )
        with pytest.raises(SystemExit) as exc_info:
            prepare_runtime(args)
        assert exc_info.value.code == 1
        # Improve coverage for the missing implement.md error path (line 165-166)
        # This also verifies that selector mode requires implement.md when no explicit prompt

    def test_prepare_runtime_selector_mode_no_implement_with_inline_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = self._setup_home_with_prompts(tmp_path, ["loop.md", "select.md"])
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        args = _make_loop_args(
            no_log=True,
            no_selector=False,
            runner="fake-runner",
            selector="fake-selector",
            prompt="implement this task",
        )
        runtime = prepare_runtime(args)
        assert runtime.implement_prompt_file is None
        assert runtime.resolved_prompt is not None
        cleanup_runtime(runtime)

    def test_prepare_runtime_selector_mode_no_implement_with_explicit_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = self._setup_home_with_prompts(tmp_path, ["loop.md", "select.md"])
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        custom_prompt = tmp_path / "custom.md"
        custom_prompt.write_text("custom instructions\n", encoding="utf-8")

        args = _make_loop_args(
            no_log=True,
            no_selector=False,
            runner="fake-runner",
            selector="fake-selector",
            prompt_file=str(custom_prompt),
        )
        runtime = prepare_runtime(args)

        # When explicit prompt is set, implement.md is not required
        assert runtime.select_invocation is not None
        assert runtime.implement_prompt_file is None
        assert runtime.resolved_prompt is not None
        cleanup_runtime(runtime)

    def test_dry_run_skips_bootstrap_prompt_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """dry_run mode skips run_prompt_command for bootstrap.

        When dry_run is enabled, the bootstrap prompt is prepared (so it
        appears in the dry-run output) but run_prompt_command is NOT called.
        """
        home = self._setup_home_with_prompts(tmp_path, ["loop.md", "select.md"])
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        # No progress file → bootstrap prompt path is taken
        run_calls: list[list[str]] = []

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            idle_timeout: float | None = None,
        ) -> str:
            run_calls.append(command)
            return ""

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        monkeypatch.setattr("agm.commands.loop.step.dry_run.enabled", lambda: True)

        args = _make_loop_args(no_log=True, no_selector=True, runner="fake-runner")
        runtime = prepare_runtime(args)

        # Bootstrap prompt should be prepared but not executed
        assert runtime.bootstrap_prompt is not None
        assert run_calls == [], "run_prompt_command must not be called in dry-run mode"
        cleanup_runtime(runtime)


# ===========================================================================
# print_startup
# ===========================================================================


class TestPrintStartup:
    def test_prints_and_logs_resolved_tasks_dir(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_file = tmp_path / "out.log"
        runtime = _make_runtime(tmp_path, log_file=log_file)

        print_startup(runtime)

        out, _ = capsys.readouterr()
        expected = f"Tasks dir: {runtime.resolved_tasks_dir}\n"
        assert out == expected
        assert log_file.read_text(encoding="utf-8") == expected

    def test_displays_tasks_dir_relative_to_current_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.chdir(tmp_path)
        log_file = tmp_path / "out.log"
        runtime = _make_runtime(tmp_path, log_file=log_file)

        print_startup(runtime)

        out, _ = capsys.readouterr()
        expected = f"Tasks dir: {Path('tasks')}\n"
        assert out == expected
        assert log_file.read_text(encoding="utf-8") == expected


# ===========================================================================
# execute_single_step
# ===========================================================================


class TestExecuteSingleStep:
    def test_no_selector_returns_true_when_output_is_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = _make_runtime(tmp_path)
        monkeypatch.setattr(
            "agm.commands.loop.step.run_prompt_command",
            lambda *a, **kw: "COMPLETE",
        )
        result = execute_single_step(runtime, step_number=1)
        assert result is True

    def test_no_selector_returns_false_when_output_is_not_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = _make_runtime(tmp_path)
        monkeypatch.setattr(
            "agm.commands.loop.step.run_prompt_command",
            lambda *a, **kw: "still working",
        )
        result = execute_single_step(runtime, step_number=1)
        assert result is False

    def test_no_selector_passes_callbacks_to_run_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = _make_runtime(tmp_path)
        captured_callbacks: dict[str, object] = {}

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            stdout_callback: object = None,
            stderr_callback: object = None,
            idle_timeout: float | None = None,
        ) -> str:
            captured_callbacks["stdout_callback"] = stdout_callback
            captured_callbacks["stderr_callback"] = stderr_callback
            return "COMPLETE"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        execute_single_step(runtime, step_number=1)
        assert callable(captured_callbacks["stdout_callback"])
        assert callable(captured_callbacks["stderr_callback"])

    def test_no_selector_appends_output_to_log_via_callback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_file = tmp_path / "out.log"
        runtime = _make_runtime(tmp_path, log_file=log_file)

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            stdout_callback: object = None,
            stderr_callback: object = None,
            idle_timeout: float | None = None,
        ) -> str:
            if callable(stdout_callback):
                stdout_callback("stdout chunk\n")
            if callable(stderr_callback):
                stderr_callback("stderr chunk\n")
            return "COMPLETE"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        execute_single_step(runtime, step_number=1)
        log_content = log_file.read_text(encoding="utf-8")
        assert "stdout chunk" in log_content
        assert "stderr chunk" in log_content

    def test_step_header_is_printed_and_logged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_file = tmp_path / "out.log"
        runtime = _make_runtime(tmp_path, log_file=log_file)
        monkeypatch.setattr(
            "agm.commands.loop.step.run_prompt_command",
            lambda *a, **kw: "COMPLETE",
        )
        execute_single_step(runtime, step_number=3)
        out, _ = capsys.readouterr()
        assert "Step 3" in out
        assert "Step 3" in log_file.read_text(encoding="utf-8")

    def test_selector_mode_returns_true_when_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompt_file = tmp_path / "select.md"
        prompt_file.write_text("select a task\n", encoding="utf-8")
        invocation = PreparedSelectInvocation(
            source_prompt_file=prompt_file,
            effective_prompt_file=prompt_file,
            command=["fake-selector"],
            command_kind="selector",
            runner_command=["fake-runner"],
            selector_command=["fake-selector"],
        )
        runtime = _make_runtime(
            tmp_path, select_invocation=invocation, loop_prompt=None,
            implement_prompt_file=tmp_path / "implement.md",
        )

        monkeypatch.setattr(
            "agm.commands.loop.step.run_prompt_command",
            lambda *a, **kw: "COMPLETE",
        )
        monkeypatch.setattr(
            "agm.commands.loop.step.selector_result",
            lambda output, tasks_dir: None,
        )
        result = execute_single_step(runtime, step_number=1)
        assert result is True

    def test_selector_mode_returns_false_and_runs_runner_for_task_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompt_file = tmp_path / "select.md"
        prompt_file.write_text("select a task\n", encoding="utf-8")
        task_file = tmp_path / "tasks" / "task-1.md"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text("do task\n", encoding="utf-8")

        implement_file = tmp_path / "implement.md"
        implement_file.write_text("implement @${TASK_FILE}\n", encoding="utf-8")

        invocation = PreparedSelectInvocation(
            source_prompt_file=prompt_file,
            effective_prompt_file=prompt_file,
            command=["fake-selector"],
            command_kind="selector",
            runner_command=["fake-runner"],
            selector_command=["fake-selector"],
        )
        runtime = _make_runtime(
            tmp_path,
            select_invocation=invocation,
            implement_prompt_file=implement_file,
            loop_prompt=None,
        )

        call_count = 0
        all_targets: list[Path] = []

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            stdout_callback: object = None,
            stderr_callback: object = None,
            idle_timeout: float | None = None,
        ) -> str:
            nonlocal call_count
            call_count += 1
            all_targets.append(target)
            return "task output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        monkeypatch.setattr(
            "agm.commands.loop.step.selector_result",
            lambda output, tasks_dir: task_file,
        )
        result = execute_single_step(runtime, step_number=1)
        assert result is False
        # selector + runner calls = 2
        assert call_count == 2
        # Runner (2nd call) uses preprocessed implement.md, not the raw task file
        assert all_targets[1] != task_file
        expanded_text = all_targets[1].read_text(encoding="utf-8")
        assert "TASK_FILE" in expanded_text or str(task_file) in expanded_text

    def test_selector_mode_implement_prompt_expands_task_file_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompt_file = tmp_path / "select.md"
        prompt_file.write_text("select a task\n", encoding="utf-8")
        task_file = tmp_path / "tasks" / "task-1.md"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text("do task\n", encoding="utf-8")

        implement_file = tmp_path / "implement.md"
        implement_file.write_text(
            "Implement the task at ${TASK_FILE}.\n", encoding="utf-8"
        )

        invocation = PreparedSelectInvocation(
            source_prompt_file=prompt_file,
            effective_prompt_file=prompt_file,
            command=["fake-selector"],
            command_kind="selector",
            runner_command=["fake-runner"],
            selector_command=["fake-selector"],
        )
        runtime = _make_runtime(
            tmp_path,
            select_invocation=invocation,
            implement_prompt_file=implement_file,
            loop_prompt=None,
        )

        all_targets: list[Path] = []
        all_envs: list[dict[str, str]] = []

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            stdout_callback: object = None,
            stderr_callback: object = None,
            idle_timeout: float | None = None,
        ) -> str:
            all_targets.append(target)
            all_envs.append(env)
            return "task output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        monkeypatch.setattr(
            "agm.commands.loop.step.selector_result",
            lambda output, tasks_dir: task_file,
        )
        execute_single_step(runtime, step_number=1)

        # Runner (2nd call) receives preprocessed implement.md with TASK_FILE expanded
        assert len(all_targets) == 2
        runner_target = all_targets[1]
        expanded_content = runner_target.read_text(encoding="utf-8")
        assert str(task_file) in expanded_content
        assert "${TASK_FILE}" not in expanded_content
        # TASK_FILE is in runner env
        assert all_envs[1]["TASK_FILE"] == str(task_file)

    def test_selector_mode_explicit_prompt_re_prepares_with_task_file_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        task_file = tmp_path / "tasks" / "task-1.md"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text("do task\n", encoding="utf-8")

        prompt_file = tmp_path / "select.md"
        prompt_file.write_text("select\n", encoding="utf-8")

        custom_prompt = tmp_path / "custom-prompt.md"
        custom_prompt.write_text("Do $TASK_FILE with $TASKS_DIR\n", encoding="utf-8")

        invocation = PreparedSelectInvocation(
            source_prompt_file=prompt_file,
            effective_prompt_file=prompt_file,
            command=["fake-selector"],
            command_kind="selector",
            runner_command=["fake-runner"],
            selector_command=["fake-selector"],
        )
        resolved_prompt = ResolvedPrompt(source=custom_prompt, effective_file=custom_prompt)
        runtime = _make_runtime(
            tmp_path,
            select_invocation=invocation,
            resolved_prompt=resolved_prompt,
            loop_prompt=None,
        )

        all_targets: list[Path] = []
        all_envs: list[dict[str, str]] = []

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            stdout_callback: object = None,
            stderr_callback: object = None,
            idle_timeout: float | None = None,
        ) -> str:
            all_targets.append(target)
            all_envs.append(env)
            return "output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        monkeypatch.setattr(
            "agm.commands.loop.step.selector_result",
            lambda output, tasks_dir: task_file,
        )
        execute_single_step(runtime, step_number=1)

        # Explicit prompt is re-prepared with TASK_FILE env
        assert len(all_targets) == 2
        runner_content = all_targets[1].read_text(encoding="utf-8")
        assert str(task_file) in runner_content
        assert all_envs[1]["TASK_FILE"] == str(task_file)

    def test_selector_mode_no_implement_no_prompt_uses_raw_task_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no implement.md and no explicit prompt, task file is used."""
        task_file = tmp_path / "tasks" / "task-1.md"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text("do task\n", encoding="utf-8")

        prompt_file = tmp_path / "select.md"
        prompt_file.write_text("select\n", encoding="utf-8")

        invocation = PreparedSelectInvocation(
            source_prompt_file=prompt_file,
            effective_prompt_file=prompt_file,
            command=["fake-selector"],
            command_kind="selector",
            runner_command=["fake-runner"],
            selector_command=["fake-selector"],
        )
        runtime = _make_runtime(
            tmp_path,
            select_invocation=invocation,
            loop_prompt=None,
            # Neither implement_prompt_file nor resolved_prompt set
        )

        all_targets: list[Path] = []
        all_envs: list[dict[str, str]] = []

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            stdout_callback: object = None,
            stderr_callback: object = None,
            idle_timeout: float | None = None,
        ) -> str:
            all_targets.append(target)
            all_envs.append(env)
            return "output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        monkeypatch.setattr(
            "agm.commands.loop.step.selector_result",
            lambda output, tasks_dir: task_file,
        )
        execute_single_step(runtime, step_number=1)

        # Raw task file is used as runner target (fallback)
        assert len(all_targets) == 2
        assert all_targets[1] == task_file
        assert "TASK_FILE" not in all_envs[1]

    def test_selector_mode_retries_until_valid_task_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompt_file = tmp_path / "select.md"
        prompt_file.write_text("select\n", encoding="utf-8")
        task_file = tmp_path / "tasks" / "task-1.md"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text("do task\n", encoding="utf-8")

        implement_file = tmp_path / "implement.md"
        implement_file.write_text("implement @${TASK_FILE}\n", encoding="utf-8")

        invocation = PreparedSelectInvocation(
            source_prompt_file=prompt_file,
            effective_prompt_file=prompt_file,
            command=["fake-selector"],
            command_kind="selector",
            runner_command=["fake-runner"],
            selector_command=["fake-selector"],
        )
        runtime = _make_runtime(
            tmp_path,
            select_invocation=invocation,
            implement_prompt_file=implement_file,
            loop_prompt=None,
        )

        selector_call_count = 0
        selector_results: list[Path | None | str] = ["not a path yet", task_file]

        def fake_selector_result(output: str, *, tasks_dir: Path) -> Path | None | str:
            nonlocal selector_call_count
            val = selector_results[selector_call_count]
            selector_call_count += 1
            return val

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", lambda *a, **kw: "output")
        monkeypatch.setattr("agm.commands.loop.step.selector_result", fake_selector_result)
        execute_single_step(runtime, step_number=1)
        assert selector_call_count == 2


# ===========================================================================
# cleanup_runtime
# ===========================================================================


class TestCleanupRuntime:
    def test_removes_temp_files(self, tmp_path: Path) -> None:
        f1 = tmp_path / "tmp1.md"
        f2 = tmp_path / "tmp2.md"
        f1.write_text("a", encoding="utf-8")
        f2.write_text("b", encoding="utf-8")
        runtime = _make_runtime(tmp_path)
        runtime.temp_files.extend([f1, f2])
        cleanup_runtime(runtime)
        assert not f1.exists()
        assert not f2.exists()

    def test_tolerates_already_deleted_temp_files(self, tmp_path: Path) -> None:
        missing = tmp_path / "gone.md"
        runtime = _make_runtime(tmp_path)
        runtime.temp_files.append(missing)
        cleanup_runtime(runtime)  # should not raise


# ===========================================================================
# run (step entry point)
# ===========================================================================


class TestStepRun:
    def _stub_prepare(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> LoopStepRuntime:
        runtime = _make_runtime(tmp_path)

        def fake_prepare(args: LoopArgs) -> LoopStepRuntime:
            return runtime

        monkeypatch.setattr("agm.commands.loop.step.prepare_runtime", fake_prepare)
        monkeypatch.setattr("agm.commands.loop.step.cleanup_runtime", lambda r: None)
        return runtime

    def test_calls_execute_single_step_and_returns(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        runtime = self._stub_prepare(tmp_path, monkeypatch)
        step_calls: list[int] = []

        def fake_execute(r: LoopStepRuntime, *, step_number: int) -> bool:
            step_calls.append(step_number)
            print(f"Step {step_number}")
            return True

        monkeypatch.setattr("agm.commands.loop.step.execute_single_step", fake_execute)
        monkeypatch.setattr("agm.commands.loop.step.dry_run.enabled", lambda: False)

        args = _make_loop_args()
        run(args)
        out, _ = capsys.readouterr()
        assert step_calls == [1]
        assert out.startswith(f"Tasks dir: {runtime.resolved_tasks_dir}\nStep 1\n")

    def test_dry_run_prints_and_does_not_execute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._stub_prepare(tmp_path, monkeypatch)
        dry_run_called = [False]

        def fake_print_dry_run(r: LoopStepRuntime) -> None:
            dry_run_called[0] = True

        execute_called = [False]

        def fake_execute(r: LoopStepRuntime, *, step_number: int) -> bool:
            execute_called[0] = True
            return True

        monkeypatch.setattr("agm.commands.loop.step.print_dry_run", fake_print_dry_run)
        monkeypatch.setattr("agm.commands.loop.step.execute_single_step", fake_execute)
        monkeypatch.setattr("agm.commands.loop.step.dry_run.enabled", lambda: True)

        args = _make_loop_args()
        run(args)
        assert dry_run_called[0] is True
        assert execute_called[0] is False

    def test_keyboard_interrupt_exits_with_130(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_prepare(tmp_path, monkeypatch)
        monkeypatch.setattr("agm.commands.loop.step.dry_run.enabled", lambda: False)
        monkeypatch.setattr(
            "agm.commands.loop.step.execute_single_step",
            lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt),
        )
        args = _make_loop_args()
        with pytest.raises(SystemExit) as exc_info:
            run(args)
        assert exc_info.value.code == 130

    def test_cleanup_is_called_even_on_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = _make_runtime(tmp_path)
        cleanup_called = [False]

        monkeypatch.setattr(
            "agm.commands.loop.step.prepare_runtime", lambda args: runtime
        )
        monkeypatch.setattr("agm.commands.loop.step.dry_run.enabled", lambda: False)

        def fake_execute(r: LoopStepRuntime, *, step_number: int) -> bool:
            raise RuntimeError("oops")

        monkeypatch.setattr("agm.commands.loop.step.execute_single_step", fake_execute)

        def fake_cleanup(r: LoopStepRuntime) -> None:
            cleanup_called[0] = True

        monkeypatch.setattr("agm.commands.loop.step.cleanup_runtime", fake_cleanup)

        args = _make_loop_args()
        with pytest.raises(RuntimeError):
            run(args)
        assert cleanup_called[0] is True

    def test_cleanup_not_called_when_prepare_runtime_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """finally block when runtime is None after prepare_runtime raises.

        When prepare_runtime raises, runtime stays None and cleanup_runtime
        must NOT be called.
        """
        monkeypatch.setattr(
            "agm.commands.loop.step.prepare_runtime",
            lambda _args: (_ for _ in ()).throw(RuntimeError("setup failed")),
        )
        cleanup_called = [False]
        monkeypatch.setattr(
            "agm.commands.loop.step.cleanup_runtime",
            lambda r: cleanup_called.__setitem__(0, True),
        )

        args = _make_loop_args()
        with pytest.raises(RuntimeError, match="setup failed"):
            run(args)
        assert cleanup_called[0] is False


# ===========================================================================
# loop.select — _dry_run_prompt_text
# ===========================================================================


class TestNextDryRunPromptText:
    def test_same_file_returns_source_path(self, tmp_path: Path) -> None:
        f = tmp_path / "select.md"
        assert next_dry_run_prompt_text(f, f) == str(f)

    def test_different_files_shows_arrow_and_label(self, tmp_path: Path) -> None:
        src = tmp_path / "select.md"
        eff = tmp_path / "select.md.tmp"
        text = next_dry_run_prompt_text(src, eff)
        assert text == f"{src} -> {eff} (preprocessed)"


# ===========================================================================
# loop.select — _print_dry_run_prompt
# ===========================================================================


class TestNextPrintDryRunPrompt:
    def test_prints_correct_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        next_print_dry_run_prompt("selector", "/tmp/select.md")
        out, _ = capsys.readouterr()
        assert out.strip() == "dry-run: prompt [selector]: /tmp/select.md"

    def test_includes_label_in_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        next_print_dry_run_prompt("custom-label", "some text")
        out, _ = capsys.readouterr()
        assert "[custom-label]" in out
        assert "some text" in out


# ===========================================================================
# loop.select — run
# ===========================================================================


class TestNextRun:
    def _make_invocation(self, tmp_path: Path) -> PreparedSelectInvocation:
        prompt = tmp_path / "select.md"
        prompt.write_text("select a task\n", encoding="utf-8")
        return PreparedSelectInvocation(
            source_prompt_file=prompt,
            effective_prompt_file=prompt,
            command=["fake-cmd"],
            command_kind="selector",
            runner_command=["fake-runner"],
            selector_command=["fake-cmd"],
        )

    def test_errors_when_no_selector_mode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agm.commands.loop.select.use_selector_mode", lambda args: False
        )

        args = _make_loop_select_args(no_selector=True)
        with pytest.raises(SystemExit) as exc_info:
            next_run(args)
        assert exc_info.value.code == 1
        _, err = capsys.readouterr()
        assert "selector" in err.lower()

    def test_dry_run_prints_configuration_and_skips_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        invocation = self._make_invocation(tmp_path)
        monkeypatch.setattr(
            "agm.commands.loop.select.use_selector_mode", lambda args: True
        )
        monkeypatch.setattr(
            "agm.commands.loop.select.prepare_select_invocation",
            lambda args, temp_files, env: invocation,
        )
        monkeypatch.setattr("agm.commands.loop.select.dry_run.enabled", lambda: True)
        monkeypatch.setattr("agm.commands.loop.select.tasks_dir", lambda args: tmp_path / "tasks")
        monkeypatch.setattr("agm.commands.loop.select.loop_env", lambda d: {})

        run_command_called = [False]

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            idle_timeout: float | None = None,
        ) -> str:
            run_command_called[0] = True
            return ""

        monkeypatch.setattr("agm.commands.loop.select.run_prompt_command", fake_run_command)

        args = _make_loop_select_args()
        next_run(args)

        assert run_command_called[0] is False
        out, _ = capsys.readouterr()
        assert "dry-run" in out

    def test_run_command_output_is_printed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        invocation = self._make_invocation(tmp_path)
        monkeypatch.setattr(
            "agm.commands.loop.select.use_selector_mode", lambda args: True
        )
        monkeypatch.setattr(
            "agm.commands.loop.select.prepare_select_invocation",
            lambda args, temp_files, env: invocation,
        )
        monkeypatch.setattr("agm.commands.loop.select.dry_run.enabled", lambda: False)
        monkeypatch.setattr("agm.commands.loop.select.tasks_dir", lambda args: tmp_path / "tasks")
        monkeypatch.setattr("agm.commands.loop.select.loop_env", lambda d: {})
        monkeypatch.setattr("agm.commands.loop.select.cleanup_temp_files", lambda files: None)
        monkeypatch.setattr(
            "agm.commands.loop.select.run_prompt_command",
            lambda command, target, *, env, idle_timeout=None: "task-1.md",
        )

        args = _make_loop_select_args()
        next_run(args)

        out, _ = capsys.readouterr()
        assert "task-1.md" in out

    def test_keyboard_interrupt_exits_130(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        invocation = self._make_invocation(tmp_path)
        monkeypatch.setattr(
            "agm.commands.loop.select.use_selector_mode", lambda args: True
        )
        monkeypatch.setattr(
            "agm.commands.loop.select.prepare_select_invocation",
            lambda args, temp_files, env: invocation,
        )
        monkeypatch.setattr("agm.commands.loop.select.dry_run.enabled", lambda: False)
        monkeypatch.setattr("agm.commands.loop.select.tasks_dir", lambda args: tmp_path / "tasks")
        monkeypatch.setattr("agm.commands.loop.select.loop_env", lambda d: {})
        monkeypatch.setattr(
            "agm.commands.loop.select.run_prompt_command",
            lambda command, target, *, env, idle_timeout=None: (
                _ for _ in ()
            ).throw(KeyboardInterrupt),
        )

        args = _make_loop_select_args()
        with pytest.raises(SystemExit) as exc_info:
            next_run(args)
        assert exc_info.value.code == 130

    def test_cleanup_is_called_even_on_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        invocation = self._make_invocation(tmp_path)
        monkeypatch.setattr(
            "agm.commands.loop.select.use_selector_mode", lambda args: True
        )
        monkeypatch.setattr(
            "agm.commands.loop.select.prepare_select_invocation",
            lambda args, temp_files, env: invocation,
        )
        monkeypatch.setattr("agm.commands.loop.select.dry_run.enabled", lambda: False)
        monkeypatch.setattr("agm.commands.loop.select.tasks_dir", lambda args: tmp_path / "tasks")
        monkeypatch.setattr("agm.commands.loop.select.loop_env", lambda d: {})
        monkeypatch.setattr(
            "agm.commands.loop.select.run_prompt_command",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        cleanup_called = [False]

        def fake_cleanup(files: list[Path]) -> None:
            cleanup_called[0] = True

        monkeypatch.setattr("agm.commands.loop.select.cleanup_temp_files", fake_cleanup)

        args = _make_loop_select_args()
        with pytest.raises(RuntimeError):
            next_run(args)
        assert cleanup_called[0] is True


# ===========================================================================
# loop.run — run
# ===========================================================================


class TestLoopRun:
    def _stub_prepare(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> LoopStepRuntime:
        runtime = _make_runtime(tmp_path)
        monkeypatch.setattr(
            "agm.commands.loop.run.step_command.prepare_runtime", lambda a: runtime
        )
        monkeypatch.setattr(
            "agm.commands.loop.run.step_command.cleanup_runtime", lambda r: None
        )
        return runtime

    def test_loops_until_execute_returns_true(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        runtime = self._stub_prepare(tmp_path, monkeypatch)
        monkeypatch.setattr("agm.commands.loop.run.dry_run.enabled", lambda: False)

        call_count = [0]

        def fake_execute(r: LoopStepRuntime, *, step_number: int) -> bool:
            call_count[0] += 1
            print(f"Step {step_number}")
            return call_count[0] >= 3

        monkeypatch.setattr("agm.commands.loop.run.step_command.execute_single_step", fake_execute)

        args = _make_loop_args()
        loop_run(args)
        out, _ = capsys.readouterr()
        assert call_count[0] == 3
        assert out.startswith(f"Tasks dir: {runtime.resolved_tasks_dir}\nStep 1\n")
        assert out.count("Tasks dir:") == 1

    def test_increments_step_number_across_iterations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_prepare(tmp_path, monkeypatch)
        monkeypatch.setattr("agm.commands.loop.run.dry_run.enabled", lambda: False)

        step_numbers: list[int] = []

        def fake_execute(r: LoopStepRuntime, *, step_number: int) -> bool:
            step_numbers.append(step_number)
            return len(step_numbers) >= 3

        monkeypatch.setattr("agm.commands.loop.run.step_command.execute_single_step", fake_execute)

        args = _make_loop_args()
        loop_run(args)
        assert step_numbers == [1, 2, 3]

    def test_dry_run_calls_print_dry_run_and_does_not_execute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_prepare(tmp_path, monkeypatch)
        monkeypatch.setattr("agm.commands.loop.run.dry_run.enabled", lambda: True)

        dry_run_called = [False]

        def fake_print_dry_run(r: LoopStepRuntime) -> None:
            dry_run_called[0] = True

        execute_called = [False]

        def fake_execute(r: LoopStepRuntime, *, step_number: int) -> bool:
            execute_called[0] = True
            return True

        monkeypatch.setattr(
            "agm.commands.loop.run.step_command.print_dry_run", fake_print_dry_run
        )
        monkeypatch.setattr(
            "agm.commands.loop.run.step_command.execute_single_step", fake_execute
        )

        args = _make_loop_args()
        loop_run(args)
        assert dry_run_called[0] is True
        assert execute_called[0] is False

    def test_keyboard_interrupt_exits_130(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_prepare(tmp_path, monkeypatch)
        monkeypatch.setattr("agm.commands.loop.run.dry_run.enabled", lambda: False)
        monkeypatch.setattr(
            "agm.commands.loop.run.step_command.execute_single_step",
            lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt),
        )

        args = _make_loop_args()
        with pytest.raises(SystemExit) as exc_info:
            loop_run(args)
        assert exc_info.value.code == 130

    def test_cleanup_is_called_even_on_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = _make_runtime(tmp_path)
        monkeypatch.setattr(
            "agm.commands.loop.run.step_command.prepare_runtime", lambda a: runtime
        )
        monkeypatch.setattr("agm.commands.loop.run.dry_run.enabled", lambda: False)

        def fake_execute(r: LoopStepRuntime, *, step_number: int) -> bool:
            raise RuntimeError("step failed")

        monkeypatch.setattr(
            "agm.commands.loop.run.step_command.execute_single_step", fake_execute
        )

        cleanup_called = [False]

        def fake_cleanup(r: LoopStepRuntime) -> None:
            cleanup_called[0] = True

        monkeypatch.setattr(
            "agm.commands.loop.run.step_command.cleanup_runtime", fake_cleanup
        )

        args = _make_loop_args()
        with pytest.raises(RuntimeError):
            loop_run(args)
        assert cleanup_called[0] is True

    def test_terminates_immediately_when_execute_returns_true_on_first_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_prepare(tmp_path, monkeypatch)
        monkeypatch.setattr("agm.commands.loop.run.dry_run.enabled", lambda: False)

        call_count = [0]

        def fake_execute(r: LoopStepRuntime, *, step_number: int) -> bool:
            call_count[0] += 1
            return True

        monkeypatch.setattr("agm.commands.loop.run.step_command.execute_single_step", fake_execute)

        args = _make_loop_args()
        loop_run(args)
        assert call_count[0] == 1

    def test_cleanup_not_called_when_prepare_runtime_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """finally block when runtime is None after prepare_runtime raises.

        When prepare_runtime raises before returning, runtime stays None and
        cleanup_runtime must NOT be called.
        """
        monkeypatch.setattr(
            "agm.commands.loop.run.step_command.prepare_runtime",
            lambda _args: (_ for _ in ()).throw(RuntimeError("prepare failed")),
        )
        cleanup_called = [False]
        monkeypatch.setattr(
            "agm.commands.loop.run.step_command.cleanup_runtime",
            lambda r: cleanup_called.__setitem__(0, True),
        )

        args = _make_loop_args()
        with pytest.raises(RuntimeError, match="prepare failed"):
            loop_run(args)
        assert cleanup_called[0] is False


class TestPrintDryRunFull:
    def test_print_dry_run_with_selector_invocation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agm.commands.loop.step import LoopStepRuntime, print_dry_run

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        progress = tasks_dir / "PROGRESS.md"
        prompt = tmp_path / "select.md"
        prompt.write_text("select\n", encoding="utf-8")

        invocation = PreparedSelectInvocation(
            source_prompt_file=prompt,
            effective_prompt_file=prompt,
            command=["selector"],
            command_kind="selector",
            runner_command=["runner"],
            selector_command=["selector"],
        )

        runtime = LoopStepRuntime(
            temp_files=[],
            resolved_tasks_dir=tasks_dir,
            resolved_progress_file=progress,
            env={},
            resolved_runner_command=["runner"],
            select_invocation=invocation,
            implement_prompt_file=None,
            loop_prompt=None,
            resolved_prompt=None,
            bootstrap_prompt=None,
            extra_prompt_source=None,
            log_file=None,
            idle_timeout=5.0,
        )

        from agm.core import dry_run

        dry_run_calls: list[Any] = []
        monkeypatch.setattr(
            dry_run, "print_configuration", lambda label: dry_run_calls.append(("config", label))
        )
        monkeypatch.setattr(
            dry_run, "print_detail", lambda k, v: dry_run_calls.append(("detail", k, v))
        )
        monkeypatch.setattr(
            dry_run,
            "print_labeled_command",
            lambda label, cmd, cwd=None: dry_run_calls.append(("cmd", label, cmd)),
        )
        monkeypatch.setattr(
            dry_run,
            "format_command",
            lambda cmd: " ".join(cmd),
        )
        monkeypatch.setattr(
            dry_run,
            "print_operation",
            lambda name, detail: dry_run_calls.append(("op", name, detail)),
        )

        print_dry_run(runtime)

        detail_calls = [c for c in dry_run_calls if c[0] == "detail"]
        assert any(d[1] == "idle timeout" and d[2] == "5.0s" for d in detail_calls)
        assert any(d[1] == "selector command" and d[2] == "selector" for d in detail_calls)

    def test_print_dry_run_with_selector_and_implement_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agm.commands.loop.step import LoopStepRuntime, print_dry_run

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        progress = tasks_dir / "PROGRESS.md"
        prompt = tmp_path / "select.md"
        prompt.write_text("select\n", encoding="utf-8")
        implement_file = tmp_path / "implement.md"
        implement_file.write_text("implement\n", encoding="utf-8")

        invocation = PreparedSelectInvocation(
            source_prompt_file=prompt,
            effective_prompt_file=prompt,
            command=["selector"],
            command_kind="selector",
            runner_command=["runner"],
            selector_command=["selector"],
        )

        runtime = LoopStepRuntime(
            temp_files=[],
            resolved_tasks_dir=tasks_dir,
            resolved_progress_file=progress,
            env={},
            resolved_runner_command=["runner"],
            select_invocation=invocation,
            implement_prompt_file=implement_file,
            loop_prompt=None,
            resolved_prompt=None,
            bootstrap_prompt=None,
            extra_prompt_source=None,
            log_file=None,
            idle_timeout=None,
        )

        print_dry_run(runtime)

        output = capsys.readouterr().out
        assert "implement.md" in output
        assert "(default)" in output

    def test_print_dry_run_selector_with_explicit_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agm.commands.loop.step import LoopStepRuntime, print_dry_run

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        progress = tasks_dir / "PROGRESS.md"
        prompt = tmp_path / "select.md"
        prompt.write_text("select\n", encoding="utf-8")

        invocation = PreparedSelectInvocation(
            source_prompt_file=prompt,
            effective_prompt_file=prompt,
            command=["selector"],
            command_kind="selector",
            runner_command=["runner"],
            selector_command=["selector"],
        )

        resolved_file = tmp_path / "custom-prompt.md"
        resolved_file.write_text("custom\n", encoding="utf-8")
        resolved_prompt = ResolvedPrompt(source=resolved_file, effective_file=resolved_file)

        runtime = LoopStepRuntime(
            temp_files=[],
            resolved_tasks_dir=tasks_dir,
            resolved_progress_file=progress,
            env={},
            resolved_runner_command=["runner"],
            select_invocation=invocation,
            implement_prompt_file=None,
            loop_prompt=None,
            resolved_prompt=resolved_prompt,
            bootstrap_prompt=None,
            extra_prompt_source=None,
            log_file=None,
            idle_timeout=None,
        )

        print_dry_run(runtime)

        output = capsys.readouterr().out
        assert "custom-prompt" in output

    def test_print_dry_run_without_selector_invocation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.commands.loop.step import LoopStepRuntime, PreparedPrompt, print_dry_run

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        progress = tasks_dir / "PROGRESS.md"
        prompt = tmp_path / "loop.md"
        prompt.write_text("loop\n", encoding="utf-8")

        loop_prompt = PreparedPrompt(
            label="loop", source_file=prompt, effective_file=prompt
        )

        runtime = LoopStepRuntime(
            temp_files=[],
            resolved_tasks_dir=tasks_dir,
            resolved_progress_file=progress,
            env={},
            resolved_runner_command=["runner"],
            select_invocation=None,
            implement_prompt_file=None,
            loop_prompt=loop_prompt,
            resolved_prompt=None,
            bootstrap_prompt=None,
            extra_prompt_source=None,
            log_file=tmp_path / "test.log",
            idle_timeout=None,
        )

        from agm.core import dry_run

        dry_run_calls: list[Any] = []
        monkeypatch.setattr(
            dry_run, "print_configuration", lambda label: dry_run_calls.append(("config", label))
        )
        monkeypatch.setattr(
            dry_run, "print_detail", lambda k, v: dry_run_calls.append(("detail", k, v))
        )
        monkeypatch.setattr(
            dry_run,
            "print_labeled_command",
            lambda label, cmd, cwd=None: dry_run_calls.append(("cmd", label, cmd)),
        )
        monkeypatch.setattr(
            dry_run,
            "format_command",
            lambda cmd: " ".join(cmd),
        )
        monkeypatch.setattr(
            dry_run,
            "print_operation",
            lambda name, detail: dry_run_calls.append(("op", name, detail)),
        )

        print_dry_run(runtime)

        detail_calls = [c for c in dry_run_calls if c[0] == "detail"]
        assert any(d[1] == "idle timeout" and d[2] == "disabled" for d in detail_calls)
        assert any(d[1] == "log file" and str(tmp_path / "test.log") in d[2] for d in detail_calls)

    def test_print_dry_run_with_explicit_prompt_detail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.commands.loop.step import LoopStepRuntime, PreparedPrompt, print_dry_run

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        progress = tasks_dir / "PROGRESS.md"
        prompt = tmp_path / "loop.md"
        prompt.write_text("loop\n", encoding="utf-8")

        loop_prompt = PreparedPrompt(
            label="loop", source_file=prompt, effective_file=prompt
        )
        resolved_prompt = ResolvedPrompt(
            source="inline text", effective_file=tmp_path / "inline.md"
        )

        runtime = LoopStepRuntime(
            temp_files=[],
            resolved_tasks_dir=tasks_dir,
            resolved_progress_file=progress,
            env={},
            resolved_runner_command=["runner"],
            select_invocation=None,
            implement_prompt_file=None,
            loop_prompt=loop_prompt,
            resolved_prompt=resolved_prompt,
            bootstrap_prompt=None,
            extra_prompt_source=None,
            log_file=None,
            idle_timeout=None,
        )

        from agm.core import dry_run

        dry_run_calls: list[Any] = []
        monkeypatch.setattr(
            dry_run, "print_configuration", lambda label: dry_run_calls.append(("config", label))
        )
        monkeypatch.setattr(
            dry_run, "print_detail", lambda k, v: dry_run_calls.append(("detail", k, v))
        )
        monkeypatch.setattr(
            dry_run,
            "print_labeled_command",
            lambda label, cmd, cwd=None: dry_run_calls.append(("cmd", label, cmd)),
        )
        monkeypatch.setattr(
            dry_run,
            "format_command",
            lambda cmd: " ".join(cmd),
        )
        monkeypatch.setattr(
            dry_run,
            "print_operation",
            lambda name, detail: dry_run_calls.append(("op", name, detail)),
        )

        print_dry_run(runtime)

        detail_calls = [c for c in dry_run_calls if c[0] == "detail"]
        assert any(d[1] == "explicit prompt" for d in detail_calls)


class TestPrepareRuntimeMissingPromptFiles:
    def test_exits_when_select_md_prompt_file_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """prepare_runtime exits when bootstrap prompt (select.md) is missing."""
        from agm.commands.loop.step import prepare_runtime

        home = tmp_path / "home"
        (home / ".agm" / "prompts").mkdir(parents=True)
        # Create loop.md so the loop prompt check passes
        (home / ".agm" / "prompts" / "loop.md").write_text("loop", encoding="utf-8")
        # No select.md!
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner="fake-runner",
            runner_args=[],
            selector=None,
            no_selector=True,
            tasks_dir=None,
            no_log=True,
            log_file=None,
            prompt=None,
            prompt_file=None,
            selector_prompt=None,
            selector_prompt_file=None,
            extra_prompt=None,
            extra_prompt_file=None,
            extra_selector_prompt=None,
            extra_selector_prompt_file=None,
            timeout=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            prepare_runtime(args)
        assert exc_info.value.code == 1


class TestExecuteSingleStepSelectorStringResult:
    def test_selector_mode_retries_when_result_is_string(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When selector_result returns a string (not Path), selector retries."""
        from agm.commands.loop.step import LoopStepRuntime, execute_single_step

        prompt = tmp_path / "select.md"
        prompt.write_text("select\n", encoding="utf-8")
        invocation = PreparedSelectInvocation(
            source_prompt_file=prompt,
            effective_prompt_file=prompt,
            command=["fake-selector"],
            command_kind="selector",
            runner_command=["fake-runner"],
            selector_command=["fake-selector"],
        )
        tasks_dir_path = tmp_path / "tasks"
        tasks_dir_path.mkdir(parents=True)
        runtime = LoopStepRuntime(
            temp_files=[],
            resolved_tasks_dir=tasks_dir_path,
            resolved_progress_file=tasks_dir_path / "PROGRESS.md",
            env={},
            resolved_runner_command=["fake-runner"],
            select_invocation=invocation,
            implement_prompt_file=None,
            loop_prompt=None,
            resolved_prompt=None,
            bootstrap_prompt=None,
            extra_prompt_source=None,
            log_file=None,
            idle_timeout=None,
        )

        call_count = 0

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            stdout_callback: object = None,
            stderr_callback: object = None,
            idle_timeout: float | None = None,
        ) -> str:
            nonlocal call_count
            call_count += 1
            # Return COMPLETE on second call (the retry)
            return "COMPLETE"

        # First returns a string (not a path), then None (COMPLETE on retry)
        selector_results = ["not-a-file-path"]
        selector_idx = 0

        def fake_selector_result(output: str, *, tasks_dir: Path) -> Path | None | str:
            nonlocal selector_idx
            if selector_idx < len(selector_results):
                val = selector_results[selector_idx]
                selector_idx += 1
                return val
            return None  # COMPLETE on third call

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        monkeypatch.setattr("agm.commands.loop.step.selector_result", fake_selector_result)
        result = execute_single_step(runtime, step_number=1)
        # After string result, selector retries; eventually COMPLETE returns True
        assert result is True


class TestExecuteSingleStepWithResolvedPrompt:
    def test_selector_mode_uses_resolved_prompt_for_runner_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When runtime has resolved_prompt and loop_prompt, runner uses loop_prompt."""
        from agm.commands.loop.step import LoopStepRuntime, execute_single_step
        from agm.commands.loop.step import PreparedPrompt as StepPrepPrompt

        prompt = tmp_path / "select.md"
        prompt.write_text("select\n", encoding="utf-8")
        invocation = PreparedSelectInvocation(
            source_prompt_file=prompt,
            effective_prompt_file=prompt,
            command=["fake-selector"],
            command_kind="selector",
            runner_command=["fake-runner"],
            selector_command=["fake-selector"],
        )

        resolved_file = tmp_path / "resolved.md"
        resolved_file.write_text("resolved\n", encoding="utf-8")
        loop_file = tmp_path / "loop.md"
        loop_file.write_text("loop\n", encoding="utf-8")

        resolved_prompt = ResolvedPrompt(source="inline", effective_file=resolved_file)
        loop_prompt = StepPrepPrompt(
            label="loop", source_file=loop_file, effective_file=loop_file
        )

        tasks_dir_path = tmp_path / "tasks"
        tasks_dir_path.mkdir(parents=True)
        runtime = LoopStepRuntime(
            temp_files=[],
            resolved_tasks_dir=tasks_dir_path,
            resolved_progress_file=tasks_dir_path / "PROGRESS.md",
            env={},
            resolved_runner_command=["fake-runner"],
            select_invocation=invocation,
            implement_prompt_file=None,
            loop_prompt=loop_prompt,
            resolved_prompt=resolved_prompt,
            bootstrap_prompt=None,
            extra_prompt_source=None,
            log_file=None,
            idle_timeout=None,
        )

        task_file = tmp_path / "tasks" / "task-1.md"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text("do task\n", encoding="utf-8")

        run_targets: list[Path] = []
        run_envs: list[dict[str, str]] = []

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            stdout_callback: object = None,
            stderr_callback: object = None,
            idle_timeout: float | None = None,
        ) -> str:
            run_targets.append(target)
            run_envs.append(env)
            return "output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        monkeypatch.setattr(
            "agm.commands.loop.step.selector_result", lambda output, tasks_dir: task_file
        )
        result = execute_single_step(runtime, step_number=1)
        assert result is False
        # Prompt is re-prepared from original source with TASK_FILE in env,
        # so the target is a new temp file (not the original loop_file)
        assert run_targets[-1] != loop_file
        # TASK_FILE env var should be set for the runner
        assert "TASK_FILE" in run_envs[-1]


class TestExecuteSingleStepExpandsTaskFileInPrompt:
    def test_task_file_expanded_in_prompt_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When a prompt file contains ${TASK_FILE}, it is expanded after task selection."""
        from agm.commands.loop.step import LoopStepRuntime, execute_single_step
        from agm.commands.loop.step import PreparedPrompt as StepPrepPrompt

        select_prompt = tmp_path / "select.md"
        select_prompt.write_text("select\n", encoding="utf-8")
        invocation = PreparedSelectInvocation(
            source_prompt_file=select_prompt,
            effective_prompt_file=select_prompt,
            command=["fake-selector"],
            command_kind="selector",
            runner_command=["fake-runner"],
            selector_command=["fake-selector"],
        )

        # Prompt file with ${TASK_FILE} placeholder
        prompt_file_path = tmp_path / "loop.md"
        prompt_file_path.write_text(
            "Work on ${TASK_FILE}\n", encoding="utf-8"
        )

        # Simulate how prepare_runtime builds resolved_prompt + loop_prompt
        resolved_prompt = ResolvedPrompt(
            source=prompt_file_path, effective_file=prompt_file_path
        )
        loop_prompt = StepPrepPrompt(
            label="loop", source_file=prompt_file_path, effective_file=prompt_file_path
        )

        tasks_dir_path = tmp_path / "tasks"
        tasks_dir_path.mkdir(parents=True)
        runtime = LoopStepRuntime(
            temp_files=[],
            resolved_tasks_dir=tasks_dir_path,
            resolved_progress_file=tasks_dir_path / "PROGRESS.md",
            env={},
            resolved_runner_command=["fake-runner"],
            select_invocation=invocation,
            implement_prompt_file=None,
            loop_prompt=loop_prompt,
            resolved_prompt=resolved_prompt,
            bootstrap_prompt=None,
            extra_prompt_source=None,
            log_file=None,
            idle_timeout=None,
        )

        task_file = tmp_path / "tasks" / "task-1.md"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text("do task\n", encoding="utf-8")

        run_targets: list[Path] = []

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            stdout_callback: object = None,
            stderr_callback: object = None,
            idle_timeout: float | None = None,
        ) -> str:
            run_targets.append(target)
            return "output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        monkeypatch.setattr(
            "agm.commands.loop.step.selector_result", lambda output, tasks_dir: task_file
        )
        result = execute_single_step(runtime, step_number=1)
        assert result is False
        # The runner target should be a new file with TASK_FILE expanded,
        # not the original prompt file that still has ${TASK_FILE}
        runner_target = run_targets[-1]
        content = runner_target.read_text(encoding="utf-8")
        assert "${TASK_FILE}" not in content
        assert str(task_file) in content

    def test_task_file_expanded_in_inline_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When inline prompt text contains ${TASK_FILE}, it is expanded after task selection."""
        from agm.commands.loop.step import LoopStepRuntime, execute_single_step
        from agm.commands.loop.step import PreparedPrompt as StepPrepPrompt

        select_prompt = tmp_path / "select.md"
        select_prompt.write_text("select\n", encoding="utf-8")
        invocation = PreparedSelectInvocation(
            source_prompt_file=select_prompt,
            effective_prompt_file=select_prompt,
            command=["fake-selector"],
            command_kind="selector",
            runner_command=["fake-runner"],
            selector_command=["fake-selector"],
        )

        # Inline prompt text with ${TASK_FILE} placeholder
        inline_text = "Work on ${TASK_FILE}\n"
        from agm.agent.loop import loop_env
        env_no_task = loop_env(tmp_path / "tasks")
        resolved_prompt = ResolvedPrompt(source=inline_text, effective_file=tmp_path / "stub")
        loop_prompt = StepPrepPrompt(
            label="loop", source_file=tmp_path / "stub", effective_file=tmp_path / "stub"
        )

        tasks_dir_path = tmp_path / "tasks"
        tasks_dir_path.mkdir(parents=True)
        runtime = LoopStepRuntime(
            temp_files=[],
            resolved_tasks_dir=tasks_dir_path,
            resolved_progress_file=tasks_dir_path / "PROGRESS.md",
            env=env_no_task,
            resolved_runner_command=["fake-runner"],
            select_invocation=invocation,
            implement_prompt_file=None,
            loop_prompt=loop_prompt,
            resolved_prompt=resolved_prompt,
            bootstrap_prompt=None,
            extra_prompt_source=None,
            log_file=None,
            idle_timeout=None,
        )

        task_file = tmp_path / "tasks" / "task-1.md"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text("do task\n", encoding="utf-8")

        run_targets: list[Path] = []

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            stdout_callback: object = None,
            stderr_callback: object = None,
            idle_timeout: float | None = None,
        ) -> str:
            run_targets.append(target)
            return "output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        monkeypatch.setattr(
            "agm.commands.loop.step.selector_result", lambda output, tasks_dir: task_file
        )
        result = execute_single_step(runtime, step_number=1)
        assert result is False
        # The runner target content should have TASK_FILE expanded
        runner_target = run_targets[-1]
        content = runner_target.read_text(encoding="utf-8")
        assert "${TASK_FILE}" not in content
        assert str(task_file) in content


# ---------------------------------------------------------------------------
# parser.py – line 562: raise ValueError in _help_text_for_path
# ---------------------------------------------------------------------------


class TestPrintDryRunWithBootstrap:
    def test_print_dry_run_with_bootstrap_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """print_dry_run includes bootstrap prompt when present."""
        from agm.commands.loop.step import LoopStepRuntime, PreparedPrompt, print_dry_run

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        progress = tasks_dir / "PROGRESS.md"
        prompt = tmp_path / "loop.md"
        prompt.write_text("loop\n", encoding="utf-8")
        bootstrap = tmp_path / "select.md"
        bootstrap.write_text("bootstrap\n", encoding="utf-8")

        loop_prompt = PreparedPrompt(
            label="loop", source_file=prompt, effective_file=prompt
        )
        bootstrap_prompt = PreparedPrompt(
            label="bootstrap", source_file=bootstrap, effective_file=bootstrap
        )

        runtime = LoopStepRuntime(
            temp_files=[],
            resolved_tasks_dir=tasks_dir,
            resolved_progress_file=progress,
            env={},
            resolved_runner_command=["runner"],
            select_invocation=None,
            implement_prompt_file=None,
            loop_prompt=loop_prompt,
            resolved_prompt=None,
            bootstrap_prompt=bootstrap_prompt,
            extra_prompt_source=None,
            log_file=None,
            idle_timeout=None,
        )

        from agm.core import dry_run

        dry_run_calls: list[Any] = []
        monkeypatch.setattr(
            dry_run, "print_configuration", lambda label: dry_run_calls.append(("config", label))
        )
        monkeypatch.setattr(
            dry_run, "print_detail", lambda k, v: dry_run_calls.append(("detail", k, v))
        )
        monkeypatch.setattr(
            dry_run,
            "print_labeled_command",
            lambda label, cmd, cwd=None: dry_run_calls.append(("cmd", label, cmd)),
        )
        monkeypatch.setattr(
            dry_run,
            "format_command",
            lambda cmd: " ".join(cmd),
        )
        monkeypatch.setattr(
            dry_run,
            "print_operation",
            lambda name, detail: dry_run_calls.append(("op", name, detail)),
        )

        print_dry_run(runtime)

        # Bootstrap command should be printed
        cmd_calls = [c for c in dry_run_calls if c[0] == "cmd"]
        assert any(c[1] == "bootstrap" for c in cmd_calls)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# commands/loop/step.py – cleanup_runtime
# ---------------------------------------------------------------------------


class TestCleanupRuntimeViaStep:
    def test_cleanup_runtime_delegates_to_cleanup_temp_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cleanup_runtime delegates to cleanup_temp_files."""
        from agm.commands.loop.step import LoopStepRuntime, cleanup_runtime

        f1 = tmp_path / "temp1.md"
        f1.write_text("temp", encoding="utf-8")

        runtime = LoopStepRuntime(
            temp_files=[f1],
            resolved_tasks_dir=tmp_path,
            resolved_progress_file=tmp_path / "PROGRESS.md",
            env={},
            resolved_runner_command=[],
            select_invocation=None,
            implement_prompt_file=None,
            loop_prompt=None,
            resolved_prompt=None,
            bootstrap_prompt=None,
            extra_prompt_source=None,
            log_file=None,
            idle_timeout=None,
        )
        cleanup_runtime(runtime)
        assert not f1.exists()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# project/layout.py – current_project_dir git_common_dir path
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# project/layout.py – current_workspace with REPO_DIR env var
# ---------------------------------------------------------------------------

class TestPrepareRuntimeExtraPromptSource:
    def test_extra_prompt_applied_to_loop_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Extra prompt source is stored in runtime and applied to loop prompt."""
        home = tmp_path / "home"
        prompt_dir = home / ".agm" / "prompts"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "loop.md").write_text("# loop prompt1", encoding="utf-8")
        (prompt_dir / "select.md").write_text("# select1", encoding="utf-8")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        # Create progress file so bootstrap is skipped
        tasks_dir_path = tmp_path / ".agent-files" / "tasks"
        tasks_dir_path.mkdir(parents=True)
        (tasks_dir_path / "PROGRESS.md").write_text("done1", encoding="utf-8")

        args = _make_loop_args(no_log=True, no_selector=True, extra_prompt="extra stuff")
        runtime = prepare_runtime(args)
        assert runtime.extra_prompt_source == "extra stuff"
        cleanup_runtime(runtime)

    def test_extra_selector_prompt_applied_to_select_invocation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Extra selector prompt is appended to the selector invocation."""
        home = tmp_path / "home"
        prompt_dir = home / ".agm" / "prompts"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "loop.md").write_text("# loop1", encoding="utf-8")
        (prompt_dir / "select.md").write_text("select $TASKS_DIR1", encoding="utf-8")
        (prompt_dir / "implement.md").write_text("# implement1", encoding="utf-8")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        tasks_dir_path = tmp_path / ".agent-files" / "tasks"
        tasks_dir_path.mkdir(parents=True)
        (tasks_dir_path / "PROGRESS.md").write_text("done1", encoding="utf-8")

        args = _make_loop_args(
            no_log=True,
            no_selector=False,
            selector="fake-selector",
            extra_selector_prompt="extra selector stuff",
        )
        runtime = prepare_runtime(args)
        assert runtime.select_invocation is not None
        effective_text = runtime.select_invocation.effective_prompt_file.read_text(
            encoding="utf-8"
        )
        assert "extra selector stuff" in effective_text
        cleanup_runtime(runtime)


class TestExecuteSingleStepWithExtraPrompt:
    def test_extra_prompt_appended_to_runner_target_in_selector_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When extra_prompt_source is set, it's appended to the runner target."""
        task_file = tmp_path / "tasks" / "task-1.md"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text("do task1", encoding="utf-8")

        prompt_file = tmp_path / "select.md"
        prompt_file.write_text("select1", encoding="utf-8")

        implement_file = tmp_path / "implement.md"
        implement_file.write_text("implement @${TASK_FILE}1", encoding="utf-8")

        invocation = PreparedSelectInvocation(
            source_prompt_file=prompt_file,
            effective_prompt_file=prompt_file,
            command=["fake-selector"],
            command_kind="selector",
            runner_command=["fake-runner"],
            selector_command=["fake-selector"],
        )
        runtime = _make_runtime(
            tmp_path,
            select_invocation=invocation,
            implement_prompt_file=implement_file,
            loop_prompt=None,
        )
        runtime = LoopStepRuntime(
            temp_files=runtime.temp_files,
            resolved_tasks_dir=runtime.resolved_tasks_dir,
            resolved_progress_file=runtime.resolved_progress_file,
            env=runtime.env,
            resolved_runner_command=runtime.resolved_runner_command,
            select_invocation=runtime.select_invocation,
            implement_prompt_file=runtime.implement_prompt_file,
            loop_prompt=runtime.loop_prompt,
            resolved_prompt=runtime.resolved_prompt,
            bootstrap_prompt=runtime.bootstrap_prompt,
            extra_prompt_source="EXTRA CONTENT",
            log_file=runtime.log_file,
            idle_timeout=runtime.idle_timeout,
        )

        all_targets: list[Path] = []

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            stdout_callback: object = None,
            stderr_callback: object = None,
            idle_timeout: float | None = None,
        ) -> str:
            all_targets.append(target)
            return "output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        monkeypatch.setattr(
            "agm.commands.loop.step.selector_result",
            lambda output, tasks_dir: task_file,
        )
        execute_single_step(runtime, step_number=1)

        assert len(all_targets) == 2
        runner_target_text = all_targets[1].read_text(encoding="utf-8")
        assert "EXTRA CONTENT" in runner_target_text

    def test_extra_prompt_appended_to_runner_target_in_implement_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With resolved_prompt and extra_prompt_source, extra is appended."""
        task_file = tmp_path / "tasks" / "task-1.md"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text("do task1", encoding="utf-8")

        prompt_file = tmp_path / "select.md"
        prompt_file.write_text("select1", encoding="utf-8")

        custom_prompt = tmp_path / "custom-prompt.md"
        custom_prompt.write_text("Do $TASK_FILE stuff", encoding="utf-8")

        invocation = PreparedSelectInvocation(
            source_prompt_file=prompt_file,
            effective_prompt_file=prompt_file,
            command=["fake-selector"],
            command_kind="selector",
            runner_command=["fake-runner"],
            selector_command=["fake-selector"],
        )
        resolved_prompt = ResolvedPrompt(
            source=custom_prompt, effective_file=custom_prompt
        )
        runtime = _make_runtime(
            tmp_path,
            select_invocation=invocation,
            resolved_prompt=resolved_prompt,
            loop_prompt=None,
        )
        runtime = LoopStepRuntime(
            temp_files=runtime.temp_files,
            resolved_tasks_dir=runtime.resolved_tasks_dir,
            resolved_progress_file=runtime.resolved_progress_file,
            env=runtime.env,
            resolved_runner_command=runtime.resolved_runner_command,
            select_invocation=runtime.select_invocation,
            implement_prompt_file=runtime.implement_prompt_file,
            loop_prompt=runtime.loop_prompt,
            resolved_prompt=runtime.resolved_prompt,
            bootstrap_prompt=runtime.bootstrap_prompt,
            extra_prompt_source="APPENDED EXTRA",
            log_file=runtime.log_file,
            idle_timeout=runtime.idle_timeout,
        )

        all_targets: list[Path] = []

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            stdout_callback: object = None,
            stderr_callback: object = None,
            idle_timeout: float | None = None,
        ) -> str:
            all_targets.append(target)
            return "output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        monkeypatch.setattr(
            "agm.commands.loop.step.selector_result",
            lambda output, tasks_dir: task_file,
        )
        execute_single_step(runtime, step_number=1)

        assert len(all_targets) == 2
        runner_target_text = all_targets[1].read_text(encoding="utf-8")
        assert "APPENDED EXTRA" in runner_target_text


class TestNextRunWithExtraSelectorPrompt:
    def test_extra_selector_prompt_appended_to_invocation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        prompt_file = tmp_path / "select.md"
        prompt_file.write_text("select content", encoding="utf-8")

        effective_prompt = tmp_path / "effective-select.md"
        effective_prompt.write_text("select content", encoding="utf-8")

        invocation = PreparedSelectInvocation(
            source_prompt_file=prompt_file,
            effective_prompt_file=effective_prompt,
            command=["fake-selector"],
            command_kind="selector",
            runner_command=["fake-runner"],
            selector_command=["fake-selector"],
        )
        monkeypatch.setattr(
            "agm.commands.loop.select.use_selector_mode", lambda args: True
        )
        monkeypatch.setattr(
            "agm.commands.loop.select.prepare_select_invocation",
            lambda args, temp_files, env: invocation,
        )
        monkeypatch.setattr("agm.commands.loop.select.dry_run.enabled", lambda: False)
        monkeypatch.setattr("agm.commands.loop.select.tasks_dir", lambda args: tmp_path / "tasks")
        monkeypatch.setattr("agm.commands.loop.select.loop_env", lambda d: {})
        monkeypatch.setattr("agm.commands.loop.select.cleanup_temp_files", lambda files: None)
        monkeypatch.setattr(
            "agm.commands.loop.select.run_prompt_command",
            lambda command, target, *, env, idle_timeout=None: "task-1.md",
        )

        args = _make_loop_select_args(
            selector="fake-selector",
            extra_selector_prompt="extra selector context",
        )
        next_run(args)

        # The effective prompt file should now include the extra selector prompt
        effective_text = invocation.effective_prompt_file.read_text(encoding="utf-8")
        assert "extra selector context" in effective_text
