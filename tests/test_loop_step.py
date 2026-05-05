"""Comprehensive tests for agm.commands.loop.step, loop.next, and loop.run."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.commands.args import LoopArgs, LoopNextArgs
from agm.commands.loop.common import PreparedSelectInvocation, ResolvedPrompt
from agm.commands.loop.next import _dry_run_prompt_text as next_dry_run_prompt_text
from agm.commands.loop.next import _print_dry_run_prompt as next_print_dry_run_prompt
from agm.commands.loop.next import run as next_run
from agm.commands.loop.run import run as loop_run
from agm.commands.loop.step import (
    LoopStepRuntime,
    PreparedPrompt,
    _append_log,
    _dry_run_prompt_text,
    _log_file,
    _prepare_prompt,
    _print_dry_run_command,
    _print_dry_run_prompt,
    _write_stream,
    cleanup_runtime,
    execute_single_step,
    prepare_runtime,
    run,
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
    command_name: str | None = None,
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
    )


def _make_loop_next_args(
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
    command_name: str | None = None,
) -> LoopNextArgs:
    return LoopNextArgs(
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
    )


def _make_runtime(
    tmp_path: Path,
    *,
    select_invocation: PreparedSelectInvocation | None = None,
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
        loop_prompt=loop_prompt,
        resolved_prompt=resolved_prompt,
        bootstrap_prompt=bootstrap_prompt,
        log_file=log_file,
    )


# ===========================================================================
# _log_file
# ===========================================================================


class TestLogFile:
    def test_returns_none_when_no_log(self, tmp_path: Path) -> None:
        args = _make_loop_args(no_log=True, log_file=None)
        assert _log_file(args) is None

    def test_returns_explicit_log_file_when_given(self, tmp_path: Path) -> None:
        explicit = str(tmp_path / "my.log")
        args = _make_loop_args(no_log=False, log_file=explicit)
        result = _log_file(args)
        assert result == Path(explicit)

    def test_generates_timestamped_log_file_in_cwd_when_no_log_file_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        args = _make_loop_args(no_log=False, log_file=None)
        result = _log_file(args)
        assert result is not None
        assert result.parent == tmp_path
        assert result.name.startswith("loop-")
        assert result.suffix == ".log"

    def test_no_log_overrides_explicit_log_file(self, tmp_path: Path) -> None:
        args = _make_loop_args(no_log=True, log_file=str(tmp_path / "ignored.log"))
        assert _log_file(args) is None


# ===========================================================================
# _append_log
# ===========================================================================


class TestAppendLog:
    def test_no_op_when_log_file_is_none(self, tmp_path: Path) -> None:
        # Should not raise; no file should be written
        _append_log(None, "some content")

    def test_no_op_when_content_is_empty(self, tmp_path: Path) -> None:
        log = tmp_path / "out.log"
        _append_log(log, "")
        assert not log.exists()

    def test_appends_content_to_file(self, tmp_path: Path) -> None:
        log = tmp_path / "out.log"
        _append_log(log, "first line\n")
        _append_log(log, "second line\n")
        assert log.read_text(encoding="utf-8") == "first line\nsecond line\n"

    def test_creates_file_if_missing(self, tmp_path: Path) -> None:
        log = tmp_path / "new.log"
        _append_log(log, "hello")
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
# _dry_run_prompt_text (step)
# ===========================================================================


class TestDryRunPromptText:
    def test_source_equals_effective_returns_just_source(self, tmp_path: Path) -> None:
        f = tmp_path / "loop.md"
        prompt = PreparedPrompt(label="loop", source_file=f, effective_file=f)
        assert _dry_run_prompt_text(prompt) == str(f)

    def test_different_files_shows_arrow_and_preprocessed_label(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "loop.md"
        eff = tmp_path / "loop.md.tmp"
        prompt = PreparedPrompt(label="loop", source_file=src, effective_file=eff)
        text = _dry_run_prompt_text(prompt)
        assert text == f"{src} -> {eff} (preprocessed)"


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
    def _setup_home_with_prompts(self, tmp_path: Path, prompts: list[str]) -> Path:
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

        def fake_run_command(command: list[str], target: Path, *, env: dict[str, str]) -> str:
            run_calls.append(command)
            return ""

        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.setattr("agm.commands.loop.step.run_command", fake_run_command)

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
        home = self._setup_home_with_prompts(tmp_path, ["loop.md", "select.md"])
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        args = _make_loop_args(
            no_log=True, no_selector=False, runner="fake-runner", selector="fake-selector"
        )
        runtime = prepare_runtime(args)

        assert runtime.select_invocation is not None
        assert runtime.loop_prompt is None
        cleanup_runtime(runtime)


# ===========================================================================
# execute_single_step
# ===========================================================================


class TestExecuteSingleStep:
    def test_no_selector_returns_true_when_output_is_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = _make_runtime(tmp_path)
        monkeypatch.setattr(
            "agm.commands.loop.step.run_command",
            lambda *a, **kw: "COMPLETE",
        )
        result = execute_single_step(runtime, step_number=1)
        assert result is True

    def test_no_selector_returns_false_when_output_is_not_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = _make_runtime(tmp_path)
        monkeypatch.setattr(
            "agm.commands.loop.step.run_command",
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
        ) -> str:
            captured_callbacks["stdout_callback"] = stdout_callback
            captured_callbacks["stderr_callback"] = stderr_callback
            return "COMPLETE"

        monkeypatch.setattr("agm.commands.loop.step.run_command", fake_run_command)
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
        ) -> str:
            if callable(stdout_callback):
                stdout_callback("stdout chunk\n")
            if callable(stderr_callback):
                stderr_callback("stderr chunk\n")
            return "COMPLETE"

        monkeypatch.setattr("agm.commands.loop.step.run_command", fake_run_command)
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
            "agm.commands.loop.step.run_command",
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
        runtime = _make_runtime(tmp_path, select_invocation=invocation, loop_prompt=None)

        monkeypatch.setattr(
            "agm.commands.loop.step.run_command",
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

        invocation = PreparedSelectInvocation(
            source_prompt_file=prompt_file,
            effective_prompt_file=prompt_file,
            command=["fake-selector"],
            command_kind="selector",
            runner_command=["fake-runner"],
            selector_command=["fake-selector"],
        )
        runtime = _make_runtime(tmp_path, select_invocation=invocation, loop_prompt=None)

        call_count = 0

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            stdout_callback: object = None,
            stderr_callback: object = None,
        ) -> str:
            nonlocal call_count
            call_count += 1
            return "task output"

        monkeypatch.setattr("agm.commands.loop.step.run_command", fake_run_command)
        monkeypatch.setattr(
            "agm.commands.loop.step.selector_result",
            lambda output, tasks_dir: task_file,
        )
        result = execute_single_step(runtime, step_number=1)
        assert result is False
        # selector + runner calls = 2
        assert call_count == 2

    def test_selector_mode_retries_until_valid_task_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompt_file = tmp_path / "select.md"
        prompt_file.write_text("select\n", encoding="utf-8")
        task_file = tmp_path / "tasks" / "task-1.md"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text("do task\n", encoding="utf-8")

        invocation = PreparedSelectInvocation(
            source_prompt_file=prompt_file,
            effective_prompt_file=prompt_file,
            command=["fake-selector"],
            command_kind="selector",
            runner_command=["fake-runner"],
            selector_command=["fake-selector"],
        )
        runtime = _make_runtime(tmp_path, select_invocation=invocation, loop_prompt=None)

        selector_call_count = 0
        selector_results: list[Path | None | str] = ["not a path yet", task_file]

        def fake_selector_result(output: str, *, tasks_dir: Path) -> Path | None | str:
            nonlocal selector_call_count
            val = selector_results[selector_call_count]
            selector_call_count += 1
            return val

        monkeypatch.setattr("agm.commands.loop.step.run_command", lambda *a, **kw: "output")
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
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_prepare(tmp_path, monkeypatch)
        step_calls: list[int] = []

        def fake_execute(r: LoopStepRuntime, *, step_number: int) -> bool:
            step_calls.append(step_number)
            return True

        monkeypatch.setattr("agm.commands.loop.step.execute_single_step", fake_execute)
        monkeypatch.setattr("agm.commands.loop.step.dry_run.enabled", lambda: False)

        args = _make_loop_args()
        run(args)
        assert step_calls == [1]

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


# ===========================================================================
# loop.next — _dry_run_prompt_text
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
# loop.next — _print_dry_run_prompt
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
# loop.next — run
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
            "agm.commands.loop.next.use_selector_mode", lambda args: False
        )

        args = _make_loop_next_args(no_selector=True)
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
            "agm.commands.loop.next.use_selector_mode", lambda args: True
        )
        monkeypatch.setattr(
            "agm.commands.loop.next.prepare_select_invocation",
            lambda args, temp_files, env: invocation,
        )
        monkeypatch.setattr("agm.commands.loop.next.dry_run.enabled", lambda: True)
        monkeypatch.setattr("agm.commands.loop.next.tasks_dir", lambda args: tmp_path / "tasks")
        monkeypatch.setattr("agm.commands.loop.next.loop_env", lambda d: {})

        run_command_called = [False]

        def fake_run_command(
            command: list[str], target: Path, *, env: dict[str, str]
        ) -> str:
            run_command_called[0] = True
            return ""

        monkeypatch.setattr("agm.commands.loop.next.run_command", fake_run_command)

        args = _make_loop_next_args()
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
            "agm.commands.loop.next.use_selector_mode", lambda args: True
        )
        monkeypatch.setattr(
            "agm.commands.loop.next.prepare_select_invocation",
            lambda args, temp_files, env: invocation,
        )
        monkeypatch.setattr("agm.commands.loop.next.dry_run.enabled", lambda: False)
        monkeypatch.setattr("agm.commands.loop.next.tasks_dir", lambda args: tmp_path / "tasks")
        monkeypatch.setattr("agm.commands.loop.next.loop_env", lambda d: {})
        monkeypatch.setattr("agm.commands.loop.next.cleanup_temp_files", lambda files: None)
        monkeypatch.setattr(
            "agm.commands.loop.next.run_command",
            lambda command, target, *, env: "task-1.md",
        )

        args = _make_loop_next_args()
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
            "agm.commands.loop.next.use_selector_mode", lambda args: True
        )
        monkeypatch.setattr(
            "agm.commands.loop.next.prepare_select_invocation",
            lambda args, temp_files, env: invocation,
        )
        monkeypatch.setattr("agm.commands.loop.next.dry_run.enabled", lambda: False)
        monkeypatch.setattr("agm.commands.loop.next.tasks_dir", lambda args: tmp_path / "tasks")
        monkeypatch.setattr("agm.commands.loop.next.loop_env", lambda d: {})
        monkeypatch.setattr(
            "agm.commands.loop.next.run_command",
            lambda command, target, *, env: (_ for _ in ()).throw(KeyboardInterrupt),
        )

        args = _make_loop_next_args()
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
            "agm.commands.loop.next.use_selector_mode", lambda args: True
        )
        monkeypatch.setattr(
            "agm.commands.loop.next.prepare_select_invocation",
            lambda args, temp_files, env: invocation,
        )
        monkeypatch.setattr("agm.commands.loop.next.dry_run.enabled", lambda: False)
        monkeypatch.setattr("agm.commands.loop.next.tasks_dir", lambda args: tmp_path / "tasks")
        monkeypatch.setattr("agm.commands.loop.next.loop_env", lambda d: {})
        monkeypatch.setattr(
            "agm.commands.loop.next.run_command",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        cleanup_called = [False]

        def fake_cleanup(files: list[Path]) -> None:
            cleanup_called[0] = True

        monkeypatch.setattr("agm.commands.loop.next.cleanup_temp_files", fake_cleanup)

        args = _make_loop_next_args()
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
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_prepare(tmp_path, monkeypatch)
        monkeypatch.setattr("agm.commands.loop.run.dry_run.enabled", lambda: False)

        call_count = [0]

        def fake_execute(r: LoopStepRuntime, *, step_number: int) -> bool:
            call_count[0] += 1
            return call_count[0] >= 3

        monkeypatch.setattr("agm.commands.loop.run.step_command.execute_single_step", fake_execute)

        args = _make_loop_args()
        loop_run(args)
        assert call_count[0] == 3

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
