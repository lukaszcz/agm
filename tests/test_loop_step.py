"""Comprehensive tests for agm.commands.loop.step, loop.select, and loop.run."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

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
    print_dry_run,
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
    extra_prompt_source: str | Path | None = None,
    idle_timeout: float | None = None,
    env: dict[str, str] | None = None,
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
        env=env if env is not None else {"TASKS_DIR": str(tasks_dir)},
        resolved_runner_command=runner_command if runner_command is not None else ["myrunner"],
        select_invocation=select_invocation,
        implement_prompt_file=implement_prompt_file,
        loop_prompt=loop_prompt,
        resolved_prompt=resolved_prompt,
        bootstrap_prompt=bootstrap_prompt,
        extra_prompt_source=extra_prompt_source,
        log_file=log_file,
        idle_timeout=idle_timeout,
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
    def test_prints_labeled_command_to_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _print_dry_run_command("runner", ["myrunner", "--verbose"])
        out, _ = capsys.readouterr()
        assert "command [runner]:" in out
        assert "myrunner" in out
        assert "--verbose" in out


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
        run_targets: list[Path] = []

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            idle_timeout: float | None = None,
        ) -> str:
            run_calls.append(command)
            run_targets.append(target)
            return ""

        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)

        args = _make_loop_args(no_log=True, no_selector=True, runner="fake-runner")
        runtime = prepare_runtime(args)

        assert runtime.bootstrap_prompt is not None
        assert runtime.bootstrap_prompt.label == "bootstrap"
        # bootstrap runner was invoked with the runner command targeting the bootstrap prompt
        assert run_calls == [["fake-runner"]]
        assert run_targets == [runtime.bootstrap_prompt.effective_file]
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
        # Verifies that selector mode requires implement.md when no explicit prompt is given.

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
        # Real selector_result: "COMPLETE" on the last line → returns None → done.
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
            # Selector call: return task filename so real selector_result resolves it.
            # Runner call: return arbitrary output.
            if command == ["fake-selector"]:
                return "task-1.md\n"
            return "task output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        # Real selector_result: "task-1.md" found in tasks_dir → returns Path.
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
            # Selector call: real selector_result resolves "task-1.md" → task_file.
            if command == ["fake-selector"]:
                return "task-1.md\n"
            return "task output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        # Real selector_result: "task-1.md" found in tasks_dir → returns Path.
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
            # Selector call: real selector_result resolves "task-1.md" → task_file.
            if command == ["fake-selector"]:
                return "task-1.md\n"
            return "output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        # Real selector_result: "task-1.md" found in tasks_dir → returns Path.
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
            # Selector call: real selector_result resolves "task-1.md" → task_file.
            if command == ["fake-selector"]:
                return "task-1.md\n"
            return "output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        # Real selector_result: "task-1.md" found in tasks_dir → returns Path.
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

        def fake_run_command(
            command: list[str],
            target: Path,
            *,
            env: dict[str, str],
            stdout_callback: object = None,
            stderr_callback: object = None,
            idle_timeout: float | None = None,
        ) -> str:
            nonlocal selector_call_count
            if command == ["fake-selector"]:
                selector_call_count += 1
                # 1st selector call: return non-resolvable text → real selector_result → str.
                # 2nd selector call: return task filename → real selector_result → Path.
                if selector_call_count == 1:
                    return "not a path yet\n"
                return "task-1.md\n"
            return "output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        # Real selector_result drives the retry: str on 1st call, Path on 2nd.
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


# ---------------------------------------------------------------------------
# Runtime builders for TestPrintDryRunFull parametrized cases
# ---------------------------------------------------------------------------


def _make_selector_invocation(tmp_path: Path) -> PreparedSelectInvocation:
    """Build a selector PreparedSelectInvocation for dry-run tests."""
    prompt = tmp_path / "select.md"
    prompt.write_text("select\n", encoding="utf-8")
    return PreparedSelectInvocation(
        source_prompt_file=prompt,
        effective_prompt_file=prompt,
        command=["selector"],
        command_kind="selector",
        runner_command=["runner"],
        selector_command=["selector"],
    )


def _dry_run_selector_idle_timeout(tmp_path: Path) -> LoopStepRuntime:
    return _make_runtime(
        tmp_path,
        select_invocation=_make_selector_invocation(tmp_path),
        loop_prompt=None,
        runner_command=["runner"],
        env={},
        idle_timeout=5.0,
    )


def _dry_run_selector_implement_prompt(tmp_path: Path) -> LoopStepRuntime:
    impl = tmp_path / "implement.md"
    impl.write_text("implement\n", encoding="utf-8")
    return _make_runtime(
        tmp_path,
        select_invocation=_make_selector_invocation(tmp_path),
        implement_prompt_file=impl,
        loop_prompt=None,
        runner_command=["runner"],
        env={},
    )


def _dry_run_selector_explicit_prompt(tmp_path: Path) -> LoopStepRuntime:
    resolved_file = tmp_path / "custom-prompt.md"
    resolved_file.write_text("custom\n", encoding="utf-8")
    rp = ResolvedPrompt(source=resolved_file, effective_file=resolved_file)
    return _make_runtime(
        tmp_path,
        select_invocation=_make_selector_invocation(tmp_path),
        resolved_prompt=rp,
        loop_prompt=None,
        runner_command=["runner"],
        env={},
    )


def _dry_run_no_selector_log_file(tmp_path: Path) -> LoopStepRuntime:
    return _make_runtime(
        tmp_path,
        runner_command=["runner"],
        env={},
        log_file=tmp_path / "test.log",
    )


def _dry_run_no_selector_explicit_prompt(tmp_path: Path) -> LoopStepRuntime:
    rp = ResolvedPrompt(source="inline text", effective_file=tmp_path / "inline.md")
    return _make_runtime(
        tmp_path,
        runner_command=["runner"],
        env={},
        resolved_prompt=rp,
    )


def _dry_run_with_bootstrap(tmp_path: Path) -> LoopStepRuntime:
    bootstrap_f = tmp_path / "select.md"
    bootstrap_f.write_text("bootstrap\n", encoding="utf-8")
    bp = PreparedPrompt(label="bootstrap", source_file=bootstrap_f, effective_file=bootstrap_f)
    return _make_runtime(
        tmp_path,
        runner_command=["runner"],
        env={},
        bootstrap_prompt=bp,
    )


class TestPrintDryRunFull:
    @pytest.mark.parametrize(
        ("build_runtime", "expected_snippets"),
        [
            pytest.param(
                _dry_run_selector_idle_timeout,
                ["idle timeout: 5.0s", "selector command: selector"],
                id="selector_idle_timeout",
            ),
            pytest.param(
                _dry_run_selector_implement_prompt,
                ["implement.md", "(default)"],
                id="selector_implement_prompt",
            ),
            pytest.param(
                _dry_run_selector_explicit_prompt,
                ["custom-prompt"],
                id="selector_explicit_prompt",
            ),
            pytest.param(
                _dry_run_no_selector_log_file,
                ["idle timeout: disabled", "test.log"],
                id="no_selector_log_file",
            ),
            pytest.param(
                _dry_run_no_selector_explicit_prompt,
                ["explicit prompt:"],
                id="no_selector_explicit_prompt",
            ),
            pytest.param(
                _dry_run_with_bootstrap,
                ["command [bootstrap]:"],
                id="bootstrap_prompt",
            ),
        ],
    )
    def test_print_dry_run_output(
        self,
        tmp_path: Path,
        build_runtime: Callable[[Path], LoopStepRuntime],
        expected_snippets: list[str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        runtime = build_runtime(tmp_path)
        print_dry_run(runtime)
        output = capsys.readouterr().out
        for snippet in expected_snippets:
            assert snippet in output
        # When a log file is configured, the dry-run output names its full path.
        if runtime.log_file is not None:
            assert str(runtime.log_file) in output


class TestPrepareRuntimeMissingPromptFiles:
    def test_exits_when_select_md_prompt_file_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """prepare_runtime exits when bootstrap prompt (select.md) is missing."""
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
        runtime = _make_runtime(
            tmp_path,
            select_invocation=invocation,
            loop_prompt=None,
            runner_command=["fake-runner"],
            env={},
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
            # 1st selector call: non-resolvable text → real selector_result → str → retry.
            # 2nd selector call: COMPLETE → real selector_result → None → done.
            if call_count == 1:
                return "not-a-file-path\n"
            return "COMPLETE\n"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        # Real selector_result: "not-a-file-path" is not a file → str (retry);
        # then "COMPLETE" on last line → None (done).
        result = execute_single_step(runtime, step_number=1)
        # After string result, selector retries; eventually COMPLETE returns True
        assert result is True
        # The str result must have triggered a retry: exactly two selector calls,
        # never a runner call (this branch never selects a task).
        assert call_count == 2


class TestExecuteSingleStepWithResolvedPrompt:
    def test_selector_mode_uses_resolved_prompt_for_runner_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When runtime has resolved_prompt and loop_prompt, runner uses loop_prompt."""
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
        loop_prompt = PreparedPrompt(
            label="loop", source_file=loop_file, effective_file=loop_file
        )

        runtime = _make_runtime(
            tmp_path,
            select_invocation=invocation,
            loop_prompt=loop_prompt,
            resolved_prompt=resolved_prompt,
            runner_command=["fake-runner"],
            env={},
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
            # Selector call: real selector_result resolves "task-1.md" → task_file.
            if command == ["fake-selector"]:
                return "task-1.md\n"
            return "output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        # Real selector_result: "task-1.md" found in tasks_dir → returns Path.
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
        loop_prompt = PreparedPrompt(
            label="loop", source_file=prompt_file_path, effective_file=prompt_file_path
        )

        runtime = _make_runtime(
            tmp_path,
            select_invocation=invocation,
            loop_prompt=loop_prompt,
            resolved_prompt=resolved_prompt,
            runner_command=["fake-runner"],
            env={},
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
            # Selector call: real selector_result resolves "task-1.md" → task_file.
            if command == ["fake-selector"]:
                return "task-1.md\n"
            return "output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        # Real selector_result: "task-1.md" found in tasks_dir → returns Path.
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
        loop_prompt = PreparedPrompt(
            label="loop", source_file=tmp_path / "stub", effective_file=tmp_path / "stub"
        )

        runtime = _make_runtime(
            tmp_path,
            select_invocation=invocation,
            loop_prompt=loop_prompt,
            resolved_prompt=resolved_prompt,
            runner_command=["fake-runner"],
            env=env_no_task,
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
            # Selector call: real selector_result resolves "task-1.md" → task_file.
            if command == ["fake-selector"]:
                return "task-1.md\n"
            return "output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        # Real selector_result: "task-1.md" found in tasks_dir → returns Path.
        result = execute_single_step(runtime, step_number=1)
        assert result is False
        # The runner target content should have TASK_FILE expanded
        runner_target = run_targets[-1]
        content = runner_target.read_text(encoding="utf-8")
        assert "${TASK_FILE}" not in content
        assert str(task_file) in content


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# commands/loop/step.py – cleanup_runtime
# ---------------------------------------------------------------------------


class TestCleanupRuntimeViaStep:
    def test_cleanup_runtime_delegates_to_cleanup_temp_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cleanup_runtime delegates to cleanup_temp_files."""
        f1 = tmp_path / "temp1.md"
        f1.write_text("temp", encoding="utf-8")

        runtime = _make_runtime(tmp_path)
        runtime.temp_files.append(f1)
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
            extra_prompt_source="EXTRA CONTENT",
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
            # Selector call: real selector_result resolves "task-1.md" → task_file.
            if command == ["fake-selector"]:
                return "task-1.md\n"
            return "output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        # Real selector_result: "task-1.md" found in tasks_dir → returns Path.
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
            extra_prompt_source="APPENDED EXTRA",
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
            # Selector call: real selector_result resolves "task-1.md" → task_file.
            if command == ["fake-selector"]:
                return "task-1.md\n"
            return "output"

        monkeypatch.setattr("agm.commands.loop.step.run_prompt_command", fake_run_command)
        # Real selector_result: "task-1.md" found in tasks_dir → returns Path.
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
