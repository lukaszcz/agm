"""Tests for the ``agm loop`` step, run, and select commands."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from agm.cli_support.args import LoopArgs, LoopCommandArgs
from agm.commands.loop.run import run as loop_run
from agm.commands.loop.select import run as select_run
from agm.commands.loop.step import (
    LoopStepRuntime,
    PreparedPrompt,
    _write_stream,
    cleanup_runtime,
    prepare_runtime,
    print_dry_run,
)
from agm.commands.loop.step import run as step_run
from agm.core import dry_run
from agm.core.log import append_log, prepare_log_file, resolve_log_file
from tests._git_helpers import init_repo
from tests._process_helpers import (
    AgentCall,
    AgentReply,
    AgentTimeout,
    FakeAgent,
    Reply,
    fake_agent,
)


def _raising_agent(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> FakeAgent:
    """Install an agent boundary whose process launch raises *error*."""

    def respond(_call: AgentCall) -> Reply:
        raise error

    return fake_agent(monkeypatch, respond)


# ---------------------------------------------------------------------------
# Loop fixtures
# ---------------------------------------------------------------------------


DEFAULT_PROMPTS = {
    "loop.md": "work on the tasks in %{TASKS_DIR}\n",
    "select.md": "select a task from %{TASKS_DIR}\n",
    "implement.md": "implement @%{TASK_FILE}\n",
}


def _loop_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompts: Mapping[str, str] | None = None,
) -> Path:
    """Set up an isolated home with loop prompts and work in *tmp_path*."""
    home = tmp_path / "home"
    prompt_dir = home / ".agm" / "prompts"
    prompt_dir.mkdir(parents=True)
    for name, text in (DEFAULT_PROMPTS if prompts is None else prompts).items():
        (prompt_dir / name).write_text(text, encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    # Runner and selector executables are looked up on PATH; no real agent
    # binary may be involved, so resolution answers for any name.
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    return home


def _isolate_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Stop git's repository search from escaping *tmp_path*.

    Default log files are placed at the containing git root, which AGM finds
    by running git for real.  Capping the search keeps the answer the same
    whether or not the directory holding the suite's temporary files happens
    to sit inside somebody's checkout, while a repository created inside
    *tmp_path* is still found normally.
    """
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))


def _tasks_dir(tmp_path: Path) -> Path:
    """The default tasks directory for a loop run inside *tmp_path*."""
    return tmp_path / ".agent-files" / "tasks"


def _write_progress(tmp_path: Path) -> Path:
    """Create the progress file, which tells the loop it needs no bootstrap."""
    progress = _tasks_dir(tmp_path) / "PROGRESS.md"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text("in progress\n", encoding="utf-8")
    return progress


def _write_task(tmp_path: Path, name: str = "task-1.md") -> Path:
    """Create a selectable task file in the default tasks directory."""
    task = _tasks_dir(tmp_path) / name
    task.parent.mkdir(parents=True, exist_ok=True)
    task.write_text("do the task\n", encoding="utf-8")
    return task


def _make_loop_args(
    *,
    no_log: bool = True,
    log_file: str | None = None,
    runner: str | None = "fake-runner",
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


def _make_select_args(
    *,
    runner: str | None = "fake-runner",
    runner_args: list[str] | None = None,
    selector: str | None = "fake-selector",
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
) -> LoopCommandArgs:
    return LoopCommandArgs(
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


def _selector_args(**overrides: object) -> LoopArgs:
    """Loop arguments in selector mode, with a distinct selector command."""
    fields: dict[str, object] = {"no_selector": False, "selector": "fake-selector"}
    fields.update(overrides)
    return _make_loop_args(**fields)


# ---------------------------------------------------------------------------
# Log files
# ---------------------------------------------------------------------------


class TestLogFile:
    @pytest.mark.parametrize("log_file", [None, "ignored.log"], ids=["default", "explicit"])
    def test_logging_disabled_yields_no_log_file(self, log_file: str | None) -> None:
        assert resolve_log_file(command_name="loop", enabled=False, log_file=log_file) is None

    def test_explicit_log_file_is_used_as_given(self, tmp_path: Path) -> None:
        explicit = tmp_path / "my.log"
        result = resolve_log_file(command_name="loop", enabled=True, log_file=str(explicit))
        assert result == explicit

    def test_relative_log_file_resolves_against_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)

        result = resolve_log_file(command_name="loop", enabled=True, log_file="logs/run.log")

        assert result == work / "logs" / "run.log"

    def test_default_log_file_lands_in_the_working_directory_outside_a_repository(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _isolate_git(monkeypatch, tmp_path)

        result = resolve_log_file(command_name="loop", enabled=True, log_file=None)

        assert result is not None
        assert result.parent == tmp_path / ".agent-files"
        assert result.name.startswith("loop-")
        assert result.suffix == ".log"

    def test_default_log_file_lands_at_the_git_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
    ) -> None:
        checkout = init_repo(tmp_path / "checkout", env)
        nested = checkout / "src"
        nested.mkdir()
        monkeypatch.chdir(nested)
        _isolate_git(monkeypatch, tmp_path)

        result = resolve_log_file(command_name="refine", enabled=True, log_file=None)

        assert result is not None
        assert result.parent == checkout / ".agent-files"
        assert result.name.startswith("refine-")

    def test_preparing_a_log_file_creates_its_directory_and_names_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_file = tmp_path / ".agent-files" / "loop.log"

        prepare_log_file(log_file)

        assert log_file.parent.is_dir()
        assert str(log_file) in capsys.readouterr().out


class TestAppendLog:
    def test_appends_successive_chunks_creating_the_file(self, tmp_path: Path) -> None:
        log = tmp_path / "out.log"

        append_log(log, "first line\n")
        append_log(log, "second line\n")

        assert log.read_text(encoding="utf-8") == "first line\nsecond line\n"

    def test_nothing_is_written_without_a_log_file_or_content(self, tmp_path: Path) -> None:
        log = tmp_path / "out.log"

        append_log(None, "some content")
        append_log(log, "")

        assert not log.exists()


class TestWriteStream:
    @pytest.mark.parametrize("stderr", [False, True], ids=["stdout", "stderr"])
    def test_empty_chunks_are_not_written(
        self, capsys: pytest.CaptureFixture[str], stderr: bool
    ) -> None:
        _write_stream("", stderr=stderr)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


# ---------------------------------------------------------------------------
# A loop step without a selector
# ---------------------------------------------------------------------------


class TestStepWithoutSelector:
    def test_step_sends_the_default_loop_prompt_to_the_runner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        home = _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        agent = fake_agent(monkeypatch, {"fake-runner": ["still working\n"]})

        step_run(_make_loop_args())

        (call,) = agent.calls
        assert call.runner == ["fake-runner"]
        assert call.prompt == f"work on the tasks in {_tasks_dir(tmp_path)}\n"
        assert call.env["TASKS_DIR"] == str(_tasks_dir(tmp_path))
        # The rendered prompt is a temporary file, cleaned up after the step;
        # the prompt it was rendered from is left alone.
        assert not call.prompt_file.exists()
        assert (home / ".agm" / "prompts" / "loop.md").exists()
        out = capsys.readouterr().out
        assert f"Tasks dir: {Path('.agent-files') / 'tasks'}" in out
        assert "Step 1" in out
        assert "still working" in out

    def test_step_reports_completion_when_the_runner_is_done(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        fake_agent(monkeypatch, {"fake-runner": ["COMPLETE\n"]})

        step_run(_make_loop_args())

        assert "Completed." in capsys.readouterr().out

    def test_loop_repeats_steps_until_the_runner_completes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        agent = fake_agent(monkeypatch, {"fake-runner": ["working\n", "working\n", "COMPLETE\n"]})

        loop_run(_make_loop_args())

        assert len(agent.calls) == 3
        out = capsys.readouterr().out
        assert "Step 1" in out
        assert "Step 3" in out
        assert out.count("Tasks dir:") == 1
        assert "Completed." in out

    def test_a_single_step_does_not_repeat_when_the_runner_is_unfinished(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        agent = fake_agent(monkeypatch, {"fake-runner": ["working\n"]})

        step_run(_make_loop_args())

        assert len(agent.calls) == 1

    def test_runner_output_is_streamed_to_the_terminal_and_the_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        log_file = tmp_path / "loop.log"
        fake_agent(
            monkeypatch,
            {"fake-runner": [AgentReply(stdout="progress report\n", stderr="a warning\n")]},
        )

        step_run(_make_loop_args(no_log=False, log_file=str(log_file)))

        captured = capsys.readouterr()
        assert "progress report" in captured.out
        assert "a warning" in captured.err
        logged = log_file.read_text(encoding="utf-8")
        assert "Tasks dir:" in logged
        assert "Step 1" in logged
        assert "progress report" in logged
        assert "a warning" in logged

    def test_an_idle_runner_leaves_the_step_unfinished_and_is_logged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        log_file = tmp_path / "loop.log"
        fake_agent(monkeypatch, {"fake-runner": [AgentTimeout()]})

        step_run(_make_loop_args(no_log=False, log_file=str(log_file)))

        logged = log_file.read_text(encoding="utf-8")
        assert "Idle timeout" in logged
        assert "Completed." not in logged

    def test_a_failed_agent_launch_is_logged_before_it_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        log_file = tmp_path / "loop.log"
        _raising_agent(monkeypatch, ValueError("runner disappeared"))

        with pytest.raises(ValueError):
            step_run(_make_loop_args(no_log=False, log_file=str(log_file)))

        assert "runner disappeared" in log_file.read_text(encoding="utf-8")

    def test_an_explicit_prompt_file_replaces_the_default_loop_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        custom = tmp_path / "custom.md"
        custom.write_text("custom work in %{TASKS_DIR}\n", encoding="utf-8")
        agent = fake_agent(monkeypatch, {"fake-runner": ["COMPLETE\n"]})

        step_run(_make_loop_args(prompt_file=str(custom)))

        assert agent.prompts == [f"custom work in {_tasks_dir(tmp_path)}\n"]

    def test_an_inline_prompt_replaces_the_default_loop_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        agent = fake_agent(monkeypatch, {"fake-runner": ["COMPLETE\n"]})

        step_run(_make_loop_args(prompt="inline work in %{TASKS_DIR}"))

        assert agent.prompts == [f"inline work in {_tasks_dir(tmp_path)}"]

    def test_an_extra_prompt_is_appended_to_the_loop_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        agent = fake_agent(monkeypatch, {"fake-runner": ["COMPLETE\n"]})

        step_run(_make_loop_args(extra_prompt="also mind %{TASKS_DIR}"))

        (prompt,) = agent.prompts
        assert prompt.startswith("work on the tasks in")
        assert prompt.endswith(f"also mind {_tasks_dir(tmp_path)}")

    def test_the_runner_command_is_expanded_at_launch_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        monkeypatch.setenv("LOOP_RUNNER", "fake-runner")
        agent = fake_agent(monkeypatch, {"fake-runner": ["COMPLETE\n"]})

        step_run(_make_loop_args(runner="%{LOOP_RUNNER}", runner_args=["--fast"]))

        assert agent.calls[0].runner == ["fake-runner", "--fast"]

    def test_a_custom_tasks_directory_reaches_the_prompt_and_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        tasks = tmp_path / "my-tasks"
        tasks.mkdir()
        (tasks / "PROGRESS.md").write_text("in progress\n", encoding="utf-8")
        agent = fake_agent(monkeypatch, {"fake-runner": ["COMPLETE\n"]})

        step_run(_make_loop_args(tasks_dir="my-tasks"))

        (call,) = agent.calls
        assert call.env["TASKS_DIR"] == str(tasks)
        assert str(tasks) in call.prompt


class TestBootstrap:
    def test_the_bootstrap_prompt_runs_once_before_the_first_step(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        agent = fake_agent(monkeypatch, {"fake-runner": ["bootstrapped\n", "COMPLETE\n"]})

        step_run(_make_loop_args())

        assert agent.prompts == [
            f"select a task from {_tasks_dir(tmp_path)}\n",
            f"work on the tasks in {_tasks_dir(tmp_path)}\n",
        ]

    def test_no_bootstrap_runs_once_the_progress_file_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        agent = fake_agent(monkeypatch, {"fake-runner": ["COMPLETE\n"]})

        step_run(_make_loop_args())

        assert len(agent.calls) == 1

    def test_an_idle_bootstrap_agent_does_not_abort_the_step(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        agent = fake_agent(monkeypatch, {"fake-runner": [AgentTimeout(), "COMPLETE\n"]})

        step_run(_make_loop_args())

        assert len(agent.calls) == 2
        assert "Completed." in capsys.readouterr().out

    def test_a_failed_bootstrap_launch_is_logged_before_it_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        log_file = tmp_path / "loop.log"
        _raising_agent(monkeypatch, ValueError("runner disappeared"))

        with pytest.raises(ValueError):
            step_run(_make_loop_args(no_log=False, log_file=str(log_file)))

        assert "runner disappeared" in log_file.read_text(encoding="utf-8")

    def test_dry_run_describes_the_bootstrap_without_running_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        agent = fake_agent(monkeypatch, "COMPLETE\n")
        dry_run.set_enabled(True)

        step_run(_make_loop_args())

        assert agent.calls == []
        assert "command [bootstrap]:" in capsys.readouterr().out


class TestPromptValidation:
    def test_a_missing_loop_prompt_stops_the_step(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch, prompts={})
        agent = fake_agent(monkeypatch, "COMPLETE\n")

        with pytest.raises(SystemExit) as exc_info:
            step_run(_make_loop_args())

        assert exc_info.value.code == 1
        assert agent.calls == []

    def test_a_missing_bootstrap_prompt_stops_the_step(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch, prompts={"loop.md": "work\n"})
        agent = fake_agent(monkeypatch, "COMPLETE\n")

        with pytest.raises(SystemExit) as exc_info:
            step_run(_make_loop_args())

        assert exc_info.value.code == 1
        assert agent.calls == []

    def test_setup_diagnostics_are_recorded_in_the_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch, prompts={})
        log_file = tmp_path / "loop.log"

        with pytest.raises(SystemExit):
            prepare_runtime(_make_loop_args(no_log=False, log_file=str(log_file)))

        assert "loop.md" in log_file.read_text(encoding="utf-8")

    def test_a_loop_prompt_cannot_reference_the_task_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Without a selector no task is ever selected, so the hole cannot bind."""
        _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        prompt = tmp_path / "prompt.md"
        prompt.write_text("Implement %{TASK_FILE}\n", encoding="utf-8")
        agent = fake_agent(monkeypatch, "COMPLETE\n")

        with pytest.raises(SystemExit) as exc_info:
            step_run(_make_loop_args(prompt_file=str(prompt)))

        assert exc_info.value.code == 1
        assert agent.calls == []
        error = capsys.readouterr().err
        assert "prompt.md" in error
        assert "TASK_FILE" in error


# ---------------------------------------------------------------------------
# A loop step driven by a selector
# ---------------------------------------------------------------------------


class TestStepWithSelector:
    def test_the_selected_task_is_handed_to_the_runner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        task = _write_task(tmp_path)
        agent = fake_agent(
            monkeypatch,
            {"fake-selector": ["task-1.md\n"], "fake-runner": ["done\n"]},
        )

        step_run(_selector_args())

        assert agent.runners == ["fake-selector", "fake-runner"]
        assert agent.prompts_of("fake-selector") == [f"select a task from {_tasks_dir(tmp_path)}\n"]
        assert agent.prompts_of("fake-runner") == [f"implement @{task}\n"]
        assert agent.calls[1].env["TASK_FILE"] == str(task)
        assert "Selected task:" in capsys.readouterr().out

    def test_the_selector_ends_the_loop_when_no_task_is_left(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        agent = fake_agent(monkeypatch, {"fake-selector": ["COMPLETE\n"]})

        loop_run(_selector_args())

        assert len(agent.calls) == 1
        assert "Completed." in capsys.readouterr().out

    def test_the_selector_is_retried_until_it_names_a_real_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_task(tmp_path)
        agent = fake_agent(
            monkeypatch,
            {
                "fake-selector": ["thinking about it\n", "task-1.md\n"],
                "fake-runner": ["done\n"],
            },
        )

        step_run(_selector_args())

        assert agent.runners == ["fake-selector", "fake-selector", "fake-runner"]

    def test_an_idle_selector_is_retried_within_the_same_step(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        agent = fake_agent(monkeypatch, {"fake-selector": [AgentTimeout(), "COMPLETE\n"]})

        step_run(_selector_args())

        assert len(agent.calls) == 2
        assert "Completed." in capsys.readouterr().out

    def test_an_idle_runner_leaves_the_step_unfinished(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_task(tmp_path)
        agent = fake_agent(
            monkeypatch,
            {"fake-selector": ["task-1.md\n"], "fake-runner": [AgentTimeout()]},
        )

        step_run(_selector_args())

        assert len(agent.calls) == 2
        assert "Completed." not in capsys.readouterr().out

    def test_the_runner_command_selects_tasks_when_no_selector_command_is_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        task = _write_task(tmp_path)
        agent = fake_agent(monkeypatch, {"fake-runner": ["task-1.md\n", "done\n"]})

        step_run(_make_loop_args(no_selector=False))

        assert agent.runners == ["fake-runner", "fake-runner"]
        assert agent.prompts[1] == f"implement @{task}\n"

    @pytest.mark.parametrize(
        ("prompt_kind", "prompt_text"),
        [("file", "Work on %{TASK_FILE}\n"), ("inline", "Work on %{TASK_FILE}")],
    )
    def test_an_explicit_prompt_is_rendered_once_the_task_is_known(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        prompt_kind: str,
        prompt_text: str,
    ) -> None:
        """Rendering before selection would fail: ``TASK_FILE`` is unbound then."""
        _loop_home(tmp_path, monkeypatch, prompts={"select.md": "select\n"})
        task = _write_task(tmp_path)
        overrides: dict[str, object] = {}
        if prompt_kind == "file":
            prompt_file = tmp_path / "prompt.md"
            prompt_file.write_text(prompt_text, encoding="utf-8")
            overrides["prompt_file"] = str(prompt_file)
        else:
            overrides["prompt"] = prompt_text
        agent = fake_agent(
            monkeypatch,
            {"fake-selector": ["task-1.md\n"], "fake-runner": ["done\n"]},
        )

        step_run(_selector_args(**overrides))

        assert agent.prompts_of("fake-runner") == [prompt_text.replace("%{TASK_FILE}", str(task))]

    @pytest.mark.parametrize(
        "prompt_override",
        [{}, {"prompt": "Work on %{TASK_FILE}"}],
        ids=["implement-prompt", "explicit-prompt"],
    )
    def test_an_extra_prompt_is_appended_to_the_selected_task_prompt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        prompt_override: dict[str, object],
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        task = _write_task(tmp_path)
        agent = fake_agent(
            monkeypatch,
            {"fake-selector": ["task-1.md\n"], "fake-runner": ["done\n"]},
        )

        step_run(_selector_args(extra_prompt="also review %{TASK_FILE}", **prompt_override))

        (runner_prompt,) = agent.prompts_of("fake-runner")
        assert str(task) in runner_prompt
        assert runner_prompt.endswith(f"also review {task}")

    def test_an_extra_selector_prompt_is_appended_to_the_selector_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        agent = fake_agent(monkeypatch, {"fake-selector": ["COMPLETE\n"]})

        step_run(_selector_args(extra_selector_prompt="prefer the oldest task"))

        (selector_prompt,) = agent.prompts_of("fake-selector")
        assert selector_prompt.startswith("select a task from")
        assert selector_prompt.endswith("prefer the oldest task")

    def test_a_custom_selector_prompt_file_replaces_the_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        selector_prompt = tmp_path / "pick.md"
        selector_prompt.write_text("pick from %{TASKS_DIR}\n", encoding="utf-8")
        agent = fake_agent(monkeypatch, {"fake-selector": ["COMPLETE\n"]})

        step_run(_selector_args(selector_prompt_file=str(selector_prompt)))

        assert agent.prompts == [f"pick from {_tasks_dir(tmp_path)}\n"]

    def test_a_missing_implement_prompt_stops_the_step(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch, prompts={"select.md": "select\n"})
        agent = fake_agent(monkeypatch, "COMPLETE\n")

        with pytest.raises(SystemExit) as exc_info:
            step_run(_selector_args())

        assert exc_info.value.code == 1
        assert agent.calls == []

    def test_an_explicit_prompt_stands_in_for_a_missing_implement_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch, prompts={"select.md": "select\n"})
        task = _write_task(tmp_path)
        agent = fake_agent(
            monkeypatch,
            {"fake-selector": ["task-1.md\n"], "fake-runner": ["done\n"]},
        )

        step_run(_selector_args(prompt="Implement %{TASK_FILE}"))

        assert agent.prompts_of("fake-runner") == [f"Implement {task}"]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"prompt": "Fix %{TASK_FILEE}"},
            {"prompt": "Fix %{TASK_FILE", "dry_run": True},
            {"prompt": "Fix %{TASK_FILE}", "extra_prompt": "Also %{TASK_FILEE}"},
            {"implement_prompt": "Fix %{TASK_FILEE}"},
            {"prompt_file": "missing.md"},
            {"extra_prompt_file": "missing-extra.md", "dry_run": True},
        ],
        ids=[
            "typo-in-prompt",
            "unterminated-hole",
            "typo-in-extra-prompt",
            "typo-in-implement-prompt",
            "missing-prompt-file",
            "missing-extra-prompt-file",
        ],
    )
    def test_a_bad_runner_prompt_is_rejected_before_the_selector_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object]
    ) -> None:
        """The selector mutates the task files, so it must not run first."""
        options = dict(overrides)
        implement_prompt = options.pop("implement_prompt", None)
        prompts = dict(DEFAULT_PROMPTS)
        if isinstance(implement_prompt, str):
            prompts["implement.md"] = implement_prompt
        _loop_home(tmp_path, monkeypatch, prompts=prompts)
        if options.pop("dry_run", False):
            dry_run.set_enabled(True)
        for key in ("prompt_file", "extra_prompt_file"):
            value = options.get(key)
            if isinstance(value, str):
                options[key] = str(tmp_path / value)
        agent = fake_agent(monkeypatch, "COMPLETE\n")

        with pytest.raises(SystemExit) as exc_info:
            step_run(_selector_args(**options))

        assert exc_info.value.code == 1
        assert agent.calls == []

    def test_the_task_file_hole_is_accepted_before_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``TASK_FILE`` binds only after selection, but is a known name up front."""
        _loop_home(tmp_path, monkeypatch, prompts={"select.md": "select\n"})
        agent = fake_agent(monkeypatch, {"fake-selector": ["COMPLETE\n"]})

        step_run(_selector_args(prompt="Implement %{TASK_FILE}"))

        assert len(agent.calls) == 1


# ---------------------------------------------------------------------------
# Dry-run reporting
# ---------------------------------------------------------------------------


class TestDryRun:
    @pytest.mark.parametrize(
        ("overrides", "prompts", "expected_snippets"),
        [
            pytest.param(
                {"no_selector": False, "selector": "fake-selector", "timeout": 5.0},
                None,
                ["idle timeout: 5.0s", "selector command: fake-selector"],
                id="selector-idle-timeout",
            ),
            pytest.param(
                {"no_selector": False, "selector": "fake-selector"},
                None,
                ["runner prompt: ", "implement.md (default) (reprocessed after task selection)"],
                id="selector-implement-prompt",
            ),
            pytest.param(
                {"no_selector": False, "selector": "fake-selector", "prompt_file": "custom.md"},
                None,
                ["runner prompt: custom.md (reprocessed after task selection)"],
                id="selector-explicit-prompt",
            ),
            pytest.param(
                {"no_selector": False, "selector": "fake-selector", "prompt": "do %{TASK_FILE}"},
                None,
                [
                    "prompt [prompt]: inline prompt",
                    "runner prompt: inline prompt (reprocessed after task selection)",
                ],
                id="selector-inline-prompt",
            ),
            pytest.param(
                {"no_log": False, "log_file": "loop.log"},
                None,
                ["idle timeout: disabled", "log file: loop.log", "selector command: disabled"],
                id="no-selector-log-file",
            ),
            pytest.param(
                {"prompt": "inline work"},
                None,
                ["explicit prompt:", "loop-runner"],
                id="no-selector-explicit-prompt",
            ),
            pytest.param(
                {"bootstrap": True},
                {"loop.md": "work\n", "select.md": "select\n"},
                ["command [bootstrap]:", "prompt [bootstrap]:"],
                id="bootstrap-prompt",
            ),
        ],
    )
    def test_dry_run_reports_the_resolved_configuration(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        overrides: dict[str, object],
        prompts: dict[str, str] | None,
        expected_snippets: list[str],
    ) -> None:
        _loop_home(tmp_path, monkeypatch, prompts=prompts)
        options = dict(overrides)
        if "prompt_file" in options:
            prompt_file = tmp_path / str(options["prompt_file"])
            prompt_file.write_text("custom %{TASK_FILE}\n", encoding="utf-8")
            options["prompt_file"] = str(prompt_file)
        if not options.pop("bootstrap", False):
            # A progress file means the loop needs no bootstrap pass.
            _write_progress(tmp_path)
        agent = fake_agent(monkeypatch, "COMPLETE\n")
        dry_run.set_enabled(True)

        step_run(_make_loop_args(**options))

        out = capsys.readouterr().out
        for snippet in expected_snippets:
            assert snippet in out
        assert agent.calls == []

    def test_dry_run_does_not_render_the_runner_prompt_before_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The runner prompt depends on ``TASK_FILE``, unknown until selection."""
        _loop_home(tmp_path, monkeypatch, prompts={"select.md": "select\n"})
        prompt = tmp_path / "prompt.md"
        prompt.write_text("Implement %{TASK_FILE}\n", encoding="utf-8")
        fake_agent(monkeypatch, "COMPLETE\n")
        dry_run.set_enabled(True)

        runtime = prepare_runtime(_selector_args(prompt_file=str(prompt)))
        try:
            # No rendered prompt file exists yet; only its source is named.
            assert runtime.temp_files == []
            print_dry_run(runtime)
        finally:
            cleanup_runtime(runtime)

        out = capsys.readouterr().out
        assert "prompt [prompt]: prompt.md" in out
        assert "runner prompt: prompt.md" in out

    def test_dry_run_of_the_whole_loop_reports_and_runs_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        agent = fake_agent(monkeypatch, "COMPLETE\n")
        dry_run.set_enabled(True)

        loop_run(_make_loop_args())

        assert agent.calls == []
        out = capsys.readouterr().out
        assert "loop configuration" in out
        assert "command [runner]:" in out

    def test_dry_run_names_the_preprocessed_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        fake_agent(monkeypatch, "COMPLETE\n")
        dry_run.set_enabled(True)

        step_run(_make_loop_args())

        out = capsys.readouterr().out
        assert "loop.md -> " in out
        assert "(preprocessed)" in out

    def test_dry_run_names_a_prompt_that_needs_no_preprocessing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch, prompts={"loop.md": "work\n", "select.md": "select\n"})
        _write_progress(tmp_path)
        fake_agent(monkeypatch, "COMPLETE\n")
        dry_run.set_enabled(True)

        step_run(_make_loop_args())

        out = capsys.readouterr().out
        assert "prompt [loop]: " in out
        assert "(preprocessed)" not in out


# ---------------------------------------------------------------------------
# Cleanup and interruption
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_rendered_prompts_are_removed_when_the_agent_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        rendered: list[Path] = []

        def respond(call: AgentCall) -> Reply:
            rendered.append(call.prompt_file)
            raise OSError("runner disappeared")

        fake_agent(monkeypatch, respond)

        # A runner that cannot be spawned is a fatal configuration error.
        with pytest.raises(SystemExit) as exc_info:
            step_run(_make_loop_args())

        assert exc_info.value.code == 1

        assert rendered and not any(path.exists() for path in rendered)

    def test_cleanup_tolerates_already_deleted_prompt_files(self, tmp_path: Path) -> None:
        runtime = LoopStepRuntime(
            temp_files=[tmp_path / "gone.md"],
            resolved_tasks_dir=tmp_path,
            resolved_progress_file=tmp_path / "PROGRESS.md",
            env={},
            resolved_runner_command=["fake-runner"],
            selector=None,
            loop_prompt=PreparedPrompt(
                label="loop",
                source_file=tmp_path / "loop.md",
                effective_file=tmp_path / "loop.md",
            ),
            prompt_source=None,
            bootstrap_prompt=None,
            extra_prompt_source=None,
            log_file=None,
            idle_timeout=None,
        )

        cleanup_runtime(runtime)

    def test_a_failed_setup_leaves_no_cleanup_to_do(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setup can fail before a runtime exists; teardown must cope with that."""
        _loop_home(tmp_path, monkeypatch, prompts={})
        fake_agent(monkeypatch, "COMPLETE\n")

        with pytest.raises(SystemExit) as exc_info:
            step_run(_make_loop_args())

        assert exc_info.value.code == 1


class TestInterruption:
    @pytest.mark.parametrize("entry_point", [step_run, loop_run], ids=["step", "loop"])
    def test_an_interrupted_step_exits_with_the_interrupt_status(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        entry_point: Callable[[LoopArgs], None],
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _write_progress(tmp_path)
        log_file = tmp_path / "loop.log"
        _raising_agent(monkeypatch, KeyboardInterrupt())

        with pytest.raises(SystemExit) as exc_info:
            entry_point(_make_loop_args(no_log=False, log_file=str(log_file)))

        assert exc_info.value.code == 130
        assert "Interrupted" in capsys.readouterr().out
        assert "Interrupted" in log_file.read_text(encoding="utf-8")

    @pytest.mark.parametrize("entry_point", [step_run, loop_run], ids=["step", "loop"])
    def test_an_interrupt_during_setup_exits_with_the_interrupt_status(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        entry_point: Callable[[LoopArgs], None],
    ) -> None:
        """The bootstrap agent runs while the runtime is still being built."""
        _loop_home(tmp_path, monkeypatch)
        _raising_agent(monkeypatch, KeyboardInterrupt())

        with pytest.raises(SystemExit) as exc_info:
            entry_point(_make_loop_args())

        assert exc_info.value.code == 130


# ---------------------------------------------------------------------------
# agm loop select
# ---------------------------------------------------------------------------


class TestLoopSelect:
    def test_selection_requires_selector_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        agent = fake_agent(monkeypatch, "task-1.md\n")

        with pytest.raises(SystemExit) as exc_info:
            select_run(_make_select_args(no_selector=True))

        assert exc_info.value.code == 1
        assert agent.calls == []
        assert capsys.readouterr().err.strip()

    def test_the_selected_task_is_printed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        agent = fake_agent(monkeypatch, {"fake-selector": ["task-1.md\n"]})

        select_run(_make_select_args())

        (call,) = agent.calls
        assert call.prompt == f"select a task from {_tasks_dir(tmp_path)}\n"
        assert "task-1.md" in capsys.readouterr().out

    def test_the_runner_selects_when_no_selector_command_is_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        agent = fake_agent(monkeypatch, {"fake-runner": ["task-1.md\n"]})

        select_run(_make_select_args(selector=None))

        assert agent.runners == ["fake-runner"]

    def test_dry_run_reports_the_configuration_without_selecting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        agent = fake_agent(monkeypatch, "task-1.md\n")
        dry_run.set_enabled(True)

        select_run(_make_select_args())

        assert agent.calls == []
        out = capsys.readouterr().out
        assert "loop-select configuration" in out
        assert "selector command: fake-selector" in out
        assert "command [selector]:" in out

    def test_an_idle_selector_is_not_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        fake_agent(monkeypatch, {"fake-selector": [AgentTimeout()]})

        select_run(_make_select_args())

        assert "task" not in capsys.readouterr().out

    def test_an_interrupted_selection_exits_with_the_interrupt_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        _raising_agent(monkeypatch, KeyboardInterrupt())

        with pytest.raises(SystemExit) as exc_info:
            select_run(_make_select_args())

        assert exc_info.value.code == 130
        assert "Interrupted" in capsys.readouterr().out

    def test_rendered_prompts_are_removed_when_selection_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        rendered: list[Path] = []

        def respond(call: AgentCall) -> Reply:
            rendered.append(call.prompt_file)
            raise RuntimeError("boom")

        fake_agent(monkeypatch, respond)

        with pytest.raises(RuntimeError):
            select_run(_make_select_args())

        assert rendered and not any(path.exists() for path in rendered)

    def test_an_extra_selector_prompt_is_appended(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch)
        agent = fake_agent(monkeypatch, {"fake-selector": ["task-1.md\n"]})

        select_run(_make_select_args(extra_selector_prompt="prefer the oldest task"))

        (call,) = agent.calls
        assert call.prompt.startswith("select a task from")
        assert call.prompt.endswith("prefer the oldest task")

    def test_a_missing_selector_prompt_stops_the_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _loop_home(tmp_path, monkeypatch, prompts={})
        agent = fake_agent(monkeypatch, "task-1.md\n")

        with pytest.raises(SystemExit) as exc_info:
            select_run(_make_select_args())

        assert exc_info.value.code == 1
        assert agent.calls == []
