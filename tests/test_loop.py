"""Focused tests for loop helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agm.agent.runner import (
    command_with_prompt_target,
    prepare_prompt_from_source,
    run_prompt_command,
    split_command,
    validate_command,
)
from agm.commands.args import LoopArgs, LoopNextArgs
from agm.commands.loop.common import (
    is_complete_output,
    loop_env,
    loop_prompt_source,
    prepare_select_invocation,
    prompt_file,
    selector_prompt_source,
    selector_result,
    use_selector_mode,
)


def test_selector_result_accepts_relative_path_from_current_working_directory(
    tmp_path: Path, monkeypatch
) -> None:
    tasks_dir = tmp_path / ".agent-files" / "tasks"
    tasks_dir.mkdir(parents=True)
    task_file = tmp_path / "custom" / "tasks" / "task-1.md"
    task_file.parent.mkdir(parents=True)
    task_file.write_text("task one\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = selector_result("custom/tasks/task-1.md\n", tasks_dir=tasks_dir)

    assert result == task_file


def test_selector_result_accepts_absolute_path_from_selector(tmp_path: Path) -> None:
    tasks_dir = tmp_path / ".agent-files" / "tasks"
    tasks_dir.mkdir(parents=True)
    task_file = tmp_path / "custom" / "tasks" / "task-1.md"
    task_file.parent.mkdir(parents=True)
    task_file.write_text("task one\n", encoding="utf-8")

    result = selector_result(f"{task_file}\n", tasks_dir=tasks_dir)

    assert result == task_file


def test_selector_result_falls_back_to_tasks_dir_when_relative_path_is_not_in_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    tasks_dir = tmp_path / ".agent-files" / "tasks"
    tasks_dir.mkdir(parents=True)
    task_file = tasks_dir / "task-1.md"
    task_file.write_text("task one\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = selector_result("task-1.md\n", tasks_dir=tasks_dir)

    assert result == task_file


def test_selector_result_uses_only_last_output_line_for_task_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks_dir = tmp_path / ".agent-files" / "tasks"
    tasks_dir.mkdir(parents=True)
    task_file = tasks_dir / "task-1.md"
    task_file.write_text("task one\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = selector_result("progress update\n task-1.md \n", tasks_dir=tasks_dir)

    assert result == task_file


def test_selector_result_uses_only_last_output_line_for_complete(tmp_path: Path) -> None:
    tasks_dir = tmp_path / ".agent-files" / "tasks"
    tasks_dir.mkdir(parents=True)

    result = selector_result("progress update\n COMPLETE \n", tasks_dir=tasks_dir)

    assert result is None


def test_is_complete_output_uses_only_last_output_line() -> None:
    assert is_complete_output("progress update\n COMPLETE \n")
    assert not is_complete_output(" COMPLETE \nprogress update\n")


def test_prompt_file_prefers_home_over_install_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    prefix_prompt_dir = prefix / ".agm" / "prompts"
    prefix_prompt_dir.mkdir(parents=True)
    prefix_prompt = prefix_prompt_dir / "loop.md"
    prefix_prompt.write_text("prefix prompt\n", encoding="utf-8")

    home = tmp_path / "home"
    home_prompt_dir = home / ".agm" / "prompts"
    home_prompt_dir.mkdir(parents=True)
    home_prompt = home_prompt_dir / "loop.md"
    home_prompt.write_text("home prompt\n", encoding="utf-8")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("agm.config.general.agm_installation_prefix", lambda: prefix)

    assert prompt_file("loop.md") == home_prompt


def test_prepare_select_invocation_prefers_selector_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    prompt_dir = home / ".agm" / "prompts"
    prompt_dir.mkdir(parents=True)
    prompt_path = prompt_dir / "select.md"
    prompt_path.write_text("update $TASKS_DIR\n", encoding="utf-8")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    monkeypatch.chdir(tmp_path)

    args = LoopNextArgs(
        command_name=None,
        runner="runner --print",
        runner_args=[],
        selector="selector",
        no_selector=False,
        tasks_dir="custom/tasks",
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
    env = {"TASKS_DIR": str(tmp_path / "custom" / "tasks")}

    invocation = prepare_select_invocation(args, temp_files=[], env=env)

    assert invocation.command == ["selector"]
    assert invocation.command_kind == "selector"
    assert invocation.source_prompt_file == prompt_path
    assert invocation.effective_prompt_file.read_text(encoding="utf-8") == (
        f"update {tmp_path / 'custom' / 'tasks'}\n"
    )


def test_prepare_select_invocation_falls_back_to_runner_without_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    prompt_dir = home / ".agm" / "prompts"
    prompt_dir.mkdir(parents=True)
    prompt_path = prompt_dir / "select.md"
    prompt_path.write_text("update progress\n", encoding="utf-8")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    monkeypatch.chdir(tmp_path)

    args = LoopNextArgs(
        command_name=None,
        runner="runner --print",
        runner_args=["--verbose"],
        selector=None,
        no_selector=False,
        tasks_dir=None,
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

    invocation = prepare_select_invocation(args, temp_files=[], env={})

    assert invocation.command == ["runner", "--print", "--verbose"]
    assert invocation.command_kind == "runner"
    assert invocation.source_prompt_file == prompt_path
    assert invocation.effective_prompt_file == prompt_path


def test_use_selector_mode_is_default_when_no_flags_or_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    args = LoopArgs(
        command_name=None,
        runner=None,
        runner_args=[],
        selector=None,
        no_selector=False,
        tasks_dir=None,
        no_log=False,
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
    assert use_selector_mode(args) is True


def test_use_selector_mode_is_disabled_by_cli_no_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    args = LoopArgs(
        command_name=None,
        runner=None,
        runner_args=[],
        selector=None,
        no_selector=True,
        tasks_dir=None,
        no_log=False,
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
    assert use_selector_mode(args) is False


def test_use_selector_mode_is_disabled_by_config_no_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text('[loop]\nno_selector = true\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    args = LoopArgs(
        command_name=None,
        runner=None,
        runner_args=[],
        selector=None,
        no_selector=False,
        tasks_dir=None,
        no_log=False,
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
    assert use_selector_mode(args) is False


def test_use_selector_mode_cli_no_selector_overrides_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Config doesn't set no_selector, but CLI does
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    args = LoopNextArgs(
        command_name=None,
        runner=None,
        runner_args=[],
        selector=None,
        no_selector=True,
        tasks_dir=None,
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
    assert use_selector_mode(args) is False


class TestResolvePromptSource:
    def test_returns_none_when_no_prompt_specified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
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
        assert loop_prompt_source(args) is None

    def test_returns_prompt_text_from_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt="do the thing",
            prompt_file=None,
            selector_prompt=None,
            selector_prompt_file=None,
        extra_prompt=None,
        extra_prompt_file=None,
        extra_selector_prompt=None,
        extra_selector_prompt_file=None,
        timeout=None,
    )
        result = loop_prompt_source(args)
        assert result == "do the thing"

    def test_returns_path_from_cli_prompt_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt=None,
            prompt_file="/path/to/prompt.md",
            selector_prompt=None,
            selector_prompt_file=None,
        extra_prompt=None,
        extra_prompt_file=None,
        extra_selector_prompt=None,
        extra_selector_prompt_file=None,
        timeout=None,
    )
        result = loop_prompt_source(args)
        assert result == Path("/path/to/prompt.md")

    def test_cli_prompt_file_relative_path_resolves_against_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt=None,
            prompt_file="prompts/prompt.md",
            selector_prompt=None,
            selector_prompt_file=None,
        extra_prompt=None,
        extra_prompt_file=None,
        extra_selector_prompt=None,
        extra_selector_prompt_file=None,
        timeout=None,
    )
        result = loop_prompt_source(args)
        assert result == work / "prompts" / "prompt.md"

    def test_cli_prompt_overrides_prompt_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt="inline text",
            prompt_file="/path/to/prompt.md",
            selector_prompt=None,
            selector_prompt_file=None,
        extra_prompt=None,
        extra_prompt_file=None,
        extra_selector_prompt=None,
        extra_selector_prompt_file=None,
        timeout=None,
    )
        result = loop_prompt_source(args)
        assert result == "inline text"

    def test_returns_prompt_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[loop]\nprompt = "config prompt"\n')
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
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
        result = loop_prompt_source(args)
        assert result == "config prompt"

    def test_returns_prompt_file_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[loop]\nprompt_file = "my-prompt.md"\n'
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
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
        result = loop_prompt_source(args)
        assert result == tmp_path / "my-prompt.md"

    def test_cli_overrides_config_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[loop]\nprompt = "config prompt"\n')
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt="cli text",
            prompt_file=None,
            selector_prompt=None,
            selector_prompt_file=None,
        extra_prompt=None,
        extra_prompt_file=None,
        extra_selector_prompt=None,
        extra_selector_prompt_file=None,
        timeout=None,
    )
        result = loop_prompt_source(args)
        assert result == "cli text"


class TestPreparePromptFromSource:
    def test_inline_text_creates_temp_file_with_env_expansion(self, tmp_path: Path) -> None:
        temp_files: list[Path] = []
        env = {"MY_VAR": "hello"}

        resolved = prepare_prompt_from_source(
            "greet $MY_VAR world", temp_files=temp_files, env=env
        )

        assert isinstance(resolved.source, str)
        assert resolved.source == "greet $MY_VAR world"
        assert resolved.effective_file.read_text(encoding="utf-8") == "greet hello world"
        assert resolved.effective_file in temp_files
        # Clean up
        resolved.effective_file.unlink()

    def test_inline_text_without_env_vars_still_creates_temp_file(self, tmp_path: Path) -> None:
        temp_files: list[Path] = []
        env: dict[str, str] = {}

        resolved = prepare_prompt_from_source(
            "no vars here", temp_files=temp_files, env=env
        )

        assert isinstance(resolved.source, str)
        assert resolved.effective_file.read_text(encoding="utf-8") == "no vars here"
        assert resolved.effective_file in temp_files
        resolved.effective_file.unlink()

    def test_file_path_processes_existing_file(self, tmp_path: Path) -> None:
        prompt_path = tmp_path / "custom-prompt.md"
        prompt_path.write_text("task: $TASK_VAR", encoding="utf-8")

        temp_files: list[Path] = []
        env = {"TASK_VAR": "testing"}

        resolved = prepare_prompt_from_source(
            prompt_path, temp_files=temp_files, env=env
        )

        assert resolved.source == prompt_path
        assert resolved.effective_file.read_text(encoding="utf-8") == "task: testing"
        assert resolved.effective_file in temp_files

    def test_file_path_without_vars_reuses_original(self, tmp_path: Path) -> None:
        prompt_path = tmp_path / "static-prompt.md"
        prompt_path.write_text("static content", encoding="utf-8")

        temp_files: list[Path] = []
        env: dict[str, str] = {}

        resolved = prepare_prompt_from_source(
            prompt_path, temp_files=temp_files, env=env
        )

        assert resolved.source == prompt_path
        assert resolved.effective_file == prompt_path
        assert resolved.effective_file not in temp_files

    def test_inline_text_expands_tasks_dir_from_loop_env(self, tmp_path: Path) -> None:
        temp_files: list[Path] = []
        env = loop_env(Path("/tmp/tasks"))

        resolved = prepare_prompt_from_source(
            "look in $TASKS_DIR for tasks", temp_files=temp_files, env=env
        )

        assert isinstance(resolved.source, str)
        assert resolved.effective_file.read_text(encoding="utf-8") == "look in /tmp/tasks for tasks"
        resolved.effective_file.unlink()

    def test_file_path_expands_tasks_dir_from_loop_env(self, tmp_path: Path) -> None:
        prompt_path = tmp_path / "custom-prompt.md"
        prompt_path.write_text("dir=$TASKS_DIR", encoding="utf-8")

        temp_files: list[Path] = []
        env = loop_env(Path("/tmp/tasks"))

        resolved = prepare_prompt_from_source(
            prompt_path, temp_files=temp_files, env=env
        )

        assert resolved.effective_file.read_text(encoding="utf-8") == "dir=/tmp/tasks"

    def test_missing_file_exits_with_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist.md"
        temp_files: list[Path] = []

        with pytest.raises(SystemExit):
            prepare_prompt_from_source(missing, temp_files=temp_files, env={})


class TestLoopEnv:
    def test_includes_tasks_dir(self) -> None:
        env = loop_env(Path("/tmp/tasks"))
        assert env["TASKS_DIR"] == "/tmp/tasks"

    def test_includes_task_file_when_provided(self) -> None:
        env = loop_env(Path("/tmp/tasks"), task_file=Path("/tmp/tasks/task1.md"))
        assert env["TASK_FILE"] == "/tmp/tasks/task1.md"

    def test_no_task_file_by_default(self) -> None:
        env = loop_env(Path("/tmp/tasks"))
        assert "TASK_FILE" not in env


class TestResolveSelectorPromptSource:
    def test_returns_none_when_no_selector_prompt_specified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
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
        assert selector_prompt_source(args) is None

    def test_returns_selector_prompt_text_from_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt=None,
            prompt_file=None,
            selector_prompt="custom selector prompt",
            selector_prompt_file=None,
        extra_prompt=None,
        extra_prompt_file=None,
        extra_selector_prompt=None,
        extra_selector_prompt_file=None,
        timeout=None,
    )
        result = selector_prompt_source(args)
        assert result == "custom selector prompt"

    def test_returns_path_from_cli_selector_prompt_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt=None,
            prompt_file=None,
            selector_prompt=None,
            selector_prompt_file="/path/to/selector-prompt.md",
            extra_prompt=None,
            extra_prompt_file=None,
            extra_selector_prompt=None,
            extra_selector_prompt_file=None,
            timeout=None,
        )
        result = selector_prompt_source(args)
        assert result == Path("/path/to/selector-prompt.md")

    def test_cli_selector_prompt_file_relative_path_resolves_against_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt=None,
            prompt_file=None,
            selector_prompt=None,
            selector_prompt_file="prompts/selector.md",
        extra_prompt=None,
        extra_prompt_file=None,
        extra_selector_prompt=None,
        extra_selector_prompt_file=None,
        timeout=None,
    )
        result = selector_prompt_source(args)
        assert result == work / "prompts" / "selector.md"

    def test_cli_selector_prompt_overrides_selector_prompt_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt=None,
            prompt_file=None,
            selector_prompt="inline text",
            selector_prompt_file="/path/to/selector-prompt.md",
            extra_prompt=None,
            extra_prompt_file=None,
            extra_selector_prompt=None,
            extra_selector_prompt_file=None,
            timeout=None,
        )
        result = selector_prompt_source(args)
        assert result == "inline text"

    def test_returns_selector_prompt_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[loop]\nselector_prompt = "config selector prompt"\n'
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
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
        result = selector_prompt_source(args)
        assert result == "config selector prompt"

    def test_returns_selector_prompt_file_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[loop]\nselector_prompt_file = "my-selector-prompt.md"\n'
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
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
        result = selector_prompt_source(args)
        assert result == tmp_path / "my-selector-prompt.md"

    def test_cli_overrides_config_selector_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[loop]\nselector_prompt = "config selector prompt"\n'
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt=None,
            prompt_file=None,
            selector_prompt="cli selector text",
            selector_prompt_file=None,
        extra_prompt=None,
        extra_prompt_file=None,
        extra_selector_prompt=None,
        extra_selector_prompt_file=None,
        timeout=None,
    )
        result = selector_prompt_source(args)
        assert result == "cli selector text"

    def test_absolute_config_path_is_not_prefixed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[loop]\nselector_prompt_file = "/absolute/selector-prompt.md"\n'
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
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
        result = selector_prompt_source(args)
        assert result == Path("/absolute/selector-prompt.md")


class TestPrepareProgressInvocationSelectorPrompt:
    def test_selector_prompt_overrides_default_prompt_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        prompt_dir = home / ".agm" / "prompts"
        prompt_dir.mkdir(parents=True)
        default_prompt = prompt_dir / "select.md"
        default_prompt.write_text("default update\n", encoding="utf-8")

        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        args = LoopNextArgs(
            command_name=None,
            runner="runner",
            runner_args=[],
            selector="selector",
            no_selector=False,
            tasks_dir=None,
            prompt=None,
            prompt_file=None,
            selector_prompt="custom selector $TASKS_DIR",
            selector_prompt_file=None,
        extra_prompt=None,
        extra_prompt_file=None,
        extra_selector_prompt=None,
        extra_selector_prompt_file=None,
        timeout=None,
    )
        env = {"TASKS_DIR": "/tmp/tasks"}

        invocation = prepare_select_invocation(args, temp_files=[], env=env)

        assert invocation.source_prompt_file != default_prompt
        assert invocation.effective_prompt_file.read_text(encoding="utf-8") == (
            "custom selector /tmp/tasks"
        )

    def test_selector_prompt_file_overrides_default_prompt_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        prompt_dir = home / ".agm" / "prompts"
        prompt_dir.mkdir(parents=True)
        default_prompt = prompt_dir / "select.md"
        default_prompt.write_text("default update\n", encoding="utf-8")

        custom_prompt = tmp_path / "custom-selector.md"
        custom_prompt.write_text("select task from $TASKS_DIR\n", encoding="utf-8")

        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        args = LoopNextArgs(
            command_name=None,
            runner="runner",
            runner_args=[],
            selector="selector",
            no_selector=False,
            tasks_dir=None,
            prompt=None,
            prompt_file=None,
            selector_prompt=None,
            selector_prompt_file=str(custom_prompt),
            extra_prompt=None,
            extra_prompt_file=None,
            extra_selector_prompt=None,
            extra_selector_prompt_file=None,
            timeout=None,
        )
        env = {"TASKS_DIR": "/my/tasks"}

        invocation = prepare_select_invocation(args, temp_files=[], env=env)

        assert invocation.source_prompt_file == custom_prompt
        assert invocation.effective_prompt_file.read_text(encoding="utf-8") == (
            "select task from /my/tasks\n"
        )

    def test_no_selector_prompt_uses_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        prompt_dir = home / ".agm" / "prompts"
        prompt_dir.mkdir(parents=True)
        default_prompt = prompt_dir / "select.md"
        default_prompt.write_text("default update $TASKS_DIR\n", encoding="utf-8")

        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        args = LoopNextArgs(
            command_name=None,
            runner="runner",
            runner_args=[],
            selector="selector",
            no_selector=False,
            tasks_dir=None,
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
        env = {"TASKS_DIR": "/default/tasks"}

        invocation = prepare_select_invocation(args, temp_files=[], env=env)

        assert invocation.source_prompt_file == default_prompt
        assert invocation.effective_prompt_file.read_text(encoding="utf-8") == (
            "default update /default/tasks\n"
        )

class TestSplitCommandEmpty:
    def test_empty_command_exits(self) -> None:

        with pytest.raises(SystemExit) as exc_info:
            split_command("", kind="runner")
        assert exc_info.value.code == 1

    def test_whitespace_only_command_exits(self) -> None:

        with pytest.raises(SystemExit) as exc_info:
            split_command("   ", kind="selector")
        assert exc_info.value.code == 1


class TestTasksDirRelativePath:
    def test_relative_tasks_dir_resolved_against_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.commands.loop.common import tasks_dir as _tasks_dir

        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir="custom/tasks",
            no_log=False,
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
        result = _tasks_dir(args)
        assert result == tmp_path / "custom" / "tasks"

    def test_relative_tasks_dir_prefixed_with_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When tasks_dir is a relative path, it is joined with cwd."""
        from agm.commands.loop.common import tasks_dir as _tasks_dir

        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir="relative/tasks",
            no_log=False,
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
        result = _tasks_dir(args)
        assert result == tmp_path / "relative" / "tasks"


# ---------------------------------------------------------------------------
# commands/loop/common.py – line 126: validate_command error when not in PATH
# ---------------------------------------------------------------------------


class TestValidateCommandNotFound:
    def test_validate_command_exits_when_not_found(self) -> None:

        with pytest.raises(SystemExit) as exc_info:
            validate_command(["nonexistent-command-xyz123"], kind="runner")
        assert exc_info.value.code == 1


class TestCommandWithPromptTarget:
    def test_replaces_percent_percent_placeholder(self) -> None:

        result = command_with_prompt_target(["runner", "%%"], Path("/tmp/prompt.md"))
        assert result == ["runner", "/tmp/prompt.md"]

    def test_replaces_prompt_file_placeholder(self) -> None:

        result = command_with_prompt_target(
            ["runner", "%{PROMPT_FILE}"], Path("/tmp/prompt.md")
        )
        assert result == ["runner", "/tmp/prompt.md"]

    def test_appends_at_target_when_no_placeholder(self) -> None:

        result = command_with_prompt_target(["runner"], Path("/tmp/prompt.md"))
        assert result == ["runner", "@/tmp/prompt.md"]

    def test_replaced_true_returns_modified_command(self, tmp_path: Path) -> None:

        target = tmp_path / "prompt.md"
        result = command_with_prompt_target(["runner", "--input", "%%", "--flag"], target)
        assert result == ["runner", "--input", str(target), "--flag"]


class TestSelectorResultEdgeCases:
    def test_returns_empty_string_when_selected_is_empty(self) -> None:

        result = selector_result("", tasks_dir=Path("/tmp/tasks"))
        assert result == ""

    def test_returns_none_when_complete(self) -> None:

        result = selector_result("COMPLETE", tasks_dir=Path("/tmp/tasks"))
        assert result is None

    def test_returns_str_for_nonexistent_absolute_path(self) -> None:

        result = selector_result("/nonexistent/path.md", tasks_dir=Path("/tmp/tasks"))
        assert result == "/nonexistent/path.md"
        assert isinstance(result, str)

    def test_returns_str_when_not_found_anywhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        monkeypatch.chdir(tmp_path)
        result = selector_result("missing-task.md", tasks_dir=tmp_path / "tasks")
        assert result == "missing-task.md"
        assert isinstance(result, str)

    def test_relative_path_found_in_tasks_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        task_file = tasks_dir / "task-1.md"
        task_file.write_text("task", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = selector_result("task-1.md\n", tasks_dir=tasks_dir)
        assert result == task_file


class TestRunCommandOutputAssembly:
    def test_run_command_returns_ordered_output_with_callbacks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When callbacks are used, ordered_output is returned."""

        # Patch run_capture to simulate callbacks being invoked
        def fake_run_capture(cmd, *, env, stdout_callback, stderr_callback, **kwargs):
            # Simulate callbacks being called
            if stdout_callback is not None:
                stdout_callback("stdout chunk\n")
            if stderr_callback is not None:
                stderr_callback("stderr chunk\n")
            return (0, "", "")

        monkeypatch.setattr("agm.agent.runner.run_capture", fake_run_capture)

        target = tmp_path / "prompt.md"
        target.write_text("test", encoding="utf-8")
        result = run_prompt_command(["cmd"], target, env={})
        assert "stdout chunk" in result
        assert "stderr chunk" in result

    def test_run_command_returns_stdout_when_no_callbacks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When no callbacks produce output, stdout+stderr is returned."""

        def fake_run_capture(cmd, *, env, stdout_callback, stderr_callback, **kwargs):
            # Don't invoke callbacks - they'll be None anyway without real IO
            return (0, "stdout text", "stderr text")

        monkeypatch.setattr("agm.agent.runner.run_capture", fake_run_capture)

        target = tmp_path / "prompt.md"
        target.write_text("test", encoding="utf-8")
        result = run_prompt_command(["cmd"], target, env={})
        assert result == "stdout textstderr text"


# ---------------------------------------------------------------------------
# loop/step.py – additional coverage gaps
# ---------------------------------------------------------------------------


class TestValidateCommandNotInPath:
    def test_validate_command_exits_for_missing_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """validate_command exits when shutil.which returns None."""
        monkeypatch.setattr("shutil.which", lambda _: None)
        with pytest.raises(SystemExit) as exc_info:
            validate_command(["missing-cmd"], kind="runner")
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# commands/loop/common.py – lines 270-271, 301, 306: run_command output assembly
# ---------------------------------------------------------------------------


class TestRunCommandOutputAssemblyFull:
    def test_run_command_with_stdout_callback_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_command with only stdout_callback still assembles output."""
        target = tmp_path / "prompt.md"
        target.write_text("prompt", encoding="utf-8")

        captured_stdout: list[str] = []

        def fake_run_capture(
            cmd: list[str],
            *,
            env: dict[str, str],
            stdout_callback: Any = None,
            stderr_callback: Any = None,
            isolate_process_group: bool = False,
            idle_timeout: float | None = None,
        ) -> tuple[int, str, str]:
            if stdout_callback is not None:
                stdout_callback("hello")
            return (0, "", "")

        monkeypatch.setattr(
            "agm.agent.runner.run_capture", fake_run_capture
        )

        output = run_prompt_command(
            ["runner"], target, env={}, stdout_callback=lambda c: captured_stdout.append(c)
        )
        assert output == "hello"
        assert captured_stdout == ["hello"]

    def test_run_command_no_ordered_output_returns_stdout_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_command without callbacks uses stdout + stderr from run_capture."""
        target = tmp_path / "prompt.md"
        target.write_text("prompt", encoding="utf-8")

        def fake_run_capture(
            cmd: list[str],
            *,
            env: dict[str, str],
            stdout_callback: Any = None,
            stderr_callback: Any = None,
            isolate_process_group: bool = False,
            idle_timeout: float | None = None,
        ) -> tuple[int, str, str]:
            return (0, "just-stdout", "")

        monkeypatch.setattr(
            "agm.agent.runner.run_capture", fake_run_capture
        )

        output = run_prompt_command(["runner"], target, env={})
        assert output == "just-stdout"

    def test_run_command_no_ordered_output_with_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_command appends stderr to stdout when no ordered_output."""
        target = tmp_path / "prompt.md"
        target.write_text("prompt", encoding="utf-8")

        def fake_run_capture(
            cmd: list[str],
            *,
            env: dict[str, str],
            stdout_callback: Any = None,
            stderr_callback: Any = None,
            isolate_process_group: bool = False,
            idle_timeout: float | None = None,
        ) -> tuple[int, str, str]:
            return (0, "the-stdout", "the-stderr")

        monkeypatch.setattr(
            "agm.agent.runner.run_capture", fake_run_capture
        )

        output = run_prompt_command(["runner"], target, env={})
        assert output == "the-stdoutthe-stderr"


# ---------------------------------------------------------------------------
# commands/loop/common.py – lines 327, 338, 346, 354: selector_result edge cases
# ---------------------------------------------------------------------------


class TestSelectorResultAdditionalEdgeCases:
    def test_empty_output_returns_empty_string(self) -> None:
        """selector_result returns empty string for empty output."""
        result = selector_result("", tasks_dir=Path("/tmp/tasks"))
        assert result == ""

    def test_absolute_path_not_a_file_returns_string(self) -> None:
        """selector_result returns raw string when absolute path is not a file."""
        result = selector_result("/nonexistent/file.md\n", tasks_dir=Path("/tmp/tasks"))
        assert result == "/nonexistent/file.md"

    def test_relative_path_not_found_anywhere_returns_string(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """selector_result returns raw string when relative path not in cwd or tasks_dir."""
        monkeypatch.chdir(tmp_path)
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        result = selector_result("missing-file.md\n", tasks_dir=tasks_dir)
        assert result == "missing-file.md"


# ---------------------------------------------------------------------------
# commands/loop/step.py – line 235: print_dry_run with bootstrap_prompt
# ---------------------------------------------------------------------------


class TestRunCommandStderrCallback:
    def test_stderr_callback_is_invoked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_command invokes stderr_callback when stderr chunks arrive."""
        target = tmp_path / "prompt.md"
        target.write_text("prompt", encoding="utf-8")

        captured_stderr: list[str] = []

        def fake_run_capture(
            cmd: list[str],
            *,
            env: dict[str, str],
            stdout_callback: Any = None,
            stderr_callback: Any = None,
            isolate_process_group: bool = False,
            idle_timeout: float | None = None,
        ) -> tuple[int, str, str]:
            if stderr_callback is not None:
                stderr_callback("error chunk")
            return (0, "", "")

        monkeypatch.setattr(
            "agm.agent.runner.run_capture", fake_run_capture
        )

        output = run_prompt_command(
            ["runner"], target, env={},
            stderr_callback=lambda c: captured_stderr.append(c),
        )
        assert output == "error chunk"
        assert captured_stderr == ["error chunk"]


# ---------------------------------------------------------------------------
# commands/loop/common.py – line 101: resolved_timeout when args.timeout is not None
# ---------------------------------------------------------------------------


class TestResolvedTimeoutFromArgs:
    def test_returns_args_timeout_when_provided(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """resolved_timeout returns args.timeout when it's provided."""
        from agm.commands.loop.common import resolved_timeout as _resolved_timeout

        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt=None,
            prompt_file=None,
            selector_prompt=None,
            selector_prompt_file=None,
            extra_prompt=None,
            extra_prompt_file=None,
            extra_selector_prompt=None,
            extra_selector_prompt_file=None,
            timeout=42.0,
        )
        result = _resolved_timeout(args)
        assert result == 42.0


# ---------------------------------------------------------------------------
# commands/loop/common.py – line 126: tasks_dir with relative path
# ---------------------------------------------------------------------------


class TestTasksDirFromConfigRelative:
    def test_config_tasks_dir_relative(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """tasks_dir joins relative config path with cwd."""
        from agm.commands.loop.common import tasks_dir as _tasks_dir

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[loop]\ntasks_dir = "my-tasks"\n', encoding="utf-8"
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
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
        result = _tasks_dir(args)
        assert result == tmp_path / "my-tasks"

    def test_absolute_tasks_dir_returned_directly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When tasks_dir is an absolute path, it is returned as-is."""
        from agm.commands.loop.common import tasks_dir as _tasks_dir

        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        abs_path = str(tmp_path / "absolute" / "tasks")
        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=abs_path,
            no_log=False,
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
        result = _tasks_dir(args)
        assert result == Path(abs_path)


# ---------------------------------------------------------------------------
# commands/loop/common.py – lines 270-271: prepare_select_invocation missing select.md
# ---------------------------------------------------------------------------


class TestPrepareSelectInvocationMissingDefault:
    def test_exits_when_default_select_md_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """prepare_select_invocation exits when no selector prompt is provided
        and the default select.md file is missing."""
        from agm.commands.args import LoopNextArgs
        from agm.commands.loop.common import prepare_select_invocation

        home = tmp_path / "home"
        (home / ".agm" / "prompts").mkdir(parents=True)
        # No select.md!
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
        monkeypatch.chdir(tmp_path)

        args = LoopNextArgs(
            command_name=None,
            runner="fake-runner",
            runner_args=[],
            selector="fake-selector",
            no_selector=False,
            tasks_dir=None,
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
        env = {"TASKS_DIR": str(tmp_path)}

        with pytest.raises(SystemExit) as exc_info:
            prepare_select_invocation(args, temp_files=[], env=env)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# project/layout.py – lines 98, 195: current_project_dir and current_checkout
# ---------------------------------------------------------------------------

class TestResolveExtraPromptSource:
    def test_returns_none_when_no_extra_prompt_specified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
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
        from agm.commands.loop.common import extra_prompt_source

        assert extra_prompt_source(args) is None

    def test_returns_extra_prompt_text_from_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt=None,
            prompt_file=None,
            selector_prompt=None,
            selector_prompt_file=None,
            extra_prompt="extra instructions",
            extra_prompt_file=None,
            extra_selector_prompt=None,
            extra_selector_prompt_file=None,
            timeout=None,
        )
        from agm.commands.loop.common import extra_prompt_source

        assert extra_prompt_source(args) == "extra instructions"

    def test_returns_extra_prompt_file_from_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt=None,
            prompt_file=None,
            selector_prompt=None,
            selector_prompt_file=None,
            extra_prompt=None,
            extra_prompt_file="/path/to/extra.md",
            extra_selector_prompt=None,
            extra_selector_prompt_file=None,
            timeout=None,
        )
        from agm.commands.loop.common import extra_prompt_source

        result = extra_prompt_source(args)
        assert result == Path("/path/to/extra.md")

    def test_cli_extra_prompt_overrides_extra_prompt_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt=None,
            prompt_file=None,
            selector_prompt=None,
            selector_prompt_file=None,
            extra_prompt="inline extra",
            extra_prompt_file="/path/to/extra.md",
            extra_selector_prompt=None,
            extra_selector_prompt_file=None,
            timeout=None,
        )
        from agm.commands.loop.common import extra_prompt_source

        result = extra_prompt_source(args)
        assert result == "inline extra"



class TestResolveExtraSelectorPromptSource:
    def test_returns_none_when_no_extra_selector_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
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
        from agm.commands.loop.common import extra_selector_prompt_source

        assert extra_selector_prompt_source(args) is None

    def test_returns_extra_selector_prompt_from_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt=None,
            prompt_file=None,
            selector_prompt=None,
            selector_prompt_file=None,
            extra_prompt=None,
            extra_prompt_file=None,
            extra_selector_prompt="extra selector text",
            extra_selector_prompt_file=None,
            timeout=None,
        )
        from agm.commands.loop.common import extra_selector_prompt_source

        result = extra_selector_prompt_source(args)
        assert result == "extra selector text"


class TestAppendExtraPrompt:
    def test_appends_inline_text_to_prompt_file(
        self, tmp_path: Path
    ) -> None:
        from agm.agent.runner import append_extra_prompt

        prompt = tmp_path / "prompt.md"
        prompt.write_text("Original content", encoding="utf-8")

        temp_files: list[Path] = []
        result = append_extra_prompt(
            prompt, "Extra instructions", temp_files=temp_files, env={}
        )

        assert result != prompt
        assert result.read_text(encoding="utf-8") == (
            "Original content" + chr(10) + "Extra instructions"
        )
        assert result in temp_files
        result.unlink()

    def test_appends_file_content_to_prompt_file(
        self, tmp_path: Path
    ) -> None:
        from agm.agent.runner import append_extra_prompt

        prompt = tmp_path / "prompt.md"
        prompt.write_text("Original content", encoding="utf-8")

        extra_file = tmp_path / "extra.md"
        extra_file.write_text("Extra from file", encoding="utf-8")

        temp_files: list[Path] = []
        result = append_extra_prompt(
            prompt, extra_file, temp_files=temp_files, env={}
        )

        assert result != prompt
        assert result.read_text(encoding="utf-8") == (
            "Original content" + chr(10) + "Extra from file"
        )
        assert result in temp_files
        result.unlink()

    def test_expands_env_vars_in_inline_text(
        self, tmp_path: Path
    ) -> None:
        from agm.agent.runner import append_extra_prompt

        prompt = tmp_path / "prompt.md"
        prompt.write_text("Original", encoding="utf-8")

        temp_files: list[Path] = []
        result = append_extra_prompt(
            prompt, "See $TASKS_DIR", temp_files=temp_files, env={"TASKS_DIR": "/tasks"}
        )

        assert result.read_text(encoding="utf-8") == "Original" + chr(10) + "See /tasks"
        result.unlink()

    def test_expands_env_vars_in_extra_file_content(
        self, tmp_path: Path
    ) -> None:
        from agm.agent.runner import append_extra_prompt

        prompt = tmp_path / "prompt.md"
        prompt.write_text("Original", encoding="utf-8")

        extra_file = tmp_path / "extra.md"
        extra_file.write_text("See $TASKS_DIR", encoding="utf-8")

        temp_files: list[Path] = []
        result = append_extra_prompt(
            prompt, extra_file, temp_files=temp_files, env={"TASKS_DIR": "/tasks"}
        )

        assert result.read_text(encoding="utf-8") == "Original" + chr(10) + "See /tasks"
        result.unlink()

    def test_missing_extra_prompt_file_exits(
        self, tmp_path: Path
    ) -> None:
        from agm.agent.runner import append_extra_prompt

        prompt = tmp_path / "prompt.md"
        prompt.write_text("Original", encoding="utf-8")

        missing = tmp_path / "missing.md"

        with pytest.raises(SystemExit):
            append_extra_prompt(prompt, missing, temp_files=[], env={})


class TestResolveExtraPromptSourceConfig:
    def test_returns_extra_prompt_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.commands.loop.common import extra_prompt_source

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[loop]\nextra_prompt = "config extra prompt"\n'
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
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
        result = extra_prompt_source(args)
        assert result == "config extra prompt"

    def test_returns_extra_prompt_file_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.commands.loop.common import extra_prompt_source

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[loop]\nextra_prompt_file = "my-extra.md"\n'
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
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
        result = extra_prompt_source(args)
        assert result == tmp_path / "my-extra.md"

    def test_cli_overrides_config_extra_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.commands.loop.common import extra_prompt_source

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[loop]\nextra_prompt = "config extra"\n'
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt=None,
            prompt_file=None,
            selector_prompt=None,
            selector_prompt_file=None,
            extra_prompt="cli extra",
            extra_prompt_file=None,
            extra_selector_prompt=None,
            extra_selector_prompt_file=None,
            timeout=None,
        )
        result = extra_prompt_source(args)
        assert result == "cli extra"


class TestResolveExtraSelectorPromptSourceConfig:
    def test_returns_extra_selector_prompt_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.commands.loop.common import extra_selector_prompt_source

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[loop]\nextra_selector_prompt = "config extra selector"\n'
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
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
        result = extra_selector_prompt_source(args)
        assert result == "config extra selector"

    def test_returns_extra_selector_prompt_file_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.commands.loop.common import extra_selector_prompt_source

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[loop]\nextra_selector_prompt_file = "my-extra-sel.md"\n'
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
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
        result = extra_selector_prompt_source(args)
        assert result == tmp_path / "my-extra-sel.md"


class TestResolveExtraPromptSourceRelativePath:
    def test_resolves_relative_extra_prompt_file_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.commands.loop.common import extra_prompt_source

        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt=None,
            prompt_file=None,
            selector_prompt=None,
            selector_prompt_file=None,
            extra_prompt=None,
            extra_prompt_file="relative/extra.md",
            extra_selector_prompt=None,
            extra_selector_prompt_file=None,
            timeout=None,
        )
        result = extra_prompt_source(args)
        assert result == tmp_path / "relative" / "extra.md"


class TestResolveExtraSelectorPromptSourceRelativePath:
    def test_resolves_relative_extra_selector_prompt_file_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.commands.loop.common import extra_selector_prompt_source

        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        args = LoopArgs(
            command_name=None,
            runner=None,
            runner_args=[],
            selector=None,
            no_selector=False,
            tasks_dir=None,
            no_log=False,
            log_file=None,
            prompt=None,
            prompt_file=None,
            selector_prompt=None,
            selector_prompt_file=None,
            extra_prompt=None,
            extra_prompt_file=None,
            extra_selector_prompt=None,
            extra_selector_prompt_file="relative/sel-extra.md",
            timeout=None,
        )
        result = extra_selector_prompt_source(args)
        assert result == tmp_path / "relative" / "sel-extra.md"
