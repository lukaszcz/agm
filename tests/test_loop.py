"""Focused tests for loop helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.commands.args import LoopArgs, LoopProgressArgs
from agm.commands.loop.common import (
    is_complete_output,
    prepare_progress_invocation,
    prompt_file,
    selector_result,
    use_selector_mode,
)
from agm.core.prompt import preprocess_prompt_file


def test_preprocess_prompt_file_expands_known_env_vars(tmp_path: Path) -> None:
    prompt_file = tmp_path / "loop.md"
    prompt_file.write_text("known=$TEST_VAR unknown=${MISSING}\n", encoding="utf-8")

    temp_files: list[Path] = []
    processed = preprocess_prompt_file(
        prompt_file,
        temp_files=temp_files,
        env={"TEST_VAR": "expanded"},
    )

    assert processed != prompt_file
    assert processed.read_text(encoding="utf-8") == "known=expanded unknown=${MISSING}\n"
    assert temp_files == [processed]


def test_preprocess_prompt_file_reuses_original_when_nothing_changes(tmp_path: Path) -> None:
    prompt_file = tmp_path / "loop.md"
    prompt_file.write_text("literal ${MISSING}\n", encoding="utf-8")

    temp_files: list[Path] = []
    processed = preprocess_prompt_file(prompt_file, temp_files=temp_files, env={})

    assert processed == prompt_file
    assert temp_files == []


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


def test_prepare_progress_invocation_prefers_selector_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    prompt_dir = home / ".agm" / "prompts"
    prompt_dir.mkdir(parents=True)
    prompt_path = prompt_dir / "update_progress.md"
    prompt_path.write_text("update $TASKS_DIR\n", encoding="utf-8")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    monkeypatch.chdir(tmp_path)

    args = LoopProgressArgs(
        command_name=None,
        runner="runner --print",
        runner_args=[],
        selector="selector",
        no_selector=False,
        tasks_dir="custom/tasks",
    )
    env = {"TASKS_DIR": str(tmp_path / "custom" / "tasks")}

    invocation = prepare_progress_invocation(args, temp_files=[], env=env)

    assert invocation.command == ["selector"]
    assert invocation.command_kind == "selector"
    assert invocation.source_prompt_file == prompt_path
    assert invocation.effective_prompt_file.read_text(encoding="utf-8") == (
        f"update {tmp_path / 'custom' / 'tasks'}\n"
    )


def test_prepare_progress_invocation_falls_back_to_runner_without_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    prompt_dir = home / ".agm" / "prompts"
    prompt_dir.mkdir(parents=True)
    prompt_path = prompt_dir / "update_progress.md"
    prompt_path.write_text("update progress\n", encoding="utf-8")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    monkeypatch.chdir(tmp_path)

    args = LoopProgressArgs(
        command_name=None,
        runner="runner --print",
        runner_args=["--verbose"],
        selector=None,
        no_selector=False,
        tasks_dir=None,
    )

    invocation = prepare_progress_invocation(args, temp_files=[], env={})

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
    )
    assert use_selector_mode(args) is False


def test_use_selector_mode_cli_no_selector_overrides_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Config doesn't set no_selector, but CLI does
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    args = LoopProgressArgs(
        command_name=None,
        runner=None,
        runner_args=[],
        selector=None,
        no_selector=True,
        tasks_dir=None,
    )
    assert use_selector_mode(args) is False
