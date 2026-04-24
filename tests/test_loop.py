"""Focused tests for loop helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.commands.loop.common import prompt_file, selector_result
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
