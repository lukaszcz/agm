"""Focused tests for loop helpers."""

from __future__ import annotations

from pathlib import Path

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
