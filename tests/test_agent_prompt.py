"""Tests for agent prompt preparation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agent.prompt import expand_prompt_env_vars, preprocess_prompt_file


def test_expand_prompt_env_vars_uses_runtime_holes_and_preserves_other_text() -> None:
    expanded = expand_prompt_env_vars(
        r"A=%{KNOWN} B=%{OTHER_2} C=\%{KNOWN} D=% E=\q F=$KNOWN G=${KNOWN}",
        env={"KNOWN": "value", "OTHER_2": "two"},
    )

    assert expanded == r"A=value B=two C=%{KNOWN} D=% E=\q F=$KNOWN G=${KNOWN}"


def test_expand_prompt_env_vars_exits_for_missing_inline_variable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        expand_prompt_env_vars("prompt %{MISSING}", env={})

    error = capsys.readouterr().err
    assert "inline prompt" in error
    assert "MISSING" in error


@pytest.mark.parametrize(
    ("content", "fragment"),
    [("prompt %{bad name}", "bad name"), ("prompt %{unfinished", "%{unfinished")],
)
def test_expand_prompt_env_vars_exits_for_malformed_inline_hole(
    content: str, fragment: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        expand_prompt_env_vars(content, env={})

    error = capsys.readouterr().err
    assert "inline prompt" in error
    assert fragment in error


def test_preprocess_prompt_file_expands_known_env_vars(tmp_path: Path) -> None:
    prompt_file = tmp_path / "loop.md"
    prompt_file.write_text("known=%{TEST_VAR}\n", encoding="utf-8")

    temp_files: list[Path] = []
    processed = preprocess_prompt_file(
        prompt_file,
        temp_files=temp_files,
        env={"TEST_VAR": "expanded"},
    )

    assert processed != prompt_file
    assert processed.read_text(encoding="utf-8") == "known=expanded\n"
    assert temp_files == [processed]


def test_preprocess_prompt_file_names_source_for_missing_variable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt_file = tmp_path / "loop.md"
    prompt_file.write_text("missing=%{MISSING}\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        preprocess_prompt_file(prompt_file, temp_files=[], env={})

    error = capsys.readouterr().err
    assert "loop.md" in error
    assert "MISSING" in error


def test_preprocess_prompt_file_names_source_for_malformed_hole(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt_file = tmp_path / "loop.md"
    prompt_file.write_text("malformed=%{bad name}\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        preprocess_prompt_file(prompt_file, temp_files=[], env={})

    error = capsys.readouterr().err
    assert "loop.md" in error
    assert "bad name" in error


def test_preprocess_prompt_file_reuses_original_when_nothing_changes(tmp_path: Path) -> None:
    prompt_file = tmp_path / "loop.md"
    prompt_file.write_text(r"literal ${MISSING} and bare %\n", encoding="utf-8")

    temp_files: list[Path] = []
    processed = preprocess_prompt_file(prompt_file, temp_files=temp_files, env={})

    assert processed == prompt_file
    assert temp_files == []
