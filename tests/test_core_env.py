"""Contract tests for environment views and shell capture."""

from __future__ import annotations

import os
from pathlib import Path
from types import MappingProxyType

import pytest

from agm.core import env


def test_resolve_env_retains_the_supplied_mapping_without_copying() -> None:
    supplied = MappingProxyType({"HOME": "/explicit"})

    assert env.resolve_env(supplied) is supplied
    assert env.resolve_env() is os.environ


def test_resolve_home_uses_only_the_supplied_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/ambient")

    assert env.resolve_home({"HOME": "/explicit"}) == Path("/explicit")
    assert env.resolve_home({}) == Path("~")


@pytest.mark.parametrize(
    ("columns", "expected"), [(20, 40), (40, 40), (60, 60), (80, 80), (120, 80)]
)
def test_help_width_clamps_terminal_columns(
    columns: int, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        env.shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((columns, 24))
    )

    assert env.help_width() == expected


@pytest.mark.parametrize("single_file", [False, True])
def test_shell_environment_sourcing_honors_cwd(tmp_path: Path, single_file: bool) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "value.txt").write_text("from-work", encoding="utf-8")
    script = tmp_path / "capture.sh"
    script.write_text('export AGM_TEST_CWD_VALUE="$(cat value.txt)"\n', encoding="utf-8")

    if single_file:
        result = env.source_env_file(script, {}, cwd=work)
    else:
        result = env.source_env_files([script], {}, cwd=work)

    assert result["AGM_TEST_CWD_VALUE"] == "from-work"


def test_shell_environment_capture_preserves_non_utf8_value_bytes(tmp_path: Path) -> None:
    script = tmp_path / "capture.sh"
    script.write_text("export AGM_TEST_RAW=\"$(printf '\\377')\"\n", encoding="utf-8")

    result = env.source_env_files([script], {})

    assert result["AGM_TEST_RAW"].encode("utf-8", errors="surrogateescape") == b"\xff"
