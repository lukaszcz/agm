"""Tests for ``agm config env`` shell output."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.commands.config.env import shell_env_delta
from agm.core.env import source_env_files


def test_shell_env_delta_exports_changed_values_with_shell_quoting() -> None:
    statements = shell_env_delta(
        before={"UNCHANGED": "same", "CHANGED": "old"},
        after={"UNCHANGED": "same", "CHANGED": "new value", "ADDED": "quote'value"},
    )

    assert statements == [
        "export ADDED='quote'\"'\"'value'",
        "export CHANGED='new value'",
    ]


def test_shell_env_delta_unsets_removed_values() -> None:
    statements = shell_env_delta(
        before={"REMOVED": "1", "UNCHANGED": "same"},
        after={"UNCHANGED": "same"},
    )

    assert statements == ["unset REMOVED"]


def test_shell_env_delta_skips_unset_for_unsafe_removed_name() -> None:
    statements = shell_env_delta(
        before={"BAD-NAME": "old", "GOOD": "keep"},
        after={"GOOD": "keep"},
    )

    assert statements == []


def test_shell_env_delta_skips_names_that_shell_cannot_assign() -> None:
    statements = shell_env_delta(
        before={"BAD-NAME": "old"},
        after={"BAD-NAME": "new", "ALSO.BAD": "1", "GOOD_NAME": "ok"},
    )

    assert statements == ["export GOOD_NAME=ok"]


def test_shell_env_delta_skips_special_shell_parameters() -> None:
    statements = shell_env_delta(
        before={
            "_": "old-last-arg",
            "PWD": "/old",
            "OLDPWD": "/previous",
            "SHLVL": "1",
            "UID": "1000",
            "REMOVED": "1",
        },
        after={
            "_": "new-last-arg",
            "PWD": "/new",
            "OLDPWD": "/old",
            "SHLVL": "2",
            "UID": "1001",
            "ADDED": "1",
        },
    )

    assert statements == [
        "unset REMOVED",
        "export ADDED=1",
    ]


class TestSourceEnvFiles:
    def test_exits_on_nonzero_returncode(self, tmp_path: Path) -> None:
        """source_env_files calls exit_with_output when bash returns non-zero."""
        env_file = tmp_path / "bad.sh"
        env_file.write_text("exit 42\n", encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            source_env_files([env_file])
        assert exc_info.value.code == 42

    def test_ignores_stdout_from_sourced_files(self, tmp_path: Path) -> None:
        env_file = tmp_path / "env.sh"
        env_file.write_text(
            f'printf "%s\\n" "{tmp_path}"\nexport VALUE=from-env\n',
            encoding="utf-8",
        )

        result = source_env_files([env_file], env={})

        assert result["VALUE"] == "from-env"
        assert all(str(tmp_path) not in key for key in result)
