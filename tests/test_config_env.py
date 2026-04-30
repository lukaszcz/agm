"""Tests for ``agm config env`` shell output."""

from __future__ import annotations

from agm.commands.config.env import shell_env_delta


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
