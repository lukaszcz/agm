"""Tests for the shared portable-relative-path predicate."""

from __future__ import annotations

import pytest

from agm.core.path import is_portable_relative_path


@pytest.mark.parametrize(
    "value",
    [
        "main.agl",
        ".hidden",
        "review_tools/main.agl",
        "review_tools-1.2.3/package.toml",
        "a/b/c.txt",
        "..hidden",
        "prompts/review.md",
    ],
)
def test_portable_relative_paths_are_accepted(value: str) -> None:
    assert is_portable_relative_path(value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".",
        "./",
        "..",
        "./a",
        "a/.",
        "a/..",
        "../a",
        "a/./b",
        "/absolute",
        "a//b",
        "a/",
        "a\\b",
        "C:/windows",
        "C:windows",
        "\\\\server\\share",
        "a\x00b",
    ],
)
def test_non_portable_relative_paths_are_rejected(value: str) -> None:
    assert not is_portable_relative_path(value)
