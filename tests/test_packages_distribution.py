"""Tests for the distribution view of a package source tree."""

from __future__ import annotations

from pathlib import Path

from agm.packages.distribution import distribution_files


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _selected(root: Path) -> set[str]:
    return {relative for relative, _ in distribution_files(root)}


def test_nested_directory_rule_excludes_matching_descendants(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "sub/.gitignore": "build/\n",
            "sub/build/out.txt": "generated\n",
            "sub/a/build/out.txt": "generated\n",
            "sub/a/keep.txt": "source\n",
            "sub/keep.txt": "source\n",
            "build/out.txt": "outside the nested rule\n",
        },
    )

    assert _selected(tmp_path) == {"sub/keep.txt", "sub/a/keep.txt", "build/out.txt"}


def test_anchored_nested_directory_rule_excludes_only_that_level(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "sub/.gitignore": "/build/\n",
            "sub/build/out.txt": "generated\n",
            "sub/a/build/out.txt": "source\n",
        },
    )

    assert _selected(tmp_path) == {"sub/a/build/out.txt"}


def test_nested_rule_with_a_middle_separator_stays_relative_to_its_file(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "sub/.gitignore": "a/b\n",
            "sub/a/b": "generated\n",
            "sub/c/a/b": "source\n",
        },
    )

    assert _selected(tmp_path) == {"sub/c/a/b"}


def test_deeper_negation_reincludes_a_directory_its_parent_rule_excluded(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "sub/.gitignore": "tmp/\n",
            "sub/a/.gitignore": "!tmp/\n",
            "sub/x/tmp/out.txt": "generated\n",
            "sub/a/deep/tmp/out.txt": "source\n",
        },
    )

    assert _selected(tmp_path) == {"sub/a/deep/tmp/out.txt"}
