"""Behavior tests for dry-run-aware filesystem copy and write primitives."""

from __future__ import annotations

import stat
from collections.abc import Callable, Generator
from pathlib import Path

import pytest

from agm.core import dry_run, fs


@pytest.fixture(autouse=True)
def reset_dry_run() -> Generator[None, None, None]:
    """Keep the global dry-run state isolated between tests."""
    previous = dry_run.enabled()
    dry_run.set_enabled(False)
    yield
    dry_run.set_enabled(previous)


def test_write_text_atomic_replaces_existing_content(tmp_path: Path) -> None:
    path = tmp_path / "index.toml"
    path.write_text("previous\n", encoding="utf-8")

    fs.write_text_atomic(path, "current\n")

    assert path.read_text(encoding="utf-8") == "current\n"
    assert [child.name for child in tmp_path.iterdir()] == ["index.toml"]


def test_write_text_atomic_preserves_existing_permissions(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("SECRET=previous\n", encoding="utf-8")
    path.chmod(0o600)

    fs.write_text_atomic(path, "SECRET=current\n")

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_write_text_atomic_leaves_no_temporary_file_when_the_write_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing-parent" / "index.toml"

    with pytest.raises(OSError):
        fs.write_text_atomic(path, "current\n")

    assert not path.exists()


def test_write_text_atomic_logs_and_does_not_write_when_dry_run_is_enabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "index.toml"
    dry_run.set_enabled(True)

    fs.write_text_atomic(path, "current\n")

    assert not path.exists()
    assert capsys.readouterr().out == f"dry-run: agm write-file {path}\n"


def test_copy_tree_copies_a_complete_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "modules").mkdir(parents=True)
    (source / "package.toml").write_text("[package]\n", encoding="utf-8")
    (source / "modules" / "main.agl").write_text("program def main() -> unit = ()\n")
    destination = tmp_path / "destination"

    fs.copy_tree(source, destination)

    assert (destination / "package.toml").read_text(encoding="utf-8") == "[package]\n"
    assert (destination / "modules" / "main.agl").read_text(encoding="utf-8") == (
        "program def main() -> unit = ()\n"
    )


def test_copy_tree_preserves_descendant_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "outside-package"
    target.write_text("not package contents", encoding="utf-8")
    (source / "linked-file").symlink_to(target)
    destination = tmp_path / "destination"

    fs.copy_tree(source, destination)

    link = destination / "linked-file"
    assert link.is_symlink()
    assert link.readlink() == target


def test_copy_tree_rejects_a_symlink_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_link = tmp_path / "source-link"
    source_link.symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError):
        fs.copy_tree(source_link, tmp_path / "destination")


def test_copy_tree_rejects_symlinks_in_an_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "modules").mkdir()
    (source / "modules" / "main.agl").write_text("contents", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "modules").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError):
        fs.copy_tree(source, destination, dirs_exist_ok=True)

    assert not (outside / "main.agl").exists()


def test_copy_tree_rejects_a_symlink_destination_ancestor_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.agl").write_text("contents", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    destination_parent = tmp_path / "destination-parent"
    destination_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError):
        fs.copy_tree(source, destination_parent / "package")

    assert not (outside / "package").exists()


_COPY_CASES: list[tuple[Callable[[Path, Path], None], str, str, str]] = [
    (fs.copy_tree, "source", "destination", "copy-tree"),
]


@pytest.mark.parametrize(("copy", "source_name", "destination_name", "operation"), _COPY_CASES)
def test_copy_primitives_log_and_do_not_write_when_dry_run_is_enabled(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    copy: Callable[[Path, Path], None],
    source_name: str,
    destination_name: str,
    operation: str,
) -> None:
    source = tmp_path / source_name
    if source.suffix:
        source.write_text("contents", encoding="utf-8")
    else:
        source.mkdir()
        (source / "file.txt").write_text("contents", encoding="utf-8")
    destination = tmp_path / destination_name
    dry_run.set_enabled(True)

    copy(source, destination)

    assert not destination.exists()
    assert capsys.readouterr().out == f"dry-run: agm {operation} {source} {destination}\n"
