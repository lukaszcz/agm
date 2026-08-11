"""Tests for installed-package RECORD creation and integrity verification."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Never

import pytest

from agm.core import fs
from agm.packages.record import RecordError, read_record, verify_record, write_record


def _package_tree(tmp_path: Path) -> Path:
    root = tmp_path / "review-tools"
    (root / "review-tools").mkdir(parents=True)
    (root / "package.toml").write_text('[package]\nname = "review-tools"\n', encoding="utf-8")
    (root / "review-tools" / "main.agl").write_text(
        "program def main() -> unit = ()\n", encoding="utf-8"
    )
    return root


def test_write_record_lists_sorted_relative_paths_and_sha256_digests(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)

    record_path = write_record(root)

    main_digest = hashlib.sha256((root / "review-tools" / "main.agl").read_bytes()).hexdigest()
    manifest_digest = hashlib.sha256((root / "package.toml").read_bytes()).hexdigest()
    assert record_path == root / "RECORD"
    assert record_path.read_text(encoding="utf-8") == (
        f"package.toml,sha256={manifest_digest}\nreview-tools/main.agl,sha256={main_digest}\n"
    )


def test_record_round_trip_verifies_all_recorded_files(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    write_record(root)

    entries = verify_record(root)

    assert tuple(entry.path for entry in entries) == ("package.toml", "review-tools/main.agl")
    assert read_record(root) == entries


def test_record_round_trip_preserves_csv_escaped_paths(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    asset = root / 'prompt,"draft".md'
    asset.write_text("prompt", encoding="utf-8")

    write_record(root)

    assert verify_record(root)[1].path == 'prompt,"draft".md'


@pytest.mark.parametrize("name", ["line\nbreak", "line\rbreak"])
def test_write_record_rejects_files_with_line_breaks_in_their_name(
    tmp_path: Path, name: str
) -> None:
    root = _package_tree(tmp_path)
    (root / name).write_text("unsafe", encoding="utf-8")

    with pytest.raises(RecordError):
        write_record(root)


@pytest.mark.parametrize("operation", [read_record, verify_record])
def test_record_reading_rejects_files_with_line_breaks_in_their_name(
    tmp_path: Path, operation: Callable[[Path], object]
) -> None:
    root = _package_tree(tmp_path)
    write_record(root)
    (root / "line\nbreak").write_text("unsafe", encoding="utf-8")

    with pytest.raises(RecordError):
        operation(root)


def test_record_order_uses_posix_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "package"
    (root / "a").mkdir(parents=True)
    (root / "a" / "file").write_text("nested", encoding="utf-8")
    (root / "aZ").write_text("sibling", encoding="utf-8")

    write_record(root)

    assert tuple(entry.path for entry in read_record(root)) == ("a/file", "aZ")


def test_read_record_rejects_a_missing_record(tmp_path: Path) -> None:
    with pytest.raises(RecordError):
        read_record(_package_tree(tmp_path))


def test_read_record_wraps_invalid_utf8(tmp_path: Path) -> None:
    root = tmp_path / "package"
    root.mkdir()
    (root / "RECORD").write_bytes(b"\xff")

    with pytest.raises(RecordError):
        read_record(root)


def test_read_record_accepts_an_empty_record(tmp_path: Path) -> None:
    root = tmp_path / "empty-package"
    root.mkdir()
    (root / "RECORD").write_text("", encoding="utf-8")

    assert read_record(root) == ()


@pytest.mark.parametrize("path", ["C:relative", r"\rooted", r"C:\rooted"])
def test_read_record_rejects_windows_drive_relative_and_rooted_paths(
    tmp_path: Path, path: str
) -> None:
    root = tmp_path / "package"
    root.mkdir()
    (root / "RECORD").write_text(path + ",sha256=" + "0" * 64 + "\n", encoding="utf-8")

    with pytest.raises(RecordError):
        read_record(root)


@pytest.mark.parametrize(
    "content",
    [
        "\n",
        "package.toml,sha256=" + "0" * 64 + "\npackage.toml,sha256=" + "0" * 64 + "\n",
        "package.toml,not-a-digest\n",
        "package.toml\n",
        '"package.toml"unexpected,sha256=' + "0" * 64 + "\n",
        '"package.toml,sha256=' + "0" * 64 + "\n",
        '"package.toml"',
    ],
)
def test_read_record_rejects_malformed_entries(tmp_path: Path, content: str) -> None:
    root = tmp_path / "package"
    root.mkdir()
    (root / "RECORD").write_text(content, encoding="utf-8")

    with pytest.raises(RecordError):
        read_record(root)


def test_write_record_rejects_a_symlink(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    (root / "linked-package.toml").symlink_to(root / "package.toml")

    with pytest.raises(RecordError):
        write_record(root)


@pytest.mark.skipif(os.name != "posix", reason="requires Unix FIFO support")
@pytest.mark.parametrize("operation", [write_record, read_record, verify_record])
def test_record_operations_reject_a_fifo(
    tmp_path: Path, operation: Callable[[Path], object]
) -> None:
    root = _package_tree(tmp_path)
    write_record(root)
    os.mkfifo(root / "pipe")

    with pytest.raises(RecordError):
        operation(root)


@pytest.mark.parametrize("operation", [write_record, read_record, verify_record])
def test_record_operations_reject_a_symlink_root(
    tmp_path: Path, operation: Callable[[Path], object]
) -> None:
    package = _package_tree(tmp_path)
    write_record(package)
    root = tmp_path / "linked-package"
    root.symlink_to(package, target_is_directory=True)

    with pytest.raises(RecordError):
        operation(root)


@pytest.mark.parametrize("operation", [read_record, verify_record])
def test_record_reading_and_verification_reject_symlink_descendants(
    tmp_path: Path, operation: Callable[[Path], object]
) -> None:
    root = _package_tree(tmp_path)
    write_record(root)
    (root / "linked-package.toml").symlink_to(root / "package.toml")

    with pytest.raises(RecordError):
        operation(root)


def test_record_wraps_traversal_io_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)

    def fail_rglob(_root: Path, _pattern: str) -> Iterator[Path]:
        raise OSError("blocked")

    monkeypatch.setattr(fs, "rglob", fail_rglob)

    with pytest.raises(RecordError, match="blocked"):
        write_record(root)


def test_record_wraps_second_traversal_io_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    original_rglob = fs.rglob
    calls = 0

    def fail_second_rglob(path: Path, pattern: str) -> Iterator[Path]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("blocked")
        return original_rglob(path, pattern)

    monkeypatch.setattr(fs, "rglob", fail_second_rglob)

    with pytest.raises(RecordError, match="blocked"):
        write_record(root)


def test_record_wraps_hash_io_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _package_tree(tmp_path)

    def fail_open(_path: Path, *_args: object, **_kwargs: object) -> Never:
        raise OSError("blocked")

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(RecordError, match="blocked"):
        write_record(root)


def _change_recorded_file(root: Path) -> None:
    (root / "review-tools" / "main.agl").write_text("changed", encoding="utf-8")


def _add_unrecorded_file(root: Path) -> None:
    (root / "extra.txt").write_text("unrecorded", encoding="utf-8")


def _traverse_outside_the_package(root: Path) -> None:
    (root / "RECORD").write_text("../outside,sha256=" + "0" * 64 + "\n", encoding="utf-8")


_TAMPERS: list[Callable[[Path], None]] = [
    _change_recorded_file,
    _add_unrecorded_file,
    _traverse_outside_the_package,
]


@pytest.mark.parametrize("tamper", _TAMPERS)
def test_verify_record_rejects_tampered_files_and_records(
    tmp_path: Path, tamper: Callable[[Path], None]
) -> None:
    root = _package_tree(tmp_path)
    write_record(root)

    tamper(root)

    with pytest.raises(RecordError):
        verify_record(root)
