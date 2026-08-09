"""Tests for deterministic package archive creation and verification."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO, cast

import pytest

import agm.packages.archive as package_archive
from agm.packages.archive import (
    ArchiveError,
    read_archive_manifest,
    read_archive_metadata,
    verify_archive,
    write_archive,
)
from agm.packages.record import RecordEntry, serialize_record


def _package_tree(tmp_path: Path) -> Path:
    root = tmp_path / "review_tools"
    (root / "review_tools").mkdir(parents=True)
    (root / "package.toml").write_text(
        '[package]\nversion = "1.2.3"\nname = "review_tools"\n', encoding="utf-8"
    )
    (root / "review_tools" / "main.agl").write_text(
        "program def main() -> unit = ()\n", encoding="utf-8"
    )
    return root


def test_write_archive_is_deterministic_and_independent_of_git_directory(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    first = tmp_path / "first.agmpkg"
    second = tmp_path / "second.agmpkg"

    first_metadata = write_archive(root, first)
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("different", encoding="utf-8")
    second_metadata = write_archive(root, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_metadata == second_metadata
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [
            "review_tools-1.2.3/RECORD",
            "review_tools-1.2.3/package.toml",
            "review_tools-1.2.3/review_tools/main.agl",
        ]
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        assert all((info.external_attr >> 16) == 0o100644 for info in archive.infolist())


def test_write_archive_excludes_private_vcs_cache_archive_and_ignored_files(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    (root / ".gitignore").write_text("ignored.txt\nnested/ignored.txt\n", encoding="utf-8")
    (root / ".hidden").write_text("hidden", encoding="utf-8")
    (root / "ignored.txt").write_text("ignored", encoding="utf-8")
    (root / "build.agmpkg").write_text("archive", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "module.pyc").write_text("cache", encoding="utf-8")
    (root / ".hg").mkdir()
    (root / ".hg" / "state").write_text("vcs", encoding="utf-8")
    (root / "nested").mkdir()
    (root / "nested" / "ignored.txt").write_text("ignored", encoding="utf-8")
    (root / "nested" / "kept.txt").write_text("kept", encoding="utf-8")

    archive_path = tmp_path / "package.agmpkg"
    write_archive(root, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    assert names == [
        "review_tools-1.2.3/RECORD",
        "review_tools-1.2.3/nested/kept.txt",
        "review_tools-1.2.3/package.toml",
        "review_tools-1.2.3/review_tools/main.agl",
    ]


def test_write_archive_refuses_symlinks_and_casefolding_collisions(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    (root / "linked.agl").symlink_to(root / "review_tools" / "main.agl")

    with pytest.raises(ArchiveError, match="symlink"):
        write_archive(root, tmp_path / "linked.agmpkg")

    (root / "linked.agl").unlink()
    (root / "README").write_text("one", encoding="utf-8")
    (root / "readme").write_text("two", encoding="utf-8")
    with pytest.raises(ArchiveError, match="case"):
        write_archive(root, tmp_path / "collision.agmpkg")


def test_archive_metadata_and_manifest_are_read_without_leaving_an_open_archive(
    tmp_path: Path,
) -> None:
    root = _package_tree(tmp_path)
    archive_path = tmp_path / "package.agmpkg"
    metadata = write_archive(root, archive_path)

    read_metadata = read_archive_metadata(archive_path)

    assert read_metadata == metadata
    assert read_metadata.manifest.name == "review_tools"
    assert str(read_metadata.manifest.version) == "1.2.3"
    assert read_archive_manifest(archive_path) == read_metadata.manifest
    assert verify_archive(archive_path) == read_metadata


def test_verify_archive_rejects_a_record_that_does_not_match_its_contents(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    archive_path = tmp_path / "package.agmpkg"
    write_archive(root, archive_path)
    prefix = "review_tools-1.2.3/"
    with zipfile.ZipFile(archive_path) as source:
        contents = {info.filename: source.read(info) for info in source.infolist()}
    contents[prefix + "review_tools/main.agl"] = b"changed"
    _write_zip(archive_path, list(contents.items()))

    with pytest.raises(ArchiveError, match="RECORD"):
        verify_archive(archive_path)


def _archive_contents(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def _write_zip(path: Path, contents: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in contents:
            archive.writestr(package_archive._zip_info(name), content)


def test_write_archive_refuses_destinations_inside_the_source_tree(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    manifest = root / "package.toml"
    original_manifest = manifest.read_bytes()

    with pytest.raises(ArchiveError):
        write_archive(root, manifest)

    assert manifest.read_bytes() == original_manifest


def test_write_archive_refuses_symlink_destination_aliasing_the_source_tree(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    manifest = root / "package.toml"
    original_manifest = manifest.read_bytes()
    destination = tmp_path / "package.agmpkg"
    destination.symlink_to(manifest)

    with pytest.raises(ArchiveError):
        write_archive(root, destination)

    assert manifest.read_bytes() == original_manifest
    assert destination.is_symlink()


def test_write_archive_refuses_hard_link_destination_aliasing_a_source_file(
    tmp_path: Path,
) -> None:
    root = _package_tree(tmp_path)
    source = root / "review_tools" / "main.agl"
    original_source = source.read_bytes()
    destination = tmp_path / "package.agmpkg"
    destination.hardlink_to(source)

    with pytest.raises(ArchiveError):
        write_archive(root, destination)

    assert source.read_bytes() == original_source
    assert destination.read_bytes() == original_source


def test_write_archive_atomically_replaces_a_destination_hardlinked_at_replace_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    source = root / "review_tools" / "main.agl"
    original_source = source.read_bytes()
    destination = tmp_path / "package.agmpkg"
    destination.write_text("previous archive", encoding="utf-8")
    replace = os.rename

    def replace_destination(
        temporary: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        assert target == destination.name
        assert src_dir_fd is not None
        assert os.stat(temporary, dir_fd=src_dir_fd).st_mode & 0o777 == 0o600
        destination.unlink()
        destination.hardlink_to(source)
        replace(temporary, target, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(package_archive.os, "rename", replace_destination)

    metadata = write_archive(root, destination)

    assert source.read_bytes() == original_source
    assert destination.stat().st_ino != source.stat().st_ino
    assert verify_archive(destination) == metadata


def test_write_archive_enforces_the_archive_entry_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    destination = tmp_path / "package.agmpkg"
    monkeypatch.setattr(package_archive, "MAX_ARCHIVE_ENTRIES", 2)

    with pytest.raises(ArchiveError, match="entries"):
        write_archive(root, destination)

    assert not destination.exists()


@pytest.mark.parametrize(
    ("limit_name", "error"),
    [
        ("MAX_ARCHIVE_ENTRY_SIZE", "entry exceeds"),
        ("MAX_ARCHIVE_TOTAL_SIZE", "total size"),
    ],
)
def test_write_archive_enforces_verifier_size_limits_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit_name: str, error: str
) -> None:
    root = _package_tree(tmp_path)
    destination = tmp_path / "package.agmpkg"
    destination.write_bytes(b"existing archive")
    monkeypatch.setattr(package_archive, limit_name, 1)

    with pytest.raises(ArchiveError, match=error):
        write_archive(root, destination)

    assert destination.read_bytes() == b"existing archive"


def test_write_archive_replaces_a_destination_symlink_without_writing_its_target(
    tmp_path: Path,
) -> None:
    root = _package_tree(tmp_path)
    target = tmp_path / "unrelated.txt"
    target.write_text("unrelated", encoding="utf-8")
    destination = tmp_path / "package.agmpkg"
    destination.symlink_to(target)

    metadata = write_archive(root, destination)

    assert target.read_text(encoding="utf-8") == "unrelated"
    assert not destination.is_symlink()
    assert verify_archive(destination) == metadata


def test_write_archive_keeps_publication_in_opened_parent_after_parent_symlink_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    destination_parent = tmp_path / "output"
    destination_parent.mkdir()
    destination = destination_parent / "package.agmpkg"
    destination.write_text("previous archive", encoding="utf-8")
    displaced_parent = tmp_path / "displaced-output"
    replace = os.rename

    def replace_destination(
        temporary: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        replace(destination_parent, displaced_parent)
        destination_parent.symlink_to(root, target_is_directory=True)
        replace(temporary, target, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(package_archive.os, "rename", replace_destination)

    metadata = write_archive(root, destination)

    assert not (root / "package.agmpkg").exists()
    assert verify_archive(displaced_parent / "package.agmpkg") == metadata


def test_write_archive_rejects_source_root_rebound_to_publication_parent_after_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    destination_parent = tmp_path / "output"
    destination = destination_parent / "package.agmpkg"
    archive_contents = package_archive._archive_contents

    def collect_then_rebind_source(
        source_root: Path, manifest: package_archive.PackageManifest
    ) -> dict[str, bytes]:
        contents = archive_contents(source_root, manifest)
        source_root.rename(destination_parent)
        root.mkdir()
        return contents

    monkeypatch.setattr(package_archive, "_archive_contents", collect_then_rebind_source)

    with pytest.raises(ArchiveError, match="inside package root"):
        write_archive(root, destination)

    assert not destination.exists()


def test_write_archive_rejects_parent_moved_into_source_after_initial_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    source = root / "review_tools" / "main.agl"
    original_source = source.read_bytes()
    destination_parent = tmp_path / "output"
    destination_parent.mkdir()
    destination = destination_parent / "package.agmpkg"
    write_zip = package_archive._write_zip

    def write_then_move_parent_into_source(
        file: IO[bytes], prefix: str, contents: dict[str, bytes]
    ) -> None:
        write_zip(file, prefix, contents)
        destination_parent.rename(root / "output")

    monkeypatch.setattr(package_archive, "_write_zip", write_then_move_parent_into_source)

    with pytest.raises(ArchiveError, match="inside package root"):
        write_archive(root, destination)

    assert source.read_bytes() == original_source
    assert not (root / "output" / destination.name).exists()
    assert not list((root / "output").glob(".package.agmpkg.*"))


@pytest.mark.parametrize("existing_destination", [False, True])
def test_write_archive_reports_and_rolls_back_a_detected_parent_move_into_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_destination: bool
) -> None:
    """Concurrent namespace mutation is unsupported but reported when detected."""

    root = _package_tree(tmp_path)
    source = root / "review_tools" / "main.agl"
    original_source = source.read_bytes()
    root_collision = root / "package.agmpkg"
    root_collision.write_bytes(b"root source collision")
    collision = root / "nested" / "package.agmpkg"
    collision.parent.mkdir()
    collision.write_bytes(b"nested source collision")
    destination_parent = tmp_path / "output"
    destination_parent.mkdir()
    destination = destination_parent / "package.agmpkg"
    if existing_destination:
        destination.write_bytes(b"previous archive")
    replace = os.rename
    moved = False

    def move_parent_then_replace(
        temporary: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal moved
        if not moved:
            moved = True
            replace(destination_parent, root / "published-output")
        replace(temporary, target, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(package_archive.os, "rename", move_parent_then_replace)

    with pytest.raises(ArchiveError, match="inside package root"):
        write_archive(root, destination)

    assert source.read_bytes() == original_source
    assert root_collision.read_bytes() == b"root source collision"
    assert collision.read_bytes() == b"nested source collision"
    published = root / "published-output" / destination.name
    if existing_destination:
        assert published.read_bytes() == b"previous archive"
    else:
        assert not published.exists()


def test_write_archive_reports_a_containment_error_when_rollback_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    destination_parent = tmp_path / "output"
    destination_parent.mkdir()
    destination = destination_parent / "package.agmpkg"
    replace = os.rename
    unlink = os.unlink
    moved = False

    def move_parent_then_replace(
        temporary: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal moved
        if not moved:
            moved = True
            replace(destination_parent, root / "published-output")
        replace(temporary, target, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    def fail_published_unlink(path: str, *, dir_fd: int | None = None) -> None:
        if path == destination.name:
            raise OSError("denied")
        unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(package_archive.os, "rename", move_parent_then_replace)
    monkeypatch.setattr(package_archive.os, "unlink", fail_published_unlink)

    with pytest.raises(ArchiveError, match="inside package root") as error:
        write_archive(root, destination)

    assert getattr(error.value, "__notes__", [])


def test_write_archive_rejects_parent_swapped_to_source_before_opening_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    destination_parent = tmp_path / "output"
    destination_parent.mkdir()
    destination = destination_parent / "package.agmpkg"
    displaced_parent = tmp_path / "displaced-output"
    open_directory = os.open

    def swap_parent_before_open(
        path: str | Path, flags: int, *args: object, **kwargs: object
    ) -> int:
        if path == destination_parent:
            destination_parent.rename(displaced_parent)
            destination_parent.symlink_to(root, target_is_directory=True)
        return open_directory(path, flags, *args, **kwargs)

    monkeypatch.setattr(package_archive.os, "open", swap_parent_before_open)

    with pytest.raises(ArchiveError, match="cannot write"):
        write_archive(root, destination)

    assert not (root / "package.agmpkg").exists()


@pytest.mark.parametrize("fail_after_open", [False, True])
def test_write_archive_reports_uninspectable_source_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_after_open: bool
) -> None:
    root = _package_tree(tmp_path)
    destination = tmp_path / "package.agmpkg"
    open_directory = os.open
    source_fd: int | None = None
    fstat = os.fstat

    def open_source_root(path: str | Path, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal source_fd
        if path == root and not fail_after_open:
            raise OSError("denied")
        fd = open_directory(path, flags, *args, **kwargs)
        if path == root:
            source_fd = fd
        return fd

    def fail_source_stat(fd: int) -> os.stat_result:
        if fail_after_open and fd == source_fd:
            raise OSError("denied")
        return fstat(fd)

    monkeypatch.setattr(package_archive.os, "open", open_source_root)
    monkeypatch.setattr(package_archive.os, "fstat", fail_source_stat)

    with pytest.raises(ArchiveError, match="cannot inspect package root"):
        write_archive(root, destination)

    assert not destination.exists()


def test_write_archive_reports_uninspectable_opened_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    destination = tmp_path / "package.agmpkg"

    def fail_duplicate(_: int) -> int:
        raise OSError("denied")

    monkeypatch.setattr(package_archive.os, "dup", fail_duplicate)

    with pytest.raises(ArchiveError, match="cannot inspect"):
        write_archive(root, destination)


def test_write_archive_fails_closed_when_directory_publication_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    manifest = root / "package.toml"
    original_manifest = manifest.read_bytes()
    destination_parent = tmp_path / "output"
    destination_parent.mkdir()
    destination = destination_parent / "package.toml"
    displaced_parent = tmp_path / "displaced-output"
    named_temporary = tempfile.NamedTemporaryFile
    monkeypatch.delattr(package_archive.os, "O_DIRECTORY")

    def swap_parent_before_temporary(*args: object, **kwargs: object) -> object:
        destination_parent.rename(displaced_parent)
        destination_parent.symlink_to(root, target_is_directory=True)
        return named_temporary(*args, **kwargs)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", swap_parent_before_temporary)

    with pytest.raises(ArchiveError, match="safe"):
        write_archive(root, destination)

    assert manifest.read_bytes() == original_manifest
    assert not list(root.glob(".package.toml.*.tmp"))


@pytest.mark.parametrize("operation", [os.open, os.rename, os.unlink, os.link])
def test_write_archive_fails_closed_before_publication_without_a_required_dir_fd_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: object
) -> None:
    root = _package_tree(tmp_path)
    destination = tmp_path / "package.agmpkg"
    supported = os.supports_dir_fd - {operation}
    monkeypatch.setattr(package_archive.os, "supports_dir_fd", supported)

    with pytest.raises(ArchiveError, match="safe"):
        write_archive(root, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".package.agmpkg.*"))


def test_write_archive_removes_temporary_file_when_replacement_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    destination = tmp_path / "package.agmpkg"

    def fail_replace(_: str, __: str, **___: object) -> None:
        raise OSError("denied")

    monkeypatch.setattr(package_archive.os, "rename", fail_replace)

    with pytest.raises(ArchiveError, match="cannot write"):
        write_archive(root, destination)

    assert not list(tmp_path.glob(".package.agmpkg.*"))


def test_write_archive_cleans_up_temporary_file_when_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    destination = tmp_path / "package.agmpkg"

    def interrupt_replace(_: str, __: str, **___: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(package_archive.os, "rename", interrupt_replace)

    with pytest.raises(KeyboardInterrupt):
        write_archive(root, destination)

    assert not list(tmp_path.glob(".package.agmpkg.*"))


def test_write_archive_tolerates_a_temporary_file_removed_before_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    destination = tmp_path / "package.agmpkg"

    def remove_temporary_then_fail(
        temporary: str, _: str, *, src_dir_fd: int | None = None, **__: object
    ) -> None:
        assert src_dir_fd is not None
        os.unlink(temporary, dir_fd=src_dir_fd)
        raise OSError("denied")

    monkeypatch.setattr(package_archive.os, "rename", remove_temporary_then_fail)

    with pytest.raises(ArchiveError):
        write_archive(root, destination)


def test_write_archive_retains_the_write_error_when_temporary_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    destination = tmp_path / "package.agmpkg"
    unlink = os.unlink

    def fail_temporary_unlink(path: str, *, dir_fd: int | None = None) -> None:
        if path.startswith(".package.agmpkg."):
            raise OSError("denied")
        unlink(path, dir_fd=dir_fd)

    def fail_replace(_: str, __: str, **___: object) -> None:
        raise OSError("denied")

    monkeypatch.setattr(package_archive.os, "unlink", fail_temporary_unlink)
    monkeypatch.setattr(package_archive.os, "rename", fail_replace)

    with pytest.raises(ArchiveError) as error:
        write_archive(root, destination)

    assert getattr(error.value.__cause__, "__notes__", [])
    monkeypatch.undo()
    for temporary in tmp_path.glob(".package.agmpkg.*"):
        temporary.unlink()


def test_write_archive_reports_uninspectable_destination_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    destination = tmp_path / "package.agmpkg"
    original_stat = Path.stat

    def fail_destination_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path == destination:
            raise OSError("denied")
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", fail_destination_stat)
    with pytest.raises(ArchiveError):
        write_archive(root, destination)

    monkeypatch.undo()
    destination.write_text("archive", encoding="utf-8")
    source = root / "review_tools" / "main.agl"

    def fail_source_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path == source and follow_symlinks:
            raise OSError("denied")
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", fail_source_stat)
    with pytest.raises(ArchiveError):
        write_archive(root, destination)


def test_write_archive_reports_invalid_roots_manifests_and_destinations(tmp_path: Path) -> None:
    with pytest.raises(ArchiveError, match="not a directory"):
        write_archive(tmp_path / "missing", tmp_path / "package.agmpkg")

    package = _package_tree(tmp_path)
    linked = tmp_path / "linked"
    linked.symlink_to(package, target_is_directory=True)
    with pytest.raises(ArchiveError, match="root is a symlink"):
        write_archive(linked, tmp_path / "package.agmpkg")

    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ArchiveError, match="manifest"):
        write_archive(root, tmp_path / "package.agmpkg")

    with pytest.raises(ArchiveError, match="cannot write"):
        write_archive(package, tmp_path / "missing" / "package.agmpkg")


def test_write_archive_reports_unreadable_source_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    unreadable = root / "unreadable"
    unreadable.mkdir()
    destination = tmp_path / "package.agmpkg"
    original_iterdir = Path.iterdir

    def fail_unreadable_directory(path: Path) -> Iterator[Path]:
        if path == unreadable:
            raise OSError("denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_unreadable_directory)
    with pytest.raises(ArchiveError, match="cannot read package directory"):
        write_archive(root, destination)

    assert not destination.exists()


def test_write_archive_reports_uninspectable_source_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    source = root / "review_tools" / "main.agl"
    original_lstat = Path.lstat

    def fail_source_lstat(path: Path) -> os.stat_result:
        if path == source:
            raise OSError("denied")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_source_lstat)
    with pytest.raises(ArchiveError, match="cannot inspect package source"):
        write_archive(root, tmp_path / "package.agmpkg")


def test_write_archive_rejects_unreadable_source_files_and_gitignores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    source = root / "source.txt"
    source.write_text("source", encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def fail_source_read(path: Path) -> bytes:
        if path == source:
            raise OSError("denied")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_source_read)
    with pytest.raises(ArchiveError, match="cannot read package file"):
        write_archive(root, tmp_path / "package.agmpkg")

    monkeypatch.undo()
    (root / ".gitignore").write_bytes(b"\xff")
    with pytest.raises(ArchiveError, match="gitignore"):
        write_archive(root, tmp_path / "package.agmpkg")


def test_write_archive_normalizes_complete_manifest_and_nested_gitignore(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    (root / "package.toml").write_text(
        """[package]
name = "review_tools"
version = "1.2.3"
description = "A review"
license = "MIT"
authors = ["Ada"]
repository = "https://example.test/review"
keywords = ["review"]

[dependencies]
alpha = "1"
tools = { version = "2", path = "../tools" }
remote = { version = "3", url = "https://example.test/remote.agmpkg", hash = "sha256:abc" }

[commands]
"review all" = { program = "review_tools/main::main", description = "Review all" }
quick = { program = "review_tools/main::main" }
""",
        encoding="utf-8",
    )
    nested = root / "nested"
    nested.mkdir()
    (nested / ".gitignore").write_text("/ignored.txt\n!kept.txt\n", encoding="utf-8")
    (nested / "ignored.txt").write_text("ignored", encoding="utf-8")
    (nested / "kept.txt").write_text("kept", encoding="utf-8")
    (root / "RECORD").write_text("old", encoding="utf-8")

    archive_path = tmp_path / "package.agmpkg"
    write_archive(root, archive_path)

    contents = _archive_contents(archive_path)
    prefix = "review_tools-1.2.3/"
    assert 'path = "../tools"' in (root / "package.toml").read_text(encoding="utf-8")
    assert prefix + "nested/ignored.txt" not in contents
    assert prefix + "nested/kept.txt" in contents
    assert contents[prefix + "package.toml"] == (
        b"[package]\n"
        b'name = "review_tools"\n'
        b'version = "1.2.3"\n'
        b'description = "A review"\n'
        b'license = "MIT"\n'
        b'repository = "https://example.test/review"\n'
        b'authors = ["Ada"]\n'
        b'keywords = ["review"]\n\n'
        b"[dependencies]\n"
        b'alpha = "1.0.0"\n'
        b'remote = { version = "3.0.0", url = "https://example.test/remote.agmpkg", '
        b'hash = "sha256:abc" }\n'
        b'tools = "2.0.0"\n\n'
        b"[commands]\n"
        b'quick = { program = "review_tools/main::main" }\n'
        b'"review all" = { program = "review_tools/main::main", description = "Review all" }\n'
    )


def test_write_archive_quotes_unicode_command_keys_in_normalized_manifest(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    (root / "package.toml").write_text(
        """[package]
name = "review_tools"
version = "1.2.3"

[commands]
"café" = { program = "review_tools/main::main" }
""",
        encoding="utf-8",
    )
    archive_path = tmp_path / "package.agmpkg"

    metadata = write_archive(root, archive_path)

    contents = _archive_contents(archive_path)
    assert (
        b'"caf\xc3\xa9" = { program = "review_tools/main::main" }'
        in contents["review_tools-1.2.3/package.toml"]
    )
    assert verify_archive(archive_path) == metadata


def test_nested_gitignore_preserves_basename_wildcard_and_negation_semantics(
    tmp_path: Path,
) -> None:
    root = _package_tree(tmp_path)
    nested = root / "nested"
    (nested / "deeper" / "assets").mkdir(parents=True)
    (nested / "assets").mkdir()
    (nested / ".gitignore").write_text("*.tmp\n!kept.tmp\nassets/*.dat\n", encoding="utf-8")
    for relative in (
        "drop.tmp",
        "kept.tmp",
        "deeper/drop.tmp",
        "deeper/kept.tmp",
        "assets/drop.dat",
        "deeper/assets/drop.dat",
    ):
        (nested / relative).write_text(relative, encoding="utf-8")

    archive_path = tmp_path / "package.agmpkg"
    write_archive(root, archive_path)

    names = _archive_contents(archive_path)
    prefix = "review_tools-1.2.3/nested/"
    assert prefix + "drop.tmp" not in names
    assert prefix + "deeper/drop.tmp" not in names
    assert prefix + "assets/drop.dat" not in names
    assert prefix + "kept.tmp" in names
    assert prefix + "deeper/kept.tmp" in names
    assert prefix + "deeper/assets/drop.dat" in names


def test_nested_gitignore_cannot_reinclude_a_file_in_an_ignored_parent(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    nested = root / "nested"
    nested.mkdir()
    (root / ".gitignore").write_text("nested/\n", encoding="utf-8")
    (nested / ".gitignore").write_text("!kept.txt\n", encoding="utf-8")
    (nested / "kept.txt").write_text("kept", encoding="utf-8")

    archive_path = tmp_path / "package.agmpkg"
    write_archive(root, archive_path)

    assert "review_tools-1.2.3/nested/kept.txt" not in _archive_contents(archive_path)


def test_gitignore_pattern_prefixing_handles_comments_empty_and_negation() -> None:
    assert package_archive._prefixed_pattern("# note", "nested") == "# note"
    assert package_archive._prefixed_pattern("", "nested") == ""
    assert package_archive._prefixed_pattern("!/kept", "nested") == "!nested/kept"
    assert package_archive._prefixed_pattern("ignored", "nested") == "nested/**/ignored"
    assert package_archive._prefixed_pattern("assets/*.tmp", "nested") == "nested/assets/*.tmp"
    assert package_archive._prefixed_pattern("/ignored", "nested") == "nested/ignored"
    assert package_archive._prefixed_pattern("ignored", ".") == "ignored"


@pytest.mark.parametrize(
    "contents",
    [
        [],
        [("review_tools-1.2.3/RECORD", b"")],
        [
            ("review_tools-1.2.3/package.toml", b""),
            ("review_tools-1.2.3/RECORD", b""),
        ],
        [
            ("other/file", b""),
            ("review_tools-1.2.3/RECORD", b""),
            ("review_tools-1.2.3/package.toml", b""),
        ],
        [
            ("review_tools-1.2.3/../escape", b""),
            ("review_tools-1.2.3/RECORD", b""),
            ("review_tools-1.2.3/package.toml", b""),
        ],
        [
            ("review_tools-1.2.3/RECORD", b""),
            ("review_tools-1.2.3/README", b""),
            ("review_tools-1.2.3/package.toml", b""),
            ("review_tools-1.2.3/readme", b""),
        ],
    ],
)
def test_read_archive_metadata_rejects_unsafe_layouts(
    tmp_path: Path, contents: list[tuple[str, bytes]]
) -> None:
    archive_path = tmp_path / "invalid.agmpkg"
    _write_zip(archive_path, contents)

    with pytest.raises(ArchiveError):
        read_archive_metadata(archive_path)


def test_read_archive_metadata_rejects_repeated_entries(tmp_path: Path) -> None:
    archive_path = tmp_path / "duplicate.agmpkg"
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_zip(
            archive_path,
            [
                ("review_tools-1.2.3/RECORD", b""),
                ("review_tools-1.2.3/package.toml", b""),
                ("review_tools-1.2.3/package.toml", b""),
            ],
        )

    with pytest.raises(ArchiveError, match="repeats"):
        read_archive_metadata(archive_path)


def test_read_archive_metadata_rejects_a_symlink_entry(tmp_path: Path) -> None:
    archive_path = tmp_path / "symlink.agmpkg"
    with zipfile.ZipFile(archive_path, "w") as archive:
        record = zipfile.ZipInfo("review_tools-1.2.3/RECORD")
        record.external_attr = 0o120777 << 16
        archive.writestr(record, b"")
        archive.writestr("review_tools-1.2.3/package.toml", b"")

    with pytest.raises(ArchiveError, match="symlink"):
        read_archive_metadata(archive_path)


def test_archive_readers_wrap_unreadable_archives(tmp_path: Path) -> None:
    missing = tmp_path / "missing.agmpkg"

    with pytest.raises(ArchiveError, match="cannot read"):
        read_archive_metadata(missing)
    with pytest.raises(ArchiveError, match="cannot read"):
        verify_archive(missing)


def test_archive_reader_rejects_manifest_prefix_and_record_errors(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    archive_path = tmp_path / "package.agmpkg"
    write_archive(root, archive_path)
    contents = _archive_contents(archive_path)
    prefix = "review_tools-1.2.3/"

    contents[prefix + "package.toml"] = b"[package\n"
    _write_zip(archive_path, list(contents.items()))
    with pytest.raises(ArchiveError, match="manifest"):
        read_archive_metadata(archive_path)

    write_archive(root, archive_path)
    contents = _archive_contents(archive_path)
    contents[prefix + "RECORD"] = b"\xff"
    _write_zip(archive_path, list(contents.items()))
    with pytest.raises(ArchiveError, match="RECORD"):
        read_archive_metadata(archive_path)

    write_archive(root, archive_path)
    contents = _archive_contents(archive_path)
    contents[prefix + "RECORD"] = contents[prefix + "RECORD"].replace(b"\n", b"\r\n")
    _write_zip(archive_path, list(contents.items()))
    with pytest.raises(ArchiveError, match="canonical"):
        read_archive_metadata(archive_path)


def test_archive_reader_rejects_unsorted_record_and_mismatched_prefix(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    archive_path = tmp_path / "package.agmpkg"
    write_archive(root, archive_path)
    contents = _archive_contents(archive_path)
    prefix = "review_tools-1.2.3/"
    record = contents[prefix + "RECORD"].decode().splitlines()
    contents[prefix + "RECORD"] = ("\n".join(reversed(record)) + "\n").encode()
    _write_zip(archive_path, list(contents.items()))
    with pytest.raises(ArchiveError, match="sorted"):
        read_archive_metadata(archive_path)

    write_archive(root, archive_path)
    contents = _archive_contents(archive_path)
    renamed = {name.replace(prefix, "other-1.2.3/"): value for name, value in contents.items()}
    _write_zip(archive_path, list(renamed.items()))
    with pytest.raises(ArchiveError, match="prefix"):
        read_archive_metadata(archive_path)


@pytest.mark.parametrize("component", ["CON", "aux.txt", "trailing.", "trailing ", "bad?.txt"])
def test_write_archive_rejects_windows_portability_invalid_components(
    tmp_path: Path, component: str
) -> None:
    root = _package_tree(tmp_path)
    (root / component).write_text("bad", encoding="utf-8")

    with pytest.raises(ArchiveError, match="component"):
        write_archive(root, tmp_path / "package.agmpkg")


def test_archive_readers_reject_noncanonical_zip_entry_metadata_and_flags(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    archive_path = tmp_path / "package.agmpkg"
    write_archive(root, archive_path)
    contents = _archive_contents(archive_path)

    _write_zip(archive_path, list(contents.items()))
    raw = bytearray(archive_path.read_bytes())
    central_directory = raw.index(b"PK\x01\x02")
    raw[central_directory + 8] |= 0x01
    archive_path.write_bytes(raw)

    with pytest.raises(ArchiveError, match="metadata"):
        read_archive_metadata(archive_path)


def test_archive_readers_reject_zip_archive_comments(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    archive_path = tmp_path / "package.agmpkg"
    write_archive(root, archive_path)
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.comment = b"noncanonical"

    with pytest.raises(ArchiveError, match="metadata"):
        read_archive_metadata(archive_path)


def test_archive_readers_reject_windows_invalid_names_and_record_paths(tmp_path: Path) -> None:
    root = _package_tree(tmp_path)
    archive_path = tmp_path / "package.agmpkg"
    write_archive(root, archive_path)
    contents = _archive_contents(archive_path)
    prefix = "review_tools-1.2.3/"
    contents[prefix + "CON"] = b"invalid"
    _write_zip(archive_path, list(contents.items()))

    with pytest.raises(ArchiveError, match="component"):
        read_archive_metadata(archive_path)

    write_archive(root, archive_path)
    contents = _archive_contents(archive_path)
    contents[prefix + "RECORD"] = ("CON,sha256=" + "0" * 64 + "\n").encode()
    _write_zip(archive_path, list(contents.items()))

    with pytest.raises(ArchiveError, match="component"):
        read_archive_metadata(archive_path)


def test_verify_archive_streams_entries_without_zipfile_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    (root / "large.agl").write_bytes(b"x" * 100_000)
    archive_path = tmp_path / "package.agmpkg"
    metadata = write_archive(root, archive_path)

    def fail_read(*_: object, **__: object) -> bytes:
        pytest.fail("verification must stream entries rather than ZipFile.read")

    monkeypatch.setattr(zipfile.ZipFile, "read", fail_read)

    assert verify_archive(archive_path) == metadata


def test_archive_readers_enforce_explicit_entry_and_size_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package_tree(tmp_path)
    archive_path = tmp_path / "package.agmpkg"
    write_archive(root, archive_path)

    monkeypatch.setattr(package_archive, "MAX_ARCHIVE_ENTRIES", 2)
    with pytest.raises(ArchiveError, match="entries"):
        read_archive_metadata(archive_path)

    monkeypatch.setattr(package_archive, "MAX_ARCHIVE_ENTRIES", 10_000)
    monkeypatch.setattr(package_archive, "MAX_ARCHIVE_ENTRY_SIZE", 1)
    with pytest.raises(ArchiveError, match="size"):
        verify_archive(archive_path)

    monkeypatch.setattr(package_archive, "MAX_ARCHIVE_ENTRY_SIZE", 64 * 1024 * 1024)
    monkeypatch.setattr(package_archive, "MAX_ARCHIVE_TOTAL_SIZE", 1)
    with pytest.raises(ArchiveError, match="total"):
        read_archive_metadata(archive_path)


def test_archive_readers_reject_directory_entries(tmp_path: Path) -> None:
    archive_path = tmp_path / "directory.agmpkg"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("review_tools-1.2.3/directory/", b"")

    with pytest.raises(ArchiveError, match="directory"):
        read_archive_metadata(archive_path)


class _BytesArchive:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def open(self, _: zipfile.ZipInfo) -> io.BytesIO:
        return io.BytesIO(self.content)


def test_stream_reading_rejects_entries_that_exceed_or_misstate_their_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(package_archive, "MAX_ARCHIVE_ENTRY_SIZE", 1)
    oversized = zipfile.ZipInfo("file")
    oversized.file_size = 1
    archive = cast(zipfile.ZipFile, _BytesArchive(b"xx"))

    with pytest.raises(ArchiveError, match="exceeds"):
        package_archive._read_entry(archive, oversized)
    with pytest.raises(ArchiveError, match="exceeds"):
        package_archive._entry_digest(archive, oversized)

    monkeypatch.setattr(package_archive, "MAX_ARCHIVE_ENTRY_SIZE", 2)
    short = zipfile.ZipInfo("file")
    short.file_size = 2
    archive = cast(zipfile.ZipFile, _BytesArchive(b"x"))

    with pytest.raises(ArchiveError, match="does not match"):
        package_archive._read_entry(archive, short)
    with pytest.raises(ArchiveError, match="does not match"):
        package_archive._entry_digest(archive, short)


def test_verify_archive_rejects_a_non_normalized_manifest_with_a_matching_record(
    tmp_path: Path,
) -> None:
    root = _package_tree(tmp_path)
    archive_path = tmp_path / "package.agmpkg"
    write_archive(root, archive_path)
    contents = _archive_contents(archive_path)
    prefix = "review_tools-1.2.3/"
    manifest_name = prefix + "package.toml"
    contents[manifest_name] = b'[package]\nversion = "1.2.3"\nname = "review_tools"\n'
    entries = tuple(
        RecordEntry(name.removeprefix(prefix), hashlib.sha256(value).hexdigest())
        for name, value in sorted(contents.items())
        if name != prefix + "RECORD"
    )
    contents[prefix + "RECORD"] = serialize_record(entries).encode()
    _write_zip(archive_path, list(contents.items()))

    with pytest.raises(ArchiveError, match="not normalized"):
        verify_archive(archive_path)
