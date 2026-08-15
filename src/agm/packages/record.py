"""Creation and verification of installed-package ``RECORD`` files."""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agm.core import fs
from agm.core.path import is_portable_relative_path

RECORD_NAME = "RECORD"
_DIGEST_PREFIX = "sha256="
_LINE_BREAKS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")


class RecordError(ValueError):
    """Raised when a package ``RECORD`` is malformed or does not verify."""


@dataclass(frozen=True, slots=True)
class RecordEntry:
    """One relative package file path and its SHA-256 digest."""

    path: str
    digest: str


def write_record(root: Path) -> Path:
    """Write a deterministic SHA-256 manifest for every file below *root*.

    ``RECORD`` itself is omitted because hashing it would be self-referential.
    """

    record_path = root / RECORD_NAME
    entries = record_entries(root)
    content = serialize_record(entries)
    fs.write_text(record_path, content)
    return record_path


def record_entries(root: Path) -> tuple[RecordEntry, ...]:
    """Return the canonical record entries for a package tree without writing it."""

    return tuple(
        RecordEntry(_record_path(path.relative_to(root).as_posix()), file_digest(path))
        for path in _package_files(root)
    )


def validate_package_tree(root: Path) -> None:
    """Check that *root* is a package tree AGM can record, without hashing it.

    Callers that only need to know a directory is eligible (regular files and
    directories, no symlinks or special nodes, portable relative paths) use
    this instead of building record entries, which would read and hash every
    file only to discard the digests.
    """

    _package_tree(root)


def read_record(root: Path) -> tuple[RecordEntry, ...]:
    """Read and validate the package ``RECORD`` below *root*."""

    _package_tree(root)
    return _read_record_entries(root / RECORD_NAME)


def _read_record_entries(record_path: Path) -> tuple[RecordEntry, ...]:
    """Read and parse ``RECORD`` content at *record_path*."""

    try:
        content = fs.read_text(record_path)
    except (OSError, UnicodeDecodeError) as exc:
        raise RecordError(f"cannot read package record {record_path}: {exc}") from exc
    return parse_record(content)


def serialize_record(entries: tuple[RecordEntry, ...]) -> str:
    """Serialize record entries in the canonical installed-package format."""

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for entry in entries:
        row: list[str] = [entry.path, f"{_DIGEST_PREFIX}{entry.digest}"]
        writer.writerow(row)
    return buffer.getvalue()


def content_hash(entries: tuple[RecordEntry, ...]) -> str:
    """Return the stable content identity represented by canonical record entries."""

    return hashlib.sha256(serialize_record(entries).encode()).hexdigest()


def parse_record(content: str) -> tuple[RecordEntry, ...]:
    """Parse and validate package ``RECORD`` content without reading a tree."""

    try:
        rows = list(csv.reader(io.StringIO(content), strict=True))
    except csv.Error as exc:
        raise RecordError(f"package record has a malformed entry: {exc}") from exc
    entries: list[RecordEntry] = []
    seen: set[str] = set()
    for row in rows:
        if len(row) != 2:
            raise RecordError("package record entries must have a path and SHA-256 digest")
        path = _record_path(row[0])
        if path in seen:
            raise RecordError(f"package record repeats path {path!r}")
        seen.add(path)
        entries.append(RecordEntry(path, _record_digest(row[1])))
    return tuple(entries)


def verify_record(root: Path) -> tuple[RecordEntry, ...]:
    """Verify that *root* exactly matches its ``RECORD`` contents."""

    record_path = root / RECORD_NAME
    paths = _package_tree(root)
    entries = _read_record_entries(record_path)
    expected = {entry.path: entry.digest for entry in entries}
    actual = {
        path.relative_to(root).as_posix(): path
        for path in paths
        if path.is_file() and path != record_path
    }
    if expected.keys() != actual.keys():
        raise RecordError("package files do not match RECORD")
    for path, digest in expected.items():
        if not hmac.compare_digest(digest, file_digest(actual[path])):
            raise RecordError(f"SHA-256 mismatch for package file {path!r}")
    return entries


def _package_files(root: Path) -> tuple[Path, ...]:
    """Return sorted package files, excluding the root ``RECORD`` itself."""

    record_path = root / RECORD_NAME
    return tuple(path for path in _package_tree(root) if path.is_file() and path != record_path)


def record_path_key(entry: RecordEntry) -> str:
    """Return the canonical ordering key for one record entry."""

    return entry.path


def posix_relative_path(root: Path) -> Callable[[Path], str]:
    """Return the portable ordering key for paths below *root*."""

    def key(path: Path) -> str:
        return path.relative_to(root).as_posix()

    return key


def walk_package_tree(
    root: Path,
    *,
    accept: Callable[[Path, int], None],
    on_directory_error: Callable[[Path, OSError], Exception],
    on_entry_error: Callable[[Path, OSError], Exception],
) -> tuple[Path, ...]:
    """Recursively enumerate every descendant of *root*, sorted by portable path.

    Every child is ``lstat``-ed and passed to *accept* along with its raw mode;
    *accept* raises the caller's own error for anything it rejects. Accepted
    directories are recursed into. Directory-listing and ``lstat`` failures are
    translated to exceptions through *on_directory_error* and *on_entry_error*.
    """

    paths: list[Path] = []

    def visit(directory: Path) -> None:
        try:
            children = list(directory.iterdir())
        except OSError as exc:
            raise on_directory_error(directory, exc) from exc
        for child in children:
            try:
                mode = child.lstat().st_mode
            except OSError as exc:
                raise on_entry_error(child, exc) from exc
            accept(child, mode)
            paths.append(child)
            if stat.S_ISDIR(mode):
                visit(child)

    visit(root)
    return tuple(sorted(paths, key=posix_relative_path(root)))


def _package_tree(root: Path) -> tuple[Path, ...]:
    """Validate and return every descendant of *root*, sorted by portable path.

    Rejects a symlinked root, symlink descendants, special filesystem nodes,
    and paths containing a line break, in a single traversal of the tree.
    """

    def wrap(_path: Path, exc: OSError) -> RecordError:
        return RecordError(f"cannot inspect package tree {root}: {exc}")

    def accept(child: Path, mode: int) -> None:
        if stat.S_ISLNK(mode):
            raise RecordError(f"package contains symlink {child}")
        if not stat.S_ISREG(mode) and not stat.S_ISDIR(mode):
            raise RecordError(f"package contains unsupported filesystem node {child}")
        if any(character in _LINE_BREAKS for character in child.relative_to(root).as_posix()):
            raise RecordError(f"package path contains a line break {child}")

    try:
        is_root_symlink = root.is_symlink()
    except OSError as exc:
        raise wrap(root, exc) from exc
    if is_root_symlink:
        raise RecordError(f"package root is a symlink {root}")
    return walk_package_tree(root, accept=accept, on_directory_error=wrap, on_entry_error=wrap)


def file_digest(path: Path) -> str:
    """Return the lowercase SHA-256 hexadecimal digest of *path*."""

    try:
        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()
    except OSError as exc:
        raise RecordError(f"cannot hash package file {path}: {exc}") from exc


def _record_path(value: str) -> str:
    """Validate and normalize one serialized relative POSIX path."""

    if (
        not is_portable_relative_path(value)
        or any(character in _LINE_BREAKS for character in value)
        or value == RECORD_NAME
    ):
        raise RecordError(f"invalid package record path {value!r}")
    return value


def _record_digest(value: str) -> str:
    """Validate and return one serialized SHA-256 digest."""

    digest = value.removeprefix(_DIGEST_PREFIX)
    if (
        not value.startswith(_DIGEST_PREFIX)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RecordError("package record has an invalid SHA-256 digest")
    return digest
