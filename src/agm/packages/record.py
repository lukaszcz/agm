"""Creation and verification of installed-package ``RECORD`` files."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from agm.core import fs

_RECORD_NAME = "RECORD"
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

    record_path = root / _RECORD_NAME
    entries = tuple(
        RecordEntry(_record_path(path.relative_to(root).as_posix()), _sha256(path))
        for path in _package_files(root)
    )
    content = serialize_record(entries)
    fs.write_text(record_path, content)
    return record_path


def read_record(root: Path) -> tuple[RecordEntry, ...]:
    """Read and validate the package ``RECORD`` below *root*."""

    _validate_package_tree(root)
    record_path = root / _RECORD_NAME
    try:
        content = fs.read_text(record_path)
    except (OSError, UnicodeDecodeError) as exc:
        raise RecordError(f"cannot read package record {record_path}: {exc}") from exc

    return parse_record(content)


def serialize_record(entries: tuple[RecordEntry, ...]) -> str:
    """Serialize record entries in the canonical installed-package format."""

    return "".join(_serialize_entry(entry) for entry in entries)


def parse_record(content: str) -> tuple[RecordEntry, ...]:
    """Parse and validate package ``RECORD`` content without reading a tree."""

    entries: list[RecordEntry] = []
    seen: set[str] = set()
    for line in content.splitlines():
        path_value, digest_value = _parse_row(line)
        path = _record_path(path_value)
        if path in seen:
            raise RecordError(f"package record repeats path {path!r}")
        seen.add(path)
        entries.append(RecordEntry(path, _record_digest(digest_value)))
    return tuple(entries)


def verify_record(root: Path) -> tuple[RecordEntry, ...]:
    """Verify that *root* exactly matches its ``RECORD`` contents."""

    entries = read_record(root)
    expected = {entry.path: entry.digest for entry in entries}
    actual = {path.relative_to(root).as_posix(): path for path in _package_files(root)}
    if expected.keys() != actual.keys():
        raise RecordError("package files do not match RECORD")
    for path, digest in expected.items():
        if not hmac.compare_digest(digest, _sha256(actual[path])):
            raise RecordError(f"SHA-256 mismatch for package file {path!r}")
    return entries


def _serialize_entry(entry: RecordEntry) -> str:
    """Serialize one two-column CSV record without depending on untyped CSV I/O."""

    return f"{_escape_field(entry.path)},{_DIGEST_PREFIX}{entry.digest}\n"


def _escape_field(value: str) -> str:
    """Escape a CSV field used for one relative path."""

    if any(character in value for character in ',"'):
        return '"' + value.replace('"', '""') + '"'
    return value


def _parse_row(line: str) -> tuple[str, str]:
    """Parse one two-column CSV row, rejecting malformed or extra columns."""

    fields: list[str] = []
    index = 0
    while index < len(line):
        field, index = _parse_field(line, index)
        fields.append(field)
        if index == len(line):
            break
        index += 1
    if len(fields) != 2:
        raise RecordError("package record entries must have a path and SHA-256 digest")
    return fields[0], fields[1]


def _parse_field(line: str, index: int) -> tuple[str, int]:
    """Parse one CSV field and return it with its following delimiter index."""

    if line[index] != '"':
        end = line.find(",", index)
        return (line[index:] if end == -1 else line[index:end], len(line) if end == -1 else end)

    index += 1
    value: list[str] = []
    while index < len(line):
        character = line[index]
        if character != '"':
            value.append(character)
            index += 1
            continue
        index += 1
        if index < len(line) and line[index] == '"':
            value.append('"')
            index += 1
            continue
        if index != len(line) and line[index] != ",":
            raise RecordError("package record has malformed quoted path")
        return "".join(value), index
    raise RecordError("package record has unterminated quoted path")


def _package_files(root: Path) -> tuple[Path, ...]:
    """Return sorted package files, excluding the root ``RECORD`` itself."""

    _validate_package_tree(root)
    files = [path for path in fs.rglob(root, "*") if path.is_file() and path != root / _RECORD_NAME]
    return tuple(sorted(files, key=_posix_relative_path(root)))


def _posix_relative_path(root: Path) -> Callable[[Path], str]:
    """Return the portable ordering key for paths below *root*."""

    def key(path: Path) -> str:
        return path.relative_to(root).as_posix()

    return key


def _validate_package_tree(root: Path) -> None:
    """Reject package roots and descendants that are symbolic links."""

    if root.is_symlink():
        raise RecordError(f"package root is a symlink {root}")
    for path in fs.rglob(root, "*"):
        if path.is_symlink():
            raise RecordError(f"package contains symlink {path}")
        if any(character in _LINE_BREAKS for character in path.relative_to(root).as_posix()):
            raise RecordError(f"package path contains a line break {path}")


def _sha256(path: Path) -> str:
    """Return the lowercase SHA-256 hexadecimal digest of *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record_path(value: str) -> str:
    """Validate and normalize one serialized relative POSIX path."""

    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or "\\" in value
        or any(character in _LINE_BREAKS for character in value)
        or path.is_absolute()
        or windows_path.drive
        or windows_path.root
        or any(part in {".", ".."} for part in path.parts)
        or path.as_posix() != value
        or value == _RECORD_NAME
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
