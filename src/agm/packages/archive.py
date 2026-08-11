"""Deterministic, portable package archive creation and verification."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import zipfile
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, TYPE_CHECKING, TypeVar, cast

from pathspec import PathSpec

from agm.packages.manifest import (
    ManifestError,
    PackageManifest,
    distribution_manifest,
    load_manifest,
    load_manifest_text,
)
from agm.packages.record import (
    RecordEntry,
    RecordError,
    content_hash,
    parse_record,
    serialize_record,
)

if TYPE_CHECKING:
    from agm.packages.model import PackageInfo

_MANIFEST_NAME = "package.toml"
_RECORD_NAME = "RECORD"
_ARCHIVE_SUFFIX = ".agmpkg"
_VCS_DIRECTORIES = frozenset({".git", ".hg", ".svn", ".bzr", "CVS"})
_CACHE_DIRECTORIES = frozenset(
    {"__pycache__", ".cache", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_FILE_MODE = 0o100644
_UTF8_FLAG = 0x800
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"
_CENTRAL_DIRECTORY_HEADER_SIZE = 46
_HASH_CHUNK_SIZE = 1024 * 1024
_DIR_FD_PUBLICATION_OPERATIONS = (os.open, os.rename, os.unlink, os.link)

# These limits apply before decompression and while reading, so untrusted archives
# cannot make metadata inspection or verification consume unbounded resources.
MAX_ARCHIVE_ENTRIES = 10_000
MAX_ARCHIVE_CENTRAL_DIRECTORY_SIZE = 16 * 1024 * 1024
MAX_ARCHIVE_PATH_COMPONENTS = 256
MAX_ARCHIVE_ENTRY_SIZE = 64 * 1024 * 1024
MAX_ARCHIVE_TOTAL_SIZE = 512 * 1024 * 1024

_WRITE_FAILURES = (
    OSError,
    RuntimeError,
    NotImplementedError,
    ValueError,
    zipfile.LargeZipFile,
    zlib.error,
)

T = TypeVar("T")


class ArchiveError(ValueError):
    """Raised when a package archive cannot be safely created or verified."""


@dataclass(frozen=True, slots=True)
class ArchiveMetadata:
    """Package identity and canonical content hash read from an archive."""

    manifest: PackageManifest
    package_hash: str


@dataclass(frozen=True, slots=True)
class _DirectoryIdentity:
    """Stable device/inode identity of an open directory."""

    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _SourceRoot:
    """An opened source-root capability retained through publication."""

    fd: int


def write_archive(package_root: Path, destination: Path) -> ArchiveMetadata:
    """Write a deterministic ``.agmpkg`` archive from a quiescent *package_root*.

    The writer never intentionally modifies a source tree that remains unchanged.
    It rejects ordinary source links and destination aliases, and rolls back a
    publication when it can verify that its opened output parent entered the
    source tree. Concurrent source or namespace mutation is unsupported: POSIX
    has no atomic operation to prove that an already-open output parent remains
    outside a concurrently renameable source tree. Detected races raise
    :class:`ArchiveError`.
    """

    root = _package_root(package_root)
    if not _directory_publication_supported():
        raise ArchiveError(
            "cannot safely publish package archive: directory-relative operations are unavailable"
        )
    source_root = _capture_source_root(root)
    try:
        _reject_source_links(root)
        _validate_destination(root, destination)
        manifest, prefix, contents = _archive_distribution(root)
        metadata = _metadata(manifest, contents)
        _write_archive_in_open_parent(source_root, destination, prefix, contents)
        return metadata
    finally:
        os.close(source_root.fd)


def validate_archive_source(
    package_root: Path, *, dependency_packages: Iterable[PackageInfo] = ()
) -> PackageManifest:
    """Validate the portable archive view of a package without writing it.

    This applies package discipline to the files selected for an archive, using
    resolved dependency modules for resource re-exports, so ignored or otherwise
    excluded resources cannot pass source-tree validation and then disappear from
    the distribution.
    """

    root = _package_root(package_root)
    _reject_source_links(root)
    manifest, _, contents = _archive_distribution(root)
    # Import lazily so ordinary archive creation stays independent of the AgL
    # parser; callers that request package creation explicitly need discipline.
    from agm.packages.discipline import DisciplineError, validate_archive_package

    try:
        validate_archive_package(
            manifest,
            archive_paths=contents,
            read_module=lambda path: contents[path].decode(),
            dependency_packages=dependency_packages,
        )
    except DisciplineError as exc:
        raise ArchiveError(f"archive package violates discipline: {exc}") from exc
    return manifest


def _directory_publication_supported() -> bool:
    """Whether the platform can address files relative to an open directory."""

    return (
        os.name == "posix"
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and all(
            operation in cast(set[object], os.supports_dir_fd)
            for operation in _DIR_FD_PUBLICATION_OPERATIONS
        )
    )


def _write_archive_in_open_parent(
    source_root: _SourceRoot,
    destination: Path,
    prefix: str,
    contents: dict[str, bytes],
) -> None:
    """Publish through an opened parent and undo a detected containment race."""

    parent_fd: int | None = None
    temporary: str | None = None
    replaced: str | None = None
    try:
        parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        _reject_open_parent_within_root(source_root, parent_fd, destination)
        temporary, temporary_fd = _create_temporary(destination.name, parent_fd)
        with os.fdopen(temporary_fd, "w+b") as file:
            _write_zip(file, prefix, contents)
        # POSIX cannot combine this ancestry check with rename.  Preserve the
        # destination by a bound-directory hard link so a relocation in that
        # seam can be rolled back without addressing the source by pathname.
        replaced = _preserve_destination(destination.name, parent_fd)
        _reject_open_parent_within_root(source_root, parent_fd, destination)
        os.rename(
            temporary,
            destination.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary = None
        try:
            _reject_open_parent_within_root(source_root, parent_fd, destination)
        except ArchiveError as error:
            _rollback_published_archive(destination.name, replaced, parent_fd, error)
            replaced = None
            raise
        _remove_temporary_at(replaced, parent_fd, RuntimeError())
        replaced = None
    except ArchiveError:
        _remove_temporary_at(temporary, parent_fd, RuntimeError())
        _remove_temporary_at(replaced, parent_fd, RuntimeError())
        raise
    except BaseException as exc:
        _remove_temporary_at(temporary, parent_fd, exc)
        _remove_temporary_at(replaced, parent_fd, exc)
        if isinstance(exc, _WRITE_FAILURES):
            raise ArchiveError(f"cannot write package archive {destination}: {exc}") from exc
        raise
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def _preserve_destination(destination_name: str, parent_fd: int) -> str | None:
    """Hard-link an existing destination to a private sibling for possible rollback."""

    preserved = f".{destination_name}.{secrets.token_hex(16)}.bak"
    try:
        os.link(
            destination_name,
            preserved,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    return preserved


def _rollback_published_archive(
    destination_name: str, replaced: str | None, parent_fd: int, error: ArchiveError
) -> None:
    """Undo a publication through the same parent capability that performed it."""

    try:
        if replaced is None:
            os.unlink(destination_name, dir_fd=parent_fd)
        else:
            os.rename(
                replaced,
                destination_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
    except OSError as exc:
        error.add_note(f"cannot roll back package archive publication: {exc}")


def _reject_open_parent_within_root(
    source_root: _SourceRoot, parent_fd: int, destination: Path
) -> None:
    """Reject a publication capability inside the retained source-root capability."""

    try:
        source_identity = _directory_identity(os.fstat(source_root.fd))
        current_fd = os.dup(parent_fd)
        try:
            while True:
                current_identity = _directory_identity(os.fstat(current_fd))
                if current_identity == source_identity:
                    raise ArchiveError(
                        f"package archive destination is inside package root {destination}"
                    )
                ancestor_fd = os.open("..", os.O_RDONLY | os.O_DIRECTORY, dir_fd=current_fd)
                ancestor_identity = _directory_identity(os.fstat(ancestor_fd))
                if ancestor_identity == current_identity:
                    os.close(ancestor_fd)
                    return
                os.close(current_fd)
                current_fd = ancestor_fd
        finally:
            os.close(current_fd)
    except ArchiveError:
        raise
    except OSError as exc:
        raise ArchiveError(
            f"cannot inspect package archive destination {destination}: {exc}"
        ) from exc


def _create_temporary(destination_name: str, parent_fd: int) -> tuple[str, int]:
    """Create a private temporary archive file relative to *parent_fd*."""

    temporary = f".{destination_name}.{secrets.token_hex(16)}.tmp"
    return temporary, os.open(
        temporary,
        os.O_RDWR | os.O_CREAT | os.O_EXCL,
        0o600,
        dir_fd=parent_fd,
    )


def _write_zip(file: IO[bytes], prefix: str, contents: dict[str, bytes]) -> None:
    """Render canonical archive contents to an already-open binary file."""

    with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(contents):
            archive.writestr(_zip_info(prefix + path), contents[path])


def _remove_temporary_at(
    temporary: str | None, parent_fd: int | None, write_error: BaseException
) -> None:
    """Remove a failed directory-relative temporary without masking its write error."""

    if temporary is None or parent_fd is None:
        return
    try:
        os.unlink(temporary, dir_fd=parent_fd)
    except FileNotFoundError:
        pass
    except OSError as exc:
        write_error.add_note(f"cannot remove temporary package archive {temporary}: {exc}")


def read_archive_manifest(archive_path: Path) -> PackageManifest:
    """Read a validated archive manifest without extracting the archive."""

    return read_archive_metadata(archive_path).manifest


def read_archive_metadata(archive_path: Path) -> ArchiveMetadata:
    """Read bounded, validated archive identity metadata without extracting it."""

    def read_metadata(archive: zipfile.ZipFile) -> ArchiveMetadata:
        prefix, manifest_name, record_name, infos = _archive_layout(archive.infolist())
        manifest = _manifest_from_bytes(_read_entry(archive, infos[manifest_name]))
        _validate_prefix(prefix, manifest)
        entries = _record_from_bytes(_read_entry(archive, infos[record_name]))
        return ArchiveMetadata(manifest, _package_hash(entries))

    return _read_archive(archive_path, read_metadata)


def verify_archive(archive_path: Path) -> ArchiveMetadata:
    """Stream-verify layout, normalized manifest, and every ``RECORD`` digest."""

    return _read_archive(archive_path, _verify_open_archive)


def verify_archive_discipline(
    archive_path: Path, *, dependency_packages: Iterable[PackageInfo] = ()
) -> ArchiveMetadata:
    """Verify an archive and validate its discipline against dependencies without extraction."""

    def verify_and_validate(archive: zipfile.ZipFile) -> ArchiveMetadata:
        metadata = _verify_open_archive(archive)
        prefix, _, _, infos = _archive_layout(archive.infolist())
        # Import lazily so archive creation and verification remain independent
        # of the AgL parser unless a caller explicitly asks for discipline.
        from agm.packages.discipline import DisciplineError, validate_archive_package

        try:
            validate_archive_package(
                metadata.manifest,
                archive_paths=(name.removeprefix(prefix) for name in infos),
                read_module=lambda path: _read_entry(archive, infos[prefix + path]).decode(),
                dependency_packages=dependency_packages,
            )
        except DisciplineError as exc:
            raise ArchiveError(f"archive package violates discipline: {exc}") from exc
        return metadata

    return _read_archive(archive_path, verify_and_validate)


def extract_archive(
    archive_path: Path, destination: Path | Callable[[ArchiveMetadata], Path]
) -> ArchiveMetadata:
    """Verify then safely extract an archive into an empty private destination.

    A callable destination is selected from the verified metadata before any
    write while the archive remains open. The caller owns publication of the
    extracted directory. Archive paths are validated before any write, and the
    extracted tree retains the archive's canonical ``RECORD`` for the
    installed-tree integrity contract.
    """

    def extract(archive: zipfile.ZipFile) -> ArchiveMetadata:
        metadata = _verify_open_archive(archive)
        extraction_root = destination(metadata) if callable(destination) else destination
        prefix, _, record_name, infos = _archive_layout(archive.infolist())
        for name, info in infos.items():
            if name == record_name:
                continue
            relative = name.removeprefix(prefix)
            target = extraction_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("xb") as output:
                while chunk := source.read(_HASH_CHUNK_SIZE):
                    output.write(chunk)
        record_target = extraction_root / _RECORD_NAME
        record_target.write_bytes(_read_entry(archive, infos[record_name]))
        return metadata

    return _read_archive(archive_path, extract)


def _verify_open_archive(archive: zipfile.ZipFile) -> ArchiveMetadata:
    """Verify an opened archive and return its identity metadata."""

    prefix, manifest_name, record_name, infos = _archive_layout(archive.infolist())
    manifest_bytes = _read_entry(archive, infos[manifest_name])
    manifest = _manifest_from_bytes(manifest_bytes)
    _validate_prefix(prefix, manifest)
    entries = _record_from_bytes(_read_entry(archive, infos[record_name]))
    expected_entries = tuple(
        RecordEntry(
            name.removeprefix(prefix),
            hashlib.sha256(manifest_bytes).hexdigest()
            if name == manifest_name
            else _entry_digest(archive, info),
        )
        for name, info in infos.items()
        if name != record_name
    )
    if manifest_bytes != _normalized_manifest(manifest):
        raise ArchiveError("archive package manifest is not normalized")
    if entries != expected_entries:
        raise ArchiveError("archive contents do not match RECORD")
    return ArchiveMetadata(manifest, _package_hash(entries))


def _read_archive(archive_path: Path, operation: Callable[[zipfile.ZipFile], T]) -> T:
    """Run an archive read operation while presenting ZIP failures as domain errors."""

    try:
        with archive_path.open("rb") as archive_file:
            _validate_central_directory_limits(archive_file)
            with zipfile.ZipFile(archive_file) as archive:
                if archive.comment:
                    raise ArchiveError("package archive has noncanonical ZIP metadata")
                return operation(archive)
    except ArchiveError:
        raise
    except (
        OSError,
        EOFError,
        KeyError,
        RuntimeError,
        NotImplementedError,
        ValueError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        zlib.error,
    ) as exc:
        raise ArchiveError(f"cannot read package archive {archive_path}: {exc}") from exc


def _validate_central_directory_limits(archive: IO[bytes]) -> None:
    """Reject ZIP64, malformed, and oversized directories before :mod:`zipfile` reads them."""

    archive.seek(0, os.SEEK_END)
    file_size = archive.tell()
    tail_size = min(file_size, 65_577)
    archive.seek(-tail_size, os.SEEK_END)
    tail = archive.read(tail_size)
    signature = b"PK\x05\x06"
    search_end = len(tail)
    while (offset := tail.rfind(signature, 0, search_end)) >= 0:
        if offset + 22 <= len(tail):
            entries = int.from_bytes(tail[offset + 10 : offset + 12], "little")
            directory_size = int.from_bytes(tail[offset + 12 : offset + 16], "little")
            comment_size = int.from_bytes(tail[offset + 20 : offset + 22], "little")
            if offset + 22 + comment_size == len(tail):
                if tail[offset - 20 : offset - 16] == _ZIP64_LOCATOR_SIGNATURE:
                    raise ArchiveError("ZIP64 package archives are not supported")
                if entries == 0xFFFF or entries > MAX_ARCHIVE_ENTRIES:
                    raise ArchiveError("package archive exceeds the entries limit")
                if directory_size > MAX_ARCHIVE_CENTRAL_DIRECTORY_SIZE:
                    raise ArchiveError("package archive central directory exceeds the size limit")
                directory_start = file_size - tail_size + offset - directory_size
                if directory_start < 0:
                    raise ArchiveError("package archive has an invalid central directory")
                archive.seek(directory_start)
                remaining = directory_size
                actual_entries = 0
                while remaining:
                    header = archive.read(_CENTRAL_DIRECTORY_HEADER_SIZE)
                    if (
                        len(header) != _CENTRAL_DIRECTORY_HEADER_SIZE
                        or header[:4] != _CENTRAL_DIRECTORY_SIGNATURE
                    ):
                        raise ArchiveError("package archive has an invalid central directory")
                    variable_size = sum(
                        int.from_bytes(header[start : start + 2], "little")
                        for start in (28, 30, 32)
                    )
                    record_size = _CENTRAL_DIRECTORY_HEADER_SIZE + variable_size
                    if record_size > remaining:
                        raise ArchiveError("package archive has an invalid central directory")
                    actual_entries += 1
                    if actual_entries > MAX_ARCHIVE_ENTRIES:
                        raise ArchiveError("package archive exceeds the entries limit")
                    archive.seek(variable_size, os.SEEK_CUR)
                    remaining -= record_size
                if actual_entries != entries:
                    raise ArchiveError("package archive central directory entry count is invalid")
                return
        search_end = offset


def _package_root(path: Path) -> Path:
    if path.is_symlink():
        raise ArchiveError(f"package root is a symlink {path}")
    if not path.is_dir():
        raise ArchiveError(f"package root is not a directory {path}")
    return path.resolve()


def _capture_source_root(root: Path) -> _SourceRoot:
    """Open *root* before collection so later rebinding cannot change its identity."""

    source_fd: int | None = None
    try:
        source_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        os.fstat(source_fd)
        return _SourceRoot(source_fd)
    except OSError as exc:
        if source_fd is not None:
            os.close(source_fd)
        raise ArchiveError(f"cannot inspect package root {root}: {exc}") from exc


def _directory_identity(result: os.stat_result) -> _DirectoryIdentity:
    """Return the identity represented by an already-open directory stat result."""

    return _DirectoryIdentity(result.st_dev, result.st_ino)


def _source_paths(root: Path) -> tuple[Path, ...]:
    """Return every source descendant, refusing directories that cannot be read."""

    paths: list[Path] = []

    def visit(directory: Path) -> None:
        children: list[Path] = []
        try:
            for child in directory.iterdir():
                children.append(child)
        except OSError as exc:
            raise ArchiveError(f"cannot read package directory {directory}: {exc}") from exc
        for child in children:
            paths.append(child)
            try:
                child_mode = child.lstat().st_mode
            except OSError as exc:
                raise ArchiveError(f"cannot inspect package source {child}: {exc}") from exc
            if (
                not stat.S_ISREG(child_mode)
                and not stat.S_ISDIR(child_mode)
                and not stat.S_ISLNK(child_mode)
            ):
                raise ArchiveError(f"package contains unsupported filesystem node {child}")
            if stat.S_ISDIR(child_mode):
                visit(child)

    visit(root)
    return tuple(sorted(paths, key=_relative_path_key(root)))


def _reject_source_links(root: Path) -> None:
    for descendant in _source_paths(root):
        if descendant.is_symlink():
            raise ArchiveError(f"package contains symlink {descendant}")


def _validate_destination(root: Path, destination: Path) -> None:
    """Reject destinations that could overwrite a source-tree file."""

    if destination.resolve().is_relative_to(root):
        raise ArchiveError(f"package archive destination is inside package root {destination}")
    try:
        destination_stat = destination.stat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ArchiveError(
            f"cannot inspect package archive destination {destination}: {exc}"
        ) from exc
    destination_identity = (destination_stat.st_dev, destination_stat.st_ino)
    for source in _source_paths(root):
        try:
            source_stat = source.stat()
        except OSError as exc:
            raise ArchiveError(f"cannot inspect package source {source}: {exc}") from exc
        if (
            stat.S_ISREG(source_stat.st_mode)
            and (
                source_stat.st_dev,
                source_stat.st_ino,
            )
            == destination_identity
        ):
            raise ArchiveError(f"package archive destination aliases package source {source}")


def _archive_distribution(root: Path) -> tuple[PackageManifest, str, dict[str, bytes]]:
    """Build the manifest and selected content set for a portable archive."""
    try:
        source_manifest = load_manifest(root / _MANIFEST_NAME)
    except ManifestError as exc:
        raise ArchiveError(f"cannot load package manifest from {root}: {exc}") from exc
    manifest = distribution_manifest(source_manifest)
    prefix = _entry_prefix(manifest)
    _archive_path(prefix + _MANIFEST_NAME)
    contents = _archive_contents(root, manifest)
    _validate_content_paths(prefix, contents)
    return manifest, prefix, contents


def _archive_contents(root: Path, manifest: PackageManifest) -> dict[str, bytes]:
    paths = _source_paths(root)
    ignored = _gitignore_spec(root, paths)
    selected: list[tuple[str, Path, int]] = []
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {_MANIFEST_NAME, _RECORD_NAME} or _excluded(relative, ignored):
            continue
        _relative_archive_path(relative)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ArchiveError(f"cannot inspect package file {path}: {exc}") from exc
        selected.append((relative, path, size))

    manifest_content = _normalized_manifest(manifest)
    record_paths = sorted([_MANIFEST_NAME, *(relative for relative, _, _ in selected)])
    record_size = len(
        serialize_record(tuple(RecordEntry(path, "0" * 64) for path in record_paths)).encode()
    )
    _validate_size_limits(
        len(selected) + 2,
        [*(size for _, _, size in selected), len(manifest_content), record_size],
    )

    contents = {_MANIFEST_NAME: manifest_content}
    remaining = MAX_ARCHIVE_TOTAL_SIZE - len(manifest_content) - record_size
    for relative, path, _ in selected:
        limit = min(MAX_ARCHIVE_ENTRY_SIZE, remaining)
        contents[relative] = _read_source_file(
            path, limit, total_limited=limit < MAX_ARCHIVE_ENTRY_SIZE
        )
        remaining -= len(contents[relative])
    entries = _record_entries(contents)
    contents[_RECORD_NAME] = serialize_record(entries).encode()
    _reject_casefolding_collisions(contents)
    return contents


def _read_source_file(path: Path, limit: int, *, total_limited: bool) -> bytes:
    """Read one source payload without exceeding archive size limits."""
    content = bytearray()
    try:
        with path.open("rb") as source:
            while chunk := source.read(min(_HASH_CHUNK_SIZE, limit - len(content) + 1)):
                content.extend(chunk)
                if len(content) > limit:
                    if total_limited:
                        raise ArchiveError("package archive exceeds the total size limit")
                    raise ArchiveError("package archive entry exceeds the size limit")
    except OSError as exc:
        raise ArchiveError(f"cannot read package file {path}: {exc}") from exc
    return bytes(content)


def _gitignore_spec(root: Path, paths: tuple[Path, ...]) -> PathSpec:
    patterns: list[str] = []
    for path in paths:
        if not path.is_file() or path.name != ".gitignore":
            continue
        relative_parent = path.relative_to(root).parent.as_posix()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise ArchiveError(f"cannot read gitignore file {path}: {exc}") from exc
        patterns.extend(_prefixed_pattern(line, relative_parent) for line in lines)
    return PathSpec.from_lines("gitignore", patterns)


def _prefixed_pattern(pattern: str, parent: str) -> str:
    """Translate a nested gitignore rule to the package-root pattern space."""

    if parent == "." or not pattern or pattern.startswith("#"):
        return pattern
    negated = pattern.startswith("!")
    body = pattern[1:] if negated else pattern
    anchored = body.startswith("/")
    body = body.removeprefix("/")
    # A slash-free rule is a basename rule and applies at every descendant,
    # unlike a rule containing a slash, which remains relative to this file.
    relative_pattern = body if anchored or "/" in body else "**/" + body
    return ("!" if negated else "") + parent + "/" + relative_pattern


def _excluded(path: str, ignored: PathSpec) -> bool:
    parts = PurePosixPath(path).parts
    if (
        any(part.startswith(".") for part in parts)
        or any(part in _VCS_DIRECTORIES or part in _CACHE_DIRECTORIES for part in parts)
        or path.casefold().endswith(_ARCHIVE_SUFFIX)
    ):
        return True
    # Git cannot re-include a child once an ancestor directory is excluded.
    # PathSpec evaluates patterns for a single path, so retain that traversal
    # rule when applying the combined nested-ignore specification.
    ancestors = PurePosixPath(path).parents
    return ignored.match_file(path) or any(
        ignored.match_file(ancestor.as_posix() + "/")
        for ancestor in ancestors
        if ancestor != PurePosixPath(".")
    )


def _normalized_manifest(manifest: PackageManifest) -> bytes:
    """Render manifest data in one stable TOML representation."""

    lines = [
        "[package]",
        f"name = {_toml_string(manifest.name)}",
        f"version = {_toml_string(str(manifest.version))}",
    ]
    for key, value in (
        ("description", manifest.description),
        ("license", manifest.license),
        ("repository", manifest.repository),
    ):
        if value is not None:
            lines.append(f"{key} = {_toml_string(value)}")
    if manifest.authors:
        lines.append("authors = " + _toml_array(manifest.authors))
    if manifest.keywords:
        lines.append("keywords = " + _toml_array(manifest.keywords))
    if manifest.dependencies:
        lines.extend(("", "[dependencies]"))
        for name in sorted(manifest.dependencies):
            dependency = manifest.dependencies[name]
            fields = [("version", str(dependency.version))]
            fields.extend(
                (key, value)
                for key, value in (("url", dependency.url), ("hash", dependency.hash))
                if value is not None
            )
            if len(fields) == 1:
                lines.append(f"{_toml_key(name)} = {_toml_string(fields[0][1])}")
            else:
                rendered = ", ".join(f"{key} = {_toml_string(value)}" for key, value in fields)
                lines.append(f"{_toml_key(name)} = {{ {rendered} }}")
    if manifest.commands:
        lines.extend(("", "[commands]"))
        for path in sorted(manifest.commands):
            command = manifest.commands[path]
            fields = [("program", command.program)]
            if command.description is not None:
                fields.append(("description", command.description))
            rendered = ", ".join(f"{key} = {_toml_string(value)}" for key, value in fields)
            lines.append(f"{_toml_key(path)} = {{ {rendered} }}")
    return ("\n".join(lines) + "\n").encode()


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_key(value: str) -> str:
    return (
        value
        if value
        and all(
            character.isascii() and (character.isalnum() or character in "_-")
            for character in value
        )
        else _toml_string(value)
    )


def _toml_array(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def _record_entries(contents: dict[str, bytes]) -> tuple[RecordEntry, ...]:
    return tuple(
        RecordEntry(path, hashlib.sha256(contents[path]).hexdigest()) for path in sorted(contents)
    )


def _metadata(manifest: PackageManifest, contents: dict[str, bytes]) -> ArchiveMetadata:
    entries = _record_from_bytes(contents[_RECORD_NAME])
    return ArchiveMetadata(manifest, _package_hash(entries))


def _package_hash(entries: tuple[RecordEntry, ...]) -> str:
    return content_hash(entries)


def _entry_prefix(manifest: PackageManifest) -> str:
    return f"{manifest.name}-{manifest.version}/"


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.create_system = 3
    info.external_attr = _FILE_MODE << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _archive_layout(
    infos: list[zipfile.ZipInfo],
) -> tuple[str, str, str, dict[str, zipfile.ZipInfo]]:
    if not infos:
        raise ArchiveError("package archive is empty")
    _validate_archive_limits(infos)
    names = [info.filename for info in infos]
    if any(stat.S_ISLNK(info.external_attr >> 16) for info in infos):
        raise ArchiveError("package archive contains a symlink")
    if any(info.is_dir() for info in infos):
        raise ArchiveError("package archive contains a directory entry")
    for info in infos:
        _archive_path(info.filename)
        _validate_zip_info(info)
    if names != sorted(names):
        raise ArchiveError("package archive entries are not sorted")
    if len(names) != len(set(names)):
        raise ArchiveError("package archive repeats an entry")
    _reject_casefolding_collisions(names)
    _reject_file_descendant_conflicts(names)
    paths = tuple(PurePosixPath(name) for name in names)
    prefix_part = paths[0].parts[0]
    if any(path.parts[0] != prefix_part or len(path.parts) < 2 for path in paths):
        raise ArchiveError("package archive entries must share one directory prefix")
    prefix = prefix_part + "/"
    manifest_name = prefix + _MANIFEST_NAME
    record_name = prefix + _RECORD_NAME
    if manifest_name not in names or record_name not in names:
        raise ArchiveError("package archive requires package.toml and RECORD")
    return prefix, manifest_name, record_name, dict(zip(names, infos, strict=True))


def _validate_archive_limits(infos: list[zipfile.ZipInfo]) -> None:
    _validate_size_limits(len(infos), (info.file_size for info in infos))


def _validate_size_limits(entry_count: int, sizes: Iterable[int]) -> None:
    """Apply the archive reader's entry and expanded-size limits."""

    if entry_count > MAX_ARCHIVE_ENTRIES:
        raise ArchiveError("package archive has too many entries")
    total_size = 0
    for size in sizes:
        if size > MAX_ARCHIVE_ENTRY_SIZE:
            raise ArchiveError("package archive entry exceeds the size limit")
        total_size += size
        if total_size > MAX_ARCHIVE_TOTAL_SIZE:
            raise ArchiveError("package archive exceeds the total size limit")


def _validate_zip_info(info: zipfile.ZipInfo) -> None:
    expected_flags = 0 if info.filename.isascii() else _UTF8_FLAG
    if (
        info.date_time != _ZIP_TIMESTAMP
        or info.create_system != 3
        or info.create_version != 20
        or info.extract_version != 20
        or info.external_attr != _FILE_MODE << 16
        or info.internal_attr != 0
        or info.flag_bits != expected_flags
        or info.compress_type != zipfile.ZIP_DEFLATED
        or info.extra
        or info.comment
        or info.volume != 0
        or info.reserved != 0
    ):
        raise ArchiveError("package archive entry has noncanonical ZIP metadata")


def _archive_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or value.endswith("/")
        or path.is_absolute()
        or windows_path.drive
        or windows_path.root
        or any(part in {".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ArchiveError(f"invalid package archive entry {value!r}")
    if len(path.parts) > MAX_ARCHIVE_PATH_COMPONENTS:
        raise ArchiveError(f"package archive entry exceeds path depth limit: {value!r}")
    _validate_portable_components(path.parts, value)
    return path


def _relative_archive_path(value: str) -> PurePosixPath:
    return _archive_path(value)


def _validate_portable_components(parts: tuple[str, ...], value: str) -> None:
    for component in parts:
        stem = component.split(".", maxsplit=1)[0].casefold()
        if (
            len(component) > 255
            or component.endswith((".", " "))
            or any(ord(character) < 32 or character in '<>:"|?*' for character in component)
            or stem in {"con", "prn", "aux", "nul"}
            or (len(stem) == 4 and stem[:3] in {"com", "lpt"} and stem[3] in "123456789¹²³")
        ):
            raise ArchiveError(f"invalid Windows-portable archive component in {value!r}")


def _validate_content_paths(prefix: str, contents: dict[str, bytes]) -> None:
    _validate_size_limits(len(contents), (len(content) for content in contents.values()))
    for path in contents:
        _archive_path(prefix + path)
    _reject_casefolding_collisions(contents)


def _validate_prefix(prefix: str, manifest: PackageManifest) -> None:
    _archive_path(prefix + _MANIFEST_NAME)
    if prefix != _entry_prefix(manifest):
        raise ArchiveError("package archive prefix does not match its manifest")


def _manifest_from_bytes(content: bytes) -> PackageManifest:
    try:
        return load_manifest_text(content.decode())
    except (ManifestError, UnicodeDecodeError) as exc:
        raise ArchiveError(f"cannot parse archive package manifest: {exc}") from exc


def _record_from_bytes(content: bytes) -> tuple[RecordEntry, ...]:
    try:
        entries = parse_record(content.decode())
    except (RecordError, UnicodeDecodeError) as exc:
        raise ArchiveError(f"cannot parse archive RECORD: {exc}") from exc
    for entry in entries:
        _relative_archive_path(entry.path)
    _reject_casefolding_collisions(entry.path for entry in entries)
    if entries != tuple(sorted(entries, key=_record_path_key)):
        raise ArchiveError("archive RECORD entries are not sorted")
    if content != serialize_record(entries).encode():
        raise ArchiveError("archive RECORD is not canonical")
    return entries


def _read_entry(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    content = bytearray()
    with archive.open(info) as file:
        while chunk := file.read(_HASH_CHUNK_SIZE):
            content.extend(chunk)
            if len(content) > MAX_ARCHIVE_ENTRY_SIZE:
                raise ArchiveError("package archive entry exceeds the size limit")
    if len(content) != info.file_size:
        raise ArchiveError("package archive entry size does not match its metadata")
    return bytes(content)


def _entry_digest(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    size = 0
    with archive.open(info) as file:
        while chunk := file.read(_HASH_CHUNK_SIZE):
            size += len(chunk)
            if size > MAX_ARCHIVE_ENTRY_SIZE:
                raise ArchiveError("package archive entry exceeds the size limit")
            digest.update(chunk)
    if size != info.file_size:
        raise ArchiveError("package archive entry size does not match its metadata")
    return digest.hexdigest()


def _relative_path_key(root: Path) -> Callable[[Path], str]:
    """Return a stable archive-sort key for paths below *root*."""

    def key(path: Path) -> str:
        return path.relative_to(root).as_posix()

    return key


def _record_path_key(entry: RecordEntry) -> str:
    """Return the canonical ordering key for one record entry."""

    return entry.path


def _reject_file_descendant_conflicts(paths: Iterable[str]) -> None:
    """Reject ZIP entries that require one path to be both file and directory."""

    names = set(paths)
    for name in names:
        path = PurePosixPath(name)
        if any(parent.as_posix() in names for parent in path.parents):
            raise ArchiveError(f"package archive file conflicts with descendant entry {name!r}")


def _reject_casefolding_collisions(paths: Iterable[str]) -> None:
    paths_by_folded_name: dict[str, str] = {}
    for path in paths:
        folded = path.casefold()
        existing = paths_by_folded_name.get(folded)
        if existing is not None and existing != path:
            raise ArchiveError(f"package paths collide by case: {existing!r} and {path!r}")
        paths_by_folded_name[folded] = path
