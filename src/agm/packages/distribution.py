"""The distribution view of a package source directory.

A package is stored and shipped as a *distribution*: its normalized manifest
plus the source files a portable archive carries, leaving behind dotfiles,
ignored files, tool caches, VCS metadata, and archives. Archive creation,
immutable directory staging, and store cleanup all read that one selection from
here, so a source directory yields the same stored tree and the same content
hash whichever route installs it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
from pathlib import Path, PurePosixPath

from pathspec import PathSpec

from agm.packages.manifest import PackageManifest
from agm.packages.record import (
    RECORD_NAME,
    RecordEntry,
    file_digest,
    record_path_key,
    walk_package_tree,
)

MANIFEST_NAME = "package.toml"

_ARCHIVE_SUFFIX = ".agmpkg"
_VCS_DIRECTORIES = frozenset({".git", ".hg", ".svn", ".bzr", "CVS"})
_CACHE_DIRECTORIES = frozenset(
    {"__pycache__", ".cache", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)


class DistributionError(ValueError):
    """Raised when a package source tree cannot be read as a distribution."""


def is_cache_or_vcs_path(path: str) -> bool:
    """Whether a relative path is tool-cache or VCS content no package distributes.

    Package trees never record such content, so the store may still acquire it
    after publication -- tools run in the store tree may write ``__pycache__``
    -- and removal has to recognize it as residue rather than package content.
    """

    return any(
        part in _VCS_DIRECTORIES or part in _CACHE_DIRECTORIES for part in PurePosixPath(path).parts
    )


def source_paths(root: Path) -> tuple[Path, ...]:
    """Return every source descendant, refusing directories that cannot be read.

    Symlinks are collected as files but are never traversed as directories.
    """

    def accept(child: Path, mode: int) -> None:
        if not stat.S_ISREG(mode) and not stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
            raise DistributionError(f"package contains unsupported filesystem node {child}")

    def directory_error(directory: Path, exc: OSError) -> DistributionError:
        return DistributionError(f"cannot read package directory {directory}: {exc}")

    def entry_error(child: Path, exc: OSError) -> DistributionError:
        return DistributionError(f"cannot inspect package source {child}: {exc}")

    return walk_package_tree(
        root, accept=accept, on_directory_error=directory_error, on_entry_error=entry_error
    )


def distribution_files(root: Path) -> tuple[tuple[str, Path], ...]:
    """Return the distributed source files as sorted ``(relative path, path)`` pairs.

    The manifest and ``RECORD`` are excluded: both are rendered rather than
    copied, so a distribution carries a normalized manifest and a record of the
    selection made here. Symlinked regular files are included and dereferenced
    by distribution writers, so stored trees and archives remain link-free.
    """

    paths = source_paths(root)
    ignored = _gitignore_spec(root, paths)
    return tuple(
        (relative, path)
        for path in paths
        if path.is_file()
        for relative in (path.relative_to(root).as_posix(),)
        if relative not in {MANIFEST_NAME, RECORD_NAME} and not _excluded(relative, ignored)
    )


def distribution_entries(root: Path, manifest: PackageManifest) -> tuple[RecordEntry, ...]:
    """Return the canonical record entries of *root*'s distribution, without writing it."""

    entries = [
        RecordEntry(MANIFEST_NAME, hashlib.sha256(normalized_manifest(manifest)).hexdigest()),
        *(RecordEntry(relative, file_digest(path)) for relative, path in distribution_files(root)),
    ]
    return tuple(sorted(entries, key=record_path_key))


def materialize_distribution(root: Path, manifest: PackageManifest, destination: Path) -> None:
    """Write *root*'s distribution into the empty directory *destination*.

    File contents cross unchanged while modes do not, matching archive
    extraction, so the same distribution installs identically from a directory
    and from an archive.
    """

    files = distribution_files(root)
    (destination / MANIFEST_NAME).write_bytes(normalized_manifest(manifest))
    for relative, path in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def normalized_manifest(manifest: PackageManifest) -> bytes:
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
    if manifest.python_dependencies:
        dependencies = _toml_array(manifest.python_dependencies)
        lines.extend(("", "[python]", f"dependencies = {dependencies}"))
    if manifest.commands:
        lines.extend(("", "[commands]"))
        for path in sorted(manifest.commands):
            command = manifest.commands[path]
            rendered = ", ".join(
                f"{key} = {_toml_string(value)}"
                for key, value in (("program", command.program), ("doc", command.doc))
                if value is not None
            )
            lines.append(f"{_toml_key(path)} = {{ {rendered} }}")
    if manifest.aliases:
        lines.extend(("", "[aliases]"))
        lines.extend(
            f"{_toml_key(alias)} = {_toml_string(target)}"
            for alias, target in sorted(manifest.aliases.items())
        )
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


def _gitignore_spec(root: Path, paths: tuple[Path, ...]) -> PathSpec:
    patterns: list[str] = []
    for path in paths:
        if not path.is_file() or path.name != ".gitignore":
            continue
        relative_parent = path.relative_to(root).parent.as_posix()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise DistributionError(f"cannot read gitignore file {path}: {exc}") from exc
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
    # A rule is anchored to this file's directory only by a separator at its
    # beginning or middle; a trailing separator merely restricts the rule to
    # directories, leaving it a name rule that applies at every descendant.
    relative_pattern = body if anchored or "/" in body.rstrip("/") else "**/" + body
    return ("!" if negated else "") + parent + "/" + relative_pattern


def _excluded(path: str, ignored: PathSpec) -> bool:
    parts = PurePosixPath(path).parts
    if (
        any(part.startswith(".") for part in parts)
        or is_cache_or_vcs_path(path)
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
