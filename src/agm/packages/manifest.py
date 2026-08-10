"""Package manifest schema and TOML loading."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from io import StringIO
from pathlib import Path

import semver
import tomlkit
from tomlkit.exceptions import TOMLKitError

from agm.agl.modules.ids import ModuleId
from agm.core.toml import TomlDict, load_toml_file, toml_dict


class ManifestError(ValueError):
    """Raised when a package manifest does not satisfy the package schema."""


@dataclass(frozen=True, slots=True)
class DependencySpec:
    """A minimum version requirement and its optional development source."""

    version: semver.Version
    path: str | None = None
    url: str | None = None
    hash: str | None = None


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """One manifest command registration."""

    program: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class PackageManifest:
    """Validated contents of a package's ``package.toml`` file."""

    name: str
    version: semver.Version
    description: str | None = None
    license: str | None = None
    authors: tuple[str, ...] = ()
    repository: str | None = None
    keywords: tuple[str, ...] = ()
    dependencies: dict[str, DependencySpec] = field(default_factory=dict)
    commands: dict[str, CommandSpec] = field(default_factory=dict)


def distribution_manifest(manifest: PackageManifest) -> PackageManifest:
    """Return a publication-safe manifest without local dependency sources.

    The source manifest remains untouched, so callers can render this view into
    a distribution without changing a development package tree.
    """

    dependencies = {
        name: replace(dependency, path=None) for name, dependency in manifest.dependencies.items()
    }
    return replace(manifest, dependencies=dependencies)


def load_manifest(path: Path) -> PackageManifest:
    """Load and validate the ``package.toml`` at *path*."""

    try:
        raw = load_toml_file(path)
    except (OSError, TOMLKitError, UnicodeDecodeError) as exc:
        raise ManifestError(f"cannot load package manifest {path}: {exc}") from exc
    return _parse_manifest(raw)


def load_manifest_text(content: str) -> PackageManifest:
    """Parse and validate package-manifest text without reading a filesystem path."""

    try:
        raw = toml_dict(tomlkit.load(StringIO(content)).unwrap())
    except TOMLKitError as exc:
        raise ManifestError(f"cannot parse package manifest: {exc}") from exc
    return _parse_manifest(raw)


def _parse_manifest(raw: TomlDict) -> PackageManifest:
    _only_keys(raw, {"package", "dependencies", "commands"}, "manifest")
    package = _required_table(raw, "package")
    _only_keys(
        package,
        {"name", "version", "description", "license", "authors", "repository", "keywords"},
        "package",
    )
    name = _package_name(_required_str(package, "name", "package"))
    return PackageManifest(
        name=name,
        version=_version(_required_str(package, "version", "package"), "package version"),
        description=_optional_str(package, "description", "package"),
        license=_optional_str(package, "license", "package"),
        authors=_optional_str_list(package, "authors", "package"),
        repository=_optional_str(package, "repository", "package"),
        keywords=_optional_str_list(package, "keywords", "package"),
        dependencies=_dependencies(_optional_table(raw, "dependencies")),
        commands=_commands(_optional_table(raw, "commands")),
    )


def _package_name(value: str) -> str:
    try:
        module_id = ModuleId.from_path(value)
    except ValueError as exc:
        raise ManifestError(f"package name {value!r} is not a valid module segment") from exc
    if len(module_id.segments) != 1:
        raise ManifestError(f"package name {value!r} is not a valid module segment")
    return value


def _dependencies(raw: TomlDict) -> dict[str, DependencySpec]:
    dependencies: dict[str, DependencySpec] = {}
    for name, value in raw.items():
        _package_name(name)
        if isinstance(value, str):
            dependencies[name] = DependencySpec(
                _minimum_version(value, f"dependency {name!r} version")
            )
        elif isinstance(value, dict):
            dependencies[name] = _dependency_table(name, toml_dict(value))
        else:
            raise ManifestError(f"dependency {name!r} must be a version string or table")
    return dependencies


def _dependency_table(name: str, raw: TomlDict) -> DependencySpec:
    _only_keys(raw, {"version", "path", "url", "hash"}, f"dependency {name!r}")
    context = f"dependency {name!r}"
    version = _minimum_version(_required_str(raw, "version", context), f"{context} version")
    path = _optional_str(raw, "path", context)
    url = _optional_str(raw, "url", context)
    content_hash = _optional_str(raw, "hash", context)
    if path is not None and url is not None:
        raise ManifestError(f"dependency {name!r} cannot specify both path and url")
    if url is not None and content_hash is None:
        raise ManifestError(f"URL dependency {name!r} requires a hash")
    if content_hash is not None and url is None:
        raise ManifestError(f"dependency {name!r} hash requires a URL source")
    if content_hash is not None and not _is_sha256_hash(content_hash):
        raise ManifestError(f"URL dependency {name!r} hash must be a SHA-256 digest")
    return DependencySpec(version, path=path, url=url, hash=content_hash)


def _is_sha256_hash(value: str) -> bool:
    prefixes = ("sha256=", "sha256:", "sha256-")
    prefix = next((candidate for candidate in prefixes if value.startswith(candidate)), None)
    if prefix is None:
        return False
    digest = value[len(prefix) :]
    return len(digest) == 64 and all(character in "0123456789abcdefABCDEF" for character in digest)


def _commands(raw: TomlDict) -> dict[str, CommandSpec]:
    commands: dict[str, CommandSpec] = {}
    for path, value in raw.items():
        if not isinstance(value, dict):
            raise ManifestError(f"command {path!r} must be a table")
        command = toml_dict(value)
        _only_keys(command, {"program", "description"}, f"command {path!r}")
        commands[path] = CommandSpec(
            program=_required_str(command, "program", f"command {path!r}"),
            description=_optional_str(command, "description", f"command {path!r}"),
        )
    return commands


def _version(value: str, label: str) -> semver.Version:
    try:
        return semver.Version.parse(value)
    except ValueError as exc:
        raise ManifestError(f"{label} must be semantic versioning syntax") from exc


def _minimum_version(value: str, label: str) -> semver.Version:
    try:
        return semver.Version.parse(value, optional_minor_and_patch=True)
    except ValueError as exc:
        raise ManifestError(f"{label} must be minimum-version syntax") from exc


def _required_table(raw: TomlDict, key: str) -> TomlDict:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ManifestError(f"manifest requires a [{key}] table")
    return toml_dict(value)


def _optional_table(raw: TomlDict, key: str) -> TomlDict:
    if key not in raw:
        return {}
    value = raw[key]
    if not isinstance(value, dict):
        raise ManifestError(f"manifest {key!r} must be a table")
    return toml_dict(value)


def _required_str(raw: TomlDict, key: str, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{context} requires a non-empty string {key!r}")
    return value


def _optional_str(raw: TomlDict, key: str, context: str) -> str | None:
    if key not in raw:
        return None
    return _required_str(raw, key, context)


def _optional_str_list(raw: TomlDict, key: str, context: str) -> tuple[str, ...]:
    if key not in raw:
        return ()
    value = raw[key]
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ManifestError(f"{context} {key!r} must be a list of non-empty strings")
    return tuple(value)


def _only_keys(raw: TomlDict, allowed: set[str], context: str) -> None:
    unexpected = set(raw).difference(allowed)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ManifestError(f"{context} has unsupported fields: {names}")
