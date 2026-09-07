"""Package manifest schema and TOML loading."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from io import StringIO
from pathlib import Path, PurePosixPath, PureWindowsPath

import semver
import tomlkit
from tomlkit.exceptions import TOMLKitError

from agm.agl.keywords import KEYWORDS
from agm.agl.modules.ids import ModuleId
from agm.command_catalog import invalid_command_path
from agm.core.toml import TomlDict, load_toml_file, toml_dict

_SHA256_PREFIXES = ("sha256=", "sha256:", "sha256-")


class ManifestError(ValueError):
    """Raised when a package manifest does not satisfy the package schema."""


@dataclass(frozen=True, slots=True)
class DependencySpec:
    """A dependency version floor and its optional development source."""

    version: semver.Version
    path: str | None = None
    url: str | None = None
    hash: str | None = None


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """One manifest command registration."""

    program: str | None = None
    description: str | None = None
    help: str | None = None


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
    aliases: dict[str, str] = field(default_factory=dict)


def command_paths_for_program(
    manifest: PackageManifest, reference: str
) -> tuple[tuple[str, ...], ...]:
    """Return the command paths *manifest* registers for a ``MODULE::PROGRAM`` *reference*.

    Each path is returned as its own words, sorted, so a program registered
    under more than one command path yields every one of them.
    """

    return tuple(
        tuple(path.split())
        for path, command in sorted(expanded_commands(manifest).items())
        if command.program == reference
    )


def expanded_commands(manifest: PackageManifest) -> dict[str, CommandSpec]:
    """Expand aliases against canonical paths, including a group's descendants.

    Alias targets always name the package's own canonical tree. This keeps
    expansion finite and independent of declaration or activation order.
    """
    commands = dict(manifest.commands)
    if not manifest.aliases:
        return commands
    canonical_paths = set(commands)
    for path in commands:
        words = path.split()
        canonical_paths.update(" ".join(words[:length]) for length in range(1, len(words)))
    for alias, target in manifest.aliases.items():
        invalid = invalid_command_path(alias)
        if invalid is not None:
            raise ManifestError(f"alias path {alias!r} {invalid}")
        if target not in canonical_paths:
            raise ManifestError(f"alias {alias!r} names unknown canonical command {target!r}")
        additions = {alias: manifest.commands.get(target, CommandSpec())}
        additions.update(
            (alias + path[len(target) :], spec)
            for path, spec in manifest.commands.items()
            if path.startswith(target + " ")
        )
        for path, spec in additions.items():
            if path in canonical_paths or path in commands:
                raise ManifestError(f"alias {alias!r} conflicts with command {path!r}")
            commands[path] = spec
    return commands


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
    _only_keys(raw, {"package", "dependencies", "commands", "aliases"}, "manifest")
    package = _required_table(raw, "package")
    _only_keys(
        package,
        {"name", "version", "description", "license", "authors", "repository", "keywords"},
        "package",
    )
    name = validate_package_name(_required_str(package, "name", "package"))
    aliases = _optional_table(raw, "aliases")
    manifest = PackageManifest(
        name=name,
        version=_version(_required_str(package, "version", "package"), "package version"),
        description=_optional_str(package, "description", "package"),
        license=_optional_str(package, "license", "package"),
        authors=_optional_str_list(package, "authors", "package"),
        repository=_optional_str(package, "repository", "package"),
        keywords=_optional_str_list(package, "keywords", "package"),
        dependencies=_dependencies(_optional_table(raw, "dependencies")),
        commands=_commands(_optional_table(raw, "commands")),
        aliases={path: _required_str(aliases, path, "aliases") for path in aliases},
    )
    expanded_commands(manifest)
    return manifest


def validate_package_name(value: str) -> str:
    """Validate a package or dependency name and return it unchanged.

    A name must be a single AgL module-path segment and not a reserved AgL
    keyword. This is the single package-name validation rule; other package
    layers translate ``ManifestError`` into their own error type.
    """
    try:
        module_id = ModuleId.from_path(value)
    except ValueError as exc:
        raise ManifestError(f"package name {value!r} is not a valid module segment") from exc
    if len(module_id.segments) != 1:
        raise ManifestError(f"package name {value!r} is not a valid module segment")
    if value in KEYWORDS:
        raise ManifestError(f"package name {value!r} is a reserved AgL keyword")
    return value


def _dependencies(raw: TomlDict) -> dict[str, DependencySpec]:
    dependencies: dict[str, DependencySpec] = {}
    for name, value in raw.items():
        validate_package_name(name)
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
    if path is not None:
        windows_path = PureWindowsPath(path)
        if PurePosixPath(path).root or windows_path.drive or windows_path.root:
            raise ManifestError(f"dependency {name!r} path must be relative")
    if path is not None and url is not None:
        raise ManifestError(f"dependency {name!r} cannot specify both path and url")
    if url is not None and content_hash is None:
        raise ManifestError(f"URL dependency {name!r} requires a hash")
    if content_hash is not None and url is None:
        raise ManifestError(f"dependency {name!r} hash requires a URL source")
    if content_hash is not None and parse_sha256(content_hash) is None:
        raise ManifestError(f"URL dependency {name!r} hash must be a SHA-256 digest")
    return DependencySpec(version, path=path, url=url, hash=content_hash)


def parse_sha256(value: str) -> str | None:
    """Return the lowercase digest of a SHA-256 declaration, or ``None``.

    All three conventional separators are accepted so a digest copied from a
    checksum file, a URL fragment, or a subresource-integrity string is
    usable unchanged.
    """

    prefix = next(
        (candidate for candidate in _SHA256_PREFIXES if value.startswith(candidate)), None
    )
    if prefix is None:
        return None
    digest = value[len(prefix) :]
    if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
        return None
    return digest.lower()


def _commands(raw: TomlDict) -> dict[str, CommandSpec]:
    commands: dict[str, CommandSpec] = {}
    for path, value in raw.items():
        if not isinstance(value, dict):
            raise ManifestError(f"command {path!r} must be a table")
        command = toml_dict(value)
        _only_keys(command, {"program", "description", "help"}, f"command {path!r}")
        commands[path] = CommandSpec(
            program=_optional_str(command, "program", f"command {path!r}"),
            description=_optional_str(command, "description", f"command {path!r}"),
            help=_optional_str(command, "help", f"command {path!r}"),
        )
    for path, spec in commands.items():
        if spec.program is None and not any(child.startswith(path + " ") for child in commands):
            raise ManifestError(f"command group {path!r} requires subcommands")
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
