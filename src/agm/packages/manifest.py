"""Package manifest schema and TOML loading."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath, PureWindowsPath

import semver
from tomlkit.exceptions import TOMLKitError

from agm.agl.keywords import is_plain_name
from agm.agl.modules.ids import ModuleId
from agm.command_catalog import invalid_command_path
from agm.core.toml import TomlDict, load_toml_file, parse_toml_doc, toml_dict
from agm.packages.record import is_sha256_hex

_SHA256_PREFIXES = ("sha256=", "sha256:", "sha256-")
_COMMAND_FIELDS = frozenset({"program", "doc"})


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
    """One manifest command registration.

    ``doc`` is the command's prose. A command naming a program takes it from
    that program's ``@doc`` every time the package's source is read, and an
    installed manifest carries the result so a host lists the command without
    compiling anything (see
    :func:`agm.packages.source_commands.package_with_source_commands`). A
    command group, which names no program, carries only what the manifest
    states.
    """

    program: str | None = None
    doc: str | None = None


@dataclass(frozen=True, slots=True)
class UnknownField:
    """One manifest key the schema does not define, and where it was found.

    Loading records these rather than refusing the file: a manifest AGM baked
    into its own store may have been written by a build with fields this one
    does not know, and a package that cannot be read at all takes the whole CLI
    down with it. Authored manifests are held to the schema by
    :func:`agm.packages.discipline.validate_package`, at the boundary where the
    author can act on the error.
    """

    context: str
    name: str


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
    unknown_fields: tuple[UnknownField, ...] = ()


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
    Assumes *manifest* already satisfies :func:`validate_command_set` — every
    complete manifest does by the time it is loaded or merged, so this only
    builds the expansion rather than re-checking it.
    """
    commands = dict(manifest.commands)
    for alias, target in manifest.aliases.items():
        commands.update(_alias_additions(manifest.commands, alias, target))
    return commands


def _alias_additions(
    commands: dict[str, CommandSpec], alias: str, target: str
) -> dict[str, CommandSpec]:
    """Return the paths *alias* contributes for canonical *target* and its descendants.

    The single statement of what an alias expands to: :func:`expanded_commands`
    applies it and :func:`validate_command_set` checks the result of applying it.
    """

    additions = {alias: commands.get(target, CommandSpec())}
    additions.update(
        (alias + path[len(target) :], spec)
        for path, spec in commands.items()
        if path.startswith(target + " ")
    )
    return additions


def validate_command_set(manifest: PackageManifest) -> None:
    """Validate the cross-entry invariants of a finished manifest's command set.

    A command group requiring subcommands, and an alias naming a known,
    non-colliding canonical command, can only be checked once every
    ``@command`` program a package's own source declares has merged into the
    manifest's command table — a manifest's ``[commands]``/``[aliases]``
    tables are otherwise validated per-entry at load time, before those
    registrations are known. A complete manifest (a store tree's, an
    archive's) validates this as part of :func:`load_manifest`; a source
    tree loaded with ``commands_complete=False`` must call this once its
    programs' commands are merged in — see
    :func:`agm.packages.source_commands.package_with_source_commands`.
    """
    commands = manifest.commands
    for path, spec in commands.items():
        if spec.program is None and not any(child.startswith(path + " ") for child in commands):
            raise ManifestError(f"command group {path!r} requires subcommands")
    if not manifest.aliases:
        return
    canonical_paths = _canonical_paths(commands)
    added: set[str] = set()
    for alias, target in manifest.aliases.items():
        if target not in canonical_paths:
            raise ManifestError(f"alias {alias!r} names unknown canonical command {target!r}")
        for path in _alias_additions(commands, alias, target):
            if path in canonical_paths or path in added:
                raise ManifestError(f"alias {alias!r} conflicts with command {path!r}")
            added.add(path)


def _canonical_paths(commands: dict[str, CommandSpec]) -> set[str]:
    """Return every command path plus each of its ancestor group prefixes."""
    canonical_paths = set(commands)
    for path in commands:
        words = path.split()
        canonical_paths.update(" ".join(words[:length]) for length in range(1, len(words)))
    return canonical_paths


def _validate_alias_syntax(aliases: dict[str, str]) -> None:
    """Validate each alias's own path syntax, independent of the command table."""
    for alias in aliases:
        invalid = invalid_command_path(alias)
        if invalid is not None:
            raise ManifestError(f"alias path {alias!r} {invalid}")


def distribution_manifest(manifest: PackageManifest) -> PackageManifest:
    """Return a publication-safe manifest without local dependency sources.

    The source manifest remains untouched, so callers can render this view into
    a distribution without changing a development package tree.
    """

    dependencies = {
        name: replace(dependency, path=None) for name, dependency in manifest.dependencies.items()
    }
    return replace(manifest, dependencies=dependencies)


def load_manifest(path: Path, *, commands_complete: bool = True) -> PackageManifest:
    """Load and validate the ``package.toml`` at *path*.

    ``commands_complete=False`` defers :func:`validate_command_set`'s
    cross-entry invariants for a source-tree caller whose package may
    register further commands through ``@command`` programs that have not
    been merged into the manifest yet; the caller must run that validation
    itself once they have (see
    :func:`agm.packages.source_commands.package_with_source_commands`) — as
    must a caller that reads only a live tree's identity and never its
    command table. A manifest that is already complete — a store tree's, an
    archive's — must load with the default so a corrupt one is still caught
    here.
    """

    try:
        raw = load_toml_file(path)
    except (OSError, TOMLKitError, UnicodeDecodeError) as exc:
        raise ManifestError(f"cannot load package manifest {path}: {exc}") from exc
    return _parse_manifest(raw, commands_complete=commands_complete)


def load_manifest_text(content: str, *, commands_complete: bool = True) -> PackageManifest:
    """Parse and validate package-manifest text without reading a filesystem path.

    See :func:`load_manifest` for ``commands_complete``.
    """

    try:
        raw = toml_dict(parse_toml_doc(content).unwrap())
    except TOMLKitError as exc:
        raise ManifestError(f"cannot parse package manifest: {exc}") from exc
    return _parse_manifest(raw, commands_complete=commands_complete)


def _parse_manifest(raw: TomlDict, *, commands_complete: bool = True) -> PackageManifest:
    unknown = _unknown_keys(raw, {"package", "dependencies", "commands", "aliases"}, "manifest")
    package = _required_table(raw, "package")
    unknown += _unknown_keys(
        package,
        {"name", "version", "description", "license", "authors", "repository", "keywords"},
        "package",
    )
    name = validate_package_name(_required_str(package, "name", "package"))
    aliases = _optional_table(raw, "aliases")
    alias_map = {path: _required_str(aliases, path, "aliases") for path in aliases}
    _validate_alias_syntax(alias_map)
    manifest = PackageManifest(
        name=name,
        version=_version(_required_str(package, "version", "package"), "package version"),
        description=_optional_str(package, "description", "package"),
        license=_optional_str(package, "license", "package"),
        authors=_optional_str_list(package, "authors", "package"),
        repository=_optional_str(package, "repository", "package"),
        keywords=_optional_str_list(package, "keywords", "package"),
        dependencies=_dependencies(_optional_table(raw, "dependencies"), unknown),
        commands=_commands(_optional_table(raw, "commands"), unknown),
        aliases=alias_map,
        unknown_fields=tuple(unknown),
    )
    if commands_complete:
        validate_command_set(manifest)
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
    if not is_plain_name(value):
        raise ManifestError(f"package name {value!r} is a reserved AgL keyword")
    return value


def _dependencies(raw: TomlDict, unknown: list[UnknownField]) -> dict[str, DependencySpec]:
    dependencies: dict[str, DependencySpec] = {}
    for name, value in raw.items():
        validate_package_name(name)
        if isinstance(value, str):
            dependencies[name] = DependencySpec(
                _minimum_version(value, f"dependency {name!r} version")
            )
        elif isinstance(value, dict):
            dependencies[name] = _dependency_table(name, toml_dict(value), unknown)
        else:
            raise ManifestError(f"dependency {name!r} must be a version string or table")
    return dependencies


def _dependency_table(name: str, raw: TomlDict, unknown: list[UnknownField]) -> DependencySpec:
    context = f"dependency {name!r}"
    unknown += _unknown_keys(raw, {"version", "path", "url", "hash"}, context)
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
    digest = value[len(prefix) :].lower()
    return digest if is_sha256_hex(digest) else None


def _commands(raw: TomlDict, unknown: list[UnknownField]) -> dict[str, CommandSpec]:
    commands: dict[str, CommandSpec] = {}
    _collect_commands(raw, prefix="", commands=commands, unknown=unknown)
    return commands


def _collect_commands(
    raw: TomlDict, *, prefix: str, commands: dict[str, CommandSpec], unknown: list[UnknownField]
) -> None:
    """Flatten nested TOML command tables into space-separated CLI paths."""

    for name, value in raw.items():
        path = f"{prefix} {name}" if prefix else name
        if not isinstance(value, dict):
            raise ManifestError(f"command {path!r} must be a table")
        command = toml_dict(value)
        metadata = {
            key: field
            for key, field in command.items()
            if key in _COMMAND_FIELDS and not isinstance(field, dict)
        }
        children: TomlDict = {
            key: child for key, child in command.items() if isinstance(child, dict)
        }
        unknown += _unknown_keys(
            {key: field for key, field in command.items() if not isinstance(field, dict)},
            set(_COMMAND_FIELDS),
            f"command {path!r}",
        )
        if not children or metadata:
            if path in commands:
                raise ManifestError(f"command {path!r} is defined more than once")
            commands[path] = CommandSpec(
                program=_optional_str(metadata, "program", f"command {path!r}"),
                doc=_optional_str(metadata, "doc", f"command {path!r}"),
            )
        _collect_commands(children, prefix=path, commands=commands, unknown=unknown)


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


def _unknown_keys(raw: TomlDict, allowed: set[str], context: str) -> list[UnknownField]:
    return [UnknownField(context, name) for name in sorted(set(raw).difference(allowed))]


def describe_unknown_fields(fields: Iterable[UnknownField]) -> str:
    """Render *fields* as one message, a clause per context that carries any."""
    by_context: dict[str, list[str]] = {}
    for unknown in fields:
        by_context.setdefault(unknown.context, []).append(unknown.name)
    return "; ".join(
        f"{context} has unsupported fields: {', '.join(sorted(names))}"
        for context, names in by_context.items()
    )
