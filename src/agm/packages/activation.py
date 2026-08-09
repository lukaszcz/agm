"""Installed-package activation, project pins, and package-root selection."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import semver
from tomlkit.exceptions import TOMLKitError

from agm.agl.modules.ids import ModuleId
from agm.config.general import load_merged_config
from agm.core.fs import mkdir, write_text
from agm.core.toml import TomlDict, load_toml_file, toml_dict
from agm.packages.manifest import ManifestError, PackageManifest, load_manifest
from agm.packages.model import PackageInfo
from agm.packages.store import package_store_path, store_root


class PackageActivationError(ValueError):
    """Raised when activation state, pins, or their requirements are invalid."""


@dataclass(frozen=True, slots=True)
class ActivePackage:
    """One globally active package selection.

    ``editable`` names a live package root.  Otherwise ``version`` identifies
    an immutable tree in the versioned package store.  ``shadow`` is retained
    in activation state for registered-command handling, which is added later.
    """

    version: semver.Version
    editable: Path | None = None
    shadow: bool = False

    def __post_init__(self) -> None:
        if self.editable is not None:
            object.__setattr__(self, "editable", self.editable.resolve())


@dataclass(frozen=True, slots=True)
class ActivationIndex:
    """The rebuildable global package selection stored under AGM home."""

    packages: dict[str, ActivePackage] = field(default_factory=dict)


def activation_index_path(*, home: Path, env: Mapping[str, str] | None = None) -> Path:
    """Return the selected AGM home's package activation-index path."""

    return store_root(home=home, env=env) / "index.toml"


def load_activation_index(*, home: Path, env: Mapping[str, str] | None = None) -> ActivationIndex:
    """Load the activation index, treating an absent index as no selections."""

    path = activation_index_path(home=home, env=env)
    if not path.exists():
        return ActivationIndex()
    try:
        raw = load_toml_file(path)
    except (OSError, TOMLKitError, UnicodeDecodeError) as exc:
        raise PackageActivationError(f"cannot load package activation index {path}: {exc}") from exc
    return _parse_activation_index(raw)


def write_activation_index(
    index: ActivationIndex, *, home: Path, env: Mapping[str, str] | None = None
) -> Path:
    """Write a canonical activation index and return its path."""

    path = activation_index_path(home=home, env=env)
    mkdir(path.parent, parents=True, exist_ok=True)
    lines: list[str] = []
    for name, active in sorted(index.packages.items()):
        _validate_package_name(name)
        lines.extend(
            (
                f"[packages.{name}]",
                f"version = {_toml_string(str(active.version))}",
            )
        )
        if active.editable is not None:
            lines.append(f"editable = {_toml_string(str(active.editable))}")
        if active.shadow:
            lines.append("shadow = true")
        lines.append("")
    write_text(path, "\n".join(lines))
    return path


def rebuild_activation_index(
    *, home: Path, env: Mapping[str, str] | None = None
) -> ActivationIndex:
    """Build the deterministic installed-package selection from store manifests.

    Store versions are side-by-side, so the highest semantic version for each
    package is active after a rebuild.  Editable and command-shadow metadata
    is intentionally not inferable from immutable installed manifests.
    """

    root = store_root(home=home, env=env)
    if not root.is_dir():
        return ActivationIndex()

    active: dict[str, ActivePackage] = {}
    for name_dir in sorted(root.iterdir()):
        if not name_dir.is_dir() or name_dir.is_symlink():
            continue
        _validate_package_name(name_dir.name)
        for version_dir in sorted(name_dir.iterdir()):
            if not version_dir.is_dir() or version_dir.is_symlink():
                continue
            manifest = _load_installed_manifest(version_dir)
            if manifest.name != name_dir.name or str(manifest.version) != version_dir.name:
                raise PackageActivationError(
                    f"installed package at {version_dir} does not match its store identity"
                )
            selected = active.get(manifest.name)
            if selected is None or manifest.version > selected.version:
                active[manifest.name] = ActivePackage(manifest.version)

    index = ActivationIndex(packages=active)
    _validate_requirements(_packages_from_index(index, home=home, env=env))
    return index


def load_package_pins(
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> dict[str, semver.Version]:
    """Read exact project package-version pins from the merged config layers.

    Only the ``[packages]`` section is interpreted here.  Other sections stay
    deliberately lenient during general configuration loading.
    """

    merged = load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd, env=env)
    raw_pins = merged.get("packages")
    if raw_pins is None:
        return {}
    if not isinstance(raw_pins, dict):
        raise PackageActivationError("[packages] must be a table")

    pins: dict[str, semver.Version] = {}
    for name, value in toml_dict(raw_pins).items():
        _validate_package_name(name)
        if not isinstance(value, str):
            raise PackageActivationError(f"package pin {name!r} must be a semantic version string")
        try:
            pins[name] = semver.Version.parse(value)
        except ValueError as exc:
            raise PackageActivationError(
                f"package pin {name!r} must be complete semantic versioning syntax"
            ) from exc
    return pins


def select_active_packages(
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> tuple[PackageInfo, ...]:
    """Return globally active packages with project pins overlaid and checked."""

    packages = _selected_active_packages(
        home=home, proj_dir=proj_dir, cwd=cwd, excluded_names=set(), env=env
    )
    _validate_requirements(packages)
    return packages


def select_package_roots(
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    development_packages: Iterable[PackageInfo] = (),
    env: Mapping[str, str] | None = None,
) -> tuple[PackageInfo, ...]:
    """Return development roots followed by non-conflicting active store roots.

    Development names are excluded before active or pinned store selections
    are resolved, so a valid development root shadows stale store state.
    """

    development = tuple(development_packages)
    development_names = {package.manifest.name for package in development}
    activated = _selected_active_packages(
        home=home,
        proj_dir=proj_dir,
        cwd=cwd,
        excluded_names=development_names,
        env=env,
    )
    selected = (
        *development,
        *(package for package in activated if package.manifest.name not in development_names),
    )
    _validate_requirements(selected)
    return selected


def _selected_active_packages(
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    excluded_names: set[str],
    env: Mapping[str, str] | None,
) -> tuple[PackageInfo, ...]:
    """Load global selections with pins, excluding development names before resolution."""

    index = load_activation_index(home=home, env=env)
    selections = dict(index.packages)
    for name, version in load_package_pins(home=home, proj_dir=proj_dir, cwd=cwd, env=env).items():
        selections[name] = ActivePackage(version)
    for name in excluded_names:
        selections.pop(name, None)
    return _packages_from_index(ActivationIndex(selections), home=home, env=env)


def _parse_activation_index(raw: TomlDict) -> ActivationIndex:
    unexpected = set(raw).difference({"packages"})
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise PackageActivationError(f"package activation index has unsupported sections: {names}")
    packages_raw = raw.get("packages", {})
    if not isinstance(packages_raw, dict):
        raise PackageActivationError("package activation index [packages] must be a table")

    packages: dict[str, ActivePackage] = {}
    for name, value in toml_dict(packages_raw).items():
        _validate_package_name(name)
        if not isinstance(value, dict):
            raise PackageActivationError(f"activation for package {name!r} must be a table")
        table = toml_dict(value)
        if set(table).difference({"version", "editable", "shadow"}):
            raise PackageActivationError(f"activation for package {name!r} has unsupported fields")
        version = _complete_version(table.get("version"), f"activation for package {name!r}")
        editable_raw = table.get("editable")
        if editable_raw is not None and (
            not isinstance(editable_raw, str)
            or not editable_raw
            or not Path(editable_raw).is_absolute()
        ):
            raise PackageActivationError(
                f"editable package {name!r} must have an absolute root path"
            )
        shadow = table.get("shadow", False)
        if not isinstance(shadow, bool):
            raise PackageActivationError(
                f"activation shadow marker for package {name!r} must be boolean"
            )
        packages[name] = ActivePackage(
            version,
            editable=None if editable_raw is None else Path(editable_raw),
            shadow=shadow,
        )
    return ActivationIndex(packages)


def _packages_from_index(
    index: ActivationIndex, *, home: Path, env: Mapping[str, str] | None
) -> tuple[PackageInfo, ...]:
    packages: list[PackageInfo] = []
    for name, active in sorted(index.packages.items()):
        if active.editable is not None:
            root = active.editable
        else:
            root = package_store_path(name, active.version, home=home, env=env).resolve()
            canonical_store_root = store_root(home=home, env=env).resolve()
            if not root.is_relative_to(canonical_store_root):
                raise PackageActivationError(
                    f"active package {name!r} resolves outside the package store root"
                )
        manifest = _load_installed_manifest(root)
        if manifest.name != name:
            raise PackageActivationError(
                f"active package {name!r} has a manifest for {manifest.name!r}"
            )
        if active.editable is None and manifest.version != active.version:
            raise PackageActivationError(
                f"active package {name!r} has a mismatched installed version"
            )
        packages.append(PackageInfo(root, manifest))
    return tuple(packages)


def _load_installed_manifest(root: Path) -> PackageManifest:
    try:
        return load_manifest(root / "package.toml")
    except ManifestError as exc:
        raise PackageActivationError(f"cannot load active package at {root}: {exc}") from exc


def _validate_requirements(packages: tuple[PackageInfo, ...]) -> None:
    selected = {package.manifest.name: package for package in packages}
    for package in packages:
        for name, requirement in package.manifest.dependencies.items():
            dependency = selected.get(name)
            if dependency is None:
                raise PackageActivationError(
                    f"selected package {package.manifest.name!r} requires package {name!r} "
                    f"at least {requirement.version}, but it is not selected"
                )
            if dependency.manifest.version < requirement.version:
                raise PackageActivationError(
                    f"selected package {package.manifest.name!r} requires package {name!r} "
                    f"at least {requirement.version}, but selected {dependency.manifest.version}"
                )


def _complete_version(value: object, context: str) -> semver.Version:
    if not isinstance(value, str):
        raise PackageActivationError(f"{context} requires a semantic version string")
    try:
        return semver.Version.parse(value)
    except ValueError as exc:
        raise PackageActivationError(
            f"{context} requires complete semantic versioning syntax"
        ) from exc


def _validate_package_name(name: str) -> None:
    try:
        module_id = ModuleId.from_path(name)
    except ValueError as exc:
        raise PackageActivationError(
            f"package name {name!r} is not a valid module segment"
        ) from exc
    if len(module_id.segments) != 1:
        raise PackageActivationError(f"package name {name!r} is not a valid module segment")


def _toml_string(value: str) -> str:
    """Return a JSON string, which is also a TOML basic string."""

    return json.dumps(value)
