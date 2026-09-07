"""Installed-package activation, project pins, and package-root selection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import cast

import semver
import tomlkit
from tomlkit.exceptions import TOMLKitError

from agm.command_catalog import invalid_command_path
from agm.config.general import agm_home_dir, load_merged_config
from agm.core.fs import mkdir, write_text_atomic
from agm.core.toml import TomlDict, dumps_toml, empty_toml_doc, load_toml_file, toml_dict
from agm.packages.layout import activation_index_path as _resolved_activation_index_path
from agm.packages.manifest import (
    ManifestError,
    PackageManifest,
    expanded_commands,
    load_manifest,
    validate_package_name,
)
from agm.packages.model import (
    STD_PACKAGE_NAME,
    PackageInfo,
    canonical_package_identity,
    is_std_package_name,
    select_satisfying,
    unmet_std_requirement,
)
from agm.packages.store import (
    StoreIdentityError,
    canonical_package_provenance_path,
    canonical_package_store_path,
    iter_installed_packages,
    iter_store_package_dirs,
)


class PackageActivationError(ValueError):
    """Raised when activation state, pins, or their requirements are invalid."""


@dataclass(frozen=True, slots=True)
class ActivePackage:
    """One globally active package selection.

    ``editable`` names a live package root. Otherwise ``version`` identifies
    an immutable tree in the versioned package store. ``registration_order``
    records when this package last claimed its manifest commands.
    """

    version: semver.Version
    editable: Path | None = None
    shadow: bool = False
    registration_order: int = 0

    def __post_init__(self) -> None:
        if self.editable is not None:
            object.__setattr__(self, "editable", self.editable.resolve())


def active_package_version(active: ActivePackage) -> semver.Version:
    """Return an active package's current version from its immutable or editable source."""

    if active.editable is None:
        return active.version
    return load_manifest(active.editable / "package.toml").version


@dataclass(frozen=True, slots=True)
class CommandRegistration:
    """One installed package command recorded in the activation index."""

    package: str
    program: str | None
    description: str | None = None
    help: str | None = None


@dataclass(frozen=True, slots=True)
class PackageProvenance:
    """Durable command-priority metadata stored beside an immutable package tree."""

    registration_order: int
    shadow: bool


@dataclass(frozen=True, slots=True)
class CommandShadow:
    """A winning package command and the active package owners it displaces."""

    path_name: str
    displaced_packages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ActivationIndex:
    """The global package selection and cached command registry."""

    packages: dict[str, ActivePackage] = field(default_factory=dict)
    commands: dict[str, CommandRegistration] = field(default_factory=dict)


def activation_index_path(*, home: Path, env: Mapping[str, str] | None = None) -> Path:
    """Return the selected AGM home's package activation-index path."""

    return _resolved_activation_index_path(agm_home_dir(home=home, env=env))


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
    """Atomically write a canonical activation index and return its path."""

    path = activation_index_path(home=home, env=env)
    doc = empty_toml_doc()

    packages_table = tomlkit.table()
    for name, active in sorted(index.packages.items()):
        _validate_active_package(name, active)
        package_table = tomlkit.table()
        package_table["version"] = str(active.version)
        if active.editable is not None:
            package_table["editable"] = str(active.editable)
        if active.shadow:
            package_table["shadow"] = True
        package_table["registration-order"] = active.registration_order
        packages_table[name] = package_table
    doc["packages"] = packages_table

    commands_table = tomlkit.table()
    for path_name, command in sorted(index.commands.items()):
        _validate_command_registration(path_name, command, index.packages)
        command_table = tomlkit.table()
        command_table["package"] = command.package
        if command.program is not None:
            command_table["program"] = command.program
        if command.help is not None:
            command_table["help"] = command.help
        if command.description is not None:
            command_table["description"] = command.description
        commands_table[path_name] = command_table
    doc["commands"] = commands_table

    try:
        mkdir(path.parent, parents=True, exist_ok=True)
        write_text_atomic(path, dumps_toml(doc))
    except OSError as exc:
        raise PackageActivationError(
            f"cannot write package activation index {path}: {exc}"
        ) from exc
    return path


def package_provenance_path(
    name: str, version: semver.Version, *, home: Path, env: Mapping[str, str] | None = None
) -> Path:
    """Return the canonical sidecar path for one immutable package version."""

    try:
        return canonical_package_provenance_path(name, version, home=home, env=env)
    except ValueError as exc:
        raise PackageActivationError(
            f"package provenance for {name!r} resolves outside the package store root"
        ) from exc


def load_package_provenance(
    name: str, version: semver.Version, *, home: Path, env: Mapping[str, str] | None = None
) -> PackageProvenance | None:
    """Load one immutable package's optional command-priority sidecar."""

    path = package_provenance_path(name, version, home=home, env=env)
    if not path.exists():
        return None
    try:
        raw = load_toml_file(path)
    except (OSError, TOMLKitError, UnicodeDecodeError) as exc:
        raise PackageActivationError(f"cannot load package provenance {path}: {exc}") from exc
    return _parse_package_provenance(raw, path)


def write_package_provenance(
    name: str,
    active: ActivePackage,
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Atomically persist activation priority outside the immutable package payload."""

    _validate_active_package(name, active)
    if active.editable is not None:
        raise PackageActivationError(f"editable package {name!r} has no store provenance")
    path = package_provenance_path(name, active.version, home=home, env=env)
    doc = empty_toml_doc()
    activation_table = tomlkit.table()
    activation_table["registration-order"] = active.registration_order
    activation_table["shadow"] = active.shadow
    doc["activation"] = activation_table
    try:
        mkdir(path.parent, parents=True, exist_ok=True)
        write_text_atomic(path, dumps_toml(doc))
    except OSError as exc:
        raise PackageActivationError(f"cannot write package provenance {path}: {exc}") from exc
    return path


def validate_activation_index(
    index: ActivationIndex,
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
    transient_packages: Mapping[str, PackageInfo] | None = None,
    packages: tuple[PackageInfo, ...] | None = None,
) -> None:
    """Ensure every activation resolves to matching packages with satisfied requirements.

    ``transient_packages`` supplies validated package manifests for selections
    that an install has planned but not yet written to the immutable store.
    ``packages`` lets a caller that already resolved and verified the active
    package set for ``index`` reuse it instead of re-resolving it here.
    """

    _validate_commands(index)
    _validate_requirements(
        _resolved_index_packages(
            index, packages, home=home, env=env, transient_packages=transient_packages
        )
    )


def rebuild_activation_index(
    *, home: Path, env: Mapping[str, str] | None = None
) -> ActivationIndex:
    """Build the deterministic installed-package selection from store manifests.

    Store versions are side-by-side, so the highest semantic version for each
    package is active after a rebuild. Package-local provenance sidecars retain
    command priority independently of package-name iteration. Editable
    selections are not inferable from the immutable store.
    """

    previous = load_activation_index(home=home, env=env)
    for name_dir in iter_store_package_dirs(home=home, env=env):
        _validate_package_name(name_dir.name)

    candidates_by_name: dict[str, list[PackageInfo]] = {}
    try:
        for installed in iter_installed_packages(
            home=home,
            env=env,
            on_manifest_error=lambda exc, version_dir: PackageActivationError(
                f"cannot load active package at {version_dir}: {exc}"
            ),
        ):
            candidates_by_name.setdefault(installed.manifest.name, []).append(installed)
    except StoreIdentityError as exc:
        raise PackageActivationError(str(exc)) from exc

    active: dict[str, ActivePackage] = {}
    for name, candidates in candidates_by_name.items():
        # ``candidates`` is only ever created together with its first
        # element, so it is always non-empty and ``select_satisfying`` always
        # returns a winner.
        selected = cast(PackageInfo, select_satisfying(candidates, previous.packages.get(name)))
        active[name] = ActivePackage(selected.manifest.version)

    provenance = {
        name: load_package_provenance(name, package.version, home=home, env=env)
        for name, package in active.items()
    }
    _validate_rebuild_provenance(provenance)
    persisted_orders = {item.registration_order for item in provenance.values() if item is not None}
    next_order = max(
        (
            *(package.registration_order for package in previous.packages.values()),
            *persisted_orders,
        ),
        default=0,
    )
    used_orders = set(persisted_orders)
    rebuilt: dict[str, ActivePackage] = {}
    for name, package in sorted(active.items()):
        persisted = provenance[name]
        previous_package = previous.packages.get(name)
        if persisted is not None:
            rebuilt[name] = ActivePackage(
                package.version,
                shadow=persisted.shadow,
                registration_order=persisted.registration_order,
            )
        elif (
            previous_package is not None
            and previous_package.editable is None
            and previous_package.registration_order not in used_orders
            and canonical_package_identity(name, previous_package.version)
            == canonical_package_identity(name, package.version)
        ):
            rebuilt[name] = ActivePackage(
                package.version,
                shadow=previous_package.shadow,
                registration_order=previous_package.registration_order,
            )
            used_orders.add(previous_package.registration_order)
        else:
            next_order += 1
            rebuilt[name] = ActivePackage(package.version, registration_order=next_order)
            used_orders.add(next_order)
    index = ActivationIndex(packages=rebuilt)
    packages = resolve_indexed_packages(index, home=home, env=env)
    _validate_requirements(packages)
    return _reconciled_commands(index, packages, provenance=provenance)


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
    index: ActivationIndex | None = None,
) -> tuple[PackageInfo, ...]:
    """Return globally active packages with project pins overlaid and checked.

    ``index`` lets a caller that already loaded the activation index reuse it
    instead of reloading it here.
    """

    packages = _selected_active_packages(
        home=home, proj_dir=proj_dir, cwd=cwd, excluded_names=set(), env=env, index=index
    )
    _validate_requirements(packages)
    return packages


def effective_command_index(
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    index: ActivationIndex | None = None,
) -> ActivationIndex:
    """Return selected-package commands with project pins applied.

    The persistent activation index caches commands for the global selection.
    A project pin changes that selection, so derive its command registry from
    the selected manifests and the persisted registration priority of each
    selected immutable version. ``index`` lets a caller that already loaded
    the activation index reuse it instead of reloading it here.
    """

    if index is None:
        index = load_activation_index(home=home, env=env)
    pins = load_package_pins(home=home, proj_dir=proj_dir, cwd=cwd, env=env)
    packages = _selected_active_packages(
        home=home,
        proj_dir=proj_dir,
        cwd=cwd,
        excluded_names=set(),
        env=env,
        index=index,
        pins=pins,
    )
    _validate_requirements(packages)
    # Without pins the selection is exactly the index, so the cached command
    # registry already describes it — unless an editable package's manifest
    # may have changed its commands since the index was written.
    if not pins and all(active.editable is None for active in index.packages.values()):
        return _reconciled_commands(index, packages)
    selections: dict[str, ActivePackage] = {}
    provenance: dict[str, ActivePackage | PackageProvenance | None] = {}
    for package in packages:
        name = package.manifest.name
        previous = index.packages.get(name)
        retains_global_selection = previous is not None and name not in pins
        metadata: ActivePackage | PackageProvenance | None
        if retains_global_selection:
            metadata = previous
        else:
            metadata = load_package_provenance(
                name,
                package.manifest.version,
                home=home,
                env=env,
            )
            if metadata is None:
                # Pre-provenance installations have only package-name priority
                # in the activation index. Preserve that compatibility fallback.
                metadata = previous
        provenance[name] = metadata
        selections[name] = ActivePackage(
            package.manifest.version,
            editable=(
                previous.editable if previous is not None and retains_global_selection else None
            ),
            shadow=metadata.shadow if metadata is not None else False,
            registration_order=metadata.registration_order if metadata is not None else 0,
        )
    return _reconciled_commands(
        ActivationIndex(selections),
        packages,
        provenance=provenance,
    )


def select_package_roots(
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    development_packages: Iterable[PackageInfo] = (),
    env: Mapping[str, str] | None = None,
    index: ActivationIndex | None = None,
) -> tuple[PackageInfo, ...]:
    """Return development roots followed by non-conflicting active store roots.

    Development names are excluded before active or pinned store selections
    are resolved, so a valid development root shadows stale store state.
    ``index`` lets a caller that already loaded the activation index reuse it
    instead of reloading it here.
    """

    development = tuple(
        package
        for package in development_packages
        if not is_std_package_name(package.manifest.name)
    )
    development_names = {package.manifest.name for package in development}
    # ``std`` is selected exclusively by ``resolve_stdlib_root`` so an
    # override or source-checkout fallback cannot also mount an active tree.
    excluded_names = {*development_names, STD_PACKAGE_NAME}
    activated = _selected_active_packages(
        home=home,
        proj_dir=proj_dir,
        cwd=cwd,
        excluded_names=excluded_names,
        env=env,
        index=index,
    )
    selected = (
        *development,
        *(package for package in activated if package.manifest.name not in excluded_names),
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
    index: ActivationIndex | None = None,
    pins: Mapping[str, semver.Version] | None = None,
) -> tuple[PackageInfo, ...]:
    """Load global selections with pins, excluding development names before resolution."""

    if index is None:
        index = load_activation_index(home=home, env=env)
    if pins is None:
        pins = load_package_pins(home=home, proj_dir=proj_dir, cwd=cwd, env=env)
    selections = dict(index.packages)
    for name, version in pins.items():
        selections[name] = ActivePackage(version)
    for name in excluded_names:
        selections.pop(name, None)
    return resolve_indexed_packages(ActivationIndex(selections), home=home, env=env)


def validate_package_command_conflicts(
    index: ActivationIndex,
    manifest: PackageManifest,
    *,
    shadow: bool,
    home: Path,
    env: Mapping[str, str] | None = None,
    transient_packages: Mapping[str, PackageInfo] | None = None,
    packages: tuple[PackageInfo, ...] | None = None,
) -> None:
    """Reject a candidate's conflicts with every other active manifest.

    The active command registry contains only each command's winner. Looking
    at manifests instead retains owners previously displaced by ``--shadow``
    when their winner is updated. ``packages`` lets a caller that already
    resolved and verified the active package set for ``index`` reuse it
    instead of re-resolving it here.
    """

    if shadow:
        return
    resolved = _resolved_index_packages(
        index, packages, home=home, env=env, transient_packages=transient_packages
    )
    candidate_paths = set(expanded_commands(manifest))
    for package in resolved:
        if package.manifest.name == manifest.name:
            continue
        conflicts = sorted(candidate_paths.intersection(expanded_commands(package.manifest)))
        if conflicts:
            path_name = conflicts[0]
            raise PackageActivationError(
                f"command {path_name!r} from package {manifest.name!r} conflicts with "
                f"package {package.manifest.name!r}; install with --shadow to replace it"
            )


def merge_package_commands(
    index: ActivationIndex, manifest: PackageManifest, *, shadow: bool
) -> ActivationIndex:
    """Replace a package's registrations, rejecting conflicting ownership.

    A package update removes its former registrations before merging its current
    manifest. ``shadow`` permits it to replace a command owned by another
    active package. The complete registry is reconciled from active manifests
    before an activation is written.
    """

    commands = {
        path_name: registration
        for path_name, registration in index.commands.items()
        if registration.package != manifest.name
    }
    for path_name, spec in sorted(expanded_commands(manifest).items()):
        existing = commands.get(path_name)
        if existing is not None and existing.package != manifest.name and not shadow:
            raise PackageActivationError(
                f"command {path_name!r} from package {manifest.name!r} conflicts with "
                f"package {existing.package!r}; install with --shadow to replace it"
            )
        commands[path_name] = CommandRegistration(
            package=manifest.name,
            program=spec.program,
            description=spec.description,
            help=spec.help,
        )
    updated = ActivationIndex(dict(index.packages), commands)
    _validate_commands(updated)
    return updated


def reconcile_package_commands(
    index: ActivationIndex,
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
    transient_packages: Mapping[str, PackageInfo] | None = None,
    packages: tuple[PackageInfo, ...] | None = None,
) -> ActivationIndex:
    """Derive current command owners from the active package manifests.

    ``packages`` lets a caller that already resolved and verified the active
    package set for ``index`` reuse it instead of re-resolving it here.
    """

    resolved = _resolved_index_packages(
        index, packages, home=home, env=env, transient_packages=transient_packages
    )
    return _reconciled_commands(index, resolved)


def _reconciled_commands(
    index: ActivationIndex,
    packages: Iterable[PackageInfo],
    *,
    provenance: Mapping[str, ActivePackage | PackageProvenance | None] | None = None,
) -> ActivationIndex:
    package_list = tuple(packages)
    _validate_command_conflict_provenance(
        package_list,
        index.packages if provenance is None else provenance,
    )

    commands: dict[str, CommandRegistration] = {}
    for package in sorted(package_list, key=partial(_command_registration_key, index)):
        for path_name, spec in sorted(expanded_commands(package.manifest).items()):
            commands[path_name] = CommandRegistration(
                package.manifest.name,
                spec.program,
                spec.description,
                spec.help,
            )
    reconciled = ActivationIndex(dict(index.packages), commands)
    _validate_commands(reconciled)
    return reconciled


def command_shadow_diagnostics(
    index: ActivationIndex,
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
    transient_packages: Mapping[str, PackageInfo] | None = None,
    packages: tuple[PackageInfo, ...] | None = None,
) -> dict[str, tuple[CommandShadow, ...]]:
    """Return active winners and the package owners each one shadows.

    ``packages`` lets a caller that already resolved and verified the active
    package set for ``index`` reuse it instead of re-resolving it here.
    """

    resolved = _resolved_index_packages(
        index, packages, home=home, env=env, transient_packages=transient_packages
    )
    by_path = _command_owners(resolved)
    diagnostics: dict[str, list[CommandShadow]] = {}
    for path_name, owners in by_path.items():
        if len(owners) < 2:
            continue
        winner = max(owners, key=partial(_command_registration_key, index))
        displaced = tuple(
            sorted(package.manifest.name for package in owners if package is not winner)
        )
        diagnostics.setdefault(winner.manifest.name, []).append(CommandShadow(path_name, displaced))
    return {
        package: tuple(sorted(shadows, key=_command_shadow_path))
        for package, shadows in diagnostics.items()
    }


def _command_registration_key(index: ActivationIndex, package: PackageInfo) -> tuple[int, str]:
    active = index.packages[package.manifest.name]
    return active.registration_order, package.manifest.name


def _command_shadow_path(shadow: CommandShadow) -> str:
    return shadow.path_name


def _parse_package_provenance(raw: TomlDict, path: Path) -> PackageProvenance:
    if set(raw) != {"activation"}:
        raise PackageActivationError(f"package provenance {path} has unsupported sections")
    activation = raw.get("activation")
    if not isinstance(activation, dict):
        raise PackageActivationError(f"package provenance {path} has an invalid activation table")
    table = toml_dict(activation)
    if set(table) != {"registration-order", "shadow"}:
        raise PackageActivationError(f"package provenance {path} has unsupported fields")
    registration_order = table.get("registration-order")
    shadow = table.get("shadow")
    if (
        not isinstance(registration_order, int)
        or isinstance(registration_order, bool)
        or registration_order < 1
        or not isinstance(shadow, bool)
    ):
        raise PackageActivationError(f"package provenance {path} has invalid activation metadata")
    return PackageProvenance(registration_order, shadow)


def _validate_rebuild_provenance(provenance: Mapping[str, PackageProvenance | None]) -> None:
    orders = [item.registration_order for item in provenance.values() if item is not None]
    if len(orders) != len(set(orders)):
        raise PackageActivationError("package provenance has duplicate registration order")


def _command_owners(packages: Iterable[PackageInfo]) -> dict[str, list[PackageInfo]]:
    owners: dict[str, list[PackageInfo]] = {}
    for package in packages:
        for path_name in expanded_commands(package.manifest):
            owners.setdefault(path_name, []).append(package)
    return owners


def _validate_command_conflict_provenance(
    packages: Iterable[PackageInfo],
    provenance: Mapping[str, ActivePackage | PackageProvenance | None],
) -> None:
    """Reject command collisions without a coherent shadow history."""

    for path_name, owners in _command_owners(packages).items():
        if len(owners) < 2:
            continue
        registered_owners: list[tuple[str, ActivePackage | PackageProvenance]] = []
        for owner in owners:
            name = owner.manifest.name
            metadata = provenance.get(name)
            if metadata is None:
                raise PackageActivationError(
                    f"cannot reconcile command {path_name!r}: its package provenance is missing"
                )
            registered_owners.append((name, metadata))
        registered_owners.sort(key=_command_provenance_order)
        for name, metadata in registered_owners[1:]:
            if not metadata.shadow:
                raise PackageActivationError(
                    f"cannot reconcile command {path_name!r}: later registration from "
                    f"package {name!r} lacks shadow intent"
                )


def _command_provenance_order(
    owner: tuple[str, ActivePackage | PackageProvenance],
) -> tuple[int, str]:
    return owner[1].registration_order, owner[0]


def _parse_activation_index(raw: TomlDict) -> ActivationIndex:
    unexpected = set(raw).difference({"packages", "commands"})
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
        if set(table).difference({"version", "editable", "shadow", "registration-order"}):
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
        registration_order = table.get("registration-order", 0)
        if (
            not isinstance(registration_order, int)
            or isinstance(registration_order, bool)
            or registration_order < 0
        ):
            raise PackageActivationError(
                f"activation registration order for package {name!r} must be a non-negative integer"
            )
        packages[name] = ActivePackage(
            version,
            editable=None if editable_raw is None else Path(editable_raw),
            shadow=shadow,
            registration_order=registration_order,
        )
    commands_raw = raw.get("commands", {})
    if not isinstance(commands_raw, dict):
        raise PackageActivationError("package activation index [commands] must be a table")
    commands: dict[str, CommandRegistration] = {}
    for path_name, value in toml_dict(commands_raw).items():
        if not isinstance(value, dict):
            raise PackageActivationError(f"command registration {path_name!r} must be a table")
        table = toml_dict(value)
        if set(table).difference({"package", "program", "description", "help"}):
            raise PackageActivationError(
                f"command registration {path_name!r} has unsupported fields"
            )
        package = table.get("package")
        program = table.get("program")
        description = table.get("description")
        help_text = table.get("help")
        if (
            not isinstance(package, str)
            or (program is not None and (not isinstance(program, str) or not program))
            or not package
        ):
            raise PackageActivationError(
                f"command registration {path_name!r} requires non-empty package and program strings"
            )
        if description is not None and (not isinstance(description, str) or not description):
            raise PackageActivationError(
                f"command registration {path_name!r} has invalid description"
            )
        if help_text is not None and (not isinstance(help_text, str) or not help_text):
            raise PackageActivationError(f"command registration {path_name!r} has invalid help")
        commands[path_name] = CommandRegistration(package, program, description, help_text)
    index = ActivationIndex(packages, commands)
    _validate_commands(index)
    return index


def _validate_active_package(name: str, active: ActivePackage) -> None:
    _validate_package_name(name)
    if (
        not isinstance(active.registration_order, int)
        or isinstance(active.registration_order, bool)
        or active.registration_order < 0
    ):
        raise PackageActivationError(
            f"activation registration order for package {name!r} must be a non-negative integer"
        )


def _validate_commands(index: ActivationIndex) -> None:
    for path_name, command in index.commands.items():
        if command.program is None and not any(
            path.startswith(path_name + " ") for path in index.commands
        ):
            raise PackageActivationError(f"command group {path_name!r} requires subcommands")
        _validate_command_registration(path_name, command, index.packages)


def _validate_command_registration(
    path_name: str, command: CommandRegistration, packages: Mapping[str, ActivePackage]
) -> None:
    invalid = invalid_command_path(path_name)
    if invalid is not None:
        raise PackageActivationError(f"command path {path_name!r} {invalid}")
    _validate_package_name(command.package)
    if command.package not in packages:
        raise PackageActivationError(
            f"command path {path_name!r} names inactive package {command.package!r}"
        )
    if command.program == "":
        raise PackageActivationError(f"command path {path_name!r} requires a program reference")
    if command.help == "":
        raise PackageActivationError(f"command path {path_name!r} has invalid help")
    if command.description is not None and not command.description:
        raise PackageActivationError(f"command path {path_name!r} has an invalid description")


def resolve_active_package(
    name: str,
    active: ActivePackage,
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
) -> PackageInfo:
    """Resolve one activation selection into its root and installed manifest.

    An editable selection names its own live root; an immutable one must
    resolve inside the store, and its installed manifest must agree with the
    activated identity. The read path does not verify tree contents against
    ``RECORD``; installation and archive verification own that check.
    """

    if active.editable is not None:
        root = active.editable
    else:
        try:
            root = canonical_package_store_path(name, active.version, home=home, env=env)
        except ValueError as exc:
            raise PackageActivationError(
                f"active package {name!r} resolves outside the package store root"
            ) from exc
    return _checked_active_package(name, active, root, _load_installed_manifest(root))


def _checked_active_package(
    name: str, active: ActivePackage, root: Path, manifest: PackageManifest
) -> PackageInfo:
    """Reject a resolved package whose manifest disagrees with its activation."""

    if manifest.name != name:
        raise PackageActivationError(
            f"active package {name!r} has a manifest for {manifest.name!r}"
        )
    if active.editable is None and canonical_package_identity(
        manifest.name, manifest.version
    ) != canonical_package_identity(name, active.version):
        raise PackageActivationError(f"active package {name!r} has a mismatched installed version")
    return PackageInfo(root, manifest)


def _resolved_index_packages(
    index: ActivationIndex,
    packages: tuple[PackageInfo, ...] | None,
    *,
    home: Path,
    env: Mapping[str, str] | None,
    transient_packages: Mapping[str, PackageInfo] | None,
) -> tuple[PackageInfo, ...]:
    """Return *packages* unchanged if given, otherwise resolve and verify *index*."""

    if packages is not None:
        return packages
    return resolve_indexed_packages(
        index, home=home, env=env, transient_packages=transient_packages
    )


def resolve_indexed_packages(
    index: ActivationIndex,
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
    transient_packages: Mapping[str, PackageInfo] | None = None,
) -> tuple[PackageInfo, ...]:
    """Resolve and verify every activation selection in *index* into a package.

    ``transient_packages`` supplies validated manifests for selections an
    install has planned but not yet written to the immutable store, standing
    in for a store manifest lookup for just those names.
    """

    packages: list[PackageInfo] = []
    for name, active in sorted(index.packages.items()):
        transient = None if transient_packages is None else transient_packages.get(name)
        if transient is None:
            packages.append(resolve_active_package(name, active, home=home, env=env))
        else:
            packages.append(
                _checked_active_package(name, active, transient.root, transient.manifest)
            )
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
            if is_std_package_name(name):
                unmet = unmet_std_requirement(requirement)
                if unmet is not None:
                    raise PackageActivationError(
                        f"selected package {package.manifest.name!r} {unmet}"
                    )
                continue
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
        validate_package_name(name)
    except ManifestError as exc:
        raise PackageActivationError(str(exc)) from exc
