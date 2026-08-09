"""Directory-package installation, dependency activation, and removal."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from agm.core import dry_run, fs
from agm.packages.activation import (
    ActivationIndex,
    ActivePackage,
    PackageActivationError,
    load_activation_index,
    validate_activation_index,
    write_activation_index,
)
from agm.packages.discipline import DisciplineError, validate_package
from agm.packages.fetch import FetchError, fetch_archive
from agm.packages.manifest import DependencySpec, ManifestError, PackageManifest, load_manifest
from agm.packages.model import PackageInfo
from agm.packages.record import RecordEntry, RecordError, verify_record, write_record
from agm.packages.store import canonical_package_store_path, store_root


class PackageInstallError(ValueError):
    """Raised when package installation or removal cannot safely proceed."""


@dataclass(slots=True)
class _InstallState:
    home: Path
    env: Mapping[str, str] | None
    index: ActivationIndex
    installing: set[Path] = field(default_factory=set)
    created: list[Path] = field(default_factory=list)
    transient_packages: dict[str, PackageInfo] = field(default_factory=dict)


def install_directory(
    source: Path,
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
    editable: bool = False,
    shadow: bool = False,
) -> PackageInfo:
    """Install a package directory or activate it as an editable package.

    Dependencies use MVS: an installed satisfying version is selected first;
    otherwise a declared local path is installed recursively.  Archive sources
    are fetched and verified before the deliberately deferred archive handoff.
    """

    try:
        index = load_activation_index(home=home, env=env)
    except PackageActivationError as exc:
        raise PackageInstallError(f"cannot load package activation: {exc}") from exc
    state = _InstallState(home=home, env=env, index=index)
    try:
        package = _install_directory(source, state=state, editable=editable, shadow=shadow)
        _commit_activation(
            state.index,
            home=home,
            env=env,
            transient_packages=state.transient_packages if dry_run.enabled() else None,
        )
    except PackageInstallError:
        _rollback_created_trees(state)
        raise
    return package


def uninstall_package(name: str, *, home: Path, env: Mapping[str, str] | None = None) -> None:
    """Remove the active package tree by its verified ``RECORD``.

    Editable packages have no copied tree or record, so removal only drops
    their activation selection.
    """

    try:
        index = load_activation_index(home=home, env=env)
    except PackageActivationError as exc:
        raise PackageInstallError(f"cannot load package activation: {exc}") from exc
    active = index.packages.get(name)
    if active is None:
        raise PackageInstallError(f"package {name!r} is not installed")
    root: Path | None = None
    entries: tuple[RecordEntry, ...] = ()
    if active.editable is None:
        try:
            root = canonical_package_store_path(name, active.version, home=home, env=env)
            entries = verify_record(root)
        except (RecordError, ValueError) as exc:
            raise PackageInstallError(
                f"package integrity check failed for {name!r}: {exc}"
            ) from exc
    packages = dict(index.packages)
    del packages[name]
    _commit_activation(ActivationIndex(packages), home=home, env=env)
    if root is not None:
        _remove_recorded_tree(root, entries)


def installed_packages(
    *, home: Path, env: Mapping[str, str] | None = None
) -> tuple[PackageInfo, ...]:
    """Return all installed immutable package versions, sorted by identity."""

    root = store_root(home=home, env=env)
    if not root.is_dir():
        return ()
    packages: list[PackageInfo] = []
    for name_dir in sorted(root.iterdir()):
        if not name_dir.is_dir() or name_dir.is_symlink():
            continue
        for version_dir in sorted(name_dir.iterdir()):
            if not version_dir.is_dir() or version_dir.is_symlink():
                continue
            try:
                package = PackageInfo(version_dir, load_manifest(version_dir / "package.toml"))
            except ManifestError as exc:
                raise PackageInstallError(
                    f"cannot load installed package at {version_dir}: {exc}"
                ) from exc
            if (
                package.manifest.name != name_dir.name
                or str(package.manifest.version) != version_dir.name
            ):
                raise PackageInstallError(
                    f"installed package at {version_dir} does not match its store identity"
                )
            packages.append(package)
    return tuple(packages)


def _install_directory(
    source: Path, *, state: _InstallState, editable: bool, shadow: bool
) -> PackageInfo:
    if source.is_symlink():
        raise PackageInstallError(f"cannot install symbolic-link package root {source}")
    root = source.resolve()
    if root in state.installing:
        raise PackageInstallError(f"cyclic path dependency at {root}")
    try:
        package = PackageInfo(root, load_manifest(root / "package.toml"))
        validate_package(package)
    except (ManifestError, DisciplineError) as exc:
        raise PackageInstallError(f"cannot install package from {source}: {exc}") from exc

    state.installing.add(root)
    try:
        _resolve_dependencies(package, state)
    finally:
        state.installing.remove(root)

    if editable:
        installed = package
    else:
        try:
            destination = canonical_package_store_path(
                package.manifest.name, package.manifest.version, home=state.home, env=state.env
            )
        except ValueError as exc:
            raise PackageInstallError(
                f"cannot install package {package.manifest.name!r}: {exc}"
            ) from exc
        installed = PackageInfo(destination, package.manifest)
        if destination.exists():
            _verify_existing_install(destination, package.manifest)
        elif not dry_run.enabled():
            fs.mkdir(destination.parent, parents=True, exist_ok=True)
            try:
                fs.copy_tree(root, destination)
                write_record(destination)
                verify_record(destination)
                state.created.append(destination)
            except (OSError, ValueError, RecordError) as exc:
                if destination.exists():
                    fs.rmtree(destination)
                raise PackageInstallError(
                    f"cannot install package {package.manifest.name!r}: {exc}"
                ) from exc
        else:
            fs.mkdir(destination.parent, parents=True, exist_ok=True)
            fs.copy_tree(root, destination)

    packages = dict(state.index.packages)
    packages[package.manifest.name] = ActivePackage(
        package.manifest.version,
        editable=root if editable else None,
        shadow=shadow,
    )
    state.index = ActivationIndex(packages)
    state.transient_packages[package.manifest.name] = installed
    return installed


def _resolve_dependencies(package: PackageInfo, state: _InstallState) -> None:
    for name, requirement in package.manifest.dependencies.items():
        selected = _installed_satisfying(name, requirement, state)
        if selected is None and requirement.path is not None:
            selected = _install_directory(
                package.root / requirement.path, state=state, editable=False, shadow=False
            )
        if selected is None and requirement.url is not None and requirement.hash is not None:
            _fetch_deferred_archive(
                name,
                str(requirement.version),
                url=requirement.url,
                expected_hash=requirement.hash,
                state=state,
            )
        if selected is None:
            raise PackageInstallError(
                f"unsatisfied package requirement {name!r} >= {requirement.version}"
            )
        if selected.manifest.name != name or selected.manifest.version < requirement.version:
            raise PackageInstallError(
                f"unsatisfied package requirement {name!r} >= {requirement.version}"
            )
        packages = dict(state.index.packages)
        packages[name] = ActivePackage(selected.manifest.version)
        state.index = ActivationIndex(packages)
        state.transient_packages[name] = selected


def _installed_satisfying(
    name: str, requirement: DependencySpec, state: _InstallState
) -> PackageInfo | None:
    candidates = [
        (package, True)
        for package in installed_packages(home=state.home, env=state.env)
        if package.manifest.name == name and package.manifest.version >= requirement.version
    ]
    if dry_run.enabled():
        candidates.extend(
            (package, False)
            for package in state.transient_packages.values()
            if package.manifest.name == name and package.manifest.version >= requirement.version
        )
    if not candidates:
        return None
    selected, verify_selected = candidates[0]
    for candidate, verify_candidate in candidates[1:]:
        if candidate.manifest.version > selected.manifest.version:
            selected = candidate
            verify_selected = verify_candidate
    if verify_selected:
        try:
            verify_record(selected.root)
        except RecordError as exc:
            raise PackageInstallError(
                f"package integrity check failed for {name!r}: {exc}"
            ) from exc
    return selected


def _fetch_deferred_archive(
    name: str, version: str, *, url: str, expected_hash: str, state: _InstallState
) -> None:
    """Fetch a verified archive and fail at M6's not-yet-implemented handoff."""

    requirement = f"{name} >= {version}"
    if dry_run.enabled():
        dry_run.print_operation("fetch-package", url)
        raise _deferred_archive_install_error(requirement)
    scratch = store_root(home=state.home, env=state.env)
    try:
        fs.mkdir(scratch, parents=True, exist_ok=True)
    except OSError as exc:
        error = FetchError(f"fetch failed for {name} >= {version}: {exc}")
        raise PackageInstallError(str(error)) from error
    try:
        fetch_archive(
            requirement=requirement,
            url=url,
            expected_hash=expected_hash,
            handoff=partial(_archive_install_deferred, requirement=requirement),
            scratch_dir=scratch if scratch.is_dir() else None,
        )
    except FetchError as exc:
        raise PackageInstallError(str(exc)) from exc


def _archive_install_deferred(_: Path, *, requirement: str) -> None:
    raise _deferred_archive_install_error(requirement)


def _deferred_archive_install_error(requirement: str) -> PackageInstallError:
    return PackageInstallError(
        f"archive package installation is not available yet for {requirement}"
    )


def _verify_existing_install(root: Path, manifest: PackageManifest) -> None:
    try:
        installed = load_manifest(root / "package.toml")
        if installed != manifest:
            raise PackageInstallError(
                f"installed package at {root} disagrees with the source manifest"
            )
        verify_record(root)
    except (ManifestError, RecordError) as exc:
        raise PackageInstallError(
            f"package integrity check failed for {manifest.name!r}: {exc}"
        ) from exc


def _commit_activation(
    index: ActivationIndex,
    *,
    home: Path,
    env: Mapping[str, str] | None,
    transient_packages: Mapping[str, PackageInfo] | None = None,
) -> None:
    """Validate and atomically publish a complete activation selection."""

    try:
        validate_activation_index(
            index,
            home=home,
            env=env,
            transient_packages=transient_packages,
        )
        write_activation_index(index, home=home, env=env)
    except PackageActivationError as exc:
        raise PackageInstallError(f"cannot write package activation: {exc}") from exc


def _rollback_created_trees(state: _InstallState) -> None:
    """Discard immutable trees created by a failed install before they became active."""

    if dry_run.enabled():
        return
    for root in reversed(state.created):
        fs.rmtree(root)


def _remove_recorded_tree(root: Path, entries: tuple[RecordEntry, ...]) -> None:
    """Remove an already-verified immutable tree using only core fs primitives."""

    for entry in entries:
        fs.unlink(root / entry.path)
    fs.unlink(root / "RECORD")
    directories = sorted((path for path in fs.rglob(root, "*") if path.is_dir()), reverse=True)
    for directory in directories:
        fs.rmdir(directory)
    fs.rmdir(root)
