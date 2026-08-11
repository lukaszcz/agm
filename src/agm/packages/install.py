"""Directory-package installation, dependency activation, and removal."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import mkdtemp

import semver

from agm.core import dry_run, fs
from agm.packages.activation import (
    ActivationIndex,
    ActivePackage,
    PackageActivationError,
    load_activation_index,
    load_package_provenance,
    merge_package_commands,
    package_provenance_path,
    reconcile_package_commands,
    validate_activation_index,
    validate_package_command_conflicts,
    write_activation_index,
    write_package_provenance,
)
from agm.packages.archive import (
    ArchiveError,
    ArchiveMetadata,
    extract_archive,
    verify_archive_discipline,
)
from agm.packages.discipline import DisciplineError, validate_package
from agm.packages.fetch import FetchError, fetch_archive
from agm.packages.manifest import DependencySpec, ManifestError, PackageManifest, load_manifest
from agm.packages.model import PackageInfo, canonical_package_identity
from agm.packages.record import (
    RecordEntry,
    RecordError,
    content_hash,
    read_record,
    record_entries,
    verify_record,
    write_record,
)
from agm.packages.store import canonical_package_store_path, package_store_path, store_root
from agm.stdlib_locator import shipped_stdlib_root
from agm.version import AGM_VERSION


class PackageInstallError(ValueError):
    """Raised when package installation or removal cannot safely proceed."""


@dataclass(frozen=True, slots=True)
class PackageInstallPlan:
    """An installed package and the complete activation selected for it."""

    package: PackageInfo
    activation_index: ActivationIndex
    transient_packages: Mapping[str, PackageInfo]


@contextmanager
def _package_operation_lock(*, home: Path, env: Mapping[str, str] | None) -> Iterator[None]:
    """Serialize package-store mutations through a persistent advisory lock."""

    if dry_run.enabled():
        yield
        return
    root = store_root(home=home, env=env)
    try:
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / ".lock").open("a+", encoding="utf-8")
    except OSError as exc:
        raise PackageInstallError(f"cannot lock package store {root}: {exc}") from exc
    with lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            raise PackageInstallError(f"cannot lock package store {root}: {exc}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _validate_managed_stdlib_install(
    manifest: PackageManifest, *, source: Path | None, editable: bool
) -> None:
    """Require ``std`` to be the immutable, lockstep shipped package."""
    if manifest.name != "std":
        return
    if canonical_package_identity(manifest.name, manifest.version) != canonical_package_identity(
        "std", semver.Version.parse(AGM_VERSION)
    ):
        raise PackageInstallError(
            f"managed std package version must exactly match AGM version {AGM_VERSION}"
        )
    if editable:
        raise PackageInstallError("the managed std package cannot be installed editable")
    if source is None or source.resolve() != shipped_stdlib_root().resolve():
        raise PackageInstallError(
            "the managed std package must be installed from AGM's shipped stdlib"
        )


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
    otherwise a declared local path is installed recursively or a URL archive
    is fetched, hash-verified, and installed.
    """

    return install_directory_with_plan(
        source,
        home=home,
        env=env,
        editable=editable,
        shadow=shadow,
    ).package


def install_directory_with_plan(
    source: Path,
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
    editable: bool = False,
    shadow: bool = False,
) -> PackageInstallPlan:
    """Install a directory package and return its resulting activation plan."""

    with _package_operation_lock(home=home, env=env):
        state = _InstallState(home=home, env=env, index=_load_install_index(home=home, env=env))
        try:
            package = _install_directory(source, state=state, editable=editable, shadow=shadow)
            state.index = _commit_activation(
                state.index,
                home=home,
                env=env,
                transient_packages=state.transient_packages if dry_run.enabled() else None,
            )
        except PackageInstallError:
            _rollback_created_trees(state)
            raise
        return PackageInstallPlan(package, state.index, dict(state.transient_packages))


def refresh_managed_stdlib(
    source: Path,
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
) -> PackageInfo:
    """Safely replace and activate AGM's shipped standard-library package.

    The complete replacement is staged beside its versioned store path while
    holding the package-operation lock. Publication only renames complete
    trees, so readers can never observe files being copied into the active
    package directory.
    """

    with _package_operation_lock(home=home, env=env):
        package = _validated_directory_package(source)
        _validate_managed_stdlib_install(package.manifest, source=package.root, editable=False)
        _validate_minimum_agm(package.manifest)
        try:
            destination = canonical_package_store_path(
                package.manifest.name, package.manifest.version, home=home, env=env
            )
        except ValueError as exc:
            raise PackageInstallError(f"cannot refresh managed std package: {exc}") from exc
        installed = PackageInfo(destination, package.manifest)
        if dry_run.enabled():
            dry_run.print_operation("refresh-managed-stdlib", str(source))
            return installed

        staging: Path | None = None
        published: tuple[Path, Path | None] | None = None
        try:
            staging = _stage_directory_package(package.root, package, destination)
            previous = _publish_staged_refresh(staging, destination)
            published = (staging, previous)
            state = _InstallState(
                home=home,
                env=env,
                index=_load_install_index(home=home, env=env),
            )
            _resolve_dependencies(package, state)
            _activate_package(installed, state, editable_root=None, shadow=False)
            _commit_activation(state.index, home=home, env=env)
            published = None
            _cleanup_previous_refreshes(destination.parent)
            return installed
        except (
            DisciplineError,
            ManifestError,
            OSError,
            PackageInstallError,
            RecordError,
            ValueError,
        ) as exc:
            if published is not None:
                published_staging, previous = published
                try:
                    _rollback_managed_refresh(destination, published_staging, previous)
                except OSError as rollback_exc:
                    raise PackageInstallError(
                        f"cannot refresh managed std package and restore its previous tree: "
                        f"{rollback_exc}"
                    ) from rollback_exc
            raise PackageInstallError(f"cannot refresh managed std package: {exc}") from exc
        finally:
            if staging is not None and staging.exists():
                fs.rmtree(staging)


def install_archive(
    archive: Path,
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
    shadow: bool = False,
) -> PackageInfo:
    """Verify, atomically extract, and activate a portable package archive."""

    return install_archive_with_plan(archive, home=home, env=env, shadow=shadow).package


def install_archive_with_plan(
    archive: Path,
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
    shadow: bool = False,
) -> PackageInstallPlan:
    """Install a portable package archive and return its resulting activation plan."""

    with _package_operation_lock(home=home, env=env):
        state = _InstallState(home=home, env=env, index=_load_install_index(home=home, env=env))
        try:
            package = _install_archive(archive, state=state, shadow=shadow)
            state.index = _commit_activation(
                state.index,
                home=home,
                env=env,
                transient_packages=state.transient_packages if dry_run.enabled() else None,
            )
        except PackageInstallError:
            _rollback_created_trees(state)
            raise
        return PackageInstallPlan(package, state.index, dict(state.transient_packages))


def uninstall_package(name: str, *, home: Path, env: Mapping[str, str] | None = None) -> None:
    """Remove the active package tree by its verified ``RECORD``.

    Editable packages have no copied tree or record, so removal only drops
    their activation selection.
    """

    with _package_operation_lock(home=home, env=env):
        _uninstall_package(name, home=home, env=env)


def _uninstall_package(name: str, *, home: Path, env: Mapping[str, str] | None = None) -> None:
    try:
        index = load_activation_index(home=home, env=env)
    except PackageActivationError as exc:
        raise PackageInstallError(f"cannot load package activation: {exc}") from exc
    if name == "std":
        raise PackageInstallError("the AGM-managed std package cannot be uninstalled")
    placeholder_version = semver.Version(0, 0, 0)
    try:
        package_store_path(name, placeholder_version, home=home, env=env)
    except ValueError as exc:
        raise PackageInstallError(f"package {name!r} is not installed") from exc
    try:
        tombstone = (
            canonical_package_store_path(name, placeholder_version, home=home, env=env).parent
            / ".uninstalling"
        )
    except ValueError as exc:
        raise PackageInstallError(f"package store path is invalid for {name!r}: {exc}") from exc
    active = index.packages.get(name)
    if active is None:
        if tombstone.exists():
            _finish_uninstall(name, tombstone, home=home, env=env)
            return
        raise PackageInstallError(f"package {name!r} is not installed")
    root: Path | None = None
    entries: tuple[RecordEntry, ...] = ()
    if active.editable is None:
        try:
            root = canonical_package_store_path(name, active.version, home=home, env=env)
            if not root.exists() and tombstone.exists():
                entries = verify_record(tombstone)
                manifest = load_manifest(tombstone / "package.toml")
                if canonical_package_identity(manifest.name, manifest.version) != (
                    name,
                    str(active.version),
                ):
                    raise RecordError("uninstall tombstone identity does not match activation")
                tombstone.replace(root)
            else:
                entries = verify_record(root)
        except (ManifestError, OSError, RecordError, ValueError) as exc:
            raise PackageInstallError(
                f"package integrity check failed for {name!r}: {exc}"
            ) from exc
    packages = dict(index.packages)
    del packages[name]
    if root is None or dry_run.enabled():
        _commit_activation(ActivationIndex(packages), home=home, env=env)
        if root is not None:
            _remove_recorded_tree(root, entries)
            _remove_package_provenance(name, active.version, home=home, env=env)
        return

    if tombstone.exists():
        _finish_uninstall(name, tombstone, home=home, env=env)
    try:
        root.replace(tombstone)
    except OSError as exc:
        raise PackageInstallError(f"cannot remove package {name!r}: {exc}") from exc
    try:
        _commit_activation(ActivationIndex(packages), home=home, env=env)
    except PackageInstallError:
        tombstone.replace(root)
        raise
    _finish_uninstall(name, tombstone, version=active.version, home=home, env=env)


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
            if (
                not version_dir.is_dir()
                or version_dir.is_symlink()
                or version_dir.name.startswith(".")
            ):
                continue
            try:
                package = PackageInfo(version_dir, load_manifest(version_dir / "package.toml"))
            except ManifestError as exc:
                raise PackageInstallError(
                    f"cannot load installed package at {version_dir}: {exc}"
                ) from exc
            if canonical_package_identity(package.manifest.name, package.manifest.version) != (
                name_dir.name,
                version_dir.name,
            ):
                raise PackageInstallError(
                    f"installed package at {version_dir} does not match its store identity"
                )
            packages.append(package)
    return tuple(packages)


def _validated_directory_package(source: Path) -> PackageInfo:
    """Load and validate one package source directory."""

    if source.is_symlink():
        raise PackageInstallError(f"cannot install symbolic-link package root {source}")
    root = source.resolve()
    try:
        package = PackageInfo(root, load_manifest(root / "package.toml"))
        validate_package(package)
    except (ManifestError, DisciplineError) as exc:
        raise PackageInstallError(f"cannot install package from {source}: {exc}") from exc
    return package


def _stage_directory_package(source: Path, package: PackageInfo, destination: Path) -> Path:
    """Copy and fully validate a package in a sibling staging directory."""

    source_root = source.resolve()
    staging_parent = destination.parent.resolve()
    if staging_parent == source_root or staging_parent.is_relative_to(source_root):
        raise PackageInstallError(f"cannot stage package inside source directory {source_root}")
    fs.mkdir(staging_parent, parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=".agm-package-", dir=staging_parent))
    try:
        fs.copy_tree(source, staging, dirs_exist_ok=True)
        staged = PackageInfo(staging, load_manifest(staging / "package.toml"))
        validate_package(staged)
        if (
            canonical_package_identity(staged.manifest.name, staged.manifest.version)
            != canonical_package_identity(package.manifest.name, package.manifest.version)
            or staged.manifest != package.manifest
        ):
            raise PackageInstallError("copied package manifest changed after source validation")
        write_record(staging)
        verify_record(staging)
        return staging
    except (DisciplineError, ManifestError, OSError, PackageInstallError, RecordError, ValueError):
        fs.rmtree(staging)
        raise


def _publish_staged_refresh(staging: Path, destination: Path) -> Path | None:
    """Publish a complete replacement, restoring the old tree if publication fails."""

    previous: Path | None = None
    if destination.exists():
        previous = Path(mkdtemp(prefix=".agm-previous-", dir=destination.parent))
        fs.rmdir(previous)
        destination.replace(previous)
    try:
        staging.replace(destination)
    except OSError:
        if previous is not None:
            previous.replace(destination)
        raise
    return previous


def _cleanup_previous_refreshes(parent: Path) -> None:
    """Best-effort cleanup after a managed refresh has committed."""
    try:
        previous_paths = tuple(parent.glob(".agm-previous-*"))
    except OSError:
        return
    for previous in previous_paths:
        try:
            fs.rmtree(previous)
        except OSError:
            continue


def _rollback_managed_refresh(destination: Path, staging: Path, previous: Path | None) -> None:
    """Restore the tree displaced by a refresh whose activation failed."""

    if previous is None:
        fs.rmtree(destination)
        return
    destination.replace(staging)
    previous.replace(destination)


def _install_directory(
    source: Path, *, state: _InstallState, editable: bool, shadow: bool
) -> PackageInfo:
    package = _validated_directory_package(source)
    root = package.root

    _validate_managed_stdlib_install(package.manifest, source=root, editable=editable)
    _validate_minimum_agm(package.manifest)
    _resolve_dependencies(package, state)

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
            try:
                package_hash = content_hash(record_entries(root))
            except (OSError, RecordError) as exc:
                raise PackageInstallError(
                    f"cannot install package {package.manifest.name!r}: {exc}"
                ) from exc
            _verify_existing_install(destination, package.manifest, package_hash)
        elif not dry_run.enabled():
            staging: Path | None = None
            try:
                staging = _stage_directory_package(root, package, destination)
                staging.replace(destination)
                state.created.append(destination)
                staging = None
            except (
                DisciplineError,
                ManifestError,
                OSError,
                PackageInstallError,
                RecordError,
                ValueError,
            ) as exc:
                raise PackageInstallError(
                    f"cannot install package {package.manifest.name!r}: {exc}"
                ) from exc
            finally:
                if staging is not None and staging.exists():
                    fs.rmtree(staging)
        else:
            try:
                record_entries(root)
            except (OSError, RecordError) as exc:
                raise PackageInstallError(
                    f"cannot install package {package.manifest.name!r}: {exc}"
                ) from exc
            fs.mkdir(destination.parent, parents=True, exist_ok=True)
            fs.copy_tree(root, destination)

    _activate_package(installed, state, editable_root=root if editable else None, shadow=shadow)
    return installed


def _install_archive(archive: Path, *, state: _InstallState, shadow: bool) -> PackageInfo:
    """Extract an archive from one verified open ZIP stream and activate it."""

    archive_path = archive.resolve()
    if dry_run.enabled():
        try:
            metadata = verify_archive_discipline(archive_path)
            _validate_managed_stdlib_install(metadata.manifest, source=None, editable=False)
            _validate_minimum_agm(metadata.manifest)
            destination = canonical_package_store_path(
                metadata.manifest.name, metadata.manifest.version, home=state.home, env=state.env
            )
            if destination.exists():
                _verify_existing_install(destination, metadata.manifest, metadata.package_hash)
        except (ArchiveError, ValueError) as exc:
            raise PackageInstallError(f"cannot install package archive {archive}: {exc}") from exc
        dry_run.print_operation("install-package-archive", str(archive))
        installed = PackageInfo(destination, metadata.manifest)
    else:
        prepared: list[tuple[Path, Path]] = []

        def prepare_staging(metadata: ArchiveMetadata) -> Path:
            """Create staging beside the canonical destination once identity is verified."""

            _validate_managed_stdlib_install(metadata.manifest, source=None, editable=False)
            _validate_minimum_agm(metadata.manifest)
            destination = canonical_package_store_path(
                metadata.manifest.name, metadata.manifest.version, home=state.home, env=state.env
            )
            fs.mkdir(destination.parent, parents=True, exist_ok=True)
            staging = Path(mkdtemp(prefix=".agm-package-", dir=destination.parent))
            prepared.append((destination, staging))
            return staging

        try:
            # Verification, destination selection, and extraction share one
            # ZipFile instance, binding accepted data to the extracted tree.
            metadata = extract_archive(archive_path, prepare_staging)
            destination, staging = prepared[-1]
            package = PackageInfo(staging, metadata.manifest)
            validate_package(package)
            verify_record(staging)
            if destination.exists():
                _verify_existing_install(destination, metadata.manifest, metadata.package_hash)
            else:
                fs.mkdir(destination.parent, parents=True, exist_ok=True)
                staging.replace(destination)
                state.created.append(destination)
            installed = PackageInfo(destination, package.manifest)
        except PackageInstallError:
            raise
        except (ArchiveError, DisciplineError, RecordError, OSError, ValueError) as exc:
            raise PackageInstallError(f"cannot install package archive {archive}: {exc}") from exc
        finally:
            for _, staging in prepared:
                if staging.exists():
                    fs.rmtree(staging)

    try:
        _resolve_dependencies(installed, state)
        _activate_package(installed, state, editable_root=None, shadow=shadow)
        return installed
    except (DisciplineError, PackageInstallError, ValueError) as exc:
        raise PackageInstallError(f"cannot install package archive {archive}: {exc}") from exc


def _validate_minimum_agm(manifest: PackageManifest) -> None:
    """Reject packages whose ``std`` requirement needs a newer AGM binary."""
    requirement = manifest.dependencies.get("std")
    if requirement is None:
        return
    if requirement.version > semver.Version.parse(AGM_VERSION):
        raise PackageInstallError(
            f"package {manifest.name!r} requires AGM at least {requirement.version} via std, "
            f"but running AGM is {AGM_VERSION}"
        )


def _resolve_dependencies(package: PackageInfo, state: _InstallState) -> None:
    root = package.root.resolve()
    if root in state.installing:
        raise PackageInstallError(f"cyclic package dependency at {root}")
    state.installing.add(root)
    try:
        _resolve_dependency_requirements(package, state)
    finally:
        state.installing.remove(root)


def _resolve_dependency_requirements(package: PackageInfo, state: _InstallState) -> None:
    for name, requirement in package.manifest.dependencies.items():
        # ``std`` is a minimum AGM-version contract, already checked before
        # dependency resolution. It is not a package-store dependency.
        if name == "std":
            continue
        selected = _installed_satisfying(name, requirement, state)
        if selected is None and requirement.path is not None:
            selected = _install_directory(
                package.root / requirement.path, state=state, editable=False, shadow=False
            )
        if selected is None and requirement.url is not None and requirement.hash is not None:
            selected = _fetch_archive_install(
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
        _resolve_dependencies(selected, state)
        if selected.manifest.name != name or selected.manifest.version < requirement.version:
            raise PackageInstallError(
                f"unsatisfied package requirement {name!r} >= {requirement.version}"
            )
        current = state.index.packages.get(name)
        if current is not None and current.editable == selected.root:
            if canonical_package_identity(name, current.version) != canonical_package_identity(
                selected.manifest.name, selected.manifest.version
            ):
                packages = dict(state.index.packages)
                packages[name] = ActivePackage(
                    selected.manifest.version,
                    editable=current.editable,
                    shadow=current.shadow,
                    registration_order=current.registration_order,
                )
                state.index = ActivationIndex(packages, state.index.commands)
            state.transient_packages[name] = selected
        elif (
            current is None
            or canonical_package_identity(name, current.version)
            != canonical_package_identity(selected.manifest.name, selected.manifest.version)
            or current.editable is not None
        ):
            _activate_package(selected, state, editable_root=None, shadow=False)
        else:
            state.transient_packages[name] = selected


def _activate_package(
    package: PackageInfo,
    state: _InstallState,
    *,
    editable_root: Path | None,
    shadow: bool,
) -> None:
    """Select a package, register its commands, and retain it for this install."""

    try:
        validate_package_command_conflicts(
            state.index,
            package.manifest,
            shadow=shadow,
            home=state.home,
            env=state.env,
            transient_packages=state.transient_packages,
        )
        registration_order = _next_registration_order(state.index, home=state.home, env=state.env)
    except PackageActivationError as exc:
        raise PackageInstallError(f"cannot register package commands: {exc}") from exc
    packages = dict(state.index.packages)
    packages[package.manifest.name] = ActivePackage(
        package.manifest.version,
        editable=editable_root,
        shadow=shadow,
        registration_order=registration_order,
    )
    state.index = ActivationIndex(packages, state.index.commands)
    state.index = merge_package_commands(state.index, package.manifest, shadow=shadow)
    state.transient_packages[package.manifest.name] = package


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
        active = state.index.packages.get(name)
        if active is None or active.editable is None:
            return None
        try:
            selected = PackageInfo(active.editable, load_manifest(active.editable / "package.toml"))
        except ManifestError as exc:
            raise PackageInstallError(
                f"cannot load active editable package {name!r}: {exc}"
            ) from exc
        return selected if selected.manifest.version >= requirement.version else None
    selected, verify_selected = candidates[0]
    active = state.index.packages.get(name)
    for candidate, verify_candidate in candidates[1:]:
        if candidate.manifest.version > selected.manifest.version or (
            candidate.manifest.version == selected.manifest.version
            and active is not None
            and active.editable is None
            and str(candidate.manifest.version) == str(active.version)
        ):
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


def _fetch_archive_install(
    name: str, version: str, *, url: str, expected_hash: str, state: _InstallState
) -> PackageInfo:
    """Fetch a content-addressed archive and install it in the current transaction."""

    requirement = f"{name} >= {version}"
    if dry_run.enabled():
        dry_run.print_operation("fetch-package", url)
        raise PackageInstallError(
            f"cannot install URL package dependency during dry-run: {requirement}"
        )
    scratch = store_root(home=state.home, env=state.env)
    try:
        fs.mkdir(scratch, parents=True, exist_ok=True)
    except OSError as exc:
        error = FetchError(f"fetch failed for {requirement}: {exc}")
        raise PackageInstallError(str(error)) from error
    selected: PackageInfo | None = None

    def handoff(archive: Path) -> None:
        nonlocal selected
        selected = _install_archive(archive, state=state, shadow=False)

    try:
        fetch_archive(
            requirement=requirement,
            url=url,
            expected_hash=expected_hash,
            handoff=handoff,
            scratch_dir=scratch if scratch.is_dir() else None,
        )
    except (FetchError, PackageInstallError) as exc:
        raise PackageInstallError(str(exc)) from exc
    if selected is None:
        raise PackageInstallError(f"fetch failed for {requirement}: archive was not installed")
    return selected


def _verify_existing_install(
    root: Path, manifest: PackageManifest, package_hash: str | None = None
) -> None:
    try:
        installed = load_manifest(root / "package.toml")
        if (
            canonical_package_identity(installed.name, installed.version)
            != canonical_package_identity(manifest.name, manifest.version)
            or installed != manifest
        ):
            raise PackageInstallError(
                f"installed package at {root} disagrees with the source manifest"
            )
        entries = verify_record(root)
        if package_hash is not None and content_hash(entries) != package_hash:
            raise PackageInstallError(
                f"installed package at {root} conflicts with the package content hash"
            )
    except (ManifestError, RecordError) as exc:
        raise PackageInstallError(
            f"package integrity check failed for {manifest.name!r}: {exc}"
        ) from exc


def _load_install_index(*, home: Path, env: Mapping[str, str] | None) -> ActivationIndex:
    try:
        index = load_activation_index(home=home, env=env)
        for name, active in index.packages.items():
            if active.editable is None:
                load_package_provenance(name, active.version, home=home, env=env)
        return reconcile_package_commands(index, home=home, env=env)
    except PackageActivationError as exc:
        raise PackageInstallError(f"cannot load package activation: {exc}") from exc


def _next_registration_order(
    index: ActivationIndex, *, home: Path, env: Mapping[str, str] | None
) -> int:
    """Allocate after active and retained immutable registration provenance."""

    highest = max((package.registration_order for package in index.packages.values()), default=0)
    for package in installed_packages(home=home, env=env):
        provenance = load_package_provenance(
            package.manifest.name,
            package.manifest.version,
            home=home,
            env=env,
        )
        if provenance is not None:
            highest = max(highest, provenance.registration_order)
    return highest + 1


def _commit_activation(
    index: ActivationIndex,
    *,
    home: Path,
    env: Mapping[str, str] | None,
    transient_packages: Mapping[str, PackageInfo] | None = None,
) -> ActivationIndex:
    """Validate and atomically publish a complete activation selection."""

    try:
        reconciled = _assign_missing_registration_orders(
            reconcile_package_commands(
                index,
                home=home,
                env=env,
                transient_packages=transient_packages,
            )
        )
        validate_activation_index(
            reconciled,
            home=home,
            env=env,
            transient_packages=transient_packages,
        )
        provenance = _snapshot_package_provenance(reconciled, home=home, env=env)
        try:
            for name, active in reconciled.packages.items():
                if active.editable is None:
                    write_package_provenance(name, active, home=home, env=env)
            write_activation_index(reconciled, home=home, env=env)
        except PackageActivationError:
            _restore_package_provenance(provenance)
            raise
        return reconciled
    except PackageActivationError as exc:
        raise PackageInstallError(f"cannot write package activation: {exc}") from exc


def _snapshot_package_provenance(
    index: ActivationIndex, *, home: Path, env: Mapping[str, str] | None
) -> dict[Path, bytes | None]:
    """Capture immutable-package sidecars before publishing an activation."""

    try:
        return {
            path: path.read_bytes() if path.exists() else None
            for name, active in index.packages.items()
            if active.editable is None
            for path in (package_provenance_path(name, active.version, home=home, env=env),)
        }
    except OSError as exc:
        raise PackageActivationError(f"cannot snapshot package provenance: {exc}") from exc


def _restore_package_provenance(provenance: Mapping[Path, bytes | None]) -> None:
    """Restore sidecars changed by an activation that could not be published."""

    try:
        for path, content in provenance.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
    except OSError as exc:
        raise PackageActivationError(f"cannot restore package provenance: {exc}") from exc


def _assign_missing_registration_orders(index: ActivationIndex) -> ActivationIndex:
    """Give legacy selections without provenance unique durable priorities."""

    next_order = max((active.registration_order for active in index.packages.values()), default=0)
    packages: dict[str, ActivePackage] = {}
    for name, active in sorted(index.packages.items()):
        if active.registration_order == 0:
            next_order += 1
            packages[name] = ActivePackage(
                active.version,
                editable=active.editable,
                shadow=active.shadow,
                registration_order=next_order,
            )
        else:
            packages[name] = active
    return ActivationIndex(packages, dict(index.commands))


def _rollback_created_trees(state: _InstallState) -> None:
    """Discard immutable trees created by a failed install before they became active."""

    if dry_run.enabled():
        return
    for root in reversed(state.created):
        fs.rmtree(root)


def _finish_uninstall(
    name: str,
    tombstone: Path,
    *,
    home: Path,
    env: Mapping[str, str] | None,
    version: semver.Version | None = None,
) -> None:
    """Finish cleanup of a hidden immutable tree, including after an interrupted attempt."""

    if tombstone.is_symlink() or not tombstone.is_dir():
        raise PackageInstallError(f"cannot remove package {name!r}: invalid uninstall tombstone")
    if version is None and (tombstone / "package.toml").is_file():
        try:
            manifest = load_manifest(tombstone / "package.toml")
        except ManifestError as exc:
            raise PackageInstallError(f"cannot remove package {name!r}: {exc}") from exc
        version = manifest.version
    if version is not None:
        _remove_package_provenance(name, version, home=home, env=env)
    try:
        entries = read_record(tombstone) if (tombstone / "RECORD").exists() else ()
        _remove_recorded_tree(tombstone, entries)
    except (OSError, RecordError) as exc:
        raise PackageInstallError(f"cannot remove package {name!r}: {exc}") from exc


def _remove_package_provenance(
    name: str,
    version: semver.Version,
    *,
    home: Path,
    env: Mapping[str, str] | None,
) -> None:
    """Remove one immutable package's activation sidecar."""

    try:
        fs.unlink(package_provenance_path(name, version, home=home, env=env), missing_ok=True)
    except (OSError, PackageActivationError) as exc:
        raise PackageInstallError(f"cannot remove package provenance for {name!r}: {exc}") from exc


def _remove_recorded_tree(root: Path, entries: tuple[RecordEntry, ...]) -> None:
    """Idempotently remove a verified or partially removed immutable tree."""

    for entry in entries:
        fs.unlink(root / entry.path, missing_ok=True)
    fs.unlink(root / "RECORD", missing_ok=True)
    directories = sorted((path for path in fs.rglob(root, "*") if path.is_dir()), reverse=True)
    for directory in directories:
        fs.rmdir(directory)
    fs.rmdir(root)
