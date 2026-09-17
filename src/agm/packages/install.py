"""Directory-package installation, dependency activation, and removal."""

from __future__ import annotations

import fcntl
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING

import semver

from agm.core import dry_run, fs
from agm.packages.activation import (
    ActivationIndex,
    ActivePackage,
    CommandShadow,
    PackageActivationError,
    command_shadow_diagnostics,
    load_activation_index,
    load_package_provenance,
    merge_package_commands,
    package_provenance_path,
    reconcile_package_commands,
    resolve_indexed_packages,
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
from agm.packages.distribution import (
    MANIFEST_NAME,
    DistributionError,
    distribution_entries,
    distribution_files,
    is_cache_or_vcs_path,
    materialize_distribution,
)
from agm.packages.errors import DisciplineError as DisciplineError
from agm.packages.errors import FetchError as FetchError
from agm.packages.errors import PackageInstallError as PackageInstallError
from agm.packages.manifest import (
    DependencySpec,
    ManifestError,
    PackageManifest,
    distribution_manifest,
    load_manifest,
)
from agm.packages.model import (
    STD_PACKAGE_NAME,
    PackageInfo,
    canonical_package_identity,
    is_std_package_name,
    unmet_std_requirement,
)
from agm.packages.record import (
    RECORD_NAME,
    RecordEntry,
    RecordError,
    content_hash,
    read_record,
    verify_record,
    write_record,
)
from agm.packages.store import (
    canonical_package_store_path,
    package_store_path,
    satisfying_installed_package,
    store_root,
)
from agm.packages.store import (
    installed_packages as installed_packages,
)
from agm.stdlib_locator import shipped_stdlib_root
from agm.version import AGM_VERSION

if TYPE_CHECKING:
    # Imported for typing only: agm.packages.discipline pulls in the AgL
    # frontend, which installation loads lazily at the call sites that need it.
    from agm.packages.discipline import PackageResolution


@dataclass(frozen=True, slots=True)
class PackageInstallPlan:
    """An installed package and the command shadows its activation produced."""

    package: PackageInfo
    command_shadows: tuple[CommandShadow, ...] = ()


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
    if not is_std_package_name(manifest.name):
        return
    if canonical_package_identity(manifest.name, manifest.version) != canonical_package_identity(
        STD_PACKAGE_NAME, semver.Version.parse(AGM_VERSION)
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
    resolved: set[Path] = field(default_factory=set)
    created: list[Path] = field(default_factory=list)
    transient_packages: dict[str, PackageInfo] = field(default_factory=dict)
    resource_packages: dict[str, PackageInfo] = field(default_factory=dict)
    installed: tuple[PackageInfo, ...] | None = None
    retained_registration_order: int | None = None
    resolved_active: tuple[PackageInfo, ...] | None = None


def install_directory_with_plan(
    source: Path,
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
    editable: bool = False,
    shadow: bool = False,
) -> PackageInstallPlan:
    """Install a package directory, or activate it as an editable package.

    Dependencies use MVS: an installed satisfying version is selected first;
    otherwise a declared local path is installed recursively or a URL archive
    is fetched, hash-verified, and installed. Returns the activation plan.
    """

    return _install_with_plan(
        lambda state: _install_directory(source, state=state, editable=editable, shadow=shadow),
        home=home,
        env=env,
        shadow=shadow,
    )


def _install_with_plan(
    install: Callable[[_InstallState], PackageInfo],
    *,
    home: Path,
    env: Mapping[str, str] | None,
    shadow: bool,
) -> PackageInstallPlan:
    """Run one locked install transaction, rolling back trees it created on failure."""

    with _package_operation_lock(home=home, env=env):
        state = _InstallState(home=home, env=env, index=_load_install_index(home=home, env=env))
        try:
            package = install(state)
            command_shadows = _commit_install_activation(state, package, report_shadows=shadow)
        except PackageInstallError:
            _rollback_created_trees(state)
            raise
        return PackageInstallPlan(package, command_shadows)


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
    from agm.packages.discipline import validate_package

    with _package_operation_lock(home=home, env=env):
        package = _validated_directory_package(source)
        _validate_managed_stdlib_install(package.manifest, source=package.root, editable=False)
        _validate_agm_compatibility(package.manifest)
        resolution = validate_package(package)
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
            staging = _stage_directory_package(
                package.root, package, destination, resolution=resolution
            )
            previous = _publish_staged_refresh(staging, destination)
            published = (staging, previous)
            state = _InstallState(
                home=home,
                env=env,
                index=_load_install_index(home=home, env=env),
            )
            _deactivate_release_incompatible_packages(state)
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


def install_archive_with_plan(
    archive: Path,
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
    shadow: bool = False,
) -> PackageInstallPlan:
    """Verify, atomically extract, and activate a portable package archive; return its plan."""

    return _install_with_plan(
        lambda state: _install_archive(archive, state=state, shadow=shadow),
        home=home,
        env=env,
        shadow=shadow,
    )


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
    if is_std_package_name(name):
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
                entries = read_record(tombstone)
                manifest = load_manifest(tombstone / "package.toml")
                if canonical_package_identity(manifest.name, manifest.version) != (
                    name,
                    str(active.version),
                ):
                    raise RecordError("uninstall tombstone identity does not match activation")
                if dry_run.enabled():
                    root = tombstone
                else:
                    tombstone.replace(root)
            else:
                entries = read_record(root)
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
    except PackageInstallError as activation_error:
        try:
            tombstone.replace(root)
        except OSError as rollback_error:
            raise PackageInstallError(
                f"cannot restore package {name!r} after activation failure: {rollback_error}"
            ) from activation_error
        raise
    _finish_uninstall(name, tombstone, version=active.version, home=home, env=env)


def _validated_directory_package(source: Path) -> PackageInfo:
    """Load and validate one package source directory.

    The returned manifest's commands already include the package's own
    source-declared registrations (:mod:`agm.packages.source_commands`), so
    every caller downstream — staging, hashing, activation — reads one
    complete command table.
    """
    from agm.packages.discipline import validate_package_structure
    from agm.packages.source_commands import package_with_source_commands

    if source.is_symlink():
        raise PackageInstallError(f"cannot install symbolic-link package root {source}")
    root = source.resolve()
    try:
        manifest = load_manifest(root / "package.toml", commands_complete=False)
        package = PackageInfo(root, manifest)
        validate_package_structure(package)
        package = package_with_source_commands(package)
    except (ManifestError, DisciplineError) as exc:
        raise PackageInstallError(f"cannot install package from {source}: {exc}") from exc
    return package


def _stage_directory_package(
    source: Path,
    package: PackageInfo,
    destination: Path,
    *,
    resolution: "PackageResolution",
) -> Path:
    """Stage and fully validate a package distribution in a sibling directory.

    The staged tree is the same distribution an archive of *source* would
    carry — its normalized manifest and its selected files — so a package has
    one stored shape and one content hash however it reaches the store. It is
    validated against *resolution*, the source tree's own, since staging copies
    that tree's modules unchanged.
    """
    from agm.packages.discipline import validate_staged_distribution

    source_root = source.resolve()
    staging_parent = destination.parent.resolve()
    if staging_parent == source_root or staging_parent.is_relative_to(source_root):
        raise PackageInstallError(f"cannot stage package inside source directory {source_root}")
    fs.mkdir(staging_parent, parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=".agm-package-", dir=staging_parent))
    try:
        distribution = distribution_manifest(package.manifest)
        materialize_distribution(source_root, distribution, staging)
        staged = PackageInfo(staging, load_manifest(staging / MANIFEST_NAME))
        validate_staged_distribution(resolution, staged)
        if (
            canonical_package_identity(staged.manifest.name, staged.manifest.version)
            != canonical_package_identity(package.manifest.name, package.manifest.version)
            or staged.manifest != distribution
        ):
            raise PackageInstallError("staged package manifest changed after source validation")
        write_record(staging)
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
    from agm.packages.discipline import validate_package, validate_package_distribution

    package = _validated_directory_package(source)
    root = package.root

    _validate_managed_stdlib_install(package.manifest, source=root, editable=editable)
    _validate_agm_compatibility(package.manifest)
    _resolve_dependencies(package, state)
    dependency_packages = tuple(state.resource_packages.values())
    try:
        resolution = validate_package(package, dependency_packages=dependency_packages)
    except DisciplineError as exc:
        raise PackageInstallError(f"cannot install package from {source}: {exc}") from exc

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
            # The store holds the distribution view of a package, so identity
            # is checked against what this source would store rather than
            # against its unfiltered development tree.
            distribution = distribution_manifest(package.manifest)
            try:
                package_hash = content_hash(distribution_entries(root, distribution))
            except (DistributionError, OSError, RecordError) as exc:
                raise PackageInstallError(
                    f"cannot install package {package.manifest.name!r}: {exc}"
                ) from exc
            _verify_existing_install(destination, distribution, package_hash)
        elif not dry_run.enabled():
            staging: Path | None = None
            try:
                staging = _stage_directory_package(
                    root,
                    package,
                    destination,
                    resolution=resolution,
                )
                staging.replace(destination)
                _record_created_tree(state, destination)
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
                distribution_files(root)
                validate_package_distribution(resolution)
            except (DisciplineError, DistributionError, OSError, RecordError) as exc:
                raise PackageInstallError(
                    f"cannot install package {package.manifest.name!r}: {exc}"
                ) from exc
            fs.mkdir(destination.parent, parents=True, exist_ok=True)
            fs.copy_tree(root, destination)

    # Dependents see what a package actually ships: the published store tree,
    # which carries only its distribution. An editable package is its own live
    # source, and a dry run publishes nothing, so both keep the source tree.
    state.resource_packages[package.manifest.name] = (
        package if editable or dry_run.enabled() else installed
    )
    _activate_package(installed, state, editable_root=root if editable else None, shadow=shadow)
    return installed


def _install_archive(archive: Path, *, state: _InstallState, shadow: bool) -> PackageInfo:
    """Extract an archive from one verified open ZIP stream and activate it."""
    from agm.packages.discipline import validate_package, validate_package_structure

    archive_path = archive.resolve()
    if dry_run.enabled():
        try:
            metadata = verify_archive_discipline(archive_path)
            _validate_managed_stdlib_install(metadata.manifest, source=None, editable=False)
            _validate_agm_compatibility(metadata.manifest)
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
            _validate_agm_compatibility(metadata.manifest)
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
            validate_package_structure(package)
            destination_exists = destination.exists()
            if destination_exists:
                _verify_existing_install(destination, metadata.manifest, metadata.package_hash)
            try:
                _resolve_dependencies(package, state)
                validate_package(package, dependency_packages=state.resource_packages.values())
            except (DisciplineError, PackageInstallError, ValueError) as exc:
                raise PackageInstallError(
                    f"cannot install package archive {archive}: {exc}"
                ) from exc
            if not destination_exists:
                fs.mkdir(destination.parent, parents=True, exist_ok=True)
                staging.replace(destination)
                _record_created_tree(state, destination)
            installed = PackageInfo(destination, package.manifest)
            state.resource_packages[installed.manifest.name] = installed
        except PackageInstallError:
            raise
        except (ArchiveError, DisciplineError, RecordError, OSError, ValueError) as exc:
            raise PackageInstallError(f"cannot install package archive {archive}: {exc}") from exc
        finally:
            for _, staging in prepared:
                if staging.exists():
                    fs.rmtree(staging)

    try:
        if dry_run.enabled():
            _resolve_dependencies(installed, state)
            revalidated = verify_archive_discipline(
                archive_path,
                dependency_packages=state.resource_packages.values(),
            )
            if revalidated != metadata:
                raise PackageInstallError(f"package archive changed while validating {archive}")
        _activate_package(installed, state, editable_root=None, shadow=shadow)
        return installed
    except (ArchiveError, DisciplineError, PackageInstallError, ValueError) as exc:
        raise PackageInstallError(f"cannot install package archive {archive}: {exc}") from exc


def _validate_agm_compatibility(manifest: PackageManifest) -> None:
    """Reject packages outside their ``std`` requirement's AGM release line."""
    requirement = manifest.dependencies.get(STD_PACKAGE_NAME)
    if requirement is None:
        return
    unmet = unmet_std_requirement(requirement)
    if unmet is not None:
        raise PackageInstallError(f"package {manifest.name!r} {unmet}")


def _transaction_installed_packages(state: _InstallState) -> tuple[PackageInfo, ...]:
    """Return the store contents, scanned once per transaction.

    Only this transaction publishes trees while it holds the store lock, so
    the scan is reused until it does.
    """

    if state.installed is None:
        state.installed = installed_packages(home=state.home, env=state.env)
    return state.installed


def _record_created_tree(state: _InstallState, destination: Path) -> None:
    """Track a published tree for rollback and invalidate the cached store scan."""

    state.created.append(destination)
    state.installed = None


def _transaction_resolved_packages(state: _InstallState) -> tuple[PackageInfo, ...]:
    """Return the transaction's active package set, resolved once and cached.

    Reused by every activation-conflict and command-shadow check within one
    locked install transaction until its activation index or transient
    selections change.
    """

    if state.resolved_active is None:
        state.resolved_active = resolve_indexed_packages(
            state.index,
            home=state.home,
            env=state.env,
            transient_packages=state.transient_packages,
        )
    return state.resolved_active


def _set_transaction_index(state: _InstallState, index: ActivationIndex) -> None:
    """Reassign the transaction's activation index, invalidating its resolved-package cache."""

    state.index = index
    state.resolved_active = None


def _deactivate_release_incompatible_packages(state: _InstallState) -> None:
    """Drop selections that cannot remain active after an AGM release-line change."""

    packages = _transaction_resolved_packages(state)
    deactivated = {
        package.manifest.name
        for package in packages
        if (requirement := package.manifest.dependencies.get(STD_PACKAGE_NAME)) is not None
        and unmet_std_requirement(requirement) is not None
    }
    while True:
        dependents = {
            package.manifest.name
            for package in packages
            if package.manifest.name not in deactivated
            and any(name in deactivated for name in package.manifest.dependencies)
        }
        if not dependents:
            break
        deactivated.update(dependents)
    if deactivated:
        _set_transaction_index(
            state,
            ActivationIndex(
                {
                    name: active
                    for name, active in state.index.packages.items()
                    if name not in deactivated
                }
            ),
        )


def _record_transient_package(state: _InstallState, name: str, package: PackageInfo) -> None:
    """Register a transient package selection, invalidating the resolved-package cache."""

    state.transient_packages[name] = package
    state.resolved_active = None


def _resolve_dependencies(package: PackageInfo, state: _InstallState) -> None:
    # Resolving a package is idempotent within a transaction, so a shared
    # dependency reached again through another path needs no second traversal.
    # Without this a diamond-shaped graph would be walked exponentially.
    root = package.root.resolve()
    if root in state.installing:
        raise PackageInstallError(f"cyclic package dependency at {root}")
    if root in state.resolved:
        return
    state.installing.add(root)
    try:
        _resolve_dependency_requirements(package, state)
    finally:
        state.installing.remove(root)
    state.resolved.add(root)


def _resolve_dependency_requirements(package: PackageInfo, state: _InstallState) -> None:
    for name, requirement in package.manifest.dependencies.items():
        # ``std`` is an AGM compatibility contract, already checked before
        # dependency resolution. It is not a package-store dependency.
        if is_std_package_name(name):
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
        state.resource_packages.setdefault(name, selected)
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
                _set_transaction_index(state, ActivationIndex(packages, state.index.commands))
            _record_transient_package(state, name, selected)
        elif (
            current is None
            or canonical_package_identity(name, current.version)
            != canonical_package_identity(selected.manifest.name, selected.manifest.version)
            or current.editable is not None
        ):
            _activate_package(selected, state, editable_root=None, shadow=False)
        else:
            _record_transient_package(state, name, selected)


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
            # ``shadow`` skips conflict resolution entirely, so resolving the
            # active set only when it is actually needed avoids forcing a
            # resolve/verify pass that its caller would otherwise skip.
            packages=None if shadow else _transaction_resolved_packages(state),
        )
        packages = dict(state.index.packages)
        packages[package.manifest.name] = ActivePackage(
            package.manifest.version,
            editable=editable_root,
            shadow=shadow,
            registration_order=_next_registration_order(state),
        )
        # Registering the commands can fail on its own, so the selection is
        # only adopted once the whole activation holds; the install then
        # reports the failure and rolls back the trees it created.
        merged = merge_package_commands(
            ActivationIndex(packages, state.index.commands), package.manifest, shadow=shadow
        )
    except PackageActivationError as exc:
        raise PackageInstallError(f"cannot register package commands: {exc}") from exc
    _set_transaction_index(state, merged)
    _record_transient_package(state, package.manifest.name, package)


def _installed_satisfying(
    name: str, requirement: DependencySpec, state: _InstallState
) -> PackageInfo | None:
    # A dry run never publishes, so planned installs stand in for store
    # trees; they were already validated when they were planned.
    planned = (
        [
            package
            for package in state.transient_packages.values()
            if package.manifest.name == name and package.manifest.version >= requirement.version
        ]
        if dry_run.enabled()
        else []
    )
    active = state.index.packages.get(name)
    try:
        return satisfying_installed_package(
            _transaction_installed_packages(state),
            name,
            requirement,
            active,
            extra_candidates=planned,
        )
    except ManifestError as exc:
        raise PackageInstallError(f"cannot load active editable package {name!r}: {exc}") from exc


def _fetch_archive_install(
    name: str, version: str, *, url: str, expected_hash: str, state: _InstallState
) -> PackageInfo:
    """Fetch a content-addressed archive and install it in the current transaction."""
    from agm.packages.fetch import fetch_archive

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


def _next_registration_order(state: _InstallState) -> int:
    """Allocate after active and retained immutable registration provenance.

    Sidecars are only rewritten when the transaction publishes, so the
    retained high-water mark is read once and reused for every activation.
    """

    if state.retained_registration_order is None:
        retained = 0
        for package in _transaction_installed_packages(state):
            provenance = load_package_provenance(
                package.manifest.name,
                package.manifest.version,
                home=state.home,
                env=state.env,
            )
            if provenance is not None:
                retained = max(retained, provenance.registration_order)
        state.retained_registration_order = retained
    active = max(
        (package.registration_order for package in state.index.packages.values()), default=0
    )
    return max(active, state.retained_registration_order) + 1


def _commit_install_activation(
    state: _InstallState, package: PackageInfo, *, report_shadows: bool
) -> tuple[CommandShadow, ...]:
    """Compute install diagnostics from one locked plan, then publish that plan."""

    # A real install has already published its trees, so activation is
    # (re)validated straight from the store; only a dry run stands its
    # transient plan in for the store, matching the cached resolution below.
    transient_packages = state.transient_packages if dry_run.enabled() else None
    prepare_packages = _transaction_resolved_packages(state) if dry_run.enabled() else None
    try:
        reconciled = _prepare_activation(
            state.index,
            home=state.home,
            env=state.env,
            transient_packages=transient_packages,
            packages=prepare_packages,
        )
        command_shadows = (
            command_shadow_diagnostics(
                reconciled,
                home=state.home,
                env=state.env,
                transient_packages=state.transient_packages,
                packages=_transaction_resolved_packages(state),
            ).get(package.manifest.name, ())
            if report_shadows
            else ()
        )
        _publish_activation(reconciled, home=state.home, env=state.env)
        return command_shadows
    except PackageActivationError as exc:
        raise PackageInstallError(f"cannot write package activation: {exc}") from exc


def _commit_activation(
    index: ActivationIndex,
    *,
    home: Path,
    env: Mapping[str, str] | None,
    transient_packages: Mapping[str, PackageInfo] | None = None,
) -> ActivationIndex:
    """Validate and atomically publish a complete activation selection."""

    try:
        reconciled = _prepare_activation(
            index,
            home=home,
            env=env,
            transient_packages=transient_packages,
        )
        _publish_activation(reconciled, home=home, env=env)
        return reconciled
    except PackageActivationError as exc:
        raise PackageInstallError(f"cannot write package activation: {exc}") from exc


def _prepare_activation(
    index: ActivationIndex,
    *,
    home: Path,
    env: Mapping[str, str] | None,
    transient_packages: Mapping[str, PackageInfo] | None,
    packages: tuple[PackageInfo, ...] | None = None,
) -> ActivationIndex:
    """Return the validated activation snapshot that is ready for publication.

    ``packages`` lets a caller that already resolved and verified *index*'s
    active package set under the same ``transient_packages`` reuse it.
    Reconciling commands and assigning registration orders never change
    which packages are selected, so the same resolution also validates the
    reconciled snapshot below without a second resolve.
    """

    resolved = (
        packages
        if packages is not None
        else resolve_indexed_packages(
            index, home=home, env=env, transient_packages=transient_packages
        )
    )
    reconciled = _assign_missing_registration_orders(
        reconcile_package_commands(
            index,
            home=home,
            env=env,
            transient_packages=transient_packages,
            packages=resolved,
        )
    )
    validate_activation_index(
        reconciled,
        home=home,
        env=env,
        transient_packages=transient_packages,
        packages=resolved,
    )
    return reconciled


def _publish_activation(
    index: ActivationIndex, *, home: Path, env: Mapping[str, str] | None
) -> None:
    """Atomically publish one already validated activation snapshot."""

    provenance = _snapshot_package_provenance(index, home=home, env=env)
    try:
        for name, active in index.packages.items():
            if active.editable is None:
                write_package_provenance(name, active, home=home, env=env)
        write_activation_index(index, home=home, env=env)
    except PackageActivationError:
        _restore_package_provenance(provenance)
        raise


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
    fs.unlink(root / RECORD_NAME, missing_ok=True)
    _remove_cache_residue(root)
    directories = sorted((path for path in fs.rglob(root, "*") if path.is_dir()), reverse=True)
    for directory in directories:
        fs.rmdir(directory)
    fs.rmdir(root)


def _remove_cache_residue(root: Path) -> None:
    """Remove tool-cache and VCS residue a package never records.

    A ``RECORD`` describes exactly the files an install created, so removal
    would otherwise strand a tree that acquired unrecorded content while it was
    installed. Only content no package distributes is cleared here; anything
    else still fails the directory sweep loudly rather than being deleted.
    """

    for path in fs.rglob(root, "*"):
        if not is_cache_or_vcs_path(path.relative_to(root).as_posix()):
            continue
        if path.is_dir() and not path.is_symlink():
            fs.rmtree(path)
        else:
            fs.unlink(path, missing_ok=True)
