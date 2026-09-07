"""Paths for the versioned installed-package store."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path, PureWindowsPath

import semver

from agm.config.general import agm_home_dir
from agm.packages.errors import PackageInstallError
from agm.packages.layout import store_root_path
from agm.packages.manifest import DependencySpec, ManifestError, load_manifest
from agm.packages.model import (
    PackageInfo,
    VersionSelection,
    canonical_package_identity,
    select_satisfying,
)


class StorePathError(ValueError):
    """Raised when a package identity cannot safely name a store directory."""


class StoreIdentityError(ValueError):
    """Raised when an installed package version disagrees with its store location."""


def store_root(*, home: Path, env: Mapping[str, str] | None = None) -> Path:
    """Return the package-store root under the selected AGM home."""

    return store_root_path(agm_home_dir(home=home, env=env))


def package_store_path(
    name: str, version: semver.Version, *, home: Path, env: Mapping[str, str] | None = None
) -> Path:
    """Return the logical extracted-tree path for one installed package version."""

    return (
        store_root(home=home, env=env)
        / _store_component(name, "package name")
        / _store_component(str(version), "package version")
    )


def package_provenance_path(
    name: str, version: semver.Version, *, home: Path, env: Mapping[str, str] | None = None
) -> Path:
    """Return the package-local sidecar path for activation provenance.

    The sidecar lives in its own namespace below the package name, keeping
    mutable activation history out of the payload covered by ``RECORD`` and
    avoiding collisions with semantic-version directory names.
    """

    package_path = package_store_path(name, version, home=home, env=env)
    return package_path.parent / ".provenance" / f"{package_path.name}.toml"


def canonical_package_provenance_path(
    name: str, version: semver.Version, *, home: Path, env: Mapping[str, str] | None = None
) -> Path:
    """Return the canonical, unlinked provenance sidecar path."""

    return _canonical_managed_path(
        package_provenance_path(name, version, home=home, env=env), home=home, env=env
    )


def canonical_package_store_path(
    name: str, version: semver.Version, *, home: Path, env: Mapping[str, str] | None = None
) -> Path:
    """Return an installed-tree path only when it is an unlinked store tree.

    The logical package-name directory and version target must not be symbolic
    links, even if resolving them would remain in the store. The store root
    itself may be relocated by a link.
    """

    return _canonical_managed_path(
        package_store_path(name, version, home=home, env=env), home=home, env=env
    )


def iter_store_package_dirs(*, home: Path, env: Mapping[str, str] | None = None) -> Iterator[Path]:
    """Yield each package-name directory in the store, in name order.

    An absent store holds no packages, and a linked name directory does not
    designate a managed tree.
    """

    root = store_root(home=home, env=env)
    if not root.is_dir():
        return
    for name_dir in sorted(root.iterdir()):
        if name_dir.is_dir() and not name_dir.is_symlink():
            yield name_dir


def iter_store_version_dirs(name_dir: Path) -> Iterator[Path]:
    """Yield one package's installed version directories, in directory order.

    Dot-prefixed entries are the store's own bookkeeping — staging trees,
    uninstall tombstones, and provenance sidecars — not installed versions.
    """

    for version_dir in sorted(name_dir.iterdir()):
        if (
            version_dir.is_dir()
            and not version_dir.is_symlink()
            and not version_dir.name.startswith(".")
        ):
            yield version_dir


def iter_installed_packages(
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
    on_manifest_error: Callable[[ManifestError, Path], Exception] | None = None,
) -> Iterator[PackageInfo]:
    """Yield every installed package version, validated against its store identity.

    Iterates store name directories, then each package's version directories,
    loading and checking ``package.toml`` against the store's own naming.
    ``on_manifest_error`` lets a caller translate a manifest-loading failure
    into its own domain error, retaining the offending version directory for
    the message; without it, ``ManifestError`` propagates unchanged. A
    version whose manifest identity disagrees with its store location raises
    :class:`StoreIdentityError`.
    """

    for name_dir in iter_store_package_dirs(home=home, env=env):
        for version_dir in iter_store_version_dirs(name_dir):
            try:
                manifest = load_manifest(version_dir / "package.toml")
            except ManifestError as exc:
                if on_manifest_error is None:
                    raise
                raise on_manifest_error(exc, version_dir) from exc
            if canonical_package_identity(manifest.name, manifest.version) != (
                name_dir.name,
                version_dir.name,
            ):
                raise StoreIdentityError(
                    f"installed package at {version_dir} does not match its store identity"
                )
            yield PackageInfo(version_dir, manifest)


def satisfying_from_store(
    store_packages: Iterable[PackageInfo],
    name: str,
    requirement: DependencySpec,
    active: VersionSelection | None,
    *,
    extra_candidates: Iterable[PackageInfo] = (),
) -> PackageInfo | None:
    """Select the MVS-satisfying candidate for one dependency from store packages.

    Candidates are filtered from *store_packages* by name and minimum
    version, then MVS-selected against *active* (see
    :func:`agm.packages.model.select_satisfying`). *extra_candidates* adds
    packages that are not in the store yet, such as a dry-run install's
    planned trees.
    """

    def satisfies(package: PackageInfo) -> bool:
        return package.manifest.name == name and package.manifest.version >= requirement.version

    candidates = [package for package in (*store_packages, *extra_candidates) if satisfies(package)]
    return select_satisfying(candidates, active)


def satisfying_installed_package(
    store_packages: Iterable[PackageInfo],
    name: str,
    requirement: DependencySpec,
    active: VersionSelection | None,
    *,
    extra_candidates: Iterable[PackageInfo] = (),
) -> PackageInfo | None:
    """Select a satisfying stored package or the active editable source."""

    selected = satisfying_from_store(
        store_packages, name, requirement, active, extra_candidates=extra_candidates
    )
    if selected is not None or active is None or active.editable is None:
        return selected
    editable = PackageInfo(active.editable, load_manifest(active.editable / "package.toml"))
    return (
        editable
        if editable.manifest.name == name and editable.manifest.version >= requirement.version
        else None
    )


def is_package_store_root(
    root: Path,
    name: str,
    version: semver.Version,
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return whether *root* is the immutable store tree for an identity."""

    return root.resolve() == canonical_package_store_path(name, version, home=home, env=env)


def _canonical_managed_path(
    logical_candidate: Path, *, home: Path, env: Mapping[str, str] | None
) -> Path:
    """Resolve a logical store path once its managed ancestry is known link-free."""

    _reject_managed_tree_symlinks(logical_candidate, store_root(home=home, env=env))
    return logical_candidate.resolve()


def _reject_managed_tree_symlinks(logical_target: Path, logical_store_root: Path) -> None:
    """Reject links in a logical package tree while permitting a relocated store root."""

    candidate = logical_target
    while candidate != logical_store_root:
        if candidate.is_symlink():
            raise StorePathError(f"package store path contains a symlink: {candidate}")
        candidate = candidate.parent


def _store_component(value: str, label: str) -> str:
    """Validate one portable, non-traversing store path component."""

    windows_path = PureWindowsPath(value)
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or windows_path.drive
        or windows_path.root
    ):
        raise StorePathError(f"{label} must be one relative path component")
    return value


def installed_packages(
    *, home: Path, env: Mapping[str, str] | None = None
) -> tuple[PackageInfo, ...]:
    """Return all installed immutable package versions, sorted by identity."""

    try:
        return tuple(
            iter_installed_packages(
                home=home,
                env=env,
                on_manifest_error=lambda exc, version_dir: PackageInstallError(
                    f"cannot load installed package at {version_dir}: {exc}"
                ),
            )
        )
    except StoreIdentityError as exc:
        raise PackageInstallError(str(exc)) from exc
