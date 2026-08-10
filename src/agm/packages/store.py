"""Paths for the versioned installed-package store."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PureWindowsPath

import semver

from agm.config.general import agm_home_dir


class StorePathError(ValueError):
    """Raised when a package identity cannot safely name a store directory."""


def store_root(*, home: Path, env: Mapping[str, str] | None = None) -> Path:
    """Return the package-store root under the selected AGM home."""

    return agm_home_dir(home=home, env=env) / "packages"


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

    The sidecar is a sibling of the immutable installed tree, keeping mutable
    activation history out of the payload covered by ``RECORD``.
    """

    package_path = package_store_path(name, version, home=home, env=env)
    return package_path.parent / f"{package_path.name}.provenance.toml"


def canonical_package_provenance_path(
    name: str, version: semver.Version, *, home: Path, env: Mapping[str, str] | None = None
) -> Path:
    """Return the canonical, unlinked provenance sidecar path."""

    logical_store_root = store_root(home=home, env=env)
    logical_package_path = package_store_path(name, version, home=home, env=env)
    _reject_managed_tree_symlinks(logical_package_path, logical_store_root)
    logical_candidate = package_provenance_path(name, version, home=home, env=env)
    if logical_candidate.is_symlink():
        raise StorePathError(f"package provenance path is a symlink: {logical_candidate}")
    return logical_candidate.resolve()


def canonical_package_store_path(
    name: str, version: semver.Version, *, home: Path, env: Mapping[str, str] | None = None
) -> Path:
    """Return an installed-tree path only when it is an unlinked store tree.

    The logical package-name directory and version target must not be symbolic
    links, even if resolving them would remain in the store. The store root
    itself may be relocated by a link.
    """

    logical_store_root = store_root(home=home, env=env)
    logical_candidate = package_store_path(name, version, home=home, env=env)
    _reject_managed_tree_symlinks(logical_candidate, logical_store_root)
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
