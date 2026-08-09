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
    """Return a provenance sidecar path only when it remains within the store."""

    canonical_store_root = store_root(home=home, env=env).resolve()
    candidate = package_provenance_path(name, version, home=home, env=env).resolve()
    if not candidate.is_relative_to(canonical_store_root):
        raise StorePathError(
            f"package provenance path resolves outside the store root: {candidate}"
        )
    return candidate


def canonical_package_store_path(
    name: str, version: semver.Version, *, home: Path, env: Mapping[str, str] | None = None
) -> Path:
    """Return an installed-tree path only when it resolves within the store.

    Resolving both paths catches a name or version ancestor that is a symbolic
    link to an external location before installation writes there or uninstall
    removes there.  The canonical store root may itself be relocated by a link.
    """

    canonical_store_root = store_root(home=home, env=env).resolve()
    candidate = package_store_path(name, version, home=home, env=env).resolve()
    if not candidate.is_relative_to(canonical_store_root):
        raise StorePathError(f"package store path resolves outside the store root: {candidate}")
    return candidate


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
