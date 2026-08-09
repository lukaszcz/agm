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
    """Return the extracted-tree path for one installed package version."""

    return (
        store_root(home=home, env=env)
        / _store_component(name, "package name")
        / _store_component(str(version), "package version")
    )


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
