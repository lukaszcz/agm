"""Discovery of package roots from a development directory.

A development tree is live source, so its manifest's command table is
incomplete until the commands its own programs register are merged in
(:mod:`agm.packages.source_commands`). Discovery loads manifests with
``commands_complete=False`` for that reason; it mounts module roots and
resolves dependencies, and never reads the command table.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import semver

from agm.packages.manifest import load_manifest
from agm.packages.model import PackageInfo
from agm.packages.store import is_package_store_root


def containing_development_package(
    anchor: Path, *, home: Path | None = None, env: Mapping[str, str] | None = None
) -> PackageInfo | None:
    """Return the nearest development package at or above *anchor*.

    ``None`` when no ancestor carries a manifest, or when the nearest one is
    an immutable store tree — a store entry is owned by active package
    selection rather than by development discovery.
    """

    package_root = _containing_package_root(anchor)
    if package_root is None:
        return None
    canonical_root = package_root.resolve()
    manifest = load_manifest(canonical_root / "package.toml", commands_complete=False)
    if home is not None and is_package_store_root(
        canonical_root, manifest.name, manifest.version, home=home, env=env
    ):
        return None
    return PackageInfo(canonical_root, manifest)


def discover_development_packages(
    anchor: Path, *, home: Path | None = None
) -> tuple[PackageInfo, ...]:
    """Return the containing development package and its path dependency closure.

    A development package is selected only by an ancestor ``package.toml`` of
    the host's source directory. Its mounted dependencies are the manifests at
    explicitly declared relative ``[dependencies]`` ``path`` sources. This
    deliberately does not scan loose module roots. Each path source must match
    its dependency key and minimum version; store activation can supply
    additional roots through the same ``package_roots`` seam later. When the
    host provides its home, an immutable store entry is left to active package
    selection rather than being reclassified as development source.
    """

    seed = containing_development_package(anchor, home=home)
    if seed is None:
        return ()
    packages: dict[Path, PackageInfo] = {}
    roots_by_name: dict[str, Path] = {}

    def add(package: PackageInfo) -> None:
        canonical_root = package.root
        manifest = package.manifest
        if home is not None and is_package_store_root(
            canonical_root, manifest.name, manifest.version, home=home
        ):
            return
        if canonical_root in packages:
            return
        previous_root = roots_by_name.setdefault(manifest.name, canonical_root)
        if previous_root != canonical_root:
            raise ValueError(
                f"development dependencies resolve package {manifest.name!r} "
                f"from both {previous_root} and {canonical_root}"
            )
        packages[canonical_root] = package
        for dependency_name, dependency in manifest.dependencies.items():
            if dependency.path is None:
                continue
            dependency_root = canonical_root / dependency.path
            if dependency_root.is_symlink():
                raise ValueError(f"cannot use symbolic-link package dependency {dependency_root}")
            dependency_root = dependency_root.resolve()
            if (dependency_root / "package.toml").is_file():
                add(_declared_dependency(dependency_root, dependency_name, dependency.version))

    add(seed)
    return tuple(packages.values())


def _declared_dependency(root: Path, name: str, minimum: semver.Version) -> PackageInfo:
    """Load the package at a declared path source, checking identity and floor."""

    manifest = load_manifest(root / "package.toml", commands_complete=False)
    if manifest.name != name:
        raise ValueError(f"development dependency {name!r} resolves package {manifest.name!r}")
    if manifest.version < minimum:
        raise ValueError(
            f"development dependency {name!r} requires at least {minimum}, "
            f"but path declares {manifest.version}"
        )
    return PackageInfo(root, manifest)


def _containing_package_root(anchor: Path) -> Path | None:
    """Find the nearest directory at or above *anchor* with a manifest."""

    directory = anchor.resolve()
    if not directory.is_dir():
        directory = directory.parent
    while True:
        if (directory / "package.toml").is_file():
            return directory
        if directory == directory.parent:
            return None
        directory = directory.parent
