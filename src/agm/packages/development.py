"""Discovery of package roots from a development directory."""

from __future__ import annotations

from pathlib import Path

import semver

from agm.packages.manifest import load_manifest
from agm.packages.model import PackageInfo
from agm.packages.store import is_package_store_root


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

    package_root = _containing_package_root(anchor)
    if package_root is None:
        return ()
    if home is not None:
        manifest = load_manifest(package_root / "package.toml")
        if is_package_store_root(package_root, manifest.name, manifest.version, home=home):
            return ()

    packages: dict[Path, PackageInfo] = {}
    roots_by_name: dict[str, Path] = {}

    def add(
        root: Path, *, expected_name: str | None = None, minimum: semver.Version | None = None
    ) -> None:
        canonical_root = root.resolve()
        manifest = load_manifest(canonical_root / "package.toml")
        if expected_name is not None and manifest.name != expected_name:
            raise ValueError(
                f"development dependency {expected_name!r} resolves package {manifest.name!r}"
            )
        if minimum is not None and manifest.version < minimum:
            raise ValueError(
                f"development dependency {expected_name!r} requires at least {minimum}, "
                f"but path declares {manifest.version}"
            )
        if canonical_root in packages:
            return
        package = PackageInfo(canonical_root, manifest)
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
            dependency_root = (canonical_root / dependency.path).resolve()
            if (dependency_root / "package.toml").is_file():
                add(
                    dependency_root,
                    expected_name=dependency_name,
                    minimum=dependency.version,
                )

    add(package_root)
    return tuple(packages.values())


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
