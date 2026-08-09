"""Discovery of package roots from a development directory."""

from __future__ import annotations

from pathlib import Path

from agm.packages.manifest import load_manifest
from agm.packages.model import PackageInfo


def discover_development_packages(anchor: Path) -> tuple[PackageInfo, ...]:
    """Return the containing package and its path-sourced dependency closure.

    A development package is selected only by an ancestor ``package.toml`` of
    the host's source directory. Its mounted dependencies are the manifests at
    explicitly declared relative ``[dependencies]`` ``path`` sources. This
    deliberately neither scans loose module roots nor treats a dependency key
    as package ownership; store activation can supply additional roots through
    the same ``package_roots`` seam later.
    """

    package_root = _containing_package_root(anchor)
    if package_root is None:
        return ()

    packages: dict[Path, PackageInfo] = {}

    def add(root: Path) -> None:
        canonical_root = root.resolve()
        if canonical_root in packages:
            return
        manifest = load_manifest(canonical_root / "package.toml")
        package = PackageInfo(canonical_root, manifest)
        packages[canonical_root] = package
        for dependency in manifest.dependencies.values():
            if dependency.path is None:
                continue
            dependency_root = (canonical_root / dependency.path).resolve()
            if (dependency_root / "package.toml").is_file():
                add(dependency_root)

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
