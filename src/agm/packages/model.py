"""Package identity and canonical module-file ownership."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agm.packages.manifest import PackageManifest


@dataclass(frozen=True, slots=True)
class PackageInfo:
    """A manifest paired with its canonical package-root directory."""

    root: Path
    manifest: PackageManifest

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.resolve())

    @property
    def module_root(self) -> Path:
        """Return the package's module-tree directory."""

        return (self.root / self.manifest.name).resolve()


def owning_package(path: Path, packages: tuple[PackageInfo, ...]) -> PackageInfo | None:
    """Return the package whose module tree canonically contains *path*.

    Assets and manifests are deliberately not owned by this map: ownership is
    the module-loader policy seam, so only files under a module tree qualify.
    """

    canonical_path = path.resolve()
    for package in packages:
        try:
            canonical_path.relative_to(package.module_root)
        except ValueError:
            continue
        return package
    return None
