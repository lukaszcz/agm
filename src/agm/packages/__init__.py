"""Pure domain types and validation for AGM packages."""

from __future__ import annotations

from agm.packages.discipline import DisciplineError, validate_package
from agm.packages.manifest import ManifestError, PackageManifest, load_manifest
from agm.packages.model import PackageInfo, owning_package

__all__ = [
    "DisciplineError",
    "ManifestError",
    "PackageInfo",
    "PackageManifest",
    "load_manifest",
    "owning_package",
    "validate_package",
]
