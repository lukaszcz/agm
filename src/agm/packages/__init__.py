"""Pure domain types and validation for AGM packages."""

from __future__ import annotations

from agm.packages.development import discover_development_packages
from agm.packages.discipline import DisciplineError, validate_package
from agm.packages.manifest import ManifestError, PackageManifest, load_manifest
from agm.packages.model import PackageInfo, owning_package
from agm.packages.record import RecordEntry, RecordError, read_record, verify_record, write_record
from agm.packages.store import StorePathError, package_store_path, store_root

__all__ = [
    "DisciplineError",
    "discover_development_packages",
    "ManifestError",
    "PackageInfo",
    "PackageManifest",
    "RecordEntry",
    "RecordError",
    "StorePathError",
    "load_manifest",
    "package_store_path",
    "read_record",
    "store_root",
    "owning_package",
    "validate_package",
    "verify_record",
    "write_record",
]
