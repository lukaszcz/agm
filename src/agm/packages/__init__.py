"""Pure domain types and validation for AGM packages."""

from __future__ import annotations

from agm.packages.activation import (
    ActivationIndex,
    ActivePackage,
    CommandRegistration,
    CommandShadow,
    PackageActivationError,
    PackageProvenance,
    command_shadow_diagnostics,
    load_activation_index,
    load_package_pins,
    load_package_provenance,
    merge_package_commands,
    rebuild_activation_index,
    select_active_packages,
    select_package_roots,
    validate_package_command_conflicts,
    write_activation_index,
    write_package_provenance,
)
from agm.packages.archive import (
    ArchiveError,
    ArchiveMetadata,
    extract_archive,
    read_archive_manifest,
    read_archive_metadata,
    verify_archive,
    write_archive,
)
from agm.packages.development import discover_development_packages
from agm.packages.discipline import DisciplineError, validate_package
from agm.packages.manifest import (
    ManifestError,
    PackageManifest,
    distribution_manifest,
    load_manifest,
)
from agm.packages.model import PackageInfo, owning_package
from agm.packages.record import RecordEntry, RecordError, read_record, verify_record, write_record
from agm.packages.store import StorePathError, package_store_path, store_root

__all__ = [
    "ActivationIndex",
    "ArchiveError",
    "ArchiveMetadata",
    "ActivePackage",
    "CommandRegistration",
    "CommandShadow",
    "DisciplineError",
    "PackageActivationError",
    "PackageProvenance",
    "discover_development_packages",
    "extract_archive",
    "distribution_manifest",
    "ManifestError",
    "PackageInfo",
    "PackageManifest",
    "RecordEntry",
    "RecordError",
    "StorePathError",
    "command_shadow_diagnostics",
    "load_activation_index",
    "load_package_provenance",
    "read_archive_manifest",
    "read_archive_metadata",
    "load_manifest",
    "load_package_pins",
    "package_store_path",
    "read_record",
    "merge_package_commands",
    "validate_package_command_conflicts",
    "rebuild_activation_index",
    "store_root",
    "select_active_packages",
    "select_package_roots",
    "owning_package",
    "validate_package",
    "verify_archive",
    "verify_record",
    "write_activation_index",
    "write_package_provenance",
    "write_archive",
    "write_record",
]
