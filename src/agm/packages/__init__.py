"""Lazy public façade for AGM package-domain types and operations."""

from __future__ import annotations

from importlib import import_module
from typing import Final, cast

_EXPORT_MODULES: Final = {
    "ActivationIndex": "activation",
    "ActivePackage": "activation",
    "ArchiveError": "archive",
    "ArchiveMetadata": "archive",
    "CommandRegistration": "activation",
    "CommandShadow": "activation",
    "DisciplineError": "discipline",
    "PackageActivationError": "activation",
    "PackageInfo": "model",
    "PackageManifest": "manifest",
    "PackageProvenance": "activation",
    "ManifestError": "manifest",
    "RecordEntry": "record",
    "RecordError": "record",
    "StorePathError": "store",
    "command_shadow_diagnostics": "activation",
    "discover_development_packages": "development",
    "distribution_manifest": "manifest",
    "extract_archive": "archive",
    "load_activation_index": "activation",
    "load_manifest": "manifest",
    "load_package_pins": "activation",
    "load_package_provenance": "activation",
    "merge_package_commands": "activation",
    "owning_package": "model",
    "package_store_path": "store",
    "read_archive_manifest": "archive",
    "read_archive_metadata": "archive",
    "read_record": "record",
    "rebuild_activation_index": "activation",
    "select_active_packages": "activation",
    "select_package_roots": "activation",
    "store_root": "store",
    "validate_package": "discipline",
    "validate_package_command_conflicts": "activation",
    "verify_archive": "archive",
    "verify_record": "record",
    "write_activation_index": "activation",
    "write_archive": "archive",
    "write_package_provenance": "activation",
    "write_record": "record",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> object:
    """Load a package-domain export only when a caller requests it."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return cast(object, getattr(import_module(f"{__name__}.{module_name}"), name))
