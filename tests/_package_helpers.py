"""Shared helpers for tests that need a real activated package on disk.

``write_installed_package`` lays out a minimal single-module package under an
AGM home's package store, writes its ``RECORD``, and activates it, so tests
that exercise installed-reference resolution can work against the same store
layout production code reads.
"""

from __future__ import annotations

from pathlib import Path

import semver

from agm.packages.activation import ActivationIndex, ActivePackage, write_activation_index
from agm.packages.record import write_record


def write_installed_package(
    home: Path, name: str, *, source: str = "param level: text\nprogram def main() -> unit = ()\n"
) -> Path:
    """Install and activate a one-module package named *name* under *home*.

    Returns the path of the package's ``main.agl`` module, so a test can edit
    or remove the entry file it will later resolve.
    """

    package_root = home / ".agm" / "packages" / name / "1.0.0"
    module = package_root / name / "main.agl"
    module.parent.mkdir(parents=True)
    (package_root / "package.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    module.write_text(source, encoding="utf-8")
    write_record(package_root)
    write_activation_index(
        ActivationIndex({name: ActivePackage(semver.Version.parse("1.0.0"))}), home=home
    )
    return module
