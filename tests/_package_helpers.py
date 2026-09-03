"""Shared helpers for tests that need a real activated package on disk.

``write_installed_package`` lays out a minimal single-module package under an
AGM home's package store, writes its ``RECORD``, and activates it, so tests
that exercise installed-reference resolution can work against the same store
layout production code reads.

``older_incompatible_std_requirement`` and ``std_compatibility_bound`` derive
``std`` requirements and bounds from the running AGM version, so release-line
tests keep testing the rule rather than one release's literals.
"""

from __future__ import annotations

from pathlib import Path

import semver

from agm.packages.activation import ActivationIndex, ActivePackage, write_activation_index
from agm.packages.record import write_record
from agm.version import AGM_VERSION


def older_incompatible_std_requirement() -> str:
    """Return a ``std`` requirement one release line below the running AGM.

    ``std`` is compatible within one minor line before 1.0 and one major line
    from 1.0 on, so stepping the running version back a whole line yields a
    requirement the running AGM must reject as too old.
    """
    running = semver.Version.parse(AGM_VERSION)
    if running.major == 0:
        return str(semver.Version(0, running.minor - 1, 0))
    return str(semver.Version(running.major - 1, 0, 0))


def std_compatibility_bound(requirement: str) -> str:
    """Return the exclusive AGM bound a ``std`` *requirement* implies."""
    version = semver.Version.parse(requirement)
    if version.major == 0:
        return str(semver.Version(0, version.minor + 1, 0))
    return str(semver.Version(version.major + 1, 0, 0))


def write_installed_package(
    home: Path,
    name: str,
    *,
    source: str = "program def main(level: text) -> unit = ()\n",
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
