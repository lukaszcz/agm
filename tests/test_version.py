"""Tests for AGM's release-version contract."""

from __future__ import annotations

import re
from importlib.metadata import version

from agm.packages.manifest import load_manifest
from agm.stdlib_locator import shipped_stdlib_root
from agm.version import AGM_VERSION


def test_agm_version_has_plain_semver_shape() -> None:
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", AGM_VERSION)
    assert AGM_VERSION == "0.2.0"


def test_release_metadata_matches_runtime_version() -> None:
    assert version("agm") == AGM_VERSION


def test_shipped_standard_library_matches_runtime_version() -> None:
    manifest = load_manifest(shipped_stdlib_root() / "package.toml")

    assert str(manifest.version) == AGM_VERSION
