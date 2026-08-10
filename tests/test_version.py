"""Tests for AGM's release-version contract."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from agm.packages.manifest import load_manifest
from agm.version import AGM_VERSION

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_agm_version_matches_project_metadata_and_plain_semver_shape() -> None:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]

    assert AGM_VERSION == project["version"]
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", AGM_VERSION)


def test_stdlib_manifest_version_matches_agm_version() -> None:
    manifest = load_manifest(_REPO_ROOT / "stdlib" / "package.toml")

    assert manifest.name == "std"
    assert str(manifest.version) == AGM_VERSION
