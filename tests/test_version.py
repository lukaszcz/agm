"""Tests for AGM's release-version contract."""

from __future__ import annotations

import re
from importlib.metadata import version

from click.testing import CliRunner
from typer.main import get_command

import agm.cli as cli
from agm.packages.manifest import load_manifest
from agm.stdlib_locator import shipped_stdlib_root
from agm.version import AGM_VERSION


def test_agm_version_has_plain_semver_shape() -> None:
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", AGM_VERSION)
    assert AGM_VERSION == "0.1.2"


def test_release_metadata_matches_runtime_version() -> None:
    assert version("agm") == AGM_VERSION


def test_shipped_standard_library_matches_runtime_version() -> None:
    manifest = load_manifest(shipped_stdlib_root() / "package.toml")

    assert str(manifest.version) == AGM_VERSION


def test_bare_version_option_displays_agm_and_agl_stdlib_versions() -> None:
    manifest = load_manifest(shipped_stdlib_root() / "package.toml")

    result = CliRunner().invoke(get_command(cli.app), ["--version"], prog_name="agm")

    assert result.exit_code == 0
    assert result.stdout == (
        f"AGM version: {AGM_VERSION}\nAgL stdlib version: {manifest.version}\n"
    )
