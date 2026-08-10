"""Tests for development-directory package discovery."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agm.packages.development import discover_development_packages


def _write_package(root: Path, name: str, dependencies: str = "") -> None:
    root.mkdir()
    (root / "package.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "1.0.0"\n' + dependencies
    )
    (root / name).mkdir()


def test_package_domain_imports_in_a_fresh_process_before_the_agl_facade() -> None:
    script = """
import sys
import agm.packages
from agm.packages.development import discover_development_packages
from agm.packages.manifest import load_manifest
from agm.packages.model import PackageInfo
assert callable(discover_development_packages)
assert callable(load_manifest)
assert PackageInfo.__name__ == "PackageInfo"
assert "agm.agl.pipeline" not in sys.modules
"""

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_agl_facade_lazily_exposes_its_public_api() -> None:
    import agm.agl as agl

    for name in agl.__all__:
        assert getattr(agl, name) is not None
    with pytest.raises(AttributeError):
        getattr(agl, "missing")


@pytest.mark.parametrize(
    "module",
    ("agm.agl.pipeline", "agm.agl.runtime.agents"),
)
def test_agl_modules_import_in_a_fresh_process_before_the_package_facade(module: str) -> None:
    script = f"""
import {module}
"""

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_package_facade_lazily_exposes_its_public_api() -> None:
    import agm.packages as packages

    assert packages.PackageInfo.__name__ == "PackageInfo"
    assert callable(packages.validate_package)
    with pytest.raises(AttributeError):
        getattr(packages, "missing")


def test_discovery_ignores_a_directory_without_a_containing_manifest(tmp_path: Path) -> None:
    assert discover_development_packages(tmp_path) == ()


def test_discovery_uses_path_dependencies_recursively_and_ignores_unavailable_sources(
    tmp_path: Path,
) -> None:
    alpha = tmp_path / "alpha"
    bravo = tmp_path / "bravo"
    _write_package(
        alpha,
        "alpha",
        "\n[dependencies]\n"
        'bravo = { version = "1", path = "../bravo" }\n'
        'missing = { version = "1", path = "../missing" }\n'
        'registry = "1"\n',
    )
    _write_package(
        bravo,
        "bravo",
        "\n[dependencies]\nalpha = { version = " + '"1", path = "../alpha" }\n',
    )

    packages = discover_development_packages(alpha / "alpha" / "main.agl")

    assert tuple(package.manifest.name for package in packages) == ("alpha", "bravo")


def test_discovery_rejects_different_roots_with_the_same_package_identity(tmp_path: Path) -> None:
    alpha = tmp_path / "alpha"
    left = tmp_path / "left"
    right = tmp_path / "right"
    shared_one = tmp_path / "shared-one"
    shared_two = tmp_path / "shared-two"
    _write_package(
        alpha,
        "alpha",
        '\n[dependencies]\nleft = { version = "1", path = "../left" }\n'
        'right = { version = "1", path = "../right" }\n',
    )
    _write_package(
        left,
        "left",
        '\n[dependencies]\nshared = { version = "1", path = "../shared-one" }\n',
    )
    _write_package(
        right,
        "right",
        '\n[dependencies]\nshared = { version = "1", path = "../shared-two" }\n',
    )
    _write_package(shared_one, "shared")
    _write_package(shared_two, "shared")

    with pytest.raises(ValueError, match="shared"):
        discover_development_packages(alpha / "alpha" / "main.agl")
