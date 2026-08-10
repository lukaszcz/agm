"""Tests for non-mutating package dependency satisfiability checks."""

from __future__ import annotations

from pathlib import Path

import pytest
import semver

from agm.packages.dependencies import DependencyError, validate_dependencies
from agm.packages.install import install_directory
from agm.packages.manifest import distribution_manifest, load_manifest
from agm.packages.model import PackageInfo
from agm.version import AGM_VERSION


def _package(root: Path, name: str, version: str, dependencies: str = "") -> PackageInfo:
    root.mkdir()
    (root / name).mkdir()
    (root / "package.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "{version}"\n' + dependencies,
        encoding="utf-8",
    )
    (root / name / "main.agl").write_text("program def main() -> unit = ()\n", encoding="utf-8")
    return PackageInfo(root, load_manifest(root / "package.toml"))


def test_dependency_check_resolves_store_then_path_and_accepts_url_sources(tmp_path: Path) -> None:
    home = tmp_path / "home"
    older = _package(tmp_path / "older", "bravo", "1.0.0")
    install_directory(older.root, home=home, env={})
    installed = _package(tmp_path / "installed", "bravo", "2.0.0")
    install_directory(installed.root, home=home, env={})
    newest = _package(tmp_path / "newest", "bravo", "10.0.0")
    install_directory(newest.root, home=home, env={})
    path_bravo = _package(tmp_path / "path-bravo", "bravo", "1.0.0")
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        "\n[dependencies]\n"
        'bravo = { version = "1", path = "../path-bravo" }\n'
        'charlie = { version = "1", url = "https://example.test/charlie.agmpkg", '
        'hash = "sha256=' + "0" * 64 + '" }\n',
    )

    validate_dependencies(alpha, home=home, env={})

    assert path_bravo.root.is_dir()


def test_distribution_dependency_check_does_not_treat_a_local_path_as_portable(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _package(tmp_path / "bravo", "bravo", "1.0.0")
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", path = "../bravo" }\n',
    )
    distribution = PackageInfo(alpha.root, distribution_manifest(alpha.manifest))

    validate_dependencies(alpha, home=home, env={})
    with pytest.raises(DependencyError, match="bravo"):
        validate_dependencies(distribution, home=home, env={})


def test_dependency_check_rejects_an_unsatisfied_store_requirement(tmp_path: Path) -> None:
    package = _package(tmp_path / "alpha", "alpha", "1.0.0", '\n[dependencies]\nbravo = "1"\n')

    with pytest.raises(DependencyError, match="bravo"):
        validate_dependencies(package, home=tmp_path / "home", env={})


def test_dependency_check_validates_std_against_the_running_agm(tmp_path: Path) -> None:
    package = _package(
        tmp_path / "alpha", "alpha", "1.0.0", f'\n[dependencies]\nstd = "{AGM_VERSION}"\n'
    )

    validate_dependencies(package, home=tmp_path / "home", env={})

    newer_agm = semver.Version.parse(AGM_VERSION).bump_major()
    package = _package(
        tmp_path / "newer-alpha", "newer_alpha", "1.0.0", f'\n[dependencies]\nstd = "{newer_agm}"\n'
    )

    with pytest.raises(DependencyError, match="AGM"):
        validate_dependencies(package, home=tmp_path / "home", env={})
    with pytest.raises(DependencyError, match="AGM"):
        validate_dependencies(
            PackageInfo(package.root, distribution_manifest(package.manifest)),
            home=tmp_path / "home",
            env={},
        )


def test_dependency_check_rejects_cyclic_missing_and_mismatched_path_sources(
    tmp_path: Path,
) -> None:
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", path = "../bravo" }\n',
    )
    _package(
        tmp_path / "bravo",
        "bravo",
        "1.0.0",
        '\n[dependencies]\nalpha = { version = "1", path = "../alpha" }\n',
    )

    with pytest.raises(DependencyError, match="cyclic"):
        validate_dependencies(alpha, home=tmp_path / "home", env={})

    missing = _package(
        tmp_path / "missing",
        "missing",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", path = "../not-there" }\n',
    )
    with pytest.raises(DependencyError, match="path dependency"):
        validate_dependencies(missing, home=tmp_path / "home", env={})

    wrong = _package(tmp_path / "wrong", "wrong", "1.0.0")
    mismatched = _package(
        tmp_path / "mismatched",
        "mismatched",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "2", path = "../wrong" }\n',
    )
    with pytest.raises(DependencyError, match="bravo"):
        validate_dependencies(mismatched, home=tmp_path / "home", env={})
    assert wrong.root.is_dir()


def test_dependency_check_selects_the_highest_satisfying_store_version(tmp_path: Path) -> None:
    home = tmp_path / "home"
    install_directory(_package(tmp_path / "older", "bravo", "1.0.0").root, home=home, env={})
    newer = install_directory(
        _package(tmp_path / "newer", "bravo", "2.0.0").root, home=home, env={}
    )
    package = _package(tmp_path / "alpha", "alpha", "1.0.0", '\n[dependencies]\nbravo = "1"\n')
    (newer.root / "bravo" / "main.agl").write_text("tampered", encoding="utf-8")

    with pytest.raises(DependencyError, match="integrity"):
        validate_dependencies(package, home=home, env={})


def test_dependency_check_wraps_an_invalid_package_store(tmp_path: Path) -> None:
    home = tmp_path / "home"
    broken = home / ".agm" / "packages" / "broken" / "1.0.0"
    broken.mkdir(parents=True)
    (broken / "package.toml").write_text("[package", encoding="utf-8")
    package = _package(tmp_path / "alpha", "alpha", "1.0.0", '\n[dependencies]\nbravo = "1"\n')

    with pytest.raises(DependencyError, match="cannot load"):
        validate_dependencies(package, home=home, env={})


def test_dependency_check_rejects_linked_or_tampered_store_sources(tmp_path: Path) -> None:
    home = tmp_path / "home"
    source = _package(tmp_path / "source", "bravo", "1.0.0")
    installed = install_directory(source.root, home=home, env={})
    package = _package(tmp_path / "alpha", "alpha", "1.0.0", '\n[dependencies]\nbravo = "1"\n')
    (installed.root / "bravo" / "main.agl").write_text("tampered", encoding="utf-8")

    with pytest.raises(DependencyError, match="integrity"):
        validate_dependencies(package, home=home, env={})

    linked = tmp_path / "linked"
    linked.symlink_to(source.root, target_is_directory=True)
    linked_package = _package(
        tmp_path / "linked-package",
        "linked_package",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", path = "../linked" }\n',
    )
    with pytest.raises(DependencyError, match="symbolic"):
        validate_dependencies(linked_package, home=tmp_path / "other-home", env={})
