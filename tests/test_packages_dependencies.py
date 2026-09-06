"""Tests for non-mutating package dependency satisfiability checks."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import semver

import agm.packages.model as package_model
from agm.packages.dependencies import DependencyError, validate_dependencies
from agm.packages.install import install_directory
from agm.packages.layout import MODULE_TREE_DIRNAME
from agm.packages.manifest import DependencySpec, distribution_manifest, load_manifest
from agm.packages.model import PackageInfo
from agm.version import AGM_VERSION


def _package(root: Path, name: str, version: str, dependencies: str = "") -> PackageInfo:
    root.mkdir()
    (root / MODULE_TREE_DIRNAME).mkdir()
    (root / "package.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "{version}"\n' + dependencies,
        encoding="utf-8",
    )
    (root / MODULE_TREE_DIRNAME / "main.agl").write_text(
        "program def main() -> unit = ()\n", encoding="utf-8"
    )
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

    resolved = validate_dependencies(alpha, home=home, env={})

    assert [(package.manifest.name, str(package.manifest.version)) for package in resolved] == [
        ("bravo", "10.0.0")
    ]
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


@pytest.mark.parametrize(
    ("running", "required", "satisfied"),
    (
        ("0.2.0", "0.2.0", True),
        ("0.2.3", "0.2.1", True),
        ("0.2.0", "0.1.0", False),
        ("0.2.0", "0.2.1", False),
        ("0.2.0", "0.3.0", False),
        ("1.4.0", "1.2.0", True),
        ("1.4.0", "1.4.1", False),
        ("1.4.0", "1.5.0", False),
        ("2.0.0", "1.9.0", False),
    ),
)
def test_std_requirement_is_a_minimum_within_the_compatible_release_line(
    monkeypatch: pytest.MonkeyPatch, running: str, required: str, satisfied: bool
) -> None:
    monkeypatch.setattr(package_model, "AGM_VERSION", running)
    requirement = DependencySpec(semver.Version.parse(required))

    assert (package_model.unmet_std_requirement(requirement) is None) is satisfied


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


def test_dependency_check_validates_path_dependency_discipline(tmp_path: Path) -> None:
    bravo = _package(tmp_path / "bravo", "bravo", "1.0.0")
    shutil.rmtree(bravo.root / MODULE_TREE_DIRNAME)
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", path = "../bravo" }\n',
    )

    with pytest.raises(DependencyError):
        validate_dependencies(alpha, home=tmp_path / "home", env={})


@pytest.mark.parametrize(
    ("second_version", "expected_source"),
    (("1.0.0", "shared-one"), ("2.0.0", "shared-two")),
)
def test_dependency_check_selects_one_path_source_per_name_in_a_diamond(
    tmp_path: Path,
    second_version: str,
    expected_source: str,
) -> None:
    sources = {
        "shared-one": _package(tmp_path / "shared-one", "shared", "1.0.0"),
        "shared-two": _package(tmp_path / "shared-two", "shared", second_version),
    }
    _package(
        tmp_path / "left",
        "left",
        "1.0.0",
        '\n[dependencies]\nshared = { version = "1", path = "../shared-one" }\n',
    )
    _package(
        tmp_path / "right",
        "right",
        "1.0.0",
        f'\n[dependencies]\nshared = {{ version = "{second_version}", path = "../shared-two" }}\n',
    )
    package = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        "\n[dependencies]\n"
        'left = { version = "1", path = "../left" }\n'
        'right = { version = "1", path = "../right" }\n',
    )

    resolved = validate_dependencies(package, home=tmp_path / "home", env={})

    selected_shared = [
        dependency for dependency in resolved if dependency.manifest.name == "shared"
    ]
    assert [dependency.root for dependency in selected_shared] == [sources[expected_source].root]


def test_dependency_check_selects_the_highest_satisfying_store_version(tmp_path: Path) -> None:
    home = tmp_path / "home"
    install_directory(_package(tmp_path / "older", "bravo", "1.0.0").root, home=home, env={})
    newer = install_directory(
        _package(tmp_path / "newer", "bravo", "2.0.0").root, home=home, env={}
    )
    package = _package(tmp_path / "alpha", "alpha", "1.0.0", '\n[dependencies]\nbravo = "1"\n')

    resolved = validate_dependencies(package, home=home, env={})

    assert [dependency.root for dependency in resolved] == [newer.root]


def test_dependency_check_preserves_active_equal_precedence_build(tmp_path: Path) -> None:
    home = tmp_path / "home"
    inactive = [
        install_directory(
            _package(tmp_path / build, "bravo", f"1.0.0+{build}").root, home=home, env={}
        )
        for build in ("linux", "windows")
    ]
    active = install_directory(
        _package(tmp_path / "macos", "bravo", "1.0.0+macos").root, home=home, env={}
    )
    package = _package(tmp_path / "alpha", "alpha", "1.0.0", '\n[dependencies]\nbravo = "1"\n')
    for installed in inactive:
        (installed.root / MODULE_TREE_DIRNAME / "main.agl").write_text("tampered", encoding="utf-8")

    resolved = validate_dependencies(package, home=home, env={})

    assert [dependency.root for dependency in resolved] == [active.root]
    assert [str(dependency.manifest.version) for dependency in resolved] == ["1.0.0+macos"]


def test_dependency_check_wraps_an_invalid_package_store(tmp_path: Path) -> None:
    home = tmp_path / "home"
    broken = home / ".agm" / "packages" / "broken" / "1.0.0"
    broken.mkdir(parents=True)
    (broken / "package.toml").write_text("[package", encoding="utf-8")
    package = _package(tmp_path / "alpha", "alpha", "1.0.0", '\n[dependencies]\nbravo = "1"\n')

    with pytest.raises(DependencyError, match="cannot load"):
        validate_dependencies(package, home=home, env={})


def test_dependency_check_rejects_linked_store_sources(tmp_path: Path) -> None:
    home = tmp_path / "home"
    source = _package(tmp_path / "source", "bravo", "1.0.0")
    install_directory(source.root, home=home, env={})

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
