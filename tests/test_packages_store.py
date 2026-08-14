"""Tests for the package-store directory layout."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import semver

from agm.packages.manifest import DependencySpec, ManifestError, load_manifest
from agm.packages.model import PackageInfo
from agm.packages.record import write_record
from agm.packages.store import (
    StoreIdentityError,
    StorePathError,
    canonical_package_provenance_path,
    canonical_package_store_path,
    is_package_store_root,
    iter_installed_packages,
    package_provenance_path,
    package_store_path,
    satisfying_from_store,
    store_root,
)


def _write_store_package(home: Path, name: str, version: str) -> Path:
    root = store_root(home=home, env={}) / name / version
    (root / name).mkdir(parents=True)
    (root / "package.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "{version}"\n', encoding="utf-8"
    )
    write_record(root)
    return root


def test_store_paths_use_the_agm_home_override(tmp_path: Path, env: dict[str, str]) -> None:
    agm_home = tmp_path / "isolated-agm-home"
    env["AGM_HOME"] = str(agm_home)

    assert store_root(home=Path(env["HOME"]), env=env) == agm_home / "packages"
    assert package_store_path(
        "review-tools", semver.Version.parse("1.2.3"), home=Path(env["HOME"]), env=env
    ) == (agm_home / "packages" / "review-tools" / "1.2.3")


def test_store_root_uses_the_installed_executable_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    index = prefix / ".agm" / "packages" / "index.toml"
    index.parent.mkdir(parents=True)
    index.write_text("")
    monkeypatch.setattr("agm.config.general.agm_installation_prefix", lambda: prefix)

    assert store_root(home=tmp_path / "home", env={}) == prefix / ".agm" / "packages"


def test_store_root_identity_supports_relocation_without_matching_editable_source(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    relocated = tmp_path / "relocated"
    immutable = relocated / "alpha" / "1.0.0"
    immutable.mkdir(parents=True)
    logical_store = home / ".agm" / "packages"
    logical_store.parent.mkdir(parents=True)
    logical_store.symlink_to(relocated, target_is_directory=True)
    version = semver.Version.parse("1.0.0")

    assert is_package_store_root(immutable, "alpha", version, home=home, env={})
    assert not is_package_store_root(
        tmp_path / "editable" / "alpha", "alpha", version, home=home, env={}
    )


def test_canonical_store_path_refuses_an_in_store_symlinked_tree_ancestor(tmp_path: Path) -> None:
    home = tmp_path / "home"
    store = home / ".agm" / "packages"
    internal_target = store / "other-package"
    internal_target.mkdir(parents=True)
    (store / "std").symlink_to(internal_target, target_is_directory=True)

    with pytest.raises(StorePathError, match="symlink"):
        canonical_package_store_path("std", semver.Version.parse("1.2.3"), home=home, env={})


def test_canonical_store_path_refuses_an_in_store_symlinked_tree_target(tmp_path: Path) -> None:
    home = tmp_path / "home"
    store = home / ".agm" / "packages"
    target = store / "other-package" / "1.2.3"
    target.mkdir(parents=True)
    logical_target = store / "std" / "1.2.3"
    logical_target.parent.mkdir()
    logical_target.symlink_to(target, target_is_directory=True)

    with pytest.raises(StorePathError, match="symlink"):
        canonical_package_store_path("std", semver.Version.parse("1.2.3"), home=home, env={})


def test_provenance_paths_do_not_collide_with_valid_package_versions(tmp_path: Path) -> None:
    home = tmp_path / "home"
    version = semver.Version.parse("1.0.0+alpha")
    colliding_version = semver.Version.parse("1.0.0+alpha.provenance.toml")

    assert package_provenance_path("alpha", version, home=home, env={}) != package_store_path(
        "alpha", colliding_version, home=home, env={}
    )


def test_canonical_provenance_path_refuses_a_symlinked_sidecar(tmp_path: Path) -> None:
    home = tmp_path / "home"
    store = home / ".agm" / "packages"
    logical_sidecar = store / "std" / ".provenance" / "1.2.3.toml"
    logical_sidecar.parent.mkdir(parents=True)
    target = store / "other-package" / "provenance.toml"
    target.parent.mkdir()
    target.write_text("", encoding="utf-8")
    logical_sidecar.symlink_to(target)

    with pytest.raises(StorePathError, match="symlink"):
        canonical_package_provenance_path("std", semver.Version.parse("1.2.3"), home=home, env={})


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "../outside", "name/version", r"name\version", "/outside", r"C:\outside"],
)
def test_package_store_path_rejects_names_that_could_escape_the_store(
    tmp_path: Path, env: dict[str, str], name: str
) -> None:
    with pytest.raises(StorePathError):
        package_store_path(name, semver.Version.parse("1.2.3"), home=Path(env["HOME"]), env=env)


@pytest.mark.parametrize("version", ["", ".", "..", "../outside", "1.2.3/extra", r"1.2.3\extra"])
def test_package_store_path_rejects_versions_that_could_escape_the_store(
    tmp_path: Path, env: dict[str, str], version: str
) -> None:
    with pytest.raises(StorePathError):
        package_store_path(
            "review-tools",
            cast(semver.Version, version),
            home=Path(env["HOME"]),
            env=env,
        )


def test_iter_installed_packages_yields_every_store_version_in_order(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_store_package(home, "alpha", "1.0.0")
    _write_store_package(home, "alpha", "2.0.0")
    _write_store_package(home, "bravo", "1.0.0")

    packages = list(iter_installed_packages(home=home, env={}))

    assert [(package.manifest.name, str(package.manifest.version)) for package in packages] == [
        ("alpha", "1.0.0"),
        ("alpha", "2.0.0"),
        ("bravo", "1.0.0"),
    ]


def test_iter_installed_packages_raises_manifest_error_without_a_translator(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    root = store_root(home=home, env={}) / "alpha" / "1.0.0"
    (root / "alpha").mkdir(parents=True)
    (root / "package.toml").write_text("", encoding="utf-8")

    with pytest.raises(ManifestError):
        list(iter_installed_packages(home=home, env={}))


def test_iter_installed_packages_translates_manifest_errors_when_given_a_translator(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    root = store_root(home=home, env={}) / "alpha" / "1.0.0"
    (root / "alpha").mkdir(parents=True)
    (root / "package.toml").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot load"):
        list(
            iter_installed_packages(
                home=home,
                env={},
                on_manifest_error=lambda exc, version_dir: ValueError(
                    f"cannot load {version_dir}: {exc}"
                ),
            )
        )


def test_iter_installed_packages_rejects_identity_mismatch(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_store_package(home, "alpha", "1.0.0")
    (store_root(home=home, env={}) / "alpha" / "1.0.0" / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "2.0.0"\n', encoding="utf-8"
    )

    with pytest.raises(StoreIdentityError, match="does not match its store identity"):
        list(iter_installed_packages(home=home, env={}))


def test_satisfying_from_store_selects_the_highest_version(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_store_package(home, "alpha", "1.0.0")
    _write_store_package(home, "alpha", "2.0.0")
    packages = list(iter_installed_packages(home=home, env={}))
    requirement = DependencySpec(semver.Version.parse("1.0.0"))

    selected = satisfying_from_store(packages, "alpha", requirement, None)

    assert selected is not None
    assert str(selected.manifest.version) == "2.0.0"


def test_satisfying_from_store_considers_candidates_outside_the_store(tmp_path: Path) -> None:
    """A dry-run install's planned tree is not in the store yet, so it must be
    offered as an extra candidate to be selectable at all."""
    home = tmp_path / "home"
    root = _write_store_package(home, "alpha", "1.0.0")
    planned = PackageInfo(root, load_manifest(root / "package.toml"))
    requirement = DependencySpec(semver.Version.parse("1.0.0"))

    selected = satisfying_from_store([], "alpha", requirement, None, extra_candidates=[planned])

    assert selected is planned
