"""Tests for the package-store directory layout."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import semver

from agm.packages.store import (
    StorePathError,
    canonical_package_provenance_path,
    canonical_package_store_path,
    package_provenance_path,
    package_store_path,
    store_root,
)


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
