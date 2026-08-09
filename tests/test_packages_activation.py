"""Tests for installed-package activation, pins, and root selection."""

from __future__ import annotations

from pathlib import Path

import pytest
import semver

from agm.cli_support.exec_roots import effective_exec_roots
from agm.packages.activation import (
    ActivationIndex,
    ActivePackage,
    PackageActivationError,
    activation_index_path,
    load_activation_index,
    load_package_pins,
    rebuild_activation_index,
    select_active_packages,
    select_package_roots,
    write_activation_index,
)
from agm.packages.manifest import load_manifest
from agm.packages.model import PackageInfo


def _write_package(
    home: Path,
    name: str,
    version: str,
    dependencies: str = "",
) -> Path:
    root = home / "packages" / name / version
    (root / name).mkdir(parents=True)
    (root / "package.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "{version}"\n' + dependencies,
        encoding="utf-8",
    )
    return root


def _write_development_package(
    root: Path, name: str, version: str, dependencies: str = ""
) -> PackageInfo:
    (root / name).mkdir(parents=True)
    (root / "package.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "{version}"\n' + dependencies,
        encoding="utf-8",
    )
    return PackageInfo(root, load_manifest(root / "package.toml"))


def test_activation_index_round_trips_all_activation_markers(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    editable = tmp_path / "editable"
    index = ActivationIndex(
        packages={
            "alpha": ActivePackage(semver.Version.parse("1.2.3"), shadow=True),
            "bravo": ActivePackage(semver.Version.parse("2.0.0"), editable=editable),
        }
    )

    env = {"AGM_HOME": str(home)}
    write_activation_index(index, home=home, env=env)

    assert activation_index_path(home=home, env=env) == home / "packages" / "index.toml"
    assert load_activation_index(home=home, env=env) == index


def test_rebuild_index_selects_latest_installed_manifest_per_package(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    _write_package(home, "alpha", "1.0.0")
    _write_package(home, "alpha", "2.0.0")
    _write_package(home, "alpha", "2.0.0-alpha")
    _write_package(home, "alpha", "1.5.0")
    _write_package(home, "bravo", "1.5.0")

    env = {"AGM_HOME": str(home)}
    rebuilt = rebuild_activation_index(home=home, env=env)
    write_activation_index(rebuilt, home=home, env=env)

    assert rebuilt == ActivationIndex(
        packages={
            "alpha": ActivePackage(semver.Version.parse("2.0.0")),
            "bravo": ActivePackage(semver.Version.parse("1.5.0")),
        }
    )
    assert load_activation_index(home=home, env=env) == rebuilt


def test_project_pin_overrides_the_global_active_version(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    one = _write_package(home, "alpha", "1.0.0")
    two = _write_package(home, "alpha", "2.0.0")
    write_activation_index(
        ActivationIndex(packages={"alpha": ActivePackage(semver.Version.parse("1.0.0"))}),
        home=home,
        env={"AGM_HOME": str(home)},
    )
    project = tmp_path / "project"
    config = project / "config"
    config.mkdir(parents=True)
    (config / "config.toml").write_text('[packages]\nalpha = "2.0.0"\n', encoding="utf-8")

    packages = select_active_packages(
        home=home, proj_dir=project, cwd=project, env={"AGM_HOME": str(home)}
    )

    assert tuple(package.root for package in packages) == (two.resolve(),)
    assert one.resolve() not in tuple(package.root for package in packages)


def test_pinned_selection_checks_package_requirements_loudly(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    _write_package(home, "bravo", "1.0.0")
    _write_package(
        home,
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = "2.0"\n',
    )
    write_activation_index(
        ActivationIndex(packages={"alpha": ActivePackage(semver.Version.parse("1.0.0"))}),
        home=home,
        env={"AGM_HOME": str(home)},
    )
    project = tmp_path / "project"
    config = project / "config"
    config.mkdir(parents=True)
    (config / "config.toml").write_text('[packages]\nbravo = "1.0.0"\n', encoding="utf-8")

    with pytest.raises(PackageActivationError, match="bravo"):
        select_active_packages(
            home=home, proj_dir=project, cwd=project, env={"AGM_HOME": str(home)}
        )


def test_unrelated_config_sections_do_not_prevent_pin_loading(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    selected = _write_package(home, "alpha", "1.0.0")
    project = tmp_path / "project"
    config = project / "config"
    config.mkdir(parents=True)
    (config / "config.toml").write_text(
        '[packages]\nalpha = "1.0.0"\n\n[unrelated]\nlog-file = "%{MISSING}"\n',
        encoding="utf-8",
    )

    packages = select_active_packages(
        home=home, proj_dir=project, cwd=project, env={"AGM_HOME": str(home)}
    )

    assert tuple(package.root for package in packages) == (selected.resolve(),)


def test_development_package_shadows_an_active_package_with_the_same_name(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    store_root = _write_package(home, "alpha", "1.0.0")
    development_root = tmp_path / "development"
    (development_root / "alpha").mkdir(parents=True)
    (development_root / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "2.0.0"\n', encoding="utf-8"
    )
    from agm.packages.manifest import load_manifest
    from agm.packages.model import PackageInfo

    development = PackageInfo(development_root, load_manifest(development_root / "package.toml"))
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex(packages={"alpha": ActivePackage(semver.Version.parse("1.0.0"))}),
        home=home,
        env=env,
    )

    packages = select_package_roots(
        home=home,
        proj_dir=None,
        cwd=tmp_path,
        development_packages=(development,),
        env=env,
    )

    assert tuple(package.root for package in packages) == (development_root.resolve(),)
    assert store_root.resolve() not in tuple(package.root for package in packages)


@pytest.mark.parametrize("selection", ("active", "pin"))
@pytest.mark.parametrize("store_state", ("missing", "mismatched"))
def test_development_package_shadows_stale_or_missing_active_or_pinned_store_selection(
    tmp_path: Path, selection: str, store_state: str
) -> None:
    home = tmp_path / "agm-home"
    if store_state == "mismatched":
        stale_root = _write_package(home, "alpha", "1.0.0")
        (stale_root / "package.toml").write_text(
            '[package]\nname = "alpha"\nversion = "2.0.0"\n', encoding="utf-8"
        )
    development = _write_development_package(tmp_path / "development", "alpha", "2.0.0")
    env = {"AGM_HOME": str(home)}
    project = tmp_path / "project"
    if selection == "active":
        write_activation_index(
            ActivationIndex(packages={"alpha": ActivePackage(semver.Version.parse("1.0.0"))}),
            home=home,
            env=env,
        )
        proj_dir = None
    else:
        config = project / "config"
        config.mkdir(parents=True)
        (config / "config.toml").write_text('[packages]\nalpha = "1.0.0"\n', encoding="utf-8")
        proj_dir = project

    packages = select_package_roots(
        home=home,
        proj_dir=proj_dir,
        cwd=project,
        development_packages=(development,),
        env=env,
    )

    assert packages == (development,)


def test_final_root_selection_validates_requirements_after_shadowing_missing_selection(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agm-home"
    _write_package(home, "bravo", "1.0.0")
    development = _write_development_package(
        tmp_path / "development",
        "alpha",
        "2.0.0",
        '\n[dependencies]\nbravo = "2.0"\n',
    )
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex(
            packages={
                "alpha": ActivePackage(semver.Version.parse("1.0.0")),
                "bravo": ActivePackage(semver.Version.parse("1.0.0")),
            }
        ),
        home=home,
        env=env,
    )

    with pytest.raises(PackageActivationError, match="bravo"):
        select_package_roots(
            home=home,
            proj_dir=None,
            cwd=tmp_path,
            development_packages=(development,),
            env=env,
        )


def test_final_root_selection_validates_development_package_requirements(tmp_path: Path) -> None:
    development = _write_development_package(
        tmp_path / "development",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = "1.0"\n',
    )

    with pytest.raises(PackageActivationError, match="bravo"):
        select_package_roots(
            home=tmp_path / "home",
            proj_dir=None,
            cwd=tmp_path,
            development_packages=(development,),
            env={},
        )


def test_final_root_selection_uses_development_package_for_active_requirements(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agm-home"
    _write_package(home, "alpha", "1.0.0", '\n[dependencies]\nbravo = "2.0"\n')
    _write_package(home, "bravo", "1.0.0")
    development = _write_development_package(tmp_path / "development", "bravo", "2.0.0")
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex(
            packages={
                "alpha": ActivePackage(semver.Version.parse("1.0.0")),
                "bravo": ActivePackage(semver.Version.parse("1.0.0")),
            }
        ),
        home=home,
        env=env,
    )

    packages = select_package_roots(
        home=home,
        proj_dir=None,
        cwd=tmp_path,
        development_packages=(development,),
        env=env,
    )

    assert tuple(package.root for package in packages) == (
        development.root,
        (home / "packages" / "alpha" / "1.0.0").resolve(),
    )


def test_effective_exec_roots_mounts_indexed_packages_under_agm_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "agm-home"
    package_root = _write_package(home, "alpha", "1.0.0")
    write_activation_index(
        ActivationIndex(packages={"alpha": ActivePackage(semver.Version.parse("1.0.0"))}),
        home=home,
        env={"AGM_HOME": str(home)},
    )
    monkeypatch.setenv("AGM_HOME", str(home))
    monkeypatch.setenv("AGM_STDLIB", str(tmp_path / "missing-stdlib"))
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    roots = effective_exec_roots(
        entry_path=None,
        module_paths=[],
        cwd=cwd,
        home=tmp_path / "user-home",
        proj_dir=None,
    )

    assert tuple(package.manifest.name for package in roots.packages) == ("alpha",)
    assert roots.packages[0].root == package_root.resolve()


def test_missing_activation_index_and_store_rebuild_to_empty(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    env = {"AGM_HOME": str(home)}

    assert load_activation_index(home=home, env=env) == ActivationIndex()
    assert rebuild_activation_index(home=home, env=env) == ActivationIndex()


def test_rebuild_ignores_non_package_store_entries(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    store = home / "packages"
    store.mkdir(parents=True)
    (store / "index.toml").write_text("", encoding="utf-8")
    name_dir = store / "alpha"
    name_dir.mkdir()
    (name_dir / "notes").write_text("ignored", encoding="utf-8")
    source = tmp_path / "linked-package"
    source.mkdir()
    (store / "linked").symlink_to(source, target_is_directory=True)

    assert rebuild_activation_index(home=home, env={"AGM_HOME": str(home)}) == ActivationIndex()


@pytest.mark.parametrize(
    "content",
    (
        "extra = true\n",
        "packages = 1\n",
        '[packages.alpha]\nversion = "1.0"\n',
        '[packages.alpha]\nversion = "1.0.0"\nextra = true\n',
        '[packages.alpha]\nversion = "1.0.0"\neditable = "relative"\n',
        '[packages.alpha]\nversion = "1.0.0"\nshadow = "yes"\n',
        '[packages]\nalpha = "1.0.0"\n',
        "[packages.alpha]\n",
        '[packages."bad/name"]\nversion = "1.0.0"\n',
    ),
)
def test_activation_index_rejects_invalid_state(tmp_path: Path, content: str) -> None:
    home = tmp_path / "agm-home"
    index_path = home / "packages" / "index.toml"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(content, encoding="utf-8")

    with pytest.raises(PackageActivationError):
        load_activation_index(home=home, env={"AGM_HOME": str(home)})


def test_activation_index_rejects_invalid_toml(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    index_path = home / "packages" / "index.toml"
    index_path.parent.mkdir(parents=True)
    index_path.write_text("[packages\n", encoding="utf-8")

    with pytest.raises(PackageActivationError):
        load_activation_index(home=home, env={"AGM_HOME": str(home)})


@pytest.mark.parametrize("pin", ("1.0", "not-a-version"))
def test_package_pins_must_be_complete_semver(tmp_path: Path, pin: str) -> None:
    project = tmp_path / "project"
    config = project / "config"
    config.mkdir(parents=True)
    (config / "config.toml").write_text(f'[packages]\nalpha = "{pin}"\n', encoding="utf-8")

    with pytest.raises(PackageActivationError):
        select_active_packages(home=tmp_path / "home", proj_dir=project, cwd=project, env={})


@pytest.mark.parametrize("config_text", ("[packages]\nalpha = 1\n", 'packages = "wrong"\n'))
def test_package_pins_must_be_a_table_of_version_strings(tmp_path: Path, config_text: str) -> None:
    project = tmp_path / "project"
    config = project / "config"
    config.mkdir(parents=True)
    (config / "config.toml").write_text(config_text, encoding="utf-8")

    with pytest.raises(PackageActivationError):
        select_active_packages(home=tmp_path / "home", proj_dir=project, cwd=project, env={})


def test_package_pins_are_empty_when_the_section_is_absent(tmp_path: Path) -> None:
    assert load_package_pins(home=tmp_path / "home", proj_dir=None, cwd=tmp_path, env={}) == {}


def test_non_editable_package_cannot_escape_the_canonical_store_root(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    external_root = tmp_path / "external"
    _write_development_package(external_root, "alpha", "1.0.0")
    store_version = home / "packages" / "alpha" / "1.0.0"
    store_version.parent.mkdir(parents=True)
    store_version.symlink_to(external_root, target_is_directory=True)
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex(packages={"alpha": ActivePackage(semver.Version.parse("1.0.0"))}),
        home=home,
        env=env,
    )

    with pytest.raises(PackageActivationError, match="store root"):
        select_active_packages(home=home, proj_dir=None, cwd=tmp_path, env=env)


def test_active_package_must_exist_and_match_its_selected_identity(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex(packages={"alpha": ActivePackage(semver.Version.parse("1.0.0"))}),
        home=home,
        env=env,
    )

    with pytest.raises(PackageActivationError):
        select_active_packages(home=home, proj_dir=None, cwd=tmp_path, env=env)

    root = _write_package(home, "alpha", "1.0.0")
    (root / "package.toml").write_text(
        '[package]\nname = "bravo"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    with pytest.raises(PackageActivationError):
        select_active_packages(home=home, proj_dir=None, cwd=tmp_path, env=env)

    (root / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "2.0.0"\n', encoding="utf-8"
    )
    with pytest.raises(PackageActivationError):
        select_active_packages(home=home, proj_dir=None, cwd=tmp_path, env=env)


def test_editable_package_uses_its_live_manifest_for_requirement_checks(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    editable = tmp_path / "editable"
    (editable / "alpha").mkdir(parents=True)
    (editable / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "2.0.0"\n\n[dependencies]\nbravo = "1.0"\n',
        encoding="utf-8",
    )
    bravo = _write_package(home, "bravo", "1.0.0")
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex(
            packages={
                "alpha": ActivePackage(semver.Version.parse("1.0.0"), editable=editable),
                "bravo": ActivePackage(semver.Version.parse("1.0.0")),
            }
        ),
        home=home,
        env=env,
    )

    packages = select_active_packages(home=home, proj_dir=None, cwd=tmp_path, env=env)

    assert tuple(package.root for package in packages) == (editable.resolve(), bravo.resolve())


def test_requirements_must_name_an_active_package(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    _write_package(home, "alpha", "1.0.0", '\n[dependencies]\nbravo = "1.0"\n')
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex(packages={"alpha": ActivePackage(semver.Version.parse("1.0.0"))}),
        home=home,
        env=env,
    )

    with pytest.raises(PackageActivationError, match="bravo"):
        select_active_packages(home=home, proj_dir=None, cwd=tmp_path, env=env)


@pytest.mark.parametrize("name", ("bad-name", "alpha/bravo"))
def test_write_activation_index_rejects_non_segment_package_names(
    tmp_path: Path, name: str
) -> None:
    with pytest.raises(PackageActivationError):
        write_activation_index(
            ActivationIndex(packages={name: ActivePackage(semver.Version.parse("1.0.0"))}),
            home=tmp_path / "home",
            env={},
        )


def test_rebuild_rejects_store_directory_that_disagrees_with_manifest(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    _write_package(home, "alpha", "1.0.0")
    root = home / "packages" / "alpha" / "1.0.0"
    (root / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "2.0.0"\n', encoding="utf-8"
    )

    with pytest.raises(PackageActivationError):
        rebuild_activation_index(home=home, env={"AGM_HOME": str(home)})
