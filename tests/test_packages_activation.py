"""Tests for installed-package activation, pins, command ownership, and root selection."""

from __future__ import annotations

from pathlib import Path
from typing import Never

import pytest
import semver

from agm.agl.keywords import KEYWORDS
from agm.cli_support.exec_roots import effective_exec_roots
from agm.packages.activation import (
    ActivationIndex,
    ActivePackage,
    CommandRegistration,
    CommandShadow,
    PackageActivationError,
    activation_index_path,
    command_shadow_diagnostics,
    effective_command_index,
    load_activation_index,
    load_package_pins,
    load_package_provenance,
    merge_package_commands,
    package_provenance_path,
    rebuild_activation_index,
    select_active_packages,
    select_package_roots,
    write_activation_index,
    write_package_provenance,
)
from agm.packages.manifest import load_manifest
from agm.packages.model import PackageInfo
from agm.packages.record import write_record
from agm.version import AGM_VERSION


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
    write_record(root)
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


def test_activation_index_round_trips_command_ownership_and_registration_order(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agm-home"
    index = ActivationIndex(
        packages={"alpha": ActivePackage(semver.Version.parse("1.2.3"), registration_order=4)},
        commands={
            "alpha run": CommandRegistration(
                package="alpha",
                program="alpha/main::main",
                description="Run alpha",
            )
        },
    )

    write_activation_index(index, home=home, env={"AGM_HOME": str(home)})

    assert load_activation_index(home=home, env={"AGM_HOME": str(home)}) == index


def test_package_provenance_is_a_record_external_sidecar(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    active = ActivePackage(semver.Version.parse("1.2.3"), shadow=True, registration_order=4)
    package = _write_package(home, "alpha", "1.2.3")

    path = write_package_provenance("alpha", active, home=home, env={"AGM_HOME": str(home)})

    provenance = load_package_provenance(
        "alpha", active.version, home=home, env={"AGM_HOME": str(home)}
    )

    assert path == package.parent / ".provenance" / "1.2.3.toml"
    assert not path.is_relative_to(package)
    assert provenance is not None
    assert provenance.registration_order == 4
    assert provenance.shadow


@pytest.mark.parametrize(
    "content",
    (
        "extra = true\n",
        "activation = 'not a table'\n",
        "[activation]\nregistration-order = 1\nshadow = true\nextra = true\n",
        "[activation]\nregistration-order = 0\nshadow = true\n",
    ),
)
def test_package_provenance_rejects_invalid_state(tmp_path: Path, content: str) -> None:
    home = tmp_path / "agm-home"
    _write_package(home, "alpha", "1.0.0")
    path = package_provenance_path(
        "alpha", semver.Version.parse("1.0.0"), home=home, env={"AGM_HOME": str(home)}
    )
    path.parent.mkdir()
    path.write_text(content, encoding="utf-8")

    with pytest.raises(PackageActivationError, match="provenance"):
        load_package_provenance(
            "alpha", semver.Version.parse("1.0.0"), home=home, env={"AGM_HOME": str(home)}
        )


def test_package_provenance_refuses_escaped_store_sidecars(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    store = home / "packages"
    external = tmp_path / "external"
    store.mkdir(parents=True)
    external.mkdir()
    (store / "alpha").symlink_to(external, target_is_directory=True)

    with pytest.raises(PackageActivationError, match="outside"):
        package_provenance_path(
            "alpha", semver.Version.parse("1.0.0"), home=home, env={"AGM_HOME": str(home)}
        )


def test_package_provenance_rejects_editable_and_write_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "agm-home"
    version = semver.Version.parse("1.0.0")
    with pytest.raises(PackageActivationError, match="editable"):
        write_package_provenance(
            "alpha", ActivePackage(version, editable=tmp_path), home=home, env={}
        )

    def fail_open(_: Path, *__: object, **___: object) -> Never:
        raise OSError("full")

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(PackageActivationError, match="cannot write"):
        write_package_provenance(
            "alpha", ActivePackage(version, registration_order=1), home=home, env={}
        )


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


def test_activation_index_write_failure_keeps_the_previous_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "agm-home"
    env = {"AGM_HOME": str(home)}
    original = ActivationIndex({"alpha": ActivePackage(semver.Version.parse("1.0.0"))})
    write_activation_index(original, home=home, env=env)

    def fail_open(_: Path, *__: object, **___: object) -> Never:
        raise OSError("full")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_open)
        with pytest.raises(PackageActivationError, match="cannot write"):
            write_activation_index(
                ActivationIndex({"alpha": ActivePackage(semver.Version.parse("2.0.0"))}),
                home=home,
                env=env,
            )

    assert load_activation_index(home=home, env=env) == original


def test_rebuild_index_merges_registered_commands_from_active_manifests(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    alpha = _write_package(home, "alpha", "1.0.0")
    bravo = _write_package(home, "bravo", "2.0.0")
    (alpha / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "1.0.0"\n\n'
        '[commands]\nalpha-run = { program = "alpha/main::main", description = "Run alpha" }\n',
        encoding="utf-8",
    )
    (bravo / "package.toml").write_text(
        '[package]\nname = "bravo"\nversion = "2.0.0"\n\n'
        '[commands]\n"bravo run" = { program = "bravo/main::main" }\n',
        encoding="utf-8",
    )
    write_record(alpha)
    write_record(bravo)

    assert rebuild_activation_index(home=home, env={"AGM_HOME": str(home)}) == ActivationIndex(
        packages={
            "alpha": ActivePackage(semver.Version.parse("1.0.0"), registration_order=1),
            "bravo": ActivePackage(semver.Version.parse("2.0.0"), registration_order=2),
        },
        commands={
            "alpha-run": CommandRegistration("alpha", "alpha/main::main", "Run alpha"),
            "bravo run": CommandRegistration("bravo", "bravo/main::main"),
        },
    )


def test_effective_command_index_reconciles_cached_immutable_command_priority(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agm-home"
    alpha = _write_package(home, "alpha", "1.0.0")
    bravo = _write_package(home, "bravo", "1.0.0")
    for root, name in ((alpha, "alpha"), (bravo, "bravo")):
        (root / "package.toml").write_text(
            f'[package]\nname = "{name}"\nversion = "1.0.0"\n\n'
            f'[commands]\nlaunch = {{ program = "{name}/main::main" }}\n',
            encoding="utf-8",
        )
        write_record(root)
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex(
            {
                "alpha": ActivePackage(semver.Version.parse("1.0.0"), registration_order=1),
                "bravo": ActivePackage(
                    semver.Version.parse("1.0.0"), shadow=True, registration_order=2
                ),
            },
            {"launch": CommandRegistration("alpha", "alpha/main::main")},
        ),
        home=home,
        env=env,
    )

    assert effective_command_index(home=home, proj_dir=None, cwd=tmp_path, env=env).commands == {
        "launch": CommandRegistration("bravo", "bravo/main::main")
    }


def test_effective_command_index_reuses_a_preresolved_index(tmp_path: Path) -> None:
    """A caller that already loaded the activation index can hand it back in."""
    home = tmp_path / "agm-home"
    _write_package(home, "alpha", "1.0.0")
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex(
            {"alpha": ActivePackage(semver.Version.parse("1.0.0"), registration_order=1)}
        ),
        home=home,
        env=env,
    )
    supplied = ActivationIndex(
        {"alpha": ActivePackage(semver.Version.parse("1.0.0"), registration_order=99)}
    )

    result = effective_command_index(
        home=home, proj_dir=None, cwd=tmp_path, env=env, index=supplied
    )

    # The supplied index's registration order wins, proving it was used
    # directly instead of the differing persisted index being reloaded.
    assert result.packages["alpha"].registration_order == 99


def test_effective_command_index_uses_live_editable_manifest_commands(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    root = tmp_path / "editable"
    (root / "alpha").mkdir(parents=True)
    (root / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "1.0.0"\n\n'
        '[commands]\nlaunch = { program = "alpha/main::current" }\n',
        encoding="utf-8",
    )
    env = {"AGM_HOME": str(home)}
    index = ActivationIndex(
        {"alpha": ActivePackage(semver.Version.parse("1.0.0"), editable=root)},
        {"launch": CommandRegistration("alpha", "alpha/main::stale")},
    )
    write_activation_index(index, home=home, env=env)

    assert effective_command_index(home=home, proj_dir=None, cwd=tmp_path, env=env).commands == {
        "launch": CommandRegistration("alpha", "alpha/main::current")
    }


def test_effective_command_index_rejects_a_live_editable_command_conflict_without_shadow(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agm-home"
    alpha = _write_package(home, "alpha", "1.0.0")
    (alpha / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "1.0.0"\n\n'
        '[commands]\nlaunch = { program = "alpha/main::main" }\n',
        encoding="utf-8",
    )
    write_record(alpha)
    bravo = tmp_path / "editable"
    (bravo / "bravo").mkdir(parents=True)
    (bravo / "package.toml").write_text(
        '[package]\nname = "bravo"\nversion = "1.0.0"\n\n'
        '[commands]\ninspect = { program = "bravo/main::main" }\n',
        encoding="utf-8",
    )
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex(
            {
                "alpha": ActivePackage(semver.Version.parse("1.0.0"), registration_order=1),
                "bravo": ActivePackage(
                    semver.Version.parse("1.0.0"),
                    editable=bravo,
                    registration_order=2,
                ),
            },
            {
                "launch": CommandRegistration("alpha", "alpha/main::main"),
                "inspect": CommandRegistration("bravo", "bravo/main::main"),
            },
        ),
        home=home,
        env=env,
    )
    (bravo / "package.toml").write_text(
        '[package]\nname = "bravo"\nversion = "1.0.0"\n\n'
        '[commands]\ninspect = { program = "bravo/main::main" }\n'
        'launch = { program = "bravo/main::main" }\n',
        encoding="utf-8",
    )

    with pytest.raises(PackageActivationError, match="launch.*shadow"):
        effective_command_index(home=home, proj_dir=None, cwd=tmp_path, env=env)


@pytest.mark.parametrize("active_kind", ("editable", "legacy"))
def test_effective_command_index_uses_pinned_version_provenance_without_losing_active_metadata(
    tmp_path: Path, active_kind: str
) -> None:
    home = tmp_path / "agm-home"
    old = _write_package(home, "alpha", "1.0.0")
    current = _write_package(home, "alpha", "2.0.0")
    for root in (old, current):
        (root / "package.toml").write_text(
            f'[package]\nname = "alpha"\nversion = "{root.name}"\n\n'
            '[commands]\nlaunch = { program = "alpha/main::main" }\n',
            encoding="utf-8",
        )
        write_record(root)
    if active_kind == "editable":
        bravo = tmp_path / "editable"
        (bravo / "bravo").mkdir(parents=True)
        (bravo / "package.toml").write_text(
            '[package]\nname = "bravo"\nversion = "1.0.0"\n\n'
            '[commands]\nlaunch = { program = "bravo/main::main" }\n',
            encoding="utf-8",
        )
        editable = bravo
    else:
        bravo = _write_package(home, "bravo", "1.0.0")
        (bravo / "package.toml").write_text(
            '[package]\nname = "bravo"\nversion = "1.0.0"\n\n'
            '[commands]\nlaunch = { program = "bravo/main::main" }\n',
            encoding="utf-8",
        )
        write_record(bravo)
        editable = None

    env = {"AGM_HOME": str(home)}
    version = semver.Version.parse("1.0.0")
    write_package_provenance(
        "alpha", ActivePackage(version, registration_order=1), home=home, env=env
    )
    write_activation_index(
        ActivationIndex(
            {
                "alpha": ActivePackage(
                    semver.Version.parse("2.0.0"), shadow=True, registration_order=3
                ),
                "bravo": ActivePackage(
                    version, editable=editable, shadow=True, registration_order=2
                ),
            },
            {"launch": CommandRegistration("alpha", "alpha/main::main")},
        ),
        home=home,
        env=env,
    )
    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "config" / "config.toml").write_text(
        '[packages]\nalpha = "1.0.0"\n', encoding="utf-8"
    )

    effective = effective_command_index(home=home, proj_dir=project, cwd=project, env=env)

    assert effective.packages["alpha"] == ActivePackage(version, registration_order=1)
    assert effective.commands == {"launch": CommandRegistration("bravo", "bravo/main::main")}


@pytest.mark.parametrize(("indexed", "registration_order"), ((True, 4), (False, 0)))
def test_effective_command_index_supports_legacy_pins_without_version_provenance(
    tmp_path: Path, indexed: bool, registration_order: int
) -> None:
    home = tmp_path / "agm-home"
    old = _write_package(home, "alpha", "1.0.0")
    (old / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "1.0.0"\n\n'
        '[commands]\nlaunch = { program = "alpha/main::main" }\n',
        encoding="utf-8",
    )
    write_record(old)
    env = {"AGM_HOME": str(home)}
    if indexed:
        _write_package(home, "alpha", "2.0.0")
        write_activation_index(
            ActivationIndex(
                {
                    "alpha": ActivePackage(
                        semver.Version.parse("2.0.0"),
                        shadow=True,
                        registration_order=registration_order,
                    )
                }
            ),
            home=home,
            env=env,
        )
    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "config" / "config.toml").write_text(
        '[packages]\nalpha = "1.0.0"\n', encoding="utf-8"
    )

    effective = effective_command_index(home=home, proj_dir=project, cwd=project, env=env)

    assert effective.packages["alpha"] == ActivePackage(
        semver.Version.parse("1.0.0"),
        shadow=indexed,
        registration_order=registration_order,
    )
    assert effective.commands == {"launch": CommandRegistration("alpha", "alpha/main::main")}


def test_effective_command_index_rebuilds_for_a_pin_with_different_build_metadata(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agm-home"
    old = _write_package(home, "alpha", "1.0.0+old")
    new = _write_package(home, "alpha", "1.0.0+new")
    for root, program in ((old, "old"), (new, "new")):
        (root / "package.toml").write_text(
            f'[package]\nname = "alpha"\nversion = "{root.name}"\n\n'
            f'[commands]\nlaunch = {{ program = "alpha/main::{program}" }}\n',
            encoding="utf-8",
        )
        write_record(root)
    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "config" / "config.toml").write_text(
        '[packages]\nalpha = "1.0.0+new"\n', encoding="utf-8"
    )
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex(
            {"alpha": ActivePackage(semver.Version.parse("1.0.0+old"))},
            {"launch": CommandRegistration("alpha", "alpha/main::old")},
        ),
        home=home,
        env=env,
    )

    assert effective_command_index(home=home, proj_dir=project, cwd=project, env=env).commands == {
        "launch": CommandRegistration("alpha", "alpha/main::new")
    }


def test_rebuild_does_not_reuse_legacy_metadata_for_different_build_metadata(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agm-home"
    _write_package(home, "alpha", "1.0.0+new")
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex(
            {"alpha": ActivePackage(semver.Version.parse("1.0.0+old"), registration_order=4)}
        ),
        home=home,
        env=env,
    )

    rebuilt = rebuild_activation_index(home=home, env=env)

    assert str(rebuilt.packages["alpha"].version) == "1.0.0+new"
    assert rebuilt.packages["alpha"].registration_order == 5


def test_rebuild_uses_the_previous_index_only_for_legacy_packages_without_sidecars(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agm-home"
    _write_package(home, "alpha", "1.0.0")
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex(
            {"alpha": ActivePackage(semver.Version.parse("1.0.0"), registration_order=4)}
        ),
        home=home,
        env=env,
    )

    rebuilt = rebuild_activation_index(home=home, env=env)

    assert rebuilt.packages["alpha"].registration_order == 4


def test_rebuild_allocates_legacy_orders_after_provenance_without_an_index(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agm-home"
    _write_package(home, "alpha", "1.0.0")
    _write_package(home, "bravo", "1.0.0")
    env = {"AGM_HOME": str(home)}
    write_package_provenance(
        "alpha",
        ActivePackage(semver.Version.parse("1.0.0"), registration_order=1),
        home=home,
        env=env,
    )

    rebuilt = rebuild_activation_index(home=home, env=env)
    write_activation_index(rebuilt, home=home, env=env)

    assert rebuilt.packages["alpha"].registration_order == 1
    assert rebuilt.packages["bravo"].registration_order == 2
    assert rebuild_activation_index(home=home, env=env) == rebuilt


def test_rebuild_rejects_duplicate_or_missing_command_provenance(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    alpha = _write_package(home, "alpha", "1.0.0")
    bravo = _write_package(home, "bravo", "1.0.0")
    env = {"AGM_HOME": str(home)}
    for package, name in ((alpha, "alpha"), (bravo, "bravo")):
        (package / "package.toml").write_text(
            f'[package]\nname = "{name}"\nversion = "1.0.0"\n\n'
            f'[commands]\nlaunch = {{ program = "{name}/main::main" }}\n',
            encoding="utf-8",
        )
        write_record(package)

    with pytest.raises(PackageActivationError, match="missing"):
        rebuild_activation_index(home=home, env=env)

    version = semver.Version.parse("1.0.0")
    for name in ("alpha", "bravo"):
        write_package_provenance(
            name,
            ActivePackage(version, registration_order=1),
            home=home,
            env=env,
        )
    with pytest.raises(PackageActivationError, match="duplicate"):
        rebuild_activation_index(home=home, env=env)


def test_rebuild_rejects_a_later_colliding_registration_without_shadow_intent(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agm-home"
    alpha = _write_package(home, "alpha", "1.0.0")
    bravo = _write_package(home, "bravo", "1.0.0")
    env = {"AGM_HOME": str(home)}
    for package, name in ((alpha, "alpha"), (bravo, "bravo")):
        (package / "package.toml").write_text(
            f'[package]\nname = "{name}"\nversion = "1.0.0"\n\n'
            f'[commands]\nlaunch = {{ program = "{name}/main::main" }}\n',
            encoding="utf-8",
        )
        write_record(package)
    version = semver.Version.parse("1.0.0")
    write_package_provenance(
        "alpha", ActivePackage(version, registration_order=1), home=home, env=env
    )
    write_package_provenance(
        "bravo", ActivePackage(version, registration_order=2), home=home, env=env
    )

    with pytest.raises(PackageActivationError, match="shadow"):
        rebuild_activation_index(home=home, env=env)


def test_command_shadow_diagnostics_and_registry_merge_are_deterministic(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    alpha = _write_package(home, "alpha", "1.0.0")
    bravo = _write_package(home, "bravo", "1.0.0")
    charlie = _write_package(home, "charlie", "1.0.0")
    for package, name, extra in (
        (alpha, "alpha", ""),
        (bravo, "bravo", 'inspect = { program = "bravo/main::main" }\n'),
        (charlie, "charlie", ""),
    ):
        (package / "package.toml").write_text(
            f'[package]\nname = "{name}"\nversion = "1.0.0"\n\n'
            f'[commands]\nlaunch = {{ program = "{name}/main::main" }}\n{extra}',
            encoding="utf-8",
        )
        write_record(package)
    version = semver.Version.parse("1.0.0")
    index = ActivationIndex(
        {
            "alpha": ActivePackage(version, registration_order=1),
            "bravo": ActivePackage(version, registration_order=3),
            "charlie": ActivePackage(version, registration_order=2),
        },
        {"launch": CommandRegistration("alpha", "alpha/main::main")},
    )

    diagnostics = command_shadow_diagnostics(index, home=home, env={"AGM_HOME": str(home)})

    assert diagnostics == {"bravo": (CommandShadow("launch", ("alpha", "charlie")),)}
    with pytest.raises(PackageActivationError, match="launch"):
        merge_package_commands(index, load_manifest(bravo / "package.toml"), shadow=False)


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
            "alpha": ActivePackage(semver.Version.parse("2.0.0"), registration_order=1),
            "bravo": ActivePackage(semver.Version.parse("1.5.0"), registration_order=2),
        }
    )
    assert load_activation_index(home=home, env=env) == rebuilt


def test_rebuild_preserves_active_equal_precedence_build(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    _write_package(home, "alpha", "1.0.0+linux")
    _write_package(home, "alpha", "1.0.0+macos")
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex({"alpha": ActivePackage(semver.Version.parse("1.0.0+macos"))}),
        home=home,
        env=env,
    )

    rebuilt = rebuild_activation_index(home=home, env=env)

    assert str(rebuilt.packages["alpha"].version) == "1.0.0+macos"


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


def test_std_is_never_selected_as_a_package_root(tmp_path: Path) -> None:
    """``std`` is mounted only through the stdlib seam, never as a package root."""
    home = tmp_path / "agm-home"
    _write_package(home, "std", "1.0.0")
    development_root = tmp_path / "development"
    (development_root / "std").mkdir(parents=True)
    (development_root / "package.toml").write_text(
        '[package]\nname = "std"\nversion = "2.0.0"\n', encoding="utf-8"
    )
    from agm.packages.manifest import load_manifest
    from agm.packages.model import PackageInfo

    development = PackageInfo(development_root, load_manifest(development_root / "package.toml"))
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex(packages={"std": ActivePackage(semver.Version.parse("1.0.0"))}),
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

    assert packages == ()


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
    ).roots

    assert tuple(package.manifest.name for package in roots.packages) == ("alpha",)
    assert roots.packages[0].root == package_root.resolve()


def test_effective_exec_roots_treats_stdlib_override_as_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An override must not also mount the active managed std package."""
    home = tmp_path / "agm-home"
    active_std = _write_package(home, "std", AGM_VERSION)
    _write_package(home, "alpha", "1.0.0", '\n[dependencies]\nstd = "0.1"\n')
    override = tmp_path / "override"
    (override / "std").mkdir(parents=True)
    write_activation_index(
        ActivationIndex(
            packages={
                "std": ActivePackage(semver.Version.parse(AGM_VERSION)),
                "alpha": ActivePackage(semver.Version.parse("1.0.0")),
            }
        ),
        home=home,
        env={"AGM_HOME": str(home)},
    )
    monkeypatch.setenv("AGM_HOME", str(home))
    monkeypatch.setenv("AGM_STDLIB", str(override))
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    roots = effective_exec_roots(
        entry_path=None,
        module_paths=[],
        cwd=cwd,
        home=tmp_path / "user-home",
        proj_dir=None,
    ).roots

    assert roots.stdlib_roots == {override.resolve()}
    assert tuple(package.manifest.name for package in roots.packages) == ("alpha",)
    assert active_std.resolve() not in roots.roots


def test_effective_exec_roots_falls_back_when_active_std_tree_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing managed tree does not block the source-checkout fallback."""
    home = tmp_path / "agm-home"
    write_activation_index(
        ActivationIndex({"std": ActivePackage(semver.Version.parse(AGM_VERSION))}),
        home=home,
        env={"AGM_HOME": str(home)},
    )
    monkeypatch.setenv("AGM_HOME", str(home))
    monkeypatch.delenv("AGM_STDLIB", raising=False)
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    roots = effective_exec_roots(
        entry_path=None,
        module_paths=[],
        cwd=cwd,
        home=tmp_path / "user-home",
        proj_dir=None,
    ).roots

    assert roots.stdlib_roots == {Path(__file__).resolve().parents[1] / "stdlib"}
    assert roots.packages == ()


def test_package_root_selection_rejects_a_std_requirement_newer_than_agm(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    newer_agm = semver.Version.parse(AGM_VERSION).bump_major()
    _write_package(home, "alpha", "1.0.0", f'\n[dependencies]\nstd = "{newer_agm}"\n')
    write_activation_index(
        ActivationIndex({"alpha": ActivePackage(semver.Version.parse("1.0.0"))}),
        home=home,
        env={"AGM_HOME": str(home)},
    )

    with pytest.raises(PackageActivationError, match="AGM"):
        select_package_roots(
            home=home,
            proj_dir=None,
            cwd=tmp_path,
            env={"AGM_HOME": str(home)},
        )


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
        '[packages.alpha]\nversion = "1.0.0"\nregistration-order = "first"\n',
        '[packages]\nalpha = "1.0.0"\n',
        "[packages.alpha]\n",
        '[packages."bad/name"]\nversion = "1.0.0"\n',
        "commands = 1\n",
        '[commands]\nlaunch = "not a table"\n',
        '[packages.alpha]\nversion = "1.0.0"\n\n[commands.launch]\nextra = true\n',
        '[packages.alpha]\nversion = "1.0.0"\n\n[commands.launch]\npackage = "alpha"\n',
        (
            '[packages.alpha]\nversion = "1.0.0"\n\n[commands.launch]\n'
            'package = "alpha"\nprogram = "alpha/main::main"\ndescription = 1\n'
        ),
        (
            '[packages.alpha]\nversion = "1.0.0"\n\n[commands.launch]\n'
            'package = "alpha"\nprogram = "alpha/main::main"\nshadowed-package = "bravo"\n'
        ),
        (
            '[packages.alpha]\nversion = "1.0.0"\n\n[commands."bad  path"]\n'
            'package = "alpha"\nprogram = "alpha/main::main"\n'
        ),
        (
            '[packages.alpha]\nversion = "1.0.0"\n\n[commands."exec launch"]\n'
            'package = "alpha"\nprogram = "alpha/main::main"\n'
        ),
        (
            '[packages.alpha]\nversion = "1.0.0"\n\n[commands.launch]\n'
            'package = "bravo"\nprogram = "bravo/main::main"\n'
        ),
    ),
)
def test_activation_index_rejects_invalid_state(tmp_path: Path, content: str) -> None:
    home = tmp_path / "agm-home"
    index_path = home / "packages" / "index.toml"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(content, encoding="utf-8")

    with pytest.raises(PackageActivationError):
        load_activation_index(home=home, env={"AGM_HOME": str(home)})


@pytest.mark.parametrize(
    ("active", "command"),
    (
        (ActivePackage(semver.Version.parse("1.0.0"), registration_order=-1), None),
        (ActivePackage(semver.Version.parse("1.0.0")), CommandRegistration("alpha", "")),
        (
            ActivePackage(semver.Version.parse("1.0.0")),
            CommandRegistration("alpha", "alpha/main::main", ""),
        ),
        (
            ActivePackage(semver.Version.parse("1.0.0")),
            CommandRegistration("alpha", "alpha/main::main", description=""),
        ),
    ),
)
def test_activation_index_rejects_invalid_in_memory_command_registration(
    tmp_path: Path, active: ActivePackage, command: CommandRegistration | None
) -> None:
    commands = {} if command is None else {"launch": command}
    with pytest.raises(PackageActivationError):
        write_activation_index(
            ActivationIndex(packages={"alpha": active}, commands=commands),
            home=tmp_path / "home",
            env={},
        )


def test_activation_index_rejects_option_shaped_command_segment(tmp_path: Path) -> None:
    with pytest.raises(PackageActivationError):
        write_activation_index(
            ActivationIndex(
                packages={"alpha": ActivePackage(semver.Version.parse("1.0.0"))},
                commands={"tools --help": CommandRegistration("alpha", "alpha/main::main")},
            ),
            home=tmp_path / "home",
            env={},
        )


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

    write_activation_index(
        ActivationIndex(
            packages={
                "alpha": ActivePackage(
                    semver.Version.parse("1.0.0"), editable=tmp_path / "missing-editable"
                )
            }
        ),
        home=home,
        env=env,
    )
    with pytest.raises(PackageActivationError):
        select_active_packages(home=home, proj_dir=None, cwd=tmp_path, env=env)

    write_activation_index(
        ActivationIndex(packages={"alpha": ActivePackage(semver.Version.parse("1.0.0"))}),
        home=home,
        env=env,
    )
    root = _write_package(home, "alpha", "1.0.0")
    (root / "package.toml").write_text(
        '[package]\nname = "bravo"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    write_record(root)
    with pytest.raises(PackageActivationError):
        select_active_packages(home=home, proj_dir=None, cwd=tmp_path, env=env)

    (root / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "2.0.0"\n', encoding="utf-8"
    )
    write_record(root)
    with pytest.raises(PackageActivationError):
        select_active_packages(home=home, proj_dir=None, cwd=tmp_path, env=env)


def test_active_immutable_package_must_match_exact_build_metadata(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    root = _write_package(home, "alpha", "1.0.0+selected")
    (root / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "1.0.0+other"\n', encoding="utf-8"
    )
    write_record(root)
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex({"alpha": ActivePackage(semver.Version.parse("1.0.0+selected"))}),
        home=home,
        env=env,
    )

    with pytest.raises(PackageActivationError, match="mismatched"):
        select_active_packages(home=home, proj_dir=None, cwd=tmp_path, env=env)


def test_editable_package_resolves_to_its_live_root(tmp_path: Path) -> None:
    home = tmp_path / "agm-home"
    editable = _write_development_package(tmp_path / "editable", "alpha", "1.0.0")
    env = {"AGM_HOME": str(home)}
    write_activation_index(
        ActivationIndex(
            {"alpha": ActivePackage(semver.Version.parse("1.0.0"), editable=editable.root)}
        ),
        home=home,
        env=env,
    )

    assert select_active_packages(home=home, proj_dir=None, cwd=tmp_path, env=env) == (editable,)


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


@pytest.mark.parametrize("name", sorted(KEYWORDS))
def test_write_activation_index_rejects_reserved_keyword_package_names(
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
