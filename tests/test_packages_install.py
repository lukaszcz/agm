"""Tests for package installation, activation, and removal."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
import semver

import agm.packages.archive as package_archive
import agm.packages.install as package_install
from agm.core import dry_run
from agm.packages.activation import (
    ActivationIndex,
    ActivePackage,
    CommandRegistration,
    PackageActivationError,
    load_activation_index,
    rebuild_activation_index,
    write_activation_index,
)
from agm.packages.archive import write_archive
from agm.packages.install import (
    PackageInstallError,
    install_archive,
    install_directory,
    installed_packages,
    uninstall_package,
)
from agm.packages.record import verify_record
from agm.version import AGM_VERSION


@pytest.fixture(autouse=True)
def _restore_dry_run() -> Generator[None, None, None]:
    previous = dry_run.enabled()
    dry_run.set_enabled(False)
    yield
    dry_run.set_enabled(previous)


def _newer_agm_requirement() -> str:
    """Return a valid minimum version no running AGM release can satisfy."""
    return str(semver.Version.parse(AGM_VERSION).bump_major())


def _package(root: Path, name: str, version: str, dependencies: str = "") -> Path:
    root.mkdir()
    (root / name).mkdir()
    (root / "package.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "{version}"\n' + dependencies,
        encoding="utf-8",
    )
    (root / name / "main.agl").write_text("program def main() -> unit = ()\n", encoding="utf-8")
    return root


def test_install_registers_stdlib_package_under_an_isolated_agm_home(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parent.parent / "stdlib"
    agm_home = tmp_path / "agm-home"

    installed = install_directory(
        source,
        home=tmp_path / "ignored-home",
        env={"AGM_HOME": str(agm_home)},
    )

    assert installed.root == agm_home / "packages" / "std" / AGM_VERSION
    assert (installed.root / "std" / "core.agl").is_file()
    assert verify_record(installed.root)
    index = load_activation_index(home=tmp_path / "ignored-home", env={"AGM_HOME": str(agm_home)})
    assert index.packages["std"].version == installed.manifest.version


def test_package_operations_create_a_store_lock(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    home = tmp_path / "home"

    install_directory(source, home=home, env={})

    lock = home / ".agm" / "packages" / ".lock"
    assert lock.is_file()
    uninstall_package("alpha", home=home, env={})
    assert lock.is_file()


def test_package_operations_report_store_lock_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    monkeypatch.setattr(
        Path,
        "mkdir",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("denied")),
    )

    with pytest.raises(PackageInstallError):
        install_directory(source, home=tmp_path / "home", env={})


def test_install_copies_package_writes_record_and_activates_it(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    home = tmp_path / "home"

    installed = install_directory(source, home=home, env={})

    assert installed.root == home / ".agm" / "packages" / "alpha" / "1.0.0"
    assert (installed.root / "alpha" / "main.agl").is_file()
    assert (installed.root.parent / ".provenance" / "1.0.0.toml").is_file()
    assert verify_record(installed.root)
    active = load_activation_index(home=home, env={}).packages["alpha"]
    assert active.version == installed.manifest.version


def test_install_keeps_provenance_separate_from_colliding_version_names(tmp_path: Path) -> None:
    home = tmp_path / "home"
    first = _package(tmp_path / "first", "alpha", "1.0.0+build")
    second = _package(tmp_path / "second", "alpha", "1.0.0+build.provenance.toml")

    install_directory(first, home=home, env={})
    install_directory(second, home=home, env={})

    versions = home / ".agm" / "packages" / "alpha"
    assert (versions / "1.0.0+build").is_dir()
    assert (versions / "1.0.0+build.provenance.toml").is_dir()
    assert (versions / ".provenance" / "1.0.0+build.toml").is_file()


def test_uninstall_refuses_the_managed_standard_library(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parent.parent / "stdlib"
    home = tmp_path / "home"
    installed = install_directory(source, home=home, env={})

    with pytest.raises(PackageInstallError, match="std"):
        uninstall_package("std", home=home, env={})

    assert installed.root.is_dir()
    assert "std" in load_activation_index(home=home, env={}).packages


def test_install_refuses_an_unmanaged_standard_library_source(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "std", AGM_VERSION)

    with pytest.raises(PackageInstallError, match="managed"):
        install_directory(source, home=tmp_path / "home", env={})


def test_install_refuses_a_standard_library_at_an_arbitrary_version(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "std", "0.0.1")

    with pytest.raises(PackageInstallError, match="AGM version"):
        install_directory(source, home=tmp_path / "home", env={})


def test_install_refuses_an_editable_standard_library(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parent.parent / "stdlib"

    with pytest.raises(PackageInstallError, match="editable"):
        install_directory(source, home=tmp_path / "home", env={}, editable=True)


def test_archive_install_refuses_the_managed_standard_library(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parent.parent / "stdlib"
    archive = tmp_path / "std.agmpkg"
    write_archive(source, archive)

    with pytest.raises(PackageInstallError, match="managed"):
        install_archive(archive, home=tmp_path / "home", env={})


def test_install_refuses_a_package_requiring_a_newer_agm(tmp_path: Path) -> None:
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        f'\n[dependencies]\nstd = "{_newer_agm_requirement()}"\n',
    )
    home = tmp_path / "home"

    with pytest.raises(PackageInstallError, match="AGM"):
        install_directory(source, home=home, env={})

    assert not (home / ".agm" / "packages" / "alpha").exists()


def test_install_accepts_a_package_requiring_the_running_agm_without_active_stdlib(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        f'\n[dependencies]\nstd = "{AGM_VERSION}"\n',
    )

    installed = install_directory(source, home=home, env={})

    assert installed.manifest.name == "alpha"
    assert load_activation_index(home=home, env={}).packages.keys() == {"alpha"}


def test_install_archive_refuses_a_package_requiring_a_newer_agm(tmp_path: Path) -> None:
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        f'\n[dependencies]\nstd = "{_newer_agm_requirement()}"\n',
    )
    archive = tmp_path / "alpha.agmpkg"
    write_archive(source, archive)

    with pytest.raises(PackageInstallError, match="AGM"):
        install_archive(archive, home=tmp_path / "home", env={})


def test_dry_run_archive_install_refuses_a_package_requiring_a_newer_agm(
    tmp_path: Path,
) -> None:
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        f'\n[dependencies]\nstd = "{_newer_agm_requirement()}"\n',
    )
    archive = tmp_path / "alpha.agmpkg"
    write_archive(source, archive)
    dry_run.set_enabled(True)

    with pytest.raises(PackageInstallError, match="AGM"):
        install_archive(archive, home=tmp_path / "home", env={})


def test_uninstall_tolerates_an_absent_provenance_sidecar(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    home = tmp_path / "home"
    installed = install_directory(source, home=home, env={})
    (installed.root.parent / ".provenance" / "1.0.0.toml").unlink()

    uninstall_package("alpha", home=home, env={})

    assert not installed.root.exists()
    assert "alpha" not in load_activation_index(home=home, env={}).packages


def test_uninstall_reports_a_provenance_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    home = tmp_path / "home"
    install_directory(source, home=home, env={})
    original_unlink = package_install.fs.unlink

    def fail_provenance_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path.parent.name == ".provenance":
            raise OSError("blocked")
        original_unlink(path)

    monkeypatch.setattr(package_install.fs, "unlink", fail_provenance_unlink)
    with pytest.raises(PackageInstallError, match="provenance"):
        uninstall_package("alpha", home=home, env={})


def test_legacy_activation_orders_are_assigned_when_an_editable_package_is_present(
    tmp_path: Path,
) -> None:
    alpha = _package(tmp_path / "alpha", "alpha", "1.0.0")
    bravo = _package(tmp_path / "bravo", "bravo", "1.0.0")
    home = tmp_path / "home"
    install_directory(alpha, home=home, env={}, editable=True)
    active_alpha = load_activation_index(home=home, env={}).packages["alpha"]
    write_activation_index(
        ActivationIndex({"alpha": ActivePackage(active_alpha.version, editable=alpha)}),
        home=home,
        env={},
    )
    install_directory(bravo, home=home, env={})

    index = load_activation_index(home=home, env={})
    assert index.packages["alpha"].registration_order > 0


def test_install_merges_commands_and_unregisters_them_on_uninstall(
    tmp_path: Path,
) -> None:
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        '\n[commands]\nlaunch = { program = "alpha/main::main", description = "Launch alpha" }\n',
    )
    home = tmp_path / "home"

    install_directory(source, home=home, env={})

    index = load_activation_index(home=home, env={})
    assert index.commands == {
        "launch": CommandRegistration("alpha", "alpha/main::main", "Launch alpha")
    }

    uninstall_package("alpha", home=home, env={})

    assert load_activation_index(home=home, env={}).commands == {}


@pytest.mark.parametrize("command_path", ("exec launch", "wsp launch"))
def test_install_refuses_builtin_and_alias_command_prefixes(
    tmp_path: Path, command_path: str
) -> None:
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        f'\n[commands]\n"{command_path}" = {{ program = "alpha/main::main" }}\n',
    )

    with pytest.raises(PackageInstallError, match="reserved"):
        install_directory(source, home=tmp_path / "home", env={})


def test_shadowed_command_is_restored_when_the_winning_package_is_uninstalled(
    tmp_path: Path,
) -> None:
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[commands]\nlaunch = { program = "alpha/main::main" }\n',
    )
    bravo = _package(
        tmp_path / "bravo",
        "bravo",
        "1.0.0",
        '\n[commands]\nlaunch = { program = "bravo/main::main" }\n',
    )
    home = tmp_path / "home"
    install_directory(alpha, home=home, env={})

    with pytest.raises(PackageInstallError, match="launch"):
        install_directory(bravo, home=home, env={})

    install_directory(bravo, home=home, env={}, shadow=True)

    assert load_activation_index(home=home, env={}).commands == {
        "launch": CommandRegistration("bravo", "bravo/main::main")
    }

    uninstall_package("bravo", home=home, env={})

    assert load_activation_index(home=home, env={}).commands == {
        "launch": CommandRegistration("alpha", "alpha/main::main")
    }


def test_rebuild_after_index_loss_preserves_a_shadow_winner_installed_in_reverse_name_order(
    tmp_path: Path,
) -> None:
    bravo = _package(
        tmp_path / "bravo",
        "bravo",
        "1.0.0",
        '\n[commands]\nlaunch = { program = "bravo/main::main" }\n',
    )
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[commands]\nlaunch = { program = "alpha/main::main" }\n',
    )
    home = tmp_path / "home"
    install_directory(bravo, home=home, env={})
    install_directory(alpha, home=home, env={}, shadow=True)

    index_path = home / ".agm" / "packages" / "index.toml"
    index_path.unlink()

    rebuilt = rebuild_activation_index(home=home, env={})

    assert rebuilt.commands == {"launch": CommandRegistration("alpha", "alpha/main::main")}


def test_rebuild_after_index_loss_refuses_corrupt_command_provenance(tmp_path: Path) -> None:
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        '\n[commands]\nlaunch = { program = "alpha/main::main" }\n',
    )
    home = tmp_path / "home"
    install_directory(source, home=home, env={})
    (home / ".agm" / "packages" / "index.toml").unlink()
    (home / ".agm" / "packages" / "alpha" / ".provenance" / "1.0.0.toml").write_text(
        "not valid = [", encoding="utf-8"
    )

    with pytest.raises(PackageActivationError, match="provenance"):
        rebuild_activation_index(home=home, env={})


def test_package_update_retains_a_displaced_owner_conflict_without_shadow(tmp_path: Path) -> None:
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[commands]\nlaunch = { program = "alpha/main::main" }\n',
    )
    bravo = _package(
        tmp_path / "bravo",
        "bravo",
        "1.0.0",
        '\n[commands]\nlaunch = { program = "bravo/main::main" }\n',
    )
    bravo_update = _package(
        tmp_path / "bravo-update",
        "bravo",
        "2.0.0",
        '\n[commands]\nlaunch = { program = "bravo/main::main" }\n',
    )
    home = tmp_path / "home"
    install_directory(alpha, home=home, env={})
    install_directory(bravo, home=home, env={}, shadow=True)

    with pytest.raises(PackageInstallError, match="launch"):
        install_directory(bravo_update, home=home, env={})

    install_directory(bravo_update, home=home, env={}, shadow=True)


def test_install_resolves_path_dependencies_before_activation(tmp_path: Path) -> None:
    bravo = _package(tmp_path / "bravo", "bravo", "1.2.0")
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1.1", path = "../bravo" }\n',
    )
    home = tmp_path / "home"

    install_directory(alpha, home=home, env={})

    index = load_activation_index(home=home, env={})
    assert set(index.packages) == {"alpha", "bravo"}
    assert index.packages["bravo"].version.major == 1
    assert (home / ".agm" / "packages" / "bravo" / "1.2.0") == (
        home / ".agm" / "packages" / "bravo" / "1.2.0"
    )
    assert bravo.is_dir()


def test_install_activates_an_installed_dependency_missing_from_the_index(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bravo = _package(
        tmp_path / "bravo",
        "bravo",
        "1.0.0",
        '\n[commands]\ninspect = { program = "bravo/main::main" }\n',
    )
    install_directory(bravo, home=home, env={})
    write_activation_index(ActivationIndex(), home=home, env={})
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = "1"\n',
    )

    install_directory(alpha, home=home, env={})

    assert load_activation_index(home=home, env={}).commands == {
        "inspect": CommandRegistration("bravo", "bravo/main::main")
    }


def test_install_rejects_an_inactive_dependency_command_conflicting_with_an_active_owner(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    bravo = _package(
        tmp_path / "bravo",
        "bravo",
        "1.0.0",
        '\n[commands]\nlaunch = { program = "bravo/main::main" }\n',
    )
    install_directory(bravo, home=home, env={})
    write_activation_index(ActivationIndex(), home=home, env={})
    charlie = _package(
        tmp_path / "charlie",
        "charlie",
        "1.0.0",
        '\n[commands]\nlaunch = { program = "charlie/main::main" }\n',
    )
    install_directory(charlie, home=home, env={})
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = "1"\n',
    )

    with pytest.raises(PackageInstallError, match="launch"):
        install_directory(alpha, home=home, env={})


def test_install_uses_an_installed_satisfying_dependency_before_path_source(tmp_path: Path) -> None:
    home = tmp_path / "home"
    install_directory(_package(tmp_path / "older", "bravo", "1.5.0"), home=home, env={})
    installed_dependency = _package(tmp_path / "installed", "bravo", "2.0.0")
    install_directory(installed_dependency, home=home, env={})
    install_directory(_package(tmp_path / "prerelease", "bravo", "2.0.0-alpha"), home=home, env={})
    _package(tmp_path / "path-bravo", "bravo", "1.0.0")
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1.0", path = "../path-bravo" }\n',
    )

    install_directory(alpha, home=home, env={})

    assert load_activation_index(home=home, env={}).packages["bravo"].version.major == 2


def test_install_activates_the_closure_of_a_stored_dependency(tmp_path: Path) -> None:
    home = tmp_path / "home"
    charlie = _package(tmp_path / "charlie", "charlie", "1.0.0")
    bravo = _package(
        tmp_path / "bravo",
        "bravo",
        "1.0.0",
        '\n[dependencies]\ncharlie = "1"\n',
    )
    install_directory(charlie, home=home, env={})
    install_directory(bravo, home=home, env={})
    write_activation_index(ActivationIndex(), home=home, env={})
    alpha = _package(tmp_path / "alpha", "alpha", "1.0.0", '\n[dependencies]\nbravo = "1"\n')

    install_directory(alpha, home=home, env={})

    assert set(load_activation_index(home=home, env={}).packages) == {"alpha", "bravo", "charlie"}


def test_install_accepts_an_active_editable_dependency(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bravo = _package(tmp_path / "bravo", "bravo", "1.0.0")
    alpha = _package(tmp_path / "alpha", "alpha", "1.0.0", '\n[dependencies]\nbravo = "1"\n')
    install_directory(bravo, home=home, env={}, editable=True)

    install_directory(alpha, home=home, env={})

    active = load_activation_index(home=home, env={}).packages["bravo"]
    assert active.editable == bravo.resolve()


def test_editable_install_mounts_live_tree_without_a_record(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    home = tmp_path / "home"

    installed = install_directory(source, home=home, env={}, editable=True)

    assert installed.root == source.resolve()
    assert not (source / "RECORD").exists()
    active = load_activation_index(home=home, env={}).packages["alpha"]
    assert active.editable == source.resolve()


def test_uninstall_verifies_record_then_removes_only_the_requested_version(tmp_path: Path) -> None:
    home = tmp_path / "home"
    one = install_directory(_package(tmp_path / "one", "alpha", "1.0.0"), home=home, env={})
    two = install_directory(_package(tmp_path / "two", "alpha", "2.0.0"), home=home, env={})

    uninstall_package("alpha", home=home, env={})

    assert not two.root.exists()
    assert one.root.exists()
    assert "alpha" not in load_activation_index(home=home, env={}).packages


def test_uninstall_refuses_tampered_record_without_removing_files(tmp_path: Path) -> None:
    home = tmp_path / "home"
    installed = install_directory(
        _package(tmp_path / "source", "alpha", "1.0.0"), home=home, env={}
    )
    module = installed.root / "alpha" / "main.agl"
    module.write_text("tampered", encoding="utf-8")

    with pytest.raises(PackageInstallError, match="integrity"):
        uninstall_package("alpha", home=home, env={})

    assert module.exists()


def test_uninstall_refuses_to_leave_an_active_package_with_unsatisfied_dependencies(
    tmp_path: Path,
) -> None:
    bravo = _package(tmp_path / "bravo", "bravo", "1.0.0")
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", path = "../bravo" }\n',
    )
    home = tmp_path / "home"
    install_directory(alpha, home=home, env={})

    with pytest.raises(PackageInstallError, match="bravo"):
        uninstall_package("bravo", home=home, env={})

    assert (home / ".agm" / "packages" / "bravo" / "1.0.0").is_dir()
    assert set(load_activation_index(home=home, env={}).packages) == {"alpha", "bravo"}
    assert bravo.is_dir()


def test_dry_run_uninstall_refuses_to_leave_unsatisfied_remaining_selection(tmp_path: Path) -> None:
    bravo = _package(tmp_path / "bravo", "bravo", "1.0.0")
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", path = "../bravo" }\n',
    )
    home = tmp_path / "home"
    install_directory(alpha, home=home, env={})
    dry_run.set_enabled(True)

    with pytest.raises(PackageInstallError, match="bravo"):
        uninstall_package("bravo", home=home, env={})

    assert (home / ".agm" / "packages" / "bravo" / "1.0.0").is_dir()
    assert set(load_activation_index(home=home, env={}).packages) == {"alpha", "bravo"}
    assert bravo.is_dir()


def test_uninstall_of_editable_package_only_removes_activation(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    home = tmp_path / "home"
    install_directory(source, home=home, env={}, editable=True)

    uninstall_package("alpha", home=home, env={})

    assert source.exists()
    assert "alpha" not in load_activation_index(home=home, env={}).packages


def test_install_rejects_invalid_and_linked_source_trees(tmp_path: Path) -> None:
    home = tmp_path / "home"
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    with pytest.raises(PackageInstallError):
        install_directory(invalid, home=home, env={})
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    linked = tmp_path / "linked"
    linked.symlink_to(source, target_is_directory=True)
    with pytest.raises(PackageInstallError):
        install_directory(linked, home=home, env={})


def test_install_and_uninstall_refuse_store_paths_redirected_outside_the_store(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    store = home / ".agm" / "packages"
    external = tmp_path / "external"
    external.mkdir()
    store.mkdir(parents=True)
    (store / "alpha").symlink_to(external, target_is_directory=True)

    with pytest.raises(PackageInstallError, match="store"):
        install_directory(source, home=home, env={})

    (store / "alpha").unlink()
    installed = install_directory(source, home=home, env={})
    external_version = external / "1.0.0"
    installed.root.rename(external_version)
    provenance = store / "alpha" / ".provenance" / "1.0.0.toml"
    provenance.unlink()
    provenance.parent.rmdir()
    (store / "alpha").rmdir()
    (store / "alpha").symlink_to(external, target_is_directory=True)

    with pytest.raises(PackageInstallError, match="store"):
        uninstall_package("alpha", home=home, env={})

    assert external_version.is_dir()
    assert "alpha" in load_activation_index(home=home, env={}).packages


def test_install_refuses_an_in_store_symlinked_package_path(tmp_path: Path) -> None:
    home = tmp_path / "home"
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    store = home / ".agm" / "packages"
    internal_target = store / "other-package"
    internal_target.mkdir(parents=True)
    (store / "alpha").symlink_to(internal_target, target_is_directory=True)

    with pytest.raises(PackageInstallError, match="symlink"):
        install_directory(source, home=home, env={})

    assert not (internal_target / "1.0.0").exists()
    assert load_activation_index(home=home, env={}).packages == {}


def test_install_refuses_cyclic_and_mismatched_path_dependencies(tmp_path: Path) -> None:
    home = tmp_path / "home"
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
    with pytest.raises(PackageInstallError, match="cyclic"):
        install_directory(alpha, home=home, env={})

    wrong = _package(tmp_path / "wrong", "wrong", "1.0.0")
    mismatched = _package(
        tmp_path / "mismatched",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", path = "../wrong" }\n',
    )
    with pytest.raises(PackageInstallError, match="bravo"):
        install_directory(mismatched, home=home, env={})
    assert wrong.is_dir()


def test_install_refuses_tampered_existing_tree_and_cleans_failed_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    installed = install_directory(source, home=home, env={})
    (installed.root / "alpha" / "main.agl").write_text("tampered", encoding="utf-8")
    with pytest.raises(PackageInstallError, match="integrity"):
        install_directory(source, home=home, env={})

    fresh = _package(tmp_path / "fresh", "bravo", "1.0.0")

    def fail_copy(_: Path, destination: Path, **__: object) -> None:
        destination.mkdir()
        raise OSError("full")

    monkeypatch.setattr(package_install.fs, "copy_tree", fail_copy)
    with pytest.raises(PackageInstallError, match="cannot install"):
        install_directory(fresh, home=home, env={})
    assert not (home / ".agm" / "packages" / "bravo" / "1.0.0").exists()

    another = _package(tmp_path / "another", "charlie", "1.0.0")
    monkeypatch.setattr(
        package_install.fs,
        "copy_tree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("full")),
    )
    with pytest.raises(PackageInstallError, match="cannot install"):
        install_directory(another, home=home, env={})


def test_install_rejects_changed_directory_content_for_an_existing_identity(tmp_path: Path) -> None:
    home = tmp_path / "home"
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    installed = install_directory(source, home=home, env={})
    changed_source = "program def main() -> unit = ()\n// changed\n"
    (source / "alpha" / "main.agl").write_text(changed_source, encoding="utf-8")

    with pytest.raises(PackageInstallError, match="content"):
        install_directory(source, home=home, env={})

    assert (installed.root / "alpha" / "main.agl").read_text(encoding="utf-8") != changed_source


def test_store_dependency_integrity_error_is_reported(tmp_path: Path) -> None:
    home = tmp_path / "home"
    installed = install_directory(_package(tmp_path / "bravo", "bravo", "1.0.0"), home=home, env={})
    (installed.root / "bravo" / "main.agl").write_text("tampered", encoding="utf-8")
    source = _package(tmp_path / "alpha", "alpha", "1.0.0", '\n[dependencies]\nbravo = "1"\n')
    with pytest.raises(PackageInstallError, match="integrity"):
        install_directory(source, home=home, env={})


def test_install_archive_verifies_extracts_records_and_activates_it(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    archive = tmp_path / "alpha.agmpkg"
    write_archive(source, archive)

    installed = install_archive(archive, home=tmp_path / "home", env={})

    assert installed.root == tmp_path / "home" / ".agm" / "packages" / "alpha" / "1.0.0"
    assert verify_record(installed.root)
    assert load_activation_index(home=tmp_path / "home", env={}).packages["alpha"].version == (
        installed.manifest.version
    )


def test_archive_command_lifecycle_restores_the_remaining_owner(tmp_path: Path) -> None:
    alpha = _package(
        tmp_path / "alpha-source",
        "alpha",
        "1.0.0",
        '\n[commands]\nlaunch = { program = "alpha/main::main" }\n',
    )
    bravo = _package(
        tmp_path / "bravo-source",
        "bravo",
        "1.0.0",
        '\n[commands]\nlaunch = { program = "bravo/main::main" }\n',
    )
    alpha_archive = tmp_path / "alpha.agmpkg"
    bravo_archive = tmp_path / "bravo.agmpkg"
    write_archive(alpha, alpha_archive)
    write_archive(bravo, bravo_archive)
    home = tmp_path / "home"
    install_archive(alpha_archive, home=home, env={})

    with pytest.raises(PackageInstallError, match="launch"):
        install_archive(bravo_archive, home=home, env={})

    install_archive(bravo_archive, home=home, env={}, shadow=True)
    assert load_activation_index(home=home, env={}).commands == {
        "launch": CommandRegistration("bravo", "bravo/main::main")
    }

    uninstall_package("bravo", home=home, env={})
    assert load_activation_index(home=home, env={}).commands == {
        "launch": CommandRegistration("alpha", "alpha/main::main")
    }


def test_editable_command_lifecycle_restores_the_remaining_owner(tmp_path: Path) -> None:
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[commands]\nlaunch = { program = "alpha/main::main" }\n',
    )
    bravo = _package(
        tmp_path / "bravo",
        "bravo",
        "1.0.0",
        '\n[commands]\nlaunch = { program = "bravo/main::main" }\n',
    )
    home = tmp_path / "home"
    install_directory(alpha, home=home, env={})
    install_directory(bravo, home=home, env={}, editable=True, shadow=True)

    assert load_activation_index(home=home, env={}).commands == {
        "launch": CommandRegistration("bravo", "bravo/main::main")
    }

    uninstall_package("bravo", home=home, env={})

    assert bravo.exists()
    assert load_activation_index(home=home, env={}).commands == {
        "launch": CommandRegistration("alpha", "alpha/main::main")
    }


def test_install_archive_reuses_an_existing_verified_tree(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    archive = tmp_path / "alpha.agmpkg"
    write_archive(source, archive)
    first = install_archive(archive, home=tmp_path / "home", env={})

    second = install_archive(archive, home=tmp_path / "home", env={})

    assert second == first


def test_install_archive_rejects_a_different_content_hash_for_an_existing_identity(
    tmp_path: Path,
) -> None:
    first_source = _package(tmp_path / "first-source", "alpha", "1.0.0")
    second_source = _package(tmp_path / "second-source", "alpha", "1.0.0")
    (second_source / "alpha" / "main.agl").write_text(
        "program def main() -> unit = ()\n// different\n", encoding="utf-8"
    )
    first_archive = tmp_path / "first.agmpkg"
    second_archive = tmp_path / "second.agmpkg"
    write_archive(first_source, first_archive)
    write_archive(second_source, second_archive)
    home = tmp_path / "home"

    install_archive(first_archive, home=home, env={})

    with pytest.raises(PackageInstallError, match="content"):
        install_archive(second_archive, home=home, env={})

    assert (home / ".agm" / "packages" / "alpha" / "1.0.0" / "alpha" / "main.agl").read_text(
        encoding="utf-8"
    ) == "program def main() -> unit = ()\n"


def test_install_archive_does_not_reopen_an_archive_after_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    replacement_source = _package(tmp_path / "replacement-source", "alpha", "1.0.0")
    (replacement_source / "alpha" / "main.agl").write_text(
        "program def main() -> unit = ()\n// replacement\n", encoding="utf-8"
    )
    archive = tmp_path / "alpha.agmpkg"
    replacement = tmp_path / "replacement.agmpkg"
    write_archive(source, archive)
    write_archive(replacement_source, replacement)

    def replace_after_verification(path: Path) -> object:
        path.write_bytes(replacement.read_bytes())
        raise AssertionError("archive verification must be bound to extraction")

    monkeypatch.setattr(package_archive, "verify_archive", replace_after_verification)

    install_archive(archive, home=tmp_path / "home", env={})

    installed_module = (
        tmp_path / "home" / ".agm" / "packages" / "alpha" / "1.0.0" / "alpha" / "main.agl"
    )
    assert installed_module.read_text(encoding="utf-8") == "program def main() -> unit = ()\n"


def test_dry_run_archive_install_reports_a_verification_failure_without_writing(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "invalid.agmpkg"
    archive.write_bytes(b"not an archive")
    dry_run.set_enabled(True)

    with pytest.raises(PackageInstallError, match="archive"):
        install_archive(archive, home=tmp_path / "home", env={})

    assert not (tmp_path / "home").exists()


def test_dry_run_archive_install_verifies_without_writing_and_reports_its_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    archive = tmp_path / "alpha.agmpkg"
    write_archive(source, archive)
    dry_run.set_enabled(True)

    installed = install_archive(archive, home=tmp_path / "home", env={})

    assert installed.manifest.name == "alpha"
    assert not (tmp_path / "home").exists()
    assert "dry-run: agm install-package-archive" in capsys.readouterr().out


def test_dry_run_archive_install_validates_an_existing_verified_tree(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    archive = tmp_path / "alpha.agmpkg"
    write_archive(source, archive)
    home = tmp_path / "home"
    installed = install_archive(archive, home=home, env={})
    activation = (home / ".agm" / "packages" / "index.toml").read_text(encoding="utf-8")
    dry_run.set_enabled(True)

    assert install_archive(archive, home=home, env={}) == installed
    assert (home / ".agm" / "packages" / "index.toml").read_text(encoding="utf-8") == activation


def test_dry_run_archive_install_rejects_a_content_conflict_with_existing_tree(
    tmp_path: Path,
) -> None:
    first_source = _package(tmp_path / "first-source", "alpha", "1.0.0")
    second_source = _package(tmp_path / "second-source", "alpha", "1.0.0")
    (second_source / "alpha" / "main.agl").write_text(
        "program def main() -> unit = ()\n// different\n", encoding="utf-8"
    )
    first_archive = tmp_path / "first.agmpkg"
    second_archive = tmp_path / "second.agmpkg"
    write_archive(first_source, first_archive)
    write_archive(second_source, second_archive)
    home = tmp_path / "home"
    install_archive(first_archive, home=home, env={})
    dry_run.set_enabled(True)

    with pytest.raises(PackageInstallError, match="content"):
        install_archive(second_archive, home=home, env={})


def test_dry_run_archive_install_rejects_a_tampered_existing_tree(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    archive = tmp_path / "alpha.agmpkg"
    write_archive(source, archive)
    home = tmp_path / "home"
    installed = install_archive(archive, home=home, env={})
    (installed.root / "alpha" / "main.agl").write_text("tampered", encoding="utf-8")
    dry_run.set_enabled(True)

    with pytest.raises(PackageInstallError, match="integrity"):
        install_archive(archive, home=home, env={})


def test_install_archive_wraps_an_unsatisfied_dependency_error(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0", '\n[dependencies]\nbravo = "1"\n')
    archive = tmp_path / "alpha.agmpkg"
    write_archive(source, archive)

    with pytest.raises(PackageInstallError, match="bravo"):
        install_archive(archive, home=tmp_path / "home", env={})


def test_archive_staging_is_not_visible_to_package_store_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    archive = tmp_path / "alpha.agmpkg"
    write_archive(source, archive)
    extract = package_install.extract_archive

    def extract_and_scan(path: Path, destination: Path) -> object:
        assert installed_packages(home=home, env={}) == ()
        return extract(path, destination)

    monkeypatch.setattr(package_install, "extract_archive", extract_and_scan)

    installed = install_archive(archive, home=home, env={})

    assert installed.manifest.name == "alpha"
    assert not list((home / ".agm").glob(".agm-package-*"))


def test_dry_run_archive_install_validates_discipline_without_writing(tmp_path: Path) -> None:
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        '\n[commands]\nlaunch = { program = "alpha/main::missing" }\n',
    )
    archive = tmp_path / "alpha.agmpkg"
    write_archive(source, archive)
    dry_run.set_enabled(True)

    with pytest.raises(PackageInstallError, match="program"):
        install_archive(archive, home=tmp_path / "home", env={})

    assert not (tmp_path / "home").exists()


def test_dry_run_archive_install_rejects_a_missing_module_tree_without_writing(
    tmp_path: Path,
) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    (source / "alpha" / "main.agl").unlink()
    (source / "alpha").rmdir()
    archive = tmp_path / "alpha.agmpkg"
    write_archive(source, archive)
    dry_run.set_enabled(True)

    with pytest.raises(PackageInstallError, match="module tree"):
        install_archive(archive, home=tmp_path / "home", env={})

    assert not (tmp_path / "home").exists()


def test_dry_run_archive_install_rejects_an_invalid_module_path_without_writing(
    tmp_path: Path,
) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    (source / "alpha" / "invalid-name.agl").write_text(
        "program def invalid() -> unit = ()\n", encoding="utf-8"
    )
    archive = tmp_path / "alpha.agmpkg"
    write_archive(source, archive)
    dry_run.set_enabled(True)

    with pytest.raises(PackageInstallError, match="invalid module path"):
        install_archive(archive, home=tmp_path / "home", env={})

    assert not (tmp_path / "home").exists()


def test_install_archive_refuses_tampering_without_publishing_a_tree(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    archive = tmp_path / "alpha.agmpkg"
    write_archive(source, archive)
    corrupted = bytearray(archive.read_bytes())
    corrupted[len(corrupted) // 2] ^= 1
    archive.write_bytes(corrupted)

    with pytest.raises(PackageInstallError, match="archive"):
        install_archive(archive, home=tmp_path / "home", env={})

    assert not (tmp_path / "home" / ".agm" / "packages" / "alpha").exists()


def test_url_dependency_hands_verified_archive_to_the_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bravo = _package(tmp_path / "bravo", "bravo", "1.0.0")
    archive = tmp_path / "bravo.agmpkg"
    write_archive(bravo, archive)
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", url = "https://example.test/bravo.agmpkg", '
        'hash = "sha256=' + "0" * 64 + '" }\n',
    )
    fetched: list[str] = []

    def fake_fetch(**kwargs: object) -> None:
        fetched.append(str(kwargs["requirement"]))
        handoff = kwargs["handoff"]
        assert callable(handoff)
        handoff(archive)

    monkeypatch.setattr(package_install, "fetch_archive", fake_fetch)
    installed = install_directory(source, home=tmp_path / "home", env={})

    assert installed.manifest.name == "alpha"
    assert fetched == ["bravo >= 1.0.0"]
    assert set(load_activation_index(home=tmp_path / "home", env={}).packages) == {"alpha", "bravo"}


def test_archive_install_wraps_activation_and_extraction_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    archive = tmp_path / "alpha.agmpkg"
    write_archive(source, archive)
    monkeypatch.setattr(
        package_install,
        "load_activation_index",
        lambda **_: (_ for _ in ()).throw(PackageActivationError("broken")),
    )

    with pytest.raises(PackageInstallError, match="activation"):
        install_archive(archive, home=tmp_path / "home", env={})

    monkeypatch.setattr(
        package_install,
        "load_activation_index",
        lambda **_: load_activation_index(home=tmp_path / "other", env={}),
    )
    monkeypatch.setattr(
        package_install,
        "extract_archive",
        lambda *_: (_ for _ in ()).throw(package_install.ArchiveError("broken")),
    )

    with pytest.raises(PackageInstallError, match="archive"):
        install_archive(archive, home=tmp_path / "home", env={})


def test_failed_activation_does_not_leave_package_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    home = tmp_path / "home"
    monkeypatch.setattr(
        package_install,
        "write_activation_index",
        lambda *_, **__: (_ for _ in ()).throw(PackageActivationError("broken")),
    )

    with pytest.raises(PackageInstallError, match="activation"):
        install_directory(source, home=home, env={})

    assert not (home / ".agm" / "packages" / "alpha" / ".provenance" / "1.0.0.toml").exists()


def test_failed_activation_restores_existing_package_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    home = tmp_path / "home"
    install_directory(source, home=home, env={})
    provenance = home / ".agm" / "packages" / "alpha" / ".provenance" / "1.0.0.toml"
    original = provenance.read_bytes()
    monkeypatch.setattr(
        package_install,
        "write_activation_index",
        lambda *_, **__: (_ for _ in ()).throw(PackageActivationError("broken")),
    )

    with pytest.raises(PackageInstallError, match="activation"):
        install_directory(source, home=home, env={})

    assert provenance.read_bytes() == original


def test_active_editable_manifest_errors_are_install_errors(
    tmp_path: Path,
) -> None:
    editable = tmp_path / "editable"
    editable.mkdir()
    (editable / "package.toml").write_text("invalid", encoding="utf-8")
    state = package_install._InstallState(
        home=tmp_path / "home",
        env={},
        index=ActivationIndex(
            {"alpha": ActivePackage(semver.Version.parse("1.0.0"), editable=editable)}, {}
        ),
    )

    with pytest.raises(PackageInstallError, match="active editable"):
        package_install._installed_satisfying(
            "alpha", package_install.DependencySpec(semver.Version.parse("1.0.0")), state
        )


def test_provenance_snapshot_and_restore_wrap_io_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = ActivationIndex({"alpha": ActivePackage(semver.Version.parse("1.0.0"))}, {})
    provenance = tmp_path / "home" / ".agm" / "packages" / "alpha" / ".provenance" / "1.0.0.toml"
    provenance.parent.mkdir(parents=True)
    provenance.write_text("original", encoding="utf-8")
    monkeypatch.setattr(Path, "read_bytes", lambda _: (_ for _ in ()).throw(OSError("broken")))

    with pytest.raises(PackageActivationError, match="snapshot"):
        package_install._snapshot_package_provenance(index, home=tmp_path / "home", env={})

    path = tmp_path / "provenance.toml"
    monkeypatch.setattr(
        Path, "unlink", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("broken"))
    )
    with pytest.raises(PackageActivationError, match="restore"):
        package_install._restore_package_provenance({path: None})


def test_fetch_and_activation_failures_become_package_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    monkeypatch.setattr(
        package_install,
        "load_activation_index",
        lambda **_: (_ for _ in ()).throw(PackageActivationError("broken")),
    )
    with pytest.raises(PackageInstallError, match="activation"):
        install_directory(source, home=tmp_path / "home", env={})

    monkeypatch.setattr(
        package_install,
        "load_activation_index",
        lambda **_: load_activation_index(home=tmp_path / "other", env={}),
    )
    monkeypatch.setattr(
        package_install,
        "write_activation_index",
        lambda *_, **__: (_ for _ in ()).throw(PackageActivationError("broken")),
    )
    with pytest.raises(PackageInstallError, match="activation"):
        install_directory(source, home=tmp_path / "home", env={})
    assert not (tmp_path / "home" / ".agm" / "packages" / "alpha" / "1.0.0").exists()


def test_installed_package_enumeration_rejects_invalid_store_identity(tmp_path: Path) -> None:
    home = tmp_path / "home"
    assert installed_packages(home=home, env={}) == ()
    root = home / ".agm" / "packages" / "alpha" / "1.0.0"
    (root / "alpha").mkdir(parents=True)
    (root / "package.toml").write_text(
        '[package]\nname = "bravo"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    with pytest.raises(PackageInstallError, match="identity"):
        installed_packages(home=home, env={})


def test_uninstall_reports_unknown_and_activation_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(PackageInstallError, match="not installed"):
        uninstall_package("alpha", home=tmp_path / "home", env={})
    monkeypatch.setattr(
        package_install,
        "load_activation_index",
        lambda **_: (_ for _ in ()).throw(PackageActivationError("broken")),
    )
    with pytest.raises(PackageInstallError, match="activation"):
        uninstall_package("alpha", home=tmp_path / "home", env={})


def test_uninstall_reports_activation_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    home = tmp_path / "home"
    install_directory(source, home=home, env={})
    monkeypatch.setattr(
        package_install,
        "write_activation_index",
        lambda *_, **__: (_ for _ in ()).throw(PackageActivationError("broken")),
    )
    with pytest.raises(PackageInstallError, match="activation"):
        uninstall_package("alpha", home=home, env={})


def test_installed_package_enumeration_skips_non_package_entries_and_bad_manifests(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    store = home / ".agm" / "packages"
    store.mkdir(parents=True)
    (store / "index.toml").write_text("", encoding="utf-8")
    (store / "alpha").mkdir()
    (store / "alpha" / "notes").write_text("not a version", encoding="utf-8")
    assert installed_packages(home=home, env={}) == ()
    broken = store / "alpha" / "1.0.0"
    broken.mkdir()
    (broken / "package.toml").write_text("[package\n", encoding="utf-8")
    with pytest.raises(PackageInstallError, match="cannot load"):
        installed_packages(home=home, env={})


def test_install_refuses_unsatisfied_and_different_existing_manifest(tmp_path: Path) -> None:
    home = tmp_path / "home"
    unsatisfied = _package(
        tmp_path / "unsatisfied", "alpha", "1.0.0", '\n[dependencies]\nbravo = "1"\n'
    )
    with pytest.raises(PackageInstallError, match="bravo"):
        install_directory(unsatisfied, home=home, env={})

    source = _package(tmp_path / "source", "bravo", "1.0.0")
    install_directory(source, home=home, env={})
    (source / "package.toml").write_text(
        '[package]\nname = "bravo"\nversion = "1.0.0"\ndescription = "changed"\n',
        encoding="utf-8",
    )
    with pytest.raises(PackageInstallError, match="disagrees"):
        install_directory(source, home=home, env={})


def test_dry_run_install_validates_the_resulting_activation_without_writing(tmp_path: Path) -> None:
    home = tmp_path / "home"
    install_directory(_package(tmp_path / "bravo-two", "bravo", "2.0.0"), home=home, env={})
    install_directory(
        _package(
            tmp_path / "alpha",
            "alpha",
            "1.0.0",
            '\n[dependencies]\nbravo = "2"\n',
        ),
        home=home,
        env={},
    )
    dry_run.set_enabled(True)

    with pytest.raises(PackageInstallError, match="bravo"):
        install_directory(_package(tmp_path / "bravo-one", "bravo", "1.0.0"), home=home, env={})

    assert not (home / ".agm" / "packages" / "bravo" / "1.0.0").exists()
    assert load_activation_index(home=home, env={}).packages["bravo"].version.major == 2


def test_dry_run_install_selects_prior_transient_dependencies(tmp_path: Path) -> None:
    bravo = _package(tmp_path / "bravo", "bravo", "1.0.0")
    _package(
        tmp_path / "charlie",
        "charlie",
        "1.0.0",
        '\n[dependencies]\nbravo = "1"\n',
    )
    alpha = _package(
        tmp_path / "alpha",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", path = "../bravo" }\n'
        'charlie = { version = "1", path = "../charlie" }\n',
    )
    home = tmp_path / "home"
    dry_run.set_enabled(True)

    installed = install_directory(alpha, home=home, env={})

    assert installed.manifest.name == "alpha"
    assert bravo.is_dir()
    assert not home.exists()

    dry_run.set_enabled(False)
    actual = install_directory(alpha, home=home, env={})

    assert actual == installed
    assert set(load_activation_index(home=home, env={}).packages) == {"alpha", "bravo", "charlie"}


def test_dry_run_url_dependency_never_fetches_or_creates_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", url = "https://example.test/bravo.agmpkg", '
        'hash = "sha256=' + "0" * 64 + '" }\n',
    )
    fetched = False

    def fail_fetch(**_: object) -> None:
        nonlocal fetched
        fetched = True

    monkeypatch.setattr(package_install, "fetch_archive", fail_fetch)
    dry_run.set_enabled(True)

    with pytest.raises(PackageInstallError, match=r"URL package.*bravo >= 1\.0\.0"):
        install_directory(source, home=tmp_path / "dry-home", env={})

    assert not fetched
    assert not (tmp_path / "dry-home").exists()


def test_url_fetch_refuses_when_the_fetch_handoff_does_not_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", url = "https://example.test/bravo.agmpkg", '
        'hash = "sha256=' + "0" * 64 + '" }\n',
    )
    monkeypatch.setattr(package_install, "fetch_archive", lambda **_: None)

    with pytest.raises(PackageInstallError, match="archive was not installed"):
        install_directory(source, home=tmp_path / "home", env={})


def test_url_fetch_scratch_creation_failure_names_the_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", url = "https://example.test/bravo.agmpkg", '
        'hash = "sha256=' + "0" * 64 + '" }\n',
    )
    monkeypatch.setattr(
        package_install.fs,
        "mkdir",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("full")),
    )

    with pytest.raises(PackageInstallError, match=r"fetch failed.*bravo >= 1\.0\.0"):
        install_directory(source, home=tmp_path / "home", env={})


def test_fetch_failure_and_dry_run_use_clean_dependency_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _package(
        tmp_path / "source",
        "alpha",
        "1.0.0",
        '\n[dependencies]\nbravo = { version = "1", url = "https://example.test/bravo.agmpkg", '
        'hash = "sha256=' + "0" * 64 + '" }\n',
    )
    monkeypatch.setattr(
        package_install,
        "fetch_archive",
        lambda **_: (_ for _ in ()).throw(
            package_install.FetchError("fetch failed for bravo >= 1.0.0: offline")
        ),
    )
    with pytest.raises(PackageInstallError, match=r"fetch failed.*bravo >= 1\.0\.0"):
        install_directory(source, home=tmp_path / "home", env={})

    dry_run.set_enabled(True)
    with pytest.raises(PackageInstallError, match=r"URL package.*bravo >= 1\.0\.0"):
        install_directory(source, home=tmp_path / "dry-home", env={})


def test_install_and_uninstall_use_dry_run_filesystem_primitives(tmp_path: Path) -> None:
    source = _package(tmp_path / "source", "alpha", "1.0.0")
    home = tmp_path / "home"
    dry_run.set_enabled(True)

    installed = install_directory(source, home=home, env={})

    assert not installed.root.exists()
    dry_run.set_enabled(False)
    install_directory(source, home=home, env={})
    dry_run.set_enabled(True)
    uninstall_package("alpha", home=home, env={})

    assert (home / ".agm" / "packages" / "alpha" / "1.0.0").exists()
