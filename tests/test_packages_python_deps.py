"""Syncing active packages' Python requirements into AGM's interpreter environment."""

from __future__ import annotations

from pathlib import Path

import pytest
import semver

from agm.core import dry_run, process
from agm.packages.activation import load_activation_index, resolve_indexed_packages
from agm.packages.archive import write_archive
from agm.packages.errors import PackageInstallError
from agm.packages.install import sync_active_python_dependencies, uninstall_package
from agm.packages.python_deps import python_dependencies, sync_python_dependencies
from agm.packages.store import installed_packages, package_store_path
from tests._package_helpers import (
    PythonInstaller,
    install_archive,
    install_directory,
    write_python_package,
)

# ``packaging`` is an AGM runtime dependency, so it is always installed; the
# ``agm-test-absent-*`` distributions never are.
_PRESENT = "packaging>=1"
_ABSENT_A = "agm-test-absent-a>=1"
_ABSENT_B = "agm-test-absent-b"
_SHARED = "agm-test-absent-shared>=2"


def test_required_dependencies_are_the_union_over_store_and_editable_packages(
    tmp_path: Path, python_installer: PythonInstaller
) -> None:
    home = python_installer.home
    install_directory(
        write_python_package(tmp_path / "zulu", "zulu", _ABSENT_A, _SHARED), home=home, env={}
    )
    install_directory(
        write_python_package(tmp_path / "alpha", "alpha", _SHARED, _ABSENT_B),
        home=home,
        env={},
        editable=True,
    )

    index = load_activation_index(home=home, env={})

    assert python_dependencies(resolve_indexed_packages(index, home=home, env={})) == (
        _SHARED,
        _ABSENT_B,
        _ABSENT_A,
    )


def test_an_unsatisfied_requirement_installs_the_whole_union_jointly(
    tmp_path: Path, python_installer: PythonInstaller
) -> None:
    install_directory(
        write_python_package(tmp_path / "alpha", "alpha", _PRESENT, _ABSENT_A),
        home=python_installer.home,
        env={},
    )

    assert python_installer.specs == [[_PRESENT, _ABSENT_A]]


def test_opposing_requirements_reach_the_installer_together(
    tmp_path: Path, python_installer: PythonInstaller
) -> None:
    home = python_installer.home
    install_directory(
        write_python_package(tmp_path / "alpha", "alpha", _PRESENT), home=home, env={}
    )
    install_directory(
        write_python_package(tmp_path / "bravo", "bravo", "packaging<1"), home=home, env={}
    )

    assert python_installer.specs == [[_PRESENT, "packaging<1"]]


def test_install_syncs_every_active_package_requirement(
    tmp_path: Path, python_installer: PythonInstaller
) -> None:
    home = python_installer.home
    install_directory(
        write_python_package(tmp_path / "alpha", "alpha", _ABSENT_A), home=home, env={}
    )
    install_directory(
        write_python_package(tmp_path / "bravo", "bravo", _ABSENT_B),
        home=home,
        env={},
        editable=True,
    )

    assert python_installer.specs == [[_ABSENT_A], [_ABSENT_A, _ABSENT_B]]


def test_install_without_unsatisfied_requirements_runs_no_installer(
    tmp_path: Path, python_installer: PythonInstaller
) -> None:
    install_directory(
        write_python_package(tmp_path / "alpha", "alpha"), home=python_installer.home, env={}
    )
    install_directory(
        write_python_package(tmp_path / "bravo", "bravo", _PRESENT),
        home=python_installer.home,
        env={},
    )

    assert python_installer.runs == []


def test_install_syncs_before_publishing_the_activation(
    tmp_path: Path, python_installer: PythonInstaller
) -> None:
    home = python_installer.home
    install_directory(
        write_python_package(tmp_path / "alpha", "alpha", _ABSENT_A), home=home, env={}
    )
    install_directory(
        write_python_package(tmp_path / "bravo", "bravo", _ABSENT_B), home=home, env={}
    )

    assert python_installer.active_during_run == [set(), {"alpha"}]
    assert set(load_activation_index(home=home, env={}).packages) == {"alpha", "bravo"}


@pytest.mark.parametrize("editable", [False, True])
def test_failed_python_install_rolls_the_store_back(
    tmp_path: Path, python_installer: PythonInstaller, editable: bool
) -> None:
    home = python_installer.home
    install_directory(write_python_package(tmp_path / "alpha", "alpha"), home=home, env={})
    before_index = load_activation_index(home=home, env={})
    before_store = installed_packages(home=home, env={})
    python_installer.returncode = 1

    with pytest.raises(PackageInstallError):
        install_directory(
            write_python_package(tmp_path / "bravo", "bravo", _ABSENT_B),
            home=home,
            env={},
            editable=editable,
        )

    assert python_installer.specs == [[_ABSENT_B]]
    assert load_activation_index(home=home, env={}) == before_index
    assert installed_packages(home=home, env={}) == before_store
    assert not package_store_path("bravo", semver.Version(1), home=home, env={}).exists()


def test_interrupted_python_install_rolls_the_store_back(
    tmp_path: Path, python_installer: PythonInstaller
) -> None:
    home = python_installer.home
    install_directory(write_python_package(tmp_path / "alpha", "alpha"), home=home, env={})
    before_index = load_activation_index(home=home, env={})
    python_installer.interrupt = True

    with pytest.raises(KeyboardInterrupt):
        install_directory(
            write_python_package(tmp_path / "bravo", "bravo", _ABSENT_B), home=home, env={}
        )

    assert load_activation_index(home=home, env={}) == before_index
    assert not package_store_path("bravo", semver.Version(1), home=home, env={}).exists()


def test_unstartable_installer_fails_the_install(
    tmp_path: Path, python_installer: PythonInstaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unstartable(cmd: list[str], **_kwargs: object) -> int:
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(process, "run_foreground", unstartable)

    with pytest.raises(PackageInstallError):
        install_directory(
            write_python_package(tmp_path / "alpha", "alpha", _ABSENT_A),
            home=python_installer.home,
            env={},
        )

    assert load_activation_index(home=python_installer.home, env={}).packages == {}


def test_dry_run_install_reports_the_requirements_without_installing(
    tmp_path: Path, python_installer: PythonInstaller, capsys: pytest.CaptureFixture[str]
) -> None:
    source = write_python_package(tmp_path / "alpha", "alpha", _PRESENT, _ABSENT_A)
    dry_run.set_enabled(True)

    install_directory(source, home=python_installer.home, env={})

    out = capsys.readouterr().out
    assert _ABSENT_A in out
    assert _PRESENT in out
    assert python_installer.runs == []
    assert not (python_installer.home / ".agm").exists()


def test_dry_run_archive_install_reports_the_requirements_without_installing(
    tmp_path: Path, python_installer: PythonInstaller, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "alpha.agmpkg"
    write_archive(write_python_package(tmp_path / "alpha", "alpha", _ABSENT_A), archive)
    capsys.readouterr()
    dry_run.set_enabled(True)

    install_archive(archive, home=python_installer.home, env={})

    assert _ABSENT_A in capsys.readouterr().out
    assert python_installer.runs == []
    assert not (python_installer.home / ".agm").exists()


def test_archive_install_syncs_the_archived_requirements(
    tmp_path: Path, python_installer: PythonInstaller
) -> None:
    archive = tmp_path / "alpha.agmpkg"
    write_archive(write_python_package(tmp_path / "alpha", "alpha", _ABSENT_A), archive)

    install_archive(archive, home=python_installer.home, env={})

    assert python_installer.specs == [[_ABSENT_A]]
    assert "alpha" in load_activation_index(home=python_installer.home, env={}).packages


def test_failed_python_install_rolls_an_archive_install_back(
    tmp_path: Path, python_installer: PythonInstaller
) -> None:
    archive = tmp_path / "alpha.agmpkg"
    write_archive(write_python_package(tmp_path / "alpha", "alpha", _ABSENT_A), archive)
    python_installer.returncode = 1

    with pytest.raises(PackageInstallError):
        install_archive(archive, home=python_installer.home, env={})

    assert load_activation_index(home=python_installer.home, env={}).packages == {}
    assert not package_store_path(
        "alpha", semver.Version(1), home=python_installer.home, env={}
    ).exists()


def test_uninstall_leaves_python_packages_alone(
    tmp_path: Path, python_installer: PythonInstaller
) -> None:
    home = python_installer.home
    install_directory(
        write_python_package(tmp_path / "alpha", "alpha", _ABSENT_A), home=home, env={}
    )
    python_installer.runs.clear()

    uninstall_package("alpha", home=home, env={})

    assert python_installer.runs == []


def test_sync_installs_the_whole_union_and_reports_the_unsatisfied(
    python_installer: PythonInstaller,
) -> None:
    assert sync_python_dependencies((_PRESENT, _ABSENT_A)) == (_ABSENT_A,)
    assert python_installer.specs == [[_PRESENT, _ABSENT_A]]


def test_sync_of_satisfied_requirements_runs_no_installer(
    python_installer: PythonInstaller,
) -> None:
    assert sync_python_dependencies((_PRESENT,)) == ()
    assert python_installer.runs == []


def test_active_sync_installs_the_whole_active_union_and_reports_the_unsatisfied(
    tmp_path: Path, python_installer: PythonInstaller
) -> None:
    home = python_installer.home
    install_directory(
        write_python_package(tmp_path / "alpha", "alpha", _PRESENT, _ABSENT_A), home=home, env={}
    )
    install_directory(
        write_python_package(tmp_path / "bravo", "bravo", _ABSENT_B),
        home=home,
        env={},
        editable=True,
    )
    python_installer.runs.clear()

    assert sync_active_python_dependencies(home=home, env={}) == (_ABSENT_A, _ABSENT_B)
    assert python_installer.specs == [[_PRESENT, _ABSENT_A, _ABSENT_B]]


def test_active_sync_of_an_unresolvable_activation_is_a_package_install_error(
    tmp_path: Path, python_installer: PythonInstaller
) -> None:
    home = python_installer.home
    source = write_python_package(tmp_path / "alpha", "alpha", _ABSENT_A)
    install_directory(source, home=home, env={}, editable=True)
    python_installer.runs.clear()
    (source / "package.toml").unlink()

    with pytest.raises(PackageInstallError):
        sync_active_python_dependencies(home=home, env={})
    assert python_installer.runs == []
