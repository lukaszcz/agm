"""Syncing active packages' Python requirements into AGM's interpreter environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
import semver

from agm.core import dry_run, process
from agm.packages.activation import load_activation_index, resolve_indexed_packages
from agm.packages.archive import write_archive
from agm.packages.errors import PackageInstallError
from agm.packages.install import uninstall_package
from agm.packages.layout import MODULE_TREE_DIRNAME
from agm.packages.python_deps import python_dependencies, sync_python_dependencies
from agm.packages.store import installed_packages, package_store_path
from tests._package_helpers import install_archive, install_directory

# ``packaging`` is an AGM runtime dependency, so it is always installed; the
# ``agm-test-absent-*`` distributions never are.
_PRESENT = "packaging>=1"
_ABSENT_A = "agm-test-absent-a>=1"
_ABSENT_B = "agm-test-absent-b"
_SHARED = "agm-test-absent-shared>=2"


@dataclass
class _Installer:
    """Records installer runs; *returncode* is what each run reports."""

    home: Path
    returncode: int = 0
    interrupt: bool = False
    runs: list[list[str]] = field(default_factory=list)
    active_during_run: list[set[str]] = field(default_factory=list)

    @property
    def specs(self) -> list[list[str]]:
        """Requirements of each run, after ``uv pip install --python <interpreter>``."""
        return [run[5:] for run in self.runs]


@pytest.fixture
def installer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Installer:
    """Replace the installer process with a recorder; nothing is ever installed."""
    recorder = _Installer(home=tmp_path / "home")

    def run_foreground(cmd: list[str], **_kwargs: object) -> int:
        recorder.runs.append(cmd)
        recorder.active_during_run.append(
            set(load_activation_index(home=recorder.home, env={}).packages)
        )
        if recorder.interrupt:
            raise KeyboardInterrupt
        return recorder.returncode

    monkeypatch.setattr(process, "run_foreground", run_foreground)
    monkeypatch.setattr("agm.core.pyenv.shutil.which", lambda _name: "/opt/tools/uv")
    return recorder


def _package(root: Path, name: str, *specs: str) -> Path:
    (root / MODULE_TREE_DIRNAME).mkdir(parents=True)
    python = f"\n[python]\ndependencies = [{', '.join(repr(s) for s in specs)}]\n" if specs else ""
    (root / "package.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "1.0.0"\n{python}'.replace("'", '"'),
        encoding="utf-8",
    )
    (root / MODULE_TREE_DIRNAME / "main.agl").write_text(
        "program def main() -> unit = ()\n", encoding="utf-8"
    )
    return root


def test_required_dependencies_are_the_union_over_store_and_editable_packages(
    tmp_path: Path, installer: _Installer
) -> None:
    home = installer.home
    install_directory(_package(tmp_path / "zulu", "zulu", _ABSENT_A, _SHARED), home=home, env={})
    install_directory(
        _package(tmp_path / "alpha", "alpha", _SHARED, _ABSENT_B), home=home, env={}, editable=True
    )

    index = load_activation_index(home=home, env={})

    assert python_dependencies(resolve_indexed_packages(index, home=home, env={})) == (
        _SHARED,
        _ABSENT_B,
        _ABSENT_A,
    )


def test_an_unsatisfied_requirement_installs_the_whole_union_jointly(
    tmp_path: Path, installer: _Installer
) -> None:
    install_directory(
        _package(tmp_path / "alpha", "alpha", _PRESENT, _ABSENT_A), home=installer.home, env={}
    )

    assert installer.specs == [[_PRESENT, _ABSENT_A]]


def test_opposing_requirements_reach_the_installer_together(
    tmp_path: Path, installer: _Installer
) -> None:
    home = installer.home
    install_directory(_package(tmp_path / "alpha", "alpha", _PRESENT), home=home, env={})
    install_directory(_package(tmp_path / "bravo", "bravo", "packaging<1"), home=home, env={})

    assert installer.specs == [[_PRESENT, "packaging<1"]]


def test_install_syncs_every_active_package_requirement(
    tmp_path: Path, installer: _Installer
) -> None:
    home = installer.home
    install_directory(_package(tmp_path / "alpha", "alpha", _ABSENT_A), home=home, env={})
    install_directory(
        _package(tmp_path / "bravo", "bravo", _ABSENT_B), home=home, env={}, editable=True
    )

    assert installer.specs == [[_ABSENT_A], [_ABSENT_A, _ABSENT_B]]


def test_install_without_unsatisfied_requirements_runs_no_installer(
    tmp_path: Path, installer: _Installer
) -> None:
    install_directory(_package(tmp_path / "alpha", "alpha"), home=installer.home, env={})
    install_directory(_package(tmp_path / "bravo", "bravo", _PRESENT), home=installer.home, env={})

    assert installer.runs == []


def test_install_syncs_before_publishing_the_activation(
    tmp_path: Path, installer: _Installer
) -> None:
    home = installer.home
    install_directory(_package(tmp_path / "alpha", "alpha", _ABSENT_A), home=home, env={})
    install_directory(_package(tmp_path / "bravo", "bravo", _ABSENT_B), home=home, env={})

    assert installer.active_during_run == [set(), {"alpha"}]
    assert set(load_activation_index(home=home, env={}).packages) == {"alpha", "bravo"}


@pytest.mark.parametrize("editable", [False, True])
def test_failed_python_install_rolls_the_store_back(
    tmp_path: Path, installer: _Installer, editable: bool
) -> None:
    home = installer.home
    install_directory(_package(tmp_path / "alpha", "alpha"), home=home, env={})
    before_index = load_activation_index(home=home, env={})
    before_store = installed_packages(home=home, env={})
    installer.returncode = 1

    with pytest.raises(PackageInstallError):
        install_directory(
            _package(tmp_path / "bravo", "bravo", _ABSENT_B), home=home, env={}, editable=editable
        )

    assert installer.specs == [[_ABSENT_B]]
    assert load_activation_index(home=home, env={}) == before_index
    assert installed_packages(home=home, env={}) == before_store
    assert not package_store_path("bravo", semver.Version(1), home=home, env={}).exists()


def test_interrupted_python_install_rolls_the_store_back(
    tmp_path: Path, installer: _Installer
) -> None:
    home = installer.home
    install_directory(_package(tmp_path / "alpha", "alpha"), home=home, env={})
    before_index = load_activation_index(home=home, env={})
    installer.interrupt = True

    with pytest.raises(KeyboardInterrupt):
        install_directory(_package(tmp_path / "bravo", "bravo", _ABSENT_B), home=home, env={})

    assert load_activation_index(home=home, env={}) == before_index
    assert not package_store_path("bravo", semver.Version(1), home=home, env={}).exists()


def test_unstartable_installer_fails_the_install(
    tmp_path: Path, installer: _Installer, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unstartable(cmd: list[str], **_kwargs: object) -> int:
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(process, "run_foreground", unstartable)

    with pytest.raises(PackageInstallError):
        install_directory(
            _package(tmp_path / "alpha", "alpha", _ABSENT_A), home=installer.home, env={}
        )

    assert load_activation_index(home=installer.home, env={}).packages == {}


def test_dry_run_install_reports_the_requirements_without_installing(
    tmp_path: Path, installer: _Installer, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _package(tmp_path / "alpha", "alpha", _PRESENT, _ABSENT_A)
    dry_run.set_enabled(True)

    install_directory(source, home=installer.home, env={})

    out = capsys.readouterr().out
    assert _ABSENT_A in out
    assert _PRESENT in out
    assert installer.runs == []
    assert not (installer.home / ".agm").exists()


def test_dry_run_archive_install_reports_the_requirements_without_installing(
    tmp_path: Path, installer: _Installer, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "alpha.agmpkg"
    write_archive(_package(tmp_path / "alpha", "alpha", _ABSENT_A), archive)
    capsys.readouterr()
    dry_run.set_enabled(True)

    install_archive(archive, home=installer.home, env={})

    assert _ABSENT_A in capsys.readouterr().out
    assert installer.runs == []
    assert not (installer.home / ".agm").exists()


def test_archive_install_syncs_the_archived_requirements(
    tmp_path: Path, installer: _Installer
) -> None:
    archive = tmp_path / "alpha.agmpkg"
    write_archive(_package(tmp_path / "alpha", "alpha", _ABSENT_A), archive)

    install_archive(archive, home=installer.home, env={})

    assert installer.specs == [[_ABSENT_A]]
    assert "alpha" in load_activation_index(home=installer.home, env={}).packages


def test_failed_python_install_rolls_an_archive_install_back(
    tmp_path: Path, installer: _Installer
) -> None:
    archive = tmp_path / "alpha.agmpkg"
    write_archive(_package(tmp_path / "alpha", "alpha", _ABSENT_A), archive)
    installer.returncode = 1

    with pytest.raises(PackageInstallError):
        install_archive(archive, home=installer.home, env={})

    assert load_activation_index(home=installer.home, env={}).packages == {}
    assert not package_store_path("alpha", semver.Version(1), home=installer.home, env={}).exists()


def test_uninstall_leaves_python_packages_alone(tmp_path: Path, installer: _Installer) -> None:
    home = installer.home
    install_directory(_package(tmp_path / "alpha", "alpha", _ABSENT_A), home=home, env={})
    installer.runs.clear()

    uninstall_package("alpha", home=home, env={})

    assert installer.runs == []


def test_sync_installs_the_whole_union_and_reports_the_unsatisfied(
    installer: _Installer,
) -> None:
    assert sync_python_dependencies((_PRESENT, _ABSENT_A)) == (_ABSENT_A,)
    assert installer.specs == [[_PRESENT, _ABSENT_A]]


def test_sync_of_satisfied_requirements_runs_no_installer(installer: _Installer) -> None:
    assert sync_python_dependencies((_PRESENT,)) == ()
    assert installer.runs == []
