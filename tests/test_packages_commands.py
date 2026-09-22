"""Command adapters for package lifecycle operations."""

from __future__ import annotations

from pathlib import Path

import pytest
import semver

import agm.commands.pkg.create as create_command
import agm.commands.pkg.info as info_command
import agm.commands.pkg.install as install_command
import agm.commands.pkg.list as list_command
import agm.commands.pkg.uninstall as uninstall_command
from agm.cli_support.args import (
    PkgCreateArgs,
    PkgInfoArgs,
    PkgInstallArgs,
    PkgListArgs,
    PkgUninstallArgs,
)
from agm.config.context import ConfigContext
from agm.core import dry_run
from agm.packages.activation import (
    ActivationIndex,
    ActivePackage,
    load_activation_index,
    write_activation_index,
)
from agm.packages.archive import write_archive
from agm.packages.layout import MODULE_TREE_DIRNAME
from agm.packages.manifest import PackageManifest
from agm.packages.model import PackageInfo
from agm.packages.record import write_record
from agm.version import AGM_VERSION
from tests._package_helpers import (
    install_archive,
    install_directory,
    older_incompatible_std_requirement,
    std_compatibility_bound,
)

_PARAM_SURFACE_PACKAGE = Path(__file__).parent / "agl" / "packages" / "param_surface"


def _context(tmp_path: Path) -> ConfigContext:
    return ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path)


def _package(tmp_path: Path, name: str = "alpha") -> PackageInfo:
    root = tmp_path / name
    root.mkdir(parents=True)
    (root / MODULE_TREE_DIRNAME).mkdir()
    (root / MODULE_TREE_DIRNAME / "main.agl").write_text("program def main() -> unit = ()\n")
    manifest = PackageManifest(name=name, version=semver.Version.parse("1.0.0"))
    (root / "package.toml").write_text(
        f'[package]\nname = "{manifest.name}"\nversion = "{manifest.version}"\n',
        encoding="utf-8",
    )
    return PackageInfo(root, manifest)


def _command_package(
    tmp_path: Path, name: str, *, version: str = "1.0.0", commands: tuple[str, ...] = ("launch",)
) -> Path:
    package = _package(tmp_path, name)
    _write_command_manifest(package.root, name, version=version, commands=commands)
    return package.root


def _write_command_manifest(
    root: Path, name: str, *, version: str, commands: tuple[str, ...]
) -> None:
    registrations = "".join(
        f'"{command}" = {{ program = "{name}/main::main" }}\n' for command in commands
    )
    (root / "package.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "{version}"\n\n[commands]\n{registrations}'
    )


@pytest.fixture(autouse=True)
def isolate_package_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(_context(tmp_path).home))
    monkeypatch.chdir(tmp_path)


def test_create_command_writes_an_archive_that_can_be_installed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _package(tmp_path)
    create_command.run(PkgCreateArgs(directory=str(package.root), output=None))

    archive = package.root.parent / "alpha-1.0.0.agmpkg"
    assert archive.is_file()
    assert archive.name in capsys.readouterr().out
    installed = install_archive(archive, home=_context(tmp_path).home)
    assert installed.manifest.name == "alpha"
    assert (installed.root / "package.toml").read_bytes() == (
        package.root / "package.toml"
    ).read_bytes()


def test_parameter_surface_fixture_installs_its_source_commands(tmp_path: Path) -> None:
    """The cross-surface fixture remains a valid installable package."""
    home = _context(tmp_path).home

    install_directory(_PARAM_SURFACE_PACKAGE, home=home, env={})

    commands = load_activation_index(home=home).commands
    assert commands["param review"].program == "param_tools/review::main"
    assert commands["param audit"].program == "param_tools/audit::main"


def test_create_command_dry_run_reports_its_plan_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _package(tmp_path)
    dry_run.set_enabled(True)

    create_command.run(PkgCreateArgs(directory=str(package.root), output="out.agmpkg"))

    assert "out.agmpkg" in capsys.readouterr().out
    assert not (tmp_path / "out.agmpkg").exists()


@pytest.mark.parametrize("invalid_manifest", [True, False], ids=["manifest", "destination"])
def test_create_command_reports_failure_without_publishing_an_archive(
    tmp_path: Path, invalid_manifest: bool
) -> None:
    package = _package(tmp_path)
    destination = tmp_path / "out.agmpkg"
    if invalid_manifest:
        (package.root / "package.toml").write_text("[broken")
    else:
        destination.mkdir()
    with pytest.raises(SystemExit) as raised:
        create_command.run(PkgCreateArgs(directory=str(package.root), output=str(destination)))

    assert raised.value.code == 1
    assert not destination.is_file()
    assert package.root.is_dir()


def test_create_rejects_an_older_incompatible_std_before_archive_publication(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    (package.root / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "1.0.0"\n\n'
        f'[dependencies]\nstd = "{older_incompatible_std_requirement()}"\n',
        encoding="utf-8",
    )
    (package.root / MODULE_TREE_DIRNAME / "main.agl").write_text(
        "program def main() -> unit = ()\n", encoding="utf-8"
    )
    destination = tmp_path / "alpha.agmpkg"

    with pytest.raises(SystemExit):
        create_command.run(PkgCreateArgs(directory=str(package.root), output=str(destination)))

    assert not destination.exists()


@pytest.mark.parametrize("archive_source", [False, True], ids=["directory", "archive"])
def test_installed_package_can_be_inspected_and_removed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], archive_source: bool
) -> None:
    package = _package(tmp_path / "source space λ")
    source = package.root
    if archive_source:
        source = tmp_path / "release space λ.agmpkg"
        write_archive(package.root, source)

    install_command.run(PkgInstallArgs(str(source), editable=False, shadow=archive_source))
    capsys.readouterr()
    info_command.run(PkgInfoArgs("alpha"))
    assert "1.0.0" in capsys.readouterr().out
    installed = _context(tmp_path).home / ".agm" / "packages" / "alpha" / "1.0.0"
    assert (installed / "package.toml").read_bytes() == (package.root / "package.toml").read_bytes()

    uninstall_command.run(PkgUninstallArgs("alpha"))
    assert not installed.exists()
    with pytest.raises(SystemExit) as raised:
        info_command.run(PkgInfoArgs("alpha"))
    assert raised.value.code == 1
    assert (package.root / "package.toml").is_file()


def test_dry_run_shadow_install_reports_the_planned_displacement_without_changing_activation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    context = _context(tmp_path)

    install_directory(_command_package(tmp_path, "alpha"), home=context.home)
    persisted = load_activation_index(home=context.home)
    candidate = _command_package(tmp_path, "bravo")
    dry_run.set_enabled(True)

    install_command.run(PkgInstallArgs(str(candidate), editable=False, shadow=True))

    assert "shadowed command launch from alpha" in capsys.readouterr().out
    assert load_activation_index(home=context.home) == persisted


@pytest.mark.parametrize("operation", ["install", "uninstall"])
def test_missing_package_is_rejected_without_activating_it(tmp_path: Path, operation: str) -> None:
    with pytest.raises(SystemExit) as raised:
        if operation == "install":
            install_command.run(PkgInstallArgs("missing", editable=True, shadow=False))
        else:
            uninstall_command.run(PkgUninstallArgs("missing"))

    assert raised.value.code == 1
    assert "missing" not in load_activation_index(home=_context(tmp_path).home).packages
    assert not (_context(tmp_path).home / ".agm" / "packages" / "missing").exists()


def test_list_command_marks_only_the_exact_build_identity_active(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for build in ("other", "selected"):
        package = _package(tmp_path / build)
        manifest = package.root / "package.toml"
        manifest.write_text(manifest.read_text().replace("1.0.0", f"1.0.0+{build}"))
        install_directory(package.root, home=_context(tmp_path).home)

    list_command.run(PkgListArgs())

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("alpha ")]
    assert lines == ["alpha 1.0.0+other installed", "alpha 1.0.0+selected active"]


@pytest.mark.parametrize("corruption", ["index", "store"])
def test_list_command_reports_corrupt_persistent_state(tmp_path: Path, corruption: str) -> None:
    store = _context(tmp_path).home / ".agm" / "packages"
    if corruption == "index":
        store.mkdir(parents=True)
        (store / "index.toml").write_text("[broken")
    else:
        version = store / "alpha" / "1.0.0"
        version.mkdir(parents=True)
        (version / "package.toml").write_text('[package]\nname = "bravo"\nversion = "1.0.0"\n')

    with pytest.raises(SystemExit) as raised:
        list_command.run(PkgListArgs())
    assert raised.value.code == 1


def test_info_command_uses_live_editable_dependency_versions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    alpha = _package(tmp_path, "alpha")
    bravo = _package(tmp_path, "bravo")
    (alpha.root / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "1.0.0"\n\n[dependencies]\nbravo = "1"\n',
        encoding="utf-8",
    )
    write_activation_index(
        ActivationIndex(
            {
                "alpha": ActivePackage(alpha.manifest.version, editable=alpha.root),
                "bravo": ActivePackage(semver.Version.parse("2.0.0"), editable=bravo.root),
            }
        ),
        home=_context(tmp_path).home,
    )

    info_command.run(PkgInfoArgs("alpha"))

    assert "requires bravo >= 1.0.0: editable 1.0.0" in capsys.readouterr().out


def test_info_command_rejects_immutable_store_escapes_and_identity_mismatches(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    active = ActivePackage(semver.Version.parse("1.0.0"))
    write_activation_index(ActivationIndex({"alpha": active}), home=_context(tmp_path).home)

    store = context.home / ".agm" / "packages"
    external = tmp_path / "external"
    external.mkdir()
    (external / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    store.mkdir(parents=True, exist_ok=True)
    (store / "alpha").symlink_to(external, target_is_directory=True)
    with pytest.raises(SystemExit):
        info_command.run(PkgInfoArgs("alpha"))

    (store / "alpha").unlink()
    root = store / "alpha" / "1.0.0"
    root.mkdir(parents=True)
    (root / "package.toml").write_text(
        '[package]\nname = "bravo"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    write_record(root)
    with pytest.raises(SystemExit):
        info_command.run(PkgInfoArgs("alpha"))

    (root / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "2.0.0"\n', encoding="utf-8"
    )
    write_record(root)
    with pytest.raises(SystemExit):
        info_command.run(PkgInfoArgs("alpha"))


def test_info_command_rejects_different_build_metadata_for_immutable_package(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    active = ActivePackage(semver.Version.parse("1.0.0+selected"))
    root = context.home / ".agm" / "packages" / "alpha" / str(active.version)
    root.mkdir(parents=True)
    (root / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "1.0.0+other"\n', encoding="utf-8"
    )
    write_record(root)
    write_activation_index(ActivationIndex({"alpha": active}), home=_context(tmp_path).home)

    with pytest.raises(SystemExit):
        info_command.run(PkgInfoArgs("alpha"))


def test_info_command_reads_manifest_metadata_and_current_dependency_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _package(tmp_path)
    manifest = package.root / "package.toml"
    metadata = (
        '[package]\nname = "alpha"\nversion = "1.0.0"\n'
        'description = "Alpha package"\nlicense = "MIT"\n'
        'authors = ["Ada", "Lin"]\nrepository = "https://example.test/alpha"\n'
        'keywords = ["agents", "tools"]\n\n'
        '[commands]\nrun = { program = "alpha/main::main", doc = "Run Alpha" }\n\n'
        f'[dependencies]\nstd = "{AGM_VERSION}"\n'
        'bravo = "1"\ncharlie = "2"\ndelta = "1"\necho = "1"\n'
    )
    manifest.write_text(metadata)
    delta = _package(tmp_path, "delta")
    index = ActivationIndex(
        {
            "alpha": ActivePackage(package.manifest.version, editable=package.root),
            "charlie": ActivePackage(semver.Version.parse("1.0.0")),
            "delta": ActivePackage(delta.manifest.version, editable=delta.root),
            "echo": ActivePackage(semver.Version.parse("1.0.0")),
        }
    )
    write_activation_index(index, home=_context(tmp_path).home)

    info_command.run(PkgInfoArgs("alpha"))

    output = capsys.readouterr().out
    assert "Alpha package" in output
    assert "license: MIT" in output
    assert "authors: Ada, Lin" in output
    assert "repository: https://example.test/alpha" in output
    assert "keywords: agents, tools" in output
    assert "run: alpha/main::main" in output
    assert "Run Alpha" in output
    bound = std_compatibility_bound(AGM_VERSION)
    assert f"requires std >= {AGM_VERSION}, < {bound}: running AGM {AGM_VERSION}" in output
    assert "requires bravo >= 1.0.0: missing" in output
    assert "requires charlie >= 2.0.0: active 1.0.0 (unsatisfied)" in output
    assert "requires delta >= 1.0.0: editable 1.0.0" in output
    assert "requires echo >= 1.0.0: active 1.0.0" in output

    newer_agm = semver.Version.parse(AGM_VERSION).bump_major()
    manifest.write_text(metadata.replace(f'std = "{AGM_VERSION}"', f'std = "{newer_agm}"'))
    info_command.run(PkgInfoArgs("alpha"))
    assert "unsatisfied" in capsys.readouterr().out.split("requires std", 1)[1]

    manifest.write_text('[package]\nname = "alpha"\nversion = "1.0.0"\n')
    info_command.run(PkgInfoArgs("alpha"))
    assert capsys.readouterr().out == "alpha 1.0.0\n"

    manifest.write_text("[broken")
    with pytest.raises(SystemExit) as raised:
        info_command.run(PkgInfoArgs("alpha"))
    assert raised.value.code == 1


def test_a_stored_manifest_field_this_build_does_not_know_keeps_the_package_usable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A package another AGM installed stays listable and inspectable.

    Its baked manifest is AGM's own artifact, rewritten whole by the next
    install; a field this build has no meaning for must not strand the store.
    """
    home = _context(tmp_path).home
    install_directory(_command_package(tmp_path, "alpha"), home=home)
    stored = home / ".agm" / "packages" / "alpha" / "1.0.0" / "package.toml"
    stored.write_text(
        '[package]\nname = "alpha"\nversion = "1.0.0"\nrelease-channel = "beta"\n\n'
        '[commands]\nlaunch = { program = "alpha/main::main", doc = "Launch", summary = "x" }\n',
        encoding="utf-8",
    )

    list_command.run(PkgListArgs())
    assert capsys.readouterr().out.splitlines() == ["alpha 1.0.0 active"]

    info_command.run(PkgInfoArgs("alpha"))
    assert "alpha 1.0.0" in capsys.readouterr().out


def test_list_reports_every_stored_version_and_each_editable_package(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _context(tmp_path).home
    alpha = _command_package(tmp_path, "alpha")
    bravo = _command_package(tmp_path, "bravo", commands=("launch", "bravo run"))
    install_directory(alpha, home=home)
    install_directory(bravo, home=home, editable=True, shadow=True)
    _write_command_manifest(alpha, "alpha", version="2.0.0", commands=("launch",))
    install_directory(alpha, home=home, shadow=True)

    list_command.run(PkgListArgs())

    lines = capsys.readouterr().out.splitlines()
    assert lines == ["alpha 1.0.0 installed", "alpha 2.0.0 active", "bravo 1.0.0 editable"]

    _write_command_manifest(bravo, "bravo", version="1.1.0", commands=("bravo updated",))
    list_command.run(PkgListArgs())
    assert "bravo 1.1.0 editable" in capsys.readouterr().out
