"""Command adapters for package lifecycle operations."""

from __future__ import annotations

from dataclasses import replace
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
from agm.packages.activation import (
    ActivationIndex,
    ActivePackage,
    CommandRegistration,
    CommandShadow,
    PackageActivationError,
)
from agm.packages.archive import ArchiveError
from agm.packages.install import PackageInstallError
from agm.packages.manifest import CommandSpec, DependencySpec, PackageManifest
from agm.packages.model import PackageInfo
from agm.version import AGM_VERSION


def _context(tmp_path: Path) -> ConfigContext:
    return ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path)


def _package(tmp_path: Path, name: str = "alpha") -> PackageInfo:
    root = tmp_path / name
    root.mkdir()
    manifest = PackageManifest(name=name, version=semver.Version.parse("1.0.0"))
    (root / "package.toml").write_text(
        f'[package]\nname = "{manifest.name}"\nversion = "{manifest.version}"\n',
        encoding="utf-8",
    )
    return PackageInfo(root, manifest)


def test_create_command_validates_and_writes_the_default_archive_beside_its_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _package(tmp_path)
    monkeypatch.setattr(create_command, "validate_archive_source", lambda _: package.manifest)
    monkeypatch.setattr(create_command, "validate_dependencies", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(create_command, "current_config_context", lambda: _context(tmp_path))
    written: list[Path] = []
    monkeypatch.setattr(
        create_command,
        "write_archive",
        lambda _root, destination: written.append(destination),
    )

    create_command.run(PkgCreateArgs(directory=str(package.root), output=None))

    assert written == [package.root.parent / "alpha-1.0.0.agmpkg"]
    assert "alpha-1.0.0.agmpkg" in capsys.readouterr().out


def test_create_command_dry_run_reports_its_plan_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _package(tmp_path)
    monkeypatch.setattr(create_command, "validate_archive_source", lambda _: package.manifest)
    monkeypatch.setattr(create_command, "validate_dependencies", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(create_command, "current_config_context", lambda: _context(tmp_path))
    monkeypatch.setattr(create_command.dry_run, "enabled", lambda: True)
    monkeypatch.setattr(
        create_command,
        "write_archive",
        lambda *_: (_ for _ in ()).throw(AssertionError("dry-run must not write an archive")),
    )

    create_command.run(PkgCreateArgs(directory=str(package.root), output="out.agmpkg"))

    assert "dry-run: agm create-package-archive" in capsys.readouterr().out


def test_create_command_reports_validation_or_archive_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package(tmp_path)
    monkeypatch.setattr(create_command, "validate_archive_source", lambda _: package.manifest)
    monkeypatch.setattr(create_command, "validate_dependencies", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(create_command, "current_config_context", lambda: _context(tmp_path))
    monkeypatch.setattr(
        create_command,
        "write_archive",
        lambda *_: (_ for _ in ()).throw(ArchiveError("broken")),
    )

    with pytest.raises(SystemExit):
        create_command.run(PkgCreateArgs(directory=str(package.root), output="out.agmpkg"))


def test_install_command_delegates_and_renders_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _package(tmp_path)
    monkeypatch.setattr(install_command, "current_config_context", lambda: _context(tmp_path))
    monkeypatch.setattr(install_command, "install_directory", lambda *args, **kwargs: package)

    install_command.run(PkgInstallArgs("source", editable=False, shadow=False))

    assert "installed alpha 1.0.0" in capsys.readouterr().out


def test_install_command_routes_an_archive_to_the_archive_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "package.agmpkg"
    archive.write_bytes(b"archive")
    package = _package(tmp_path)
    monkeypatch.setattr(install_command, "current_config_context", lambda: _context(tmp_path))
    monkeypatch.setattr(install_command, "install_archive", lambda *args, **kwargs: package)

    install_command.run(PkgInstallArgs(str(archive), editable=False, shadow=True))


def test_shadow_install_reports_unavailable_shadow_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package(tmp_path)
    monkeypatch.setattr(install_command, "current_config_context", lambda: _context(tmp_path))
    monkeypatch.setattr(install_command, "install_directory", lambda *args, **kwargs: package)

    def fail_index(**_: object) -> ActivationIndex:
        raise PackageActivationError("broken")

    monkeypatch.setattr(install_command, "load_activation_index", fail_index)
    with pytest.raises(SystemExit):
        install_command.run(PkgInstallArgs("source", editable=False, shadow=True))


def test_shadow_install_renders_displaced_command_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _package(tmp_path)
    monkeypatch.setattr(install_command, "current_config_context", lambda: _context(tmp_path))
    monkeypatch.setattr(install_command, "install_directory", lambda *args, **kwargs: package)
    monkeypatch.setattr(install_command, "load_activation_index", lambda **_: ActivationIndex())
    monkeypatch.setattr(
        install_command,
        "command_shadow_diagnostics",
        lambda *args, **kwargs: {"alpha": (CommandShadow("launch", ("bravo",)),)},
    )

    install_command.run(PkgInstallArgs("source", editable=False, shadow=True))

    assert "shadowed command launch from bravo" in capsys.readouterr().out


def test_install_command_reports_domain_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_command, "current_config_context", lambda: _context(tmp_path))

    def fail(*_: object, **__: object) -> PackageInfo:
        raise PackageInstallError("bad package")

    monkeypatch.setattr(install_command, "install_directory", fail)
    with pytest.raises(SystemExit):
        install_command.run(PkgInstallArgs("source", editable=True, shadow=False))


def test_uninstall_command_delegates_and_reports_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(uninstall_command, "current_config_context", lambda: _context(tmp_path))
    monkeypatch.setattr(uninstall_command, "uninstall_package", lambda *args, **kwargs: None)
    uninstall_command.run(PkgUninstallArgs("alpha"))
    assert "uninstalled alpha" in capsys.readouterr().out

    def fail(*_: object, **__: object) -> None:
        raise PackageInstallError("bad package")

    monkeypatch.setattr(uninstall_command, "uninstall_package", fail)
    with pytest.raises(SystemExit):
        uninstall_command.run(PkgUninstallArgs("alpha"))


def test_list_command_prints_commands_only_below_their_active_or_editable_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    alpha_old = PackageInfo(
        tmp_path / "alpha-old",
        PackageManifest(name="alpha", version=semver.Version.parse("1.0.0")),
    )
    alpha = PackageInfo(
        tmp_path / "alpha-current",
        PackageManifest(name="alpha", version=semver.Version.parse("2.0.0")),
    )
    bravo = _package(tmp_path, "bravo")
    index = ActivationIndex(
        {
            "alpha": ActivePackage(alpha.manifest.version),
            "bravo": ActivePackage(semver.Version.parse("2.0.0"), editable=bravo.root),
        },
        {
            "launch": CommandRegistration("alpha", "alpha/main::main"),
            "bravo run": CommandRegistration("bravo", "bravo/main::main"),
        },
    )
    monkeypatch.setattr(list_command, "current_config_context", lambda: _context(tmp_path))
    monkeypatch.setattr(list_command, "load_activation_index", lambda **_: index)
    monkeypatch.setattr(list_command, "installed_packages", lambda **_: (alpha_old, alpha))
    monkeypatch.setattr(
        list_command,
        "command_shadow_diagnostics",
        lambda *args, **kwargs: {"alpha": (CommandShadow("launch", ("bravo",)),)},
    )

    list_command.run(PkgListArgs())

    assert capsys.readouterr().out.splitlines() == [
        "alpha 1.0.0 installed",
        "alpha 2.0.0 active",
        "  command launch (shadows bravo)",
        "bravo 1.0.0 editable",
        "  command bravo run",
    ]


@pytest.mark.parametrize("error", [PackageActivationError("bad"), PackageInstallError("bad")])
def test_list_command_reports_index_or_store_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(list_command, "current_config_context", lambda: _context(tmp_path))
    if isinstance(error, PackageActivationError):
        monkeypatch.setattr(
            list_command, "load_activation_index", lambda **_: (_ for _ in ()).throw(error)
        )
    else:
        monkeypatch.setattr(list_command, "load_activation_index", lambda **_: ActivationIndex())
        monkeypatch.setattr(
            list_command, "installed_packages", lambda **_: (_ for _ in ()).throw(error)
        )
    with pytest.raises(SystemExit):
        list_command.run(PkgListArgs())


def test_info_command_uses_live_editable_dependency_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    alpha = _package(tmp_path, "alpha")
    bravo = _package(tmp_path, "bravo")
    (alpha.root / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "1.0.0"\n\n'
        '[dependencies]\nbravo = "1"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(info_command, "current_config_context", lambda: _context(tmp_path))
    monkeypatch.setattr(
        info_command,
        "load_activation_index",
        lambda **_: ActivationIndex(
            {
                "alpha": ActivePackage(alpha.manifest.version, editable=alpha.root),
                "bravo": ActivePackage(semver.Version.parse("2.0.0"), editable=bravo.root),
            }
        ),
    )

    info_command.run(PkgInfoArgs("alpha"))

    assert "requires bravo >= 1.0.0: editable 1.0.0" in capsys.readouterr().out


def test_info_command_rejects_immutable_store_escapes_and_identity_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    active = ActivePackage(semver.Version.parse("1.0.0"))
    monkeypatch.setattr(info_command, "current_config_context", lambda: context)
    monkeypatch.setattr(
        info_command,
        "load_activation_index",
        lambda **_: ActivationIndex({"alpha": active}),
    )

    store = context.home / ".agm" / "packages"
    external = tmp_path / "external"
    external.mkdir()
    (external / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    store.mkdir(parents=True)
    (store / "alpha").symlink_to(external, target_is_directory=True)
    with pytest.raises(SystemExit):
        info_command.run(PkgInfoArgs("alpha"))

    (store / "alpha").unlink()
    root = store / "alpha" / "1.0.0"
    root.mkdir(parents=True)
    (root / "package.toml").write_text(
        '[package]\nname = "bravo"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        info_command.run(PkgInfoArgs("alpha"))

    (root / "package.toml").write_text(
        '[package]\nname = "alpha"\nversion = "2.0.0"\n', encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        info_command.run(PkgInfoArgs("alpha"))


def test_info_command_renders_metadata_and_reports_unknown_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _package(tmp_path)
    manifest = PackageManifest(
        name="alpha",
        version=semver.Version.parse("1.0.0"),
        description="Alpha package",
        license="MIT",
        authors=("Ada", "Lin"),
        repository="https://example.test/alpha",
        keywords=("agents", "tools"),
        dependencies={
            "std": DependencySpec(semver.Version.parse(AGM_VERSION)),
            "bravo": DependencySpec(semver.Version.parse("1.0.0")),
            "charlie": DependencySpec(semver.Version.parse("2.0.0")),
            "delta": DependencySpec(semver.Version.parse("1.0.0")),
            "echo": DependencySpec(semver.Version.parse("1.0.0")),
        },
        commands={"run": CommandSpec("alpha/main::main", "Run Alpha")},
    )
    monkeypatch.setattr(info_command, "current_config_context", lambda: _context(tmp_path))
    monkeypatch.setattr(
        info_command,
        "load_activation_index",
        lambda **_: ActivationIndex(
            {
                "alpha": ActivePackage(manifest.version, editable=package.root),
                "charlie": ActivePackage(semver.Version.parse("1.0.0")),
                "delta": ActivePackage(semver.Version.parse("1.0.0"), editable=package.root),
                "echo": ActivePackage(semver.Version.parse("1.0.0")),
            }
        ),
    )
    monkeypatch.setattr(info_command, "load_manifest", lambda _: manifest)

    info_command.run(PkgInfoArgs("alpha"))

    output = capsys.readouterr().out
    assert "Alpha package" in output
    assert "license: MIT" in output
    assert "authors: Ada, Lin" in output
    assert "repository: https://example.test/alpha" in output
    assert "keywords: agents, tools" in output
    assert "commands:" in output
    assert "run: alpha/main::main (Run Alpha)" in output
    assert f"requires std >= {AGM_VERSION}: running AGM {AGM_VERSION}" in output
    assert "requires bravo >= 1.0.0: missing" in output
    assert "requires charlie >= 2.0.0: active 1.0.0 (unsatisfied)" in output
    assert "requires delta >= 1.0.0: editable 1.0.0" in output
    assert "requires echo >= 1.0.0: active 1.0.0" in output

    newer_agm = semver.Version.parse(AGM_VERSION).bump_major()
    monkeypatch.setattr(
        info_command,
        "load_manifest",
        lambda _: replace(manifest, dependencies={"std": DependencySpec(newer_agm)}),
    )
    info_command.run(PkgInfoArgs("alpha"))
    assert (
        f"requires std >= {newer_agm}: running AGM {AGM_VERSION} (unsatisfied)"
        in capsys.readouterr().out
    )

    no_description = PackageManifest(name="alpha", version=manifest.version)
    monkeypatch.setattr(info_command, "load_manifest", lambda _: no_description)
    info_command.run(PkgInfoArgs("alpha"))
    monkeypatch.setattr(info_command, "load_activation_index", lambda **_: ActivationIndex())
    with pytest.raises(SystemExit):
        info_command.run(PkgInfoArgs("alpha"))

    monkeypatch.setattr(
        info_command,
        "load_activation_index",
        lambda **_: ActivationIndex(
            {"alpha": ActivePackage(manifest.version, editable=package.root)}
        ),
    )
    monkeypatch.setattr(
        info_command,
        "load_manifest",
        lambda _: (_ for _ in ()).throw(info_command.ManifestError("broken")),
    )
    with pytest.raises(SystemExit):
        info_command.run(PkgInfoArgs("alpha"))
