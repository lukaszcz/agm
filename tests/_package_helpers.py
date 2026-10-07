"""Shared helpers for tests that need a real activated package on disk.

``write_installed_package`` lays out a minimal single-module package under an
AGM home's package store, writes its ``RECORD``, and activates it, so tests
that exercise installed-reference resolution can work against the same store
layout production code reads.

``older_incompatible_std_requirement`` and ``std_compatibility_bound`` derive
``std`` requirements and bounds from the running AGM version, so release-line
tests keep testing the rule rather than one release's literals.

``archive_contents`` and ``write_zip`` read and rewrite a ``.agmpkg`` archive's
raw ZIP entries, so tests can graft tampered or hand-built content onto an
otherwise valid archive.

``package_info`` reads a package source's manifest into its ``PackageInfo``.

``install_directory`` and ``install_archive`` install a package and return only
its ``PackageInfo``.

``write_python_package`` writes a package source declaring ``[python]``
requirements; ``PythonInstaller`` records installer runs in place of the real
installer (see the ``python_installer`` fixture).

``write_rollback_package``, ``replace_rollback_payload``, and
``InstalledPackageSnapshot`` share failed-reinstall setup and restoration checks.
"""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import semver
from click.testing import CliRunner
from typer.main import get_command

import agm.packages.archive as package_archive
from agm.packages.activation import (
    ActivationIndex,
    ActivePackage,
    load_activation_index,
    write_activation_index,
)
from agm.packages.install import install_archive_with_plan, install_directory_with_plan
from agm.packages.layout import MODULE_TREE_DIRNAME
from agm.packages.manifest import load_manifest
from agm.packages.model import PackageInfo
from agm.packages.record import verify_record, write_record
from agm.version import AGM_VERSION


def older_incompatible_std_requirement() -> str:
    """Return a ``std`` requirement one release line below the running AGM.

    ``std`` is compatible within one minor line before 1.0 and one major line
    from 1.0 on, so stepping the running version back a whole line yields a
    requirement the running AGM must reject as too old.
    """
    running = semver.Version.parse(AGM_VERSION)
    if running.major == 0:
        return str(semver.Version(0, running.minor - 1, 0))
    return str(semver.Version(running.major - 1, 0, 0))


def std_compatibility_bound(requirement: str) -> str:
    """Return the exclusive AGM bound a ``std`` *requirement* implies."""
    version = semver.Version.parse(requirement)
    if version.major == 0:
        return str(semver.Version(0, version.minor + 1, 0))
    return str(semver.Version(version.major + 1, 0, 0))


def write_installed_package(
    home: Path,
    name: str,
    *,
    source: str = "program def main(level: text) -> unit = ()\n",
    commands: Mapping[str, str] | None = None,
    module_path: str = "main",
) -> Path:
    """Install and activate a one-module package named *name* under *home*.

    *commands* registers manifest command paths against program references, so
    a test can exercise a package that owns CLI commands.  *module_path* is the
    module's ``/``-separated path below the package name, so a test can install
    a module nested inside the module tree.

    Returns the path of the package's module file, so a test can edit or remove
    the entry file it will later resolve.
    """

    package_root = home / ".agm" / "packages" / name / "1.0.0"
    module = package_root.joinpath(MODULE_TREE_DIRNAME, *module_path.split("/")).with_suffix(".agl")
    module.parent.mkdir(parents=True)
    registrations = "".join(
        f'"{path}" = {{ program = "{program}" }}\n' for path, program in (commands or {}).items()
    )
    command_table = f"\n[commands]\n{registrations}" if registrations else ""
    (package_root / "package.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "1.0.0"\n{command_table}', encoding="utf-8"
    )
    module.write_text(source, encoding="utf-8")
    write_record(package_root)
    write_activation_index(
        ActivationIndex({name: ActivePackage(semver.Version.parse("1.0.0"))}), home=home
    )
    return module


def archive_contents(path: Path) -> dict[str, bytes]:
    """Read every entry of the ``.agmpkg`` archive at *path* by name."""
    with zipfile.ZipFile(path) as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def write_zip(path: Path, contents: list[tuple[str, bytes]]) -> None:
    """Write a canonical-metadata ZIP at *path* from ``(name, content)`` pairs.

    Tests use this to graft tampered or hand-built entries onto an archive,
    reusing the same per-entry metadata :mod:`agm.packages.archive` writes.
    """
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in contents:
            archive.writestr(package_archive._zip_info(name), content)


def package_info(root: Path) -> PackageInfo:
    """The package whose manifest is ``root/package.toml``."""
    return PackageInfo(root, load_manifest(root / "package.toml"))


def install_directory(
    source: Path,
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
    editable: bool = False,
    shadow: bool = False,
) -> PackageInfo:
    """Install a package directory and return the installed package."""
    return install_directory_with_plan(
        source, home=home, env=env, editable=editable, shadow=shadow
    ).package


def install_archive(
    archive: Path,
    *,
    home: Path,
    env: Mapping[str, str] | None = None,
    shadow: bool = False,
) -> PackageInfo:
    """Install a package archive and return the installed package."""
    return install_archive_with_plan(archive, home=home, env=env, shadow=shadow).package


def write_python_package(root: Path, name: str, *specs: str) -> Path:
    """Write a one-module package source at *root* requiring the PEP 508 *specs*."""
    (root / MODULE_TREE_DIRNAME).mkdir(parents=True)
    listed = ", ".join(f'"{spec}"' for spec in specs)
    python = f"\n[python]\ndependencies = [{listed}]\n" if specs else ""
    (root / "package.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "1.0.0"\n{python}', encoding="utf-8"
    )
    (root / MODULE_TREE_DIRNAME / "main.agl").write_text(
        "program def main() -> unit = ()\n", encoding="utf-8"
    )
    return root


def write_rollback_package(root: Path, *specs: str) -> Path:
    """Write an executable original package with resources and an old-only file."""
    source = write_python_package(root, "alpha", *specs)
    manifest = source / "package.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8")
        + '\n[commands]\noriginal = { program = "alpha/main::main" }\n',
        encoding="utf-8",
    )
    (source / MODULE_TREE_DIRNAME / "main.agl").write_text(
        "import std/fs\nprogram def main() -> unit =\n"
        '  print("original behavior")\n  print(fs::read(resource("payload.txt")))\n',
        encoding="utf-8",
    )
    (source / "payload.txt").write_text("original resource", encoding="utf-8")
    (source / "obsolete.txt").write_text("old only", encoding="utf-8")
    return source


def replace_rollback_payload(source: Path) -> None:
    """Change behavior, resource bytes, and file membership before a failed reinstall."""
    (source / MODULE_TREE_DIRNAME / "main.agl").write_text(
        'program def main() -> unit = print("replacement behavior")\n', encoding="utf-8"
    )
    (source / "payload.txt").write_text("replacement resource", encoding="utf-8")
    (source / "obsolete.txt").unlink()
    (source / "new-only.txt").write_text("new only", encoding="utf-8")


def _file_snapshot(root: Path) -> dict[Path, bytes]:
    """Snapshot file membership and bytes, not modes or empty directories."""
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@dataclass
class InstalledPackageSnapshot:
    """Original files and activation state that a failed reinstall must restore."""

    root: Path
    home: Path
    files: dict[Path, bytes]
    index: ActivationIndex
    index_bytes: bytes

    @property
    def index_path(self) -> Path:
        return self.home / ".agm" / "packages" / "index.toml"

    @classmethod
    def capture(cls, package: PackageInfo, *, home: Path) -> InstalledPackageSnapshot:
        """Remember the installed package and both forms of its activation state."""
        return cls(
            package.root,
            home,
            _file_snapshot(package.root),
            load_activation_index(home=home, env={}),
            (home / ".agm" / "packages" / "index.toml").read_bytes(),
        )

    def assert_restored(self) -> None:
        """Verify rollback integrity, cleanup, and the original CLI behavior."""
        import agm.cli as cli

        assert _file_snapshot(self.root) == self.files
        assert verify_record(self.root)
        assert load_activation_index(home=self.home, env={}) == self.index
        assert self.index_path.read_bytes() == self.index_bytes
        assert not tuple(self.root.parent.glob(".agm-previous-*"))
        assert not tuple(self.root.parent.glob(".agm-package-*"))
        result = CliRunner().invoke(
            get_command(cli.app), ["original"], env={"HOME": str(self.home)}, catch_exceptions=False
        )
        assert result.exit_code == 0
        assert result.stdout.splitlines() == ["original behavior", "original resource"]


@dataclass
class PythonInstaller:
    """Records installer runs under AGM home *home*; each run returns *returncode*."""

    home: Path
    returncode: int = 0
    interrupt: bool = False
    runs: list[list[str]] = field(default_factory=list)
    active_during_run: list[set[str]] = field(default_factory=list)

    @property
    def specs(self) -> list[list[str]]:
        """Requirements of each run, after ``uv pip install --python <interpreter>``."""
        return [run[5:] for run in self.runs]

    def run_foreground(self, cmd: list[str], **_kwargs: object) -> int:
        """Stand-in for ``process.run_foreground``; never installs anything."""
        self.runs.append(cmd)
        self.active_during_run.append(set(load_activation_index(home=self.home, env={}).packages))
        if self.interrupt:
            raise KeyboardInterrupt
        return self.returncode
