"""One resolver for the ``agm exec`` argument: inline source, on-disk file, or
installed ``PACKAGE/MODULE::PROGRAM`` reference.

Execution (``commands/exec.py`` and ``exec_program.run_registered``) and the
advisory ``--help``/shell-completion surfaces (``cli.py``, ``completion.py``)
must classify a bare ``agm exec`` positional argument identically: it names an
installed reference when no inline ``-c/--command`` source was given, the
argument contains ``::``, and no on-disk file exists at that path. This module
supplies that one classification rule (:func:`is_installed_reference`) plus
the installed-reference resolution (module path -> active package -> entry
file) shared by execution, help, and completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agm.agl.modules.ids import ModuleId
from agm.packages.activation import select_active_packages
from agm.packages.model import PackageInfo


def is_installed_reference(file: str | None, *, command: str | None) -> bool:
    """Return whether *file* names an installed package program reference.

    True exactly when no inline ``-c/--command`` source was given, *file* is
    present, contains ``::``, and no on-disk file exists at that path. This is
    the single classification rule shared by execution, ``--help``, and shell
    completion.
    """
    return command is None and file is not None and "::" in file and not Path(file).is_file()


@dataclass(frozen=True, slots=True)
class InlineSource:
    """The exec argument is inline ``-c/--command`` source."""


@dataclass(frozen=True, slots=True)
class FileEntry:
    """The exec argument names an on-disk file."""

    path: Path


@dataclass(frozen=True, slots=True)
class PackageProgramReference:
    """An installed reference resolved to its owning package and entry file."""

    package: PackageInfo
    packages: tuple[PackageInfo, ...]
    module_id: ModuleId
    entry_path: Path
    declaration_path: str


@dataclass(frozen=True, slots=True)
class ExecTargetError:
    """The exec argument failed classification or installed-reference resolution."""

    message: str


type ExecTarget = InlineSource | FileEntry | PackageProgramReference


def resolve_installed_reference(
    reference: str,
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    package_name: str | None = None,
) -> PackageProgramReference | ExecTargetError:
    """Resolve an installed ``MODULE::DECLARATION`` reference to its entry file.

    *package_name* selects the active package explicitly instead of using the
    reference's own leading module segment; registered-command dispatch passes
    its cached package name so a stale registration is reported against the
    package it was cached against rather than whatever the reference happens
    to name.
    """
    module_path, separator, declaration_path = reference.partition("::")
    if not separator or not declaration_path:
        return ExecTargetError(f"invalid installed program reference {reference!r}.")
    try:
        module_id = ModuleId.from_path(module_path)
    except ValueError as exc:
        return ExecTargetError(f"invalid installed program reference {reference!r}: {exc}")
    try:
        packages = select_active_packages(home=home, proj_dir=proj_dir, cwd=cwd)
    except ValueError as exc:
        return ExecTargetError(f"cannot resolve active packages: {exc}")
    name = module_id.segments[0] if package_name is None else package_name
    package = next((candidate for candidate in packages if candidate.manifest.name == name), None)
    if package is None:
        return ExecTargetError(
            f"installed program reference {reference!r} does not name an active package."
        )
    try:
        entry_path = package.module_path(module_id.segments)
    except ValueError:
        return ExecTargetError(
            f"installed program reference {reference!r} names no module of package {name!r}."
        )
    return PackageProgramReference(
        package=package,
        packages=packages,
        module_id=module_id,
        entry_path=entry_path,
        declaration_path=declaration_path,
    )


def resolve_exec_target(
    *,
    file: str | None,
    command: str | None,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
) -> ExecTarget | ExecTargetError:
    """Classify an ``agm exec`` argument and resolve it if it names an installed reference."""
    if command is not None:
        return InlineSource()
    if file is None:
        return ExecTargetError("exec requires either a FILE or -c/--command")
    if not is_installed_reference(file, command=command):
        return FileEntry(Path(file))
    return resolve_installed_reference(file, home=home, proj_dir=proj_dir, cwd=cwd)
