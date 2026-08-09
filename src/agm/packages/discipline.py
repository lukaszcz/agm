"""Filesystem and command discipline for package directories."""

from __future__ import annotations

from pathlib import Path

from agm.agl.modules.ids import ModuleId
from agm.agl.parser import AglSyntaxError, parse_program
from agm.agl.syntax.nodes import static_function_items
from agm.command_catalog import COMMAND_NAMES
from agm.core import fs
from agm.packages.model import PackageInfo
from agm.util.ident import is_identifier

# These are the command-tree aliases that do not appear in COMMAND_NAMES.
_RESERVED_COMMAND_NAMES = frozenset(COMMAND_NAMES) | frozenset({"wsp", "wt", "cp", "copy"})


class DisciplineError(ValueError):
    """Raised when a package directory violates package discipline."""


def validate_package(package: PackageInfo) -> None:
    """Validate a package's module tree, commands, and program references."""

    _validate_package_name(package.manifest.name)
    modules = _module_files(package)
    for command_path, command in package.manifest.commands.items():
        _validate_command_path(command_path)
        _validate_program_reference(package, command.program, modules)


def _validate_package_name(name: str) -> None:
    if name in _RESERVED_COMMAND_NAMES:
        raise DisciplineError(f"package name {name!r} is reserved by AGM")


def _module_files(package: PackageInfo) -> dict[ModuleId, Path]:
    module_root = package.module_root
    if not fs.is_dir(module_root):
        raise DisciplineError(
            f"package {package.manifest.name!r} requires module tree {module_root.name!r}"
        )

    modules: dict[ModuleId, Path] = {}
    for path in sorted(fs.rglob(module_root, "*.agl")):
        if not fs.is_file(path):
            continue
        relative = path.relative_to(package.root).with_suffix("")
        try:
            module_id = ModuleId.from_path(relative.as_posix())
        except ValueError as exc:
            raise DisciplineError(f"invalid module path {relative.as_posix()!r}") from exc
        modules[module_id] = path.resolve()
    return modules


def _validate_command_path(command_path: str) -> None:
    words = command_path.split()
    if not words or " ".join(words) != command_path:
        raise DisciplineError(f"command path {command_path!r} must be space-separated words")
    if words[0] in _RESERVED_COMMAND_NAMES:
        raise DisciplineError(f"command path {command_path!r} begins with a reserved AGM command")


def _validate_program_reference(
    package: PackageInfo, reference: str, modules: dict[ModuleId, Path]
) -> None:
    module_path, separator, declaration_path = reference.partition("::")
    if not separator or not module_path or not declaration_path:
        raise DisciplineError(
            f"program reference {reference!r} must include module and declaration paths"
        )
    try:
        module_id = ModuleId.from_path(module_path)
    except ValueError as exc:
        raise DisciplineError(
            f"program reference {reference!r} has an invalid module path"
        ) from exc
    declaration = tuple(declaration_path.split("::"))
    if not all(is_identifier(segment) for segment in declaration):
        raise DisciplineError(f"program reference {reference!r} has an invalid declaration path")
    if module_id.segments[0] != package.manifest.name:
        raise DisciplineError(
            f"program reference {reference!r} is outside package {package.manifest.name!r}"
        )
    source_path = modules.get(module_id)
    if source_path is None:
        raise DisciplineError(f"program reference {reference!r} names no package module")

    try:
        program = parse_program(fs.read_text(source_path))
    except (AglSyntaxError, OSError, UnicodeDecodeError) as exc:
        raise DisciplineError(f"cannot parse program module {source_path}: {exc}") from exc
    candidates = {
        tuple(segment.name for segment in function.scope_path) + (function.name,)
        for function in static_function_items(program.body.items)
        if function.is_program
    }
    if declaration not in candidates:
        raise DisciplineError(f"program reference {reference!r} names no program declaration")
