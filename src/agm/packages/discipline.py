"""Filesystem and command discipline for package directories and archives."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import TypeVar

from agm.agl.modules.ids import ModuleId
from agm.agl.parser import AglSyntaxError, parse_program
from agm.agl.syntax.nodes import Call, VarRef, static_function_items
from agm.agl.syntax.resources import ResourceError, resolve_resource, resource_path
from agm.agl.syntax.visitor import walk
from agm.command_catalog import RESERVED_COMMAND_NAMES
from agm.core import fs
from agm.packages.manifest import PackageManifest
from agm.packages.model import PackageInfo
from agm.util.ident import is_identifier

T = TypeVar("T")


class DisciplineError(ValueError):
    """Raised when a package directory violates package discipline."""


def validate_package(package: PackageInfo) -> None:
    """Validate a package's module tree, commands, and program references."""

    _validate_package_name(package.manifest.name)
    modules = _module_files(package)
    _validate_commands(package.manifest, modules, fs.read_text)
    _validate_resources(
        modules,
        fs.read_text,
        exists=lambda relative: _resource_exists(package.root, relative),
    )


def validate_archive_package(
    manifest: PackageManifest,
    *,
    archive_paths: Iterable[str],
    read_module: Callable[[str], str],
) -> None:
    """Validate archived module content without extracting it to disk."""

    _validate_package_name(manifest.name)
    paths = tuple(archive_paths)
    module_root = manifest.name + "/"
    if not any(path.startswith(module_root) for path in paths):
        raise DisciplineError(f"package {manifest.name!r} requires module tree {manifest.name!r}")
    modules: dict[ModuleId, str] = {}
    for path in paths:
        if not path.startswith(module_root) or not path.endswith(".agl"):
            continue
        try:
            module_id = ModuleId.from_path(PurePosixPath(path).with_suffix("").as_posix())
        except ValueError as exc:
            raise DisciplineError(f"invalid module path {path.removesuffix('.agl')!r}") from exc
        modules[module_id] = path
    _validate_commands(manifest, modules, read_module)
    path_set = frozenset(paths)
    _validate_resources(
        modules,
        read_module,
        exists=lambda relative: _archive_resource_exists(relative, path_set),
    )


def _archive_resource_exists(relative: str, paths: frozenset[str]) -> bool:
    """Return whether a package-root-relative archive resource is present."""
    normalized = PurePosixPath(relative).as_posix()
    return normalized == "." or normalized in paths


def _resource_exists(root: Path, relative: str) -> bool:
    """Return whether a resource exists without leaving a package root."""
    try:
        resolve_resource(root, relative)
    except ResourceError:
        return False
    return True


def _validate_resources(
    modules: Mapping[ModuleId, T],
    read_module: Callable[[T], str],
    *,
    exists: Callable[[str], bool],
) -> None:
    """Verify that every literal resource target in package modules is present."""
    for module_path in modules.values():
        try:
            program = parse_program(read_module(module_path))
        except (AglSyntaxError, OSError, UnicodeDecodeError) as exc:
            raise DisciplineError(f"cannot parse package module {module_path}: {exc}") from exc
        calls: list[tuple[Call, bool]] = []

        def collect_resource_call(node: object) -> None:
            if (
                isinstance(node, Call)
                and isinstance(node.callee, VarRef)
                and node.callee.name in {"resource", "resource-dir"}
            ):
                calls.append((node, node.callee.name == "resource-dir"))

        walk(program, collect_resource_call)
        for call, is_directory in calls:
            try:
                relative = resource_path(call, is_directory=is_directory)
            except ResourceError as exc:
                raise DisciplineError(f"invalid resource call in {module_path}: {exc}") from exc
            if relative is not None and not exists(relative):
                raise DisciplineError(
                    f"resource {relative!r} referenced by {module_path} does not exist"
                )


def _validate_package_name(name: str) -> None:
    if name in RESERVED_COMMAND_NAMES:
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


def _validate_commands(
    manifest: PackageManifest, modules: Mapping[ModuleId, T], read_module: Callable[[T], str]
) -> None:
    for command_path, command in manifest.commands.items():
        _validate_command_path(command_path)
        _validate_program_reference(manifest, command.program, modules, read_module)


def _validate_command_path(command_path: str) -> None:
    words = command_path.split()
    if not words or " ".join(words) != command_path:
        raise DisciplineError(f"command path {command_path!r} must be space-separated words")
    if words[0] in RESERVED_COMMAND_NAMES:
        raise DisciplineError(f"command path {command_path!r} begins with a reserved AGM command")


def _validate_program_reference(
    manifest: PackageManifest,
    reference: str,
    modules: Mapping[ModuleId, T],
    read_module: Callable[[T], str],
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
    if module_id.segments[0] != manifest.name:
        raise DisciplineError(
            f"program reference {reference!r} is outside package {manifest.name!r}"
        )
    source = modules.get(module_id)
    if source is None:
        raise DisciplineError(f"program reference {reference!r} names no package module")

    try:
        program = parse_program(read_module(source))
    except (AglSyntaxError, OSError, UnicodeDecodeError) as exc:
        raise DisciplineError(f"cannot parse program module {source}: {exc}") from exc
    candidates = {
        tuple(segment.name for segment in function.scope_path) + (function.name,)
        for function in static_function_items(program.body.items)
        if function.is_program
    }
    if declaration not in candidates:
        raise DisciplineError(f"program reference {reference!r} names no program declaration")
