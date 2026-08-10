"""Filesystem and command discipline for package directories and archives."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import TypeVar

from agm.agl.modules.ids import ModuleId
from agm.agl.parser import AglSyntaxError, parse_program
from agm.agl.scope import BuiltinKind
from agm.agl.scope.reexports import ReexportCycleError, converge_reexports
from agm.agl.syntax.nodes import (
    Call,
    ExportDecl,
    FuncDef,
    ImportDecl,
    LetDecl,
    ParamDecl,
    Program,
    VarDecl,
    VarRef,
    static_function_items,
    static_items,
)
from agm.agl.syntax.resources import ResourceError, resolve_resource, resource_path
from agm.agl.syntax.types import ImportMode
from agm.agl.syntax.visitor import walk
from agm.command_catalog import RESERVED_COMMAND_NAMES
from agm.core import fs
from agm.packages.manifest import PackageManifest
from agm.packages.model import PackageInfo
from agm.util.ident import is_identifier

T = TypeVar("T")
_ResourcePaths = dict[tuple[str, ...], BuiltinKind]


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
    """Return whether an archive contains a file or nonempty directory resource."""
    normalized = PurePosixPath(relative).as_posix()
    return (
        normalized == "."
        or normalized in paths
        or any(path.startswith(normalized + "/") for path in paths)
    )


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
    programs: dict[ModuleId, tuple[T, Program]] = {}
    for module_id, module_path in modules.items():
        try:
            programs[module_id] = (module_path, parse_program(read_module(module_path)))
        except (AglSyntaxError, OSError, UnicodeDecodeError) as exc:
            raise DisciplineError(f"cannot parse package module {module_path}: {exc}") from exc

    parsed_modules = {module_id: program for module_id, (_, program) in programs.items()}
    exports = _resource_exports(parsed_modules)
    for module_path, program in programs.values():
        calls = _resource_calls(program, _resource_imports(program, exports))
        for call, is_directory in calls:
            try:
                relative = resource_path(call, is_directory=is_directory)
            except ResourceError as exc:
                raise DisciplineError(f"invalid resource call in {module_path}: {exc}") from exc
            if relative is not None and not exists(relative):
                raise DisciplineError(
                    f"resource {relative!r} referenced by {module_path} does not exist"
                )


_RESOURCE_BUILTINS: _ResourcePaths = {
    ("resource",): BuiltinKind.RESOURCE,
    ("resource-dir",): BuiltinKind.RESOURCE_DIR,
}


def _resource_exports(programs: Mapping[ModuleId, Program]) -> dict[ModuleId, _ResourcePaths]:
    """Resolve package re-exports that preserve a standard resource builtin."""

    exports: dict[ModuleId, _ResourcePaths] = {module_id: {} for module_id in programs}
    declarations = {
        module_id: tuple(
            item for item in static_items(program.body.items) if isinstance(item, ExportDecl)
        )
        for module_id, program in programs.items()
    }

    def propagate() -> bool:
        changed = False
        for module_id, module_declarations in declarations.items():
            module_exports = exports[module_id]
            for declaration in module_declarations:
                target = _resource_export_target(declaration.module_path, exports)
                for path, kind in _select_resource_paths(declaration, target).items():
                    if module_exports.get(path) != kind:
                        module_exports[path] = kind
                        changed = True
        return changed

    try:
        converge_reexports(sum(map(len, declarations.values())), propagate)
    except ReexportCycleError as exc:
        raise DisciplineError(str(exc)) from exc
    return exports


def _resource_export_target(
    module_path: tuple[str, ...], exports: Mapping[ModuleId, _ResourcePaths]
) -> Mapping[tuple[str, ...], BuiltinKind]:
    if module_path == ("std", "core"):
        return _RESOURCE_BUILTINS
    return exports.get(ModuleId(module_path), {})


def _select_resource_paths(
    declaration: ImportDecl | ExportDecl,
    paths: Mapping[tuple[str, ...], BuiltinKind],
) -> dict[tuple[str, ...], BuiltinKind]:
    """Apply an import/export selection while retaining resource declaration identity."""

    prefix = tuple(segment.name for segment in declaration.scope_path)
    if declaration.mode is ImportMode.ALL:
        return {(*prefix, *path): kind for path, kind in paths.items()}
    if declaration.mode is ImportMode.HIDING:
        hidden = {
            (*tuple(segment.name for segment in item.scope_path), item.name)
            for item in declaration.items
        }
        return {
            (*prefix, *path): kind
            for path, kind in paths.items()
            if not any(path[: len(item)] == item for item in hidden)
        }

    selected: dict[tuple[str, ...], BuiltinKind] = {}
    for item in declaration.items:
        source = (*tuple(segment.name for segment in item.scope_path), item.name)
        for path, kind in paths.items():
            if path[: len(source)] != source:
                continue
            exposed = (item.rename or source[-1], *path[len(source) :])
            selected[(*prefix, *exposed)] = kind
    return selected


def _resource_imports(
    program: Program, exports: Mapping[ModuleId, _ResourcePaths]
) -> _ResourcePaths:
    """Resolve the visible paths that identify standard resource declarations."""

    paths = dict(_RESOURCE_BUILTINS)
    for declaration in static_items(program.body.items):
        if not isinstance(declaration, ImportDecl):
            continue
        target = _resource_export_target(declaration.module_path, exports)
        prefix = tuple(segment.name for segment in declaration.scope_path)
        selected = _select_resource_paths(declaration, target)
        # A scoped import limits bare reach but leaves its module route global.
        route_paths = {path[len(prefix) :]: kind for path, kind in selected.items()}
        routes = (
            ((declaration.alias,),)
            if declaration.alias is not None
            else tuple(
                declaration.module_path[index:] for index in range(len(declaration.module_path))
            )
        )
        for route in routes:
            for path, kind in route_paths.items():
                paths[(*route, *path)] = kind
        if declaration.is_open or declaration.mode is ImportMode.USING:
            paths.update(
                {path: kind for path, kind in selected.items() if len(path) == len(prefix) + 1}
            )

    for function in static_function_items(program.body.items):
        if not function.scope_path:
            paths.pop((function.name,), None)
    return paths


def _resource_calls(
    program: Program, paths: Mapping[tuple[str, ...], BuiltinKind]
) -> list[tuple[Call, bool]]:
    """Return calls whose resolved declaration denotes a resource builtin."""

    calls: list[tuple[Call, bool]] = []

    def collect_resource_call(node: object, scope_path: tuple[str, ...]) -> None:
        if not isinstance(node, Call) or not isinstance(node.callee, VarRef):
            return
        route = (
            ()
            if node.callee.qualifier is None
            else tuple(
                part
                for segment in node.callee.qualifier.segments
                for part in segment.name.split("/")
            )
        )
        # Bare imports reach every lexically nested named scope.
        callee_paths = tuple(
            (*route, *scope_path[:index], node.callee.name)
            for index in range(len(scope_path), -1, -1)
        )
        kind = next((paths.get(path) for path in callee_paths if paths.get(path) is not None), None)
        if kind in {BuiltinKind.RESOURCE, BuiltinKind.RESOURCE_DIR}:
            calls.append((node, kind is BuiltinKind.RESOURCE_DIR))

    scoped_items = (FuncDef, LetDecl, ParamDecl, VarDecl)
    for item in static_items(program.body.items):
        scope_path = (
            tuple(segment.name for segment in item.scope_path)
            if isinstance(item, scoped_items)
            else ()
        )
        walk(item, lambda node: collect_resource_call(node, scope_path))
    return calls


def _validate_package_name(name: str) -> None:
    if name in RESERVED_COMMAND_NAMES:
        raise DisciplineError(f"package name {name!r} is reserved by AGM")


def _module_files(package: PackageInfo) -> dict[ModuleId, Path]:
    module_root = package.module_root
    if not module_root.is_relative_to(package.root):
        raise DisciplineError(
            f"package {package.manifest.name!r} module tree escapes the package root"
        )
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
    if words[0].startswith("-"):
        raise DisciplineError(f"command path {command_path!r} begins with an option")


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
