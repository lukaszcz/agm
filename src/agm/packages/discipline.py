"""Filesystem and command discipline for package directories and archives."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TypeVar, cast

from agm.agl.diagnostics import AglError
from agm.agl.modules.ids import STD_CORE_ID, ModuleId
from agm.agl.modules.loader import build_repl_graph
from agm.agl.modules.roots import RootSet
from agm.agl.parser import AglSyntaxError, parse_program
from agm.agl.parser.parser import parse_program_seeded
from agm.agl.scope import BuiltinKind
from agm.agl.scope.reexports import ReexportCycleError, converge_reexports
from agm.agl.syntax.nodes import (
    Call,
    ExportDecl,
    FuncDef,
    ImportDecl,
    LetDecl,
    OpenDecl,
    ParamDecl,
    Program,
    ScopeRegion,
    VarDecl,
    VarRef,
    static_function_items,
    static_items,
)
from agm.agl.syntax.resources import ResourceError, resolve_resource, resource_path
from agm.agl.syntax.types import ImportMode, UnitT
from agm.agl.syntax.visitor import walk
from agm.command_catalog import RESERVED_COMMAND_NAMES
from agm.core import fs
from agm.packages.manifest import PackageManifest
from agm.packages.model import PackageInfo
from agm.stdlib_locator import shipped_stdlib_root
from agm.util.ident import is_identifier

T = TypeVar("T")
U = TypeVar("U")
_ResourcePaths = dict[tuple[str, ...], BuiltinKind]
_ResourceBindings = dict[tuple[str, ...], BuiltinKind | None]


class DisciplineError(ValueError):
    """Raised when a package directory violates package discipline."""


def validate_package(
    package: PackageInfo, *, dependency_packages: Iterable[PackageInfo] = ()
) -> None:
    """Validate a package's module tree, commands, imports, resources, and references."""

    dependencies = tuple(dependency_packages)
    modules = _validate_package_structure(package, dependency_packages=dependencies)
    _validate_imports(package, modules, dependency_packages=dependencies)


def validate_package_structure(package: PackageInfo) -> None:
    """Validate package structure before its dependency closure is available."""

    _validate_package_structure(package, dependency_packages=())


def _validate_package_structure(
    package: PackageInfo, *, dependency_packages: tuple[PackageInfo, ...]
) -> dict[ModuleId, Path]:
    """Run validation stages that do not require loading the import graph."""

    _validate_package_name(package.manifest.name)
    modules = _module_files(package)
    dependency_modules = {
        module_id: path
        for dependency in dependency_packages
        for module_id, path in _module_files(dependency).items()
    }
    _validate_commands(package.manifest, modules, fs.read_text)
    _validate_resources(
        modules,
        fs.read_text,
        exists=lambda relative: _resource_exists(package.root, relative),
        export_modules=dependency_modules,
        read_export_module=fs.read_text,
    )
    return modules


def _validate_imports(
    package: PackageInfo,
    modules: Mapping[ModuleId, Path],
    *,
    dependency_packages: tuple[PackageInfo, ...],
) -> None:
    """Load every package module with runtime package-visibility rules."""

    if not modules:
        return
    stdlib_root = (
        package.root if package.manifest.name == "std" else shipped_stdlib_root().resolve()
    )
    mounted_packages = (package, *dependency_packages)
    roots = RootSet(
        roots=frozenset({stdlib_root, *(mounted.root for mounted in mounted_packages)}),
        packages=mounted_packages,
        stdlib_roots=frozenset({stdlib_root}),
    )
    entry, next_id = _package_graph_entry(package.manifest.name)
    try:
        build_repl_graph(entry, next_id, path=None, cached={}, roots=roots)
    except (AglError, OSError, UnicodeDecodeError) as exc:
        raise DisciplineError(
            f"cannot load imports for package {package.manifest.name!r}: {exc}"
        ) from exc


def _package_graph_entry(package_name: str) -> tuple[Program, int]:
    """Build a synthetic wildcard entry without reparsing the package name."""

    program, next_id = parse_program_seeded("import package_root/*\n", start_id=0)
    declaration = replace(
        cast(ImportDecl, program.body.items[0]),
        module_path=(package_name,),
    )
    return replace(program, body=replace(program.body, items=(declaration,))), next_id


def validate_archive_package(
    manifest: PackageManifest,
    *,
    archive_paths: Iterable[str],
    read_module: Callable[[str], str],
    dependency_packages: Iterable[PackageInfo] = (),
) -> None:
    """Validate archived module content against its resolved dependencies."""

    _validate_package_name(manifest.name)
    dependencies = tuple(dependency_packages)
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
    dependency_modules = {
        module_id: path
        for dependency in dependencies
        for module_id, path in _module_files(dependency).items()
    }
    _validate_commands(manifest, modules, read_module)
    path_set = frozenset(paths)
    _validate_resources(
        modules,
        read_module,
        exists=lambda relative: _archive_resource_exists(relative, path_set),
        export_modules=dependency_modules,
        read_export_module=fs.read_text,
    )
    available_dependencies = {dependency.manifest.name for dependency in dependencies}
    if all(name == "std" or name in available_dependencies for name in manifest.dependencies):
        _validate_archive_imports(
            manifest,
            modules,
            archive_paths=path_set,
            read_module=read_module,
            dependency_packages=dependencies,
        )


def _validate_archive_imports(
    manifest: PackageManifest,
    modules: Mapping[ModuleId, str],
    *,
    archive_paths: frozenset[str],
    read_module: Callable[[str], str],
    dependency_packages: tuple[PackageInfo, ...],
) -> None:
    """Materialize archived modules so the filesystem graph loader can validate them."""

    try:
        with TemporaryDirectory(prefix="agm-package-check-") as temporary:
            package = PackageInfo(Path(temporary), manifest)
            materialized: dict[ModuleId, Path] = {}
            for module_id, archive_path in modules.items():
                module_path = package.root / module_id.relpath()
                module_path.parent.mkdir(parents=True, exist_ok=True)
                module_path.write_text(read_module(archive_path), encoding="utf-8")
                companion_path = str(PurePosixPath(archive_path).with_suffix(".py"))
                if companion_path in archive_paths:
                    module_path.with_suffix(".py").write_text("", encoding="utf-8")
                materialized[module_id] = module_path
            _validate_imports(
                package,
                materialized,
                dependency_packages=dependency_packages,
            )
    except (OSError, UnicodeDecodeError) as exc:
        raise DisciplineError(
            f"cannot load imports for archive package {manifest.name!r}: {exc}"
        ) from exc


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
    export_modules: Mapping[ModuleId, U],
    read_export_module: Callable[[U], str],
) -> None:
    """Verify that every literal resource target in package modules is present."""
    programs: dict[ModuleId, tuple[T, Program]] = {}
    for module_id, module_path in modules.items():
        try:
            programs[module_id] = (module_path, parse_program(read_module(module_path)))
        except (AglSyntaxError, OSError, UnicodeDecodeError) as exc:
            raise DisciplineError(f"cannot parse package module {module_path}: {exc}") from exc

    dependency_programs: dict[ModuleId, Program] = {}
    for module_id, dependency_path in export_modules.items():
        try:
            dependency_programs[module_id] = parse_program(read_export_module(dependency_path))
        except (AglSyntaxError, OSError, UnicodeDecodeError) as exc:
            raise DisciplineError(
                f"cannot parse dependency module {dependency_path}: {exc}"
            ) from exc
    parsed_modules = {
        **dependency_programs,
        **{module_id: program for module_id, (_, program) in programs.items()},
    }
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
                targets = _resource_export_targets(
                    declaration.module_path, wildcard=declaration.wildcard, exports=exports
                )
                for _, target in targets:
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


def _resource_export_targets(
    module_path: tuple[str, ...],
    *,
    wildcard: bool,
    exports: Mapping[ModuleId, _ResourcePaths],
) -> tuple[tuple[ModuleId, Mapping[tuple[str, ...], BuiltinKind]], ...]:
    """Return concrete resource-bearing modules matched by an import or export."""
    known = {**exports, STD_CORE_ID: _RESOURCE_BUILTINS}
    if not wildcard:
        module_id = ModuleId(module_path)
        return ((module_id, known.get(module_id, {})),)
    return tuple(
        (module_id, paths)
        for module_id, paths in known.items()
        if module_id.segments[: len(module_path)] == module_path
    )


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
            exposed = path if item.rename is None else (item.rename, *path[len(source) :])
            selected[(*prefix, *exposed)] = kind
    return selected


def _resource_imports(
    program: Program, exports: Mapping[ModuleId, _ResourcePaths]
) -> _ResourceBindings:
    """Resolve the visible paths that identify standard resource declarations."""

    paths: _ResourceBindings = dict(_RESOURCE_BUILTINS)
    for declaration in static_items(program.body.items):
        if not isinstance(declaration, ImportDecl):
            continue
        targets = _resource_export_targets(
            declaration.module_path, wildcard=declaration.wildcard, exports=exports
        )
        prefix = tuple(segment.name for segment in declaration.scope_path)
        for module_id, target in targets:
            selected = _select_resource_paths(declaration, target)
            # A scoped import limits bare reach but leaves its module route global.
            route_paths = {path[len(prefix) :]: kind for path, kind in selected.items()}
            routes = (
                ((declaration.alias,),)
                if declaration.alias is not None
                else tuple(module_id.segments[index:] for index in range(len(module_id.segments)))
            )
            for route in routes:
                for path, kind in route_paths.items():
                    paths[(*route, *path)] = kind
            if declaration.is_open or declaration.mode is ImportMode.USING:
                paths.update(selected)

    _apply_resource_opens(program, paths)
    for function in static_function_items(program.body.items):
        declaration_path = tuple(segment.name for segment in function.scope_path) + (function.name,)
        paths[declaration_path] = None
    return paths


def _apply_resource_opens(program: Program, paths: _ResourceBindings) -> None:
    """Apply lexical ``open`` selections to known resource-bearing paths."""

    def visit(items: tuple[object, ...], scope_path: tuple[str, ...]) -> None:
        for item in items:
            if isinstance(item, ScopeRegion):
                visit(item.items, (*scope_path, item.segment.name))
                continue
            if not isinstance(item, OpenDecl):
                continue
            requested = (
                *item.scope_ref.module_route,
                *(segment.name for segment in item.scope_ref.scope_path),
            )
            candidates = (
                *(
                    (*scope_path[:index], *requested)
                    for index in range(len(scope_path), -1, -1)
                    if not item.scope_ref.module_route
                ),
                requested,
            )
            target_prefix = next(
                (
                    prefix
                    for prefix in candidates
                    if any(
                        path[: len(prefix)] == prefix and len(path) > len(prefix) for path in paths
                    )
                ),
                None,
            )
            if target_prefix is None:
                continue
            target = {
                path[len(target_prefix) :]: kind
                for path, kind in paths.items()
                if path[: len(target_prefix)] == target_prefix and len(path) > len(target_prefix)
            }
            selected = _select_open_resource_paths(item, target)
            paths.update({(*scope_path, *path): kind for path, kind in selected.items()})

    visit(program.body.items, ())


def _select_open_resource_paths(
    declaration: OpenDecl, paths: _ResourceBindings
) -> _ResourceBindings:
    """Apply one open declaration's using/hiding selection."""
    if declaration.mode is ImportMode.ALL:
        return dict(paths)
    prefixes = {
        (*tuple(segment.name for segment in item.scope_path), item.name): item.rename
        for item in declaration.items
    }
    if declaration.mode is ImportMode.HIDING:
        return {
            path: kind
            for path, kind in paths.items()
            if not any(path[: len(prefix)] == prefix for prefix in prefixes)
        }
    selected: _ResourceBindings = {}
    for prefix, rename in prefixes.items():
        for path, kind in paths.items():
            if path[: len(prefix)] != prefix:
                continue
            exposed = path if rename is None else (rename, *path[len(prefix) :])
            selected[exposed] = kind
    return selected


def _resource_calls(
    program: Program, paths: Mapping[tuple[str, ...], BuiltinKind | None]
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
        kind = next((paths[path] for path in callee_paths if path in paths), None)
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
        target = path.resolve()
        if not target.is_relative_to(module_root):
            raise DisciplineError(f"package module {relative.as_posix()!r} escapes its module tree")
        try:
            module_id = ModuleId.from_path(relative.as_posix())
        except ValueError as exc:
            raise DisciplineError(f"invalid module path {relative.as_posix()!r}") from exc
        modules[module_id] = target
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
    if any(word.startswith("-") for word in words):
        raise DisciplineError(f"command path {command_path!r} contains an option")


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
        tuple(segment.name for segment in function.scope_path) + (function.name,): function
        for function in static_function_items(program.body.items)
        if function.is_program
    }
    function = candidates.get(declaration)
    if function is None:
        raise DisciplineError(f"program reference {reference!r} names no program declaration")
    if function.type_param_slots or function.params or not isinstance(function.return_type, UnitT):
        raise DisciplineError(
            f"registered program {reference!r} must declare no parameters "
            "and an explicit unit result"
        )
