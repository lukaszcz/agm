"""Filesystem and command discipline for package directories and archives."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TypeVar, cast

from agm.agl.diagnostics import AglError
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.loader import build_repl_graph
from agm.agl.modules.roots import RootSet
from agm.agl.parser.parser import parse_program_seeded
from agm.agl.scope import BuiltinKind, ModuleResolution, resolve_program
from agm.agl.syntax.nodes import Call, ImportDecl, Program, static_function_items
from agm.agl.syntax.resources import ResourceError, resolve_resource, resource_path
from agm.agl.syntax.types import UnitT
from agm.agl.syntax.visitor import walk
from agm.command_catalog import RESERVED_COMMAND_NAMES, invalid_command_path
from agm.core import fs
from agm.packages.manifest import PackageManifest
from agm.packages.model import PackageInfo, is_std_package_name
from agm.stdlib_locator import shipped_stdlib_root
from agm.util.ident import is_identifier

T = TypeVar("T")
_ModuleResolutions = Mapping[ModuleId, ModuleResolution]


class DisciplineError(ValueError):
    """Raised when a package directory violates package discipline."""


def validate_package(
    package: PackageInfo, *, dependency_packages: Iterable[PackageInfo] = ()
) -> None:
    """Validate a package's module tree, commands, imports, resources, and references."""

    modules = validate_package_structure(package)
    resolutions = _resolve_package_modules(
        package, modules, dependency_packages=tuple(dependency_packages)
    )
    _validate_command_programs(package.manifest, resolutions)
    _validate_resources(
        modules, resolutions, exists=lambda relative: _resource_exists(package.root, relative)
    )


def validate_package_structure(package: PackageInfo) -> dict[ModuleId, Path]:
    """Run the manifest and module-tree checks that read no module source."""

    _validate_package_name(package.manifest.name)
    _validate_command_paths(package.manifest)
    return _module_files(package)


def _resolve_package_modules(
    package: PackageInfo,
    modules: Mapping[ModuleId, Path],
    *,
    dependency_packages: tuple[PackageInfo, ...],
) -> dict[ModuleId, ModuleResolution]:
    """Load and name-resolve a package under runtime package-visibility rules.

    A synthetic wildcard entry pulls the whole module tree into one graph, so
    every module is parsed exactly once and later checks read the compiler's
    own name resolution instead of re-deriving it.
    """

    if not modules:
        return {}
    stdlib_root = (
        package.root
        if is_std_package_name(package.manifest.name)
        else shipped_stdlib_root().resolve()
    )
    mounted_packages = (package, *dependency_packages)
    roots = RootSet(
        roots=frozenset({stdlib_root, *(mounted.root for mounted in mounted_packages)}),
        packages=mounted_packages,
        stdlib_roots=frozenset({stdlib_root}),
    )
    entry, next_id = _package_graph_entry(package.manifest.name)
    try:
        graph, _next_id, _loaded = build_repl_graph(
            entry, next_id, path=None, cached={}, roots=roots
        )
        resolved = resolve_program(graph)
    except (AglError, OSError, UnicodeDecodeError) as exc:
        raise DisciplineError(f"cannot resolve package {package.manifest.name!r}: {exc}") from exc
    return {module_id: resolved.modules[module_id].resolved for module_id in modules}


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
    _validate_command_paths(manifest)
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
    available_dependencies = {dependency.manifest.name for dependency in dependencies}
    if all(
        is_std_package_name(name) or name in available_dependencies
        for name in manifest.dependencies
    ):
        _validate_archive_content(
            manifest,
            modules,
            archive_paths=frozenset(paths),
            read_module=read_module,
            dependency_packages=dependencies,
        )


def _validate_archive_content(
    manifest: PackageManifest,
    modules: Mapping[ModuleId, str],
    *,
    archive_paths: frozenset[str],
    read_module: Callable[[str], str],
    dependency_packages: tuple[PackageInfo, ...],
) -> None:
    """Materialize archived modules so the filesystem graph loader can resolve them."""

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
            resolutions = _resolve_package_modules(
                package, materialized, dependency_packages=dependency_packages
            )
    except (OSError, UnicodeDecodeError) as exc:
        raise DisciplineError(f"cannot load archive package {manifest.name!r}: {exc}") from exc
    _validate_command_programs(manifest, resolutions)
    _validate_resources(
        modules,
        resolutions,
        exists=lambda relative: _archive_resource_exists(relative, archive_paths),
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
    resolutions: _ModuleResolutions,
    *,
    exists: Callable[[str], bool],
) -> None:
    """Verify that every literal resource target in package modules is present."""

    for module_id, module_path in modules.items():
        for call, kind in _resource_calls(resolutions[module_id]):
            try:
                relative = resource_path(call, is_directory=kind is BuiltinKind.RESOURCE_DIR)
            except ResourceError as exc:
                raise DisciplineError(f"invalid resource call in {module_path}: {exc}") from exc
            if relative is not None and not exists(relative):
                raise DisciplineError(
                    f"resource {relative!r} referenced by {module_path} does not exist"
                )


def _resource_calls(resolution: ModuleResolution) -> list[tuple[Call, BuiltinKind]]:
    """Return the calls the scope pass resolved to a resource builtin, in source order."""

    calls: list[tuple[Call, BuiltinKind]] = []

    def collect(node: object) -> None:
        if not isinstance(node, Call):
            return
        kind = resolution.builtin_calls.get(node.node_id)
        if kind is BuiltinKind.RESOURCE or kind is BuiltinKind.RESOURCE_DIR:
            calls.append((node, kind))

    walk(resolution.program, collect)
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


def _validate_command_paths(manifest: PackageManifest) -> None:
    for command_path in manifest.commands:
        invalid = invalid_command_path(command_path)
        if invalid is not None:
            raise DisciplineError(f"command path {command_path!r} {invalid}")


def _validate_command_programs(manifest: PackageManifest, resolutions: _ModuleResolutions) -> None:
    for command in manifest.commands.values():
        _validate_program_reference(manifest, command.program, resolutions)


def _validate_program_reference(
    manifest: PackageManifest, reference: str, resolutions: _ModuleResolutions
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
    resolution = resolutions.get(module_id)
    if resolution is None:
        raise DisciplineError(f"program reference {reference!r} names no package module")

    candidates = {
        tuple(segment.name for segment in function.scope_path) + (function.name,): function
        for function in static_function_items(resolution.program.body.items)
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
