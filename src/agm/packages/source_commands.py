"""Discovery of the commands a package's own programs register.

A ``program def`` may register itself as a package command via ``@command``
(see :mod:`agm.agl.attributes`), and every command's prose is the ``@doc`` of
the program behind it — the manifest authors none. Discovery reads only the
AST of every module in a package's own module tree — no scope resolution, no
dependency graph — so a source tree yields its command table, documentation
included, without paying for a full compile. Those ASTs come from the shared parsed-module cache
(:mod:`agm.agl.modules.parsed_module_cache`), so a scan reuses whatever a
previous scan or graph load already parsed and leaves its own parses there
for them; no module is ever parsed twice for one command.

The result merges into the manifest before anything else (install, ``pkg
check``, archive creation) sees it, so every downstream consumer reads one
complete command table and an immutable store package is never rescanned.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path

from agm.agl.attributes import ProgramRegistration
from agm.agl.diagnostics import AglError
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.loader import load_parsed_module
from agm.agl.scope import recognize_program_registration
from agm.agl.syntax.nodes import FuncDef, Program, static_function_items
from agm.packages.discipline import DisciplineError, package_module_files
from agm.packages.manifest import CommandSpec, ManifestError, validate_command_set
from agm.packages.model import PackageInfo

__all__ = ["package_with_source_commands"]


def package_with_source_commands(package: PackageInfo) -> PackageInfo:
    """Return *package* with its manifest's command table completed from its own source.

    Its programs' ``@command`` registrations merge in, and every command whose
    program lives in this package — registered in source or declared by the
    manifest — takes that program's ``@doc`` as its prose. A path a program
    discovers that the manifest's own ``[commands]`` table also declares is
    legal only when both name the same program; an installed manifest's baked
    table therefore merges back in unchanged and its content hash stays
    stable. Two programs claiming one path is still rejected. The merged
    manifest then runs :func:`agm.packages.manifest.validate_command_set`
    unconditionally, even when no program registers anything, since a lenient
    :func:`agm.packages.manifest.load_manifest` load defers that validation in
    every case — so a conflict with ``[aliases]`` or a now-unsatisfiable
    command group is caught here rather than by a downstream reader that can
    no longer report where the merge happened.

    Raises :class:`DisciplineError` for a module that cannot be read, decoded,
    or parsed, an invalid ``@command`` attribute, two programs claiming the
    same command path, a discovered path that conflicts with a different
    manifest registration, or a merged manifest that fails its own
    consistency rules.
    """
    discovered, docs = _source_commands(package)
    manifest = package.manifest
    conflicting = sorted(
        path
        for path in discovered.keys() & manifest.commands.keys()
        if manifest.commands[path].program != discovered[path].program
    )
    if conflicting:
        path = conflicting[0]
        raise DisciplineError(
            f"command path {path!r} is registered by both the package manifest and "
            f"program {discovered[path].program!r}"
        )
    commands = {
        path: _documented(spec, docs) for path, spec in {**manifest.commands, **discovered}.items()
    }
    if commands != manifest.commands:
        manifest = replace(manifest, commands=commands)
    try:
        validate_command_set(manifest)
    except ManifestError as exc:
        raise DisciplineError(str(exc)) from exc
    return package if manifest is package.manifest else replace(package, manifest=manifest)


def _documented(spec: CommandSpec, docs: Mapping[str, str]) -> CommandSpec:
    """Return *spec* carrying the documentation of the program behind it.

    A command group, and a command naming a program outside this package,
    keep what they already carry.
    """
    if spec.program is None or spec.program not in docs:
        return spec
    return replace(spec, doc=docs[spec.program])


def _source_commands(package: PackageInfo) -> tuple[dict[str, CommandSpec], dict[str, str]]:
    """Return *package*'s source-registered commands and its programs' documentation.

    The documentation is keyed by ``MODULE::DECL`` reference, so a command the
    manifest declares for one of these programs takes its prose too.
    """
    specs: dict[str, CommandSpec] = {}
    docs: dict[str, str] = {}
    for module_id, path in package_module_files(package).items():
        for reference, registration in _module_programs(module_id, path):
            if registration.doc is not None:
                docs[reference] = registration.doc
            if registration.command is None:
                continue
            owner = specs.get(registration.command)
            if owner is not None:
                raise DisciplineError(
                    f"command path {registration.command!r} is registered by both "
                    f"{owner.program!r} and {reference!r}"
                )
            specs[registration.command] = CommandSpec(program=reference, doc=registration.doc)
    return specs, docs


def _module_programs(module_id: ModuleId, path: Path) -> Iterator[tuple[str, ProgramRegistration]]:
    """Yield ``(program reference, registration)`` for one module's programs."""
    program = _parse_module(module_id, path)
    try:
        registrations = [
            (function, recognize_program_registration(function))
            for function in static_function_items(program.body.items)
            if function.is_program
        ]
    except AglError as exc:
        raise DisciplineError(f"invalid attribute in {path}: {exc}") from exc
    for function, registration in registrations:
        yield _declaration_reference(module_id, function), registration


def _parse_module(module_id: ModuleId, path: Path) -> Program:
    """Return one module's AST from the shared parsed-module cache.

    Discovery keys the same cache the module loader keys, so a module parsed
    for a command scan is the one a later graph load reuses, and a module the
    loader already parsed is never parsed again here.
    """
    try:
        return load_parsed_module(module_id, path).program
    except (OSError, UnicodeDecodeError) as exc:
        raise DisciplineError(f"cannot read package module {path}: {exc}") from exc
    except AglError as exc:
        raise DisciplineError(f"cannot parse package module {path}: {exc}") from exc


def _declaration_reference(module_id: ModuleId, function: FuncDef) -> str:
    """Return the ``MODULE::DECL`` reference a manifest command names for *function*."""
    segments = (*(segment.name for segment in function.scope_path), function.name)
    declaration_path = "::".join(segments)
    return f"{module_id.path_str()}::{declaration_path}"
