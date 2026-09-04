"""``program def`` declaration discovery and entry-program selection.

:func:`discover_program_declarations_from_source` and
:func:`discover_program_declarations_from_installed_reference` run
``PipelineDriver.discover_programs`` (:class:`~agm.agl.pipeline.ProgramDiscovery`)
as their own standalone pipeline pass.

:func:`select_entry_program` is the one place a requested ``-p``/``--program``
name is matched against the entry module's own ``program def`` declarations,
shared by ``commands.exec_program.run`` (which turns an unresolved selection
into a host diagnostic) and ``cli._exec_print_help`` (which degrades an
unresolved selection to a usage-line listing), so the two surfaces can never
disagree about which program a given name selects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agm.agl.modules.roots import RootSet
from agm.agl.runtime.request import AgentResponse

if TYPE_CHECKING:
    from agm.agl.runtime.types import ProgramDeclInfo
    from agm.cli_support.exec_target import PackageProgramReference

__all__ = [
    "ProgramSelection",
    "discover_program_declarations_from_installed_reference",
    "discover_program_declarations_from_source",
    "discover_programs_for_target",
    "select_entry_program",
]


def discover_program_declarations_from_source(
    source: str,
    *,
    inline_source: bool = False,
    entry_path: Path | None = None,
    roots: RootSet | None = None,
    default_stdlib: bool = True,
) -> "tuple[ProgramDeclInfo, ...]":
    """Discover declared ``program def`` signatures from AgL *source*, degrading to ``()`` on error.

    Shared by the help and shell-completion paths, which both need only the
    discovered programs and must tolerate unreadable/unparsable sources.
    Inline sources receive the same pure synthetic-main wrapper as ``agm exec
    -c``; file sources retain their ordinary unwrapped behavior. Supplying
    their *entry_path* lets the loader discover imports relative to that
    file.
    """
    try:
        from dataclasses import replace

        from agm.agl import PipelineDriver

        runtime = PipelineDriver(agent_dispatcher=lambda request: AgentResponse(content=""))
        if inline_source:
            parsed = runtime.parse_entry(source)
            if parsed.program is not None:
                from agm.agl.parser import wrap_inline_program

                program, next_id = wrap_inline_program(parsed.program, next_node_id=parsed.next_id)
                parsed = replace(parsed, program=program, next_id=next_id)
            prepared = runtime.prepare_parsed_entry(
                parsed, roots=roots, default_stdlib=default_stdlib
            )
        else:
            prepared = runtime.prepare_program(
                source, entry_path=entry_path, roots=roots, default_stdlib=default_stdlib
            )
        return runtime.discover_programs(prepared).programs
    except (Exception, SystemExit):
        return ()


def discover_program_declarations_from_installed_reference(
    resolved: "PackageProgramReference",
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    default_stdlib: bool = True,
) -> "tuple[ProgramDeclInfo, ...]":
    """Discover ``program def`` signatures for an already-resolved installed program reference.

    Help and completion resolve ``PACKAGE/MODULE::PROGRAM`` through the same
    active package selection as execution, then pass the result here. They
    are advisory surfaces, so an unreadable entry or a program that no
    longer parses degrades to no programs.
    """
    try:
        from agm.cli_support.exec_roots import effective_exec_roots

        source = resolved.entry_path.read_text(encoding="utf-8")
        exec_roots = effective_exec_roots(
            entry_path=resolved.entry_path,
            module_paths=[],
            cwd=cwd,
            home=home,
            proj_dir=proj_dir,
        )
        return discover_program_declarations_from_source(
            source,
            entry_path=resolved.entry_path,
            roots=exec_roots.roots,
            default_stdlib=default_stdlib,
        )
    except (Exception, SystemExit):
        return ()


def discover_programs_for_target(
    *,
    file: str | None,
    command: str | None,
    module_paths: "list[str] | None",
    no_stdlib: bool,
) -> "tuple[tuple[ProgramDeclInfo, ...], str | None]":
    """Resolve an ``agm exec`` source selector and discover its ``program def`` declarations.

    The single advisory discovery path behind both ``agm exec --help``'s
    ``Program arguments:`` section and ``agm exec``'s shell completion: it
    resolves the same target the execution path would
    (``exec_target.resolve_exec_target``), assembles the same module roots
    (``exec_roots.effective_exec_roots``), and runs
    :func:`discover_program_declarations_from_source` or
    :func:`discover_program_declarations_from_installed_reference` for it. A
    single implementation is what keeps help and completion from disagreeing
    about which programs a given selector offers.

    Returns the discovered programs and, for an installed
    ``PACKAGE/MODULE::PROGRAM`` reference, the declaration path that reference
    already names — the implicit ``-p``/``--program`` selection a caller
    applies when none was given explicitly. Every failure degrades to
    ``((), None)``: these are advisory surfaces, so an unreadable entry, an
    unresolvable target, or a source that no longer parses shows no program
    arguments rather than failing the command.
    """
    from agm.cli_support.exec_roots import effective_exec_roots
    from agm.cli_support.exec_target import (
        FileEntry,
        InlineSource,
        PackageProgramReference,
        resolve_exec_target,
    )
    from agm.config.context import current_config_context
    from agm.core.fs import read_text_arg

    try:
        context = current_config_context()
        target = resolve_exec_target(
            file=file,
            command=command,
            home=context.home,
            proj_dir=context.proj_dir,
            cwd=context.cwd,
        )
        if isinstance(target, PackageProgramReference):
            programs = discover_program_declarations_from_installed_reference(
                target,
                home=context.home,
                proj_dir=context.proj_dir,
                cwd=context.cwd,
                default_stdlib=not no_stdlib,
            )
            return programs, target.declaration_path
        if isinstance(target, (InlineSource, FileEntry)):
            source = command if isinstance(target, InlineSource) else read_text_arg(target.path)
            entry_path = target.path if isinstance(target, FileEntry) else None
            exec_roots = effective_exec_roots(
                entry_path=entry_path,
                module_paths=[] if module_paths is None else module_paths,
                cwd=context.cwd,
                home=context.home,
                proj_dir=context.proj_dir,
            )
            assert source is not None
            programs = discover_program_declarations_from_source(
                source,
                inline_source=isinstance(target, InlineSource),
                entry_path=entry_path,
                roots=exec_roots.roots,
                default_stdlib=not no_stdlib,
            )
            return programs, None
    except (Exception, SystemExit):
        return (), None
    return (), None


@dataclass(frozen=True, slots=True)
class ProgramSelection:
    """The entry-module programs a host may run, and which one *requested* names.

    ``entry_programs`` is the discovered programs filtered to the entry
    module's own declarations (``ProgramDeclInfo.is_entry``) — an
    imported module's own ``program def`` is discoverable but never runnable
    directly. ``selected`` is the sole entry program when there is exactly
    one and *requested* names none of them, *requested*'s own match when it
    names one, or ``None`` when several are declared and *requested* is
    absent or matches none of them. ``requested_unmatched`` tells those two
    ``None`` causes apart: ``True`` only when *requested* was given and
    matched nothing.
    """

    entry_programs: "tuple[ProgramDeclInfo, ...]"
    selected: "ProgramDeclInfo | None"
    requested_unmatched: bool


def select_entry_program(
    programs: "tuple[ProgramDeclInfo, ...]", *, requested: str | None
) -> ProgramSelection:
    """Filter *programs* to the entry module's own declarations and apply *requested*."""
    entry_programs = tuple(program for program in programs if program.is_entry)
    if requested is not None:
        selected = next(
            (program for program in entry_programs if program.declaration_path == requested), None
        )
        return ProgramSelection(
            entry_programs=entry_programs, selected=selected, requested_unmatched=selected is None
        )
    selected = entry_programs[0] if len(entry_programs) == 1 else None
    return ProgramSelection(
        entry_programs=entry_programs, selected=selected, requested_unmatched=False
    )
