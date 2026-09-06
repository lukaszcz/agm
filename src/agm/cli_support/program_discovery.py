"""``program def`` declaration discovery and entry-program selection.

:func:`discover_program_declarations_from_source` and
:func:`discover_program_declarations_from_installed_reference` run
``PipelineDriver.discover_programs`` (:class:`~agm.agl.pipeline.ProgramDiscovery`)
as their own standalone pipeline pass.

:func:`select_entry_program` matches a requested ``-p``/``--program`` name
against the entry module's own ``program def`` declarations, shared by
``commands.exec_program.run`` (which turns an unresolved selection into a host
diagnostic) and ``cli._exec_print_help`` (which degrades an unresolved
selection to the host command's own help), so the two surfaces can never
disagree about which program a given name selects. :func:`select_declared_program`
holds the selection rule itself, for the one caller that must apply it before
declarations exist.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from agm.agl.modules.roots import RootSet
from agm.agl.runtime.request import AgentResponse

if TYPE_CHECKING:
    from agm.agl.runtime.types import ProgramDeclInfo
    from agm.cli_support.exec_target import PackageProgramReference
    from agm.cli_support.program_options import ProgramCommand

_ProgramT = TypeVar("_ProgramT")

__all__ = [
    "ExecProgramDiscovery",
    "ProgramSelection",
    "select_declared_program",
    "discover_program_declarations_from_installed_reference",
    "discover_program_declarations_from_source",
    "discover_programs_for_target",
    "program_candidates",
    "select_entry_program",
    "unmatched_program_message",
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

    The single advisory discovery path behind both ``agm exec``'s help
    rendering and its shell completion: it
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
    from agm.core.fs import read_text

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
            source = command if isinstance(target, InlineSource) else read_text(target.path)
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


class ExecProgramDiscovery:
    """One ``agm exec`` invocation's advisory discovery, memoized per source selector.

    The advisory surfaces of a single invocation ask about the same source
    repeatedly: the tail split probes candidate FILE tokens
    (:meth:`command_for_file`), and the help and completion surfaces that
    follow then ask about the token it settled on (:meth:`programs`). Each
    answer costs a full static pipeline pass, so they are memoized for the
    life of the invocation — nothing the pass reads changes within it.

    Held for one invocation rather than in a process-wide cache, so a host
    that runs several invocations in one process (the test suite, a shell
    completion server) never serves one invocation's answer to another's
    configuration context.
    """

    def __init__(
        self,
        *,
        command: str | None,
        requested_program: str | None,
        module_paths: "list[str] | None",
        no_stdlib: bool,
    ) -> None:
        self._command = command
        self._requested_program = requested_program
        self._module_paths = module_paths
        self._no_stdlib = no_stdlib
        self._programs: dict[str | None, tuple[tuple[ProgramDeclInfo, ...], str | None]] = {}

    def programs(self, file: str | None) -> "tuple[tuple[ProgramDeclInfo, ...], str | None]":
        """Return :func:`discover_programs_for_target`'s answer for *file*, once."""
        cached = self._programs.get(file)
        if cached is None:
            cached = discover_programs_for_target(
                file=file,
                command=self._command,
                module_paths=self._module_paths,
                no_stdlib=self._no_stdlib,
            )
            self._programs[file] = cached
        return cached

    def selection(self, file: str | None) -> "ProgramSelection":
        """Return the entry-program selection *file* offers under this invocation's ``-p``."""
        programs, referenced_program = self.programs(file)
        requested = (
            self._requested_program if self._requested_program is not None else referenced_program
        )
        return select_entry_program(programs, requested=requested)

    def command_for_file(self, file: str) -> "ProgramCommand | None":
        """Return the selected program's command for a potential FILE token.

        The tail parser calls this only when pre-FILE program options make a
        spelling-only FILE scan ambiguous, so it may ask about several
        candidate tokens; ``None`` means the token does not name one usable,
        selected program.
        """
        from agm.cli_support.program_options import program_command_for

        return program_command_for(self.selection(file).selected)


def unmatched_program_message(
    requested: str | None, entry_programs: "tuple[ProgramDeclInfo, ...]"
) -> str:
    """Return the diagnostic for a program name that selects no entry program.

    Shared by execution (``commands.exec_program.run``) and the help surface
    (``cli._exec_print_help``), so a name matching nothing is reported the
    same way — and fails the same way — whether the invocation asked to run
    the program or to describe it.
    """
    candidates = program_candidates(entry_programs)
    suffix = f" Candidates: {candidates}" if candidates else ""
    return f"Error: no program matches '{requested}'.{suffix}"


def program_candidates(entry_programs: "tuple[ProgramDeclInfo, ...]") -> str:
    """Return *entry_programs*' declaration paths as one comma-separated list.

    How every inline host diagnostic names the programs a reader may select
    with ``-p``, so the "no program matches" and "multiple programs declared"
    messages spell the same candidates the same way.
    """
    return ", ".join(program.declaration_path for program in entry_programs)


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
    matched nothing. ``requested`` is the name the selection was made with —
    an explicit ``-p`` or the declaration path an installed reference names —
    so a caller reporting an unmatched request names what was actually asked
    for rather than re-deriving it.
    """

    entry_programs: "tuple[ProgramDeclInfo, ...]"
    selected: "ProgramDeclInfo | None"
    requested_unmatched: bool
    requested: str | None = None


def select_declared_program(
    candidates: "Sequence[_ProgramT]",
    *,
    requested: str | None,
    declaration_path: "Callable[[_ProgramT], str]",
) -> "_ProgramT | None":
    """Apply the entry-program selection rule to *candidates*.

    A *requested* name selects only the candidate whose declaration path it
    spells, never a differently named one; with no request, a sole candidate
    is selected implicitly and several are left unselected. Held here so the
    AST pre-pass in ``commands.exec_program.run`` — which must resolve the
    selected program's qualified engine-config table before the pipeline that
    produces :class:`ProgramDeclInfo` values can be built — applies exactly
    the rule :func:`select_entry_program` applies later.
    """
    if requested is not None:
        return next(
            (candidate for candidate in candidates if declaration_path(candidate) == requested),
            None,
        )
    return candidates[0] if len(candidates) == 1 else None


def select_entry_program(
    programs: "tuple[ProgramDeclInfo, ...]", *, requested: str | None
) -> ProgramSelection:
    """Filter *programs* to the entry module's own declarations and apply *requested*."""
    entry_programs = tuple(program for program in programs if program.is_entry)
    selected = select_declared_program(
        entry_programs,
        requested=requested,
        declaration_path=lambda program: program.declaration_path,
    )
    return ProgramSelection(
        entry_programs=entry_programs,
        selected=selected,
        requested_unmatched=requested is not None and selected is None,
        requested=requested,
    )
