"""Incremental lowering support for the AgL REPL."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from agm.agl.ir.contracts import ContractPayload
from agm.agl.ir.ids import NominalId, SourceId, SymbolId
from agm.agl.ir.program import ExecutableProgram, NominalDescriptor, SourceFile
from agm.agl.ir.validate import validate_ir
from agm.agl.lower.lowerer import _LinkState, _Lowerer
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.syntax.nodes import Binder, Declaration
from agm.agl.typecheck.env import CheckedProgram
from agm.agl.typecheck.graph import CheckedModuleGraph
from agm.util.text import normalize_newlines

__all__ = ["LinkImage", "LoweredReplEntry", "lower_repl_entry", "lower_repl_graph"]


@dataclass(slots=True)
class LinkImage:
    """Persistent linker allocation and metadata image for one REPL session."""

    _state: _LinkState = field(default_factory=_LinkState)
    _linked_modules: set[ModuleId] = field(default_factory=set)

    def symbol_for_decl(self, decl_node_id: int) -> SymbolId | None:
        """Return the persistent symbol allocated for an AST declaration."""
        return self._state.decl_to_sym.get(decl_node_id)

    def mark_linked(self, module_ids: "Iterable[ModuleId]") -> None:
        """Record library modules as persistently linked.

        Called by the REPL session only after an entry evaluates successfully,
        so a runtime failure never leaves a module marked linked without a
        matching cached ``LoadedModule`` (which would skip re-lowering on the
        next import while the reloaded module carries fresh declaration IDs).
        """
        self._linked_modules.update(module_ids)

    def snapshot_nominals(self) -> dict[NominalId, NominalDescriptor]:
        """Return a rollback snapshot of persistent nominal descriptors."""
        return dict(self._state.nominals)

    def restore_nominals(
        self,
        snapshot: Mapping[NominalId, NominalDescriptor],
        nominal_ids: Iterable[NominalId],
    ) -> None:
        """Restore selected nominal descriptors from *snapshot*.

        Runtime-failed REPL entries may have linked type declarations that were
        not promoted statically. Nominals are keyed by stable module/name rather
        than declaration node id, so unpromoted redeclarations must be restored
        explicitly to keep constructor values in later entries consistent with
        the restored type environment.
        """
        for nominal in nominal_ids:
            previous = snapshot.get(nominal)
            if previous is None:
                self._state.nominals.pop(nominal, None)
            else:
                self._state.nominals[nominal] = previous


@dataclass(frozen=True, slots=True)
class LoweredReplEntry:
    """One entry linked into a persistent image.

    ``trailing_expression`` is the initializer index whose value the REPL echoes,
    or ``None`` when the entry does not end in a bare expression.
    """

    program: ExecutableProgram
    trailing_expression: int | None


def lower_repl_entry(
    checked_entry: CheckedProgram,
    *,
    image: LinkImage,
    source_text: str,
    source_label: str,
    validate: bool = False,
    contract_payloads: Mapping[int, ContractPayload] | None = None,
) -> LoweredReplEntry:
    """Link one checked REPL entry into ``image`` without resetting any IDs."""
    link = image._state
    source_id = SourceId(link.next_source)
    link.next_source += 1
    link.sources[source_id] = SourceFile(
        display_name=source_label,
        normalized_text=normalize_newlines(source_text),
    )
    lowerer = _Lowerer(
        checked_entry,
        link,
        ENTRY_ID,
        source_id,
        source_text,
        contract_payloads=contract_payloads,
    )
    program = lowerer.lower()
    items = checked_entry.resolved.program.body.items
    last = items[-1]
    trailing_expression = (
        len(program.modules[ENTRY_ID].initializers) - 1
        if not isinstance(last, (Binder, Declaration))
        else None
    )
    if validate:
        validate_ir(program)
    return LoweredReplEntry(program=program, trailing_expression=trailing_expression)


def lower_repl_graph(
    checked_graph: CheckedModuleGraph,
    *,
    image: LinkImage,
    source_text: str,
    validate: bool = False,
    companion_paths: "Mapping[ModuleId, Path | None] | None" = None,
    contract_payloads: Mapping[int, ContractPayload] | None = None,
) -> LoweredReplEntry:
    """Incrementally link a checked module graph into a REPL image.

    ``companion_paths``, when given, threads each newly-linked module's
    companion path (as recorded by the loader) into every extern lowered from
    it; omitted (the default) when the caller has not loaded that metadata,
    in which case such externs lower with ``companion_path=None``.
    """
    from agm.agl.lower.graph import lower_graph

    # NOTE: ``image._linked_modules`` is intentionally NOT updated here. Linking
    # a module allocates persistent IDs, but the entry may still fail at runtime;
    # marking modules linked before the entry succeeds would desync the image
    # from the session's cached ``LoadedModule`` set. The session calls
    # ``LinkImage.mark_linked`` once the entry has evaluated successfully.
    program = lower_graph(
        checked_graph,
        validate=validate,
        companion_paths=companion_paths,
        _link=image._state,
        _already_linked=frozenset(image._linked_modules),
        _entry_source_text=source_text,
        contract_payloads=contract_payloads,
    )
    entry = checked_graph.modules[checked_graph.entry_id].resolved.program
    last = entry.body.items[-1]
    marker = (
        len(program.modules[program.entry_module].initializers) - 1
        if not isinstance(last, (Binder, Declaration))
        else None
    )
    return LoweredReplEntry(program=program, trailing_expression=marker)
