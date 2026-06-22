"""Incremental lowering support for the AgL REPL."""

from __future__ import annotations

from dataclasses import dataclass, field

from agm.agl._text import normalize_newlines
from agm.agl.ir.ids import SourceId, SymbolId
from agm.agl.ir.program import ExecutableProgram, SourceFile
from agm.agl.ir.validate import validate_ir
from agm.agl.lower.lowerer import _LinkState, _Lowerer
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.syntax.nodes import Binder, Declaration
from agm.agl.typecheck.env import CheckedProgram
from agm.agl.typecheck.graph import CheckedModuleGraph

__all__ = ["LinkImage", "LoweredReplEntry", "lower_repl_entry", "lower_repl_graph"]


@dataclass(slots=True)
class LinkImage:
    """Persistent linker allocation and metadata image for one REPL session."""

    _state: _LinkState = field(default_factory=_LinkState)
    _linked_modules: set[ModuleId] = field(default_factory=set)

    def symbol_for_decl(self, decl_node_id: int) -> SymbolId | None:
        """Return the persistent symbol allocated for an AST declaration."""
        return self._state.decl_to_sym.get(decl_node_id)


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
) -> LoweredReplEntry:
    """Link one checked REPL entry into ``image`` without resetting any IDs."""
    link = image._state
    source_id = SourceId(link.next_source)
    link.next_source += 1
    link.sources[source_id] = SourceFile(
        display_name=source_label,
        normalized_text=normalize_newlines(source_text),
    )
    lowerer = _Lowerer(checked_entry, link, ENTRY_ID, source_id, source_text)
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
) -> LoweredReplEntry:
    """Incrementally link a checked module graph into a REPL image."""
    from agm.agl.lower.graph import lower_graph

    program = lower_graph(
        checked_graph,
        validate=validate,
        _link=image._state,
        _already_linked=frozenset(image._linked_modules),
        _entry_source_text=source_text,
    )
    image._linked_modules.update(mid for mid in checked_graph.modules if not mid.is_entry)
    entry = checked_graph.modules[checked_graph.entry_id].resolved.program
    last = entry.body.items[-1]
    marker = (
        len(program.modules[program.entry_module].initializers) - 1
        if not isinstance(last, (Binder, Declaration))
        else None
    )
    return LoweredReplEntry(program=program, trailing_expression=marker)
