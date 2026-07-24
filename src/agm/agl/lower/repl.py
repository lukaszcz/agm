"""Incremental lowering support for the AgL REPL."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from agm.agl.ir.contracts import ContractPayload
from agm.agl.ir.ids import NominalId, SourceId, SymbolId
from agm.agl.ir.program import ExecutableProgram, NominalDescriptor, SourceFile
from agm.agl.ir.validate import validate_ir
from agm.agl.lower.lowerer import _LinkState, _Lowerer
from agm.agl.matchcompile import MatchCompiledModule, MatchCompiledProgram
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.self_validation import self_validation_enabled
from agm.agl.syntax.nodes import (
    AgentDecl,
    AssignStmt,
    Binder,
    Declaration,
    EnumDef,
    ExceptionDef,
    FuncDef,
    InfixDecl,
    Item,
    LetDecl,
    ParamDecl,
    ProgramDecl,
    RecordDef,
    TypeAlias,
    VarDecl,
    pattern_binder_candidates,
    simple_let_pattern_name,
)
from agm.util.text import normalize_newlines

if TYPE_CHECKING:
    from agm.agl.typecheck.env import CheckedModule

__all__ = [
    "LinkImage",
    "LoweredReplEntry",
    "ReplPromotionPlan",
    "lower_repl_entry",
    "lower_repl_program",
]


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
class ReplPromotionPlan:
    """Relate source declarations to the entry initializers that complete them.

    Function closures execute in a leading initializer group even when their
    declarations appear later in source. Non-function source declarations become
    eligible when execution reaches their source-order initializer frontier.
    This makes partial REPL promotion depend on completed IR initializers rather
    than diagnostic source locations.
    """

    source_declaration_ids: tuple[frozenset[int], ...]
    initializer_declaration_ids: tuple[frozenset[int], ...]
    initializer_source_indices: tuple[int, ...]
    initializer_is_function: tuple[bool, ...]
    declaration_dependencies: tuple[tuple[int, frozenset[int]], ...]

    def completed_declaration_ids(self, completed_initializer_count: int) -> frozenset[int]:
        """Return completed declarations whose entry dependencies are also completed."""
        assert 0 <= completed_initializer_count <= len(self.initializer_declaration_ids)
        completed = set().union(*self.initializer_declaration_ids[:completed_initializer_count])
        pending_source_indices = (
            source_index
            for source_index, is_function in zip(
                self.initializer_source_indices[completed_initializer_count:],
                self.initializer_is_function[completed_initializer_count:],
                strict=True,
            )
            if not is_function
        )
        source_frontier = next(pending_source_indices, len(self.source_declaration_ids))
        for declaration_ids in self.source_declaration_ids[:source_frontier]:
            completed.update(declaration_ids)

        dependencies = dict(self.declaration_dependencies)
        while unsafe := {
            declaration_id
            for declaration_id in completed
            if dependencies.get(declaration_id, frozenset()) - completed
        }:
            completed.difference_update(unsafe)
        return frozenset(completed)


@dataclass(frozen=True, slots=True)
class LoweredReplEntry:
    """One entry linked into a persistent image.

    ``trailing_expression`` is the initializer index whose value the REPL echoes
    for a bare expression. ``trailing_let_value_symbol`` retains a trailing
    destructuring let's complete initializer value without changing the let
    item's language-level unit result. ``promotion_plan`` maps completed entry
    initializers to declarations that may persist after a runtime failure.
    """

    program: ExecutableProgram
    trailing_expression: int | None
    trailing_let_value_symbol: SymbolId | None
    promotion_plan: ReplPromotionPlan


def _item_declaration_ids(item: Item, checked: "CheckedModule") -> frozenset[int]:
    """Return session-promotable declaration ids introduced by one source item."""
    if isinstance(item, LetDecl):
        return frozenset(
            binding.decl_node_id
            for candidate in pattern_binder_candidates(item.pattern)
            if (binding := checked.pattern_binding_for(candidate.node_id)) is not None
        )
    if isinstance(
        item,
        (
            AgentDecl,
            EnumDef,
            ExceptionDef,
            FuncDef,
            InfixDecl,
            ParamDecl,
            ProgramDecl,
            RecordDef,
            TypeAlias,
            VarDecl,
        ),
    ):
        return frozenset({item.node_id})
    return frozenset()


def _nominal_dependencies(typ: object, type_declaration_ids: Mapping[str, int]) -> set[int]:
    """Return entry nominal declarations reachable from a resolved semantic type."""
    from agm.agl.semantics.types import (
        DictType,
        EnumType,
        ExceptionType,
        FunctionType,
        ListType,
        RecordType,
    )

    if isinstance(typ, (RecordType, EnumType)):
        return (
            {type_declaration_ids[typ.name]}
            if typ.module_id.is_entry and typ.name in type_declaration_ids
            else set()
        ) | set().union(
            *(_nominal_dependencies(arg, type_declaration_ids) for arg in typ.type_args)
        )
    if isinstance(typ, ExceptionType):
        return (
            {type_declaration_ids[typ.name]}
            if typ.module_id.is_entry and typ.name in type_declaration_ids
            else set()
        )
    if isinstance(typ, ListType):
        return _nominal_dependencies(typ.elem, type_declaration_ids)
    if isinstance(typ, DictType):
        return _nominal_dependencies(typ.value, type_declaration_ids)
    if isinstance(typ, FunctionType):
        return set().union(
            *(
                _nominal_dependencies(part, type_declaration_ids)
                for part in (*typ.params, typ.result)
            )
        )
    return set()


def _declaration_dependencies(
    item: Item,
    checked: "CheckedModule",
    entry_declaration_ids: frozenset[int],
    type_declaration_ids: Mapping[str, int],
) -> frozenset[int]:
    """Return current-entry runtime and nominal dependencies of one source item."""
    from agm.agl.syntax.nodes import ElseSentinel, FuncDef
    from agm.agl.syntax.types import AppliedT, NameT
    from agm.agl.syntax.visitor import walk

    type_parameters = (
        frozenset(item.type_params)
        if isinstance(item, (EnumDef, ExceptionDef, FuncDef, RecordDef, TypeAlias))
        else frozenset()
    )
    dependencies: set[int] = set()

    def collect(node: object) -> None:
        if isinstance(node, ElseSentinel):
            return
        node_id = cast(Item, node).node_id
        if (
            isinstance(node, (AppliedT, NameT))
            and node.module_qualifier is None
            and node.name not in type_parameters
        ):
            dependency = type_declaration_ids.get(node.name)
            if dependency is not None:
                dependencies.add(dependency)
        binding = checked.binding_for(node_id)
        if binding is not None and binding.decl_node_id in entry_declaration_ids:
            dependencies.add(binding.decl_node_id)
        constructor = checked.constructor_ref_for(node_id)
        if constructor is not None and constructor.owner_decl_node_id in entry_declaration_ids:
            dependencies.add(constructor.owner_decl_node_id)
        typ = checked.node_types.get(node_id)
        if typ is not None:
            dependencies.update(_nominal_dependencies(typ, type_declaration_ids))

    walk(item, collect)
    if isinstance(item, FuncDef):
        signature = checked.type_env.get_function_signature_by_node_id(item.node_id)
        assert signature is not None, f"compiler bug: no signature for {item.name!r}"
        for parameter in signature.params:
            dependencies.update(_nominal_dependencies(parameter.type, type_declaration_ids))
        dependencies.update(_nominal_dependencies(signature.result, type_declaration_ids))
    type_name = (
        item.name if isinstance(item, (EnumDef, ExceptionDef, RecordDef, TypeAlias)) else None
    )
    typedef = checked.type_env.type_table.get(checked.module_id, type_name) if type_name else None
    if typedef is not None:
        for _, field_type in typedef.fields:
            dependencies.update(_nominal_dependencies(field_type, type_declaration_ids))
        for _, fields in typedef.variants:
            for _, field_type in fields:
                dependencies.update(_nominal_dependencies(field_type, type_declaration_ids))
        if typedef.base is not None:
            base_module, base_name = typedef.base
            if base_module.is_entry and base_name in type_declaration_ids:
                dependencies.add(type_declaration_ids[base_name])
    return frozenset(dependencies)


def _promotion_plan(checked: "CheckedModule", program: ExecutableProgram) -> ReplPromotionPlan:
    """Build the source-to-initializer and dependency-safe completion plan."""
    items = checked.resolved.program.body.items
    source_declaration_ids = tuple(_item_declaration_ids(item, checked) for item in items)
    entry_declaration_ids = frozenset().union(*source_declaration_ids)
    type_declaration_ids = {
        item.name: item.node_id
        for item in items
        if isinstance(item, (EnumDef, ExceptionDef, RecordDef, TypeAlias))
    }
    declaration_dependencies = tuple(
        (
            declaration_id,
            _declaration_dependencies(item, checked, entry_declaration_ids, type_declaration_ids)
            - {declaration_id},
        )
        for item, declaration_ids in zip(items, source_declaration_ids, strict=True)
        for declaration_id in declaration_ids
    )
    function_initializers: list[tuple[frozenset[int], int, bool]] = []
    other_initializers: list[tuple[frozenset[int], int, bool]] = []
    for source_index, item in enumerate(items):
        declaration_ids = source_declaration_ids[source_index]
        if isinstance(item, FuncDef) and not item.is_builtin:
            function_initializers.append((declaration_ids, source_index, True))
        elif isinstance(item, (AgentDecl, AssignStmt, LetDecl, VarDecl)):
            other_initializers.append((declaration_ids, source_index, False))
        elif not isinstance(item, Declaration):
            other_initializers.append((frozenset(), source_index, False))
    initializers = (*function_initializers, *other_initializers)
    entry_initializers = program.modules[program.entry_module].initializers
    assert len(initializers) == len(entry_initializers), "compiler bug: REPL promotion plan drifted"
    return ReplPromotionPlan(
        source_declaration_ids=source_declaration_ids,
        initializer_declaration_ids=tuple(item[0] for item in initializers),
        initializer_source_indices=tuple(item[1] for item in initializers),
        initializer_is_function=tuple(item[2] for item in initializers),
        declaration_dependencies=declaration_dependencies,
    )


def lower_repl_entry(
    compiled_entry: MatchCompiledModule,
    *,
    image: LinkImage,
    source_text: str,
    source_label: str,
    contract_payloads: Mapping[int, ContractPayload] | None = None,
) -> LoweredReplEntry:
    """Link one match-compiled REPL entry into ``image`` without resetting any IDs."""
    # ``compiled_entry`` validated itself when it was constructed; lowering adds
    # the IR self-check over its own output below.
    checked_entry = compiled_entry.checked
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
        compiled_entry.sites,
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
    trailing_let_value_symbol = (
        link.let_value_symbols.get(last.node_id)
        if isinstance(last, LetDecl) and simple_let_pattern_name(last.pattern) is None
        else None
    )
    if self_validation_enabled():
        validate_ir(program)
    return LoweredReplEntry(
        program=program,
        trailing_expression=trailing_expression,
        trailing_let_value_symbol=trailing_let_value_symbol,
        promotion_plan=_promotion_plan(checked_entry, program),
    )


def lower_repl_program(
    compiled: MatchCompiledProgram,
    *,
    image: LinkImage,
    source_text: str,
    contract_payloads: Mapping[int, ContractPayload] | None = None,
) -> LoweredReplEntry:
    """Incrementally link a match-compiled module graph into a REPL image."""
    from agm.agl.lower.program import lower_program

    # NOTE: ``image._linked_modules`` is intentionally NOT updated here. Linking
    # a module allocates persistent IDs, but the entry may still fail at runtime;
    # marking modules linked before the entry succeeds would desync the image
    # from the session's cached ``LoadedModule`` set. The session calls
    # ``LinkImage.mark_linked`` once the entry has evaluated successfully.
    program = lower_program(
        compiled,
        _link=image._state,
        _already_linked=frozenset(image._linked_modules),
        _entry_source_text=source_text,
        contract_payloads=contract_payloads,
    )
    checked = compiled.checked
    entry = checked.modules[checked.entry_id].resolved.program
    last = entry.body.items[-1]
    marker = (
        len(program.modules[program.entry_module].initializers) - 1
        if not isinstance(last, (Binder, Declaration))
        else None
    )
    trailing_let_value_symbol = (
        image._state.let_value_symbols.get(last.node_id)
        if isinstance(last, LetDecl) and simple_let_pattern_name(last.pattern) is None
        else None
    )
    return LoweredReplEntry(
        program=program,
        trailing_expression=marker,
        trailing_let_value_symbol=trailing_let_value_symbol,
        promotion_plan=_promotion_plan(checked.modules[checked.entry_id], program),
    )
