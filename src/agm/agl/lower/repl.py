"""Incremental lowering support for the AgL REPL."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from agm.agl.ir.contracts import ContractPayload
from agm.agl.ir.ids import NominalId, SourceId, SymbolId
from agm.agl.ir.program import ExecutableProgram, NominalDescriptor, SourceFile
from agm.agl.ir.validate import validate_ir
from agm.agl.lower.lowerer import InitializerOrigin, _LinkState, _Lowerer
from agm.agl.matchcompile import MatchCompiledModule, MatchCompiledProgram
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.self_validation import self_validation_enabled
from agm.agl.semantics.types import iter_nominal_types
from agm.agl.syntax.nodes import (
    AgentDecl,
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
    from agm.agl.semantics.types import Type
    from agm.agl.typecheck.env import CheckedModule

__all__ = [
    "LinkImage",
    "LoweredReplEntry",
    "ParamOrigin",
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
class ParamOrigin:
    """Pair a source parameter declaration with its lowering symbol identity."""

    declaration_id: int
    symbol: SymbolId


@dataclass(frozen=True, slots=True)
class ReplPromotionPlan:
    """Relate source declarations to the entry initializers that complete them.

    Function closures execute in a leading initializer group even when their
    declarations appear later in source. Non-function source declarations become
    eligible when execution reaches their source-order initializer frontier.
    This makes partial REPL promotion depend on completed IR initializers rather
    than diagnostic source locations. Entry parameters are installed by a
    pre-pass, so their completion comes from the symbols the interpreter
    actually installed rather than from source position.
    """

    source_declaration_ids: tuple[frozenset[int], ...]
    initializers: tuple[InitializerOrigin, ...]
    params: tuple[ParamOrigin, ...]
    declaration_dependencies: Mapping[int, frozenset[int]]

    def completed_declaration_ids(
        self,
        completed_initializer_count: int,
        installed_param_symbols: Collection[SymbolId],
    ) -> frozenset[int]:
        """Return complete declarations, including only installed params.

        The result is dependency-safe and needs no post-correction by callers.
        Parameters are installed by a pre-pass rather than an initializer, so
        their source position cannot establish completion.
        """
        assert 0 <= completed_initializer_count <= len(self.initializers)
        completed: set[int] = set()
        for origin in self.initializers[:completed_initializer_count]:
            completed.update(self.source_declaration_ids[origin.source_index])
        source_frontier = next(
            (
                origin.source_index
                for origin in self.initializers[completed_initializer_count:]
                if not origin.is_function
            ),
            len(self.source_declaration_ids),
        )
        for declaration_ids in self.source_declaration_ids[:source_frontier]:
            completed.update(declaration_ids)

        completed.difference_update(
            origin.declaration_id
            for origin in self.params
            if origin.symbol not in installed_param_symbols
        )

        dependencies = self.declaration_dependencies
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


def _nominal_dependencies(typ: "Type", type_declaration_ids: Mapping[str, int]) -> set[int]:
    """Return entry nominal declarations reachable from a resolved semantic type."""
    return {
        type_declaration_ids[nominal.name]
        for nominal in iter_nominal_types(typ)
        if nominal.module_id.is_entry and nominal.name in type_declaration_ids
    }


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
            and (node.module_qualifier is None or not node.module_qualifier.segments)
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


def _promotion_plan(
    checked: "CheckedModule",
    initializer_origins: tuple[InitializerOrigin, ...],
    decl_to_sym: Mapping[int, SymbolId],
) -> ReplPromotionPlan:
    """Consume lowering's origins and add dependency-safe promotion metadata."""
    items = checked.resolved.program.body.items
    source_declaration_ids = tuple(_item_declaration_ids(item, checked) for item in items)
    params = tuple(
        ParamOrigin(declaration_id=item.node_id, symbol=decl_to_sym[item.node_id])
        for item in items
        if isinstance(item, ParamDecl)
    )
    entry_declaration_ids = frozenset().union(*source_declaration_ids)
    type_declaration_ids = {
        item.name: item.node_id
        for item in items
        if isinstance(item, (EnumDef, ExceptionDef, RecordDef, TypeAlias))
    }
    declaration_dependencies: dict[int, frozenset[int]] = {}
    for item, declaration_ids in zip(items, source_declaration_ids, strict=True):
        if not declaration_ids:
            continue
        item_dependencies = _declaration_dependencies(
            item, checked, entry_declaration_ids, type_declaration_ids
        )
        for declaration_id in declaration_ids:
            declaration_dependencies[declaration_id] = item_dependencies - {declaration_id}
    return ReplPromotionPlan(
        source_declaration_ids=source_declaration_ids,
        initializers=initializer_origins,
        params=params,
        declaration_dependencies=MappingProxyType(declaration_dependencies),
    )


def _trailing_let_value_symbol(last: Item, link: _LinkState) -> SymbolId | None:
    """Return the root value symbol of a trailing destructuring let, if any.

    A simple-name let echoes through its own binding symbol, so only a
    destructuring pattern needs the site's retained root value.
    """
    if not isinstance(last, LetDecl) or simple_let_pattern_name(last.pattern) is not None:
        return None
    return link.let_value_symbols.get(last.node_id)


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
    trailing_let_value_symbol = _trailing_let_value_symbol(last, link)
    if self_validation_enabled():
        validate_ir(program)
    return LoweredReplEntry(
        program=program,
        trailing_expression=trailing_expression,
        trailing_let_value_symbol=trailing_let_value_symbol,
        promotion_plan=_promotion_plan(
            checked_entry,
            link.initializer_origins[ENTRY_ID],
            link.decl_to_sym,
        ),
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
    trailing_let_value_symbol = _trailing_let_value_symbol(last, image._state)
    return LoweredReplEntry(
        program=program,
        trailing_expression=marker,
        trailing_let_value_symbol=trailing_let_value_symbol,
        promotion_plan=_promotion_plan(
            checked.modules[checked.entry_id],
            image._state.initializer_origins[program.entry_module],
            image._state.decl_to_sym,
        ),
    )
