"""Incremental lowering support for the AgL REPL."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from agm.agl.ir.contracts import ContractPayload
from agm.agl.ir.ids import SymbolId
from agm.agl.ir.program import ExecutableProgram
from agm.agl.lower.lowerer import InitializerOrigin, _LinkState
from agm.agl.matchcompile import MatchCompiledProgram
from agm.agl.modules.ids import ModuleId
from agm.agl.semantics.types import iter_nominal_types
from agm.agl.syntax.nodes import (
    Binder,
    Declaration,
    EnumDef,
    ExceptionDef,
    FuncDef,
    InfixDecl,
    Item,
    LetDecl,
    RecordDef,
    ScopeRegion,
    TypeAlias,
    VarDecl,
    VariantDef,
    pattern_binder_candidates,
    simple_let_pattern_name,
    static_items,
)

if TYPE_CHECKING:
    from agm.agl.semantics.types import Type
    from agm.agl.typecheck.env import CheckedModule

__all__ = [
    "LinkImage",
    "LoweredReplEntry",
    "ReplPromotionPlan",
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
        """Record initialized library modules as persistently linked.

        The REPL calls this together with caching the matching ``LoadedModule``
        objects, either after full entry success or for the dependency-complete
        library subset left by a partially failed run. An incomplete module is
        never marked linked with fresh declaration IDs that a later reload
        would replace.
        """
        self._linked_modules.update(module_ids)

    def snapshot_state(self) -> _LinkState:
        """Return an independent rollback snapshot of incremental linker state."""
        state = self._state
        return _LinkState(
            next_sym=state.next_sym,
            next_fn=state.next_fn,
            next_source=state.next_source,
            next_contract=state.next_contract,
            decl_to_sym=dict(state.decl_to_sym),
            fn_node_to_sym=dict(state.fn_node_to_sym),
            fn_node_to_id=dict(state.fn_node_to_id),
            symbols=dict(state.symbols),
            functions=dict(state.functions),
            nominals=dict(state.nominals),
            builtin_nominals=state.builtin_nominals,
            sources=dict(state.sources),
            contracts=dict(state.contracts),
            let_value_symbols=dict(state.let_value_symbols),
            initializer_origins=dict(state.initializer_origins),
        )

    def restore_state(self, snapshot: _LinkState) -> None:
        """Restore a previously snapshotted incremental linker state."""
        self._state = snapshot


@dataclass(frozen=True, slots=True)
class ReplPromotionPlan:
    """Relate source declarations to the entry initializers that complete them.

    Function closures execute in a leading initializer group even when their
    declarations appear later in source. Non-function source declarations become
    eligible when execution reaches their source-order initializer frontier.
    This makes partial REPL promotion depend on completed IR initializers rather
    than diagnostic source locations. Runtime references into imported modules
    are retained separately so promotion can require the owning modules to be
    available.
    """

    source_declaration_ids: tuple[frozenset[int], ...]
    initializers: tuple[InitializerOrigin, ...]
    declaration_dependencies: Mapping[int, frozenset[int]]
    imported_module_dependencies: Mapping[int, frozenset[ModuleId]]
    scope_region_source_indices: Mapping[tuple[str, ...], tuple[int, ...]] = field(
        default_factory=dict
    )

    def _source_frontier(self, completed_initializer_indices: Collection[int]) -> int:
        completed_indices = set(completed_initializer_indices)
        return min(
            (
                origin.source_index
                for index, origin in enumerate(self.initializers)
                if index not in completed_indices and not origin.is_function
            ),
            default=len(self.source_declaration_ids),
        )

    def completed_scope_region_paths(
        self, completed_initializer_indices: Collection[int]
    ) -> frozenset[tuple[str, ...]]:
        """Return explicit regions reached before the failed source frontier."""
        frontier = self._source_frontier(completed_initializer_indices)
        return frozenset(
            path
            for path, indices in self.scope_region_source_indices.items()
            if any(index <= frontier for index in indices)
        )

    def completed_declaration_ids(
        self,
        completed_initializer_indices: Collection[int],
        available_module_ids: Collection[ModuleId],
    ) -> frozenset[int]:
        """Return declarations whose local and imported dependencies are available.

        Imported runtime references are safe only when their owning library
        module initialized completely or was already retained by the session.
        """
        completed_indices = set(completed_initializer_indices)
        assert all(0 <= index < len(self.initializers) for index in completed_indices)
        completed: set[int] = set()
        for index in sorted(completed_indices):
            completed.update(self.source_declaration_ids[self.initializers[index].source_index])
        source_frontier = self._source_frontier(completed_indices)
        for declaration_ids in self.source_declaration_ids[:source_frontier]:
            completed.update(declaration_ids)

        dependencies = self.declaration_dependencies
        available_modules = set(available_module_ids)
        while unsafe := {
            declaration_id
            for declaration_id in completed
            if dependencies.get(declaration_id, frozenset()) - completed
            or self.imported_module_dependencies.get(declaration_id, frozenset())
            - available_modules
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
    """Return session-promotable declaration ids introduced by one leaf source item.

    Called only on the flattened, region-transparent sequence ``static_items``
    produces — the same one lowering assigns source indices over — so a
    region's own declaration ids come from its members individually, never
    from the region node itself.
    """
    if isinstance(item, LetDecl):
        return frozenset(
            candidate.node_id
            for candidate in pattern_binder_candidates(item.pattern)
            if checked.pattern_binding_for(candidate.node_id) is not None
        )
    if isinstance(
        item,
        (
            EnumDef,
            ExceptionDef,
            FuncDef,
            InfixDecl,
            RecordDef,
            TypeAlias,
            VarDecl,
        ),
    ):
        return frozenset({item.node_id})
    return frozenset()


def _nominal_dependencies(
    typ: "Type", nominal_dependency_ids: Mapping[int, int], entry_id: ModuleId
) -> set[int]:
    """Map nominal identities in *typ* to their promotable source declarations.

    Only a nominal declared by the entry module itself is promotable; one from a
    library module travels with that module's retained image instead.
    """
    return {
        nominal_dependency_ids[nominal.decl_id]
        for nominal in iter_nominal_types(typ)
        if nominal.module_id == entry_id and nominal.decl_id in nominal_dependency_ids
    }


def _declaration_dependencies(
    item: Item,
    checked: "CheckedModule",
    entry_declaration_ids: frozenset[int],
    nominal_dependency_ids: Mapping[int, int],
    alias_declaration_ids: Mapping[str, frozenset[int]],
    library_module_ids: Collection[ModuleId],
) -> tuple[frozenset[int], frozenset[ModuleId]]:
    """Return local declaration and imported runtime dependencies of one leaf item.

    Called only on the flattened, region-transparent sequence ``static_items``
    produces, so *item* is never a ``ScopeRegion`` itself. Imported nominal
    types are metadata dependencies rather than module-initialization dependencies;
    imported binding and method references require their module's runtime image.
    """
    from agm.agl.syntax.nodes import ElseSentinel, FuncDef
    from agm.agl.syntax.types import AppliedT, NameT
    from agm.agl.syntax.visitor import walk

    type_parameters = (
        frozenset(item.type_params)
        if isinstance(item, (EnumDef, ExceptionDef, FuncDef, RecordDef, TypeAlias))
        else frozenset()
    )
    dependencies: set[int] = set()
    imported_modules: set[ModuleId] = set()

    def collect(node: object) -> None:
        if isinstance(node, ElseSentinel):
            return
        node_id = cast(Item, node).node_id
        if (
            isinstance(node, (AppliedT, NameT))
            and (node.qualifier is None or not node.qualifier.segments)
            and node.name not in type_parameters
        ):
            dependencies.update(alias_declaration_ids.get(node.name, frozenset()))
        binding = checked.binding_for(node_id)
        if binding is not None:
            if binding.decl_node_id in entry_declaration_ids:
                dependencies.add(binding.decl_node_id)
            elif binding.module_id in library_module_ids:
                imported_modules.add(binding.module_id)
        method = checked.method_selection_for(node_id)
        if method is not None and method.module_id in library_module_ids:
            imported_modules.add(method.module_id)
        constructor = checked.constructor_ref_for(node_id)
        if constructor is not None and constructor.owner_decl_node_id in entry_declaration_ids:
            dependencies.add(constructor.owner_decl_node_id)
        typ = checked.node_types.get(node_id)
        if typ is not None:
            dependencies.update(
                _nominal_dependencies(typ, nominal_dependency_ids, checked.module_id)
            )

    walk(item, collect)
    if isinstance(item, FuncDef):
        signature = checked.type_env.get_function_signature_by_node_id(item.node_id)
        assert signature is not None, f"compiler bug: no signature for {item.name!r}"
        for parameter in signature.params:
            dependencies.update(
                _nominal_dependencies(parameter.type, nominal_dependency_ids, checked.module_id)
            )
        dependencies.update(
            _nominal_dependencies(signature.result, nominal_dependency_ids, checked.module_id)
        )
    typedef = (
        checked.type_env.type_table.get(
            checked.module_id,
            item.name,
            tuple(segment.name for segment in item.scope_path),
        )
        if isinstance(item, (EnumDef, ExceptionDef, RecordDef))
        else None
    )
    if typedef is not None:
        for _, field_type in typedef.fields:
            dependencies.update(
                _nominal_dependencies(field_type, nominal_dependency_ids, checked.module_id)
            )
        for member in typedef.members:
            dependencies.update(
                _nominal_dependencies(member, nominal_dependency_ids, checked.module_id)
            )
            for field_type in checked.type_env.type_table.record_fields(member).values():
                dependencies.update(
                    _nominal_dependencies(field_type, nominal_dependency_ids, checked.module_id)
                )
        if typedef.base is not None:
            base_typedef = checked.type_env.type_table.get_by_id(typedef.base)
            if (
                base_typedef is not None
                and base_typedef.module_id == checked.module_id
                and base_typedef.decl_node_id in nominal_dependency_ids
            ):
                dependencies.add(nominal_dependency_ids[base_typedef.decl_node_id])
    if isinstance(item, TypeAlias):
        alias_template = checked.type_env.source_type_template_qname(
            checked.module_id,
            item.name,
            scope_path=tuple(segment.name for segment in item.scope_path),
        )
        assert alias_template is not None, f"compiler bug: no type alias for {item.name!r}"
        dependencies.update(
            _nominal_dependencies(
                alias_template.template, nominal_dependency_ids, checked.module_id
            )
        )
    return frozenset(dependencies), frozenset(imported_modules)


def _promotion_plan(
    checked: "CheckedModule",
    initializer_origins: tuple[InitializerOrigin, ...],
    library_module_ids: Collection[ModuleId],
) -> ReplPromotionPlan:
    """Consume lowering's origins and add dependency-safe promotion metadata.

    Walks the same flattened, region-transparent sequence lowering assigns
    source indices over (``static_items``), so ``source_declaration_ids`` lines
    up with ``InitializerOrigin.source_index`` member for member rather than
    region for region.
    """
    leaf_items = tuple(static_items(checked.resolved.program.body.items))
    source_declaration_ids = tuple(_item_declaration_ids(item, checked) for item in leaf_items)
    entry_declaration_ids = frozenset().union(*source_declaration_ids, frozenset())
    # A top-level nominal handle promotes with its own source item. Synthetic
    # inline-member records promote with their EnumDef owner instead, so a
    # function mentioning ``E::A`` cannot survive without ``E``.
    nominal_dependency_ids: dict[int, int] = {}
    for item in leaf_items:
        if not isinstance(item, (EnumDef, ExceptionDef, RecordDef)):
            continue
        typedef = checked.type_env.type_table.get(
            checked.module_id,
            item.name,
            tuple(segment.name for segment in item.scope_path),
        )
        assert typedef is not None
        nominal_dependency_ids[typedef.decl_node_id] = item.node_id
        if isinstance(item, EnumDef):
            for source_member, member_handle in zip(item.members, typedef.members, strict=True):
                if isinstance(source_member, VariantDef):
                    nominal_dependency_ids[member_handle.decl_id] = item.node_id
    # Aliases are transparent in semantic types, so retain their syntactic
    # declaration dependency separately from nominal identity.
    alias_declaration_ids: dict[str, frozenset[int]] = {}
    for item in leaf_items:
        if isinstance(item, TypeAlias):
            alias_declaration_ids[item.name] = alias_declaration_ids.get(
                item.name, frozenset()
            ) | frozenset({item.node_id})
    region_indices: dict[tuple[str, ...], list[int]] = {}

    def collect_regions(items: tuple[Item, ...], parent: tuple[str, ...] = ()) -> None:
        for item in items:
            if not isinstance(item, ScopeRegion):
                continue
            path = (*parent, item.segment.name)
            source_index = sum(
                leaf.span.start_offset < item.span.start_offset for leaf in leaf_items
            )
            region_indices.setdefault(path, []).append(source_index)
            collect_regions(cast(tuple[Item, ...], item.items), path)

    collect_regions(checked.resolved.program.body.items)
    declaration_dependencies: dict[int, frozenset[int]] = {}
    imported_module_dependencies: dict[int, frozenset[ModuleId]] = {}
    for item, declaration_ids in zip(leaf_items, source_declaration_ids, strict=True):
        if not declaration_ids:
            continue
        item_dependencies, imported_modules = _declaration_dependencies(
            item,
            checked,
            entry_declaration_ids,
            nominal_dependency_ids,
            alias_declaration_ids,
            library_module_ids,
        )
        for declaration_id in declaration_ids:
            declaration_dependencies[declaration_id] = item_dependencies - {declaration_id}
            imported_module_dependencies[declaration_id] = imported_modules
    return ReplPromotionPlan(
        source_declaration_ids=source_declaration_ids,
        initializers=initializer_origins,
        declaration_dependencies=MappingProxyType(declaration_dependencies),
        imported_module_dependencies=MappingProxyType(imported_module_dependencies),
        scope_region_source_indices=MappingProxyType(
            {path: tuple(indices) for path, indices in region_indices.items()}
        ),
    )


def _trailing_let_value_symbol(last: Item, link: _LinkState) -> SymbolId | None:
    """Return the root value symbol of a trailing destructuring let, if any.

    A simple-name let echoes through its own binding symbol, so only a
    destructuring pattern needs the site's retained root value.
    """
    if not isinstance(last, LetDecl) or simple_let_pattern_name(last.pattern) is not None:
        return None
    return link.let_value_symbols.get(last.node_id)


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
    # a module allocates persistent IDs, but evaluation still determines which
    # modules initialized completely. The session calls ``LinkImage.mark_linked``
    # together with caching exactly that completed library set.
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
        if not isinstance(last, (Binder, Declaration, ScopeRegion))
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
            frozenset(checked.modules) - {checked.entry_id},
        ),
    )
