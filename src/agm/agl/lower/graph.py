"""Whole-graph module lowering for the AgL typeless execution IR.

``lower_graph`` links a match-compiled module graph into a single
:class:`~agm.agl.ir.program.ExecutableProgram` with one shared
symbol/function/nominal table and per-module initializer sequences.
"""

from __future__ import annotations

from collections.abc import Mapping

from agm.agl.ir.contracts import ContractPayload
from agm.agl.ir.ids import NominalId, SourceId
from agm.agl.ir.nodes import IrExpr
from agm.agl.ir.program import (
    DryRunEntry,
    ExecutableModule,
    ExecutableProgram,
    NominalDescriptor,
    NominalKind,
    SourceFile,
    VariantDescriptor,
)
from agm.agl.ir.validate import validate_ir
from agm.agl.lower.lowerer import (
    _add_builtin_nominals,
    _contract_has_schema,
    _LinkState,
    _Lowerer,
)
from agm.agl.matchcompile.stage import (
    MatchCompiledModuleGraph,
    validate_match_compiled_graph,
)
from agm.agl.modules.ids import STD_CORE_ID, ModuleId
from agm.agl.semantics.types import EnumType, ExceptionType, RecordType
from agm.agl.syntax.nodes import AgentDecl, FuncDef
from agm.util.text import normalize_newlines

__all__ = ["lower_graph"]


def lower_graph(
    compiled_graph: MatchCompiledModuleGraph,
    *,
    validate: bool = False,
    _link: _LinkState | None = None,
    _already_linked: frozenset[ModuleId] = frozenset(),
    _entry_source_text: str | None = None,
    contract_payloads: Mapping[int, ContractPayload] | None = None,
) -> ExecutableProgram:
    """Lower a whole-graph match-compiled artifact to an
    :class:`~agm.agl.ir.program.ExecutableProgram`.

    :param compiled_graph: the statically match-compiled module graph to lower.
    :param validate: when ``True``, run ``validate_ir(deep=True)`` before returning.
    :returns: the linked ``ExecutableProgram`` ready for evaluation.
    """
    validate_match_compiled_graph(compiled_graph)
    checked_graph = compiled_graph.checked_graph
    link = _link if _link is not None else _LinkState()

    # Every per-module TypeEnvironment shares one TypeTable instance (built
    # during checking); pick the entry module's env to reach it.
    type_table = checked_graph.modules[checked_graph.entry_id].type_env.type_table

    # Step 1: Register a SourceFile for every module.
    module_source_ids: dict[ModuleId, SourceId] = {}
    for mid, cm in checked_graph.modules.items():
        if mid in _already_linked or mid == STD_CORE_ID:
            continue
        source_id = SourceId(link.next_source)
        link.next_source += 1
        display_name = mid.dotted() if not mid.is_entry else "<entry>"
        module_source_text = (
            _entry_source_text
            if mid.is_entry and _entry_source_text is not None
            else cm.source_text
        )
        normalized = normalize_newlines(module_source_text)
        link.sources[source_id] = SourceFile(
            display_name=display_name,
            normalized_text=normalized,
        )
        module_source_ids[mid] = source_id

    # Step 2: Build nominals from graph_type_table + builtins.
    for (mid, name), typ in checked_graph.graph_type_table.items():
        # Skip type aliases: only register the canonical declaration site.
        # A type alias `type Foo = Bar` creates an entry (mid, "Foo") -> RecordType(Bar)
        # where Bar's name differs from "Foo"; skipping it avoids a spurious nominal.
        if isinstance(typ, RecordType):
            if name != typ.name or mid != typ.module_id:
                continue
            nominal = NominalId(typ.module_id, name)
            link.nominals[nominal] = NominalDescriptor(
                nominal=nominal,
                display_name=name,
                kind=NominalKind.RECORD,
                fields=tuple(type_table.record_fields(typ).keys()),
                variants=(),
            )
        elif isinstance(typ, EnumType):
            if name != typ.name or mid != typ.module_id:
                continue
            nominal = NominalId(typ.module_id, name)
            variants = tuple(
                VariantDescriptor(name=vname, fields=tuple(vfields.keys()))
                for vname, vfields in type_table.enum_variants(typ).items()
            )
            link.nominals[nominal] = NominalDescriptor(
                nominal=nominal,
                display_name=name,
                kind=NominalKind.ENUM,
                fields=(),
                variants=variants,
            )
        elif isinstance(typ, ExceptionType):
            if name != typ.name or mid != typ.module_id:
                continue
            nominal = NominalId(typ.module_id, name)
            link.nominals[nominal] = NominalDescriptor(
                nominal=nominal,
                display_name=name,
                kind=NominalKind.EXCEPTION,
                fields=tuple(type_table.exception_fields(typ).keys()),
                variants=(),
            )

    _add_builtin_nominals(link.nominals, type_table)

    # Generic declarations live outside graph_type_table. Runtime nominal
    # identity erases type arguments, so register each generic template once.
    # Field/variant NAMES are read directly off the registered TypeDef (never
    # instantiated — a generic template has no concrete type_args).
    for cm in checked_graph.modules.values():
        for name, generic in cm.type_env.all_generic_types().items():
            typ = generic.template
            nominal = NominalId(typ.module_id, name)
            typedef = type_table.get(typ.module_id, name)
            assert typedef is not None, (
                f"compiler bug: generic type {name!r} has no TypeDef registered"
            )
            if isinstance(typ, RecordType):
                link.nominals[nominal] = NominalDescriptor(
                    nominal=nominal,
                    display_name=name,
                    kind=NominalKind.RECORD,
                    fields=tuple(fname for fname, _ in typedef.fields),
                )
            else:
                link.nominals[nominal] = NominalDescriptor(
                    nominal=nominal,
                    display_name=name,
                    kind=NominalKind.ENUM,
                    variants=tuple(
                        VariantDescriptor(vname, tuple(fname for fname, _ in vfields))
                        for vname, vfields in typedef.variants
                    ),
                )

    # Step 3: Phase 1 — pre-allocate FunctionId + symbol for every FuncDef
    # across ALL modules before any body is lowered (enables cross-module calls).
    module_lowerers: dict[ModuleId, _Lowerer] = {}
    for mid, cm in checked_graph.modules.items():
        if mid in _already_linked or mid == STD_CORE_ID:
            continue
        source_id = module_source_ids[mid]
        lowerer = _Lowerer(
            cm,
            link,
            mid,
            source_id,
            _entry_source_text
            if mid.is_entry and _entry_source_text is not None
            else cm.source_text,
            compiled_graph.cases_by_module[mid],
            contract_payloads=contract_payloads,
        )
        module_lowerers[mid] = lowerer
        body = cm.resolved.program.body
        for item in body.items:
            if isinstance(item, FuncDef) and not item.is_builtin:
                lowerer._prealloc_funcdef(item)
            elif isinstance(item, AgentDecl):
                lowerer._alloc_sym(
                    item.node_id,
                    name=item.name,
                    mutable=False,
                    public=mid.is_entry,
                    owner=mid,
                )

    # Step 4: Phase 2 — lower bodies.
    # Library modules first, entry last, so the insertion order of
    # executable_modules matches dependency order (Python dicts preserve order).
    ordered_mids = [mid for mid in checked_graph.modules if not mid.is_entry and mid != STD_CORE_ID]
    ordered_mids = [mid for mid in ordered_mids if mid not in _already_linked]
    ordered_mids.append(checked_graph.entry_id)

    executable_modules: dict[ModuleId, ExecutableModule] = {
        mid: ExecutableModule(module_id=mid, initializers=()) for mid in _already_linked
    }
    for mid in ordered_mids:
        cm = checked_graph.modules[mid]
        lowerer = module_lowerers[mid]
        body = cm.resolved.program.body
        function_initializers: list[IrExpr] = []
        other_initializers: list[IrExpr] = []
        for item in body.items:
            if mid.is_entry or isinstance(item, FuncDef):
                ir = lowerer.lower_item(item, top_level=mid.is_entry)
                if ir is not None:
                    target = (
                        function_initializers if isinstance(item, FuncDef) else other_initializers
                    )
                    target.append(ir)
        executable_modules[mid] = ExecutableModule(
            module_id=mid,
            initializers=tuple((*function_initializers, *other_initializers)),
        )

    # Collect entry-module params (only the entry module contributes params).
    entry_lowerer = module_lowerers[checked_graph.entry_id]
    payloads = contract_payloads if contract_payloads is not None else {}
    dry_run_entries: list[DryRunEntry] = []
    for cm in checked_graph.modules.values():
        for csr in cm.call_sites:
            dry_run_entries.append(
                DryRunEntry(
                    callee=csr.callee,
                    codec_name=csr.codec_name,
                    target_type_label=repr(csr.target_type),
                    has_schema=_contract_has_schema(
                        cm.contract_specs.get(csr.node_id),
                        payloads.get(csr.node_id),
                    ),
                    parse_policy=csr.parse_policy,
                    line=csr.line,
                    col=csr.col,
                )
            )
    dry_run_inventory = tuple(dry_run_entries)
    program = ExecutableProgram(
        entry_module=checked_graph.entry_id,
        modules=executable_modules,
        symbols=dict(link.symbols),
        nominals=dict(link.nominals),
        sources=dict(link.sources),
        functions=dict(link.functions),
        params=tuple(entry_lowerer._params),
        contracts=dict(link.contracts),
        dry_run_inventory=dry_run_inventory,
    )
    if validate:
        validate_ir(program, deep=True)
    return program
