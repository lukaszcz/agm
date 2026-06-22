"""Whole-graph module lowering for the AgL typeless execution IR (M5).

``lower_graph`` links a :class:`~agm.agl.typecheck.graph.CheckedModuleGraph`
into a single :class:`~agm.agl.ir.program.ExecutableProgram` with one shared
symbol/function/nominal table and per-module initializer sequences.
"""
from __future__ import annotations

from agm.agl._text import normalize_newlines
from agm.agl.ir.ids import NominalId, SourceId
from agm.agl.ir.nodes import IrExpr
from agm.agl.ir.program import (
    ExecutableModule,
    ExecutableProgram,
    NominalDescriptor,
    NominalKind,
    SourceFile,
    VariantDescriptor,
)
from agm.agl.ir.validate import validate_ir
from agm.agl.lower.lowerer import _LinkState, _Lowerer
from agm.agl.modules.ids import PRELUDE_ID, ModuleId
from agm.agl.syntax.nodes import FuncDef
from agm.agl.typecheck.graph import CheckedModuleGraph
from agm.agl.typecheck.types import BUILTIN_EXCEPTIONS, EnumType, ExceptionType, RecordType

__all__ = ["lower_graph"]


def lower_graph(
    checked_graph: CheckedModuleGraph,
    *,
    validate: bool = False,
) -> ExecutableProgram:
    """Lower a whole-graph :class:`~agm.agl.typecheck.graph.CheckedModuleGraph` to an
    :class:`~agm.agl.ir.program.ExecutableProgram`.

    :param checked_graph: the type-checked module graph to lower.
    :param validate: when ``True``, run ``validate_ir(deep=True)`` before returning.
    :returns: the linked ``ExecutableProgram`` ready for evaluation.
    """
    link = _LinkState()

    # Step 1: Register a SourceFile for every module.
    module_source_ids: dict[ModuleId, SourceId] = {}
    for mid, cm in checked_graph.modules.items():
        source_id = SourceId(link.next_source)
        link.next_source += 1
        display_name = mid.dotted() if not mid.is_entry else "<entry>"
        normalized = normalize_newlines(cm.source_text)
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
                fields=tuple(typ.fields.keys()),
                variants=(),
            )
        elif isinstance(typ, EnumType):
            if name != typ.name or mid != typ.module_id:
                continue
            nominal = NominalId(typ.module_id, name)
            variants = tuple(
                VariantDescriptor(name=vname, fields=tuple(vfields.keys()))
                for vname, vfields in typ.variants.items()
            )
            link.nominals[nominal] = NominalDescriptor(
                nominal=nominal,
                display_name=name,
                kind=NominalKind.ENUM,
                fields=(),
                variants=variants,
            )
        elif isinstance(typ, ExceptionType):  # pragma: no cover
            nominal = NominalId(PRELUDE_ID, name)
            link.nominals[nominal] = NominalDescriptor(
                nominal=nominal,
                display_name=name,
                kind=NominalKind.EXCEPTION,
                fields=tuple(typ.fields.keys()),
                variants=(),
            )

    # Add built-in exceptions (ExceptionType not yet supported by the grammar, so
    # these slots are always empty at this point).
    for exc_name, exc_type in BUILTIN_EXCEPTIONS.items():
        nominal = NominalId(PRELUDE_ID, exc_name)
        link.nominals[nominal] = NominalDescriptor(
            nominal=nominal,
            display_name=exc_name,
            kind=NominalKind.EXCEPTION,
            fields=tuple(exc_type.fields.keys()),
            variants=(),
        )

    # Step 3: Phase 1 — pre-allocate FunctionId + symbol for every FuncDef
    # across ALL modules before any body is lowered (enables cross-module calls).
    module_lowerers: dict[ModuleId, _Lowerer] = {}
    for mid, cm in checked_graph.modules.items():
        source_id = module_source_ids[mid]
        lowerer = _Lowerer(cm, link, mid, source_id, cm.source_text)
        module_lowerers[mid] = lowerer
        body = cm.resolved.program.body
        for item in body.items:
            if isinstance(item, FuncDef):
                lowerer._prealloc_funcdef(item)

    # Step 4: Phase 2 — lower bodies.
    # Library modules first, entry last, so the insertion order of
    # executable_modules matches dependency order (Python dicts preserve order).
    ordered_mids = [mid for mid in checked_graph.modules if not mid.is_entry]
    ordered_mids.append(checked_graph.entry_id)

    executable_modules: dict[ModuleId, ExecutableModule] = {}
    for mid in ordered_mids:
        cm = checked_graph.modules[mid]
        lowerer = module_lowerers[mid]
        body = cm.resolved.program.body
        ir_items: list[IrExpr] = []
        for item in body.items:
            if mid.is_entry or isinstance(item, FuncDef):
                ir = lowerer.lower_item(item, top_level=mid.is_entry)
                if ir is not None:
                    ir_items.append(ir)
        executable_modules[mid] = ExecutableModule(
            module_id=mid,
            initializers=tuple(ir_items),
        )

    program = ExecutableProgram(
        entry_module=checked_graph.entry_id,
        modules=executable_modules,
        symbols=dict(link.symbols),
        nominals=dict(link.nominals),
        sources=dict(link.sources),
        functions=dict(link.functions),
    )
    if validate:
        validate_ir(program, deep=True)
    return program
