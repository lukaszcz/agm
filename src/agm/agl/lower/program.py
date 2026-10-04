"""Whole-program module lowering for the AgL typeless execution IR.

``lower_program`` links a match-compiled program into a single
:class:`~agm.agl.ir.program.ExecutableProgram` with one shared
symbol/function/nominal table and per-module initializer sequences.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import cast

from agm.agl.ir.builtin_vars import BuiltinVarKey, builtin_var_key
from agm.agl.ir.contracts import ContractPayload, ExceptionFieldEncode, ParamDecoder
from agm.agl.ir.ids import FunctionId, NominalId, SourceId, SymbolId
from agm.agl.ir.nodes import IrExpr
from agm.agl.ir.program import (
    ExecutableModule,
    ExecutableProgram,
    FunctionDescriptor,
    IrProgramParam,
    NominalDescriptor,
    NominalKind,
    SourceFile,
    SymbolDescriptor,
)
from agm.agl.ir.static_keys import StaticBindingKey, static_binding_key
from agm.agl.ir.validate import validate_ir
from agm.agl.lower import module as module_cache
from agm.agl.lower.lowerer import (
    _add_builtin_nominals,
    _LinkState,
    _Lowerer,
    builtin_nominals_from_declarations,
    reserved_fallback_superseded,
    reserved_field_defaults,
)
from agm.agl.lower.nominal_descriptors import (
    enum_descriptor,
    exception_descriptor,
    record_descriptor,
)
from agm.agl.matchcompile import MatchCompiledProgram
from agm.agl.modules.ids import STD_ENV_ID, ModuleId
from agm.agl.self_validation import self_validation_enabled
from agm.agl.semantics.type_table import TypeDef, TypeTable
from agm.agl.semantics.types import EnumType, ExceptionType, RecordType
from agm.agl.syntax.nodes import (
    BuiltinVarDecl,
    FuncDef,
    static_binding_name,
    static_items,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.type_schema import build_exception_field_encodes, build_param_decoder
from agm.agl.typecheck.env import CheckedModule, FunctionSignature
from agm.agl.typecheck.program import program_funcdefs
from agm.util.text import normalize_newlines

__all__ = ["lower_program"]


def _superseded_reserved(typedef: TypeDef, type_table: TypeTable) -> bool:
    """Return whether *typedef* is a reserved fallback a standard declaration owns.

    A reserved enum's member records go with their enum: a loaded declaration
    brings its own members, so the fallback's are unreachable too.
    """
    if not typedef.module_id.is_reserved:
        return False
    owner_name = typedef.scope_path[0] if typedef.scope_path else typedef.name
    return reserved_fallback_superseded(owner_name, type_table)


def _descriptor_for_skipped_identity(
    nominal: NominalId,
    type_table: TypeTable,
    field_defaults: Mapping[NominalId, tuple[IrExpr | None, ...]],
) -> NominalDescriptor:
    """Build a descriptor for an identity a kept declaration references but that was itself skipped.

    A retained enum's member, or a retained exception's base, may itself be a
    reserved fallback superseded by a loaded standard declaration -- excluded
    from the main declaration pass, but still needed for IR identity and
    validation, so it is added here with ``bears_name_path=False``.
    """
    typedef = type_table.typedef_of(nominal.value)
    handle = typedef.handle()
    if isinstance(handle, RecordType):
        return record_descriptor(
            typedef, handle, type_table, bears_name_path=False, field_defaults=field_defaults
        )
    return exception_descriptor(
        typedef,
        cast(ExceptionType, handle),
        type_table,
        bears_name_path=False,
        field_defaults=field_defaults,
    )


def _add_missing_enum_member_descriptors(
    nominals: dict[NominalId, NominalDescriptor],
    type_table: TypeTable,
    field_defaults: Mapping[NominalId, tuple[IrExpr | None, ...]],
) -> None:
    """Close the nominal table over records referenced by retained enums.

    Usually an enum and all its members are emitted by the same declaration
    pass. A retained enum may instead reference a reserved member whose owning
    fallback enum was superseded, so that record still needs a non-name-bearing
    descriptor for IR identity and validation.
    """
    missing = {
        variant.member
        for descriptor in nominals.values()
        if descriptor.kind is NominalKind.ENUM
        for variant in descriptor.variants
        if variant.member not in nominals
    }
    for nominal in missing:
        nominals[nominal] = _descriptor_for_skipped_identity(nominal, type_table, field_defaults)


def _add_missing_exception_base_descriptors(
    nominals: dict[NominalId, NominalDescriptor],
    type_table: TypeTable,
    field_defaults: Mapping[NominalId, tuple[IrExpr | None, ...]],
) -> None:
    """Close the nominal table over exception bases referenced by retained exceptions.

    A kept exception may extend a reserved base whose owning fallback was
    itself superseded by a loaded standard declaration, so that base still
    needs a non-name-bearing descriptor, transitively up its own base chain.
    """
    pending = [
        descriptor.base
        for descriptor in nominals.values()
        if descriptor.kind is NominalKind.EXCEPTION
    ]
    while pending:
        nominal = pending.pop()
        if nominal is None or nominal in nominals:
            continue
        descriptor = _descriptor_for_skipped_identity(nominal, type_table, field_defaults)
        nominals[nominal] = descriptor
        pending.append(descriptor.base)


def _add_exception_field_encodes(
    encodes: dict[NominalId, tuple[ExceptionFieldEncode, ...]],
    nominals: Mapping[NominalId, NominalDescriptor],
    type_table: TypeTable,
) -> None:
    """Compile every registered exception's field encodes not yet in *encodes*.

    An exception's fields are fixed by its declaration identity, so a REPL
    link state keeps earlier entries' encodes.
    """
    for nominal, descriptor in nominals.items():
        if descriptor.kind is not NominalKind.EXCEPTION or nominal in encodes:
            continue
        typedef = type_table.typedef_of(nominal.value)
        encodes[nominal] = build_exception_field_encodes(typedef.exception_handle(), type_table)


def _program_signature(sig: FunctionSignature, type_table: TypeTable) -> tuple[IrProgramParam, ...]:
    """Build one program's host-facing parameter signature from its checked type."""
    return tuple(
        IrProgramParam(
            name=param.name,
            kind=param.kind,
            required=not param.has_default,
            external_decoder=build_param_decoder(param.type, type_table),
        )
        for param in sig.params
    )


def _program_signatures(
    modules: Mapping[ModuleId, CheckedModule],
    fn_node_to_sym: Mapping[int, SymbolId],
    type_table: TypeTable,
) -> dict[SymbolId, tuple[IrProgramParam, ...]]:
    """Build every linked ``program def``'s host-facing parameter signature."""
    result: dict[SymbolId, tuple[IrProgramParam, ...]] = {}
    for _mid, cm, item in program_funcdefs(modules):
        sig = cm.type_env.function_signature_of(item.node_id)
        result[fn_node_to_sym[item.node_id]] = _program_signature(sig, type_table)
    return result


def _param_tables(
    modules: Mapping[ModuleId, CheckedModule],
    decl_to_sym: Mapping[int, SymbolId],
    type_table: TypeTable,
) -> tuple[
    dict[StaticBindingKey, SymbolId],
    dict[StaticBindingKey, ParamDecoder],
    dict[StaticBindingKey, SourceSpan],
]:
    """Build host seed identities, decoders, and spans for every ``@param`` binding."""
    bindings: dict[StaticBindingKey, SymbolId] = {}
    decoders: dict[StaticBindingKey, ParamDecoder] = {}
    spans: dict[StaticBindingKey, SourceSpan] = {}
    for module_id, checked_module in modules.items():
        for binding in checked_module.resolved.param_bindings():
            name = static_binding_name(binding.item)
            key = static_binding_key(module_id, binding.scope_path, name)
            bindings[key] = decl_to_sym[binding.node_id]
            binding_type = checked_module.type_env.binding_type_of(binding.node_id)
            decoders[key] = build_param_decoder(binding_type, type_table)
            spans[key] = binding.item.span
    return bindings, decoders, spans


def _live_functions_and_symbols(
    link: _LinkState,
    live_modules: Iterable[ModuleId],
) -> tuple[dict[FunctionId, FunctionDescriptor], dict[SymbolId, SymbolDescriptor]]:
    """Drop function/symbol descriptors left behind by modules this program excludes.

    A REPL entry that fails partway through initializing newly imported
    library modules deliberately keeps the link image's allocation delta for
    the incomplete modules (see ``LinkImage.mark_linked``), so a later reload
    of one of them reuses the same declaration IDs. Those modules are then
    never retained by any later program: they stay out of every future
    entry's ``program.modules`` because nothing re-imports a module that
    failed to load. Their allocated ``FunctionDescriptor``/``SymbolDescriptor``
    entries would otherwise linger in ``link.functions``/``link.symbols``
    forever, orphaned from any module in the emitted program -- exactly what
    ``ir.validate`` treats as invalid. Filtering the emitted tables down to
    what the program's own modules own is a no-op for a non-REPL whole-program
    lowering, where ``live_modules`` already covers every module in ``link``.
    """
    live = frozenset(live_modules)
    functions = {
        fn_id: fn_desc for fn_id, fn_desc in link.functions.items() if fn_desc.module_id in live
    }
    symbols = {
        sym_id: sym_desc
        for sym_id, sym_desc in link.symbols.items()
        if (
            sym_desc.owner in live
            if isinstance(sym_desc.owner, ModuleId)
            else sym_desc.owner in functions
        )
    }
    return functions, symbols


def lower_program(
    compiled: MatchCompiledProgram,
    *,
    _link: _LinkState | None = None,
    _already_linked: frozenset[ModuleId] = frozenset(),
    _entry_source_text: str | None = None,
    contract_payloads: Mapping[int, ContractPayload] | None = None,
) -> ExecutableProgram:
    """Lower a whole-program match-compiled artifact to an
    :class:`~agm.agl.ir.program.ExecutableProgram`.

    :param compiled: the statically match-compiled program to lower.
    :returns: the linked ``ExecutableProgram`` ready for evaluation.
    """
    # ``compiled`` validated itself when it was constructed; lowering adds
    # the IR self-check over its own output below.
    checked = compiled.checked
    link = _link if _link is not None else _LinkState()

    # Every per-module TypeEnvironment shares one TypeTable instance (built
    # during checking); pick the entry module's env to reach it.
    type_table = checked.modules[checked.entry_id].type_env.type_table

    # Rebuilt from scratch rather than folded into the link's prior table: the
    # shared ``TypeTable`` is itself what accumulates across REPL entries (a
    # new entry's type environment is seeded from the session's, merging in
    # every prior entry's declarations and its orphan set -- see
    # ``TypeEnvironment.seed_from``/``TypeTable.merge_from``), so rebuilding
    # from it on every lowering already keeps an earlier entry's ``builtin``
    # declaration live for a later entry that does not redeclare it, while
    # automatically dropping one an entry never promoted.
    link.builtin_nominals = builtin_nominals_from_declarations(type_table)

    # Step 1: Register a SourceFile for every module.
    module_source_ids: dict[ModuleId, SourceId] = {}
    for mid, cm in checked.modules.items():
        if mid in _already_linked:
            continue
        source_id = SourceId(cm.resolved.program.node_id if _link is None else link.next_source)
        link.next_source += 1
        # The module's own frontend span already carries the exact label
        # static diagnostics use for it (canonical file path, "<repl>",
        # "<command>", ...; see ``modules.loader.entry_source_id``) -- reuse
        # it rather than the module identity's internal display spelling, so
        # a runtime raise's source (``RunError.source``) matches.
        display_name = cm.resolved.program.span.source.label
        module_source_text = (
            _entry_source_text
            if mid == checked.entry_id and _entry_source_text is not None
            else cm.source_text
        )
        normalized = normalize_newlines(module_source_text)
        link.sources[source_id] = SourceFile(
            display_name=display_name,
            normalized_text=normalized,
            file_name=cm.resolved.program.span.source.label,
        )
        module_source_ids[mid] = source_id

    # An INLINE member record (one declared bare inside its enum body, whose
    # scope path is the enum's own) is a semantic implementation detail of
    # that enum: it receives a descriptor only so runtime identity remains
    # complete, but must not be exposed as a companion namespace leaf of its
    # own (where e.g. ``Step::Continue`` would overwrite ``Step.Continue``).
    # A REFERENCED member (a qualified name naming a record declared
    # elsewhere, e.g. ``M::Go``) keeps its own independent name path -- it
    # may be a companion namespace leaf in its own right, and other enums may
    # reference the same record, so listing it as one enum's member must not
    # suppress it.
    inline_member_ids = {
        typedef.decl_node_id for typedef in type_table.entries() if typedef.is_inline_enum_member
    }

    # Step 2: Phase 1 — pre-allocate every static runtime symbol before any
    # body is lowered. Function ids enable calls across root and named-scope
    # declaration paths; binding symbols make library lets/vars available to
    # their functions while their initializers retain dependency order below.
    module_lowerers: dict[ModuleId, _Lowerer] = {}
    for mid, cm in checked.modules.items():
        if mid in _already_linked:
            continue
        source_id = module_source_ids[mid]
        lowerer = _Lowerer(
            cm,
            link,
            mid,
            source_id,
            _entry_source_text
            if mid == checked.entry_id and _entry_source_text is not None
            else cm.source_text,
            compiled.sites_by_module[mid],
            checked.resource_roots[mid],
            has_std_env=STD_ENV_ID in checked.modules,
            stable_ids=_link is None,
            contract_payloads=contract_payloads,
        )
        module_lowerers[mid] = lowerer
        lowerer.prealloc_static_symbols(cm.resolved.program.body)

    # Step 3: Phase 2 — lower bodies in the loader's dependency/SCC order.
    # Type checking intentionally preserves its own presentation order, so it
    # retains the loader's reverse-topological components separately for this
    # execution-sensitive pass. Within an import cycle, the loader's stable
    # member ordering is the only valid tie-break; the entry remains last.
    ordered_mids = [
        mid
        for component in checked.import_sccs
        for mid in component
        if mid != checked.entry_id and mid not in _already_linked
    ]
    if checked.entry_id not in _already_linked:
        ordered_mids.append(checked.entry_id)

    executable_modules: dict[ModuleId, ExecutableModule] = {
        mid: ExecutableModule(module_id=mid, initializers=()) for mid in _already_linked
    }
    builtin_setting_defaults: dict[BuiltinVarKey | str, IrExpr] = {}
    program_configs: dict[SymbolId, tuple[tuple[StaticBindingKey, IrExpr], ...]] = {}
    for mid in ordered_mids:
        cm = checked.modules[mid]
        lowerer = module_lowerers[mid]
        fingerprint = checked.module_fingerprints.get(mid) if _link is None else None
        key = None
        if fingerprint is not None:
            context = (
                checked.resource_roots[mid],
                STD_ENV_ID in checked.modules,
                sorted(link.builtin_nominals.declared.items()),
                sorted(link.builtin_nominals.members.items()),
                sorted(link.builtin_nominals.standard_members.items()),
                [(nid, (contract_payloads or {}).get(nid)) for nid in sorted(cm.contract_specs)],
            )
            key = hashlib.sha256(fingerprint + repr(context).encode()).digest()
        cached = module_cache.load(key) if key is not None else None
        if cached is not None:
            cached.link_into(link)
            executable_modules[mid] = cached.module
            builtin_setting_defaults.update(cached.defaults)
            program_configs.update(cached.program_configs)
            continue
        body = cm.resolved.program.body
        initializers = lowerer.lower_initializers(body, top_level=True)
        executable_module = ExecutableModule(module_id=mid, initializers=initializers)
        executable_modules[mid] = executable_module
        with lowerer.substituted_constants():
            defaults: dict[BuiltinVarKey | str, IrExpr] = {
                builtin_var_key(
                    mid, (segment.name for segment in item.scope_path), item.name
                ): lowerer.lower_expr(item.default)
                for item in static_items(body.items)
                if isinstance(item, BuiltinVarDecl) and item.default is not None
            }
            module_program_configs: dict[SymbolId, tuple[tuple[StaticBindingKey, IrExpr], ...]] = {
                link.fn_node_to_sym[item.node_id]: tuple(
                    (
                        cm.program_config_targets[entry.key.node_id],
                        lowerer.lower_coerced(entry.value, cm.node_types[entry.key.node_id]),
                    )
                    for entry in raw_entries
                )
                for _mid, _cm, item in program_funcdefs({mid: cm})
                if (raw_entries := cm.resolved.attributes.program_configs.get(item.node_id, ()))
            }
        builtin_setting_defaults.update(defaults)
        program_configs.update(module_program_configs)
        if key is not None:
            seed = cm.resolved.program.node_id << 32
            module_cache.save(
                key,
                module_cache.capture(
                    executable_module,
                    link,
                    defaults,
                    module_program_configs,
                    tuple(lowerer.resources),
                    seed,
                    seed + (1 << 32),
                    dict(lowerer.field_defaults),
                ),
            )

    # Step 4: Build nominals from the authoritative TypeTable declarations.
    # Aliases do not have a TypeDef, so this also excludes their transparent
    # source spellings without comparing concatenated scope names. ``entries()``
    # yields every declaration the table retains -- including a superseded one
    # and one from an unpromoted REPL entry -- so each descriptor also records
    # whether its identity currently bears its own name path, via the same
    # name index ``TypeTable.get`` itself resolves through: an authoritative,
    # order-independent answer to "which declaration does this name mean now?"
    # that the extern boundary later uses to resolve a companion's bare/dotted
    # nominal lookup. A seeded reserved shape a standard-library declaration
    # supersedes is skipped: the source declaration is the identity the host
    # mints for that name.
    for typedef in type_table.entries():
        if _superseded_reserved(typedef, type_table):
            continue
        nominal = NominalId(typedef.decl_node_id)
        bears_name_path = (
            type_table.is_current(typedef) and typedef.decl_node_id not in inline_member_ids
        )
        if (reserved_defaults := reserved_field_defaults(typedef)) is not None:
            link.field_defaults[nominal] = reserved_defaults
        handle = typedef.handle()
        match handle:
            case RecordType():
                link.nominals[nominal] = record_descriptor(
                    typedef,
                    handle,
                    type_table,
                    bears_name_path=bears_name_path,
                    field_defaults=link.field_defaults,
                )
            case EnumType():
                link.nominals[nominal] = enum_descriptor(
                    typedef, handle, type_table, bears_name_path=bears_name_path
                )
            case _:
                link.nominals[nominal] = exception_descriptor(
                    typedef,
                    handle,
                    type_table,
                    bears_name_path=bears_name_path,
                    field_defaults=link.field_defaults,
                )

    _add_builtin_nominals(link.nominals, type_table, link.field_defaults)

    # Generic declarations live outside program_type_table. Runtime nominal
    # identity erases type arguments, so register each generic template once.
    # Field/variant NAMES are read directly off the registered TypeDef (never
    # instantiated — a generic template has no concrete type_args).
    # ``bears_name_path`` compares the identity being registered against the
    # one ``generic_typedef``'s NAME lookup landed on, which is exactly the
    # name-index answer ``TypeTable.is_current`` gives for a non-generic
    # declaration above. Inline members remain excluded just as they are in
    # that pass, because a generic member also appears in this template loop.
    for cm in checked.modules.values():
        for generic in cm.type_env.all_generic_types().values():
            typ = generic.template
            nominal = NominalId(typ.decl_id)
            generic_typedef = type_table.named(typ.module_id, typ.name, typ.scope_path)
            bears_name_path = (
                generic_typedef.decl_node_id == typ.decl_id and typ.decl_id not in inline_member_ids
            )
            if isinstance(typ, RecordType):
                link.nominals[nominal] = record_descriptor(
                    generic_typedef,
                    typ,
                    type_table,
                    bears_name_path=bears_name_path,
                    field_defaults=link.field_defaults,
                )
            else:
                link.nominals[nominal] = enum_descriptor(
                    generic_typedef, typ, type_table, bears_name_path=bears_name_path
                )

    _add_missing_enum_member_descriptors(link.nominals, type_table, link.field_defaults)
    _add_missing_exception_base_descriptors(link.nominals, type_table, link.field_defaults)

    # Inventory every declaration, including ones without defaults and ones
    # retained from earlier REPL entries, so structural validation can verify
    # the module, scope path, and name of each structured host-backed key.
    builtin_var_declarations = frozenset(
        builtin_var_key(mid, (segment.name for segment in item.scope_path), item.name)
        for mid, checked_module in checked.modules.items()
        for item in static_items(checked_module.resolved.program.body.items)
        if isinstance(item, BuiltinVarDecl)
    )

    _add_exception_field_encodes(link.exception_field_encodes, link.nominals, type_table)
    live_functions, live_symbols = _live_functions_and_symbols(link, executable_modules)
    program_symbols = {
        item.node_id: link.fn_node_to_sym[item.node_id]
        for _mid, _cm, item in program_funcdefs(checked.modules)
    }
    param_bindings, param_decoders, param_spans = _param_tables(
        checked.modules, link.decl_to_sym, type_table
    )
    program = ExecutableProgram(
        entry_module=checked.entry_id,
        modules=executable_modules,
        symbols=live_symbols,
        nominals=dict(link.nominals),
        sources=dict(link.sources),
        functions=live_functions,
        program_symbols=program_symbols,
        program_functions={
            symbol: link.fn_node_to_id[node_id] for node_id, symbol in program_symbols.items()
        },
        synthetic_main_symbol=next(
            (
                link.fn_node_to_sym[item.node_id]
                for item in static_items(
                    checked.modules[checked.entry_id].resolved.program.body.items
                )
                if isinstance(item, FuncDef) and item.is_synthetic
            ),
            None,
        ),
        program_signatures=_program_signatures(checked.modules, link.fn_node_to_sym, type_table),
        param_bindings=param_bindings,
        param_decoders=param_decoders,
        param_spans=param_spans,
        contracts=dict(link.contracts),
        target_contracts=dict(link.target_contracts),
        builtin_nominals=link.builtin_nominals,
        builtin_var_declarations=builtin_var_declarations,
        exception_field_encodes=dict(link.exception_field_encodes),
        builtin_setting_defaults=builtin_setting_defaults,
        program_configs=program_configs,
    )
    if self_validation_enabled():
        validate_ir(program, deep=True)
    return program
