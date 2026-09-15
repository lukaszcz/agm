"""Whole-program module lowering for the AgL typeless execution IR.

``lower_program`` links a match-compiled program into a single
:class:`~agm.agl.ir.program.ExecutableProgram` with one shared
symbol/function/nominal table and per-module initializer sequences.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

from agm.agl.ir.builtin_vars import BuiltinVarKey, builtin_var_key
from agm.agl.ir.contracts import ContractPayload, ExceptionFieldEncode, ParamDecoder
from agm.agl.ir.ids import FunctionId, NominalId, SourceId, SymbolId
from agm.agl.ir.nodes import IrExpr
from agm.agl.ir.program import (
    DryRunEntry,
    ExecutableModule,
    ExecutableProgram,
    FunctionDescriptor,
    IrProgramParam,
    NominalDescriptor,
    NominalKind,
    SourceFile,
    SymbolDescriptor,
    VariantDescriptor,
)
from agm.agl.ir.static_keys import StaticBindingKey, static_binding_key
from agm.agl.ir.validate import validate_ir
from agm.agl.lower import module as module_cache
from agm.agl.lower.lowerer import (
    _add_builtin_nominals,
    _contract_has_schema,
    _LinkState,
    _Lowerer,
    builtin_nominals_from_declarations,
    reserved_fallback_superseded,
)
from agm.agl.matchcompile import MatchCompiledProgram
from agm.agl.modules.ids import STD_ENV_ID, ModuleId
from agm.agl.self_validation import self_validation_enabled
from agm.agl.semantics.arguments import positional_field_names
from agm.agl.semantics.type_table import TypeDef, TypeTable, is_json_convertible
from agm.agl.semantics.types import EnumType, ExceptionType, RecordType
from agm.agl.syntax.nodes import (
    BuiltinVarDecl,
    FuncDef,
    LetDecl,
    VarDecl,
    simple_let_pattern_name,
    static_items,
)
from agm.agl.type_schema import build_encode_plan, build_param_decoder
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


def _exception_field_encodes(
    type_table: TypeTable,
) -> dict[NominalId, tuple[ExceptionFieldEncode, ...]]:
    """Compile reporting provenance for every exception field, JSON-convertible or not.

    Each field's JSON name is its effective external name (``@json-name`` ??
    ``@name`` ?? declared) — the uncaught-exception report is keyed by it,
    covering every field so no two fields can collide on a fallback declared
    key. A field with no JSON form carries no encode plan and is reported via
    the value-directed serializer instead (see ``pipeline.exception_value_to_run_error``).
    """
    result: dict[NominalId, tuple[ExceptionFieldEncode, ...]] = {}
    for typedef in type_table.entries():
        if typedef.kind != "exception":
            continue
        if _superseded_reserved(typedef, type_table):
            continue
        handle = typedef.handle()
        assert isinstance(handle, ExceptionType)
        result[NominalId(typedef.decl_node_id)] = tuple(
            ExceptionFieldEncode(
                field_name,
                json_name,
                build_encode_plan(field_type, type_table)
                if is_json_convertible(field_type, type_table)
                else None,
            )
            for field_name, json_name, field_type in type_table.json_fields(handle)
        )
    return result


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
        sig = cm.type_env.get_function_signature_by_node_id(item.node_id)
        assert sig is not None, f"compiler bug: no function signature for program {item.name!r}"
        result[fn_node_to_sym[item.node_id]] = _program_signature(sig, type_table)
    return result


def _param_tables(
    modules: Mapping[ModuleId, CheckedModule],
    decl_to_sym: Mapping[int, SymbolId],
    type_table: TypeTable,
) -> tuple[dict[StaticBindingKey, SymbolId], dict[StaticBindingKey, ParamDecoder]]:
    """Build host seed identities and decoders for every linked ``@param`` binding."""
    bindings: dict[StaticBindingKey, SymbolId] = {}
    decoders: dict[StaticBindingKey, ParamDecoder] = {}
    for module_id, checked_module in modules.items():
        attributes = checked_module.resolved.attributes
        for item in static_items(checked_module.resolved.program.body.items):
            if isinstance(item, VarDecl):
                binding_node_id = item.node_id
                name = item.name
            elif isinstance(item, LetDecl):
                binding_node_id = item.pattern.node_id
                let_name = simple_let_pattern_name(item.pattern)
                if let_name is None:
                    continue
                name = let_name
            else:
                continue
            if binding_node_id not in attributes.params:
                continue
            key = static_binding_key(module_id, (segment.name for segment in item.scope_path), name)
            bindings[key] = decl_to_sym[binding_node_id]
            binding_type = checked_module.type_env.get_binding_type(binding_node_id)
            assert binding_type is not None, (
                f"compiler bug: parameter binding {name!r} has no checked type"
            )
            decoders[key] = build_param_decoder(binding_type, type_table)
    return bindings, decoders


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
        display_name = mid.display()
        module_source_text = (
            _entry_source_text
            if mid == checked.entry_id and _entry_source_text is not None
            else cm.source_text
        )
        normalized = normalize_newlines(module_source_text)
        link.sources[source_id] = SourceFile(
            display_name=display_name,
            normalized_text=normalized,
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

    # Step 2: Build nominals from the authoritative TypeTable declarations.
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
        handle = typedef.handle()
        match handle:
            case RecordType():
                link.nominals[nominal] = NominalDescriptor(
                    nominal=nominal,
                    module_id=typedef.module_id,
                    scope_path=typedef.scope_path,
                    declared_name=typedef.name,
                    kind=NominalKind.RECORD,
                    fields=tuple(name for name, _ in typedef.fields),
                    mutable_fields=typedef.mutable_fields,
                    variants=(),
                    positional_fields=positional_field_names(type_table.field_kinds(handle)),
                    bears_name_path=bears_name_path,
                )
            case EnumType():
                link.nominals[nominal] = NominalDescriptor(
                    nominal=nominal,
                    module_id=typedef.module_id,
                    scope_path=typedef.scope_path,
                    declared_name=typedef.name,
                    kind=NominalKind.ENUM,
                    fields=(),
                    variants=tuple(
                        VariantDescriptor(
                            name, tuple(type_table.record_fields(member)), NominalId(member.decl_id)
                        )
                        for name, member in type_table.enum_member_names(handle).items()
                    ),
                    bears_name_path=bears_name_path,
                )
            case _:
                link.nominals[nominal] = NominalDescriptor(
                    nominal=nominal,
                    module_id=typedef.module_id,
                    scope_path=typedef.scope_path,
                    declared_name=typedef.name,
                    kind=NominalKind.EXCEPTION,
                    fields=tuple(type_table.exception_fields(handle).keys()),
                    variants=(),
                    positional_fields=positional_field_names(type_table.field_kinds(handle)),
                    bears_name_path=bears_name_path,
                )

    _add_builtin_nominals(link.nominals, type_table)

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
        for name, generic in cm.type_env.all_generic_types().items():
            typ = generic.template
            nominal = NominalId(typ.decl_id)
            generic_typedef = type_table.get(typ.module_id, typ.name, typ.scope_path)
            assert generic_typedef is not None, (
                f"compiler bug: generic type {name!r} has no TypeDef registered"
            )
            bears_name_path = (
                generic_typedef.decl_node_id == typ.decl_id and typ.decl_id not in inline_member_ids
            )
            if isinstance(typ, RecordType):
                link.nominals[nominal] = NominalDescriptor(
                    nominal=nominal,
                    module_id=typ.module_id,
                    scope_path=typ.scope_path,
                    declared_name=typ.name,
                    kind=NominalKind.RECORD,
                    fields=tuple(fname for fname, _ in generic_typedef.fields),
                    mutable_fields=generic_typedef.mutable_fields,
                    positional_fields=positional_field_names(type_table.field_kinds(typ)),
                    bears_name_path=bears_name_path,
                )
            else:
                link.nominals[nominal] = NominalDescriptor(
                    nominal=nominal,
                    module_id=typ.module_id,
                    scope_path=typ.scope_path,
                    declared_name=typ.name,
                    kind=NominalKind.ENUM,
                    variants=tuple(
                        VariantDescriptor(
                            vname,
                            tuple(type_table.record_fields(member)),
                            NominalId(member.decl_id),
                        )
                        for vname, member in type_table.enum_member_names(typ).items()
                    ),
                    bears_name_path=bears_name_path,
                )

    # Step 3: Phase 1 — pre-allocate every static runtime symbol before any
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
            checked.resource_roots.get(mid),
            has_std_env=STD_ENV_ID in checked.modules,
            stable_ids=_link is None,
            contract_payloads=contract_payloads,
        )
        module_lowerers[mid] = lowerer
        lowerer.prealloc_static_symbols(cm.resolved.program.body)

    # Step 4: Phase 2 — lower bodies in the loader's dependency/SCC order.
    # Type checking intentionally preserves its own presentation order, so it
    # retains the loader's reverse-topological components separately for this
    # execution-sensitive pass. Within an import cycle, the loader's stable
    # member ordering is the only valid tie-break; the entry remains last.
    import_sccs = checked.import_sccs or (tuple(checked.modules),)
    ordered_mids = [
        mid
        for component in import_sccs
        for mid in component
        if mid != checked.entry_id and mid not in _already_linked
    ]
    if checked.entry_id not in _already_linked:
        ordered_mids.append(checked.entry_id)

    executable_modules: dict[ModuleId, ExecutableModule] = {
        mid: ExecutableModule(module_id=mid, initializers=()) for mid in _already_linked
    }
    builtin_setting_defaults: dict[BuiltinVarKey | str, IrExpr] = {}
    for mid in ordered_mids:
        cm = checked.modules[mid]
        lowerer = module_lowerers[mid]
        fingerprint = checked.module_fingerprints.get(mid) if _link is None else None
        key = None
        if fingerprint is not None:
            context = (
                checked.resource_roots.get(mid),
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
            continue
        body = cm.resolved.program.body
        initializers = lowerer.lower_initializers(body, top_level=True)
        executable_module = ExecutableModule(module_id=mid, initializers=initializers)
        executable_modules[mid] = executable_module
        defaults: dict[BuiltinVarKey | str, IrExpr] = {
            builtin_var_key(
                mid, (segment.name for segment in item.scope_path), item.name
            ): lowerer.lower_expr(item.default)
            for item in static_items(body.items)
            if isinstance(item, BuiltinVarDecl) and item.default is not None
        }
        builtin_setting_defaults.update(defaults)
        if key is not None:
            seed = cm.resolved.program.node_id << 32
            module_cache.save(
                key,
                module_cache.capture(
                    executable_module,
                    link,
                    defaults,
                    tuple(lowerer.resources),
                    seed,
                    seed + (1 << 32),
                ),
            )

    # Inventory every declaration, including ones without defaults and ones
    # retained from earlier REPL entries, so structural validation can verify
    # the module, scope path, and name of each structured host-backed key.
    builtin_var_declarations = frozenset(
        builtin_var_key(mid, (segment.name for segment in item.scope_path), item.name)
        for mid, checked_module in checked.modules.items()
        for item in static_items(checked_module.resolved.program.body.items)
        if isinstance(item, BuiltinVarDecl)
    )

    payloads = contract_payloads if contract_payloads is not None else {}
    dry_run_entries: list[DryRunEntry] = []
    runtime_modules = checked.runtime_modules or frozenset(checked.modules)
    for module_id, cm in checked.modules.items():
        if module_id not in runtime_modules:
            continue
        for csr in cm.call_sites:
            dry_run_entries.append(
                DryRunEntry(
                    module=module_id,
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
    exception_field_encodes = _exception_field_encodes(type_table)
    live_functions, live_symbols = _live_functions_and_symbols(link, executable_modules)
    program_symbols = {
        item.node_id: link.fn_node_to_sym[item.node_id]
        for _mid, _cm, item in program_funcdefs(checked.modules)
    }
    param_bindings, param_decoders = _param_tables(checked.modules, link.decl_to_sym, type_table)
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
        contracts=dict(link.contracts),
        dry_run_inventory=dry_run_inventory,
        builtin_nominals=link.builtin_nominals,
        builtin_var_declarations=builtin_var_declarations,
        exception_field_encodes=exception_field_encodes,
        builtin_setting_defaults=builtin_setting_defaults,
    )
    if self_validation_enabled():
        validate_ir(program, deep=True)
    return program
