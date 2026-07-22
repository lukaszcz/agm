"""Program-level type-checking pass for the AgL module system.

``check_program(resolved, capabilities)`` runs the full type-checking pass
over a :class:`~agm.agl.scope.program.ResolvedProgram`, producing a
:class:`CheckedProgram`.

Algorithm
---------
1. **Program type pre-pass** — collect ALL public type declarations across every
   module and resolve their bodies (stamping each
   ``RecordType``/``EnumType``/``ExceptionType`` handle with the owning
   ``ModuleId``), building the shared ``program_type_table``.
   ``RecordType``/``EnumType``/``ExceptionType`` carry no field/variant data
   of their own (their shapes live in the shared ``TypeTable``), so a
   reference to another module's type is a valid handle whether or not that
   type's own body has been resolved yet — body resolution is therefore
   order-free.

   The pre-pass is genuinely whole-program two-phase:

   a. **Headers** — every declared name's handle (or, for a generic
      declaration, its ``GenericTypeDef``) is registered into the shared
      ``program_type_table`` first; type aliases are registered as lazy program
      alias keys because aliases are transparent and have no handle shell.
   b. **Bodies, order-free** — each declaration's body (field/variant type
      expressions) is resolved in a fixed deterministic order (sorted by
      ``(ModuleId.segments, name)``), with no dependency-ordering
      constraint — nominal cycles (same-module, mutual, or spanning any
      number of modules) are legal, since every nominal reference resolves to
      a handle regardless of build order; alias references resolve lazily and
      still reject transparent alias cycles.
   c. **Inhabitation** — once every body is resolved, the whole-program
      inhabitation fixpoint
      (:func:`~agm.agl.semantics.analyses.compute_uninhabited`) rejects the
      first declaration (across the whole program) that has no finite value,
      consistent with the single-module ``_TypeBuilder`` check.

2. **Program function-signature pre-pass** — resolve the parameter and return type
   annotations for EVERY top-level ``FuncDef`` in EVERY module (using the
   ``program_type_table`` and each module's ``ImportEnv`` for cross-module type
   refs), producing a ``program_func_sig_table`` mapping each ``FuncDef.node_id``
   to ``FunctionSignatureRecord``. No function body is checked in this
   phase. The result is used in Phase 3 to
   seed EVERY module's env with ALL function binding types before any body is
   checked, enabling cross-file mutual recursion: a call to a not-yet-checked
   module's function resolves its callee type from the pre-pass table.

3. **Per-module type-check** — for each module, build a module-aware
   :class:`~agm.agl.typecheck.env.TypeEnvironment` seeded with the module's own
   types (from the program table), ALL function binding types (from the
   function-signature pre-pass), and the program table + import env for cross-module
   lookups, then run the existing :class:`~agm.agl.typecheck.builder._TypeBuilder`
   and :class:`~agm.agl.typecheck.checker._Checker` logic.

Single-module equivalence
-------------------------
A single-module (entry-only) program checked via :func:`check_program` is
equivalent to calling :func:`~agm.agl.typecheck.checker.check` directly on the
entry's :class:`~agm.agl.scope.symbols.ModuleResolution`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from agm.agl.capabilities import HostCapabilities
from agm.agl.diagnostics import Diagnostic
from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import ImportEnv
from agm.agl.scope.program import ResolvedProgram
from agm.agl.scope.symbols import ModuleResolution
from agm.agl.semantics.analyses import compute_uninhabited, uninhabitable_message
from agm.agl.semantics.type_table import TypeTable, create_seeded_type_table, decl_key_sort_key
from agm.agl.semantics.types import Type
from agm.agl.syntax.nodes import (
    BuiltinVarDecl,
    EnumDef,
    ExceptionDef,
    FuncDef,
    ParamKind,
    Program,
    RecordDef,
    TypeAlias,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.builder import _TypeBuilder
from agm.agl.typecheck.checker import _check_prepared_module
from agm.agl.typecheck.env import (
    AglTypeError,
    CheckedModule,
    ConstructorSignature,
    FunctionSignature,
    GenericAliasDef,
    GenericTypeDef,
    TypeEnvironment,
    _assert_checked_types_closed,
    assert_checked_output_closed,
)
from agm.agl.typecheck.function_inference import (
    FunctionReturnSource,
    FunctionSignatureRecord,
    resolve_function_header,
)

# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckedProgram:
    """Immutable output of :func:`check_program`.

    ``modules``
        Maps each :class:`~agm.agl.modules.ids.ModuleId` to its
        :class:`CheckedModule`.
    ``entry_id``
        Always :data:`~agm.agl.modules.ids.ENTRY_ID`.
    ``program_type_table``
        Whole-program type table mapping ``(ModuleId, name)`` to the fully-built
        :class:`~agm.agl.semantics.types.Type` object stamped with the owning
        ``module_id``.  Built in the program pre-pass; shared (read-only) across
        all per-module environments.
    ``warnings``
        All non-fatal type-check diagnostics collected across all modules, in
        module-traversal order.
    """

    modules: dict[ModuleId, CheckedModule]
    entry_id: ModuleId
    program_type_table: dict[tuple[ModuleId, str], Type]
    warnings: tuple[Diagnostic, ...]
    capabilities: HostCapabilities | None = None


def _assert_checked_module_closed(module: CheckedModule) -> None:
    """Assert that one program module is safe to pass to the lowerer."""
    assert_checked_output_closed(
        node_types=module.node_types,
        contract_specs=module.contract_specs,
        call_sites=module.call_sites,
        type_env=module.type_env,
        function_signatures=module.function_signatures,
        cast_specs=module.cast_specs,
        argument_bindings=module.argument_bindings,
        owner=f"checked module {module.module_id.path_str()}",
    )


def assert_checked_program_closed(checked: CheckedProgram) -> None:
    """Assert that all program-level checked output is safe to lower."""
    for module in checked.modules.values():
        _assert_checked_module_closed(module)
    _assert_checked_types_closed(checked.program_type_table.values(), owner="checked module graph")
    # The remaining whole-program tables (the shared TypeTable and the generic /
    # alias / constructor maps) are the same instances on every module env, so
    # validate them once here rather than on every per-module seal.
    checked.modules[checked.entry_id].type_env.assert_shared_tables_closed()


# ---------------------------------------------------------------------------
# Program type pre-pass helpers
# ---------------------------------------------------------------------------


def _collect_shells_only(builder: _TypeBuilder, program: object) -> None:
    """Run only phase 1 (shell registration) of ``_TypeBuilder.collect``.

    Delegates to :meth:`~agm.agl.typecheck.builder._TypeBuilder.collect_shells_only`,
    the public API added to ``_TypeBuilder`` for this purpose.
    """
    assert isinstance(program, Program)
    builder.collect_shells_only(program)


def _sync_program_env_extensions(
    mid: ModuleId,
    env: TypeEnvironment,
    program_generic_table: dict[tuple[ModuleId, str], GenericTypeDef],
    program_ctor_sig_table: dict[tuple[ModuleId, str, str | None], ConstructorSignature],
    program_ctor_field_kinds_table: dict[
        tuple[ModuleId, str, str | None], tuple[tuple[str, ParamKind], ...]
    ],
) -> None:
    """Copy generic type and constructor metadata built for one module into program tables."""
    for generic_name, gdef in env.all_generic_types().items():
        program_generic_table[(mid, generic_name)] = gdef
    for (owner_name, variant), sig in env.all_constructor_sigs():
        program_ctor_sig_table[(mid, owner_name, variant)] = sig
    for (owner_name, variant), kinds in env.all_constructor_field_kinds():
        program_ctor_field_kinds_table[(mid, owner_name, variant)] = kinds


def _resolve_body_for_one(
    mid: ModuleId,
    name: str,
    per_module_builders: dict[ModuleId, _TypeBuilder],
    program_type_table: dict[tuple[ModuleId, str], Type],
    program_generic_table: dict[tuple[ModuleId, str], GenericTypeDef],
    program_alias_table: dict[tuple[ModuleId, str], GenericAliasDef],
    program_ctor_sig_table: dict[tuple[ModuleId, str, str | None], ConstructorSignature],
    program_ctor_field_kinds_table: dict[
        tuple[ModuleId, str, str | None], tuple[tuple[str, ParamKind], ...]
    ],
    resolved: ResolvedProgram,
    cross_envs: dict[ModuleId, TypeEnvironment],
) -> None:
    """Resolve the body of one type ``(mid, name)`` and update the program table.

    Called once per key in a fixed deterministic order (no dependency
    ordering): every type reference is a handle, valid regardless of whether
    the referenced type's own body has been resolved yet.
    """
    cross_env = cross_envs[mid]
    builder = per_module_builders[mid]

    program = resolved.modules[mid].resolved.program
    assert isinstance(program, Program)

    for item in program.body.items:
        if isinstance(item, RecordDef) and item.name == name:
            builder.build_record(item.name)
            t = cross_env.get_type(item.name)
            if t is not None:
                # Non-generic record: update the program table with the fully-built type.
                program_type_table[(mid, item.name)] = t
            # Generic record: body registered in _generic_types (no _types entry);
            # program_type_table retains the handle from Step A.  Cross-module generic
            # constructor calls use _program_generic_table / _program_ctor_sig_table instead.
            break
        if isinstance(item, EnumDef) and item.name == name:
            builder.build_enum(item.name)
            t = cross_env.get_type(item.name)
            if t is not None:
                program_type_table[(mid, item.name)] = t
            break
        if isinstance(item, ExceptionDef) and item.name == name:
            builder.build_exception(name)
            typ = cross_env.get_type(name)
            assert typ is not None, f"Exception type {name!r} not registered"
            program_type_table[(mid, name)] = typ
            break
        if isinstance(item, TypeAlias) and item.name == name:
            if item.type_params:
                builder.validate_alias(item)
                template = cross_env.resolve_type_expr(
                    item.type_expr,
                    span=item.span,
                    type_vars=frozenset(item.type_params),
                )
                program_alias_table[(mid, item.name)] = GenericAliasDef(
                    type_params=item.type_params,
                    template=template,
                )
            else:
                alias_type = cross_env.resolve_type_expr(item.type_expr, span=item.span)
                program_type_table[(mid, item.name)] = alias_type
            break
    else:
        # Unreachable: called only for keys produced by _collect_all_type_keys,
        # which iterates the same program.body.items.
        raise AssertionError(f"type '{name}' not found in module '{mid}'")  # pragma: no cover

    _sync_program_env_extensions(
        mid,
        cross_env,
        program_generic_table,
        program_ctor_sig_table,
        program_ctor_field_kinds_table,
    )


def _collect_all_type_keys(
    resolved: ResolvedProgram,
) -> set[tuple[ModuleId, str]]:
    """Collect the set of all user-declared type keys across all modules.

    Returns ``{(ModuleId, name)}`` for every ``RecordDef``, ``EnumDef``, and
    ``TypeAlias`` in every module.  This includes type aliases whose resolved type
    is a primitive (e.g. ``type Number = int``).  Builtin-shadowing types are
    never present here because ``_collect_shells_only`` rejects them earlier.

    This set is the fixed order in which Step C below resolves every
    declaration's body (sorted by :func:`decl_key_sort_key`). It is LARGER
    than ``program_type_table`` during the shell-collection step because
    aliases are not yet resolved to shells there — the program table is only
    populated with record/enum shells and is updated with alias resolutions
    as each body is resolved.
    """
    all_keys: set[tuple[ModuleId, str]] = set()
    for mid, rmod in resolved.modules.items():
        program = rmod.resolved.program
        assert isinstance(program, Program)
        for item in program.body.items:
            if isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
                # Builtin/prelude shadowing is rejected in _collect_shells_only
                # (Step A of _build_program_type_table), which is called before this
                # function.  Only non-builtin types reach this point.
                all_keys.add((mid, item.name))
    return all_keys


def _find_type_decl_span(resolved: ResolvedProgram, key: tuple[ModuleId, str]) -> SourceSpan | None:
    """Return the declaration span for *key*, or ``None`` if it cannot be found.

    Used to attach a real source span to the whole-program inhabitation error
    (see :func:`_build_program_type_table`): the resolved module ASTs are
    already in hand, so the span is a plain lookup rather than anything
    carried through the type table itself (a ``TypeDef`` has no span — it is
    a pure semantic description, shared with standalone-module
    building).
    """
    mid, name = key
    rmod = resolved.modules.get(mid)
    if rmod is None:
        return None
    program = rmod.resolved.program
    assert isinstance(program, Program)
    for item in program.body.items:
        if isinstance(item, (RecordDef, EnumDef, ExceptionDef)) and item.name == name:
            return item.span
    return None


def _raise_first_uninhabited(
    uninhabited: frozenset[tuple[ModuleId, str]],
    type_table: TypeTable,
    resolved: ResolvedProgram,
) -> None:
    """Raise ``AglTypeError`` for the first uninhabited key, sorted deterministically."""
    mid, name = sorted(uninhabited, key=decl_key_sort_key)[0]
    typedef = type_table.get(mid, name)
    assert typedef is not None
    span = _find_type_decl_span(resolved, (mid, name))
    raise AglTypeError(uninhabitable_message(typedef.kind, name), span=span)


def _build_program_type_table(
    resolved: ResolvedProgram,
    *,
    type_table: TypeTable | None = None,
    entry_seed_env: TypeEnvironment | None = None,
) -> tuple[
    dict[tuple[ModuleId, str], Type],
    dict[tuple[ModuleId, str], GenericTypeDef],
    dict[tuple[ModuleId, str], GenericAliasDef],
    dict[tuple[ModuleId, str, str | None], ConstructorSignature],
    dict[tuple[ModuleId, str, str | None], tuple[tuple[str, ParamKind], ...]],
]:
    """Phase 1: collect and resolve all public type declarations across all modules.

    Returns a ``program_type_table`` mapping ``(ModuleId, name)`` → the
    ``RecordType``/``EnumType``/``ExceptionType`` handle or resolved-alias
    ``Type``.

    The pre-pass is genuinely whole-program two-phase:

    Step A: Register every declared name's handle for ALL modules.  Records,
            enums, and exceptions get their handle entered into
            ``program_type_table`` directly (a handle carries no field/variant
            data, so there is nothing left to fill in later — forward
            references within or across modules are valid immediately).
            Type aliases are registered as lazy program alias keys (their target
            type is not known until the alias body is resolved, so they have
            no handle entry yet).

    Step B: Resolve every type body in a fixed deterministic order (sorted by
            ``(ModuleId.segments, name)``), with no dependency-ordering
            constraint — every nominal field/variant/element type reference is
            a handle, valid regardless of whether the referenced declaration's
            own body has been resolved yet. Transparent aliases are resolved
            lazily when referenced so alias dependencies do not impose a body
            ordering, while recursive aliases are still rejected.

    Step C: Once every body is resolved, run the inhabitation fixpoint
            (:func:`~agm.agl.semantics.analyses.compute_uninhabited`) over the
            whole shared table and reject the first uninhabited declaration
            (sorted by :func:`decl_key_sort_key`), at its declaration span.

    Cross-module type cycles are allowed (a cycle may span any modules, the
    same as same-module mutual recursion) as long as the declarations
    involved are inhabited — e.g. modA imports modB for a Color enum used in
    modA.Foo fields, and modB imports modA for modA.Foo used in modB.Bar
    fields via a ``list``/``dict`` field or an enum base-case variant.
    """
    # Shared TypeTable: one instance dual-written by every cross-module env
    # below, so declarations from all modules land in the same table.  Step A's
    # per-module envs are transient header-only scaffolding and never
    # dual-write, so they keep their own private (default) tables.
    shared_type_table = type_table if type_table is not None else create_seeded_type_table()
    if entry_seed_env is not None:
        shared_type_table.merge_from(entry_seed_env.type_table)

    # Step A: register every declared name's handle for all modules.
    # For records/enums: register the handle in both the per-module env AND
    # program_type_table.  For aliases: register in the per-module env only
    # (their entry is added to program_type_table in Step C, once resolved).
    per_module_envs: dict[ModuleId, TypeEnvironment] = {}

    for mid, rmod in resolved.modules.items():
        env = TypeEnvironment(module_id=mid)
        if mid.is_entry and entry_seed_env is not None:
            env.seed_from(entry_seed_env)
        # The builder is transient: it only collects headers into ``env``
        # (which bootstraps ``program_type_table`` below).  Body resolution
        # uses the cross-module builders built later, not this one.
        _collect_shells_only(_TypeBuilder(env, module_id=mid), rmod.resolved.program)
        per_module_envs[mid] = env

    # Collect record/enum handles into the shared program type table.
    # Aliases are NOT added here — their entries will be written in Step C
    # after their target type is resolved.
    program_type_table: dict[tuple[ModuleId, str], Type] = {}
    for mid, env in per_module_envs.items():
        for name, t in env.non_builtin_type_items():
            program_type_table[(mid, name)] = t

    # Cross-module generic type definitions carry no shape (a GenericTypeDef is
    # just a type-parameter count plus a TypeVarType-stamped template — the
    # same "shell" data a non-generic handle carries), so — like
    # program_type_table above — they are collected here in Step A rather than
    # gated on that module's own body-resolution order in Step C: a qualified
    # generic application (e.g. ``lib::Box[int]``) inside a field of a type
    # declared in a module that sorts before ``lib`` in the fixed body-resolution
    # order must still resolve.  Aliases need resolved targets rather than
    # shells, so program environments resolve them lazily; constructor signatures
    # and constructor field kinds genuinely need a resolved body (field/target
    # types), so those remain filled as each type body is resolved in Step C.
    program_generic_table: dict[tuple[ModuleId, str], GenericTypeDef] = {}
    for mid, env in per_module_envs.items():
        for name, gdef in env.all_generic_types().items():
            program_generic_table[(mid, name)] = gdef
    program_alias_table: dict[tuple[ModuleId, str], GenericAliasDef] = {}
    alias_decls: dict[tuple[ModuleId, str], TypeAlias] = {}
    for mid, rmod in resolved.modules.items():
        program = rmod.resolved.program
        assert isinstance(program, Program)
        for item in program.body.items:
            if isinstance(item, TypeAlias):
                alias_decls[(mid, item.name)] = item
    program_alias_keys = frozenset(alias_decls)
    program_ctor_sig_table: dict[tuple[ModuleId, str, str | None], ConstructorSignature] = {}
    program_ctor_field_kinds_table: dict[
        tuple[ModuleId, str, str | None], tuple[tuple[str, ParamKind], ...]
    ] = {}

    # Build per-module cross-module-aware environments and builders for
    # body resolution.  Each env knows the full program_type_table and its own
    # module's ImportEnv so qualified and open-imported type refs resolve.
    cross_envs: dict[ModuleId, TypeEnvironment] = {}
    cross_builders: dict[ModuleId, _TypeBuilder] = {}
    resolving_aliases: set[tuple[ModuleId, str]] = set()

    def _resolve_program_alias(
        alias_mid: ModuleId, alias_name: str, span: SourceSpan | None
    ) -> Type | None:
        key = (alias_mid, alias_name)
        item = alias_decls[key]
        if key in resolving_aliases:
            raise AglTypeError(
                f"Type alias '{alias_name}' is part of a cycle.",
                span=span,
            )
        resolving_aliases.add(key)
        try:
            env = cross_envs[alias_mid]
            if item.type_params:
                template = env.resolve_type_expr(
                    item.type_expr,
                    span=item.span,
                    type_vars=frozenset(item.type_params),
                )
                program_alias_table[key] = GenericAliasDef(
                    type_params=item.type_params,
                    template=template,
                )
                return None
            resolved = env.resolve_type_expr(item.type_expr, span=item.span)
            program_type_table[key] = resolved
            return resolved
        finally:
            resolving_aliases.remove(key)

    for mid, rmod in resolved.modules.items():
        import_env = rmod.import_env
        cross_env = TypeEnvironment(
            program_type_table=program_type_table,
            program_generic_table=program_generic_table,
            program_alias_table=program_alias_table,
            program_alias_keys=program_alias_keys,
            program_alias_resolver=_resolve_program_alias,
            program_ctor_sig_table=program_ctor_sig_table,
            program_ctor_field_kinds_table=program_ctor_field_kinds_table,
            import_env=import_env,
            private_info=resolved.private_info,
            module_id=mid,
            type_table=shared_type_table,
        )
        if mid.is_entry and entry_seed_env is not None:
            cross_env.seed_from(entry_seed_env)
        # Seed with own type shells so bare-name local refs resolve.
        for name, t in per_module_envs[mid].non_builtin_type_items():
            cross_env.register_type(name, t)
        cross_envs[mid] = cross_env
        # Build a _TypeBuilder that uses the cross-module env and has the
        # headers and alias targets registered (for build_record/build_enum/
        # build_exception to work).
        builder = _TypeBuilder(cross_env, module_id=mid)
        _collect_shells_only(builder, rmod.resolved.program)
        cross_builders[mid] = builder

    # Use the COMPLETE set of declared type keys (including aliases), NOT just
    # the record/enum handles in program_type_table, as the fixed resolution
    # order for Step B below.
    all_type_keys = _collect_all_type_keys(resolved)

    def _resolve_one(key: tuple[ModuleId, str]) -> None:
        mid, name = key
        _resolve_body_for_one(
            mid=mid,
            name=name,
            per_module_builders=cross_builders,
            program_type_table=program_type_table,
            program_generic_table=program_generic_table,
            program_alias_table=program_alias_table,
            program_ctor_sig_table=program_ctor_sig_table,
            program_ctor_field_kinds_table=program_ctor_field_kinds_table,
            resolved=resolved,
            cross_envs=cross_envs,
        )

    # Step B: resolve every type body in a fixed deterministic order — no
    # dependency-ordering constraint of any kind, since every reference
    # (including an exception's ``extends`` base) is a handle, valid whether
    # or not the referenced declaration's own body has been resolved yet.
    body_order = sorted(all_type_keys, key=decl_key_sort_key)

    for key in body_order:
        _resolve_one(key)

    # Step C: every body is now resolved, so the inhabitation fixpoint can
    # run over the whole shared table (this program's declarations plus the
    # builtin/prelude defs, all trivially inhabited). The per-module builder
    # re-check in Phase 3 (``_check_module``) skips its own inhabitation pass
    # (``check_inhabitation=False``) precisely because this whole-program check
    # has already run.
    uninhabited = compute_uninhabited(shared_type_table)
    if uninhabited:
        _raise_first_uninhabited(uninhabited, shared_type_table, resolved)

    return (
        program_type_table,
        program_generic_table,
        program_alias_table,
        program_ctor_sig_table,
        program_ctor_field_kinds_table,
    )


# ---------------------------------------------------------------------------
# Phase 2: whole-program function-signature pre-pass
# ---------------------------------------------------------------------------


def _build_program_func_sig_table(
    resolved: ResolvedProgram,
    program_type_table: dict[tuple[ModuleId, str], Type],
    program_generic_table: dict[tuple[ModuleId, str], GenericTypeDef],
    program_alias_table: dict[tuple[ModuleId, str], GenericAliasDef],
    entry_seed_env: TypeEnvironment | None = None,
) -> dict[int, FunctionSignatureRecord]:
    """Phase 2: compute function signatures for ALL top-level FuncDefs across all modules.

    Returns a table mapping each ``FuncDef.node_id`` to immutable named
    metadata: the declared name, full signature, erased value type, declaration
    status, and reserved candidate evidence. This is internal coordination
    state; semantic function types remain unchanged.

    This pre-pass builds per-module ``TypeEnvironment``s seeded with the
    program-wide type table and the module's ``ImportEnv`` so that parameter/return
    type annotations that reference cross-module types (e.g. ``lib::Color``) resolve
    correctly.  No function body is checked — only the type-expression annotations
    in each ``FuncDef``'s parameter and return-type declarations are resolved.

    The result is used in :func:`check_program` (Phase 3) to seed EVERY module's
    ``TypeEnvironment`` with ALL reachable function binding types BEFORE any body is
    checked.  Because ``node_id`` is globally unique, seeding the whole table into
    every module's env is safe and collision-free.  This makes cross-file mutual
    recursion work: both ``A::f`` (calling ``B::g``) and ``B::g`` (calling ``A::f``)
    have the other's binding type available regardless of the per-module checking
    order.

    Reuses the shared function-header resolver: a temporary per-module
    ``TypeEnvironment`` (seeded with own types + program table + import env) is
    constructed for each module, then each ``FuncDef`` in that module has its
    signature resolved through the normal ``TypeEnvironment.resolve_type_expr``
    path — the exact same path used in the real per-module check.
    """
    from agm.agl.typecheck.checker import _BUILTIN_FUNC_NAMES, _BUILTIN_TYPE_NAMES

    result: dict[int, FunctionSignatureRecord] = {}

    for mid, rmod in resolved.modules.items():
        program = rmod.resolved.program
        assert isinstance(program, Program)

        import_env = rmod.import_env
        # Build a cross-module-aware env for this module, seeded with its own
        # types so bare-name local type refs in param annotations resolve.
        env = TypeEnvironment(
            program_type_table=program_type_table,
            program_generic_table=program_generic_table,
            program_alias_table=program_alias_table,
            import_env=import_env,
            private_info=resolved.private_info,
            module_id=mid,
        )
        if mid.is_entry and entry_seed_env is not None:
            env.seed_from(entry_seed_env)
        for (t_mid, t_name), t in program_type_table.items():
            if t_mid == mid:
                env.register_type(t_name, t)
        # Also seed the module's own generic types so bare-name local generic
        # refs in param/return annotations (e.g. `o: Option[T]`) resolve here —
        # mirroring the register_type seeding above for non-generic types.
        for (g_mid, g_name), gdef in program_generic_table.items():
            if g_mid == mid:
                env.register_generic_type(g_name, gdef)

        for item in program.body.items:
            if not isinstance(item, FuncDef):
                continue
            # Defer invalid user shadowing declarations to the ordinary checker,
            # which reports them at the declaration. A ``builtin def`` deliberately
            # uses a builtin call name, so it must still receive program metadata.
            if item.name in _BUILTIN_TYPE_NAMES or (
                item.name in _BUILTIN_FUNC_NAMES and not item.is_builtin
            ):
                continue
            if item.return_type is None:
                continue

            signature, function_type = resolve_function_header(
                env, item, result_type=item.return_type
            )
            result[item.node_id] = FunctionSignatureRecord(
                declaration_node_id=item.node_id,
                name=item.name,
                signature=signature,
                function_type=function_type,
                is_builtin=item.is_builtin,
                is_extern=item.is_extern,
                return_source=FunctionReturnSource.DECLARED,
            )

    return result


def _build_program_builtin_var_table(
    resolved: ResolvedProgram,
) -> dict[int, Type]:
    """Compute binding types for every ``builtin var`` across all modules.

    A ``builtin var`` names a fixed engine key whose type is canonical (from the
    engine-key registry), so no type-expression resolution or per-module env is
    needed.  The table is keyed by the declaration node id (globally unique), so
    seeding it into every module's env makes each engine setting readable and
    assignable from any module that imports its owner (e.g. ``std/config``).
    Unknown-key declarations are omitted; the owning module's own check rejects
    them with a clear error.
    """
    from agm.agl.semantics.engine_keys import get_engine_key_type

    result: dict[int, Type] = {}
    for _mid, loaded in resolved.modules.items():
        for item in loaded.resolved.program.body.items:
            if isinstance(item, BuiltinVarDecl):
                key_type = get_engine_key_type(item.name)
                if key_type is not None:
                    result[item.node_id] = key_type
    return result


# ---------------------------------------------------------------------------
# Per-module checking helper
# ---------------------------------------------------------------------------


def _module_function_signatures(
    program: Program, env: TypeEnvironment
) -> dict[str, FunctionSignature]:
    """Return checked schemes declared by *program*, keyed by their local names.

    A program environment carries imported schemes during checking, keyed by
    declaration node id.  They support resolved occurrences but are not
    declarations of the module being published.
    """
    signatures: dict[str, FunctionSignature] = {}
    for item in program.body.items:
        if not isinstance(item, FuncDef):
            continue
        signature = env.get_function_signature_by_node_id(item.node_id)
        assert signature is not None, f"No checked signature for '{item.name}'"
        signatures[item.name] = signature
    return signatures


def _check_module(
    mid: ModuleId,
    resolved: ModuleResolution,
    source_text: str,
    capabilities: HostCapabilities,
    program_type_table: dict[tuple[ModuleId, str], Type],
    import_env_map: Mapping[ModuleId, object],
    private_info: Mapping[tuple[ModuleId, str], bool],
    program_func_sig_table: dict[int, FunctionSignatureRecord],
    program_builtin_var_table: dict[int, Type],
    program_generic_table: dict[tuple[ModuleId, str], GenericTypeDef],
    program_alias_table: dict[tuple[ModuleId, str], GenericAliasDef],
    program_ctor_sig_table: dict[tuple[ModuleId, str, str | None], ConstructorSignature],
    program_ctor_field_kinds_table: dict[
        tuple[ModuleId, str, str | None], tuple[tuple[str, ParamKind], ...]
    ],
    type_table: TypeTable,
    entry_seed_env: TypeEnvironment | None = None,
) -> CheckedModule:
    """Type-check one module with a module-aware ``TypeEnvironment``.

    The env is seeded with:
    - The module's own types (from ``program_type_table``).
    - The program table + import env for cross-module lookups.
    - Binding types (function signatures) from the whole-program function-signature
      pre-pass (``program_func_sig_table``), seeded BEFORE any body is checked so
      that cross-module function calls — including those in import cycles — can
      look up callee types regardless of per-module checking order.  The pre-pass
      computed named signature records for ALL top-level ``FuncDef``s across ALL modules;
      ``node_id``s are globally unique, so seeding the whole table into
      every module's env is safe and collision-free.
    - ``type_table``: the single ``TypeTable`` instance shared by every module
      in this program (the same one built and dual-written in the type pre-pass),
      so this module's own re-check dual-writes into the same table.
    - ``entry_seed_env``: when given and ``mid`` is the entry module, the session
      type env is seeded first so that prior REPL bindings are available.
    """
    import_env = import_env_map[mid]
    assert isinstance(import_env, ImportEnv)

    env = TypeEnvironment(
        program_type_table=program_type_table,
        program_generic_table=program_generic_table,
        program_alias_table=program_alias_table,
        program_ctor_sig_table=program_ctor_sig_table,
        program_ctor_field_kinds_table=program_ctor_field_kinds_table,
        import_env=import_env,
        private_info=private_info,
        module_id=mid,
        type_table=type_table,
    )

    # Seed from the REPL session type env first (for the entry module in REPL
    # program context).  Program tables override on collision, so the entry's own types
    # and function signatures always shadow any session binding with the same name.
    if mid.is_entry and entry_seed_env is not None:
        env.seed_from(entry_seed_env)

    # Seed env with the module's own fully-resolved types so they're
    # accessible by bare name (no qualifier needed within the module).
    for (t_mid, t_name), t in program_type_table.items():
        if t_mid == mid:
            env.register_type(t_name, t)

    # Seed binding types from the whole-program function-signature pre-pass.
    # This makes EVERY module's function signatures available in EVERY module's
    # env before any body is checked, enabling cross-file mutual recursion (the current rules).
    #
    # Three tables are seeded:
    # - _binding_types (node_id-keyed, globally unique): used by _check_varref to
    #   look up the callee type when the callee VarRef resolves to a cross-module
    #   FuncDef's decl_node_id.  Seeding the entire pre-pass table is safe because
    #   node_ids are globally unique.
    # - _function_signatures_by_node_id (node_id-keyed, globally unique): used by
    #   _check_declared_name_call to look up the CORRECT signature for any callee
    #   by its globally-unique decl_node_id.  Unlike the name-keyed table below,
    #   this table never suffers from same-name collisions across modules.
    # - _function_signatures (name-keyed): used as a fallback by
    #   _check_declared_name_call when no node-id lookup is available (single-
    #   program path).  Same-named functions from different modules may collide
    #   here; the current module's own signatures always win because
    #   builder.collect() → _preregister_funcdef re-registers them AFTER this
    #   seeding step, overwriting any cross-module collision for bare-name calls.
    for node_id, record in program_func_sig_table.items():
        env.set_binding_type(node_id, record.function_type)
        env.register_function_signature_by_node_id(node_id, record.signature)
        env.register_function_signature(record.name, record.signature)
        if record.is_extern:
            env.register_extern_node_id(node_id)

    # Seed builtin-var binding types (engine settings) from the whole-program
    # pre-pass so a ``std/config::key`` read/assign in any module resolves its
    # type.  Keyed by globally-unique decl node id, so seeding the whole table
    # into every module's env is safe and collision-free.
    for var_node_id, var_type in program_builtin_var_table.items():
        env.set_binding_type(var_node_id, var_type)

    # Build the checked output through the same close/finalize boundary as a
    # single module. The program pre-pass already checked inhabitation over the
    # shared table, so this per-module pass only skips that redundant analysis.
    cp = _check_prepared_module(
        resolved,
        capabilities,
        env=env,
        module_id=mid,
        check_inhabitation=False,
    )
    return replace(
        cp,
        module_id=mid,
        import_env=import_env,
        source_text=source_text,
        function_signatures=_module_function_signatures(resolved.program, cp.type_env),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_program(
    resolved: ResolvedProgram,
    capabilities: HostCapabilities,
    entry_seed_env: TypeEnvironment | None = None,
) -> CheckedProgram:
    """Run the full type-checking pass over a :class:`ResolvedProgram`.

    Parameters
    ----------
    resolved:
        Output of :func:`~agm.agl.scope.program.resolve_program`.
    capabilities:
        Immutable host capability catalog (agents, codecs, renderers).
    entry_seed_env:
        When given, the entry module's ``TypeEnvironment`` is seeded from this
        environment before the program type table and function signatures are
        installed.  Used by the REPL program context to make prior session
        bindings available in program entries.

    Returns
    -------
    CheckedProgram
        Per-module type side tables plus the shared program type table.

    Raises
    ------
    AglTypeError
        On the first static type violation in any module (first-error abort).
    """
    # One TypeTable shared by every module in this program: the type pre-pass
    # dual-writes into it below, and Phase 3 re-checks each module's own
    # declarations against the SAME instance, so the whole program's declarations
    # land in one table regardless of per-module checking order.
    shared_type_table = create_seeded_type_table()

    # Phase 1: build the program-wide type table with all module types stamped
    # with their owning module_id.  Also collects cross-module generic type defs,
    # parameterized aliases, constructor signatures, and constructor field kinds
    # from the per-module envs built during body resolution.
    (
        program_type_table,
        program_generic_table,
        program_alias_table,
        program_ctor_sig_table,
        program_ctor_field_kinds_table,
    ) = _build_program_type_table(
        resolved,
        type_table=shared_type_table,
        entry_seed_env=entry_seed_env,
    )

    # Phase 2: build the program-wide function-signature table.
    # Resolves parameter/return TypeExprs for every top-level FuncDef in every
    # module using the program_type_table (so cross-module type refs in annotations
    # resolve), WITHOUT checking any function body.  Keyed by FuncDef.node_id
    # .
    program_func_sig_table = _build_program_func_sig_table(
        resolved,
        program_type_table,
        program_generic_table,
        program_alias_table,
        entry_seed_env=entry_seed_env,
    )

    # Phase 2b: canonical binding types for every ``builtin var`` (engine
    # settings), keyed by decl node id, seeded into every module's env below.
    program_builtin_var_table = _build_program_builtin_var_table(resolved)

    # Collect import envs for per-module checking.
    import_env_map: dict[ModuleId, object] = {
        mid: rmod.import_env for mid, rmod in resolved.modules.items()
    }

    # Phase 3: type-check each module's bodies.
    # Non-entry modules are checked first, then entry (ordering kept for
    # determinism and for any future ordering-sensitive checks), but function
    # signature availability no longer depends on this order — the whole-program
    # pre-pass in Phase 2 seeds all binding types before any body is checked,
    # so cross-file mutual recursion is handled correctly.
    checked_modules: dict[ModuleId, CheckedModule] = {}
    all_warnings: list[Diagnostic] = []

    # Check non-entry modules first, then entry.
    ordered_mids = [mid for mid in resolved.modules if not mid.is_entry] + [resolved.entry_id]

    for mid in ordered_mids:
        rmod = resolved.modules[mid]
        cm = _check_module(
            mid,
            rmod.resolved,
            rmod.source_text,
            capabilities,
            program_type_table,
            import_env_map,
            resolved.private_info,
            program_func_sig_table,
            program_builtin_var_table,
            program_generic_table,
            program_alias_table,
            program_ctor_sig_table,
            program_ctor_field_kinds_table,
            shared_type_table,
            entry_seed_env=entry_seed_env if mid.is_entry else None,
        )
        checked_modules[mid] = cm
        all_warnings.extend(cm.warnings)

    checked = CheckedProgram(
        modules=checked_modules,
        entry_id=resolved.entry_id,
        program_type_table=program_type_table,
        warnings=tuple(all_warnings),
        capabilities=capabilities,
    )
    assert_checked_program_closed(checked)
    return checked
