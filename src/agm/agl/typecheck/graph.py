"""Graph-aware type-checking pass for the AgL module system.

``check_graph(resolved_graph, capabilities)`` runs the full type-checking pass
over a :class:`~agm.agl.scope.graph.ResolvedModuleGraph`, producing a
:class:`CheckedModuleGraph`.

Algorithm
---------
1. **Graph type pre-pass** — collect ALL public type declarations across every
   module and resolve their bodies (stamping each
   ``RecordType``/``EnumType``/``ExceptionType`` handle with the owning
   ``ModuleId``), building the shared ``graph_type_table``.
   ``RecordType``/``EnumType``/``ExceptionType`` carry no field/variant data
   of their own (their shapes live in the shared ``TypeTable``), so a
   reference to another module's type is a valid handle whether or not that
   type's own body has been resolved yet — body resolution is therefore
   order-free.

   The pre-pass is genuinely whole-graph two-phase:

   a. **Headers** — every declared name's handle (or, for a generic
      declaration, its ``GenericTypeDef``) is registered into the shared
      ``graph_type_table`` first; type aliases are registered as lazy graph
      alias keys because aliases are transparent and have no handle shell.
   b. **Bodies, order-free** — each declaration's body (field/variant type
      expressions) is resolved in a fixed deterministic order (sorted by
      ``(ModuleId.segments, name)``), with no dependency-ordering
      constraint — nominal cycles (same-module, mutual, or spanning any
      number of modules) are legal, since every nominal reference resolves to
      a handle regardless of build order; alias references resolve lazily and
      still reject transparent alias cycles.
   c. **Inhabitation** — once every body is resolved, the whole-graph
      inhabitation fixpoint
      (:func:`~agm.agl.semantics.analyses.compute_uninhabited`) rejects the
      first declaration (across the whole graph) that has no finite value,
      consistent with the single-module ``_TypeBuilder`` check.

2. **Graph function-signature pre-pass** — resolve the parameter and return type
   annotations for EVERY top-level ``FuncDef`` in EVERY module (using the
   ``graph_type_table`` and each module's ``ImportEnv`` for cross-module type
   refs), producing a ``graph_func_sig_table`` mapping each ``FuncDef.node_id``
   to ``(FunctionSignature, FunctionType)``. No function body is checked in this
   phase. The result is used in Phase 3 to
   seed EVERY module's env with ALL function binding types before any body is
   checked, enabling cross-file mutual recursion: a call to a not-yet-checked
   module's function resolves its callee type from the pre-pass table.

3. **Per-module type-check** — for each module, build a module-aware
   :class:`~agm.agl.typecheck.env.TypeEnvironment` seeded with the module's own
   types (from the graph table), ALL function binding types (from the
   function-signature pre-pass), and the graph table + import env for cross-module
   lookups, then run the existing :class:`~agm.agl.typecheck.builder._TypeBuilder`
   and :class:`~agm.agl.typecheck.checker._Checker` logic.

Single-module equivalence
-------------------------
A single-module (entry-only) graph checked via :func:`check_graph` is
equivalent to calling :func:`~agm.agl.typecheck.checker.check` directly on the
entry's :class:`~agm.agl.scope.symbols.ResolvedProgram`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from agm.agl.capabilities import HostCapabilities
from agm.agl.diagnostics import Diagnostic
from agm.agl.modules.ids import ModuleId
from agm.agl.scope.graph import ResolvedModuleGraph
from agm.agl.scope.imports import ImportEnv
from agm.agl.scope.symbols import ResolvedProgram
from agm.agl.semantics.analyses import compute_uninhabited, uninhabitable_message
from agm.agl.semantics.type_table import TypeTable, create_seeded_type_table, decl_key_sort_key
from agm.agl.semantics.types import (
    CastSpec,
    FunctionType,
    Type,
)
from agm.agl.syntax.nodes import (
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
from agm.agl.typecheck.checker import _Checker
from agm.agl.typecheck.env import (
    AglTypeError,
    ArgumentBindings,
    CallSiteRecord,
    ConstructorSignature,
    FunctionSignature,
    GenericAliasDef,
    GenericTypeDef,
    OutputContractSpec,
    ParamSpec,
    PartialCallSpec,
    TypeEnvironment,
)

# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckedModule:
    """Per-module output of the graph type-checking pass.

    ``module_id``
        The :class:`~agm.agl.modules.ids.ModuleId` of this module.
    ``resolved``
        The :class:`~agm.agl.scope.symbols.ResolvedProgram` from graph scope resolution.
    ``node_types``
        Maps ``node_id`` → resolved :class:`~agm.agl.semantics.types.Type`
        for every expression node that was type-checked in this module.
    ``contract_specs``
        Maps ``AgentCall.node_id`` →
        :class:`~agm.agl.typecheck.env.OutputContractSpec` for call sites that
        parse output.
    ``call_sites``
        Tuple of :class:`~agm.agl.typecheck.env.CallSiteRecord` — one per
        agent-call/exec site in this module, in source order.
    ``warnings``
        Non-fatal type-check diagnostics from this module.
    ``function_signatures``
        Maps function name → :class:`~agm.agl.typecheck.env.FunctionSignature`
        for all top-level ``def`` declarations in this module.
    ``cast_specs``
        Maps ``Cast.node_id`` → :class:`~agm.agl.semantics.types.CastSpec`
        for every cast expression in this module.
    ``type_env``
        The module-aware :class:`~agm.agl.typecheck.env.TypeEnvironment`
        built during the pass.
    ``import_env``
        The immutable source import environment used to resolve this module.
        It is retained explicitly so later checked-program consumers can use
        source aliases and exposure rules without reaching into ``type_env``.
    """

    module_id: ModuleId
    resolved: ResolvedProgram
    node_types: dict[int, Type]
    contract_specs: dict[int, OutputContractSpec]
    call_sites: tuple[CallSiteRecord, ...]
    warnings: tuple[Diagnostic, ...]
    function_signatures: dict[str, FunctionSignature]
    cast_specs: dict[int, CastSpec]
    type_env: TypeEnvironment
    import_env: ImportEnv
    source_text: str
    argument_bindings: ArgumentBindings
    partial_calls: dict[int, PartialCallSpec]


@dataclass(frozen=True, slots=True)
class CheckedModuleGraph:
    """Immutable output of :func:`check_graph`.

    ``modules``
        Maps each :class:`~agm.agl.modules.ids.ModuleId` to its
        :class:`CheckedModule`.
    ``entry_id``
        Always :data:`~agm.agl.modules.ids.ENTRY_ID`.
    ``graph_type_table``
        Whole-graph type table mapping ``(ModuleId, name)`` to the fully-built
        :class:`~agm.agl.semantics.types.Type` object stamped with the owning
        ``module_id``.  Built in the graph pre-pass; shared (read-only) across
        all per-module environments.
    ``warnings``
        All non-fatal type-check diagnostics collected across all modules, in
        module-traversal order.
    """

    modules: dict[ModuleId, CheckedModule]
    entry_id: ModuleId
    graph_type_table: dict[tuple[ModuleId, str], Type]
    warnings: tuple[Diagnostic, ...]


# ---------------------------------------------------------------------------
# Graph type pre-pass helpers
# ---------------------------------------------------------------------------


def _collect_shells_only(builder: _TypeBuilder, program: object) -> None:
    """Run only phase 1 (shell registration) of ``_TypeBuilder.collect``.

    Delegates to :meth:`~agm.agl.typecheck.builder._TypeBuilder.collect_shells_only`,
    the public API added to ``_TypeBuilder`` for this purpose.
    """
    assert isinstance(program, Program)
    builder.collect_shells_only(program)


def _sync_graph_env_extensions(
    mid: ModuleId,
    env: TypeEnvironment,
    graph_generic_table: dict[tuple[ModuleId, str], GenericTypeDef],
    graph_ctor_sig_table: dict[tuple[ModuleId, str, str | None], ConstructorSignature],
    graph_ctor_field_kinds_table: dict[
        tuple[ModuleId, str, str | None], tuple[tuple[str, ParamKind], ...]
    ],
) -> None:
    """Copy generic type and constructor metadata built for one module into graph tables."""
    for generic_name, gdef in env.all_generic_types().items():
        graph_generic_table[(mid, generic_name)] = gdef
    for (owner_name, variant), sig in env.all_constructor_sigs():
        graph_ctor_sig_table[(mid, owner_name, variant)] = sig
    for (owner_name, variant), kinds in env.all_constructor_field_kinds():
        graph_ctor_field_kinds_table[(mid, owner_name, variant)] = kinds


def _resolve_body_for_one(
    mid: ModuleId,
    name: str,
    per_module_builders: dict[ModuleId, _TypeBuilder],
    graph_type_table: dict[tuple[ModuleId, str], Type],
    graph_generic_table: dict[tuple[ModuleId, str], GenericTypeDef],
    graph_alias_table: dict[tuple[ModuleId, str], GenericAliasDef],
    graph_ctor_sig_table: dict[tuple[ModuleId, str, str | None], ConstructorSignature],
    graph_ctor_field_kinds_table: dict[
        tuple[ModuleId, str, str | None], tuple[tuple[str, ParamKind], ...]
    ],
    resolved_graph: ResolvedModuleGraph,
    cross_envs: dict[ModuleId, TypeEnvironment],
) -> None:
    """Resolve the body of one type ``(mid, name)`` and update the graph table.

    Called once per key in a fixed deterministic order (no dependency
    ordering): every type reference is a handle, valid regardless of whether
    the referenced type's own body has been resolved yet.
    """
    cross_env = cross_envs[mid]
    builder = per_module_builders[mid]

    program = resolved_graph.modules[mid].resolved.program
    assert isinstance(program, Program)

    for item in program.body.items:
        if isinstance(item, RecordDef) and item.name == name:
            builder.build_record(item.name)
            t = cross_env.get_type(item.name)
            if t is not None:
                # Non-generic record: update the graph table with the fully-built type.
                graph_type_table[(mid, item.name)] = t
            # Generic record: body registered in _generic_types (no _types entry);
            # graph_type_table retains the handle from Step A.  Cross-module generic
            # constructor calls use _graph_generic_table / _graph_ctor_sig_table instead.
            break
        if isinstance(item, EnumDef) and item.name == name:
            builder.build_enum(item.name)
            t = cross_env.get_type(item.name)
            if t is not None:
                graph_type_table[(mid, item.name)] = t
            break
        if isinstance(item, ExceptionDef) and item.name == name:
            builder.build_exception(name)
            typ = cross_env.get_type(name)
            assert typ is not None, f"Exception type {name!r} not registered"
            graph_type_table[(mid, name)] = typ
            break
        if isinstance(item, TypeAlias) and item.name == name:
            if item.type_params:
                builder.validate_alias(item)
                template = cross_env.resolve_type_expr(
                    item.type_expr,
                    span=item.span,
                    type_vars=frozenset(item.type_params),
                )
                graph_alias_table[(mid, item.name)] = GenericAliasDef(
                    type_params=item.type_params,
                    template=template,
                )
            else:
                resolved = cross_env.resolve_type_expr(item.type_expr, span=item.span)
                graph_type_table[(mid, item.name)] = resolved
            break
    else:
        # Unreachable: called only for keys produced by _collect_all_type_keys,
        # which iterates the same program.body.items.
        raise AssertionError(f"type '{name}' not found in module '{mid}'")  # pragma: no cover

    _sync_graph_env_extensions(
        mid,
        cross_env,
        graph_generic_table,
        graph_ctor_sig_table,
        graph_ctor_field_kinds_table,
    )


def _collect_all_type_keys(
    resolved_graph: ResolvedModuleGraph,
) -> set[tuple[ModuleId, str]]:
    """Collect the set of all user-declared type keys across all modules.

    Returns ``{(ModuleId, name)}`` for every ``RecordDef``, ``EnumDef``, and
    ``TypeAlias`` in every module.  This includes type aliases whose resolved type
    is a primitive (e.g. ``type Number = int``).  Builtin-shadowing types are
    never present here because ``_collect_shells_only`` rejects them earlier.

    This set is the fixed order in which Step C below resolves every
    declaration's body (sorted by :func:`decl_key_sort_key`). It is LARGER
    than ``graph_type_table`` during the shell-collection step because
    aliases are not yet resolved to shells there — the graph table is only
    populated with record/enum shells and is updated with alias resolutions
    as each body is resolved.
    """
    all_keys: set[tuple[ModuleId, str]] = set()
    for mid, rmod in resolved_graph.modules.items():
        program = rmod.resolved.program
        assert isinstance(program, Program)
        for item in program.body.items:
            if isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
                # Builtin/prelude shadowing is rejected in _collect_shells_only
                # (Step A of _build_graph_type_table), which is called before this
                # function.  Only non-builtin types reach this point.
                all_keys.add((mid, item.name))
    return all_keys


def _find_type_decl_span(
    resolved_graph: ResolvedModuleGraph, key: tuple[ModuleId, str]
) -> SourceSpan | None:
    """Return the declaration span for *key*, or ``None`` if it cannot be found.

    Used to attach a real source span to the whole-graph inhabitation error
    (see :func:`_build_graph_type_table`): the resolved module ASTs are
    already in hand, so the span is a plain lookup rather than anything
    carried through the type table itself (a ``TypeDef`` has no span — it is
    a pure semantic description, shared with non-graph single-module
    building).
    """
    mid, name = key
    rmod = resolved_graph.modules.get(mid)
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
    resolved_graph: ResolvedModuleGraph,
) -> None:
    """Raise ``AglTypeError`` for the first uninhabited key, sorted deterministically."""
    mid, name = sorted(uninhabited, key=decl_key_sort_key)[0]
    typedef = type_table.get(mid, name)
    assert typedef is not None
    span = _find_type_decl_span(resolved_graph, (mid, name))
    raise AglTypeError(uninhabitable_message(typedef.kind, name), span=span)


def _build_graph_type_table(
    resolved_graph: ResolvedModuleGraph,
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

    Returns a ``graph_type_table`` mapping ``(ModuleId, name)`` → the
    ``RecordType``/``EnumType``/``ExceptionType`` handle or resolved-alias
    ``Type``.

    The pre-pass is genuinely whole-graph two-phase:

    Step A: Register every declared name's handle for ALL modules.  Records,
            enums, and exceptions get their handle entered into
            ``graph_type_table`` directly (a handle carries no field/variant
            data, so there is nothing left to fill in later — forward
            references within or across modules are valid immediately).
            Type aliases are registered as lazy graph alias keys (their target
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
    # graph_type_table.  For aliases: register in the per-module env only
    # (their entry is added to graph_type_table in Step C, once resolved).
    per_module_envs: dict[ModuleId, TypeEnvironment] = {}

    for mid, rmod in resolved_graph.modules.items():
        env = TypeEnvironment(module_id=mid)
        if mid.is_entry and entry_seed_env is not None:
            env.seed_from(entry_seed_env)
        # The builder is transient: it only collects headers into ``env``
        # (which bootstraps ``graph_type_table`` below).  Body resolution
        # uses the cross-module builders built later, not this one.
        _collect_shells_only(_TypeBuilder(env, module_id=mid), rmod.resolved.program)
        per_module_envs[mid] = env

    # Collect record/enum handles into the shared graph type table.
    # Aliases are NOT added here — their entries will be written in Step C
    # after their target type is resolved.
    graph_type_table: dict[tuple[ModuleId, str], Type] = {}
    for mid, env in per_module_envs.items():
        for name, t in env.non_builtin_type_items():
            graph_type_table[(mid, name)] = t

    # Cross-module generic type definitions carry no shape (a GenericTypeDef is
    # just a type-parameter count plus a TypeVarType-stamped template — the
    # same "shell" data a non-generic handle carries), so — like
    # graph_type_table above — they are collected here in Step A rather than
    # gated on that module's own body-resolution order in Step C: a qualified
    # generic application (e.g. ``lib::Box[int]``) inside a field of a type
    # declared in a module that sorts before ``lib`` in the fixed body-resolution
    # order must still resolve.  Aliases need resolved targets rather than
    # shells, so graph environments resolve them lazily; constructor signatures
    # and constructor field kinds genuinely need a resolved body (field/target
    # types), so those remain filled as each type body is resolved in Step C.
    graph_generic_table: dict[tuple[ModuleId, str], GenericTypeDef] = {}
    for mid, env in per_module_envs.items():
        for name, gdef in env.all_generic_types().items():
            graph_generic_table[(mid, name)] = gdef
    graph_alias_table: dict[tuple[ModuleId, str], GenericAliasDef] = {}
    alias_decls: dict[tuple[ModuleId, str], TypeAlias] = {}
    for mid, rmod in resolved_graph.modules.items():
        program = rmod.resolved.program
        assert isinstance(program, Program)
        for item in program.body.items:
            if isinstance(item, TypeAlias):
                alias_decls[(mid, item.name)] = item
    graph_alias_keys = frozenset(alias_decls)
    graph_ctor_sig_table: dict[tuple[ModuleId, str, str | None], ConstructorSignature] = {}
    graph_ctor_field_kinds_table: dict[
        tuple[ModuleId, str, str | None], tuple[tuple[str, ParamKind], ...]
    ] = {}

    # Build per-module cross-module-aware environments and builders for
    # body resolution.  Each env knows the full graph_type_table and its own
    # module's ImportEnv so qualified and open-imported type refs resolve.
    cross_envs: dict[ModuleId, TypeEnvironment] = {}
    cross_builders: dict[ModuleId, _TypeBuilder] = {}
    resolving_aliases: set[tuple[ModuleId, str]] = set()

    def _resolve_graph_alias(
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
                graph_alias_table[key] = GenericAliasDef(
                    type_params=item.type_params,
                    template=template,
                )
                return None
            resolved = env.resolve_type_expr(item.type_expr, span=item.span)
            graph_type_table[key] = resolved
            return resolved
        finally:
            resolving_aliases.remove(key)

    for mid, rmod in resolved_graph.modules.items():
        import_env = rmod.import_env
        cross_env = TypeEnvironment(
            graph_type_table=graph_type_table,
            graph_generic_table=graph_generic_table,
            graph_alias_table=graph_alias_table,
            graph_alias_keys=graph_alias_keys,
            graph_alias_resolver=_resolve_graph_alias,
            graph_ctor_sig_table=graph_ctor_sig_table,
            graph_ctor_field_kinds_table=graph_ctor_field_kinds_table,
            import_env=import_env,
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
    # the record/enum handles in graph_type_table, as the fixed resolution
    # order for Step B below.
    all_type_keys = _collect_all_type_keys(resolved_graph)

    def _resolve_one(key: tuple[ModuleId, str]) -> None:
        mid, name = key
        _resolve_body_for_one(
            mid=mid,
            name=name,
            per_module_builders=cross_builders,
            graph_type_table=graph_type_table,
            graph_generic_table=graph_generic_table,
            graph_alias_table=graph_alias_table,
            graph_ctor_sig_table=graph_ctor_sig_table,
            graph_ctor_field_kinds_table=graph_ctor_field_kinds_table,
            resolved_graph=resolved_graph,
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
    # run over the whole shared table (this graph's declarations plus the
    # builtin/prelude defs, all trivially inhabited). The per-module builder
    # re-check in Phase 3 (``_check_module``) skips its own inhabitation pass
    # (``check_inhabitation=False``) precisely because this whole-graph check
    # has already run.
    uninhabited = compute_uninhabited(shared_type_table)
    if uninhabited:
        _raise_first_uninhabited(uninhabited, shared_type_table, resolved_graph)

    return (
        graph_type_table,
        graph_generic_table,
        graph_alias_table,
        graph_ctor_sig_table,
        graph_ctor_field_kinds_table,
    )


# ---------------------------------------------------------------------------
# Phase 2: whole-graph function-signature pre-pass
# ---------------------------------------------------------------------------


def _build_graph_func_sig_table(
    resolved_graph: ResolvedModuleGraph,
    graph_type_table: dict[tuple[ModuleId, str], Type],
    graph_generic_table: dict[tuple[ModuleId, str], GenericTypeDef],
    graph_alias_table: dict[tuple[ModuleId, str], GenericAliasDef],
    entry_seed_env: TypeEnvironment | None = None,
) -> dict[int, tuple[str, FunctionSignature, FunctionType, bool]]:
    """Phase 2: compute function signatures for ALL top-level FuncDefs across all modules.

    Returns a table mapping each ``FuncDef.node_id``
    to ``(name, FunctionSignature, FunctionType, is_extern)`` — the declared
    function name, the full declared signature (names, types, has-default
    flags), the erased value type, and whether the declaration is an
    ``extern def`` (so calls to an imported extern are recorded the same way
    as calls to an extern declared in the calling module).

    This pre-pass builds per-module ``TypeEnvironment``s seeded with the
    graph-wide type table and the module's ``ImportEnv`` so that parameter/return
    type annotations that reference cross-module types (e.g. ``lib::Color``) resolve
    correctly.  No function body is checked — only the type-expression annotations
    in each ``FuncDef``'s parameter and return-type declarations are resolved.

    The result is used in :func:`check_graph` (Phase 3) to seed EVERY module's
    ``TypeEnvironment`` with ALL reachable function binding types BEFORE any body is
    checked.  Because ``node_id`` is globally unique, seeding the whole table into
    every module's env is safe and collision-free.  This makes cross-file mutual
    recursion work: both ``A::f`` (calling ``B::g``) and ``B::g`` (calling ``A::f``)
    have the other's binding type available regardless of the per-module checking
    order.

    Reuses :meth:`~agm.agl.typecheck.checker._Checker._preregister_funcdef` logic
    without duplicating type-expression resolution: a temporary per-module
    ``TypeEnvironment`` (seeded with own types + graph table + import env) is
    constructed for each module, then each ``FuncDef`` in that module has its
    signature resolved through the normal ``TypeEnvironment.resolve_type_expr``
    path — the exact same path used in the real per-module check.
    """
    from agm.agl.typecheck.checker import _BUILTIN_FUNC_NAMES, _BUILTIN_TYPE_NAMES

    result: dict[int, tuple[str, FunctionSignature, FunctionType, bool]] = {}

    for mid, rmod in resolved_graph.modules.items():
        program = rmod.resolved.program
        assert isinstance(program, Program)

        import_env = rmod.import_env
        # Build a cross-module-aware env for this module, seeded with its own
        # types so bare-name local type refs in param annotations resolve.
        env = TypeEnvironment(
            graph_type_table=graph_type_table,
            graph_generic_table=graph_generic_table,
            graph_alias_table=graph_alias_table,
            import_env=import_env,
            module_id=mid,
        )
        if mid.is_entry and entry_seed_env is not None:
            env.seed_from(entry_seed_env)
        for (t_mid, t_name), t in graph_type_table.items():
            if t_mid == mid:
                env.register_type(t_name, t)
        # Also seed the module's own generic types so bare-name local generic
        # refs in param/return annotations (e.g. `o: Option[T]`) resolve here —
        # mirroring the register_type seeding above for non-generic types.
        for (g_mid, g_name), gdef in graph_generic_table.items():
            if g_mid == mid:
                env.register_generic_type(g_name, gdef)

        for item in program.body.items:
            if not isinstance(item, FuncDef):
                continue
            # Skip shadowing-check names — they are rejected during body check.
            # Here we only skip the resolution to avoid raising prematurely on
            # names that the body-checker will report with a better span.
            if item.name in _BUILTIN_TYPE_NAMES or item.name in _BUILTIN_FUNC_NAMES:
                continue
            if item.return_type is None:
                continue

            type_vars: frozenset[str] = frozenset(item.type_params)
            params: list[ParamSpec] = []
            # required-after-defaulted check (positional-fillable zone only).
            _Checker._check_required_after_defaulted(item.params)
            for p in item.params:
                pt = env.resolve_type_expr(p.type_expr, span=p.span, type_vars=type_vars)
                params.append(
                    ParamSpec(name=p.name, type=pt, kind=p.kind, has_default=p.default is not None)
                )

            result_type = env.resolve_type_expr(
                item.return_type, span=item.span, type_vars=type_vars
            )
            sig = FunctionSignature(
                params=tuple(params), result=result_type, type_params=item.type_params
            )
            func_type = FunctionType(
                params=tuple(p.type for p in params),
                result=result_type,
            )
            result[item.node_id] = (item.name, sig, func_type, item.is_extern)

    return result


# ---------------------------------------------------------------------------
# Per-module checking helper
# ---------------------------------------------------------------------------


def _check_module(
    mid: ModuleId,
    resolved: ResolvedProgram,
    source_text: str,
    capabilities: HostCapabilities,
    graph_type_table: dict[tuple[ModuleId, str], Type],
    import_env_map: Mapping[ModuleId, object],
    graph_func_sig_table: dict[int, tuple[str, FunctionSignature, FunctionType, bool]],
    graph_generic_table: dict[tuple[ModuleId, str], GenericTypeDef],
    graph_alias_table: dict[tuple[ModuleId, str], GenericAliasDef],
    graph_ctor_sig_table: dict[tuple[ModuleId, str, str | None], ConstructorSignature],
    graph_ctor_field_kinds_table: dict[
        tuple[ModuleId, str, str | None], tuple[tuple[str, ParamKind], ...]
    ],
    type_table: TypeTable,
    entry_seed_env: TypeEnvironment | None = None,
) -> CheckedModule:
    """Type-check one module with a module-aware ``TypeEnvironment``.

    The env is seeded with:
    - The module's own types (from ``graph_type_table``).
    - The graph table + import env for cross-module lookups.
    - Binding types (function signatures) from the whole-graph function-signature
      pre-pass (``graph_func_sig_table``), seeded BEFORE any body is checked so
      that cross-module function calls — including those in import cycles — can
      look up callee types regardless of per-module checking order.  The pre-pass
      computed ``(name, FunctionSignature,
      FunctionType)`` for ALL top-level ``FuncDef``s across ALL modules;
      ``node_id``s are globally unique, so seeding the whole table into
      every module's env is safe and collision-free.
    - ``type_table``: the single ``TypeTable`` instance shared by every module
      in this graph (the same one built and dual-written in the type pre-pass),
      so this module's own re-check dual-writes into the same table.
    - ``entry_seed_env``: when given and ``mid`` is the entry module, the session
      type env is seeded first so that prior REPL bindings are available.
    """
    import_env = import_env_map[mid]
    assert isinstance(import_env, ImportEnv)

    env = TypeEnvironment(
        graph_type_table=graph_type_table,
        graph_generic_table=graph_generic_table,
        graph_alias_table=graph_alias_table,
        graph_ctor_sig_table=graph_ctor_sig_table,
        graph_ctor_field_kinds_table=graph_ctor_field_kinds_table,
        import_env=import_env,
        module_id=mid,
        type_table=type_table,
    )

    # Seed from the REPL session type env first (for the entry module in REPL
    # graph mode).  Graph tables override on collision, so the entry's own types
    # and function signatures always shadow any session binding with the same name.
    if mid.is_entry and entry_seed_env is not None:
        env.seed_from(entry_seed_env)

    # Seed env with the module's own fully-resolved types so they're
    # accessible by bare name (no qualifier needed within the module).
    for (t_mid, t_name), t in graph_type_table.items():
        if t_mid == mid:
            env.register_type(t_name, t)

    # Seed binding types from the whole-graph function-signature pre-pass.
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
    for node_id, (name, sig, func_type, is_extern) in graph_func_sig_table.items():
        env.set_binding_type(node_id, func_type)
        env.register_function_signature_by_node_id(node_id, sig)
        env.register_function_signature(name, sig)
        if is_extern:
            env.register_extern_node_id(node_id)

    # Run the full _TypeBuilder + _Checker pipeline on this module's program.
    # builder.collect() → _preregister_funcdef re-registers this module's own
    # function signatures (both _binding_types and _function_signatures), so
    # any same-named collision seeded above is corrected for the current module.
    # check_inhabitation=False: the whole-graph pre-pass (_build_graph_type_table)
    # already ran the inhabitation fixpoint over every module's declarations
    # before Phase 3 started; re-running it per module would be redundant.
    builder = _TypeBuilder(env, module_id=mid)
    builder.collect(resolved.program, check_inhabitation=False)

    checker = _Checker(env=env, resolved=resolved, capabilities=capabilities)
    checker.check_program(resolved.program)

    cp = checker.result(resolved)
    return CheckedModule(
        module_id=mid,
        resolved=resolved,
        node_types=cp.node_types,
        contract_specs=cp.contract_specs,
        call_sites=cp.call_sites,
        warnings=cp.warnings,
        function_signatures=cp.function_signatures,
        cast_specs=cp.cast_specs,
        type_env=cp.type_env,
        import_env=import_env,
        source_text=source_text,
        argument_bindings=cp.argument_bindings,
        partial_calls=cp.partial_calls,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_graph(
    resolved_graph: ResolvedModuleGraph,
    capabilities: HostCapabilities,
    entry_seed_env: TypeEnvironment | None = None,
) -> CheckedModuleGraph:
    """Run the full type-checking pass over a :class:`ResolvedModuleGraph`.

    Parameters
    ----------
    resolved_graph:
        Output of :func:`~agm.agl.scope.graph.resolve_graph`.
    capabilities:
        Immutable host capability catalog (agents, codecs, renderers).
    entry_seed_env:
        When given, the entry module's ``TypeEnvironment`` is seeded from this
        environment before the graph type table and function signatures are
        installed.  Used by the REPL graph mode to make prior session
        bindings available in graph-mode entries.

    Returns
    -------
    CheckedModuleGraph
        Per-module type side tables plus the shared graph type table.

    Raises
    ------
    AglTypeError
        On the first static type violation in any module (first-error abort).
    """
    # One TypeTable shared by every module in this graph: the type pre-pass
    # dual-writes into it below, and Phase 3 re-checks each module's own
    # declarations against the SAME instance, so the whole graph's declarations
    # land in one table regardless of per-module checking order.
    shared_type_table = create_seeded_type_table()

    # Phase 1: build the graph-wide type table with all module types stamped
    # with their owning module_id.  Also collects cross-module generic type defs,
    # parameterized aliases, constructor signatures, and constructor field kinds
    # from the per-module envs built during body resolution.
    (
        graph_type_table,
        graph_generic_table,
        graph_alias_table,
        graph_ctor_sig_table,
        graph_ctor_field_kinds_table,
    ) = _build_graph_type_table(
        resolved_graph,
        type_table=shared_type_table,
        entry_seed_env=entry_seed_env,
    )

    # Phase 2: build the graph-wide function-signature table.
    # Resolves parameter/return TypeExprs for every top-level FuncDef in every
    # module using the graph_type_table (so cross-module type refs in annotations
    # resolve), WITHOUT checking any function body.  Keyed by FuncDef.node_id
    #.
    graph_func_sig_table = _build_graph_func_sig_table(
        resolved_graph,
        graph_type_table,
        graph_generic_table,
        graph_alias_table,
        entry_seed_env=entry_seed_env,
    )

    # Collect import envs for per-module checking.
    import_env_map: dict[ModuleId, object] = {
        mid: rmod.import_env for mid, rmod in resolved_graph.modules.items()
    }

    # Phase 3: type-check each module's bodies.
    # Non-entry modules are checked first, then entry (ordering kept for
    # determinism and for any future ordering-sensitive checks), but function
    # signature availability no longer depends on this order — the whole-graph
    # pre-pass in Phase 2 seeds all binding types before any body is checked,
    # so cross-file mutual recursion is handled correctly.
    checked_modules: dict[ModuleId, CheckedModule] = {}
    all_warnings: list[Diagnostic] = []

    # Check non-entry modules first, then entry.
    ordered_mids = [
        mid for mid in resolved_graph.modules if not mid.is_entry
    ] + [resolved_graph.entry_id]

    for mid in ordered_mids:
        rmod = resolved_graph.modules[mid]
        cm = _check_module(
            mid,
            rmod.resolved,
            rmod.source_text,
            capabilities,
            graph_type_table,
            import_env_map,
            graph_func_sig_table,
            graph_generic_table,
            graph_alias_table,
            graph_ctor_sig_table,
            graph_ctor_field_kinds_table,
            shared_type_table,
            entry_seed_env=entry_seed_env if mid.is_entry else None,
        )
        checked_modules[mid] = cm
        all_warnings.extend(cm.warnings)

    return CheckedModuleGraph(
        modules=checked_modules,
        entry_id=resolved_graph.entry_id,
        graph_type_table=graph_type_table,
        warnings=tuple(all_warnings),
    )
