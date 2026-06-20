"""Graph-aware type-checking pass for the AgL module system (M4).

``check_graph(resolved_graph, capabilities)`` runs the full type-checking pass
over a :class:`~agm.agl.scope.graph.ResolvedModuleGraph`, producing a
:class:`CheckedModuleGraph`.

Algorithm
---------
1. **Graph type pre-pass** — collect ALL public type declarations across every
   module, resolve their bodies (stamping each ``RecordType``/``EnumType`` with
   the owning ``ModuleId``), and build the shared ``graph_type_table``.  Because
   imported modules are declaration-only and type bodies cannot be structurally
   recursive across files (cycles are allowed in the *import* graph, but not in
   the *structural-type-definition* graph), every type in the table can be built
   before any function or expression body is checked.

   The pre-pass is genuinely whole-graph two-phase:

   a. **Shells** — ALL type shells for ALL modules are registered into the shared
      ``graph_type_table`` first (records and enums as empty shells; type aliases
      are tracked separately in a per-module env).
   b. **Bodies in topological order** — the structural type-definition dependency
      graph (a record/enum/alias depends on every module-qualified or unqualified
      type named in its field/variant/alias-target type expressions, across modules)
      is computed, and bodies are resolved in topological order so that each
      referenced type is fully built (fields/variants populated) BEFORE it is
      embedded by-value as another type's field/variant/element type.  Ties are
      broken by ``(ModuleId.segments, name)`` for determinism.  A genuine
      structural type cycle (type that contains itself infinitely) is a static
      error, consistent with the existing single-module behaviour.

2. **Per-module type-check** — for each module, build a module-aware
   :class:`~agm.agl.typecheck.env.TypeEnvironment` seeded with the module's own
   types (from the graph table) and the graph table + import env for cross-module
   lookups, then run the existing :class:`~agm.agl.typecheck.checker._TypeBuilder`
   and :class:`~agm.agl.typecheck.checker._Checker` logic.

Single-module equivalence
-------------------------
A single-module (entry-only) graph checked via :func:`check_graph` is
equivalent to calling :func:`~agm.agl.typecheck.checker.check` directly on the
entry's :class:`~agm.agl.scope.symbols.ResolvedProgram`.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Mapping

from agm.agl.capabilities import HostCapabilities
from agm.agl.diagnostics import Diagnostic
from agm.agl.modules.ids import ModuleId
from agm.agl.scope.graph import ResolvedModuleGraph
from agm.agl.scope.imports import ImportEnv
from agm.agl.scope.symbols import ResolvedProgram
from agm.agl.syntax.nodes import EnumDef, Program, RecordDef, TypeAlias
from agm.agl.typecheck.checker import _Checker, _TypeBuilder
from agm.agl.typecheck.env import (
    CallSiteRecord,
    FunctionSignature,
    OutputContractSpec,
    TypeEnvironment,
)
from agm.agl.typecheck.types import (
    BUILTIN_EXCEPTIONS,
    BUILTIN_PRELUDE_TYPES,
    CastSpec,
    EnumType,
    RecordType,
    Type,
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
        The :class:`~agm.agl.scope.symbols.ResolvedProgram` from M3.
    ``node_types``
        Maps ``node_id`` → resolved :class:`~agm.agl.typecheck.types.Type`
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
        Maps ``Cast.node_id`` → :class:`~agm.agl.typecheck.types.CastSpec`
        for every cast expression in this module.
    ``type_env``
        The module-aware :class:`~agm.agl.typecheck.env.TypeEnvironment`
        built during the pass.
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
        :class:`~agm.agl.typecheck.types.Type` object stamped with the owning
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
    """Run only phase 1 (shell registration) of ``_TypeBuilder.collect``."""
    assert isinstance(program, Program)
    for item in program.body.items:
        if isinstance(item, RecordDef):
            builder._register_name(item.name, item.span)  # noqa: SLF001
            builder._env.unregister_name(item.name)  # noqa: SLF001
            builder._env.register_type(  # noqa: SLF001
                item.name,
                RecordType(name=item.name, fields={}, module_id=builder._module_id),  # noqa: SLF001
            )
            builder._record_defs[item.name] = item  # noqa: SLF001
        elif isinstance(item, EnumDef):
            builder._register_name(item.name, item.span)  # noqa: SLF001
            builder._env.unregister_name(item.name)  # noqa: SLF001
            builder._env.register_type(  # noqa: SLF001
                item.name,
                EnumType(name=item.name, variants={}, module_id=builder._module_id),  # noqa: SLF001
            )
            builder._enum_defs[item.name] = item  # noqa: SLF001
        elif isinstance(item, TypeAlias):
            builder._register_name(item.name, item.span)  # noqa: SLF001
            builder._env.unregister_name(item.name)  # noqa: SLF001
            builder._env.register_alias(item.name, item.type_expr)  # noqa: SLF001


def _resolve_body_for_one(
    mid: ModuleId,
    name: str,
    per_module_builders: dict[ModuleId, _TypeBuilder],
    graph_type_table: dict[tuple[ModuleId, str], Type],
    resolved_graph: ResolvedModuleGraph,
    cross_envs: dict[ModuleId, TypeEnvironment],
) -> None:
    """Resolve the body of one type ``(mid, name)`` and update the graph table.

    Called in topological order so all types this one depends on are already
    fully built before their types are captured by-value here.
    """
    cross_env = cross_envs[mid]
    builder = per_module_builders[mid]

    program = resolved_graph.modules[mid].resolved.program
    assert isinstance(program, Program)

    for item in program.body.items:
        if isinstance(item, RecordDef) and item.name == name:
            builder._ensure_built_record(item.name)  # noqa: SLF001
            t = cross_env.get_type(item.name)
            assert t is not None, f"record '{item.name}' missing after build"
            graph_type_table[(mid, item.name)] = t
            return
        if isinstance(item, EnumDef) and item.name == name:
            builder._ensure_built_enum(item.name)  # noqa: SLF001
            t = cross_env.get_type(item.name)
            assert t is not None, f"enum '{item.name}' missing after build"
            graph_type_table[(mid, item.name)] = t
            return
        if isinstance(item, TypeAlias) and item.name == name:
            resolved = cross_env.resolve_type_expr(item.type_expr, span=item.span)
            graph_type_table[(mid, item.name)] = resolved
            return
    # Unreachable: called only for keys produced by _collect_all_type_keys,
    # which iterates the same program.body.items.
    raise AssertionError(f"type '{name}' not found in module '{mid}'")  # pragma: no cover


def _collect_all_type_keys(
    resolved_graph: ResolvedModuleGraph,
) -> set[tuple[ModuleId, str]]:
    """Collect the set of all user-declared type keys across all modules.

    Returns ``{(ModuleId, name)}`` for every ``RecordDef``, ``EnumDef``, and
    ``TypeAlias`` in every module.  This includes type aliases whose resolved type
    is a primitive (e.g. ``type Number = int``).  Builtin-shadowing types are
    never present here because ``_collect_shells_only`` rejects them earlier.

    This set is used as the universe for the structural type-definition dependency
    graph.  It is LARGER than ``graph_type_table`` during the shell-collection
    step because aliases are not yet resolved to shells there — the graph table is
    only populated with record/enum shells and is updated with alias resolutions
    during topological body resolution.
    """
    all_keys: set[tuple[ModuleId, str]] = set()
    for mid, rmod in resolved_graph.modules.items():
        program = rmod.resolved.program
        assert isinstance(program, Program)
        for item in program.body.items:
            if isinstance(item, (RecordDef, EnumDef, TypeAlias)):
                # Builtin/prelude shadowing is rejected in _collect_shells_only
                # (Step A of _build_graph_type_table), which is called before this
                # function.  Only non-builtin types reach this point.
                all_keys.add((mid, item.name))
    return all_keys


def _collect_type_expr_deps(
    type_expr: object,
    owning_mid: ModuleId,
    import_env: ImportEnv,
    all_type_keys: set[tuple[ModuleId, str]],
) -> list[tuple[ModuleId, str]]:
    """Return the list of ``(ModuleId, name)`` keys that *type_expr* depends on.

    This is used to build the structural type-definition dependency graph used
    by the topological sort in ``_build_graph_type_table``.

    A type expression ``A`` depends on ``(mid_B, B)`` if A's field/variant/alias
    body names B (either qualified or unqualified via open import), AND B is a
    user-declared type (present in ``all_type_keys`` — records, enums, and
    aliases; but NOT built-ins/prelude types which are always available).

    Only ``NameT`` (named type references) create dependencies.  Container types
    (``ListT``, ``DictT``, ``FuncT``) recurse structurally.
    """
    from agm.agl.syntax.types import DictT, FuncT, ListT, NameT

    deps: list[tuple[ModuleId, str]] = []

    def _walk(te: object) -> None:
        if isinstance(te, NameT):
            if te.module_qualifier is not None:
                # Qualified: resolve through import env.
                qualifier = te.module_qualifier
                if not qualifier.segments:
                    # Self-reference ::Name
                    key = (owning_mid, te.name)
                    if key in all_type_keys:
                        deps.append(key)
                else:
                    handle = qualifier.segments
                    handle_map = import_env.qualified.get(handle)
                    if handle_map is not None:
                        qname = handle_map.get(te.name)
                        if qname is not None:
                            key = (qname[0], qname[1])
                            if key in all_type_keys:
                                deps.append(key)
            else:
                # Unqualified: could be own module or open-imported.
                own_key = (owning_mid, te.name)
                if own_key in all_type_keys:
                    deps.append(own_key)
                else:
                    candidates = import_env.unqualified.get(te.name, frozenset())
                    for qn in candidates:
                        key = (qn[0], qn[1])
                        if key in all_type_keys:
                            deps.append(key)
        elif isinstance(te, (ListT, DictT)):
            inner = te.elem if isinstance(te, ListT) else te.value
            _walk(inner)
        elif isinstance(te, FuncT):
            for p in te.params:
                _walk(p)
            _walk(te.result)

    _walk(type_expr)
    return deps


def _compute_type_deps(
    resolved_graph: ResolvedModuleGraph,
    all_type_keys: set[tuple[ModuleId, str]],
) -> dict[tuple[ModuleId, str], list[tuple[ModuleId, str]]]:
    """Compute the structural type-definition dependency graph.

    Returns a dict mapping each ``(ModuleId, name)`` in ``all_type_keys`` to the
    list of ``(ModuleId, name)`` keys it structurally depends on (i.e. whose
    fully-built type it needs before its own body can be resolved by-value).

    Dependencies on built-ins/prelude types are omitted (they are always
    available and do not need to be in the topo sort).
    """
    deps: dict[tuple[ModuleId, str], list[tuple[ModuleId, str]]] = {}

    for mid, rmod in resolved_graph.modules.items():
        program = rmod.resolved.program
        import_env = rmod.import_env
        assert isinstance(program, Program)

        for item in program.body.items:
            if isinstance(item, (RecordDef, EnumDef, TypeAlias)):
                key = (mid, item.name)
                # key is guaranteed to be in all_type_keys: _collect_all_type_keys
                # adds every RecordDef/EnumDef/TypeAlias under the same conditions
                # used here.  Builtin shadowing is rejected in _collect_shells_only
                # before either function is called, so no shadowing key ever appears.
                item_deps: list[tuple[ModuleId, str]] = []
                if isinstance(item, RecordDef):
                    for fd in item.fields:
                        item_deps.extend(
                            _collect_type_expr_deps(
                                fd.type_expr, mid, import_env, all_type_keys
                            )
                        )
                elif isinstance(item, EnumDef):
                    for vd in item.variants:
                        for fd in vd.fields:
                            item_deps.extend(
                                _collect_type_expr_deps(
                                    fd.type_expr, mid, import_env, all_type_keys
                                )
                            )
                else:
                    # Must be TypeAlias (outer isinstance guarantees one of the three).
                    item_deps.extend(
                        _collect_type_expr_deps(
                            item.type_expr, mid, import_env, all_type_keys
                        )
                    )
                # Remove self-deps (would cause a false cycle — handled by
                # _TypeBuilder's within-module structural recursion detection).
                deps[key] = [d for d in item_deps if d != key]

    return deps


def _topological_sort_types(
    all_type_keys: set[tuple[ModuleId, str]],
    deps: dict[tuple[ModuleId, str], list[tuple[ModuleId, str]]],
) -> list[tuple[ModuleId, str]]:
    """Kahn's algorithm: return keys in topological order (leaves first).

    Ties broken by ``(ModuleId.segments, name)`` for determinism.

    Returns the list of keys in an order suitable for body resolution: each key
    appears AFTER all of its dependencies.

    Raises ``_CycleInTypeDeps`` if a structural type cycle is detected (a type
    that contains itself, directly or indirectly, by value).  Structural self-
    recursion within a single module is also detected here and converted to the
    standard error by the per-type resolution step.
    """

    def _sort_key(k: tuple[ModuleId, str]) -> tuple[tuple[str, ...], str]:
        return (k[0].segments, k[1])

    # Build in-degree and adjacency list (for Kahn's).
    # adj[u] = list of nodes that depend ON u (u must be resolved before them).
    in_degree: dict[tuple[ModuleId, str], int] = {k: 0 for k in all_type_keys}
    adj: dict[tuple[ModuleId, str], list[tuple[ModuleId, str]]] = {
        k: [] for k in all_type_keys
    }

    for node, node_deps in deps.items():
        for dep in node_deps:
            # dep is guaranteed to be in adj: _collect_type_expr_deps only
            # adds keys that are in all_type_keys, from which adj is built.
            adj[dep].append(node)
            in_degree[node] = in_degree.get(node, 0) + 1

    # Kahn's: start with all zero-in-degree nodes, sorted for determinism.
    ready: list[tuple[ModuleId, str]] = sorted(
        (k for k, d in in_degree.items() if d == 0),
        key=_sort_key,
    )
    order: list[tuple[ModuleId, str]] = []

    while ready:
        node = ready.pop(0)
        order.append(node)
        dependents = sorted(adj.get(node, []), key=_sort_key)
        for dep in dependents:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                bisect.insort(ready, dep, key=_sort_key)

    if len(order) < len(all_type_keys):
        # There is a cycle in the structural type-definition dependency graph.
        # Find the cycle participants and raise an error.
        remaining = {k for k in all_type_keys if k not in set(order)}
        raise _CycleInTypeDeps(remaining)

    return order


class _CycleInTypeDeps(Exception):
    """Internal: raised when a structural type cycle is detected."""

    def __init__(self, cycle_keys: set[tuple[ModuleId, str]]) -> None:
        self.cycle_keys = cycle_keys


def _build_graph_type_table(
    resolved_graph: ResolvedModuleGraph,
) -> dict[tuple[ModuleId, str], Type]:
    """Phase 1: collect and resolve all public type declarations across all modules.

    Returns a ``graph_type_table`` mapping ``(ModuleId, name)`` → fully-built
    ``RecordType`` / ``EnumType`` / resolved-alias ``Type``.

    The pre-pass is genuinely whole-graph two-phase:

    Step A: Register ALL type shells for ALL modules.  Record and enum shells
            (empty fields/variants) are entered into ``graph_type_table`` so that
            forward references within the same module work during body resolution.
            Type aliases are registered in per-module envs (their target type is
            not known until the alias body is resolved, so they have no shell in
            the table).  All shells are registered before any body is resolved.

    Step B: Compute the structural type-definition dependency graph across all
            modules using the complete set of declared type keys (records, enums,
            and aliases).  A type definition depends on every type named in its
            field/variant/alias-target type expressions, cross-module included.

    Step C: Topologically sort the type definitions by the dependency graph
            (Kahn's algorithm, ties broken by ``(ModuleId.segments, name)``).
            Resolve each type body in that order so every referenced type is
            fully built (fields/variants populated) before it is embedded
            by-value as another type's field/variant/element type.

    A genuine structural type cycle (a type that structurally contains itself,
    making it infinitely sized) is detected and reported as an ``AglTypeError``,
    consistent with how single-module structural recursion is handled by the
    existing ``_TypeBuilder``.

    Import-graph cycles (D8) are allowed and do NOT imply structural type cycles
    — the structural dependency graph may be acyclic even when the import graph
    has cycles (e.g. modA imports modB for a Color enum used in modA.Foo fields,
    and modB imports modA for modA.Foo used in modB.Bar fields: the structural
    order is Color → Foo → Bar, which is acyclic).
    """
    from agm.agl.typecheck.env import AglTypeError

    # Step A: register all type shells for all modules.
    # For records/enums: register empty shells in both the per-module env AND
    # graph_type_table.  For aliases: register in the per-module env only
    # (no shell in graph_type_table — the resolved type is added in Step C).
    per_module_envs: dict[ModuleId, TypeEnvironment] = {}
    per_module_builders: dict[ModuleId, _TypeBuilder] = {}

    for mid, rmod in resolved_graph.modules.items():
        env = TypeEnvironment(module_id=mid)
        builder = _TypeBuilder(env, module_id=mid)
        _collect_shells_only(builder, rmod.resolved.program)
        per_module_envs[mid] = env
        per_module_builders[mid] = builder

    # Collect record/enum shells into the shared graph type table.
    # Aliases are NOT added here — their entries will be written in Step C
    # after their target type is resolved.
    graph_type_table: dict[tuple[ModuleId, str], Type] = {}
    for mid, env in per_module_envs.items():
        for name in env._types:  # noqa: SLF001 – internal table
            if name in BUILTIN_EXCEPTIONS or name in BUILTIN_PRELUDE_TYPES:
                continue
            graph_type_table[(mid, name)] = env._types[name]  # noqa: SLF001

    # Build per-module cross-module-aware environments and builders for
    # body resolution.  Each env knows the full graph_type_table and its own
    # module's ImportEnv so qualified and open-imported type refs resolve.
    cross_envs: dict[ModuleId, TypeEnvironment] = {}
    cross_builders: dict[ModuleId, _TypeBuilder] = {}

    for mid, rmod in resolved_graph.modules.items():
        import_env = rmod.import_env
        cross_env = TypeEnvironment(
            graph_type_table=graph_type_table,
            import_env=import_env,
            module_id=mid,
        )
        # Seed with own type shells so bare-name local refs resolve.
        for name, t in per_module_envs[mid]._types.items():  # noqa: SLF001
            if name not in BUILTIN_EXCEPTIONS and name not in BUILTIN_PRELUDE_TYPES:
                cross_env.register_type(name, t)
        cross_envs[mid] = cross_env
        # Build a _TypeBuilder that uses the cross-module env and has the
        # shells and alias targets registered (for _ensure_built_* to work).
        builder = _TypeBuilder(cross_env, module_id=mid)
        _collect_shells_only(builder, rmod.resolved.program)
        cross_builders[mid] = builder

    # Step B: compute the structural type-definition dependency graph.
    # Use the COMPLETE set of declared type keys (including aliases), NOT just
    # the record/enum shells in graph_type_table.  Aliases depend on their
    # target types just like record fields do.
    all_type_keys = _collect_all_type_keys(resolved_graph)
    deps = _compute_type_deps(resolved_graph, all_type_keys)

    # Step C: resolve bodies in topological order.
    # This guarantees that when type A's body references type B, B's
    # RecordType/EnumType is already fully populated (fields/variants filled in)
    # before A captures it by-value.
    try:
        topo_order = _topological_sort_types(all_type_keys, deps)
    except _CycleInTypeDeps as exc:
        # Structural type cycle across modules: surface a clear error.
        # For same-module structural cycles the _TypeBuilder's within-module
        # detection fires naturally (cycle key in _building set).  For cross-
        # module cycles we report a graph-level error naming one participant.
        def _cycle_sort_key(k: tuple[ModuleId, str]) -> tuple[tuple[str, ...], str]:
            return (k[0].segments, k[1])

        sorted_keys = sorted(exc.cycle_keys, key=_cycle_sort_key)
        mid, name = sorted_keys[0]
        raise AglTypeError(
            f"Type '{name}' in module '{mid.dotted()}' is part of a structural "
            "type cycle (a type that directly or indirectly contains itself by "
            "value). Recursive types are not supported in v1.",
            span=None,
        ) from exc

    for key in topo_order:
        mid, name = key
        _resolve_body_for_one(
            mid=mid,
            name=name,
            per_module_builders=cross_builders,
            graph_type_table=graph_type_table,
            resolved_graph=resolved_graph,
            cross_envs=cross_envs,
        )

    return graph_type_table


# ---------------------------------------------------------------------------
# Per-module checking helper
# ---------------------------------------------------------------------------


def _check_module(
    mid: ModuleId,
    resolved: ResolvedProgram,
    capabilities: HostCapabilities,
    graph_type_table: dict[tuple[ModuleId, str], Type],
    import_env_map: Mapping[ModuleId, object],
    checked_so_far: Mapping[ModuleId, CheckedModule],
) -> CheckedModule:
    """Type-check one module with a module-aware ``TypeEnvironment``.

    The env is seeded with:
    - The module's own types (from ``graph_type_table``).
    - The graph table + import env for cross-module lookups.
    - Binding types (function signatures) from already-checked modules so that
      cross-module function calls can look up callee types.
    """
    import_env = import_env_map[mid]
    assert isinstance(import_env, ImportEnv)

    env = TypeEnvironment(
        graph_type_table=graph_type_table,
        import_env=import_env,
        module_id=mid,
    )

    # Seed env with the module's own fully-resolved types so they're
    # accessible by bare name (no qualifier needed within the module).
    for (t_mid, t_name), t in graph_type_table.items():
        if t_mid == mid:
            env.register_type(t_name, t)

    # Seed binding types from already-checked modules (for cross-module calls).
    # This allows the entry module to look up binding types for imported functions.
    #
    # Ordering dependency: seed_binding_types_from is called BEFORE
    # builder.collect() runs for the current module.  _function_signatures is
    # keyed by bare name, so a same-named function in two different modules would
    # overwrite the seeded value.  This is safe here because real cross-module
    # call type resolution goes through the node-id-keyed _binding_types table
    # (which is globally unique), NOT through _function_signatures.  The
    # _function_signatures seeding is only used so that a callee's signature is
    # readable by name during the entry module's own declared-name call site
    # checking.  The subsequent builder.collect() re-registers the current
    # module's own signatures, which win over any seeded same-named ones.
    for other_cm in checked_so_far.values():
        env.seed_binding_types_from(other_cm.type_env)

    # Run the full _TypeBuilder + _Checker pipeline on this module's program.
    builder = _TypeBuilder(env, module_id=mid)
    builder.collect(resolved.program)

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
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_graph(
    resolved_graph: ResolvedModuleGraph,
    capabilities: HostCapabilities,
) -> CheckedModuleGraph:
    """Run the full type-checking pass over a :class:`ResolvedModuleGraph`.

    Parameters
    ----------
    resolved_graph:
        Output of :func:`~agm.agl.scope.graph.resolve_graph`.
    capabilities:
        Immutable host capability catalog (agents, codecs, renderers).

    Returns
    -------
    CheckedModuleGraph
        Per-module type side tables plus the shared graph type table.

    Raises
    ------
    AglTypeError
        On the first static type violation in any module (first-error abort).
    """
    # Phase 1: build the graph-wide type table with all module types stamped
    # with their owning module_id.
    graph_type_table = _build_graph_type_table(resolved_graph)

    # Collect import envs for per-module checking.
    import_env_map: dict[ModuleId, object] = {
        mid: rmod.import_env for mid, rmod in resolved_graph.modules.items()
    }

    # Phase 2: type-check each module's bodies.
    # Non-entry modules are checked first so their binding types (function
    # signatures) are available when the entry module is checked.  The entry
    # module calls imported functions and needs their binding types seeded.
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
            capabilities,
            graph_type_table,
            import_env_map,
            checked_modules,
        )
        checked_modules[mid] = cm
        all_warnings.extend(cm.warnings)

    return CheckedModuleGraph(
        modules=checked_modules,
        entry_id=resolved_graph.entry_id,
        graph_type_table=graph_type_table,
        warnings=tuple(all_warnings),
    )
