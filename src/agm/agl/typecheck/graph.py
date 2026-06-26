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

2. **Graph function-signature pre-pass** — resolve the parameter and return type
   annotations for EVERY top-level ``FuncDef`` in EVERY module (using the
   ``graph_type_table`` and each module's ``ImportEnv`` for cross-module type
   refs), producing a ``graph_func_sig_table`` mapping each ``FuncDef.node_id``
   (globally unique per M2) to ``(FunctionSignature, FunctionType)``.  No
   function body is checked in this phase.  The result is used in Phase 3 to
   seed EVERY module's env with ALL function binding types before any body is
   checked, enabling cross-file mutual recursion (D8/§8.2): a call to a not-yet-
   checked module's function resolves its callee type from the pre-pass table.

3. **Per-module type-check** — for each module, build a module-aware
   :class:`~agm.agl.typecheck.env.TypeEnvironment` seeded with the module's own
   types (from the graph table), ALL function binding types (from the
   function-signature pre-pass), and the graph table + import env for cross-module
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
from agm.agl.semantics.types import (
    CastSpec,
    FunctionType,
    Type,
)
from agm.agl.syntax.nodes import EnumDef, ExceptionDef, FuncDef, Program, RecordDef, TypeAlias
from agm.agl.typecheck.checker import _Checker, _TypeBuilder
from agm.agl.typecheck.env import (
    CallSiteRecord,
    ConstructorSignature,
    FunctionSignature,
    GenericTypeDef,
    OutputContractSpec,
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
        The :class:`~agm.agl.scope.symbols.ResolvedProgram` from M3.
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
    source_text: str


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

    Delegates to :meth:`~agm.agl.typecheck.checker._TypeBuilder.collect_shells_only`,
    the public API added to ``_TypeBuilder`` for this purpose.
    """
    assert isinstance(program, Program)
    builder.collect_shells_only(program)


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
            builder.ensure_built_record(item.name)
            t = cross_env.get_type(item.name)
            if t is not None:
                # Non-generic record: update the graph table with the fully-built type.
                graph_type_table[(mid, item.name)] = t
            # Generic record: body registered in _generic_types (no _types entry);
            # graph_type_table retains the shell from Step A.  Cross-module generic
            # constructor calls use _graph_generic_table / _graph_ctor_sig_table instead.
            return
        if isinstance(item, EnumDef) and item.name == name:
            builder.ensure_built_enum(item.name)
            t = cross_env.get_type(item.name)
            if t is not None:
                graph_type_table[(mid, item.name)] = t
            return
        if isinstance(item, ExceptionDef) and item.name == name:
            builder.ensure_built_exception(name)
            typ = cross_envs[mid].get_type(name)
            assert typ is not None, f"Exception type {name!r} not registered"
            graph_type_table[(mid, name)] = typ
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
            if isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
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
    from agm.agl.syntax.types import AppliedT, DictT, FuncT, ListT, NameT

    deps: list[tuple[ModuleId, str]] = []

    def _walk(te: object) -> None:
        if isinstance(te, (NameT, AppliedT)):
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
            if isinstance(te, AppliedT):
                for arg in te.args:
                    _walk(arg)
        elif isinstance(te, (ListT, DictT)):
            inner = te.elem if isinstance(te, ListT) else te.value
            _walk(inner)
        elif isinstance(te, FuncT):
            for p in te.params:
                _walk(p)
            _walk(te.result)

    _walk(type_expr)
    return deps


def _collect_unqualified_type_name_deps(
    name: str,
    owning_mid: ModuleId,
    import_env: ImportEnv,
    all_type_keys: set[tuple[ModuleId, str]],
) -> list[tuple[ModuleId, str]]:
    """Return graph dependencies for an unqualified type name."""
    own_key = (owning_mid, name)
    if own_key in all_type_keys:
        return [own_key]

    deps: list[tuple[ModuleId, str]] = []
    candidates = import_env.unqualified.get(name, frozenset())
    for qn in candidates:
        key = (qn[0], qn[1])
        if key in all_type_keys:
            deps.append(key)
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
            if isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
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
                elif isinstance(item, ExceptionDef):
                    for fd in item.fields:
                        item_deps.extend(
                            _collect_type_expr_deps(
                                fd.type_expr, mid, import_env, all_type_keys
                            )
                        )
                    if item.base is not None:
                        item_deps.extend(
                            _collect_unqualified_type_name_deps(
                                item.base, mid, import_env, all_type_keys
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
) -> tuple[
    dict[tuple[ModuleId, str], Type],
    dict[tuple[ModuleId, str], GenericTypeDef],
    dict[tuple[ModuleId, str, str | None], ConstructorSignature],
]:
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

    for mid, rmod in resolved_graph.modules.items():
        env = TypeEnvironment(module_id=mid)
        # The builder is transient: it only collects shells into ``env`` (which
        # bootstraps ``graph_type_table`` below).  Body resolution uses the
        # cross-module builders built later, not this one.
        _collect_shells_only(_TypeBuilder(env, module_id=mid), rmod.resolved.program)
        per_module_envs[mid] = env

    # Collect record/enum shells into the shared graph type table.
    # Aliases are NOT added here — their entries will be written in Step C
    # after their target type is resolved.
    graph_type_table: dict[tuple[ModuleId, str], Type] = {}
    for mid, env in per_module_envs.items():
        for name, t in env.non_builtin_type_items():
            graph_type_table[(mid, name)] = t

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
        for name, t in per_module_envs[mid].non_builtin_type_items():
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

    # Collect cross-module generic type definitions and constructor signatures from
    # the per-module envs (which now have fully resolved bodies).  These are indexed
    # by (ModuleId, name) so that the per-module checker can look them up when it
    # encounters a module-qualified generic constructor call (e.g. lib::Box[int](v:1)).
    graph_generic_table: dict[tuple[ModuleId, str], GenericTypeDef] = {}
    graph_ctor_sig_table: dict[tuple[ModuleId, str, str | None], ConstructorSignature] = {}
    for mid, cross_env in cross_envs.items():
        for name, gdef in cross_env.all_generic_types().items():
            graph_generic_table[(mid, name)] = gdef
        for (owner_name, variant), sig in cross_env.all_constructor_sigs():
            graph_ctor_sig_table[(mid, owner_name, variant)] = sig

    return graph_type_table, graph_generic_table, graph_ctor_sig_table


# ---------------------------------------------------------------------------
# Phase 2: whole-graph function-signature pre-pass
# ---------------------------------------------------------------------------


def _build_graph_func_sig_table(
    resolved_graph: ResolvedModuleGraph,
    graph_type_table: dict[tuple[ModuleId, str], Type],
    graph_generic_table: dict[tuple[ModuleId, str], GenericTypeDef],
) -> dict[int, tuple[str, FunctionSignature, FunctionType]]:
    """Phase 2: compute function signatures for ALL top-level FuncDefs across all modules.

    Returns a table mapping each ``FuncDef.node_id`` (globally unique per M2)
    to ``(name, FunctionSignature, FunctionType)`` — the declared function name,
    the full declared signature (names, types, has-default flags), and the erased
    value type.

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
    from agm.agl.typecheck.env import AglTypeError

    result: dict[int, tuple[str, FunctionSignature, FunctionType]] = {}

    for mid, rmod in resolved_graph.modules.items():
        program = rmod.resolved.program
        assert isinstance(program, Program)

        import_env = rmod.import_env
        # Build a cross-module-aware env for this module, seeded with its own
        # types so bare-name local type refs in param annotations resolve.
        env = TypeEnvironment(
            graph_type_table=graph_type_table,
            graph_generic_table=graph_generic_table,
            import_env=import_env,
            module_id=mid,
        )
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

            type_vars: frozenset[str] = frozenset(item.type_params)
            params: list[tuple[str, Type, bool]] = []
            seen_required = True
            for p in item.params:
                pt = env.resolve_type_expr(p.type_expr, span=p.span, type_vars=type_vars)
                has_default = p.default is not None
                if seen_required and has_default:
                    seen_required = False
                elif not seen_required and not has_default:
                    raise AglTypeError(
                        f"Parameter '{p.name}' has no default but follows a defaulted "
                        "parameter. Required parameters must come before parameters with "
                        "defaults.",
                        span=p.span,
                    )
                params.append((p.name, pt, has_default))

            result_type = env.resolve_type_expr(
                item.return_type, span=item.span, type_vars=type_vars
            )
            sig = FunctionSignature(
                params=tuple(params), result=result_type, type_params=item.type_params
            )
            func_type = FunctionType(
                params=tuple(pt for _, pt, _ in params),
                result=result_type,
            )
            result[item.node_id] = (item.name, sig, func_type)

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
    graph_func_sig_table: dict[int, tuple[str, FunctionSignature, FunctionType]],
    graph_generic_table: dict[tuple[ModuleId, str], GenericTypeDef],
    graph_ctor_sig_table: dict[tuple[ModuleId, str, str | None], ConstructorSignature],
    entry_seed_env: TypeEnvironment | None = None,
) -> CheckedModule:
    """Type-check one module with a module-aware ``TypeEnvironment``.

    The env is seeded with:
    - The module's own types (from ``graph_type_table``).
    - The graph table + import env for cross-module lookups.
    - Binding types (function signatures) from the whole-graph function-signature
      pre-pass (``graph_func_sig_table``), seeded BEFORE any body is checked so
      that cross-module function calls — including those in import cycles (D8,
      §8.2 cross-file mutual recursion) — can look up callee types regardless of
      per-module checking order.  The pre-pass computed ``(name, FunctionSignature,
      FunctionType)`` for ALL top-level ``FuncDef``s across ALL modules;
      ``node_id``s are globally unique (M2), so seeding the whole table into
      every module's env is safe and collision-free.
    - ``entry_seed_env``: when given and ``mid`` is the entry module, the session
      type env is seeded first so that prior REPL bindings are available (M6).
    """
    import_env = import_env_map[mid]
    assert isinstance(import_env, ImportEnv)

    env = TypeEnvironment(
        graph_type_table=graph_type_table,
        graph_generic_table=graph_generic_table,
        graph_ctor_sig_table=graph_ctor_sig_table,
        import_env=import_env,
        module_id=mid,
    )

    # Seed from the REPL session type env first (for the entry module in M6 REPL
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
    # env before any body is checked, enabling cross-file mutual recursion (D8/§8.2).
    #
    # Three tables are seeded:
    # - _binding_types (node_id-keyed, globally unique): used by _check_varref to
    #   look up the callee type when the callee VarRef resolves to a cross-module
    #   FuncDef's decl_node_id.  Seeding the entire pre-pass table is safe because
    #   node_ids are globally unique per M2.
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
    for node_id, (name, sig, func_type) in graph_func_sig_table.items():
        env.set_binding_type(node_id, func_type)
        env.register_function_signature_by_node_id(node_id, sig)
        env.register_function_signature(name, sig)

    # Run the full _TypeBuilder + _Checker pipeline on this module's program.
    # builder.collect() → _preregister_funcdef re-registers this module's own
    # function signatures (both _binding_types and _function_signatures), so
    # any same-named collision seeded above is corrected for the current module.
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
        source_text=source_text,
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
        installed.  Used by the REPL graph mode (M6) to make prior session
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
    # Phase 1: build the graph-wide type table with all module types stamped
    # with their owning module_id.  Also collects cross-module generic type defs
    # and constructor signatures from the per-module envs built during body resolution.
    graph_type_table, graph_generic_table, graph_ctor_sig_table = _build_graph_type_table(
        resolved_graph
    )

    # Phase 2: build the graph-wide function-signature table.
    # Resolves parameter/return TypeExprs for every top-level FuncDef in every
    # module using the graph_type_table (so cross-module type refs in annotations
    # resolve), WITHOUT checking any function body.  Keyed by FuncDef.node_id
    # (globally unique per M2).
    graph_func_sig_table = _build_graph_func_sig_table(
        resolved_graph, graph_type_table, graph_generic_table
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
    # so cross-file mutual recursion (D8/§8.2) is handled correctly.
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
            graph_ctor_sig_table,
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
