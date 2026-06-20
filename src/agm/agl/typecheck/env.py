"""Type environment and output types for the AgL type checker.

``TypeEnvironment`` holds:
- The user-declared type namespace (records, enums, aliases, exceptions).
- The variable → Type binding table derived from the scope side tables.

``CheckedProgram`` is the frozen output of the type-checking pass.
``OutputContractSpec`` records the statically derived codec + target type per
``AgentCall`` node.
``AglTypeError`` is the fatal type error raised by the checker.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agm.agl.diagnostics import AglError, Diagnostic
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.scope.imports import ImportEnv, QName
from agm.agl.scope.symbols import BindingRef, ResolvedProgram
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.types import (
    BUILTIN_EXCEPTIONS,
    BUILTIN_PRELUDE_TYPE_NAMES,
    BUILTIN_PRELUDE_TYPES,
    AgentType,
    BoolType,
    CastSpec,
    DecimalType,
    DictType,
    EnumType,
    FunctionType,
    IntType,
    JsonType,
    ListType,
    TextType,
    Type,
    UnitType,
)

# ---------------------------------------------------------------------------
# FunctionSignature — full declared signature of a top-level def
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FunctionSignature:
    """Full declared signature of a top-level ``def``.

    Carries named/default information needed for declared-name call sites.
    The value type (FunctionType) erases names/defaults (plan R7).

    ``params`` — ordered list of (name, type, has_default).
    ``result`` — the declared return type.
    """

    params: tuple[tuple[str, Type, bool], ...]  # (name, type, has_default)
    result: Type


# ---------------------------------------------------------------------------
# AglTypeError
# ---------------------------------------------------------------------------


class AglTypeError(AglError):
    """A fatal static type error.

    Raised by the type checker on the first type violation (Q4: first-error
    abort).  Carries an optional ``SourceSpan`` for source location.
    """


# ---------------------------------------------------------------------------
# OutputContractSpec — per-call contract descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CallSiteRecord:
    """Static call-site descriptor recorded by the checker for one agent/exec call.

    Captured in ``_check_agent_call`` — the one place where the call's resolved
    callee kind, parse policy, and source span are all already in hand — so the
    ``--dry-run`` inventory (design §10.1) is derived from the checker's work
    rather than from a second AST walk.

    ``node_id``
        The ``AgentCall`` node id; keys into ``contract_specs`` and the host's
        materialized-contract table when the call has an output contract.
    ``callee``
        The agent or executor name (``"ask"``, ``"exec"``, or a registered
        agent name).
    ``target_type`` / ``codec_name``
        Inventory data for the call. ``codec_name`` is ``"none"`` for a
        ``unit`` target, which has no output contract.
    ``parse_policy``
        ``"abort"`` / ``"retry[N]"`` / ``"default"`` (when the call set no
        explicit ``on_parse_error`` policy).
    ``line`` / ``col``
        1-based source line and column of the call site.
    """

    node_id: int
    callee: str
    target_type: Type
    codec_name: str
    parse_policy: str
    line: int
    col: int


@dataclass(frozen=True, slots=True)
class OutputContractSpec:
    """Statically derived output contract for one ``AgentCall`` node.

    ``target_type``
        The resolved semantic type the agent's output will be parsed into.
    ``codec_name``
        The codec selected for this call (e.g. ``"text"`` in M1, ``"json"``
        in M2).  When ``structured_exec`` is ``True`` this field holds the
        placeholder value ``"text"`` and is **unused** — S5 will branch on
        ``structured_exec`` to skip codec lookup and return the raw
        ``ExecResult`` handle instead.
    ``strict_json``
        The effective strict-JSON flag for this call (``None`` means the
        codec is not JSON-based and the flag is irrelevant; in M1 this is
        always ``None`` since the only codec is ``"text"``).
    ``structured_exec``
        ``True`` for the structured ``exec`` form (target is ``ExecResult``):
        returns the raw result record, does not parse stdout, does not raise
        on nonzero exit.  ``False`` (the default) for all other calls.  S5
        must branch on this flag to skip the codec/parse pipeline entirely.
    """

    target_type: Type
    codec_name: str
    strict_json: bool | None
    structured_exec: bool = False


# ---------------------------------------------------------------------------
# CheckedProgram — output of the type-checking pass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckedProgram:
    """Immutable output of the type-checking pass.

    ``resolved``
        The ``ResolvedProgram`` from the scope pass (carries the original
        ``Program`` and scope side tables).
    ``node_types``
        Maps ``node_id`` → resolved ``Type`` for every expression node that
        was type-checked.  Statement nodes are not entered here.
    ``contract_specs``
        Maps ``AgentCall.node_id`` → ``OutputContractSpec`` for call sites that
        parse output. ``unit`` agent calls are omitted because they have no
        output contract.
    ``call_sites``
        Tuple of ``CallSiteRecord`` — one per agent-call/exec site, in source
        order — captured by the checker.  The ``--dry-run`` inventory (§10.1) is
        built from this plus ``contract_specs``; it is never re-derived by
        re-walking the AST.
    ``warnings``
        Tuple of warning-severity ``Diagnostic`` records collected during the
        pass.  The checker raises on the first *error*; warnings are
        accumulated and returned here.
    ``type_env``
        The complete ``TypeEnvironment`` built during the pass.  It carries the
        full user-declared type namespace (records, enums, aliases, built-in
        exceptions) so downstream consumers (the interpreter) can resolve
        constructors without reconstructing it.  This is the public contract
        for type-namespace access after checking.
    """

    resolved: ResolvedProgram
    node_types: dict[int, Type]
    contract_specs: dict[int, OutputContractSpec]
    call_sites: tuple[CallSiteRecord, ...]
    warnings: tuple[Diagnostic, ...]
    type_env: TypeEnvironment
    function_signatures: dict[str, FunctionSignature]
    cast_specs: dict[int, CastSpec]


# ---------------------------------------------------------------------------
# TypeEnvironment — mutable state during type checking
# ---------------------------------------------------------------------------


class TypeEnvironment:
    """Mutable type environment used during the type-checking pass.

    Holds the type-declaration namespace (records, enums, exception types, and
    aliases) and a mapping from declaration ``node_id`` → ``Type`` for binding
    resolution during the check pass.

    The ``TypeEnvironment`` is constructed and populated by the
    ``_TypeBuilder`` pre-pass; the main ``_Checker`` visitor then queries it.

    Graph mode (M4)
    ---------------
    When ``graph_type_table``, ``import_env``, and ``module_id`` are supplied,
    the environment becomes module-aware:

    - ``graph_type_table`` maps ``(ModuleId, name)`` to the fully-built
      ``Type`` objects stamped with their owning ``module_id``.  Built once by
      the graph pre-pass; shared (read-only) across all per-module envs.
    - ``import_env`` is the per-module :class:`~agm.agl.scope.imports.ImportEnv`
      produced by M3.  Used to resolve qualified and open-imported type names.
    - ``module_id`` is the owning module of the current env.  ``::Name``
      (empty-segment qualifier) resolves against this module's own types.

    These fields are ``None`` in the single-program path; the resolution logic
    in ``resolve_type_expr`` and ``_resolve_name_type`` checks for ``None``
    before entering the module-aware branches, so the existing path is
    unchanged when these are absent.
    """

    def __init__(
        self,
        *,
        graph_type_table: Mapping[tuple[ModuleId, str], Type] | None = None,
        import_env: ImportEnv | None = None,
        module_id: ModuleId = ENTRY_ID,
    ) -> None:
        # user-declared types (records, enums) — name → Type
        self._types: dict[str, Type] = {}
        # alias targets — name → resolved Type (cycle detection uses seen set)
        self._alias_targets: dict[str, object] = {}  # stores raw TypeExpr until resolved
        # Binding node_id → Type (populated as declarations are checked).
        self._binding_types: dict[int, Type] = {}
        # Function signatures — name → FunctionSignature (for declared-name calls).
        self._function_signatures: dict[str, FunctionSignature] = {}
        # Node-id-keyed function signatures — decl_node_id → FunctionSignature.
        # Populated in graph mode by the whole-graph function-signature pre-pass so
        # that _check_declared_name_call can look up the CORRECT signature for a
        # cross-module callee by its globally-unique decl_node_id rather than by
        # bare name (which would pick the wrong signature when two modules define
        # functions with the same name but different signatures).
        self._function_signatures_by_node_id: dict[int, FunctionSignature] = {}
        # Graph-mode context (M4): None in single-program path.
        self._graph_type_table: Mapping[tuple[ModuleId, str], Type] | None = graph_type_table
        self._import_env: ImportEnv | None = import_env
        self._module_id: ModuleId = module_id
        # Built-in exception types are always available.
        for exc_name, exc_type in BUILTIN_EXCEPTIONS.items():
            self._types[exc_name] = exc_type
        # Built-in prelude types (AgL v2: ExecResult, ParsePolicy) are always available.
        for prelude_name, prelude_type in BUILTIN_PRELUDE_TYPES.items():
            self._types[prelude_name] = prelude_type

    # --- Type namespace queries ---

    def has_type(self, name: str) -> bool:
        return name in self._types

    def get_type(self, name: str) -> Type | None:
        return self._types.get(name)

    def register_type(self, name: str, typ: Type) -> None:
        self._types[name] = typ

    def unregister_name(self, name: str) -> None:
        """Remove a user *name* from BOTH the type and alias namespace tables.

        Used by the type-builder when an incremental-session entry redeclares a
        *seeded* name with a different kind (e.g. a seeded ``record R`` redefined
        as ``type R = int``).  ``_types`` (records/enums/exceptions) and
        ``_alias_targets`` (aliases) are separate tables, so a cross-kind
        redefinition would otherwise leave a stale entry in the other table and
        make ``get_type`` disagree with annotation/constructor resolution.
        Dropping the name from both tables before the new kind is registered
        keeps the two namespaces mutually exclusive for user names.

        Built-in exception names and built-in prelude type names are never
        removed: they are non-shadowable (rejected earlier by
        ``_BUILTIN_TYPE_NAMES``), so the builder never calls this for them,
        but the guard makes the helper safe to call defensively.
        """
        if name in BUILTIN_EXCEPTIONS or name in BUILTIN_PRELUDE_TYPE_NAMES:
            return
        self._types.pop(name, None)
        self._alias_targets.pop(name, None)

    def register_alias(self, name: str, target_expr: object) -> None:
        """Store the raw TypeExpr for *name*; resolved lazily by resolve_type_expr."""
        self._alias_targets[name] = target_expr

    def get_alias_target_expr(self, name: str) -> object | None:
        return self._alias_targets.get(name)

    def resolve_named_type(self, name: str) -> Type | None:
        """Resolve a type *name* alias-transparently to a semantic ``Type``.

        Returns the resolved ``Type`` for a record/enum/exception name or an
        alias chain (multi-hop, alias-of-alias) that bottoms out in a named
        type; ``None`` if the name is unknown or names a non-nominal alias
        target (e.g. an alias of ``list[int]``, which has no single name).
        Used for alias-transparent qualifier resolution in qualified
        constructors and ``is`` tests (design §5.4).

        In graph mode, also searches open-imported types when the name is not
        found locally.
        """
        if name in self._types or name in self._alias_targets:
            try:
                return self._resolve_name_type(name, span=None, _resolving=frozenset())
            except AglTypeError:
                return None
        # Graph mode: look up via open imports.
        if self._import_env is not None and self._graph_type_table is not None:
            candidates = self._import_env.unqualified.get(name, frozenset())
            type_candidates = [
                qn for qn in candidates if (qn[0], qn[1]) in self._graph_type_table
            ]
            if len(type_candidates) == 1:
                return self._graph_type_table.get((type_candidates[0][0], type_candidates[0][1]))
        return None

    # --- Function signature table ---

    def register_function_signature(self, name: str, sig: FunctionSignature) -> None:
        self._function_signatures[name] = sig

    def get_function_signature(self, name: str) -> FunctionSignature | None:
        return self._function_signatures.get(name)

    def all_function_signatures(self) -> dict[str, FunctionSignature]:
        return dict(self._function_signatures)

    def register_function_signature_by_node_id(
        self, node_id: int, sig: FunctionSignature
    ) -> None:
        """Register a function signature keyed by its declaration ``node_id``.

        Used by the graph pre-pass to seed every module's env with ALL
        function signatures before any body is checked.  Because ``node_id``
        is globally unique (M2), signatures from different modules never
        collide here even when two modules define functions with the same name.
        """
        self._function_signatures_by_node_id[node_id] = sig

    def get_function_signature_by_node_id(self, node_id: int) -> FunctionSignature | None:
        """Return the function signature for a callee's declaration ``node_id``.

        Used by ``_check_declared_name_call`` to look up the correct signature
        for a callee by its globally-unique declaration node id, avoiding the
        name collision problem when two modules define same-named functions with
        different signatures.

        Populated in graph mode by the whole-graph function-signature pre-pass
        (via :meth:`register_function_signature_by_node_id`) AND in single-
        program mode by ``_preregister_funcdef`` (which always seeds both the
        name-keyed and node-id-keyed tables).  Returns ``None`` only for
        syntactically impossible cases (e.g. a callee node_id not registered
        because the function body check raised before registration).
        """
        return self._function_signatures_by_node_id.get(node_id)

    # --- Binding type table ---

    def set_binding_type(self, node_id: int, typ: Type) -> None:
        self._binding_types[node_id] = typ

    def get_binding_type(self, node_id: int) -> Type | None:
        return self._binding_types.get(node_id)

    def resolve_binding(self, ref: BindingRef) -> Type | None:
        """Return the declared type for a ``BindingRef``."""
        return self._binding_types.get(ref.decl_node_id)

    # --- Type expression resolution ---

    def resolve_type_expr(
        self,
        type_expr: object,
        *,
        span: SourceSpan | None = None,
        _resolving: frozenset[str] | None = None,
    ) -> Type:
        """Resolve a ``TypeExpr`` AST node to a semantic ``Type``.

        Aliases are resolved transitively; cycles are detected and reported
        as ``AglTypeError``.

        Parameters
        ----------
        type_expr:
            A ``TypeExpr`` node from ``agm.agl.syntax.types``.
        span:
            Override span for error messages (defaults to the node's span).
        _resolving:
            Internal: set of alias names currently being resolved (cycle
            detection).
        """
        from agm.agl.syntax.types import (
            AgentT,
            BoolT,
            DecimalT,
            DictT,
            FuncT,
            IntT,
            JsonT,
            ListT,
            NameT,
            TextT,
            UnitT,
        )

        if _resolving is None:
            _resolving = frozenset()

        if isinstance(type_expr, TextT):
            return TextType()
        if isinstance(type_expr, JsonT):
            return JsonType()
        if isinstance(type_expr, BoolT):
            return BoolType()
        if isinstance(type_expr, IntT):
            return IntType()
        if isinstance(type_expr, DecimalT):
            return DecimalType()
        if isinstance(type_expr, UnitT):
            return UnitType()
        if isinstance(type_expr, AgentT):
            return AgentType()
        if isinstance(type_expr, FuncT):
            params = tuple(
                self.resolve_type_expr(p, _resolving=_resolving) for p in type_expr.params
            )
            result = self.resolve_type_expr(type_expr.result, _resolving=_resolving)
            return FunctionType(params=params, result=result)
        if isinstance(type_expr, ListT):
            elem = self.resolve_type_expr(type_expr.elem, _resolving=_resolving)
            return ListType(elem=elem)
        if isinstance(type_expr, DictT):
            val = self.resolve_type_expr(type_expr.value, _resolving=_resolving)
            return DictType(value=val)
        if isinstance(type_expr, NameT):
            eff_span = span if span is not None else type_expr.span
            if type_expr.module_qualifier is not None:
                return self._resolve_qualified_name_type(
                    type_expr.module_qualifier, type_expr.name, span=eff_span
                )
            return self._resolve_name_type(
                type_expr.name,
                span=eff_span,
                _resolving=_resolving,
            )
        raise AglTypeError(
            f"Unknown type expression: {type_expr!r}",
            span=span,
        )

    def _resolve_name_type(
        self,
        name: str,
        *,
        span: SourceSpan | None,
        _resolving: frozenset[str],
    ) -> Type:
        # Check alias table first (aliases are raw TypeExpr, resolved on demand).
        if name in self._alias_targets:
            if name in _resolving:
                raise AglTypeError(
                    f"Type alias '{name}' is part of a cycle.",
                    span=span,
                )
            target_expr = self._alias_targets[name]
            return self.resolve_type_expr(
                target_expr,
                span=span,
                _resolving=_resolving | {name},
            )
        # Direct named type (record, enum, exception, prelude).
        typ = self._types.get(name)
        if typ is not None:
            return typ
        # Graph mode: unqualified lookup through open-imported names.
        if self._import_env is not None and self._graph_type_table is not None:
            candidates = self._import_env.unqualified.get(name, frozenset())
            # Filter to candidates that are type names in the graph type table.
            type_candidates: list[QName] = [
                qn for qn in candidates if (qn[0], qn[1]) in self._graph_type_table
            ]
            if len(type_candidates) == 1:
                qn = type_candidates[0]
                return self._graph_type_table[(qn[0], qn[1])]
            elif len(type_candidates) > 1:
                # Ambiguous: multiple modules export this type name.
                sorted_candidates = sorted(
                    f"{qn[0].dotted()}::{qn[1]}" for qn in type_candidates
                )
                raise AglTypeError(
                    f"Ambiguous type '{name}': it is exported by multiple modules "
                    f"({', '.join(sorted_candidates)}). "
                    f"Use a qualified reference to disambiguate.",
                    span=span,
                )
        raise AglTypeError(
            f"Unknown type '{name}'.",
            span=span,
        )

    def _resolve_qualified_name_type(
        self,
        qualifier: object,  # Qualifier from agm.agl.syntax.types
        name: str,
        *,
        span: SourceSpan | None,
    ) -> Type:
        """Resolve a module-qualified type reference ``QUALIFIER::Name``.

        Called only in graph mode.  Falls back to the local type namespace
        (prelude / built-ins) when the qualifier is empty (``::Name``
        self-reference to the current module) and no graph context exists.
        """
        from agm.agl.syntax.types import Qualifier

        assert isinstance(qualifier, Qualifier)

        if self._graph_type_table is None:
            # Single-module path: module qualifiers have no graph table to consult.
            # ::Name (empty segments) = current module's own type.
            if not qualifier.segments:
                return self._resolve_name_type(name, span=span, _resolving=frozenset())
            raise AglTypeError(
                f"Module qualifier '{'.'.join(qualifier.segments)}::' cannot be resolved "
                "outside of a module graph.",
                span=span,
            )

        # ``::Name`` — self-reference to the current module's own type.
        if not qualifier.segments:
            key = (self._module_id, name)
            t = self._graph_type_table.get(key)
            if t is not None:
                return t
            # Fall back to own local types (covers built-ins and prelude).
            return self._resolve_name_type(name, span=span, _resolving=frozenset())

        # Qualified reference: resolve via ImportEnv.
        assert self._import_env is not None
        handle = qualifier.segments
        handle_map = self._import_env.qualified.get(handle)
        if handle_map is None:
            raise AglTypeError(
                f"Unknown module qualifier '{'.'.join(handle)}::'. "
                "The module has not been imported or the qualifier is not in scope.",
                span=span,
            )
        qname = handle_map.get(name)
        if qname is None:
            raise AglTypeError(
                f"Type '{name}' is not accessible via qualifier '{'.'.join(handle)}::'. "
                "It may not be in the imported set S, or may not be exported.",
                span=span,
            )
        t = self._graph_type_table.get((qname[0], qname[1]))
        if t is not None:
            return t
        # Name is in S but isn't a type in the graph table — it might be a function.
        raise AglTypeError(
            f"'{'.'.join(handle)}::{name}' does not name a type.",
            span=span,
        )

    def non_builtin_type_items(self) -> list[tuple[str, Type]]:
        """Return ``(name, type)`` pairs for all non-builtin registered types.

        Used by the graph pre-pass to collect type shells into the shared
        ``graph_type_table`` without accessing the private ``_types`` dict.
        Builtins (exception types and prelude types) are excluded.
        """
        builtin = frozenset(BUILTIN_EXCEPTIONS) | BUILTIN_PRELUDE_TYPE_NAMES
        return [(name, t) for name, t in self._types.items() if name not in builtin]

    def all_declared_type_names(self) -> frozenset[str]:
        """Return the set of all registered type names (including built-ins)."""
        return frozenset(self._types) | frozenset(self._alias_targets)

    def get_open_imported_enum_candidates(
        self, variant_name: str
    ) -> list[tuple[EnumType, str]]:
        """Return all open-imported ``EnumType`` values that have *variant_name* as a variant.

        Used by the unqualified constructor checker (graph mode) to find enum
        types that are open-imported but not locally declared.  Returns a list
        of ``(EnumType, variant_name)`` pairs, deterministically ordered by
        ``(module_id, enum_name)``.

        Returns an empty list in single-program mode (no graph table).
        """
        if self._import_env is None or self._graph_type_table is None:
            return []
        results: list[tuple[EnumType, str]] = []
        # Iterate over unqualified open-imported names, looking for enum types
        # whose variants include variant_name.
        seen: set[tuple[ModuleId, str]] = set()
        for _exposed_name, qnames in self._import_env.unqualified.items():
            for qn in qnames:
                key = (qn[0], qn[1])
                if key in seen:
                    continue
                seen.add(key)
                t = self._graph_type_table.get(key)
                if isinstance(t, EnumType) and variant_name in t.variants:
                    results.append((t, variant_name))
        # Deterministic order: sort by (module_id segments, enum_name).
        def _sort_key(pair: tuple[EnumType, str]) -> tuple[tuple[str, ...], str]:
            return (pair[0].module_id.segments, pair[0].name)

        results.sort(key=_sort_key)
        return results

    # --- Seeding support ---

    def seed_from(self, other: TypeEnvironment) -> None:
        """Copy *other*'s user-declared types, aliases, and binding types in.

        Used to pre-populate a fresh environment with a session's accumulated
        state before checking a new entry.  Built-in exception types and
        built-in prelude types are already present in every fresh environment
        and are not copied from the source.  Binding types are keyed by
        globally-unique ``decl_node_id`` so they never collide across entries.
        """
        builtin = frozenset(BUILTIN_EXCEPTIONS) | BUILTIN_PRELUDE_TYPE_NAMES
        for name, typ in other._types.items():
            if name not in builtin:
                self._types[name] = typ
        self._alias_targets.update(other._alias_targets)
        self._binding_types.update(other._binding_types)
        self._function_signatures.update(other._function_signatures)
        self._function_signatures_by_node_id.update(other._function_signatures_by_node_id)
