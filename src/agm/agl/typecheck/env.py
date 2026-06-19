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

from dataclasses import dataclass

from agm.agl.diagnostics import AglError, Diagnostic
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
    """

    def __init__(self) -> None:
        # user-declared types (records, enums) — name → Type
        self._types: dict[str, Type] = {}
        # alias targets — name → resolved Type (cycle detection uses seen set)
        self._alias_targets: dict[str, object] = {}  # stores raw TypeExpr until resolved
        # Binding node_id → Type (populated as declarations are checked).
        self._binding_types: dict[int, Type] = {}
        # Function signatures — name → FunctionSignature (for declared-name calls).
        self._function_signatures: dict[str, FunctionSignature] = {}
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
        """
        if name not in self._types and name not in self._alias_targets:
            return None
        try:
            return self._resolve_name_type(name, span=None, _resolving=frozenset())
        except AglTypeError:
            return None

    # --- Function signature table ---

    def register_function_signature(self, name: str, sig: FunctionSignature) -> None:
        self._function_signatures[name] = sig

    def get_function_signature(self, name: str) -> FunctionSignature | None:
        return self._function_signatures.get(name)

    def all_function_signatures(self) -> dict[str, FunctionSignature]:
        return dict(self._function_signatures)

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
            return self._resolve_name_type(
                type_expr.name,
                span=span if span is not None else type_expr.span,
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
        # Direct named type (record, enum, exception).
        typ = self._types.get(name)
        if typ is not None:
            return typ
        raise AglTypeError(
            f"Unknown type '{name}'.",
            span=span,
        )

    def all_declared_type_names(self) -> frozenset[str]:
        """Return the set of all registered type names (including built-ins)."""
        return frozenset(self._types) | frozenset(self._alias_targets)

    # --- Seeding support (incremental REPL sessions) ---

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
