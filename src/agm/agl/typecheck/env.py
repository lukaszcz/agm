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
from typing import cast

from agm.agl.diagnostics import AglError, Diagnostic
from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID, ModuleId
from agm.agl.scope.imports import ImportEnv, QName
from agm.agl.scope.symbols import BindingRef, ResolvedProgram
from agm.agl.semantics.types import (
    BUILTIN_EXCEPTIONS,
    BUILTIN_PRELUDE_TYPE_NAMES,
    BUILTIN_PRELUDE_TYPES,
    AgentType,
    BoolType,
    CastSpec,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
    UnitType,
)
from agm.agl.syntax.nodes import Expr, ParamKind, Pattern
from agm.agl.syntax.spans import SourceSpan

# ---------------------------------------------------------------------------
# ParamSpec — per-parameter descriptor in a FunctionSignature
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """Full descriptor for one parameter in a ``FunctionSignature``.

    ``name``        — the declared parameter name.
    ``type``        — the resolved semantic type.
    ``kind``        — positional-only, standard, or named-only (from the AST).
    ``has_default`` — whether a default expression was provided.
    """

    name: str
    type: Type
    kind: ParamKind
    has_default: bool


# ---------------------------------------------------------------------------
# FunctionSignature — full declared signature of a top-level def
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FunctionSignature:
    """Full declared signature of a top-level ``def``.

    Carries named/default/kind information needed for declared-name call sites.
    The value type (FunctionType) erases names/defaults/kinds.

    ``params``      — ordered list of ``ParamSpec`` (name, type, kind, has_default).
    ``result``      — the declared return type.
    ``type_params`` — tuple of type-parameter names for generic functions
                      (empty for non-generic functions).
    """

    params: tuple[ParamSpec, ...]
    result: Type
    type_params: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GenericTypeDef:
    """Template for a generic record or enum definition.

    ``kind``        — ``"record"`` or ``"enum"``.
    ``type_params`` — ordered tuple of type-parameter names.
    ``template``    — a ``RecordType`` or ``EnumType`` whose fields/variants
                      may contain ``TypeVarType`` nodes.
    """

    kind: str  # "record" | "enum"
    type_params: tuple[str, ...]
    template: RecordType | EnumType


@dataclass(frozen=True, slots=True)
class GenericAliasDef:
    """Resolved template for a parameterized type alias.

    ``type_params`` — ordered tuple of type-parameter names.
    ``template``    — alias body resolved in the alias-defining module with its
                      parameters represented as ``TypeVarType`` nodes.
    """

    type_params: tuple[str, ...]
    template: Type


@dataclass(frozen=True, slots=True)
class ConstructorSignature:
    """Signature for a record constructor or enum variant constructor.

    ``owner_name``      — name of the owning record or enum type.
    ``variant``         — variant name for enum constructors; ``None`` for
                          record constructors.
    ``field_names``     — ordered field names accepted by the constructor.
    ``field_templates`` — field types (may contain ``TypeVarType`` nodes for
                          generic types).
    ``result_template`` — the return type template (may contain TypeVarType).
    ``type_params``     — type-parameter names for instantiation.
    """

    owner_name: str
    variant: str | None
    field_names: tuple[str, ...]
    field_templates: tuple[Type, ...]
    result_template: Type
    # No default: a constructor always belongs to a concrete generic type whose
    # type_params are known at registration time; () would silently mask a bug.
    type_params: tuple[str, ...]


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
    ``--dry-run`` inventory is derived from the checker's work
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
        The codec selected for this call (e.g. ``"text"`` or ``"json"``).
        When ``structured_exec`` is ``True`` this field holds the placeholder
        value ``"text"`` and is **unused** —  will branch on
        ``structured_exec`` to skip codec lookup and return the raw ``ExecResult``
        handle instead.
    ``strict_json``
        The effective strict-JSON flag for this call (``None`` means the
        codec is not JSON-based and the flag is irrelevant; this is
        always ``None`` since the only codec is ``"text"``).
    ``structured_exec``
        ``True`` for the structured ``exec`` form (target is ``ExecResult``):
        returns the raw result record, does not parse stdout, does not raise
        on nonzero exit.  ``False`` (the default) for all other calls.  
        must branch on this flag to skip the codec/parse pipeline entirely.
    """

    target_type: Type
    codec_name: str
    strict_json: bool | None
    structured_exec: bool = False


# ---------------------------------------------------------------------------
# ArgumentBindings — checker-computed call/pattern argument bindings
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArgumentBindings:
    """Checker-computed argument bindings for call-like constructs, keyed by node_id.

    The checker is the single source of truth for how each construct's
    positional, named, and bare-name-shorthand arguments map onto declared
    parameters or fields.  The lowerer reads these instead of re-running the
    binder.

    ``function_calls``
        Direct user-function ``Call.node_id`` → declaration-order argument tuple
        (one entry per parameter; ``None`` means "use the parameter's default").
    ``constructor_calls``
        Record/enum/exception constructor ``Call.node_id`` → ordered
        ``{field_name: expr}`` mapping (every field bound; constructors have no
        defaults).
    ``constructor_patterns``
        Constructor-pattern ``Pattern.node_id`` → ordered ``(field_name,
        sub_pattern)`` pairs (partial patterns omit unmentioned fields).
    """

    function_calls: dict[int, tuple[Expr | None, ...]]
    constructor_calls: dict[int, dict[str, Expr]]
    constructor_patterns: dict[int, tuple[tuple[str, Pattern], ...]]


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
        order — captured by the checker.  The ``--dry-run`` inventory is
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
    argument_bindings: ArgumentBindings


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

    Graph mode
    ---------------
    When ``graph_type_table``, ``import_env``, and ``module_id`` are supplied,
    the environment becomes module-aware:

    - ``graph_type_table`` maps ``(ModuleId, name)`` to the fully-built
      ``Type`` objects stamped with their owning ``module_id``.  Built once by
      the graph pre-pass; shared (read-only) across all per-module envs.
    - ``graph_generic_table`` and ``graph_alias_table`` carry cross-module
      templates for applied nominal types and parameterized aliases.
    - ``import_env`` is the per-module :class:`~agm.agl.scope.imports.ImportEnv`
      produced by graph scope resolution. Used to resolve qualified and open-imported type names.
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
        graph_generic_table: Mapping[tuple[ModuleId, str], GenericTypeDef] | None = None,
        graph_alias_table: Mapping[tuple[ModuleId, str], GenericAliasDef] | None = None,
        graph_ctor_sig_table: Mapping[
            tuple[ModuleId, str, str | None], ConstructorSignature
        ] | None = None,
        graph_ctor_field_kinds_table: Mapping[
            tuple[ModuleId, str, str | None], tuple[tuple[str, ParamKind], ...]
        ] | None = None,
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
        # Generic type definitions — name → GenericTypeDef.
        self._generic_types: dict[str, GenericTypeDef] = {}
        # Constructor signatures — (owner_name, variant | None) → ConstructorSignature.
        self._constructor_sigs: dict[tuple[str, str | None], ConstructorSignature] = {}
        # Alias type-params — name → tuple of type-param names.
        self._alias_type_params: dict[str, tuple[str, ...]] = {}
        # Node-id-keyed function signatures — decl_node_id → FunctionSignature.
        # Populated in graph mode by the whole-graph function-signature pre-pass so
        # that _check_declared_name_call can look up the CORRECT signature for a
        # cross-module callee by its globally-unique decl_node_id rather than by
        # bare name (which would pick the wrong signature when two modules define
        # functions with the same name but different signatures).
        self._function_signatures_by_node_id: dict[int, FunctionSignature] = {}
        # Constructor field-kinds registry — (owner_name, variant | None) → ordered
        # (field_name, ParamKind) pairs.  Populated by _TypeBuilder for every
        # record/enum/exception; consumed by the checker and lowerer to build
        # bind_arguments param lists without round-tripping through the AST.
        self._constructor_field_kinds: dict[
            tuple[str, str | None], tuple[tuple[str, ParamKind], ...]
        ] = {}
        # Cross-module constructor field-kinds table: (ModuleId, owner_name, variant) → kinds.
        self._graph_ctor_field_kinds_table: Mapping[
            tuple[ModuleId, str, str | None], tuple[tuple[str, ParamKind], ...]
        ] | None = graph_ctor_field_kinds_table
        # Graph-mode context: None in single-program path.
        self._graph_type_table: Mapping[tuple[ModuleId, str], Type] | None = graph_type_table
        # Cross-module generic type definitions: (ModuleId, name) → GenericTypeDef.
        # Populated by the graph type pre-pass for qualified generic constructor calls.
        self._graph_generic_table: Mapping[
            tuple[ModuleId, str], GenericTypeDef
        ] | None = graph_generic_table
        # Cross-module parameterized type aliases: (ModuleId, name) → GenericAliasDef.
        self._graph_alias_table: Mapping[tuple[ModuleId, str], GenericAliasDef] = (
            graph_alias_table if graph_alias_table is not None else {}
        )
        # Cross-module constructor signatures: (ModuleId, owner_name, variant) → sig.
        self._graph_ctor_sig_table: Mapping[
            tuple[ModuleId, str, str | None], ConstructorSignature
        ] | None = graph_ctor_sig_table
        self._import_env: ImportEnv | None = import_env
        self._module_id: ModuleId = module_id
        # Built-in exception types are always available.
        for exc_name, exc_type in BUILTIN_EXCEPTIONS.items():
            self._types[exc_name] = exc_type
        # Built-in prelude types (AgL: ExecResult, ParsePolicy) are always available.
        for prelude_name, prelude_type in BUILTIN_PRELUDE_TYPES.items():
            self._types[prelude_name] = prelude_type
            if isinstance(prelude_type, RecordType):
                self._constructor_field_kinds[(prelude_name, None)] = tuple(
                    (fname, ParamKind.NAMED_ONLY) for fname in prelude_type.fields
                )
                continue
            enum_type = cast(EnumType, prelude_type)
            for variant, fields in enum_type.variants.items():
                self._constructor_field_kinds[(prelude_name, variant)] = tuple(
                    (fname, ParamKind.NAMED_ONLY) for fname in fields
                )
        # Pre-register builtin exception field kinds (all non-trace_id fields, NAMED_ONLY).
        for exc_name, exc_type in BUILTIN_EXCEPTIONS.items():
            if not exc_type.abstract:
                _kinds: tuple[tuple[str, ParamKind], ...] = tuple(
                    (fname, ParamKind.NAMED_ONLY)
                    for fname in exc_type.fields
                    if fname != "trace_id"
                )
                self._constructor_field_kinds[(exc_name, None)] = _kinds

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

    def register_alias(
        self, name: str, target_expr: object, *, type_params: tuple[str, ...] = ()
    ) -> None:
        """Store the raw TypeExpr for *name*; resolved lazily by resolve_type_expr.

        ``type_params`` must be provided for parameterized type aliases (e.g.
        ``type Wrapper[T] = list[T]``); defaults to ``()`` for plain aliases.
        """
        self._alias_targets[name] = target_expr
        self._alias_type_params[name] = type_params

    def get_alias_target_expr(self, name: str) -> object | None:
        return self._alias_targets.get(name)

    def get_alias_type_params(self, name: str) -> tuple[str, ...]:
        """Return the type-parameter names for a parameterized alias, or ``()``."""
        return self._alias_type_params.get(name, ())

    # --- Generic type registry ---

    def register_generic_type(self, name: str, gdef: GenericTypeDef) -> None:
        """Register a generic type definition under *name*."""
        self._generic_types[name] = gdef

    def get_generic_type(self, name: str) -> GenericTypeDef | None:
        """Return the ``GenericTypeDef`` for *name*, or ``None`` if unknown."""
        return self._generic_types.get(name)

    def instantiate_nominal(
        self,
        name: str,
        args: tuple[Type, ...],
        span: SourceSpan | None = None,
    ) -> RecordType | EnumType:
        """Instantiate a generic type named *name* with *args*.

        Substitutes the type parameters in the template with *args* and
        returns a concrete ``RecordType`` or ``EnumType`` with ``type_args``
        set to the supplied arguments.

        Raises ``AglTypeError`` for unknown names or arity mismatches.
        The optional *span* is forwarded to the error for source-location reporting.
        """
        gdef = self._generic_types.get(name)
        if gdef is None:
            raise AglTypeError(f"Unknown generic type '{name}'.", span=span)
        return self.instantiate_from_gdef(name, gdef, args, span=span)

    def instantiate_from_gdef(
        self,
        name: str,
        gdef: GenericTypeDef,
        args: tuple[Type, ...],
        span: SourceSpan | None = None,
    ) -> RecordType | EnumType:
        """Instantiate a :class:`GenericTypeDef` with *args*, substituting type parameters.

        Unlike :meth:`instantiate_nominal`, this method accepts a pre-looked-up
        ``GenericTypeDef`` directly, so callers that already have the definition
        (e.g. cross-module generic constructor checks) do not need to register it
        in the own-module ``_generic_types`` table.
        The optional *span* is forwarded to any ``AglTypeError`` raised.
        """
        from agm.agl.semantics.types import substitute as _subst

        if len(args) != len(gdef.type_params):
            raise AglTypeError(
                f"Type '{name}' requires {len(gdef.type_params)} type argument(s), "
                f"got {len(args)}.",
                span=span,
            )
        subst = dict(zip(gdef.type_params, args))
        template = gdef.template
        if isinstance(template, RecordType):
            new_fields = {k: _subst(v, subst) for k, v in template.fields.items()}
            return RecordType(
                name=name, fields=new_fields, type_args=args, module_id=template.module_id
            )
        # EnumType: substitute into each variant's field types.
        new_variants = {
            vname: {k: _subst(v, subst) for k, v in vfields.items()}
            for vname, vfields in template.variants.items()
        }
        return EnumType(
            name=name, variants=new_variants, type_args=args, module_id=template.module_id
        )

    def instantiate_alias(
        self,
        name: str,
        alias_def: GenericAliasDef,
        args: tuple[Type, ...],
        span: SourceSpan | None = None,
    ) -> Type:
        """Instantiate a resolved parameterized type alias template."""
        from agm.agl.semantics.types import substitute as _subst

        if len(args) != len(alias_def.type_params):
            raise AglTypeError(
                f"Alias '{name}' requires {len(alias_def.type_params)} type argument(s), "
                f"got {len(args)}.",
                span=span,
            )
        return _subst(alias_def.template, dict(zip(alias_def.type_params, args)))

    # --- Constructor signature registry ---

    def register_constructor_signature(self, sig: ConstructorSignature) -> None:
        """Register a constructor signature for a record or enum variant."""
        self._constructor_sigs[(sig.owner_name, sig.variant)] = sig

    def get_constructor_signature(
        self, owner_name: str, variant: str | None
    ) -> ConstructorSignature | None:
        """Return the constructor signature for *owner_name* / *variant*, or ``None``."""
        return self._constructor_sigs.get((owner_name, variant))

    def resolve_named_type(self, name: str) -> Type | None:
        """Resolve a type *name* alias-transparently to a semantic ``Type``.

        Returns the resolved ``Type`` for a record/enum/exception name or an
        alias chain (multi-hop, alias-of-alias) that bottoms out in a named
        type; ``None`` if the name is unknown or names a non-nominal alias
        target (e.g. an alias of ``list[int]``, which has no single name).
        Used for alias-transparent qualifier resolution in qualified
        constructors and ``is`` tests.

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
        is globally unique, signatures from different modules never
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
        type_vars: frozenset[str] = frozenset(),
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
        type_vars:
            Set of names that are in scope as rigid type variables.  A
            ``NameT`` whose name is in this set resolves to a ``TypeVarType``
            instead of being looked up in the type namespace.
        """
        from agm.agl.syntax.types import (
            AgentT,
            AppliedT,
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
                self.resolve_type_expr(p, _resolving=_resolving, type_vars=type_vars)
                for p in type_expr.params
            )
            result = self.resolve_type_expr(
                type_expr.result, _resolving=_resolving, type_vars=type_vars
            )
            return FunctionType(params=params, result=result)
        if isinstance(type_expr, ListT):
            elem = self.resolve_type_expr(
                type_expr.elem, _resolving=_resolving, type_vars=type_vars
            )
            return ListType(elem=elem)
        if isinstance(type_expr, DictT):
            val = self.resolve_type_expr(
                type_expr.value, _resolving=_resolving, type_vars=type_vars
            )
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
                type_vars=type_vars,
            )
        if isinstance(type_expr, AppliedT):
            name = type_expr.name
            eff_span = span if span is not None else type_expr.span
            resolved_args = tuple(
                self.resolve_type_expr(
                    a, span=None, _resolving=_resolving, type_vars=type_vars
                )
                for a in type_expr.args
            )
            qualifier = type_expr.module_qualifier
            if qualifier is not None and qualifier.segments:
                return self._resolve_qualified_applied_type(
                    qualifier, name, resolved_args, span=eff_span
                )
            gdef = self._generic_types.get(name)
            if gdef is not None:
                return self.instantiate_nominal(name, resolved_args, span=eff_span)
            alias_expr = self._alias_targets.get(name)
            if alias_expr is not None:
                if name in _resolving:
                    raise AglTypeError(
                        f"Type alias '{name}' is part of a cycle.",
                        span=eff_span,
                    )
                alias_params = self._alias_type_params.get(name, ())
                if len(resolved_args) != len(alias_params):
                    raise AglTypeError(
                        f"Alias '{name}' requires {len(alias_params)} type argument(s), "
                        f"got {len(resolved_args)}.",
                        span=eff_span,
                    )
                body_type = self.resolve_type_expr(
                    alias_expr,
                    span=span,
                    _resolving=_resolving | {name},
                    type_vars=type_vars | frozenset(alias_params),
                )
                return self.instantiate_alias(
                    name,
                    GenericAliasDef(type_params=alias_params, template=body_type),
                    resolved_args,
                    span=eff_span,
                )
            if name in self._types:
                raise AglTypeError(
                    f"Type '{name}' does not take type arguments.",
                    span=eff_span,
                )
            if qualifier is None:
                imported = self._resolve_open_imported_applied_type(
                    name, resolved_args, span=eff_span
                )
                if imported is not None:
                    return imported
            raise AglTypeError(
                f"Unknown type '{name}'.",
                span=eff_span,
            )
        raise AglTypeError(
            f"Unknown type expression: {type_expr!r}",
            span=span,
        )

    def _resolve_open_imported_applied_type(
        self,
        name: str,
        args: tuple[Type, ...],
        *,
        span: SourceSpan | None,
    ) -> Type | None:
        """Resolve an unqualified generic application through open imports."""
        if self._import_env is None or self._graph_generic_table is None:
            return None
        candidates = self._import_env.unqualified.get(name, frozenset())
        type_candidates = [
            qname
            for qname in candidates
            if (qname[0], qname[1]) in self._graph_generic_table
            or (qname[0], qname[1]) in self._graph_alias_table
            or (
                self._graph_type_table is not None
                and (qname[0], qname[1]) in self._graph_type_table
            )
        ]
        if len(type_candidates) > 1:
            labels = sorted(f"{qname[0].dotted()}::{qname[1]}" for qname in type_candidates)
            raise AglTypeError(
                f"Ambiguous type '{name}': it is exported by multiple modules "
                f"({', '.join(labels)}). Use a qualified reference to disambiguate.",
                span=span,
            )
        if not type_candidates:
            return None
        module_id, source_name = type_candidates[0]
        gdef = self._graph_generic_table.get((module_id, source_name))
        if gdef is not None:
            return self.instantiate_from_gdef(source_name, gdef, args, span=span)
        alias_def = self._graph_alias_table.get((module_id, source_name))
        if alias_def is not None:
            return self.instantiate_alias(source_name, alias_def, args, span=span)
        raise AglTypeError(
            f"Type '{name}' does not take type arguments.",
            span=span,
        )

    def _resolve_qualified_applied_type(
        self,
        qualifier: object,
        name: str,
        args: tuple[Type, ...],
        *,
        span: SourceSpan | None,
    ) -> Type:
        """Resolve ``module::Name[args]`` through the module import environment."""
        from agm.agl.syntax.types import Qualifier

        assert isinstance(qualifier, Qualifier)
        if self._import_env is None or self._graph_generic_table is None:
            raise AglTypeError(
                f"Module qualifier '{'.'.join(qualifier.segments)}::' cannot be resolved "
                "outside of a module graph.",
                span=span,
            )
        handle_map = self._import_env.qualified.get(qualifier.segments)
        if handle_map is None:
            raise AglTypeError(
                f"Unknown module qualifier '{'.'.join(qualifier.segments)}::'. "
                "The module has not been imported or the qualifier is not in scope.",
                span=span,
            )
        qname = handle_map.get(name)
        if qname is None:
            raise AglTypeError(
                f"Type '{name}' is not accessible via qualifier "
                f"'{'.'.join(qualifier.segments)}::'.",
                span=span,
            )
        gdef = self._graph_generic_table.get((qname[0], qname[1]))
        if gdef is not None:
            return self.instantiate_from_gdef(qname[1], gdef, args, span=span)
        alias_def = self._graph_alias_table.get((qname[0], qname[1]))
        if alias_def is not None:
            return self.instantiate_alias(qname[1], alias_def, args, span=span)
        if self._graph_type_table is not None and (qname[0], qname[1]) in self._graph_type_table:
            raise AglTypeError(
                f"Type '{'.'.join(qualifier.segments)}::{name}' does not take type arguments.",
                span=span,
            )
        raise AglTypeError(
            f"'{'.'.join(qualifier.segments)}::{name}' does not name a type.",
            span=span,
        )

    def _resolve_name_type(
        self,
        name: str,
        *,
        span: SourceSpan | None,
        _resolving: frozenset[str],
        type_vars: frozenset[str] = frozenset(),
    ) -> Type:
        # Type variables take priority over the type namespace.
        if name in type_vars:
            return TypeVarType(name)
        # Reject a bare reference to a generic type that requires type arguments.
        gdef = self._generic_types.get(name)
        if gdef is not None and len(gdef.type_params) > 0:
            raise AglTypeError(
                f"Generic type '{name}' requires {len(gdef.type_params)} type argument(s); "
                f"use '{name}[...]' to apply it.",
                span=span,
            )
        # Reject a bare reference to a parameterized alias.
        alias_params = self._alias_type_params.get(name, ())
        if name in self._alias_targets and len(alias_params) > 0:
            raise AglTypeError(
                f"Parameterized alias '{name}' requires {len(alias_params)} type argument(s); "
                f"use '{name}[...]' to apply it.",
                span=span,
            )
        # Check alias table (aliases are raw TypeExpr, resolved on demand).
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
                type_vars=type_vars,
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
        """Return the full declared type-name set for type-namespace enumeration.

        Combines registered nominal types (records, enums, exceptions, and
        built-ins) with registered alias names.  Retained for generic type
        resolution in future work, where the complete type-namespace must
        be enumerable to validate applied-type names (e.g. ``Pair[int, text]``).
        """
        return frozenset(self._types) | frozenset(self._alias_targets)

    def resolve_type_by_module_id(
        self, module_id: ModuleId, name: str
    ) -> Type | None:
        """Directly look up a type by owning module and name in the graph type table.

        Used for cross-module constructor references when the owning module is
        already known from scope resolution (e.g. ``mylib::Color.Red``).

        Returns ``None`` in single-program mode or if not found.
        """
        if self._graph_type_table is None:
            return None
        return self._graph_type_table.get((module_id, name))

    def get_generic_type_from_module(
        self, module_id: ModuleId, name: str
    ) -> GenericTypeDef | None:
        """Look up a cross-module ``GenericTypeDef`` by owning module and name.

        Used to detect and instantiate generic constructors referenced via a
        module qualifier (e.g. ``lib::Box[int](value: 1)``).  Returns ``None``
        in single-program mode or when the type is not generic.
        """
        if self._graph_generic_table is None:
            return None
        return self._graph_generic_table.get((module_id, name))

    def get_open_imported_generic_type(
        self, exposed_name: str
    ) -> tuple[ModuleId, str, GenericTypeDef] | None:
        """Return the unique generic type exposed by an open-imported name."""
        matches = self._open_imported_generic_type_matches(exposed_name)
        if len(matches) == 1:
            return matches[0]
        return None

    def resolve_unapplied_generic_type(
        self,
        name: str,
        *,
        span: SourceSpan | None = None,
    ) -> tuple[str, GenericTypeDef] | None:
        """Resolve a bare generic type name without applying type arguments.

        This serves REPL type-definition display only. Normal type-expression
        resolution still rejects unapplied generics because they are not concrete
        value-level types.
        """
        gdef = self._generic_types.get(name)
        if gdef is not None:
            return name, gdef
        matches = self._open_imported_generic_type_matches(name)
        if len(matches) > 1:
            labels = sorted(
                f"{module_id.dotted()}::{source_name}"
                for module_id, source_name, _ in matches
            )
            raise AglTypeError(
                f"Ambiguous generic type '{name}': it is exported by multiple modules "
                f"({', '.join(labels)}). Use a qualified reference to disambiguate.",
                span=span,
            )
        if len(matches) == 1:
            return name, matches[0][2]
        return None

    def resolve_qualified_unapplied_generic_type(
        self,
        qualifier: object,
        name: str,
        *,
        span: SourceSpan | None = None,
    ) -> tuple[str, GenericTypeDef] | None:
        """Resolve a module-qualified generic type name without applying arguments."""
        from agm.agl.syntax.types import Qualifier

        assert isinstance(qualifier, Qualifier)
        if not qualifier.segments:
            gdef = self._generic_types.get(name)
            if gdef is not None:
                return name, gdef
            if self._graph_generic_table is None:
                return None
            gdef = self._graph_generic_table.get((self._module_id, name))
            return (name, gdef) if gdef is not None else None
        if self._import_env is None or self._graph_generic_table is None:
            return None
        handle_map = self._import_env.qualified.get(qualifier.segments)
        if handle_map is None:
            return None
        qname = handle_map.get(name)
        if qname is None:
            return None
        gdef = self._graph_generic_table.get((qname[0], qname[1]))
        if gdef is None:
            return None
        qualified_name = f"{'.'.join(qualifier.segments)}::{name}"
        return qualified_name, gdef

    def _open_imported_generic_type_matches(
        self, exposed_name: str
    ) -> list[tuple[ModuleId, str, GenericTypeDef]]:
        if self._import_env is None or self._graph_generic_table is None:
            return []
        matches: list[tuple[ModuleId, str, GenericTypeDef]] = []
        for module_id, source_name in self._import_env.unqualified.get(
            exposed_name, frozenset()
        ):
            gdef = self._graph_generic_table.get((module_id, source_name))
            if gdef is not None:
                matches.append((module_id, source_name, gdef))
        return matches

    def get_open_imported_type(self, exposed_name: str) -> tuple[ModuleId, str, Type] | None:
        """Return the unique concrete type exposed by an open-imported name."""
        if self._import_env is None or self._graph_type_table is None:
            return None
        matches = []
        for module_id, source_name in self._import_env.unqualified.get(
            exposed_name, frozenset()
        ):
            typ = self._graph_type_table.get((module_id, source_name))
            if typ is not None:
                matches.append((module_id, source_name, typ))
        if len(matches) == 1:
            return matches[0]
        return None

    def get_ctor_sig_from_module(
        self, module_id: ModuleId, owner_name: str, variant: str | None
    ) -> ConstructorSignature | None:
        """Look up a cross-module ``ConstructorSignature`` by owning module, name, and variant.

        Companion to :meth:`get_generic_type_from_module` for checking cross-module
        generic constructor calls.  Returns ``None`` in single-program mode or when
        not found.
        """
        if self._graph_ctor_sig_table is None:
            return None
        return self._graph_ctor_sig_table.get((module_id, owner_name, variant))

    def register_constructor_field_kinds(
        self,
        owner_name: str,
        variant: str | None,
        fields: tuple[tuple[str, ParamKind], ...],
    ) -> None:
        """Register ordered (field_name, ParamKind) pairs for a constructor."""
        self._constructor_field_kinds[(owner_name, variant)] = fields

    def get_constructor_field_kinds(
        self,
        owner_name: str,
        variant: str | None,
        *,
        module_id: ModuleId | None = None,
    ) -> tuple[tuple[str, ParamKind], ...] | None:
        """Return the ordered field-kind pairs for a constructor, or ``None`` if unknown.

        First checks the own-module registry; falls back to the cross-module graph table
        when ``module_id`` is provided (used for cross-module constructor calls where the
        owner type carries its declaring module's ``module_id``).
        """
        result = self._constructor_field_kinds.get((owner_name, variant))
        if result is not None:
            return result
        if module_id is not None and self._graph_ctor_field_kinds_table is not None:
            return self._graph_ctor_field_kinds_table.get((module_id, owner_name, variant))
        return None

    def get_constructor_field_kinds_for_type(
        self,
        typ: Type | None,
        owner_name: str,
        variant: str | None,
        *,
        module_id: ModuleId | None = None,
    ) -> tuple[tuple[str, ParamKind], ...] | None:
        """Return field-kinds for a constructor identified by its resolved owner *typ*.

        Encapsulates the registry-key convention in one place. ``RecordType`` and
        ``EnumType`` carry their own ``module_id``. ``ExceptionType`` does not, so
        callers that resolved a cross-module exception must provide its defining
        ``module_id``; built-in and same-module exceptions still resolve through
        the local registry before any graph lookup. Enum variants are keyed by
        *variant*; records and exceptions are keyed under ``variant=None``.
        """
        if isinstance(typ, EnumType):
            lookup_module_id: ModuleId = typ.module_id
            lookup_variant = variant
        elif isinstance(typ, RecordType):
            lookup_module_id = typ.module_id
            lookup_variant = None
        elif isinstance(typ, ExceptionType) and module_id is not None:
            lookup_module_id = module_id
            lookup_variant = None
        else:
            # ExceptionType without a caller-provided module_id (same-module or
            # built-in exception) or an unexpected typ value.  Both same-module
            # exceptions and built-in prelude exceptions are pre-registered in the
            # LOCAL registry, so get_constructor_field_kinds() returns at the first
            # check and never reaches the graph-table lookup.  PRELUDE_ID is used
            # here only to satisfy the type; the graph lookup is always a no-op for
            # this branch.
            lookup_module_id = PRELUDE_ID
            lookup_variant = None
        return self.get_constructor_field_kinds(
            owner_name, lookup_variant, module_id=lookup_module_id
        )

    def all_constructor_field_kinds(
        self,
    ) -> list[tuple[tuple[str, str | None], tuple[tuple[str, ParamKind], ...]]]:
        """Return all own-module constructor field-kind entries as (key, kinds) pairs."""
        return list(self._constructor_field_kinds.items())

    def all_generic_types(self) -> dict[str, GenericTypeDef]:
        """Return the own-module generic type map (name → GenericTypeDef)."""
        return self._generic_types

    def all_constructor_sigs(self) -> list[tuple[tuple[str, str | None], ConstructorSignature]]:
        """Return all own-module constructor signatures as (key, sig) pairs."""
        return list(self._constructor_sigs.items())

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
        self._generic_types.update(other._generic_types)
        self._constructor_sigs.update(other._constructor_sigs)
        self._constructor_field_kinds.update(other._constructor_field_kinds)
        self._alias_type_params.update(other._alias_type_params)
        self._function_signatures_by_node_id.update(other._function_signatures_by_node_id)
