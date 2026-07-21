"""Type environment and output types for the AgL type checker.

``TypeEnvironment`` holds:
- The user-declared type namespace (records, enums, aliases, exceptions).
- The variable → Type binding table derived from the scope side tables.

``CheckedModule`` is the frozen output of the type-checking pass.
``OutputContractSpec`` records the statically derived codec + target type per
``AgentCall`` node.
``AglTypeError`` is the fatal type error raised by the checker.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

from agm.agl.diagnostics import AglError, Diagnostic
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.scope.imports import (
    ImportEnv,
    QName,
    QualResolutionFound,
    contribution_routes,
    qualifier_contributes,
    resolve_qualified,
    resolve_qualified_member,
    try_resolve_qualified_member,
)
from agm.agl.scope.symbols import BindingRef, ConstructorRef, ModuleResolution
from agm.agl.semantics.persistent import PersistentDict
from agm.agl.semantics.type_table import (
    BUILTIN_PRELUDE_TYPE_DEFS,
    TypeTable,
    create_seeded_type_table,
)
from agm.agl.semantics.types import (
    BUILTIN_EXCEPTIONS,
    BUILTIN_PRELUDE_TYPE_NAMES,
    BUILTIN_PRELUDE_TYPES,
    AgentType,
    BoolType,
    CastSpec,
    DecimalType,
    DictType,
    EnumOwnerForm,
    EnumOwnerFormKind,
    EnumType,
    ExceptionType,
    FunctionType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
    TypeTemplate,
    TypeTemplateMatch,
    TypeVarType,
    UnitType,
    contains_inference_var,
)
from agm.agl.syntax.nodes import Expr, ParamKind, Pattern
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import Qualifier

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
    ``template``    — a bare ``RecordType``/``EnumType`` handle whose
                      ``type_args`` are ``TypeVarType`` nodes for each of
                      ``type_params``; field/variant shapes are looked up by
                      handle in the shared ``TypeTable``.
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
        Output-discarding unit calls use ``"none"``. When ``structured_exec``
        is ``True`` this field holds the placeholder
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
    ``function_param_types``
        Direct user-function ``Call.node_id`` → concrete declaration-order parameter
        types after generic type-argument substitution.  The lowerer uses these
        types to insert call-site coercions without re-inferring generic arguments.
    ``constructor_calls``
        Record/enum/exception constructor ``Call.node_id`` → ordered
        ``{field_name: expr}`` mapping (every field bound; constructors have no
        defaults).
    ``constructor_patterns``
        Constructor-pattern ``Pattern.node_id`` → ordered ``(field_name,
        sub_pattern)`` pairs (partial patterns omit unmentioned fields).
    """

    function_calls: dict[int, tuple[Expr | None, ...]]
    function_param_types: dict[int, tuple[Type, ...]]
    constructor_calls: dict[int, dict[str, Expr]]
    constructor_patterns: dict[int, tuple[tuple[str, Pattern], ...]]


@dataclass(frozen=True, slots=True)
class PartialCallSpec:
    """Checker-computed routing metadata for a call that produces a function.

    ``callee_kind`` identifies which lowering path the underlying call uses.
    ``argument_holes`` is ordered like the checked call binding for that callee;
    each item is the produced-function parameter index for a placeholder slot,
    or ``None`` for a supplied non-placeholder argument or a defaulted slot.
    """

    argument_holes: tuple[int | None, ...]
    callee_kind: Literal["declared", "constructor", "value"] = "declared"


# ---------------------------------------------------------------------------
# Pattern-slot dereferencing — shared by the checker and its checked output
# ---------------------------------------------------------------------------


def dereference_slot_binding(
    node_id: int,
    *,
    resolution: Mapping[int, BindingRef],
    slot_resolution: Mapping[int, BindingRef],
) -> BindingRef | None:
    """Return *node_id*'s binding, following a pattern slot to its selection.

    Scope-created references to a field-directed pattern slot carry no final
    meaning of their own; the checker's ``slot_resolution`` supplies it.  The
    checker uses this while it builds the table and ``CheckedModule`` uses it
    afterwards, so both agree on what a slot reference denotes.
    """
    raw_binding = resolution.get(node_id)
    if raw_binding is None or raw_binding.slot_id is None:
        return raw_binding
    return slot_resolution.get(raw_binding.slot_id)


def dereference_slot_constructor_ref(
    node_id: int,
    *,
    resolution: Mapping[int, BindingRef],
    constructor_refs: Mapping[int, ConstructorRef],
    slot_constructor_refs: Mapping[int, ConstructorRef],
) -> ConstructorRef | None:
    """Return *node_id*'s constructor reference, following a pattern slot."""
    raw_binding = resolution.get(node_id)
    if raw_binding is None or raw_binding.slot_id is None:
        return constructor_refs.get(node_id)
    return slot_constructor_refs.get(raw_binding.slot_id)


# ---------------------------------------------------------------------------
# CheckedModule — output of the type-checking pass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckedModule:
    """Output of the type-checking pass.

    ``resolved``
        The original ``ModuleResolution`` carrying the ``Program`` and scope
        side tables. Field-directed branch references remain immutable,
        scope-created pattern-slot ``BindingRef`` instances; checker selections
        below provide their final meanings without changing scope output.
    ``pattern_classifications``
        Final field-directed classification of bare patterns: ``Pattern.node_id``
        → ``None`` for a binder, or the ``ConstructorRef`` it matches. This is
        the authoritative source for match normalization and lowering.
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
    ``slot_resolution`` / ``slot_constructor_refs``
        Checker-owned, fully dereferenced pattern-slot selections.
        ``binding_for`` and ``constructor_ref_for`` apply them to scope-created
        slot references; consumers must use these accessors for references that
        may be slots.
    """

    resolved: ModuleResolution
    node_types: dict[int, Type]
    contract_specs: dict[int, OutputContractSpec]
    call_sites: tuple[CallSiteRecord, ...]
    warnings: tuple[Diagnostic, ...]
    type_env: TypeEnvironment
    function_signatures: dict[str, FunctionSignature]
    cast_specs: dict[int, CastSpec]
    argument_bindings: ArgumentBindings
    pattern_classifications: dict[int, ConstructorRef | None]
    partial_calls: dict[int, PartialCallSpec]
    module_id: ModuleId = ENTRY_ID
    import_env: ImportEnv = field(default_factory=lambda: ImportEnv({}, {}))
    source_text: str = ""
    slot_resolution: dict[int, BindingRef] = field(default_factory=dict)
    slot_constructor_refs: dict[int, ConstructorRef] = field(default_factory=dict)

    def binding_for(self, node_id: int) -> BindingRef | None:
        """Return *node_id*'s checked binding, dereferencing a pattern slot."""
        return dereference_slot_binding(
            node_id,
            resolution=self.resolved.resolution,
            slot_resolution=self.slot_resolution,
        )

    def constructor_ref_for(self, node_id: int) -> ConstructorRef | None:
        """Return *node_id*'s checked constructor reference through a slot."""
        return dereference_slot_constructor_ref(
            node_id,
            resolution=self.resolved.resolution,
            constructor_refs=self.resolved.constructor_refs,
            slot_constructor_refs=self.slot_constructor_refs,
        )


def _assert_checked_types_closed(types: Iterable[Type], *, owner: str) -> None:
    """Reject solver-local types that escape a checked-output boundary."""
    if any(contains_inference_var(typ) for typ in types):
        raise AssertionError(f"inference variable leaked from checked output ({owner})")


def assert_checked_output_closed(
    *,
    node_types: Mapping[int, Type],
    contract_specs: Mapping[int, OutputContractSpec],
    call_sites: Iterable[CallSiteRecord],
    type_env: "TypeEnvironment",
    function_signatures: Mapping[str, FunctionSignature],
    cast_specs: Mapping[int, CastSpec],
    argument_bindings: ArgumentBindings,
    owner: str,
) -> None:
    """Assert that all type-bearing checked metadata is concrete or rigid.

    Partial-call routing is deliberately absent: it is syntax-and-binding metadata
    only; its function type is published in ``node_types``.  Rigid declaration
    variables remain valid in generic templates and signatures, while flexible
    ``InferenceVarType`` instances are a compiler invariant failure here.
    """
    _assert_checked_types_closed(
        (
            *node_types.values(),
            *(spec.target_type for spec in contract_specs.values()),
            *(site.target_type for site in call_sites),
            *(signature.result for signature in function_signatures.values()),
            *(
                param.type
                for signature in function_signatures.values()
                for param in signature.params
            ),
            *(spec.target_type for spec in cast_specs.values()),
            *(
                param_type
                for param_types in argument_bindings.function_param_types.values()
                for param_type in param_types
            ),
        ),
        owner=owner,
    )
    type_env.seal()


def assert_checked_module_closed(checked: CheckedModule) -> None:
    """Assert that a single-module checked program is safe to lower."""
    assert_checked_output_closed(
        node_types=checked.node_types,
        contract_specs=checked.contract_specs,
        call_sites=checked.call_sites,
        type_env=checked.type_env,
        function_signatures=checked.function_signatures,
        cast_specs=checked.cast_specs,
        argument_bindings=checked.argument_bindings,
        owner="checked program",
    )
    checked.type_env.assert_shared_tables_closed()


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

    Program context
    ---------------
    When ``program_type_table``, ``import_env``, and ``module_id`` are supplied,
    the environment becomes module-aware:

    - ``program_type_table`` maps ``(ModuleId, name)`` to the fully-built
      ``Type`` objects stamped with their owning ``module_id``.  Built once by
      the program pre-pass; shared (read-only) across all per-module envs.
    - ``program_generic_table`` and ``program_alias_table`` carry cross-module
      templates for applied nominal types and parameterized aliases; during
      program type-table construction, ``program_alias_keys`` and
      ``program_alias_resolver`` let transparent cross-module aliases resolve
      lazily before their sorted body-resolution turn.
    - ``import_env`` is the per-module :class:`~agm.agl.scope.imports.ImportEnv`
      produced by program scope resolution. Used to resolve qualified and open-imported type names.
    - ``module_id`` is the owning module of the current env.  ``::Name``
      (empty-segment qualifier) resolves against this module's own types.

    These fields are ``None`` in the module path; the resolution logic
    in ``resolve_type_expr`` and ``_resolve_name_type`` checks for ``None``
    before entering the module-aware branches, so the existing path is
    unchanged when these are absent.
    """

    def __init__(
        self,
        *,
        program_type_table: Mapping[tuple[ModuleId, str], Type] | None = None,
        program_generic_table: Mapping[tuple[ModuleId, str], GenericTypeDef] | None = None,
        program_alias_table: Mapping[tuple[ModuleId, str], GenericAliasDef] | None = None,
        program_alias_keys: frozenset[tuple[ModuleId, str]] | None = None,
        program_alias_resolver: Callable[[ModuleId, str, SourceSpan | None], Type | None]
        | None = None,
        program_ctor_sig_table: Mapping[tuple[ModuleId, str, str | None], ConstructorSignature]
        | None = None,
        program_ctor_field_kinds_table: Mapping[
            tuple[ModuleId, str, str | None], tuple[tuple[str, ParamKind], ...]
        ]
        | None = None,
        import_env: ImportEnv | None = None,
        private_info: Mapping[tuple[ModuleId, str], bool] | None = None,
        module_id: ModuleId = ENTRY_ID,
        type_table: TypeTable | None = None,
    ) -> None:
        # Shared nominal type-declaration table (dual-write target alongside
        # ``_types``): defaults to a fresh table seeded with built-in prelude
        # defs; program context passes one shared instance across per-module envs.
        self._type_table: TypeTable = (
            type_table if type_table is not None else create_seeded_type_table()
        )
        # user-declared types (records, enums) — name → Type
        self._types: dict[str, Type] = {}
        # alias targets — name → resolved Type (cycle detection uses seen set)
        self._alias_targets: dict[str, object] = {}  # stores raw TypeExpr until resolved
        # Binding node_id → Type (populated as declarations are checked).
        self._binding_types: PersistentDict[int, Type] = PersistentDict()
        # Function signatures — name → FunctionSignature (for declared-name calls).
        self._function_signatures: dict[str, FunctionSignature] = {}
        # Generic type definitions — name → GenericTypeDef.
        self._generic_types: dict[str, GenericTypeDef] = {}
        # Constructor signatures — (owner_name, variant | None) → ConstructorSignature.
        self._constructor_sigs: dict[tuple[str, str | None], ConstructorSignature] = {}
        # Alias type-params — name → tuple of type-param names.
        self._alias_type_params: dict[str, tuple[str, ...]] = {}
        # Node-id-keyed function signatures — decl_node_id → FunctionSignature.
        # Populated in program context by the whole-program function-signature pre-pass so
        # that _check_declared_name_call can look up the CORRECT signature for a
        # cross-module callee by its globally-unique decl_node_id rather than by
        # bare name (which would pick the wrong signature when two modules define
        # functions with the same name but different signatures).
        self._function_signatures_by_node_id: dict[int, FunctionSignature] = {}
        # Declaration node_ids of ``extern def``s, keyed by the same globally-unique
        # decl_node_id as ``_function_signatures_by_node_id``.  Populated by
        # ``_preregister_funcdef`` (module mode) and by the program
        # function-signature pre-pass seeding (program context, including imported
        # externs).  Consulted by ``_check_declared_name_call`` to decide whether a
        # declared-name call site is an extern call site to record.
        self._extern_node_ids: set[int] = set()
        # Constructor field-kinds registry — (owner_name, variant | None) → ordered
        # (field_name, ParamKind) pairs.  Populated by _TypeBuilder for every
        # record/enum/exception; consumed by the checker and lowerer to build
        # bind_arguments param lists without round-tripping through the AST.
        self._constructor_field_kinds: dict[
            tuple[str, str | None], tuple[tuple[str, ParamKind], ...]
        ] = {}
        # Cross-module constructor field-kinds table: (ModuleId, owner_name, variant) → kinds.
        self._program_ctor_field_kinds_table: (
            Mapping[tuple[ModuleId, str, str | None], tuple[tuple[str, ParamKind], ...]] | None
        ) = program_ctor_field_kinds_table
        # Program context: None in module path.
        self._program_type_table: Mapping[tuple[ModuleId, str], Type] | None = program_type_table
        # Cross-module generic type definitions: (ModuleId, name) → GenericTypeDef.
        # Populated by the program type pre-pass for qualified generic constructor calls.
        self._program_generic_table: Mapping[tuple[ModuleId, str], GenericTypeDef] | None = (
            program_generic_table
        )
        # Cross-module parameterized type aliases: (ModuleId, name) → GenericAliasDef.
        self._program_alias_table: Mapping[tuple[ModuleId, str], GenericAliasDef] = (
            program_alias_table if program_alias_table is not None else {}
        )
        self._program_alias_keys: frozenset[tuple[ModuleId, str]] = (
            program_alias_keys if program_alias_keys is not None else frozenset()
        )
        self._program_alias_resolver: (
            Callable[[ModuleId, str, SourceSpan | None], Type | None] | None
        ) = program_alias_resolver
        # Cross-module constructor signatures: (ModuleId, owner_name, variant) → sig.
        self._program_ctor_sig_table: (
            Mapping[tuple[ModuleId, str, str | None], ConstructorSignature] | None
        ) = program_ctor_sig_table
        self._import_env: ImportEnv | None = import_env
        # Scope collects this once from declarations; sharing it here preserves
        # the public/private distinction for qualified type diagnostics.
        self._private_info: Mapping[tuple[ModuleId, str], bool] = (
            private_info if private_info is not None else {}
        )
        self._module_id: ModuleId = module_id
        self._sealed = False
        # Memo for the own-type-name enumeration, which rebuilds a whole-namespace
        # answer and is asked for repeatedly (once per owner-form resolution).  It
        # is populated only once ``seal`` has frozen the declaration namespace, so
        # a still-mutating environment never serves a stale answer.
        self._sealed_own_source_type_names: frozenset[str] | None = None
        # Memos for the enum owner-form enumeration and its variant-level
        # counterpart.  Both rescan the whole type namespace (and, for imports,
        # every contribution route), and match compilation asks for them once
        # per case.  Like the memo above, they are populated only once ``seal``
        # has frozen the declaration namespace.
        self._sealed_enum_owner_forms: tuple[EnumOwnerForm, ...] | None = None
        self._sealed_blocked_enum_variants: Mapping[tuple[str, ...], frozenset[str]] | None = None
        # Built-in exception types are always available.
        for exc_name, exc_type in BUILTIN_EXCEPTIONS.items():
            self._types[exc_name] = exc_type
        # Built-in prelude types (AgL: ExecResult, ParsePolicy) are always
        # available.  Field/variant names for constructor-kind registration
        # come from the shared prelude ``TypeDef`` literals — the handles
        # themselves carry no shape data.
        for prelude_name, prelude_type in BUILTIN_PRELUDE_TYPES.items():
            self._types[prelude_name] = prelude_type
            typedef = BUILTIN_PRELUDE_TYPE_DEFS[prelude_name]
            if typedef.kind == "record":
                self._constructor_field_kinds[(prelude_name, None)] = tuple(
                    (fname, ParamKind.STANDARD) for fname, _ in typedef.fields
                )
                continue
            for variant, vfields in typedef.variants:
                self._constructor_field_kinds[(prelude_name, variant)] = tuple(
                    (fname, ParamKind.STANDARD) for fname, _ in vfields
                )
        # Exception constructor field kinds are NOT pre-registered here: each
        # exception's own fields honor their declared @pos/@std/@named marker
        # (stored on its TypeDef as ``field_kinds``, alongside ``fields``),
        # same as a record's fields.  ``get_constructor_field_kinds_for_type``
        # derives the full flattened (base-chain-inherited + own) kinds
        # directly from ``type_table.exception_field_kinds`` on demand instead
        # of a pre-registration step, since that requires no build ordering.

    @property
    def type_table(self) -> TypeTable:
        """The shared ``TypeTable`` populated alongside ``_types`` (dual-write)."""
        return self._type_table

    @property
    def is_sealed(self) -> bool:
        """Whether this environment has been validated and frozen for seeding."""
        return self._sealed

    def _assert_mutable(self) -> None:
        if self._sealed:
            raise AssertionError("cannot mutate a sealed type environment")

    def seal(self) -> None:
        """Validate this environment once and freeze it as checked output."""
        self.assert_closed()
        self._sealed = True

    # --- Type namespace queries ---

    def has_type(self, name: str) -> bool:
        return name in self._types

    def get_type(self, name: str) -> Type | None:
        return self._types.get(name)

    def has_qualified_import_member(self, qualifier: Qualifier, name: str) -> bool:
        """Return whether a qualifier route contributes *name* after filtering."""
        return self._import_env is not None and qualifier_contributes(
            self._import_env, qualifier.segments, name, anchored=qualifier.anchored
        )

    def _resolve_import_qname(
        self,
        qualifier: "Qualifier",
        name: str,
        *,
        span: SourceSpan | None,
        required: bool = True,
    ) -> QName | None:
        """Resolve a qualified member through the shared contribution resolver."""
        import_env = cast(ImportEnv, self._import_env)
        if not required:
            return try_resolve_qualified_member(import_env, qualifier, name)
        return resolve_qualified_member(
            import_env,
            qualifier,
            name,
            self._private_info,
            unknown_qualifier=lambda rendered: AglTypeError(
                f"Unknown module qualifier '{rendered}::'.", span=span
            ),
            private_member=lambda module: AglTypeError(
                f"Type '{name}' in module '{module.path_str()}' is declared private "
                "and cannot be accessed from outside the module.",
                span=span,
            ),
            missing_member=lambda rendered: AglTypeError(
                f"Type '{name}' is not accessible via qualifier '{rendered}::'.", span=span
            ),
            ambiguous=lambda message: AglTypeError(message, span=span),
        )

    def register_type(self, name: str, typ: Type) -> None:
        self._assert_mutable()
        self._types[name] = typ

    def unregister_name(self, name: str) -> None:
        """Remove a user *name* from the type, alias, and type-table namespaces.

        Used by the type-builder when an incremental-session entry redeclares a
        *seeded* name — either with a different kind (e.g. a seeded ``record R``
        redefined as ``type R = int``) or a different shape (e.g. ``record R``
        redefined with different fields).  Type handles, aliases, generic
        templates, constructor metadata, and alias parameter metadata live in
        separate tables, so a cross-kind redefinition would otherwise leave a
        stale entry in another table and make ``get_type`` disagree with
        annotation/constructor resolution.  Dropping the name from all
        namespaces before the new declaration is registered keeps them mutually
        consistent, and lets the type table's dual-write ``register`` calls
        treat every registration as a fresh one rather than a conflicting
        re-registration of the same key.

        Built-in exception names and built-in prelude type names are never
        removed: they are non-shadowable (rejected earlier by
        ``_BUILTIN_TYPE_NAMES``), so the builder never calls this for them,
        but the guard makes the helper safe to call defensively.
        """
        self._assert_mutable()
        if name in BUILTIN_EXCEPTIONS or name in BUILTIN_PRELUDE_TYPE_NAMES:
            return
        self._types.pop(name, None)
        self._alias_targets.pop(name, None)
        self._generic_types.pop(name, None)
        self._alias_type_params.pop(name, None)
        for key in tuple(self._constructor_sigs):
            if key[0] == name:
                self._constructor_sigs.pop(key, None)
        for key in tuple(self._constructor_field_kinds):
            if key[0] == name:
                self._constructor_field_kinds.pop(key, None)
        self._type_table.unregister(self._module_id, name)

    def register_alias(
        self, name: str, target_expr: object, *, type_params: tuple[str, ...] = ()
    ) -> None:
        """Store the raw TypeExpr for *name*; resolved lazily by resolve_type_expr.

        ``type_params`` must be provided for parameterized type aliases (e.g.
        ``type Wrapper[T] = list[T]``); defaults to ``()`` for plain aliases.
        """
        self._assert_mutable()
        self._alias_targets[name] = target_expr
        self._alias_type_params[name] = type_params

    def get_alias_type_params(self, name: str) -> tuple[str, ...]:
        """Return the type-parameter names for a parameterized alias, or ``()``."""
        return self._alias_type_params.get(name, ())

    # --- Generic type registry ---

    def register_generic_type(self, name: str, gdef: GenericTypeDef) -> None:
        """Register a generic type definition under *name*."""
        self._assert_mutable()
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

        Returns a ``RecordType``/``EnumType`` handle with ``type_args`` set to
        the supplied arguments; field/variant shapes are resolved later, by
        handle, from the shared ``TypeTable``.

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
        if len(args) != len(gdef.type_params):
            raise AglTypeError(
                f"Type '{name}' requires {len(gdef.type_params)} type argument(s), "
                f"got {len(args)}.",
                span=span,
            )
        # No field/variant substitution: the result is a bare handle with the
        # supplied type_args; field/variant shapes are looked up by handle in
        # the shared TypeTable (which substitutes type_args into the
        # registered TypeDef's templates on demand).
        template = gdef.template
        if isinstance(template, RecordType):
            return RecordType(name=name, type_args=args, module_id=template.module_id)
        return EnumType(name=name, type_args=args, module_id=template.module_id)

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
        self._assert_mutable()
        self._constructor_sigs[(sig.owner_name, sig.variant)] = sig

    def get_constructor_signature(
        self, owner_name: str, variant: str | None
    ) -> ConstructorSignature | None:
        """Return the constructor signature for *owner_name* / *variant*, or ``None``."""
        return self._constructor_sigs.get((owner_name, variant))

    def resolve_constructor_owner_name(self, name: str) -> str | None:
        """Resolve local nominal aliases to the name holding their constructor metadata."""
        from agm.agl.syntax.types import AppliedT, NameT

        seen: set[str] = set()
        while name not in seen:
            seen.add(name)
            target = self._alias_targets.get(name)
            if target is None:
                return name
            if isinstance(target, (NameT, AppliedT)) and target.module_qualifier is None:
                name = target.name
                continue
            return None
        return None

    def resolve_named_type(self, name: str) -> Type | None:
        """Resolve a type *name* alias-transparently to a semantic ``Type``.

        Returns the resolved ``Type`` for a record/enum/exception name or an
        alias chain (multi-hop, alias-of-alias) that bottoms out in a named
        type; ``None`` if the name is unknown or names a non-nominal alias
        target (e.g. an alias of ``list[int]``, which has no single name).
        Used for alias-transparent qualifier resolution in qualified
        constructors and ``is`` tests.

        In program context, also searches open-imported types when the name is not
        found locally.
        """
        if name in self._types or name in self._alias_targets:
            try:
                return self._resolve_name_type(name, span=None, _resolving=frozenset())
            except AglTypeError:
                return None
        # Program context: look up via open imports.
        if self._import_env is not None and self._program_type_table is not None:
            candidates = self._import_env.unqualified.get(name, frozenset())
            type_candidates = [qn for qn in candidates if self._is_program_type_candidate(qn)]
            if len(type_candidates) == 1:
                try:
                    return self._resolve_program_qname_as_bare_type(
                        type_candidates[0], name, span=None
                    )
                except AglTypeError:
                    return None
        return None

    def _is_program_type_candidate(self, qname: QName) -> bool:
        """Return whether a program-qualified name denotes any type-namespace declaration."""
        key = (qname[0], qname[1])
        return (
            (self._program_type_table is not None and key in self._program_type_table)
            or (self._program_generic_table is not None and key in self._program_generic_table)
            or key in self._program_alias_table
            or key in self._program_alias_keys
        )

    def _ensure_program_alias_resolved(
        self, module_id: ModuleId, name: str, span: SourceSpan | None
    ) -> Type | None:
        """Resolve a program alias lazily when the program pre-pass is still building it."""
        key = (module_id, name)
        if key not in self._program_alias_keys or self._program_alias_resolver is None:
            return None
        return self._program_alias_resolver(module_id, name, span)

    def _resolve_program_qname_as_bare_type(
        self, qname: QName, exposed_name: str, *, span: SourceSpan | None
    ) -> Type | None:
        """Resolve a program-qualified name used as an unapplied type expression."""
        assert self._program_type_table is not None
        key = (qname[0], qname[1])
        typ = self._program_type_table.get(key)
        if typ is not None:
            return typ
        typ = self._ensure_program_alias_resolved(qname[0], qname[1], span)
        if typ is not None:
            return typ
        alias_def = self._program_alias_table.get(key)
        if alias_def is not None and alias_def.type_params:
            raise AglTypeError(
                f"Parameterized alias '{exposed_name}' requires "
                f"{len(alias_def.type_params)} type argument(s); "
                f"use '{exposed_name}[...]' to apply it.",
                span=span,
            )
        return self._program_type_table.get(key)

    # --- Function signature table ---

    def register_function_signature(self, name: str, sig: FunctionSignature) -> None:
        self._assert_mutable()
        self._function_signatures[name] = sig

    def get_function_signature(self, name: str) -> FunctionSignature | None:
        return self._function_signatures.get(name)

    def all_function_signatures(self) -> dict[str, FunctionSignature]:
        return dict(self._function_signatures)

    def register_function_signature_by_node_id(self, node_id: int, sig: FunctionSignature) -> None:
        """Register a function signature keyed by its declaration ``node_id``.

        Used by the program pre-pass to seed every module's env with ALL
        function signatures before any body is checked.  Because ``node_id``
        is globally unique, signatures from different modules never
        collide here even when two modules define functions with the same name.
        """
        self._assert_mutable()
        self._function_signatures_by_node_id[node_id] = sig

    def get_function_signature_by_node_id(self, node_id: int) -> FunctionSignature | None:
        """Return the function signature for a callee's declaration ``node_id``.

        Used by ``_check_declared_name_call`` to look up the correct signature
        for a callee by its globally-unique declaration node id, avoiding the
        name collision problem when two modules define same-named functions with
        different signatures.

        Populated in program context by the whole-program function-signature pre-pass
        (via :meth:`register_function_signature_by_node_id`) AND in single-
        program mode by ``_preregister_funcdef`` (which always seeds both the
        name-keyed and node-id-keyed tables).  Returns ``None`` only for
        syntactically impossible cases (e.g. a callee node_id not registered
        because the function body check raised before registration).
        """
        return self._function_signatures_by_node_id.get(node_id)

    def register_extern_node_id(self, node_id: int) -> None:
        """Mark a function declaration ``node_id`` as an ``extern def``.

        Consulted by ``_check_declared_name_call`` so that direct calls to an
        extern (own-module or imported) are recorded as dry-run call sites the
        same way ``ask``/``exec`` calls are.
        """
        self._assert_mutable()
        self._extern_node_ids.add(node_id)

    def is_extern_node_id(self, node_id: int) -> bool:
        """Return ``True`` if *node_id* names a declared ``extern def``."""
        return node_id in self._extern_node_ids

    # --- Binding type table ---

    def set_binding_type(self, node_id: int, typ: Type) -> None:
        self._assert_mutable()
        self._binding_types[node_id] = typ

    def get_binding_type(self, node_id: int) -> Type | None:
        return self._binding_types.get(node_id)

    def remove_binding_types(self, node_ids: Iterable[int]) -> None:
        """Forget binding-type metadata for the given declaration node ids."""
        self._assert_mutable()
        for node_id in node_ids:
            self._binding_types.pop(node_id, None)
            self._function_signatures_by_node_id.pop(node_id, None)
            self._extern_node_ids.discard(node_id)

    def restore_binding_metadata_from(
        self,
        other: "TypeEnvironment",
        node_ids: Iterable[int],
        function_names: Iterable[str],
    ) -> None:
        """Restore selected binding/signature metadata from an earlier environment.

        Incremental hosts may check a whole entry before a runtime failure
        determines which declarations were actually installed.  This removes
        metadata for declarations that did not commit, then restores any
        pre-existing entries (normally none because declaration ids are
        globally unique).  Function names need separate handling because their
        convenient name-keyed signature table is not keyed by declaration id.
        """
        self._assert_mutable()
        node_id_set = set(node_ids)
        self.remove_binding_types(node_id_set)
        for node_id in node_id_set:
            binding_type = other._binding_types.get(node_id)
            if binding_type is not None:
                self._binding_types[node_id] = binding_type
            signature = other._function_signatures_by_node_id.get(node_id)
            if signature is not None:
                self._function_signatures_by_node_id[node_id] = signature
            if node_id in other._extern_node_ids:
                self._extern_node_ids.add(node_id)
        for name in function_names:
            self._function_signatures.pop(name, None)
            signature = other._function_signatures.get(name)
            if signature is not None:
                self._function_signatures[name] = signature

    def assert_closed(self) -> None:
        """Assert that this env's module-local metadata has no solver variables.

        Rigid ``TypeVarType`` nodes are valid inside persisted rank-1 schemes;
        flexible ``InferenceVarType`` nodes are owned by an expression region
        and must be finalized before an environment is seeded or retained.

        Only per-module state is walked here.  The whole-program tables that every
        module env of a graph shares (the ``TypeTable`` and the ``_graph_*``
        maps) are validated once per program by :meth:`assert_shared_tables_closed`
        rather than redundantly on every per-module seal.
        """
        types: list[Type] = [
            *self._types.values(),
            *self._binding_types.changed_values(),
            *(param.type for sig in self._function_signatures.values() for param in sig.params),
            *(sig.result for sig in self._function_signatures.values()),
            *(
                param.type
                for sig in self._function_signatures_by_node_id.values()
                for param in sig.params
            ),
            *(sig.result for sig in self._function_signatures_by_node_id.values()),
            *(generic.template for generic in self._generic_types.values()),
            *(
                template
                for sig in self._constructor_sigs.values()
                for template in sig.field_templates
            ),
            *(sig.result_template for sig in self._constructor_sigs.values()),
        ]
        if any(contains_inference_var(typ) for typ in types):
            raise AssertionError("inference variable leaked into a persistent type environment")

    def assert_shared_tables_closed(self) -> None:
        """Assert the whole-program tables shared across module envs are closed.

        The ``TypeTable`` and the cross-module ``_graph_*`` maps are the same
        instances on every module env of a graph, so they are validated once per
        program (or once per single-module program) instead of on every seal.  The
        program type table itself is validated by the caller from the authoritative
        :class:`CheckedProgram` field, so it is not re-walked here.
        """
        constructor_sigs = tuple((self._program_ctor_sig_table or {}).values())
        types: list[Type] = [
            *(alias.template for alias in self._program_alias_table.values()),
            *(generic.template for generic in (self._program_generic_table or {}).values()),
            *(template for sig in constructor_sigs for template in sig.field_templates),
            *(sig.result_template for sig in constructor_sigs),
            *(
                field_type
                for typedef in self._type_table.entries()
                for _, field_type in typedef.fields
            ),
            *(
                field_type
                for typedef in self._type_table.entries()
                for _, fields in typedef.variants
                for _, field_type in fields
            ),
        ]
        if any(contains_inference_var(typ) for typ in types):
            raise AssertionError("inference variable leaked into a persistent type environment")

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
                self.resolve_type_expr(a, span=None, _resolving=_resolving, type_vars=type_vars)
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
            program_alias_def = self._program_alias_table.get((self._module_id, name))
            if program_alias_def is not None:
                return self.instantiate_alias(name, program_alias_def, resolved_args, span=eff_span)
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
        if self._import_env is None or self._program_generic_table is None:
            return None
        candidates = self._import_env.unqualified.get(name, frozenset())
        type_candidates = [qname for qname in candidates if self._is_program_type_candidate(qname)]
        if len(type_candidates) > 1:
            labels = sorted(f"{qname[0].path_str()}::{qname[1]}" for qname in type_candidates)
            raise AglTypeError(
                f"Ambiguous type '{name}': it is exported by multiple modules "
                f"({', '.join(labels)}). Use a qualified reference to disambiguate.",
                span=span,
            )
        if not type_candidates:
            return None
        module_id, source_name = type_candidates[0]
        gdef = self._program_generic_table.get((module_id, source_name))
        if gdef is not None:
            return self.instantiate_from_gdef(source_name, gdef, args, span=span)
        alias_def = self._program_alias_table.get((module_id, source_name))
        if alias_def is None and (module_id, source_name) in self._program_alias_keys:
            self._ensure_program_alias_resolved(module_id, source_name, span)
            alias_def = self._program_alias_table.get((module_id, source_name))
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
        rendered = qualifier.render()
        if self._import_env is None or self._program_generic_table is None:
            raise AglTypeError(
                f"Module qualifier '{rendered}::' cannot be resolved outside of a module graph.",
                span=span,
            )
        qname = self._resolve_import_qname(qualifier, name, span=span)
        assert qname is not None
        gdef = self._program_generic_table.get((qname[0], qname[1]))
        if gdef is not None:
            return self.instantiate_from_gdef(qname[1], gdef, args, span=span)
        alias_def = self._program_alias_table.get((qname[0], qname[1]))
        if alias_def is None and (qname[0], qname[1]) in self._program_alias_keys:
            self._ensure_program_alias_resolved(qname[0], qname[1], span)
            alias_def = self._program_alias_table.get((qname[0], qname[1]))
        if alias_def is not None:
            return self.instantiate_alias(qname[1], alias_def, args, span=span)
        if (
            self._program_type_table is not None
            and (qname[0], qname[1]) in self._program_type_table
        ):
            raise AglTypeError(
                f"Type '{rendered}::{name}' does not take type arguments.",
                span=span,
            )
        raise AglTypeError(f"'{rendered}::{name}' does not name a type.", span=span)

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
        program_alias_def = self._program_alias_table.get((self._module_id, name))
        if program_alias_def is not None:
            raise AglTypeError(
                f"Parameterized alias '{name}' requires "
                f"{len(program_alias_def.type_params)} type argument(s); "
                f"use '{name}[...]' to apply it.",
                span=span,
            )
        # Direct named type (record, enum, exception, prelude).
        typ = self._types.get(name)
        if typ is not None:
            return typ
        # Program context: unqualified lookup through open-imported names.
        if self._import_env is not None and self._program_type_table is not None:
            candidates = self._import_env.unqualified.get(name, frozenset())
            # Filter to candidates that are type names in the program type namespace.
            type_candidates: list[QName] = [
                qn for qn in candidates if self._is_program_type_candidate(qn)
            ]
            if len(type_candidates) == 1:
                qn = type_candidates[0]
                typ = self._resolve_program_qname_as_bare_type(qn, name, span=span)
                if typ is not None:
                    return typ
            elif len(type_candidates) > 1:
                # Ambiguous: multiple modules export this type name.
                sorted_candidates = sorted(f"{qn[0].path_str()}::{qn[1]}" for qn in type_candidates)
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

        Called only in program context.  Falls back to the local type namespace
        (prelude / built-ins) when the qualifier is empty (``::Name``
        self-reference to the current module) and no program context exists.
        """
        from agm.agl.syntax.types import Qualifier

        assert isinstance(qualifier, Qualifier)
        rendered = qualifier.render()

        if self._program_type_table is None:
            # Single-module path: module qualifiers have no program table to consult.
            # ::Name (empty segments) = current module's own type.
            if not qualifier.segments:
                return self._resolve_name_type(name, span=span, _resolving=frozenset())
            raise AglTypeError(
                f"Module qualifier '{rendered}::' cannot be resolved outside of a module graph.",
                span=span,
            )

        # ``::Name`` — self-reference to the current module's own type.
        if not qualifier.segments:
            key = (self._module_id, name)
            t = self._program_type_table.get(key)
            if t is not None:
                return t
            # Fall back to own local types (covers built-ins and prelude).
            return self._resolve_name_type(name, span=span, _resolving=frozenset())

        # Qualified reference: resolve through the contribution environment.
        assert self._import_env is not None
        qname = self._resolve_import_qname(qualifier, name, span=span)
        assert qname is not None
        if not self._is_program_type_candidate(qname):
            raise AglTypeError(f"'{rendered}::{name}' does not name a type.", span=span)
        typ = self._resolve_program_qname_as_bare_type(qname, name, span=span)
        if typ is not None:
            return typ
        raise AglTypeError(f"'{rendered}::{name}' does not name a type.", span=span)

    def non_builtin_type_items(self) -> list[tuple[str, Type]]:
        """Return ``(name, type)`` pairs for all non-builtin registered types.

        Used by the program pre-pass to collect type shells into the shared
        ``program_type_table`` without accessing the private ``_types`` dict.
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

    def resolve_type_by_module_id(self, module_id: ModuleId, name: str) -> Type | None:
        """Directly look up a type by owning module and name in the program type table.

        Used for cross-module constructor references when the owning module is
        already known from scope resolution (e.g. ``mylib::Color::Red``).

        Returns ``None`` in module mode or if not found.
        """
        if self._program_type_table is None:
            return None
        return self._program_type_table.get((module_id, name))

    def resolve_constructible_type_by_module_id(
        self, module_id: ModuleId, name: str
    ) -> RecordType | EnumType | ExceptionType | None:
        """Resolve a program type name or transparent alias to a nominal target.

        Scope tentatively classifies aliases with a named or applied body as
        constructor bindings. This confirms that classification after aliases
        have resolved, without exposing a non-nominal alias to constructor
        lookup.
        """
        template = self.source_type_template_qname(module_id, name)
        if template is None:
            return None
        target = template.template
        if isinstance(target, (RecordType, EnumType, ExceptionType)):
            return target
        return None

    def match_source_type_qname(
        self,
        module_id: ModuleId,
        name: str,
        concrete: Type,
    ) -> TypeTemplateMatch | None:
        """Alias-transparently match a checked source type QName to ``concrete``.

        This is the public checked-type boundary for downstream consumers that
        retain source import names. It exposes only an immutable semantic match,
        not the mutable program type/generic/alias registries. Non-generic aliases,
        generic aliases, alias chains, transformed arguments, and ordinary
        nominal declarations all share the exact template matcher.
        """
        template = self.source_type_template_qname(module_id, name)
        return None if template is None else template.match(concrete)

    def source_type_template(self, name: str) -> TypeTemplate | None:
        """Return immutable checked template data for an own-module source type."""
        return self.source_type_template_qname(self._module_id, name)

    def source_type_template_qname(self, module_id: ModuleId, name: str) -> TypeTemplate | None:
        """Return immutable checked template data for one source type QName.

        Program aliases are already cycle-checked and resolved by the checked
        program boundary. Own-module fallback supports the module path
        without exposing any mutable type-environment registry.
        """
        key = (module_id, name)
        alias_def = self._program_alias_table.get(key)
        if alias_def is not None:
            return TypeTemplate(alias_def.template, alias_def.type_params)
        if self._program_generic_table is not None:
            generic_def = self._program_generic_table.get(key)
            if generic_def is not None:
                return TypeTemplate(generic_def.template, generic_def.type_params)
        if self._program_type_table is not None:
            resolved = self._program_type_table.get(key)
            if resolved is not None:
                return TypeTemplate(resolved)
        if module_id != self._module_id:
            return None
        local_generic = self._generic_types.get(name)
        if local_generic is not None:
            return TypeTemplate(local_generic.template, local_generic.type_params)
        alias_expr = self._alias_targets.get(name)
        if alias_expr is not None:
            type_params = self._alias_type_params.get(name, ())
            template = self.resolve_type_expr(
                alias_expr,
                _resolving=frozenset({name}),
                type_vars=frozenset(type_params),
            )
            return TypeTemplate(template, type_params)
        resolved = self._types.get(name)
        return None if resolved is None else TypeTemplate(resolved)

    def _own_source_type_names(self) -> frozenset[str]:
        cached = self._sealed_own_source_type_names
        if cached is not None:
            return cached
        names = set(self._types) | set(self._alias_targets) | set(self._generic_types)
        names.update(
            name for module_id, name in self._program_alias_table if module_id == self._module_id
        )
        if self._program_generic_table is not None:
            names.update(
                name
                for module_id, name in self._program_generic_table
                if module_id == self._module_id
            )
        if self._program_type_table is not None:
            names.update(
                name for module_id, name in self._program_type_table if module_id == self._module_id
            )
        own_names = frozenset(names)
        if self._sealed:
            self._sealed_own_source_type_names = own_names
        return own_names

    def resolve_enum_owner_form(
        self,
        kind: EnumOwnerFormKind,
        owner_name: str,
        module_qualifier: Qualifier | None = None,
        *,
        span: SourceSpan | None = None,
    ) -> EnumOwnerForm | None:
        """Resolve one exact enum-owner source form through checked visibility."""
        source_module_id: ModuleId
        source_name: str
        expected_qualifier: tuple[str, ...] | None
        if kind in (EnumOwnerFormKind.LOCAL, EnumOwnerFormKind.SELF):
            if owner_name not in self._own_source_type_names():
                return None
            source_module_id = self._module_id
            source_name = owner_name
            expected_qualifier = None if kind is EnumOwnerFormKind.LOCAL else ()
        elif kind is EnumOwnerFormKind.OPEN_IMPORT:
            if self._import_env is None or owner_name in self._own_source_type_names():
                return None
            type_qnames = tuple(
                qname
                for qname in self._import_env.unqualified.get(owner_name, frozenset())
                if self._is_program_type_candidate(qname)
            )
            if len(type_qnames) != 1:
                return None
            source_module_id, source_name = type_qnames[0]
            expected_qualifier = None
        else:
            if (
                self._import_env is None
                or module_qualifier is None
                or not module_qualifier.segments
            ):
                return None
            # A qualified enum spelling must preserve the shared resolver's
            # verdict.  In particular, an unknown route or ambiguity is not a
            # statement that the subject is "not an enum".
            qname = self._resolve_import_qname(module_qualifier, owner_name, span=span)
            assert qname is not None
            if not self._is_program_type_candidate(qname):
                return None
            source_module_id, source_name = qname
            expected_qualifier = module_qualifier.segments
        template = self.source_type_template_qname(source_module_id, source_name)
        assert template is not None
        return EnumOwnerForm(
            owner_name,
            expected_qualifier,
            kind=kind,
            source_module_id=source_module_id,
            source_name=source_name,
            type_template=template,
            qualifier_anchored=(
                module_qualifier.anchored if module_qualifier is not None else False
            ),
        )

    def resolve_unqualified_enum_owner_form(self, owner_name: str) -> EnumOwnerForm | None:
        """Resolve ``Owner::variant`` with local-before-open precedence."""
        if owner_name in self._own_source_type_names():
            return self.resolve_enum_owner_form(EnumOwnerFormKind.LOCAL, owner_name)
        return self.resolve_enum_owner_form(EnumOwnerFormKind.OPEN_IMPORT, owner_name)

    def _blocked_short_variants(self, form: EnumOwnerForm) -> frozenset[str]:
        """Return variants whose short owner spelling is occupied by a module route.

        Only a ``LOCAL``/``OPEN_IMPORT`` form can be shadowed this way: those
        are the only kinds writable as a bare ``owner_name`` qualifier, which
        is exactly the qualifier a same-named module route also competes for.
        """
        template = form.type_template
        if (
            self._import_env is None
            or form.kind not in (EnumOwnerFormKind.LOCAL, EnumOwnerFormKind.OPEN_IMPORT)
            or template is None
            or not isinstance(template.template, EnumType)
        ):
            return frozenset()
        owner_qualifier = (form.owner_name or "",)
        return frozenset(
            variant
            for variant in self.type_table.enum_variants(template.template)
            if qualifier_contributes(self._import_env, owner_qualifier, variant)
        )

    def enum_owner_forms(self) -> tuple[EnumOwnerForm, ...]:
        """Enumerate finite checked owner forms writable in this environment.

        Memoized once the environment is sealed, so consumers that need the
        owner forms per case (match compilation) can simply ask the environment.
        """
        cached = self._sealed_enum_owner_forms
        if cached is not None:
            return cached
        forms: set[EnumOwnerForm] = set()
        for owner_name in self._own_source_type_names():
            for kind in (EnumOwnerFormKind.LOCAL, EnumOwnerFormKind.SELF):
                own_form = cast(EnumOwnerForm, self.resolve_enum_owner_form(kind, owner_name))
                forms.add(own_form)
        if self._import_env is not None:
            for owner_name in self._import_env.unqualified:
                form = self.resolve_enum_owner_form(EnumOwnerFormKind.OPEN_IMPORT, owner_name)
                if form is None:
                    continue
                forms.add(form)
            for contribution in self._import_env.contributions.values():
                routes = contribution_routes(contribution)
                for owner_name, qname in contribution.members.items():
                    if not self._is_program_type_candidate(qname):
                        continue
                    template = cast(TypeTemplate, self.source_type_template_qname(*qname))
                    for qualifier, anchored in routes:
                        resolved = resolve_qualified(
                            self._import_env, qualifier, owner_name, anchored=anchored
                        )
                        if not isinstance(resolved, QualResolutionFound):
                            continue
                        forms.add(
                            EnumOwnerForm(
                                owner_name,
                                qualifier,
                                kind=EnumOwnerFormKind.QUALIFIED_IMPORT,
                                source_module_id=qname[0],
                                source_name=qname[1],
                                type_template=template,
                                qualifier_anchored=anchored,
                            )
                        )

        def form_key(form: EnumOwnerForm) -> tuple[str, tuple[str, ...], bool, str]:
            assert form.kind is not None
            return (
                form.owner_name or "",
                form.module_qualifier or (),
                form.qualifier_anchored,
                form.kind.value,
            )

        ordered = tuple(sorted(forms, key=form_key))
        if self._sealed:
            self._sealed_enum_owner_forms = ordered
        return ordered

    def blocked_enum_variants(self) -> Mapping[tuple[str, ...], frozenset[str]]:
        """Map each short enum-owner qualifier to the variants a module route blocks.

        A match-compile consumer selecting a source spelling for one concrete
        enum constructor needs this alongside ``enum_owner_forms``: an
        ``EnumOwnerForm`` describes only an owner spelling, never which of
        that owner's variants a same-named module route makes ambiguous.
        The key is the same ``(owner_name,)`` qualifier
        ``qualifier_contributes`` checks the module routes against.

        Memoized on the same terms as ``enum_owner_forms``.
        """
        cached = self._sealed_blocked_enum_variants
        if cached is not None:
            return cached
        owner_forms = self.enum_owner_forms()
        blocked: dict[tuple[str, ...], frozenset[str]] = {}
        for form in owner_forms:
            if form.kind not in (EnumOwnerFormKind.LOCAL, EnumOwnerFormKind.OPEN_IMPORT):
                continue
            variants = self._blocked_short_variants(form)
            if variants:
                blocked[(form.owner_name or "",)] = variants
        result: Mapping[tuple[str, ...], frozenset[str]] = MappingProxyType(blocked)
        if self._sealed:
            self._sealed_blocked_enum_variants = result
        return result

    def get_generic_type_from_module(self, module_id: ModuleId, name: str) -> GenericTypeDef | None:
        """Look up a cross-module ``GenericTypeDef`` by owning module and name.

        Used to detect and instantiate generic constructors referenced via a
        module qualifier (e.g. ``lib::Box[int](value: 1)``).  Returns ``None``
        in module mode or when the type is not generic.
        """
        if self._program_generic_table is None:
            return None
        return self._program_generic_table.get((module_id, name))

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
                f"{module_id.path_str()}::{source_name}" for module_id, source_name, _ in matches
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
            if self._program_generic_table is None:
                return None
            gdef = self._program_generic_table.get((self._module_id, name))
            return (name, gdef) if gdef is not None else None
        if self._import_env is None or self._program_generic_table is None:
            return None
        qname = self._resolve_import_qname(qualifier, name, span=span, required=False)
        if qname is None:
            return None
        gdef = self._program_generic_table.get((qname[0], qname[1]))
        if gdef is None:
            return None
        rendered = qualifier.render()
        qualified_name = f"{rendered}::{name}"
        return qualified_name, gdef

    def _open_imported_generic_type_matches(
        self, exposed_name: str
    ) -> list[tuple[ModuleId, str, GenericTypeDef]]:
        if self._import_env is None or self._program_generic_table is None:
            return []
        matches: list[tuple[ModuleId, str, GenericTypeDef]] = []
        for module_id, source_name in self._import_env.unqualified.get(exposed_name, frozenset()):
            gdef = self._program_generic_table.get((module_id, source_name))
            if gdef is not None:
                matches.append((module_id, source_name, gdef))
        return matches

    def get_ctor_sig_from_module(
        self, module_id: ModuleId, owner_name: str, variant: str | None
    ) -> ConstructorSignature | None:
        """Look up a cross-module ``ConstructorSignature`` by owning module, name, and variant.

        Companion to :meth:`get_generic_type_from_module` for checking cross-module
        generic constructor calls.  Returns ``None`` in module mode or when
        not found.
        """
        if self._program_ctor_sig_table is None:
            return None
        return self._program_ctor_sig_table.get((module_id, owner_name, variant))

    def register_constructor_field_kinds(
        self,
        owner_name: str,
        variant: str | None,
        fields: tuple[tuple[str, ParamKind], ...],
    ) -> None:
        """Register ordered (field_name, ParamKind) pairs for a constructor."""
        self._assert_mutable()
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
        if module_id is not None and self._program_ctor_field_kinds_table is not None:
            return self._program_ctor_field_kinds_table.get((module_id, owner_name, variant))
        return None

    def get_constructor_field_kinds_for_type(
        self,
        typ: Type | None,
        owner_name: str,
        variant: str | None,
    ) -> tuple[tuple[str, ParamKind], ...] | None:
        """Return field-kinds for a constructor identified by its resolved owner *typ*.

        ``RecordType``, ``EnumType``, and ``ExceptionType`` all carry their own
        ``module_id``, so the owning module is read directly off the handle —
        no caller-supplied module id is needed.  Exception field kinds are
        derived directly from ``type_table.exception_field_kinds``, which
        flattens the ``extends`` base chain (base kinds first, then own kinds,
        each honoring its declaration's ``@pos``/``@std``/``@named`` marker —
        exactly like a record's fields) and excludes ``trace_id`` (auto-filled
        at construction time, never supplied by the caller), rather than
        through the registered-kinds table records/enums use, since an
        exception's kinds are never pre-registered (see ``TypeEnvironment.
        __init__``).  ``exception_field_kinds`` returns ``ParamKind.value``
        strings rather than the enum (``semantics`` may not import
        ``syntax.nodes``), so each is converted back with ``ParamKind(...)``
        here, in the ``typecheck`` layer.  Enum variants are keyed by
        *variant*; records are keyed under ``variant=None``.
        """
        if isinstance(typ, ExceptionType):
            return tuple(
                (fname, ParamKind(kind_value))
                for fname, kind_value in self._type_table.exception_field_kinds(typ)
            )
        if isinstance(typ, EnumType):
            return self.get_constructor_field_kinds(owner_name, variant, module_id=typ.module_id)
        assert isinstance(typ, RecordType), f"unexpected constructor owner type {typ!r}"
        return self.get_constructor_field_kinds(owner_name, None, module_id=typ.module_id)

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

        Also merges *other*'s ``type_table`` entries in: *other* is treated as
        authoritative, so an entry under a key already present in this
        environment's table is overwritten (last-write-wins). For names present
        in *other*'s type namespace, stale metadata in this environment is
        cleared before copying so cross-kind REPL redefinitions do not leave old
        generic/constructor/alias tables behind. See :meth:`TypeTable.merge_from`.
        """
        self._assert_mutable()
        if not other.is_sealed:
            raise AssertionError("cannot seed from an unsealed type environment")
        builtin = frozenset(BUILTIN_EXCEPTIONS) | BUILTIN_PRELUDE_TYPE_NAMES
        incoming_type_names = (
            {name for name in other._types if name not in builtin}
            | set(other._alias_targets)
            | set(other._generic_types)
            | set(other._alias_type_params)
        )
        incoming_type_names |= {owner_name for owner_name, _variant in other._constructor_sigs}
        incoming_type_names |= {
            owner_name for owner_name, _variant in other._constructor_field_kinds
        }
        for name in incoming_type_names:
            self.unregister_name(name)
        self._type_table.merge_from(other._type_table)
        for name, typ in other._types.items():
            if name not in builtin:
                self._types[name] = typ
        self._alias_targets.update(other._alias_targets)
        self._binding_types = other._binding_types.fork()
        self._function_signatures.update(other._function_signatures)
        self._generic_types.update(other._generic_types)
        self._constructor_sigs.update(other._constructor_sigs)
        self._constructor_field_kinds.update(other._constructor_field_kinds)
        self._alias_type_params.update(other._alias_type_params)
        self._function_signatures_by_node_id.update(other._function_signatures_by_node_id)
        self._extern_node_ids.update(other._extern_node_ids)

    def restore_type_names_from(self, other: TypeEnvironment, names: Iterable[str]) -> None:
        """Restore selected type-namespace names from *other*.

        Used by the REPL after partial runtime failure: checking an entry builds
        metadata for every declaration in the entry, but only declarations before
        the failure are promoted. For each unpromoted type name, remove the
        checked-entry metadata and restore the previous session definition when
        one existed.
        """
        self._assert_mutable()
        builtin = frozenset(BUILTIN_EXCEPTIONS) | BUILTIN_PRELUDE_TYPE_NAMES
        for name in names:
            if name in builtin:
                continue
            self.unregister_name(name)
            typedef = other._type_table.get(other._module_id, name)
            if typedef is not None:
                self._type_table.register(typedef)
            if name in other._types:
                self._types[name] = other._types[name]
            if name in other._alias_targets:
                self._alias_targets[name] = other._alias_targets[name]
            if name in other._generic_types:
                self._generic_types[name] = other._generic_types[name]
            if name in other._alias_type_params:
                self._alias_type_params[name] = other._alias_type_params[name]
            for key, sig in other._constructor_sigs.items():
                if key[0] == name:
                    self._constructor_sigs[key] = sig
            for key, kinds in other._constructor_field_kinds.items():
                if key[0] == name:
                    self._constructor_field_kinds[key] = kinds
