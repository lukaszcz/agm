"""Type environment and output types for the AgL type checker.

``TypeEnvironment`` holds:
- The user-declared type namespace (records, enums, aliases, exceptions).
- The variable → Type binding table derived from the scope side tables.

``CheckedModule`` is the frozen output of the type-checking pass.
``OutputContractSpec`` records the statically derived codec + target type per
``AgentCall`` node.
``AglTypeError`` is the fatal type error raised by the checker.
``EnvironmentFacts`` is the replayable journal of a ``TypeEnvironment``'s own facts.
``CheckedModuleImage`` is a data-only ``CheckedModule`` for cache persistence;
:meth:`CheckedModule.image` builds one, :meth:`CheckedModuleImage.rehydrate`
turns it back into an equivalent ``CheckedModule`` over a freshly prepared
environment.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, Protocol, cast

if TYPE_CHECKING:
    from agm.agl.scope.program import ResolvedModule
    from agm.agl.typecheck.function_inference import FunctionSignatureRecord

from agm.agl.constraints import ConstraintBounds
from agm.agl.diagnostics import AglTypeError as AglTypeError
from agm.agl.diagnostics import Diagnostic
from agm.agl.ir.ids import NominalId
from agm.agl.ir.reserved_nominals import NO_DECL_ID, require_reserved_nominal_id
from agm.agl.modules.ids import ENTRY_ID, ModuleId, spell_declaration
from agm.agl.scope.imports import (
    EMPTY_IMPORT_ENV,
    ImportEnv,
    NameAtom,
    QName,
    QualResolutionFound,
    contribution_routes,
    qualification_repair_guidance,
    qualifier_candidates,
    qualifier_contributes,
    resolve_qualified,
    resolve_qualified_member,
    try_resolve_qualified_member,
)
from agm.agl.scope.symbols import (
    BindingRef,
    ConstructorRef,
    ModuleResolution,
    ScopeNode,
    ScopePath,
    contributed_declarations,
    resolve_bare_contribution_layer,
    to_bare_atom,
)
from agm.agl.scope.type_names import TypeNameSite, bare_type_selection, type_name_selection
from agm.agl.self_validation import self_validation_enabled
from agm.agl.semantics.persistent import PersistentDict
from agm.agl.semantics.type_table import (
    DeclKey,
    MethodDef,
    NominalOwner,
    TypeDef,
    TypeTable,
    create_seeded_type_table,
)
from agm.agl.semantics.types import (
    BUILTIN_ALIAS_TARGETS,
    BUILTIN_EXCEPTIONS,
    BUILTIN_PRELUDE_TYPE_NAMES,
    BUILTIN_PRELUDE_TYPES,
    ArrayType,
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
    RecordType,
    TextType,
    Type,
    TypeTemplate,
    TypeVarType,
    UnitType,
    contains_inference_var,
)
from agm.agl.syntax.nodes import Expr, Pattern, QualifierAnchor, QualifierChain
from agm.agl.syntax.qualifiers import enclosing_scope_bases
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import AppliedT, NameT, TypeExpr
from agm.agl.zones import ParamZone

#: Every built-in name a module's type namespace carries a reserved fallback
#: binding for, whether or not any source declares it.
_BUILTIN_FALLBACK_TYPE_NAMES: frozenset[str] = frozenset(BUILTIN_EXCEPTIONS) | (
    BUILTIN_PRELUDE_TYPE_NAMES
)


def _split_scoped_type_name(name: str) -> tuple[ScopePath, str]:
    """Split a source spelling only at the environment's UI boundary."""
    *scope_path, declared_name = name.split("::")
    return tuple(scope_path), declared_name


def owner_type_expr(qualifier: QualifierChain) -> NameT | AppliedT:
    """Return the type expression a non-empty qualifier's last segment names as an owner."""
    owner_segment = qualifier.segments[-1]
    prefix_segments = qualifier.segments[:-1]
    owner_qualifier = (
        None
        if not prefix_segments and qualifier.anchor is None
        else QualifierChain(
            anchor=qualifier.anchor,
            segments=prefix_segments,
            member=owner_segment.name,
            span=qualifier.span,
            node_id=qualifier.node_id,
        )
    )
    if owner_segment.type_args is None:
        return NameT(
            name=owner_segment.name,
            qualifier=owner_qualifier,
            span=owner_segment.span,
            node_id=owner_segment.node_id,
        )
    return AppliedT(
        name=owner_segment.name,
        args=owner_segment.type_args,
        qualifier=owner_qualifier,
        span=owner_segment.span,
        node_id=owner_segment.node_id,
    )


def _is_own_builtin_declaration(name: str, typ: Type) -> bool:
    """Return whether *typ* is a program's own ``builtin`` declaration of reserved *name*.

    A canonical binding for a built-in exception or prelude type name carries
    that name's fixed reserved identity (``ir.reserved_nominals``); a
    source ``builtin`` declaration of the same name instead carries its own
    declaration identity, wherever it is declared (a standard-library module
    included), which never equals the reserved one. *name* is always one of the reserved
    names, so it always has a reserved identity to compare against.

    ``NO_DECL_ID`` is excluded too: it is the identity a handle carries when
    none is attached at all, never a real declaration's, so a binding
    carrying it is neither the canonical one nor a program's own and must not
    displace the canonical default.
    """
    if not isinstance(typ, (RecordType, EnumType, ExceptionType)):
        return False
    if typ.decl_id == NO_DECL_ID:
        return False
    return typ.decl_id != require_reserved_nominal_id(name)


def _render_type_atom(atom: NameAtom) -> str:
    """Render a structured type atom for diagnostics only."""
    return atom if isinstance(atom, str) else "::".join(atom)


def _member_values(source: object, names: tuple[str, ...]) -> tuple[object, ...]:
    """Read *names* off *source*, in order."""
    values: list[object] = []
    for name in names:
        value: object = getattr(source, name)
        values.append(value)
    return tuple(values)


# ---------------------------------------------------------------------------
# _Record — shared behavior of this module's data records
# ---------------------------------------------------------------------------


class _Record:
    """Base of this module's frozen, slotted data records.

    ``__match_args__`` names a record's fields in declaration order, which is
    all its state amounts to, so the hooks below move that state by name.
    Restoring one ``CheckedModuleImage`` runs them thousands of times, which
    is why they read that tuple rather than the field tuple
    ``dataclass(frozen=True, slots=True)``'s own hooks rebuild per object.
    ``image``/``rehydrate`` carry a record's state across the same way.
    """

    __slots__ = ()
    __match_args__: ClassVar[tuple[str, ...]] = ()

    def __getstate__(self) -> tuple[object, ...]:
        return _member_values(self, self.__match_args__)

    def __setstate__(self, state: tuple[object, ...]) -> None:
        for name, value in zip(self.__match_args__, state):
            object.__setattr__(self, name, value)


def _pickles_by_name[T: _Record](cls: type[T]) -> type[T]:
    """Reinstate :class:`_Record`'s pickle hooks on a record class.

    ``dataclass(slots=True)`` rebuilds the class and puts its own hooks in the
    new body, where they would otherwise shadow the inherited ones.
    """
    setattr(cls, "__getstate__", _Record.__getstate__)
    setattr(cls, "__setstate__", _Record.__setstate__)
    return cls


# ---------------------------------------------------------------------------
# ParamSpec — per-parameter descriptor in a FunctionSignature
# ---------------------------------------------------------------------------


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class ParamSpec(_Record):
    """Full descriptor for one parameter in a ``FunctionSignature``.

    ``name``        — the declared parameter name.
    ``type``        — the resolved semantic type.
    ``kind``        — positional-only, standard, or named-only (from the AST).
    ``has_default`` — whether a default expression was provided.
    """

    name: str
    type: Type
    kind: ParamZone
    has_default: bool


# ---------------------------------------------------------------------------
# FunctionSignature — full declared signature of a static def
# ---------------------------------------------------------------------------


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class FunctionSignature(_Record):
    """Full declared signature of a root or named-scope ``def``.

    Carries named/default/kind information needed for declared-name call sites.
    The value type (FunctionType) erases names/defaults/kinds.

    ``params``      — ordered list of ``ParamSpec`` (name, type, kind, has_default).
    ``result``      — the declared return type.
    ``type_params`` — tuple of type-parameter names for generic functions
                      (empty for non-generic functions).
    ``target_params`` — an ``extern def``'s type parameters no value parameter
                      (receiver included) mentions, in declaration order; each
                      call site resolves them and delivers their contracts.
                      Empty for every other function.
    ``bounds``      — the declaration's constraint block, type-parameter name to
                      its closed constraint kinds (``Hashable`` implies ``Eq``).
                      Empty for a declaration with no ``{...}`` block.
    """

    params: tuple[ParamSpec, ...]
    result: Type
    type_params: tuple[str, ...] = ()
    target_params: tuple[str, ...] = ()
    bounds: ConstraintBounds = MappingProxyType({})

    @property
    def value_type(self) -> FunctionType:
        """The function value type: parameter types to the result."""
        return FunctionType(params=tuple(param.type for param in self.params), result=self.result)


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class GenericTypeDef(_Record):
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


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class GenericAliasDef(_Record):
    """Resolved template for a parameterized type alias.

    ``type_params`` — ordered tuple of type-parameter names.
    ``template``    — alias body resolved in the alias-defining module with its
                      parameters represented as ``TypeVarType`` nodes.
    """

    type_params: tuple[str, ...]
    template: Type


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class ConstructorSignature(_Record):
    """Signature for one record or exception constructor.

    ``owner_name``      — the spelling the constructor was written with.
    ``field_names``     — ordered field names accepted by the constructor.
    ``field_templates`` — field types (may contain ``TypeVarType`` nodes for
                          generic types).
    ``result_template`` — the constructed nominal (may contain TypeVarType).
    ``type_params``     — type-parameter names for instantiation.
    """

    owner_name: str
    field_names: tuple[str, ...]
    field_templates: tuple[Type, ...]
    result_template: RecordType | ExceptionType
    type_params: tuple[str, ...]


class OwnerMember(NamedTuple):
    """An inline enum member selected through its qualifying owner.

    ``member``      — the member record at the owner's type arguments.
    ``type_params`` — the owner alias's own type parameters, which ``member``
                      may mention; empty for an applied owner.
    """

    member: RecordType
    type_params: tuple[str, ...]


# ---------------------------------------------------------------------------
# OutputContractSpec — per-call contract descriptor
# ---------------------------------------------------------------------------


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class CallSiteRecord(_Record):
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


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class OutputContractSpec(_Record):
    """Statically derived output contract for one ``ask``/``exec`` call, or for one
    target parameter of a type-directed extern occurrence (always strict ``json``).

    ``target_type``
        The resolved semantic type the output will be parsed into.
    ``codec_name``
        The codec selected for this call (e.g. ``"text"`` or ``"json"``).
        Output-discarding unit calls use ``"none"``. When ``structured_exec``
        is ``True`` this field holds the unused placeholder ``"text"``.
    ``strict_json``
        The strict-JSON flag; ``None`` when unset or the codec is not JSON.
    ``structured_exec``
        ``True`` for the structured ``exec`` form (target is ``ExecResult``):
        returns the raw result record, does not parse stdout, does not raise
        on nonzero exit.  ``False`` (the default) for all other calls.
    """

    target_type: Type
    codec_name: str
    strict_json: bool | None
    structured_exec: bool = False


# ---------------------------------------------------------------------------
# ArgumentBindings — checker-computed call/pattern argument bindings
# ---------------------------------------------------------------------------


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class ArgumentBindings(_Record):
    """Checker-computed argument bindings for call-like constructs, keyed by node_id.

    The checker is the single source of truth for how each construct's
    positional, named, and bare-name-shorthand arguments map onto declared
    parameters or fields.  The lowerer reads these instead of re-running the
    binder.

    ``function_calls``
        Complete direct user-function ``Call.node_id`` → declaration-order
        argument tuple (one entry per parameter; ``None`` means "use the
        parameter's default"). A partial call's binding is its
        :class:`PartialCallSpec`.
    ``function_param_types``
        Direct user-function ``Call.node_id`` → concrete declaration-order parameter
        types after generic type-argument substitution.  The lowerer uses these
        types to insert call-site coercions without re-inferring generic arguments.
    ``constructor_calls``
        Complete record/enum/exception constructor ``Call.node_id`` → ordered
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


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class PartialCallSpec(_Record):
    """Checker-computed binding of a partial call, which produces a function.

    ``callee_kind`` identifies which lowering path the underlying call uses.
    ``arguments`` is ordered like the checked call binding for that callee
    (declaration order; field order for a constructor): each slot is the
    supplied argument expression, the produced-function parameter index a
    placeholder fills, or ``None`` for a defaulted parameter.
    """

    arguments: tuple[Expr | int | None, ...]
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


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class CheckedModule(_Record):
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
    ``target_contract_specs``
        Maps the node id of each type-directed extern occurrence (call,
        reference, method projection, partial application) → one strict JSON
        ``OutputContractSpec`` per target parameter of the callee
        (``FunctionSignature.target_params``), in declaration order.
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
    ``selected_constructor_refs``
        The constructor the checker selected where it differs from the
        scope-resolved owner (alias spellings, multi-candidate ``is`` tests).
        ``constructor_ref_for`` exposes the selection to lowering without
        rewriting scope's resolution table.
    ``pattern_binding_refs`` / ``pattern_constructor_refs``
        Checker-selected meanings for individual pattern occurrences. They
        preserve scope's immutable candidates while making final selections
        available to later frontend stages. ``pattern_constructor_owners``
        publishes the concrete nominal identity selected for each applied
        constructor pattern because a source-spelling ``ConstructorRef`` can
        name a transparent alias; downstream passes compare this identity
        instead of re-running candidate selection.
    ``method_selections``
        The method selected for each field access that named one; a field
        access absent here selected a field. Lowering consumes these decisions
        rather than repeating member lookup, and reads the selection on a
        call's callee to choose the receiver-first direct call.
    ``explicit_builtin_targets``
        Maps a built-in call's ``Call.node_id`` → the resolved ``Type`` of its
        explicit ``::[T]`` type argument, for every built-in that accepts one
        (``print``, ``render``, ``copy``, ``shallow_copy``, ``ask``,
        ``ask-request``, ``exec``). Omitted entirely when the call has no
        explicit type argument. ``print``/``render`` lowering reads this to
        recover the coercion target their own checked result type discards
        (``unit``/``text``); ``copy``/``shallow_copy`` publish the same type
        here as their checked result in ``node_types``, so either source works
        for them.
    ``published_binding_types``
        This module's exported static ``let``/``var`` binding types, keyed by
        declaration node id (see ``syntax.nodes.static_binding_node_id``). Lets
        a cache hit answer the program-level static-binding pre-pass
        (``typecheck/program.py``) without re-checking the module.
    ``program_config_targets``
        Resolved ``@config`` targets, keyed by each entry's key expression node
        id: ``(module id, scope path, name)``, the same shape as
        ``ir.static_keys.StaticBindingKey`` (this module may not import
        ``ir``). Lowering reads this instead of re-resolving
        ``resolved.attributes.program_configs``' raw keys.
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
    published_signatures: dict[int, FunctionSignatureRecord] | None = None
    published_binding_types: dict[int, Type] | None = None
    module_id: ModuleId = ENTRY_ID
    import_env: ImportEnv = field(default_factory=lambda: ImportEnv({}, {}))
    source_text: str = ""
    slot_resolution: dict[int, BindingRef] = field(default_factory=dict)
    slot_constructor_refs: dict[int, ConstructorRef] = field(default_factory=dict)
    selected_constructor_refs: dict[int, ConstructorRef] = field(default_factory=dict)
    pattern_binding_refs: dict[int, BindingRef] = field(default_factory=dict)
    pattern_constructor_refs: dict[int, ConstructorRef] = field(default_factory=dict)
    pattern_constructor_owners: dict[int, NominalId] = field(default_factory=dict)
    method_selections: dict[int, MethodDef] = field(default_factory=dict)
    explicit_builtin_targets: dict[int, Type] = field(default_factory=dict)
    program_config_targets: dict[int, tuple[ModuleId, tuple[str, ...], str]] = field(
        default_factory=dict
    )
    target_contract_specs: dict[int, tuple[OutputContractSpec, ...]] = field(default_factory=dict)

    def binding_for(self, node_id: int) -> BindingRef | None:
        """Return *node_id*'s checked binding, dereferencing a pattern slot."""
        return dereference_slot_binding(
            node_id,
            resolution=self.resolved.resolution,
            slot_resolution=self.slot_resolution,
        )

    def constructor_ref_for(self, node_id: int) -> ConstructorRef | None:
        """Return *node_id*'s scope or checker-selected constructor reference."""
        selected = self.selected_constructor_refs.get(node_id)
        if selected is not None:
            return selected
        return dereference_slot_constructor_ref(
            node_id,
            resolution=self.resolved.resolution,
            constructor_refs=self.resolved.constructor_refs,
            slot_constructor_refs=self.slot_constructor_refs,
        )

    def method_selection_for(self, node_id: int) -> MethodDef | None:
        """Return the method selected for one member-access node, if any."""
        return self.method_selections.get(node_id)

    @property
    def interface(self) -> ModuleTypeInterface:
        """Closed type metadata this module contributes to importers."""
        return self.type_env.module_interface()

    @property
    def environment_facts(self) -> EnvironmentFacts:
        """This module's own-facts journal, as its image persists it."""
        return self.type_env.own_facts()

    def image(self) -> CheckedModuleImage:
        """Build a data-only image for cache persistence and rehydration.

        Every image field is this module's member of the same name: the
        checked side tables directly, ``interface`` and ``environment_facts``
        through the type environment. The four members with no image field —
        ``resolved``, ``type_env``, ``import_env``, ``source_text`` — are
        recovered from the current ``ResolvedModule`` on rehydration. Only a
        journaled ``type_env`` (program path) has facts to persist.
        """
        image = object.__new__(CheckedModuleImage)
        image.__setstate__(_member_values(self, CheckedModuleImage.__match_args__))
        return image


def _assert_checked_types_closed(types: Iterable[Type], *, owner: str) -> None:
    """Reject solver-local types that escape a checked-output boundary."""
    if any(contains_inference_var(typ) for typ in types):
        raise AssertionError(f"inference variable leaked from checked output ({owner})")


def assert_checked_output_closed(
    *,
    node_types: Mapping[int, Type],
    contract_specs: Mapping[int, OutputContractSpec],
    target_contract_specs: Mapping[int, tuple[OutputContractSpec, ...]],
    call_sites: Iterable[CallSiteRecord],
    function_signatures: Mapping[str, FunctionSignature],
    cast_specs: Mapping[int, CastSpec],
    argument_bindings: ArgumentBindings,
    explicit_builtin_targets: Mapping[int, Type],
    owner: str,
) -> None:
    """Assert that all type-bearing checked metadata is concrete or rigid.

    Partial-call routing is deliberately absent: it is syntax-and-binding metadata
    only; its function type is published in ``node_types``.  Rigid declaration
    variables remain valid in generic templates and signatures, while flexible
    ``InferenceVarType`` instances are a compiler invariant failure here.  Purely
    a self-check: it does not seal the type environment (see
    :meth:`TypeEnvironment.seal`, which every caller runs unconditionally at the
    same point regardless of this check).
    """
    _assert_checked_types_closed(
        (
            *node_types.values(),
            *(spec.target_type for spec in contract_specs.values()),
            *(spec.target_type for specs in target_contract_specs.values() for spec in specs),
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
            *explicit_builtin_targets.values(),
        ),
        owner=owner,
    )


def assert_checked_module_output_closed(checked: CheckedModule) -> None:
    """Assert that one module's own checked output is safe to lower."""
    assert_checked_output_closed(
        node_types=checked.node_types,
        contract_specs=checked.contract_specs,
        target_contract_specs=checked.target_contract_specs,
        call_sites=checked.call_sites,
        function_signatures=checked.function_signatures,
        cast_specs=checked.cast_specs,
        argument_bindings=checked.argument_bindings,
        explicit_builtin_targets=checked.explicit_builtin_targets,
        owner=f"checked module {checked.module_id.path_str()}",
    )
    # The environment's own binding table is not one of the side tables above
    # (it is validated as a whole by ``assert_closed``, independent of which
    # published node/call/signature happens to reference each binding).
    checked.type_env.assert_closed()


def assert_checked_module_closed(checked: CheckedModule) -> None:
    """Assert that one module's checked output and the shared tables it reads are safe to lower."""
    assert_checked_module_output_closed(checked)
    checked.type_env.assert_shared_tables_closed()


# ---------------------------------------------------------------------------
# TypeEnvironment — mutable state during type checking
# ---------------------------------------------------------------------------


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class ModuleTypeInterface(_Record):
    """Closed declarations a compiled module contributes to its importers."""

    types: dict[DeclKey, Type]
    generics: dict[DeclKey, GenericTypeDef]
    aliases: dict[DeclKey, GenericAliasDef]
    definitions: tuple[TypeDef, ...]


class PublishedModuleSurface(Protocol):
    """Shared surface of ``CheckedModule`` and ``CheckedModuleImage``.

    Exposes a module's published interface, signatures, and static-binding
    types so the whole-program pre-passes (:mod:`agm.agl.typecheck.program`) —
    ``_build_program_type_table``, ``_build_program_func_sig_table``, and
    ``_build_program_static_binding_table`` — can read a reused module's
    closed type interface, published signatures, and published binding types
    whether it is a live ``CheckedModule`` or a rehydration-ready
    ``CheckedModuleImage``.
    """

    @property
    def interface(self) -> ModuleTypeInterface: ...

    @property
    def published_signatures(self) -> dict[int, FunctionSignatureRecord] | None: ...

    @property
    def published_binding_types(self) -> dict[int, Type] | None: ...


# ---------------------------------------------------------------------------
# EnvironmentFacts — TypeEnvironment's mutation journal
# ---------------------------------------------------------------------------


class EnvironmentFact(_Record):
    """One recorded ``TypeEnvironment`` mutator call, replayable onto another.

    ``_MUTATOR`` names the method the call went to, and a fact's fields are
    that method's arguments spelled with its parameter names, so replaying one
    is a keyword call. A fact whose replay is more than that forward overrides
    :meth:`apply`.
    """

    __slots__ = ()
    _MUTATOR: ClassVar[str]

    def apply(self, env: TypeEnvironment) -> None:
        """Replay this call on *env*."""
        mutator: Callable[..., None] = getattr(env, self._MUTATOR)
        mutator(**dict(zip(self.__match_args__, _member_values(self, self.__match_args__))))


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class BindingTypeFact(EnvironmentFact):
    """Journaled :meth:`TypeEnvironment.set_binding_type` call."""

    _MUTATOR = "set_binding_type"

    node_id: int
    typ: Type


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class FunctionSignatureFact(EnvironmentFact):
    """Journaled :meth:`TypeEnvironment.register_function_signature` call."""

    _MUTATOR = "register_function_signature"

    name: str
    sig: FunctionSignature
    scope_path: ScopePath


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class FunctionSignatureByNodeIdFact(EnvironmentFact):
    """Journaled :meth:`TypeEnvironment.register_function_signature_by_node_id` call."""

    _MUTATOR = "register_function_signature_by_node_id"

    node_id: int
    sig: FunctionSignature


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class ExternNodeIdFact(EnvironmentFact):
    """Journaled :meth:`TypeEnvironment.register_extern_node_id` call."""

    _MUTATOR = "register_extern_node_id"

    node_id: int


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class TypeFact(EnvironmentFact):
    """Journaled :meth:`TypeEnvironment.register_type` call."""

    _MUTATOR = "register_type"

    name: str
    typ: Type


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class GenericTypeFact(EnvironmentFact):
    """Journaled :meth:`TypeEnvironment.register_generic_type` call."""

    _MUTATOR = "register_generic_type"

    name: str
    gdef: GenericTypeDef


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class AliasFact(EnvironmentFact):
    """Journaled :meth:`TypeEnvironment.register_alias` call."""

    _MUTATOR = "register_alias"

    name: str
    target_expr: TypeExpr
    type_params: tuple[str, ...]

    def apply(self, env: TypeEnvironment) -> None:
        """Replay the registration, unless it is already in place.

        Structural ``==`` on syntax type nodes: skipping an identical
        registration keeps a frozen alias frozen.
        """
        if env.has_alias_registration(self.name, self.target_expr, self.type_params):
            return
        super().apply(env)


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class UnregisteredNameFact(EnvironmentFact):
    """Journaled :meth:`TypeEnvironment.unregister_name` call."""

    _MUTATOR = "unregister_name"

    name: str


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class FrozenAliasFact(EnvironmentFact):
    """Journaled :meth:`TypeEnvironment.freeze_alias` call."""

    _MUTATOR = "freeze_alias"

    name: str
    template: Type
    type_params: tuple[str, ...]


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class MethodHeaderFact(EnvironmentFact):
    """Journaled :meth:`TypeEnvironment.register_method_def` call."""

    _MUTATOR = "register_method_def"

    receiver: NominalOwner | str
    method: MethodDef


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class EnvironmentFacts(_Record):
    """Ordered journal of a ``TypeEnvironment``'s journaled mutator calls.

    Data only, so it serializes under the artifact allow-list.
    :meth:`TypeEnvironment.replay` applies each entry in order to reproduce
    the recorded mutations on another environment.

    Two windows record one: a module's header preparation
    (:mod:`agm.agl.typecheck.program`) and its authoritative body check,
    whose snapshot :meth:`TypeEnvironment.own_facts` publishes as part of a
    :class:`CheckedModuleImage`.
    """

    entries: tuple[EnvironmentFact, ...] = ()


# ---------------------------------------------------------------------------
# CheckedModuleImage — data-only CheckedModule for cache persistence
# ---------------------------------------------------------------------------


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class CheckedModuleImage(_Record):
    """Data-only image of a ``CheckedModule``, ready to persist and rehydrate.

    Every ``CheckedModule`` field except ``resolved``, ``type_env``,
    ``import_env``, and ``source_text`` — those are recovered from the
    current ``ResolvedModule`` on rehydration — plus ``interface`` (this
    module's closed type contribution) and ``environment_facts`` (its
    environment's own-facts journal). Built by :meth:`CheckedModule.image`;
    turned back into an equivalent ``CheckedModule`` by :meth:`rehydrate`.
    Both directions move a field by its name, so every field here names the
    ``CheckedModule`` member it mirrors.
    """

    node_types: dict[int, Type]
    contract_specs: dict[int, OutputContractSpec]
    call_sites: tuple[CallSiteRecord, ...]
    warnings: tuple[Diagnostic, ...]
    function_signatures: dict[str, FunctionSignature]
    cast_specs: dict[int, CastSpec]
    argument_bindings: ArgumentBindings
    pattern_classifications: dict[int, ConstructorRef | None]
    partial_calls: dict[int, PartialCallSpec]
    interface: ModuleTypeInterface
    environment_facts: EnvironmentFacts
    published_signatures: dict[int, FunctionSignatureRecord] | None
    published_binding_types: dict[int, Type] | None
    module_id: ModuleId
    slot_resolution: dict[int, BindingRef]
    slot_constructor_refs: dict[int, ConstructorRef]
    selected_constructor_refs: dict[int, ConstructorRef]
    pattern_binding_refs: dict[int, BindingRef]
    pattern_constructor_refs: dict[int, ConstructorRef]
    pattern_constructor_owners: dict[int, NominalId]
    method_selections: dict[int, MethodDef]
    explicit_builtin_targets: dict[int, Type]
    program_config_targets: dict[int, tuple[ModuleId, tuple[str, ...], str]]
    target_contract_specs: dict[int, tuple[OutputContractSpec, ...]]

    def rehydrate(self, resolved_module: ResolvedModule, env: TypeEnvironment) -> CheckedModule:
        """Reconstruct an equivalent ``CheckedModule`` over a freshly prepared *env*.

        *env* is a per-module environment the current compilation already
        prepared by the whole-program pre-passes
        (:mod:`agm.agl.typecheck.program`): its own program-wide headers and
        candidates are seeded, but no journal has started yet. Replaying this
        image's own facts reproduces exactly the mutations the authoritative
        body-check would have made on it, so ``env.own_facts()`` afterward
        equals ``environment_facts`` and a later :meth:`CheckedModule.image`
        of the rehydrated module is complete.
        """
        env.begin_facts()
        env.replay(self.environment_facts)
        env.seal()
        # The live members an image cannot carry; every other field of the
        # rehydrated module is this image's field of the same name.
        live: dict[str, object] = {
            "resolved": resolved_module.resolved,
            "type_env": env,
            "import_env": resolved_module.import_env,
            "source_text": resolved_module.source_text,
        }
        state: list[object] = []
        for name in CheckedModule.__match_args__:
            if name in live:
                state.append(live[name])
                continue
            retained: object = getattr(self, name)
            state.append(retained)
        module = object.__new__(CheckedModule)
        module.__setstate__(tuple(state))
        return module


@_pickles_by_name
@dataclass(frozen=True, slots=True)
class DeclaredHeaderSeed(_Record):
    """The declared-header tables every module environment in a program starts from.

    The whole-program function-signature pre-pass yields the same declared
    headers for every module, so they are collected once and copied into each
    environment instead of being re-registered per module.
    """

    binding_types: PersistentDict[int, Type]
    signatures: dict[str, FunctionSignature]
    signatures_by_node_id: dict[int, FunctionSignature]
    extern_node_ids: set[int]


class TypeEnvironment:
    """Mutable type environment used during the type-checking pass.

    Holds the type-declaration namespace (records, enums, exception types, and
    aliases) and a mapping from declaration ``node_id`` → ``Type`` for binding
    resolution during the check pass.

    The ``TypeEnvironment`` is populated by the ``_TypeBuilder`` pre-pass.
    Program checking seeds explicit headers first, then import-SCC candidate
    inference publishes closed unannotated signatures before the main
    ``_Checker`` visitor queries the environment.

    Program context
    ---------------
    When ``program_type_table``, ``import_env``, and ``module_id`` are supplied,
    the environment becomes module-aware:

    - ``program_type_table`` maps each ``DeclKey`` ``(ModuleId, scope_path,
      name)`` to the fully-built ``Type`` stamped with its owning
      ``module_id``.  Built once by the program pre-pass; shared (read-only)
      across all per-module envs.
    - ``program_generic_table`` and ``program_alias_table`` carry cross-module
      templates for applied nominal types and parameterized aliases; during
      program type-table construction, ``program_alias_keys`` and
      ``program_alias_resolver`` let transparent cross-module aliases resolve
      lazily before their sorted body-resolution turn.
    - ``import_env`` is the per-module :class:`~agm.agl.scope.imports.ImportEnv`
      produced by program scope resolution. Used to resolve qualified and
      import-tail-exposed type names.
    - ``module_id`` is the owning module of the current env.  ``::Name``
      (empty-segment qualifier) resolves against this module's own types.

    Every environment the checker or match compiler actually queries carries
    these fields. Absent program tables are empty: on the transient,
    shell-collection-only environments the whole-program type pre-pass builds
    in Step A (``typecheck/program.py::_build_program_type_table``) to register
    every module's declaration headers before any body is resolved, and on a
    REPL session's seed environment.
    """

    def __init__(
        self,
        *,
        program_type_table: Mapping[DeclKey, Type] | None = None,
        program_generic_table: Mapping[DeclKey, GenericTypeDef] | None = None,
        program_alias_table: Mapping[DeclKey, GenericAliasDef] | None = None,
        program_alias_keys: frozenset[DeclKey] | None = None,
        program_alias_resolver: Callable[[DeclKey, SourceSpan | None], Type | None] | None = None,
        import_env: ImportEnv | None = None,
        local_scope_paths: frozenset[ScopePath] = frozenset(),
        scope_nodes: Mapping[ScopePath, ScopeNode] | None = None,
        module_id: ModuleId = ENTRY_ID,
        type_table: TypeTable | None = None,
        declared_seed: DeclaredHeaderSeed | None = None,
    ) -> None:
        # Shared nominal type-declaration table (dual-write target alongside
        # ``_types``): defaults to a fresh table seeded with built-in prelude
        # defs; program context passes one shared instance across per-module envs.
        self._type_table: TypeTable = (
            type_table if type_table is not None else create_seeded_type_table()
        )
        # user-declared types (records, enums) — name → Type
        self._types: dict[str, Type] = {}
        # Alias syntax is retained for diagnostics and cycle detection, while the
        # resolved template preserves the nominal identities selected when the
        # alias was declared (including across REPL supersession).
        self._alias_targets: dict[str, TypeExpr] = {}
        self._resolved_aliases: dict[str, GenericAliasDef] = {}
        # Binding node_id → Type (populated as declarations are checked).
        self._binding_types: PersistentDict[int, Type] = (
            PersistentDict() if declared_seed is None else declared_seed.binding_types.fork()
        )
        # Root-scope function signatures by unqualified name: a compatibility
        # map for standalone callers that only know an unqualified spelling;
        # resolved calls use declaration ids.
        self._function_signatures: dict[str, FunctionSignature] = (
            {} if declared_seed is None else dict(declared_seed.signatures)
        )
        # Generic type definitions — name → GenericTypeDef.
        self._generic_types: dict[str, GenericTypeDef] = {}
        # Alias type-params — name → tuple of type-param names.
        self._alias_type_params: dict[str, tuple[str, ...]] = {}
        # Node-id-keyed function signatures — decl_node_id → FunctionSignature.
        # Program environments receive explicit headers from the whole-program
        # pre-pass, then closed unannotated candidates from import-SCC inference,
        # so _check_declared_name_call can look up the correct cross-module callee
        # by globally unique decl_node_id rather than by bare name (which would
        # collide when modules define different same-named functions).
        self._function_signatures_by_node_id: dict[int, FunctionSignature] = (
            {} if declared_seed is None else dict(declared_seed.signatures_by_node_id)
        )
        # Declaration node_ids of ``extern def``s, keyed by the same globally-unique
        # decl_node_id as ``_function_signatures_by_node_id``.  Populated by
        # ``_preregister_funcdef`` (this module's own externs) and by the program
        # function-signature pre-pass seeding (imported externs).  Consulted by
        # ``_check_declared_name_call`` to decide whether a declared-name call
        # site is an extern call site to record.
        self._extern_node_ids: set[int] = (
            set() if declared_seed is None else set(declared_seed.extern_node_ids)
        )
        # Cross-module nominal types; empty outside a module graph.
        self._program_type_table: Mapping[DeclKey, Type] = (
            {} if program_type_table is None else program_type_table
        )
        # Cross-module generic type definitions, for qualified generic constructor calls.
        self._program_generic_table: Mapping[DeclKey, GenericTypeDef] = (
            {} if program_generic_table is None else program_generic_table
        )
        # Cross-module parameterized type aliases.
        self._program_alias_table: Mapping[DeclKey, GenericAliasDef] = (
            {} if program_alias_table is None else program_alias_table
        )
        self._program_alias_keys: frozenset[DeclKey] = (
            program_alias_keys if program_alias_keys is not None else frozenset()
        )
        self._program_alias_resolver: Callable[[DeclKey, SourceSpan | None], Type | None] | None = (
            program_alias_resolver
        )
        self._import_env: ImportEnv | None = import_env
        self._module_id: ModuleId = module_id
        # Scope resolution supplies every local path, including regions with no
        # type declarations, so failed qualified type references retain their
        # scoped diagnostic instead of being mistaken for module routes.
        self._local_scope_paths = local_scope_paths
        # Scope's bare contribution layers are shared by value and type lookup.
        # They retain selection, renaming, provenance, and region boundaries.
        self._scope_nodes = scope_nodes if scope_nodes is not None else {}
        # Type references inside a named scope use the same lexical layers as
        # value references.  Scoped declarations remain stored under their
        # full path; this frame only maps a bare source spelling to that path.
        self._type_scope: tuple[str, ...] = ()
        self._sealed = False
        # Mutation journal, recording from begin_facts() until end_facts() takes
        # it; seal() leaves it in place, where no further mutator can reach it,
        # so own_facts() keeps answering from it. rewind_from,
        # restore_binding_types, remove_binding_types and seed_from never run
        # in either window; not journaled.
        # ``_resolve_name_type`` does write ``_resolved_aliases`` directly (a memo
        # re-derivable from ``_alias_targets``) on a query path, but every declared
        # alias is frozen during header preparation, so that write is unreachable
        # once the body-check window opens.
        self._journal: list[EnvironmentFact] = []
        self._journaling = False
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
        # Built-in prelude types (AgL: ExecResult, ParsePolicy) are always available.
        self._types.update(BUILTIN_PRELUDE_TYPES)

    def module_interface(self) -> ModuleTypeInterface:
        """Export this module's closed type metadata without another header pass."""

        def own[V](table: Mapping[DeclKey, V]) -> dict[DeclKey, V]:
            return {key: value for key, value in table.items() if key[0] == self._module_id}

        return ModuleTypeInterface(
            own(self._program_type_table),
            own(self._program_generic_table),
            own(self._program_alias_table),
            tuple(td for td in self._type_table.entries() if td.module_id == self._module_id),
        )

    @property
    def type_table(self) -> TypeTable:
        """The shared ``TypeTable`` populated alongside ``_types`` (dual-write)."""
        return self._type_table

    # --- Shared type-table registration ---

    def register_method_def(self, receiver: NominalOwner | str, method: MethodDef) -> None:
        """Publish a resolved method header in the shared table, journaled.

        A built-in receiver is named by its type constructor, a nominal one by
        its handle.
        """
        if isinstance(receiver, str):
            self._type_table.register_builtin_method(receiver, method)
        else:
            self._type_table.register_method(receiver, method)
        self._record_fact(MethodHeaderFact(receiver=receiver, method=method))

    def seal(self) -> None:
        """Freeze this environment as checked output.

        Validates the environment first when self-validation is enabled; sealing
        itself — the functional state that enables namespace memoization —
        always happens, regardless of the flag. No journaled mutator runs on a
        sealed environment, so what :meth:`own_facts` reports can no longer
        change.
        """
        if self_validation_enabled():
            self.assert_closed()
        self._sealed = True

    # --- Mutation journal ---

    def begin_facts(self) -> None:
        """Start recording this environment's mutator calls into a journal."""
        self._journal = []
        self._journaling = True

    def end_facts(self) -> EnvironmentFacts:
        """Close the active journal and return what it recorded.

        The header-preparation window's counterpart to :meth:`seal`, which
        closes the body-check window instead. Leaves the environment mutable
        and ready for a later window.
        """
        facts = EnvironmentFacts(tuple(self._journal))
        self._journal = []
        self._journaling = False
        return facts

    def own_facts(self) -> EnvironmentFacts:
        """Return a snapshot of this environment's own-facts journal.

        Empty for an environment that never journaled (module path, REPL seed).
        """
        return EnvironmentFacts(tuple(self._journal))

    def has_alias_registration(
        self, name: str, target_expr: TypeExpr, type_params: tuple[str, ...]
    ) -> bool:
        """Whether *name* is registered as an alias of *target_expr* with *type_params*."""
        return (
            name in self._alias_targets
            and self._alias_targets[name] == target_expr
            and self._alias_type_params.get(name, ()) == type_params
        )

    def replay(self, facts: EnvironmentFacts) -> None:
        """Apply *facts* in order through the journaled mutators.

        Recording is suspended while applying; an active journal is then
        extended with *facts* verbatim, so a skipped identical alias is kept.
        """
        journaling, self._journaling = self._journaling, False
        try:
            for fact in facts.entries:
                fact.apply(self)
        finally:
            self._journaling = journaling
        if journaling:
            self._journal.extend(facts.entries)

    def _record_fact(self, fact: EnvironmentFact) -> None:
        """Append a fact while the own-facts journal is active."""
        if self._journaling:
            self._journal.append(fact)

    # --- Type namespace queries ---

    def get_type(self, name: str) -> Type | None:
        return self._types.get(name)

    def nominal_by_declaration(self, module_id: ModuleId, scope_path: ScopePath) -> NominalOwner:
        """Return the handle of a non-generic nominal declaration scope resolved."""
        return self._type_table.named(module_id, scope_path[-1], scope_path[:-1]).handle()

    def has_qualified_import_member(self, qualifier: QualifierChain, name: str) -> bool:
        """Return whether a qualifier route contributes *name* after filtering."""
        if self._import_env is None:
            return False
        route = tuple(part for part in qualifier.segments[0].name.split("/"))
        atom_path = (*tuple(segment.name for segment in qualifier.segments[1:]), name)
        atom: NameAtom = atom_path[0] if len(atom_path) == 1 else atom_path
        return qualifier_contributes(self._import_env, route, atom, anchored=qualifier.anchored)

    def _ensure_qualified_type_route_unambiguous(
        self,
        qualifier: QualifierChain,
        name: str,
        selected_key: DeclKey,
        *,
        selected_route: Literal["type name", "use route"],
        span: SourceSpan | None,
    ) -> None:
        """Reject a qualified local/import collision unless both routes select one declaration."""
        if qualifier.anchor is not None or not self.has_qualified_import_member(qualifier, name):
            return
        imported_qname = self._try_resolve_import_qname(qualifier, name)
        if imported_qname is not None and self._qname_decl_key(imported_qname) == selected_key:
            return
        raise AglTypeError(
            f"Qualifier '{qualifier.render()}' is both a {selected_route} and a module route "
            f"for '{name}'. {qualification_repair_guidance()}",
            span=span,
        )

    def _import_route_member(
        self, qualifier: QualifierChain, name: str
    ) -> tuple[ImportEnv, tuple[str, ...], NameAtom]:
        """Split a module-route qualifier and type path for import resolution."""
        import_env = cast(ImportEnv, self._import_env)
        route = tuple(part for part in qualifier.segments[0].name.split("/"))
        atom_path = (*tuple(segment.name for segment in qualifier.segments[1:]), name)
        atom: NameAtom = atom_path[0] if len(atom_path) == 1 else atom_path
        return import_env, route, atom

    def _try_resolve_import_qname(self, qualifier: QualifierChain, name: str) -> QName | None:
        """Resolve a module route followed by one structured type path, if it resolves."""
        import_env, route, atom = self._import_route_member(qualifier, name)
        return try_resolve_qualified_member(import_env, route, atom, anchored=qualifier.anchored)

    def _resolve_import_qname(
        self, qualifier: QualifierChain, name: str, *, span: SourceSpan | None
    ) -> QName:
        """Resolve a module route followed by one structured type path."""
        import_env, route, atom = self._import_route_member(qualifier, name)
        return resolve_qualified_member(
            import_env,
            route,
            atom,
            anchored=qualifier.anchored,
            unknown_qualifier=lambda rendered: AglTypeError(
                f"Unknown module qualifier '{rendered}::'.", span=span
            ),
            missing_member=lambda rendered: AglTypeError(
                f"Type '{name}' is not accessible via qualifier '{rendered}::'.", span=span
            ),
            ambiguous=lambda message: AglTypeError(message, span=span),
        )

    def resolve_owner_applied_inline_member_type(
        self,
        qualifier: QualifierChain,
        member: str,
        *,
        type_vars: frozenset[str],
        span: SourceSpan | None,
    ) -> RecordType | None:
        """Resolve a member type whose enum owner is applied; a parameterized alias must be."""
        selected = self.select_owner_inline_member(
            qualifier, member, type_vars=type_vars, span=span
        )
        if selected is None:
            return None
        if selected.type_params:
            owner = qualifier.segments[-1].name
            raise AglTypeError(
                f"Parameterized alias '{owner}' requires "
                f"{len(selected.type_params)} type argument(s); "
                f"use '{owner}[...]::{member}' to select its member.",
                span=span,
            )
        return selected.member

    def select_owner_inline_member(
        self,
        qualifier: QualifierChain,
        member: str,
        *,
        type_vars: frozenset[str],
        span: SourceSpan | None,
    ) -> OwnerMember | None:
        """Select an inline enum member from the applied enum its qualifier names.

        ``Source[T]::Member`` applies ``T`` to ``Source``.  An inline member
        captures only the owner parameters used by its fields, so selecting it
        from the instantiated enum yields its concrete record handle directly.
        An alias of an enum is its target, so ``Alias::Member`` selects the
        member of the alias's template, quantified over the alias's own type
        parameters. ``None`` when the qualifier names no such owner.
        """
        if not qualifier.segments:
            return None
        owner_expr = owner_type_expr(qualifier)
        if isinstance(owner_expr, NameT):
            alias = self._enum_alias_template(owner_expr, span)
            if alias is None:
                return None
            enum_template, type_params = alias
            selected = self.type_table.inline_member(enum_template, member)
            return None if selected is None else OwnerMember(selected, type_params)
        owner = self.resolve_type_expr(owner_expr, span=span, type_vars=type_vars)
        if not isinstance(owner, EnumType):
            raise AglTypeError(f"'{owner_expr.name}' is not a generic enum type.", span=span)
        selected = self.type_table.inline_member(owner, member)
        return None if selected is None else OwnerMember(selected, ())

    def _enum_alias_template(
        self, owner: NameT, span: SourceSpan | None
    ) -> tuple[EnumType, tuple[str, ...]] | None:
        """Return the enum template and parameters of the alias *owner* names, if any."""
        key = self.type_name_declaration(owner, span=span)
        if key is None:
            return None
        module_id, scope_path, name = key
        self._ensure_program_alias_resolved(key, span)
        template = self.source_type_template_qname(module_id, name, scope_path=scope_path)
        if template is None:
            return None
        enum_type = template.template
        if not isinstance(enum_type, EnumType) or (
            (enum_type.module_id, enum_type.scope_path, enum_type.name) == key
        ):
            return None
        return enum_type, template.type_params

    def register_type(self, name: str, typ: Type) -> None:
        self._types[name] = typ
        self._record_fact(TypeFact(name=name, typ=typ))

    def unregister_name(self, name: str) -> None:
        """Remove a user *name* from the tables that only ever expose ONE definition.

        Used by the type-builder before registering a redeclared name —
        whether it redeclares under a different kind (e.g. a seeded
        ``record R`` redefined as ``type R = int``) or a different shape
        (e.g. ``record R`` redefined with different fields). ``_types``,
        alias targets/params, and generic templates are each keyed by NAME
        alone with exactly one slot per name path, and each answers a "what
        does this name mean right now" question — the newest declaration's
        answer, and nothing else, must ever be reachable through them. A
        redeclaration that does not itself repopulate a slot (e.g. a
        previously generic ``Box`` redeclared as a plain record never writes
        ``_generic_types["Box"]`` again) would otherwise leave the superseded
        declaration's answer live in a table the new one never touches, and
        make ``get_type`` disagree with annotation/constructor resolution.
        Dropping the name from these namespaces before the new declaration is
        registered keeps them consistent with that single-newest-answer rule.

        The shared ``type_table`` is likewise NOT touched here: it is keyed
        by declaration identity, not name, and its own ``register`` call
        retains a superseded declaration under its own identity while
        repointing the name index at the newest one — exactly the behavior
        this method's callers rely on for the identity-keyed table.

        Built-in exception names and built-in prelude type names are never
        removed: they are non-shadowable (rejected earlier by
        ``_BUILTIN_TYPE_NAMES``), so the builder never calls this for them,
        but the guard makes the helper safe to call defensively.
        """
        if name in _BUILTIN_FALLBACK_TYPE_NAMES:
            return
        self._types.pop(name, None)
        self._alias_targets.pop(name, None)
        self._resolved_aliases.pop(name, None)
        self._generic_types.pop(name, None)
        self._alias_type_params.pop(name, None)
        self._record_fact(UnregisteredNameFact(name=name))

    def register_alias(
        self, name: str, target_expr: TypeExpr, *, type_params: tuple[str, ...] = ()
    ) -> None:
        """Store the raw TypeExpr for *name*; resolved lazily by resolve_type_expr.

        ``type_params`` must be provided for parameterized type aliases (e.g.
        ``type Wrapper[T] = array[T]``); defaults to ``()`` for plain aliases.
        """
        self._alias_targets[name] = target_expr
        self._resolved_aliases.pop(name, None)
        self._alias_type_params[name] = type_params
        self._record_fact(AliasFact(name=name, target_expr=target_expr, type_params=type_params))

    def freeze_alias(self, name: str, template: Type, *, type_params: tuple[str, ...] = ()) -> None:
        """Preserve an alias's resolved template under its declaring identities."""
        self._resolved_aliases[name] = GenericAliasDef(type_params=type_params, template=template)
        self._record_fact(FrozenAliasFact(name=name, template=template, type_params=type_params))

    # --- Generic type registry ---

    def register_generic_type(self, name: str, gdef: GenericTypeDef) -> None:
        """Register a generic type definition under *name*."""
        self._generic_types[name] = gdef
        self._record_fact(GenericTypeFact(name=name, gdef=gdef))

    def get_generic_type(self, name: str) -> GenericTypeDef | None:
        """Return the ``GenericTypeDef`` for *name*, or ``None`` if unknown."""
        return self._generic_types.get(name)

    def generic_type_of(self, name: str) -> GenericTypeDef:
        """Return the ``GenericTypeDef`` of a generic declaration this module registered."""
        return self._generic_types[name]

    def get_generic_type_by_declaration(
        self, module_id: ModuleId, scope_path: ScopePath
    ) -> GenericTypeDef | None:
        """Look up a generic declaration by its module and complete scope path."""
        name = "::".join(scope_path)
        if module_id == self._module_id:
            local = self._generic_types.get(name)
            if local is not None:
                return local
        return self._program_generic_table.get((module_id, scope_path[:-1], scope_path[-1]))

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
        # registered TypeDef's templates on demand). decl_id carries over
        # unchanged: instantiating at different type arguments still names
        # the same declaration.
        template = gdef.template
        if isinstance(template, RecordType):
            return RecordType(
                name=template.name,
                type_args=args,
                module_id=template.module_id,
                scope_path=template.scope_path,
                decl_id=template.decl_id,
            )
        return EnumType(
            name=template.name,
            type_args=args,
            module_id=template.module_id,
            scope_path=template.scope_path,
            decl_id=template.decl_id,
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

    def resolve_named_type(self, name: str, *, span: SourceSpan | None = None) -> Type | None:
        """Resolve a type *name* alias-transparently to a semantic ``Type``.

        Returns the resolved ``Type`` for a record/enum/exception name or an
        alias chain (multi-hop, alias-of-alias) that bottoms out in a named
        type; ``None`` if the name is unknown or names a non-nominal alias
        target (e.g. an alias of ``array[int]``, which has no single name).
        Used for alias-transparent qualifier resolution in qualified
        constructors, patterns and ``is`` tests: a generic type or
        parameterized alias resolves to its template over its own parameters.

        In program context, also searches types exposed bare by ``use`` declarations
        and import tails when the name is not found locally.

        An ambiguity complaint about a name contributed by several routes
        propagates: it names the problem better than the "unknown type" the
        caller would otherwise report, and callers pass *span* so it lands on
        the reference.
        """
        local_name = self._lexical_type_name(name)
        if not (
            local_name in self._generic_types
            or local_name in self._types
            or local_name in self._alias_targets
        ):
            # Bare contributions from a root ``use`` and import tails share one
            # resolution rank. A contribution from a nearer named region still
            # shadows the module-root rank.
            resolved = self._bare_type_key(name, span)
            if resolved is None:
                return None
            key = resolved[0]
            own_alias = self._own_alias_name_for_key(key)
            if own_alias is None:
                return self._program_named_type(key, name, span)
            local_name = own_alias
        if local_name in self._generic_types:
            return self._generic_types[local_name].template
        if self._alias_type_params.get(local_name):
            return self._own_alias_template(local_name).template
        try:
            return self._resolve_name_type(
                local_name, span=span, _resolving=frozenset(), lexical=False
            )
        except AglTypeError:
            return None

    def _program_named_type(self, key: DeclKey, name: str, span: SourceSpan | None) -> Type | None:
        """Resolve a contributed declaration, not an own alias; a generic one to its template."""
        generic = self._program_generic_table.get(key)
        if generic is not None:
            return generic.template
        self._ensure_program_alias_resolved(key, span)
        alias = self._program_alias_table.get(key)
        if alias is not None:
            return alias.template
        return self._resolve_type_key_as_bare(key, name, span=span)

    def type_name_declaration(
        self, type_expr: NameT | AppliedT, *, span: SourceSpan | None = None
    ) -> DeclKey | None:
        """Return the declaration identity an annotation's type name selects.

        Follows resolution's selection order without resolving, so a host can
        see through the transparent aliases resolution erases: this module's
        own type namespace, then scope-use and import contributions, then a
        qualified name's module route. ``None`` when no route selects one, as
        for a builtin alias target no declaration reaches or a name that is
        not a type. A name several routes contribute raises at *span*.
        """
        selected = type_name_selection(self._type_name_site(), type_expr)
        return self._unique_bare_type_key(
            type_expr.name, {self._qname_decl_key(qname) for qname in selected}, span
        )

    @staticmethod
    def _qname_decl_key(qname: QName) -> DeclKey:
        atom = qname[1]
        path = (atom,) if isinstance(atom, str) else atom
        return (qname[0], path[:-1], path[-1])

    def _is_program_type_candidate(self, qname: QName) -> bool:
        """Return whether a program-qualified name denotes any type-namespace declaration."""
        key = self._qname_decl_key(qname)
        return self._in_program_type_tables(key) or key in self._program_alias_keys

    def _ensure_program_alias_resolved(self, key: DeclKey, span: SourceSpan | None) -> Type | None:
        """Resolve a program alias lazily while retaining its declaration path."""
        if key not in self._program_alias_keys or self._program_alias_resolver is None:
            return None
        return self._program_alias_resolver(key, span)

    def _resolve_program_qname_as_bare_type(
        self, qname: QName, exposed_name: str, *, span: SourceSpan | None
    ) -> Type | None:
        """Resolve a program-qualified name used as an unapplied type expression."""
        key = self._qname_decl_key(qname)
        typ = self._program_type_table.get(key)
        if typ is not None:
            return typ
        typ = self._ensure_program_alias_resolved(key, span)
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
        return None

    # --- Function signature table ---

    def register_function_signature(
        self, name: str, sig: FunctionSignature, *, scope_path: ScopePath = ()
    ) -> None:
        """Register a root-scope signature under its unqualified name.

        Non-root registrations (``scope_path`` set) record the fact for replay
        but leave the compatibility map alone; resolved calls to a scoped
        function use its declaration id instead.
        """
        if not scope_path:
            self._function_signatures[name] = sig
        self._record_fact(FunctionSignatureFact(name=name, sig=sig, scope_path=scope_path))

    def all_function_signatures(self) -> dict[str, FunctionSignature]:
        return dict(self._function_signatures)

    def register_function_signature_by_node_id(self, node_id: int, sig: FunctionSignature) -> None:
        """Register a function signature keyed by its declaration ``node_id``.

        The program pre-pass seeds explicit signatures into every module env;
        import-SCC inference later publishes each closed candidate into the
        environments that need it. Because ``node_id`` is globally unique,
        signatures from different modules never collide even when their
        functions have the same name.
        """
        self._function_signatures_by_node_id[node_id] = sig
        self._record_fact(FunctionSignatureByNodeIdFact(node_id=node_id, sig=sig))

    def get_function_signature_by_node_id(self, node_id: int) -> FunctionSignature | None:
        """Return the function signature for a callee's declaration ``node_id``.

        Used by ``_check_declared_name_call`` to look up the correct signature
        for a callee by its globally-unique declaration node id, avoiding the
        name collision problem when two modules define same-named functions with
        different signatures.

        In program context, explicit signatures arrive from the whole-program
        pre-pass and closed unannotated candidates arrive from import-SCC
        inference (both via :meth:`register_function_signature_by_node_id`).
        In single-program mode, ``_preregister_funcdef`` seeds both the
        name-keyed and node-id-keyed tables. Returns ``None`` only for
        syntactically impossible cases (for example, a callee declaration that
        failed before registration).
        """
        return self._function_signatures_by_node_id.get(node_id)

    def function_signature_of(self, node_id: int) -> FunctionSignature:
        """Return the signature of a function declaration a checked program resolved to."""
        return self._function_signatures_by_node_id[node_id]

    def register_extern_node_id(self, node_id: int) -> None:
        """Mark a function declaration ``node_id`` as an ``extern def``.

        Consulted by ``_check_declared_name_call`` so that direct calls to an
        extern (own-module or imported) are recorded as dry-run call sites the
        same way ``ask``/``exec`` calls are.
        """
        self._extern_node_ids.add(node_id)
        self._record_fact(ExternNodeIdFact(node_id=node_id))

    def is_extern_node_id(self, node_id: int) -> bool:
        """Return ``True`` if *node_id* names a declared ``extern def``."""
        return node_id in self._extern_node_ids

    # --- Binding type table ---

    def set_binding_type(self, node_id: int, typ: Type) -> None:
        self._binding_types[node_id] = typ
        self._record_fact(BindingTypeFact(node_id=node_id, typ=typ))

    def snapshot_binding_types(self) -> PersistentDict[int, Type]:
        """Return a restorable snapshot of transient binding-type metadata."""
        return self._binding_types.fork()

    def restore_binding_types(self, snapshot: PersistentDict[int, Type]) -> None:
        """Restore binding metadata after a disposable checking session."""
        self._binding_types = snapshot

    def get_binding_type(self, node_id: int) -> Type | None:
        return self._binding_types.get(node_id)

    def binding_type_of(self, node_id: int) -> Type:
        """Return the type of a binding declaration a checked program resolved to."""
        return self._binding_types[node_id]

    def remove_binding_types(self, node_ids: Iterable[int]) -> None:
        """Forget binding-type metadata for the given declaration node ids."""
        for node_id in node_ids:
            self._binding_types.pop(node_id, None)
            self._function_signatures_by_node_id.pop(node_id, None)
            self._extern_node_ids.discard(node_id)

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
        types: list[Type] = [
            *(alias.template for alias in self._program_alias_table.values()),
            *(generic.template for generic in self._program_generic_table.values()),
            *(
                field_type
                for typedef in self._type_table.entries()
                for _, field_type in typedef.fields
            ),
        ]
        if any(contains_inference_var(typ) for typ in types):
            raise AssertionError("inference variable leaked into a persistent type environment")

    def resolve_binding(self, ref: BindingRef) -> Type | None:
        """Return the declared type for a ``BindingRef``."""
        return self._binding_types.get(ref.decl_node_id)

    # --- Type expression resolution ---

    @contextmanager
    def type_scope(self, path: tuple[str, ...]) -> Iterator[None]:
        """Resolve type expressions with *path* as their lexical scope layer."""
        previous = self._type_scope
        self._type_scope = path
        try:
            yield
        finally:
            self._type_scope = previous

    def _unique_bare_type_key(
        self, name: NameAtom, keys: set[DeclKey], span: SourceSpan | None
    ) -> DeclKey | None:
        """Select one declaration identity after deduplicating contribution routes."""
        if not keys:
            return None
        if len(keys) == 1:
            return next(iter(keys))
        labels = ", ".join(
            sorted(
                spell_declaration(module, (*path, declared_name), local_to=self._module_id)
                for module, path, declared_name in keys
            )
        )
        raise AglTypeError(
            f"Ambiguous type '{_render_type_atom(name)}': contributed by multiple routes "
            f"({labels}). Use a qualified reference to disambiguate.",
            span=span,
        )

    def _type_name_site(self) -> TypeNameSite:
        """Return the site of a type name written in the current type scope."""
        return TypeNameSite(
            module_id=self._module_id,
            scope_path=self._type_scope,
            import_env=EMPTY_IMPORT_ENV if self._import_env is None else self._import_env,
            declares=lambda path: self._has_own_type_name("::".join(path)),
            contributions=self._type_contributions,
            is_type=self._is_program_type_candidate,
        )

    def _type_contributions(self, name: NameAtom) -> tuple[ScopePath, frozenset[QName]] | None:
        """Return the nearest region contributing type *name* and what it contributes."""
        scope = self._scope_nodes.get(self._type_scope)
        if scope is None:
            return None
        resolved = resolve_bare_contribution_layer(
            scope, name, predicate=self._is_type_contribution
        )
        return None if resolved is None else contributed_declarations(*resolved)

    def _opened_type_key(self, name: NameAtom, span: SourceSpan | None) -> DeclKey | None:
        """Return the unique type declaration contributed to this type region."""
        resolved = self._type_contributions(name)
        qnames = frozenset() if resolved is None else resolved[1]
        return self._unique_bare_type_key(
            name, {self._qname_decl_key(qname) for qname in qnames}, span
        )

    def _bare_type_key(self, name: str, span: SourceSpan | None) -> tuple[DeclKey, bool] | None:
        """Resolve a bare type across equally ranked root use and import routes.

        Reports alongside the identity whether it came from the nearest
        region's own contributions, which callers need to choose between an
        unapplied and a bare resolution. One selection answers both.
        """
        selected, contributed = bare_type_selection(self._type_name_site(), name)
        key = self._unique_bare_type_key(
            name, {self._qname_decl_key(qname) for qname in selected}, span
        )
        return (
            None
            if key is None
            else (key, key in {self._qname_decl_key(qname) for qname in contributed})
        )

    def _is_type_contribution(self, ref: BindingRef) -> bool:
        """Whether a shared bare contribution refers to a type declaration."""
        key = (ref.module_id, ref.scope_path, ref.name)
        if key in self._program_type_table or key in self._program_generic_table:
            return True
        if key in self._program_alias_keys or key in self._program_alias_table:
            return True
        local_name = "::".join((*ref.scope_path, ref.name))
        return ref.module_id == self._module_id and (
            local_name in self._types
            or local_name in self._generic_types
            or local_name in self._alias_targets
        )

    def _own_alias_name_for_key(self, key: DeclKey) -> str | None:
        """Return the root-stored name for an own-module alias identity."""
        module, path, source_name = key
        local_name = "::".join((*path, source_name))
        if module == self._module_id and local_name in self._alias_targets:
            return local_name
        return None

    def _resolve_type_key_as_bare(
        self, key: DeclKey, exposed_name: str, *, span: SourceSpan | None
    ) -> Type | None:
        """Resolve one deduplicated declaration identity as a bare type."""
        module, path, source_name = key
        qname: QName = (module, source_name if not path else (*path, source_name))
        resolved = self._resolve_program_qname_as_bare_type(qname, exposed_name, span=span)
        if resolved is not None:
            return resolved
        alias_name = self._own_alias_name_for_key(key)
        if alias_name is None:
            return None
        return self._resolve_name_type(alias_name, span=span, _resolving=frozenset(), lexical=False)

    def _resolve_type_key_unapplied(
        self, key: DeclKey, name: NameAtom, span: SourceSpan | None
    ) -> Type:
        """Resolve one region-contributed declaration identity, rejecting a bare generic."""
        generic = self._program_generic_table.get(key)
        if generic is not None:
            rendered = _render_type_atom(name)
            raise AglTypeError(
                f"Generic type '{rendered}' requires {len(generic.type_params)} type argument(s); "
                f"use '{rendered}[...]' to apply it.",
                span=span,
            )
        # A region-contributed key names a declared type.
        return cast(Type, self._resolve_type_key_as_bare(key, _render_type_atom(name), span=span))

    def _resolve_bare_type(self, name: str, span: SourceSpan | None) -> Type | None:
        """Resolve a bare type across root uses and import tails at the same rank."""
        resolved = self._bare_type_key(name, span)
        if resolved is None:
            return None
        key, from_region = resolved
        if from_region:
            return self._resolve_type_key_unapplied(key, name, span)
        return self._resolve_type_key_as_bare(key, name, span=span)

    def _resolve_applied_type_key(
        self,
        key: DeclKey,
        name: NameAtom,
        args: tuple[Type, ...],
        span: SourceSpan | None,
    ) -> Type:
        """Apply arguments to one deduplicated bare type declaration identity."""
        generic = self._program_generic_table.get(key)
        if generic is not None:
            return self.instantiate_from_gdef(key[2], generic, args, span=span)
        self._ensure_program_alias_resolved(key, span)
        alias = self._program_alias_table.get(key)
        if alias is not None:
            return self.instantiate_alias(key[2], alias, args, span=span)
        alias_name = self._own_alias_name_for_key(key)
        if alias_name is not None:
            return self._instantiate_local_alias(
                alias_name, self._alias_targets[alias_name], args, span
            )
        raise AglTypeError(
            f"Type '{_render_type_atom(name)}' does not take type arguments.", span=span
        )

    def _resolve_local_applied_alias(
        self,
        name: str,
        args: tuple[Type, ...],
        span: SourceSpan | None,
        *,
        body_span: SourceSpan | None = None,
        resolving: frozenset[str] = frozenset(),
        type_vars: frozenset[str] = frozenset(),
    ) -> Type | None:
        """Resolve and instantiate an alias stored in the root type environment."""
        alias_expr = self._alias_targets.get(name)
        if alias_expr is None:
            return None
        return self._instantiate_local_alias(
            name,
            alias_expr,
            args,
            span,
            body_span=body_span,
            resolving=resolving,
            type_vars=type_vars,
        )

    def _instantiate_local_alias(
        self,
        name: str,
        alias_expr: TypeExpr,
        args: tuple[Type, ...],
        span: SourceSpan | None,
        *,
        body_span: SourceSpan | None = None,
        resolving: frozenset[str] = frozenset(),
        type_vars: frozenset[str] = frozenset(),
    ) -> Type:
        """Instantiate the root-stored alias *name*, whose target is *alias_expr*."""
        if name in resolving:
            raise AglTypeError(f"Type alias '{name}' is part of a cycle.", span=span)
        alias_params = self._alias_type_params.get(name, ())
        if len(args) != len(alias_params):
            raise AglTypeError(
                f"Alias '{name}' requires {len(alias_params)} type argument(s), got {len(args)}.",
                span=span,
            )
        with self.type_scope(_split_scoped_type_name(name)[0]):
            body_type = self.resolve_type_expr(
                alias_expr,
                span=body_span,
                _resolving=resolving | {name},
                type_vars=type_vars | frozenset(alias_params),
            )
        return self.instantiate_alias(
            name,
            GenericAliasDef(type_params=alias_params, template=body_type),
            args,
            span=span,
        )

    def _resolve_bare_applied_type(
        self, name: str, args: tuple[Type, ...], span: SourceSpan | None
    ) -> Type | None:
        """Resolve a bare generic across root uses and import tails at the same rank."""
        resolved = self._bare_type_key(name, span)
        if resolved is None:
            return None
        return self._resolve_applied_type_key(resolved[0], name, args, span)

    def _lexical_type_name(self, name: str) -> str:
        """Return the nearest scoped spelling of an unqualified type name."""
        for end in range(len(self._type_scope), 0, -1):
            candidate = "::".join((*self._type_scope[:end], name))
            if (
                candidate in self._types
                or candidate in self._generic_types
                or candidate in self._alias_targets
                or (self._module_id, (), candidate) in self._program_type_table
                or (self._module_id, (), candidate) in self._program_generic_table
                or (self._module_id, (), candidate) in self._program_alias_table
            ):
                return candidate
        return name

    def _has_own_type_name(self, name: str) -> bool:
        """Whether *name* is declared by this module in any type namespace."""
        return (
            name in self._types
            or name in self._generic_types
            or name in self._alias_targets
            or (self._module_id, (), name) in self._program_type_table
            or (self._module_id, (), name) in self._program_generic_table
            or (self._module_id, (), name) in self._program_alias_table
        )

    def _local_qualified_type_name(self, qualifier: QualifierChain, name: str) -> str | None:
        """Find the local type selected by *qualifier* without routing imports.

        ``::`` starts at this module's root, ``/`` always denotes an import
        route, and an unanchored qualifier walks the active lexical layers.
        Keeping this selection separate from route resolution prevents a type
        lookup from inheriting a stale lexical context or treating a module
        anchor as a local path.
        """
        if qualifier.anchor is QualifierAnchor.MODULE:
            return None
        bases = enclosing_scope_bases(
            self._type_scope, rooted=qualifier.anchor is QualifierAnchor.CURRENT_MODULE
        )
        for base in bases:
            candidate = "::".join((*base, *qualifier.route_segments, name))
            if self._has_own_type_name(candidate):
                return candidate
        return None

    def _has_local_scope_prefix(self, qualifier: QualifierChain) -> bool:
        """Whether a local scope begins the qualifier in an active lexical layer."""
        if qualifier.anchor is QualifierAnchor.MODULE:
            return False
        bases = enclosing_scope_bases(
            self._type_scope, rooted=qualifier.anchor is QualifierAnchor.CURRENT_MODULE
        )
        for base in bases:
            path = (*base, *qualifier.route_segments)
            if any(
                len(scope_path) > len(base)
                and len(scope_path) <= len(path)
                and path[: len(scope_path)] == scope_path
                for scope_path in self._local_scope_paths
            ):
                return True
        return False

    def _is_missing_local_scoped_type(self, qualifier: QualifierChain, name: str) -> bool:
        """Whether an unresolved qualifier belongs to a local scope, not a route."""
        if qualifier.anchor is QualifierAnchor.CURRENT_MODULE:
            return True
        has_import_route = self._import_env is not None and qualifier_candidates(
            self._import_env, qualifier.route_segments, anchored=qualifier.anchored
        )
        has_bare_import = (
            self._import_env is not None
            and not qualifier.anchored
            and to_bare_atom((*qualifier.route_segments, name)) in self._import_env.unqualified
        )
        return self._has_local_scope_prefix(qualifier) and not (has_import_route or has_bare_import)

    @staticmethod
    def _unknown_scoped_type_message(qualifier: QualifierChain, name: str) -> str:
        return f"Unknown scoped type '{'::'.join((*qualifier.route_segments, name))}'."

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
            AppliedT,
            ArrayT,
            BoolT,
            DecimalT,
            DictT,
            FuncT,
            IntT,
            JsonT,
            NameT,
            TextT,
            UnitT,
        )

        if _resolving is None:
            _resolving = frozenset()

        if isinstance(type_expr, (NameT, AppliedT)) and type_expr.qualifier is not None:
            for segment in type_expr.qualifier.segments:
                for type_arg in segment.type_args or ():
                    self.resolve_type_expr(type_arg, _resolving=_resolving, type_vars=type_vars)

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
        if isinstance(type_expr, FuncT):
            params = tuple(
                self.resolve_type_expr(p, _resolving=_resolving, type_vars=type_vars)
                for p in type_expr.params
            )
            result = self.resolve_type_expr(
                type_expr.result, _resolving=_resolving, type_vars=type_vars
            )
            return FunctionType(params=params, result=result)
        if isinstance(type_expr, ArrayT):
            elem = self.resolve_type_expr(
                type_expr.elem, _resolving=_resolving, type_vars=type_vars
            )
            return ArrayType(elem=elem)
        if isinstance(type_expr, DictT):
            key = self.resolve_type_expr(type_expr.key, _resolving=_resolving, type_vars=type_vars)
            val = self.resolve_type_expr(
                type_expr.value, _resolving=_resolving, type_vars=type_vars
            )
            return DictType(key=key, value=val)
        if isinstance(type_expr, NameT):
            eff_span = span if span is not None else type_expr.span
            if type_expr.qualifier is not None:
                owner_member = self.resolve_owner_applied_inline_member_type(
                    type_expr.qualifier,
                    type_expr.name,
                    type_vars=type_vars,
                    span=eff_span,
                )
                if owner_member is not None:
                    return owner_member
                return self.resolve_qualified_name_type(
                    type_expr.qualifier, type_expr.name, span=eff_span
                )
            return self._resolve_name_type(
                type_expr.name,
                span=eff_span,
                _resolving=_resolving,
                type_vars=type_vars,
            )
        if isinstance(type_expr, AppliedT):
            name = (
                self._lexical_type_name(type_expr.name)
                if type_expr.qualifier is None
                else type_expr.name
            )
            eff_span = span if span is not None else type_expr.span
            qualifier = type_expr.qualifier
            rendered_owner = "" if qualifier is None else qualifier.render()
            owner_member = (
                None
                if qualifier is None
                else self.resolve_owner_applied_inline_member_type(
                    qualifier, name, type_vars=type_vars, span=eff_span
                )
            )
            if qualifier is not None:
                local_name = self._local_qualified_type_name(qualifier, name)
                if local_name is not None:
                    local_path, declared_name = _split_scoped_type_name(local_name)
                    self._ensure_qualified_type_route_unambiguous(
                        qualifier,
                        name,
                        (self._module_id, local_path, declared_name),
                        selected_route="type name",
                        span=eff_span,
                    )
                    name = local_name
                    qualifier = None
            resolved_args = tuple(
                self.resolve_type_expr(a, span=None, _resolving=_resolving, type_vars=type_vars)
                for a in type_expr.args
            )
            if owner_member is not None:
                raise AglTypeError(
                    f"Type '{rendered_owner}::{type_expr.name}' does not take type arguments.",
                    span=eff_span,
                )
            if qualifier is not None and qualifier.anchor is None:
                opened_atom = to_bare_atom(
                    (*tuple(segment.name for segment in qualifier.segments), type_expr.name)
                )
                opened_key = self._opened_type_key(opened_atom, eff_span)
                if opened_key is not None:
                    opened = self._resolve_applied_type_key(
                        opened_key, opened_atom, resolved_args, eff_span
                    )
                    self._ensure_qualified_type_route_unambiguous(
                        qualifier,
                        name,
                        opened_key,
                        selected_route="use route",
                        span=eff_span,
                    )
                    return opened
            if qualifier is not None and qualifier.route_segments:
                return self._resolve_qualified_applied_type(
                    qualifier, name, resolved_args, span=eff_span
                )
            gdef = self._generic_types.get(name)
            if gdef is not None:
                return self.instantiate_nominal(name, resolved_args, span=eff_span)
            alias_def = self._resolved_aliases.get(name)
            if alias_def is not None:
                return self.instantiate_alias(name, alias_def, resolved_args, span=eff_span)
            local_alias = self._resolve_local_applied_alias(
                name,
                resolved_args,
                eff_span,
                body_span=span,
                resolving=_resolving,
                type_vars=type_vars,
            )
            if local_alias is not None:
                return local_alias
            program_alias_def = self._program_alias_table.get((self._module_id, (), name))
            if program_alias_def is not None:
                return self.instantiate_alias(name, program_alias_def, resolved_args, span=eff_span)
            if name in self._types:
                raise AglTypeError(
                    f"Type '{name}' does not take type arguments.",
                    span=eff_span,
                )
            if qualifier is None:
                bare = self._resolve_bare_applied_type(type_expr.name, resolved_args, span=eff_span)
                if bare is not None:
                    return bare
            raise AglTypeError(
                f"Unknown type '{name}'.",
                span=eff_span,
            )
        raise AglTypeError(
            f"Unknown type expression: {type_expr!r}",
            span=span,
        )

    def _resolve_qualified_applied_type(
        self,
        qualifier: QualifierChain,
        name: str,
        args: tuple[Type, ...],
        *,
        span: SourceSpan | None,
    ) -> Type:
        """Resolve ``module::Name[args]`` through the module import environment."""
        rendered = qualifier.render()
        if self._is_missing_local_scoped_type(qualifier, name):
            raise AglTypeError(self._unknown_scoped_type_message(qualifier, name), span=span)
        if self._import_env is None:
            raise AglTypeError(
                f"Module qualifier '{rendered}::' cannot be resolved outside of a module graph.",
                span=span,
            )
        qname = self._resolve_import_qname(qualifier, name, span=span)
        key = self._qname_decl_key(qname)
        source_name = key[2]
        gdef = self._program_generic_table.get(key)
        if gdef is not None:
            return self.instantiate_from_gdef(source_name, gdef, args, span=span)
        alias_def = self._program_alias_table.get(key)
        if alias_def is None and key in self._program_alias_keys:
            self._ensure_program_alias_resolved(key, span)
            alias_def = self._program_alias_table.get(key)
        if alias_def is not None:
            return self.instantiate_alias(source_name, alias_def, args, span=span)
        if key in self._program_type_table:
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
        lexical: bool = True,
    ) -> Type:
        # Type variables take priority over the type namespace.
        if name in type_vars:
            return TypeVarType(name)
        if lexical:
            name = self._lexical_type_name(name)
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
        # Check aliases through their declaration-time resolved templates.
        alias_def = self._resolved_aliases.get(name)
        if alias_def is not None:
            return self.instantiate_alias(name, alias_def, (), span=span)
        if name in self._alias_targets:
            if name in _resolving:
                raise AglTypeError(
                    f"Type alias '{name}' is part of a cycle.",
                    span=span,
                )
            target_expr = self._alias_targets[name]
            with self.type_scope(_split_scoped_type_name(name)[0]):
                target = self.resolve_type_expr(
                    target_expr,
                    span=span,
                    _resolving=_resolving | {name},
                    type_vars=type_vars,
                )
            self._resolved_aliases[name] = GenericAliasDef(type_params=(), template=target)
            return target
        program_alias_def = self._program_alias_table.get((self._module_id, (), name))
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
            return self._selected_builtin_type(name, typ, span)
        bare = self._resolve_bare_type(name, span)
        if bare is not None:
            return bare
        reserved_alias = BUILTIN_ALIAS_TARGETS.get(name)
        if reserved_alias is not None:
            return reserved_alias
        raise AglTypeError(
            f"Unknown type '{name}'.",
            span=span,
        )

    def resolve_qualified_name_type(
        self,
        qualifier: QualifierChain,
        name: str,
        *,
        span: SourceSpan | None,
    ) -> Type:
        """Resolve a module-qualified type reference ``QUALIFIER::Name``.

        Falls back to the local type namespace (prelude / built-ins) when the
        qualifier is empty (``::Name`` self-reference to the current module)
        and no program context exists.
        """
        rendered = qualifier.render()
        local_name = self._local_qualified_type_name(qualifier, name)
        if local_name is not None:
            local_path, declared_name = _split_scoped_type_name(local_name)
            self._ensure_qualified_type_route_unambiguous(
                qualifier,
                name,
                (self._module_id, local_path, declared_name),
                selected_route="type name",
                span=span,
            )
            return self._resolve_name_type(
                local_name, span=span, _resolving=frozenset(), lexical=False
            )

        if qualifier.anchor is None:
            opened_atom = to_bare_atom(
                (*tuple(segment.name for segment in qualifier.segments), name)
            )
            opened_key = self._opened_type_key(opened_atom, span)
            if opened_key is not None:
                opened = self._resolve_type_key_unapplied(opened_key, opened_atom, span)
                self._ensure_qualified_type_route_unambiguous(
                    qualifier,
                    name,
                    opened_key,
                    selected_route="use route",
                    span=span,
                )
                return opened
        if self._is_missing_local_scoped_type(qualifier, name):
            raise AglTypeError(self._unknown_scoped_type_message(qualifier, name), span=span)
        if self._import_env is None:
            raise AglTypeError(
                f"Module qualifier '{rendered}::' cannot be resolved outside of a module graph.",
                span=span,
            )

        qname = self._resolve_import_qname(qualifier, name, span=span)
        if not self._is_program_type_candidate(qname):
            raise AglTypeError(f"'{rendered}::{name}' does not name a type.", span=span)
        typ = self._resolve_program_qname_as_bare_type(qname, name, span=span)
        if typ is not None:
            return typ
        raise AglTypeError(f"'{rendered}::{name}' does not name a type.", span=span)

    def non_builtin_type_items(self) -> list[tuple[str, Type]]:
        """Return source-owned ``(name, type)`` pairs from the type namespace.

        Used by the program pre-pass to collect type shells into the shared
        ``program_type_table`` without accessing the private ``_types`` dict.
        Canonical fallback bindings are excluded, while a parsed ``builtin``
        declaration of the same reserved name is included like any other
        source declaration.
        """
        return [
            (name, typ)
            for name, typ in self._types.items()
            if name not in _BUILTIN_FALLBACK_TYPE_NAMES or _is_own_builtin_declaration(name, typ)
        ]

    def all_declared_type_names(self) -> frozenset[str]:
        """Return the full declared type-name set for type-namespace enumeration.

        Combines registered nominal types (records, enums, exceptions, and
        built-ins) with registered alias names and generic templates. Retained
        for generic type resolution in future work, where the complete
        type-namespace must be enumerable to validate applied-type names (e.g.
        ``Pair[int, text]``), and for REPL rollback of a type-owned subtree.
        """
        return (
            frozenset(self._types) | frozenset(self._alias_targets) | frozenset(self._generic_types)
        )

    def _selected_builtin_type(self, name: str, typ: Type, span: SourceSpan | None) -> Type:
        """Return the declaration the built-in *name* denotes in this module.

        Every module's type namespace carries the host's reserved fallback for
        each built-in name, whether or not anything declares it. Once a
        standard-library module declares that name, the loaded declaration is
        what the name means, and a reference here reaches it through the
        ordinary bare-name routes. Type references and owner-form enumeration
        both go through this, so they can never disagree about which
        declaration a built-in name denotes.
        """
        if name not in _BUILTIN_FALLBACK_TYPE_NAMES or _is_own_builtin_declaration(name, typ):
            return typ
        selected = self._resolve_bare_type(name, span)
        return typ if selected is None else selected

    def source_type_template_qname(
        self, module_id: ModuleId, name: str, *, scope_path: ScopePath = ()
    ) -> TypeTemplate | None:
        """Return immutable checked template data for one source type QName."""
        key = (module_id, scope_path, name)
        if self._in_program_type_tables(key):
            return self._program_table_template(key)
        local_name = "::".join((*scope_path, name))
        if module_id != self._module_id or not self._declares_local_type(local_name):
            return None
        return self._own_local_template(local_name)

    def declared_type_template(
        self, module_id: ModuleId, name: str, *, scope_path: ScopePath = ()
    ) -> TypeTemplate:
        """Return the template of a declared source type, forcing a lazy program alias first."""
        key = (module_id, scope_path, name)
        self._ensure_program_alias_resolved(key, None)
        if self._in_program_type_tables(key):
            return self._program_table_template(key)
        return self._own_local_template("::".join((*scope_path, name)))

    def _in_program_type_tables(self, key: DeclKey) -> bool:
        return (
            key in self._program_alias_table
            or key in self._program_generic_table
            or key in self._program_type_table
        )

    def _program_table_template(self, key: DeclKey) -> TypeTemplate:
        """Return the template of *key*, an entry of the program type tables."""
        alias_def = self._program_alias_table.get(key)
        if alias_def is not None:
            return TypeTemplate(alias_def.template, alias_def.type_params)
        generic_def = self._program_generic_table.get(key)
        if generic_def is not None:
            return TypeTemplate(generic_def.template, generic_def.type_params)
        return TypeTemplate(self._program_type_table[key])

    def _declares_local_type(self, local_name: str) -> bool:
        return (
            local_name in self._generic_types
            or local_name in self._resolved_aliases
            or local_name in self._alias_targets
            or local_name in self._types
        )

    def _own_local_template(self, local_name: str) -> TypeTemplate:
        """Return the template of *local_name*, a type this module declares locally."""
        local_generic = self._generic_types.get(local_name)
        if local_generic is not None:
            return TypeTemplate(local_generic.template, local_generic.type_params)
        if local_name in self._resolved_aliases or local_name in self._alias_targets:
            return self._own_alias_template(local_name)
        return TypeTemplate(self._selected_builtin_type(local_name, self._types[local_name], None))

    def _own_alias_template(self, local_name: str) -> TypeTemplate:
        """Return the template of this module's alias *local_name* over its parameters."""
        alias_def = self._resolved_aliases.get(local_name)
        if alias_def is not None:
            return TypeTemplate(alias_def.template, alias_def.type_params)
        type_params = self._alias_type_params.get(local_name, ())
        with self.type_scope(_split_scoped_type_name(local_name)[0]):
            template = self.resolve_type_expr(
                self._alias_targets[local_name],
                _resolving=frozenset({local_name}),
                type_vars=frozenset(type_params),
            )
        return TypeTemplate(template, type_params)

    def _own_source_type_names(self) -> frozenset[str]:
        cached = self._sealed_own_source_type_names
        if cached is not None:
            return cached
        names = set(self._types) | set(self._alias_targets) | set(self._generic_types)
        names.update(
            "::".join((*scope_path, name))
            for module_id, scope_path, name in self._program_alias_table
            if module_id == self._module_id
        )
        names.update(
            "::".join((*scope_path, name))
            for module_id, scope_path, name in (
                *self._program_generic_table,
                *self._program_type_table,
            )
            if module_id == self._module_id
        )
        own_names = frozenset(names)
        if self._sealed:
            self._sealed_own_source_type_names = own_names
        return own_names

    def resolve_enum_owner_form(
        self,
        kind: Literal[
            EnumOwnerFormKind.LOCAL, EnumOwnerFormKind.SELF, EnumOwnerFormKind.OPEN_IMPORT
        ],
        owner_name: str,
        module_qualifier: QualifierChain | None = None,
    ) -> EnumOwnerForm | None:
        """Resolve one exact unrouted enum-owner source form through checked visibility."""
        if kind is EnumOwnerFormKind.OPEN_IMPORT:
            if self._import_env is None or owner_name in self._own_source_type_names():
                return None
            type_qnames = tuple(
                qname
                for qname in self._import_env.unqualified.get(owner_name, frozenset())
                if self._is_program_type_candidate(qname)
            )
            if len(type_qnames) != 1:
                return None
            key = self._qname_decl_key(type_qnames[0])
            return self._enum_owner_form(kind, owner_name, module_qualifier, None, key)
        if owner_name not in self._own_source_type_names():
            return None
        return self._own_enum_owner_form(kind, owner_name, module_qualifier)

    def _own_enum_owner_form(
        self,
        kind: Literal[EnumOwnerFormKind.LOCAL, EnumOwnerFormKind.SELF],
        owner_name: str,
        module_qualifier: QualifierChain | None = None,
    ) -> EnumOwnerForm:
        """Build the owner form of *owner_name*, a type this module declares."""
        expected_qualifier = None if kind is EnumOwnerFormKind.LOCAL else ()
        key = (self._module_id, (), owner_name)
        return self._enum_owner_form(kind, owner_name, module_qualifier, expected_qualifier, key)

    def resolve_imported_enum_owner_form(
        self, module_qualifier: QualifierChain, owner_name: str, *, span: SourceSpan | None
    ) -> EnumOwnerForm | None:
        """Resolve a module-routed enum-owner source form through checked visibility.

        A qualified enum spelling preserves the shared resolver's verdict: an
        unknown route or ambiguity raises rather than reading as "not an enum".
        """
        qname = self._resolve_import_qname(module_qualifier, owner_name, span=span)
        if not self._is_program_type_candidate(qname):
            return None
        return self._enum_owner_form(
            EnumOwnerFormKind.QUALIFIED_IMPORT,
            owner_name,
            module_qualifier,
            module_qualifier.route_segments,
            self._qname_decl_key(qname),
        )

    def _enum_owner_form(
        self,
        kind: EnumOwnerFormKind,
        owner_name: str,
        module_qualifier: QualifierChain | None,
        expected_qualifier: tuple[str, ...] | None,
        key: DeclKey,
    ) -> EnumOwnerForm:
        source_module_id, source_scope_path, source_name = key
        return EnumOwnerForm(
            owner_name,
            expected_qualifier,
            kind=kind,
            source_module_id=source_module_id,
            source_name=source_name,
            type_template=self.declared_type_template(
                source_module_id, source_name, scope_path=source_scope_path
            ),
            qualifier_anchored=(
                module_qualifier.anchored if module_qualifier is not None else False
            ),
        )

    def _blocked_short_variants(self, form: EnumOwnerForm) -> frozenset[str]:
        """Return variants whose short owner spelling is occupied by a module route.

        Only a ``LOCAL``/``OPEN_IMPORT`` form can be shadowed this way: those
        are the only kinds writable as a bare ``owner_name`` qualifier, which
        is exactly the qualifier a same-named module route also competes for.
        """
        template = form.type_template.template
        if (
            self._import_env is None
            or form.kind not in (EnumOwnerFormKind.LOCAL, EnumOwnerFormKind.OPEN_IMPORT)
            or not isinstance(template, EnumType)
        ):
            return frozenset()
        owner_qualifier = (form.owner_name,)
        return frozenset(
            variant
            for variant in self.type_table.enum_member_names(template)
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
            forms.add(self._own_enum_owner_form(EnumOwnerFormKind.LOCAL, owner_name))
            forms.add(self._own_enum_owner_form(EnumOwnerFormKind.SELF, owner_name))
        if self._import_env is not None:
            for exposed_name in self._import_env.unqualified:
                if not isinstance(exposed_name, str):
                    continue
                form = self.resolve_enum_owner_form(EnumOwnerFormKind.OPEN_IMPORT, exposed_name)
                if form is None:
                    continue
                forms.add(form)
            for contribution in self._import_env.contributions.values():
                routes = contribution_routes(contribution)
                for exposed_name, qname in contribution.members.items():
                    if not isinstance(exposed_name, str) or not self._is_program_type_candidate(
                        qname
                    ):
                        continue
                    _, source_scope_path, source_name = self._qname_decl_key(qname)
                    template = self.declared_type_template(
                        qname[0], source_name, scope_path=source_scope_path
                    )
                    for qualifier, anchored in routes:
                        resolved = resolve_qualified(
                            self._import_env, qualifier, exposed_name, anchored=anchored
                        )
                        if not isinstance(resolved, QualResolutionFound) or resolved.qname != qname:
                            continue
                        forms.add(
                            EnumOwnerForm(
                                exposed_name,
                                qualifier,
                                kind=EnumOwnerFormKind.QUALIFIED_IMPORT,
                                source_module_id=qname[0],
                                source_name=source_name,
                                type_template=template,
                                qualifier_anchored=anchored,
                            )
                        )

        def form_key(form: EnumOwnerForm) -> tuple[str, tuple[str, ...], bool, str]:
            return (
                form.owner_name,
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
                blocked[(form.owner_name,)] = variants
        result: Mapping[tuple[str, ...], frozenset[str]] = MappingProxyType(blocked)
        if self._sealed:
            self._sealed_blocked_enum_variants = result
        return result

    def get_generic_type_from_module(
        self, module_id: ModuleId, name: str, *, scope_path: ScopePath = ()
    ) -> GenericTypeDef | None:
        """Look up a cross-module ``GenericTypeDef`` by structured owner identity."""
        return self._program_generic_table.get((module_id, scope_path, name))

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
        resolved = self._bare_type_key(name, span)
        if resolved is None:
            return None
        gdef = self._program_generic_table.get(resolved[0])
        return None if gdef is None else (name, gdef)

    def resolve_qualified_unapplied_generic_type(
        self,
        qualifier: QualifierChain,
        name: str,
        *,
        span: SourceSpan | None = None,
    ) -> tuple[str, GenericTypeDef] | None:
        """Resolve a module-qualified generic type name without applying arguments."""
        if not qualifier.route_segments:
            gdef = self._generic_types.get(name)
            if gdef is not None:
                return name, gdef
            gdef = self._program_generic_table.get((self._module_id, (), name))
            return (name, gdef) if gdef is not None else None
        # A non-empty qualifier may name a local named scope rather than an
        # import route (e.g. a REPL-retained ``scope A`` generic record
        # displayed as ``A::Box``).  Try exact local resolution first; a ``/``
        # route (``QualifierAnchor.MODULE``) is always an import route and is
        # never probed locally (``_local_qualified_type_name`` enforces this).
        local_name = self._local_qualified_type_name(qualifier, name)
        if (
            local_name is not None
            and (local_gdef := self._generic_types.get(local_name)) is not None
        ):
            local_path, declared_name = _split_scoped_type_name(local_name)
            self._ensure_qualified_type_route_unambiguous(
                qualifier,
                name,
                (self._module_id, local_path, declared_name),
                selected_route="type name",
                span=span,
            )
            return local_name, local_gdef
        if qualifier.anchor is None:
            opened_atom = to_bare_atom(
                (*tuple(segment.name for segment in qualifier.segments), name)
            )
            opened_key = self._opened_type_key(opened_atom, span)
            if opened_key is not None:
                opened_gdef = self._program_generic_table.get(opened_key)
                if opened_gdef is None:
                    return None
                self._ensure_qualified_type_route_unambiguous(
                    qualifier,
                    name,
                    opened_key,
                    selected_route="use route",
                    span=span,
                )
                rendered = qualifier.render()
                return f"{rendered}::{name}", opened_gdef
        if self._import_env is None or not qualifier.segments:
            return None
        qname = self._try_resolve_import_qname(qualifier, name)
        if qname is None:
            return None
        gdef = self._program_generic_table.get(self._qname_decl_key(qname))
        if gdef is None:
            return None
        rendered = qualifier.render()
        qualified_name = f"{rendered}::{name}"
        return qualified_name, gdef

    def all_generic_types(self) -> dict[str, GenericTypeDef]:
        """Return the own-module generic type map (name → GenericTypeDef)."""
        return self._generic_types

    # --- Seeding support ---

    def seed_from(self, other: TypeEnvironment, *, merge_type_table: bool = True) -> None:
        """Copy *other*'s user-declared types, aliases, and binding types in.

        Used to pre-populate a fresh environment with a session's accumulated
        state before checking a new entry.  Built-in exception types and
        built-in prelude types are already present in every fresh environment
        and are not copied from the source -- UNLESS *other*'s own binding for
        one is a program's own ``builtin`` declaration rather than the
        canonical one (:func:`_is_own_builtin_declaration`), which must carry
        forward exactly like any other declared name so a later entry's
        ``catch``/host-call resolution keeps agreeing with the identity the
        host has been minting since the declaring entry.  Binding types are
        keyed by globally-unique ``decl_node_id`` so they never collide across
        entries.

        Also merges *other*'s ``type_table`` entries in, unless
        *merge_type_table* is ``False``: *other* is treated as authoritative,
        so an entry under a key already present in this environment's table
        is overwritten (last-write-wins). For names present in *other*'s type
        namespace, stale metadata in this environment is cleared before
        copying so cross-kind REPL redefinitions do not leave old
        generic/constructor/alias tables behind. See :meth:`TypeTable.merge_from`.

        *merge_type_table* is ``False`` for a caller whose ``_type_table``
        already holds *other*'s table: *other*'s table was merged into it when
        the program type table was built, and this entry's own declarations
        were registered on top. Merging *other*'s table (a snapshot from
        before this entry) again would hand a name this entry redeclared back
        to its superseded owner; every other table this method copies is
        per-environment state the caller still needs.
        """
        incoming_type_names = (
            {name for name in other._types if name not in _BUILTIN_FALLBACK_TYPE_NAMES}
            | set(other._alias_targets)
            | set(other._generic_types)
            | set(other._alias_type_params)
        )
        for name in incoming_type_names:
            self.unregister_name(name)
        if merge_type_table:
            self._type_table.merge_from(other._type_table)
        for name, typ in other._types.items():
            if name not in _BUILTIN_FALLBACK_TYPE_NAMES or _is_own_builtin_declaration(name, typ):
                self._types[name] = typ
        self._alias_targets.update(other._alias_targets)
        self._resolved_aliases.update(other._resolved_aliases)
        self._binding_types = other._binding_types.fork()
        self._function_signatures.update(other._function_signatures)
        self._generic_types.update(other._generic_types)
        self._alias_type_params.update(other._alias_type_params)
        self._function_signatures_by_node_id.update(other._function_signatures_by_node_id)
        self._extern_node_ids.update(other._extern_node_ids)

    def rewind_from(
        self,
        previous: TypeEnvironment,
        *,
        type_names: Iterable[str],
        binding_node_ids: Iterable[int],
        functions: Mapping[int, str],
    ) -> None:
        """Undo what an incremental entry's UNPROMOTED declarations wrote, restoring *previous*.

        Checking an entry builds metadata for every declaration in it, but a
        runtime failure promotes only the declarations before the failure.
        This takes the three forms a declaration reaches this environment
        under — a type name, a value binding's node id, and a function's
        node id and name — and, for each, drops what the checked entry
        registered and puts back the definition *previous* held. The
        seed-forward counterpart is :meth:`seed_from`; a table added here
        later is rewound only once this method rewinds it.

        *type_names* normally comes from the entry's OWN declarations. A
        failed enum redeclaration also adds previously retained inline-member
        names whose namespace metadata the new enum cleared, so those
        survivors are restored in the same pass. A reserved built-in
        exception/prelude name can appear only when this entry itself wrote a
        ``builtin`` declaration of that name — the reserved names are
        non-shadowable other than by one — and is rolled back like any other
        unpromoted declaration, unlike in :meth:`seed_from`, which instead has
        to tell a program's own carried-forward declaration apart from the
        canonical default it must not clobber.

        The shared ``type_table`` is keyed by declaration identity, not name,
        so an unpromoted declaration stays registered under its own identity
        exactly as a superseded one does — the link image derives its nominal
        descriptors from this table on every lowering and needs the entry to
        correct the descriptor the failed entry already linked. What does need
        saying is that the declaration never took effect
        (:meth:`TypeTable.orphan`): unlike a superseded declaration, whose
        surviving values keep its members meaningful, an unpromoted one must
        answer no whole-table query about what the session declares. The
        previous declaration's own identity, methods, and base chain were
        never touched by the redeclaration, so restoring it is just
        ``register`` below reclaiming its name.

        Node ids are globally unique, so a binding normally has nothing to put
        back. Functions carry both forms because their convenient name-keyed
        signature table is not keyed by declaration id, while their methods
        are.

        *previous* is read while this environment is written, so it must be a
        separate environment with its own ``type_table`` -- the caller's
        accumulated state itself, since a fresh environment already seeds
        every reserved name and a copy of it would answer identically.
        """
        for name in type_names:
            self.unregister_name(name)
            scope_path, declared_name = _split_scoped_type_name(name)
            # Read the unpromoted declaration before ``register`` repoints the
            # name index at the survivor; the seeded table resolves this name
            # to the checked entry's own declaration, which is the one being
            # rolled back.
            unpromoted = self._type_table.get(self._module_id, declared_name, scope_path)
            typedef = previous._type_table.get(previous._module_id, declared_name, scope_path)
            if typedef is not None:
                self._type_table.register(typedef)
            # A failed enum redeclaration can clear a prior inline member's
            # namespace metadata without registering a replacement member.
            # In that case both lookups name the same survivor, which must be
            # restored rather than orphaned.
            if unpromoted is not None and (
                typedef is None or unpromoted.decl_node_id != typedef.decl_node_id
            ):
                self._type_table.orphan(unpromoted.decl_node_id)
            if name in previous._types:
                self._types[name] = previous._types[name]
            if name in previous._alias_targets:
                self._alias_targets[name] = previous._alias_targets[name]
            if name in previous._resolved_aliases:
                self._resolved_aliases[name] = previous._resolved_aliases[name]
            if name in previous._generic_types:
                self._generic_types[name] = previous._generic_types[name]
            if name in previous._alias_type_params:
                self._alias_type_params[name] = previous._alias_type_params[name]
        node_id_set = set(binding_node_ids)
        self.remove_binding_types(node_id_set)
        for node_id in node_id_set:
            binding_type = previous._binding_types.get(node_id)
            if binding_type is not None:
                self._binding_types[node_id] = binding_type
            signature = previous._function_signatures_by_node_id.get(node_id)
            if signature is not None:
                self._function_signatures_by_node_id[node_id] = signature
            if node_id in previous._extern_node_ids:
                self._extern_node_ids.add(node_id)
        for function_name in functions.values():
            self._function_signatures.pop(function_name, None)
            function_signature = previous._function_signatures.get(function_name)
            if function_signature is not None:
                self._function_signatures[function_name] = function_signature
        self._type_table.rewind_methods_from(previous._type_table, functions.keys())
