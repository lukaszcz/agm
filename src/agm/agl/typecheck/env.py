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
from typing import TYPE_CHECKING, Literal, Protocol, cast

if TYPE_CHECKING:
    from agm.agl.scope.program import ResolvedModule
    from agm.agl.syntax.types import AppliedT, NameT
    from agm.agl.typecheck.function_inference import FunctionSignatureRecord

from agm.agl.diagnostics import AglError, Diagnostic
from agm.agl.ir.ids import NominalId
from agm.agl.ir.reserved_nominals import NO_DECL_ID, require_reserved_nominal_id
from agm.agl.modules.ids import ENTRY_ID, RESERVED_ID, ModuleId, spell_declaration
from agm.agl.scope.imports import (
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
    resolve_bare_contribution_layer,
)
from agm.agl.self_validation import self_validation_enabled
from agm.agl.semantics.persistent import PersistentDict
from agm.agl.semantics.type_table import (
    DeclKey,
    MethodDef,
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
    TypeTemplateMatch,
    TypeVarType,
    UnitType,
    contains_inference_var,
    match_nominal_owner_template,
    substitute,
)
from agm.agl.syntax.nodes import Expr, Pattern, QualifierAnchor, QualifierChain
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import TypeExpr
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


def _type_path_atom(path: ScopePath) -> NameAtom:
    """Represent a contributed type path without flattening its identity."""
    return path[0] if len(path) == 1 else path


def _render_type_atom(atom: NameAtom) -> str:
    """Render a structured type atom for diagnostics only."""
    return atom if isinstance(atom, str) else "::".join(atom)


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
    kind: ParamZone
    has_default: bool


# ---------------------------------------------------------------------------
# FunctionSignature — full declared signature of a static def
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FunctionSignature:
    """Full declared signature of a root or named-scope ``def``.

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
    """Signature for one record constructor.

    ``owner_name``      — name of the record declaration.
    ``field_names``     — ordered field names accepted by the constructor.
    ``field_templates`` — field types (may contain ``TypeVarType`` nodes for
                          generic types).
    ``result_template`` — the return type template (may contain TypeVarType).
    ``type_params``     — type-parameter names for instantiation.
    """

    owner_name: str
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

    Raised by the type checker on the first type violation.  Carries an
    optional ``SourceSpan`` for source location.
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
    ``is_test_constructor_refs``
        Checker-selected constructors for bare ``is`` tests whose scope result
        retained multiple candidates. ``constructor_ref_for`` exposes the
        selection to lowering without rewriting scope's resolution table.
    ``let_matched_types``
        Complete concrete matched type for every immutable ``let`` site. This
        is distinct from each binder's type, which is keyed by its pattern node
        in ``type_env``.
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
    is_test_constructor_refs: dict[int, ConstructorRef] = field(default_factory=dict)
    let_matched_types: dict[int, Type] = field(default_factory=dict)
    pattern_binding_refs: dict[int, BindingRef] = field(default_factory=dict)
    pattern_constructor_refs: dict[int, ConstructorRef] = field(default_factory=dict)
    pattern_constructor_owners: dict[int, NominalId] = field(default_factory=dict)
    method_selections: dict[int, MethodDef] = field(default_factory=dict)
    explicit_builtin_targets: dict[int, Type] = field(default_factory=dict)

    def binding_for(self, node_id: int) -> BindingRef | None:
        """Return *node_id*'s checked binding, dereferencing a pattern slot."""
        return dereference_slot_binding(
            node_id,
            resolution=self.resolved.resolution,
            slot_resolution=self.slot_resolution,
        )

    def constructor_ref_for(self, node_id: int) -> ConstructorRef | None:
        """Return *node_id*'s scope or checker-selected constructor reference."""
        selected = self.is_test_constructor_refs.get(node_id)
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

    def pattern_binding_for(self, node_id: int) -> BindingRef | None:
        """Return the immutable binding selected for one pattern occurrence."""
        return self.pattern_binding_refs.get(node_id)

    def pattern_constructor_ref_for(self, node_id: int) -> ConstructorRef | None:
        """Return the constructor selected for one pattern occurrence."""
        return self.pattern_constructor_refs.get(node_id)

    def pattern_constructor_owner_for(self, node_id: int) -> NominalId | None:
        """Return the resolved nominal owner, distinct from source spelling."""
        return self.pattern_constructor_owners.get(node_id)

    @property
    def interface(self) -> ModuleTypeInterface:
        """Closed type metadata this module contributes to importers."""
        return self.type_env.module_interface()

    def image(self) -> CheckedModuleImage:
        """Build a data-only image for cache persistence and rehydration.

        Retains every field except ``resolved``/``type_env``/``import_env``/
        ``source_text`` — recovered from the current ``ResolvedModule`` on
        rehydration — plus the module's closed type interface and its
        environment's own-facts journal. Raises if ``type_env`` never started
        an own-facts journal (the single-module and REPL-seed paths never do).
        """
        return CheckedModuleImage(
            node_types=self.node_types,
            contract_specs=self.contract_specs,
            call_sites=self.call_sites,
            warnings=self.warnings,
            function_signatures=self.function_signatures,
            cast_specs=self.cast_specs,
            argument_bindings=self.argument_bindings,
            pattern_classifications=self.pattern_classifications,
            partial_calls=self.partial_calls,
            interface=self.interface,
            environment_facts=self.type_env.own_facts(),
            published_signatures=self.published_signatures,
            published_binding_types=self.published_binding_types,
            module_id=self.module_id,
            slot_resolution=self.slot_resolution,
            slot_constructor_refs=self.slot_constructor_refs,
            is_test_constructor_refs=self.is_test_constructor_refs,
            let_matched_types=self.let_matched_types,
            pattern_binding_refs=self.pattern_binding_refs,
            pattern_constructor_refs=self.pattern_constructor_refs,
            pattern_constructor_owners=self.pattern_constructor_owners,
            method_selections=self.method_selections,
            explicit_builtin_targets=self.explicit_builtin_targets,
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
    function_signatures: Mapping[str, FunctionSignature],
    cast_specs: Mapping[int, CastSpec],
    argument_bindings: ArgumentBindings,
    let_matched_types: Mapping[int, Type],
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
            *let_matched_types.values(),
            *explicit_builtin_targets.values(),
        ),
        owner=owner,
    )


def assert_checked_module_closed(checked: CheckedModule) -> None:
    """Assert that one module's checked output is safe to lower."""
    assert_checked_output_closed(
        node_types=checked.node_types,
        contract_specs=checked.contract_specs,
        call_sites=checked.call_sites,
        function_signatures=checked.function_signatures,
        cast_specs=checked.cast_specs,
        argument_bindings=checked.argument_bindings,
        let_matched_types=checked.let_matched_types,
        explicit_builtin_targets=checked.explicit_builtin_targets,
        owner="checked program",
    )
    # The environment's own binding table is not one of the side tables above
    # (it is validated as a whole by ``assert_closed``, independent of which
    # published node/call/signature happens to reference each binding).
    checked.type_env.assert_closed()
    checked.type_env.assert_shared_tables_closed()


# ---------------------------------------------------------------------------
# TypeEnvironment — mutable state during type checking
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModuleTypeInterface:
    """Closed declarations a compiled module contributes to its importers."""

    types: dict[DeclKey, Type]
    generics: dict[DeclKey, GenericTypeDef]
    aliases: dict[DeclKey, GenericAliasDef]
    constructors: dict[DeclKey, ConstructorSignature]
    field_kinds: dict[DeclKey, tuple[tuple[str, ParamZone], ...]]
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
# EnvironmentFacts — TypeEnvironment's own-facts journal
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BindingTypeFact:
    """Journaled :meth:`TypeEnvironment.set_binding_type` call."""

    node_id: int
    typ: Type

    def apply(self, env: TypeEnvironment) -> None:
        env.set_binding_type(node_id=self.node_id, typ=self.typ)


@dataclass(frozen=True, slots=True)
class FunctionSignatureFact:
    """Journaled :meth:`TypeEnvironment.register_function_signature` call."""

    name: str
    sig: FunctionSignature
    scope_path: ScopePath

    def apply(self, env: TypeEnvironment) -> None:
        env.register_function_signature(name=self.name, sig=self.sig, scope_path=self.scope_path)


@dataclass(frozen=True, slots=True)
class FunctionSignatureByNodeIdFact:
    """Journaled :meth:`TypeEnvironment.register_function_signature_by_node_id` call."""

    node_id: int
    sig: FunctionSignature

    def apply(self, env: TypeEnvironment) -> None:
        env.register_function_signature_by_node_id(node_id=self.node_id, sig=self.sig)


@dataclass(frozen=True, slots=True)
class ExternNodeIdFact:
    """Journaled :meth:`TypeEnvironment.register_extern_node_id` call."""

    node_id: int

    def apply(self, env: TypeEnvironment) -> None:
        env.register_extern_node_id(node_id=self.node_id)


@dataclass(frozen=True, slots=True)
class TypeFact:
    """Journaled :meth:`TypeEnvironment.register_type` call."""

    name: str
    typ: Type

    def apply(self, env: TypeEnvironment) -> None:
        env.register_type(name=self.name, typ=self.typ)


@dataclass(frozen=True, slots=True)
class GenericTypeFact:
    """Journaled :meth:`TypeEnvironment.register_generic_type` call."""

    name: str
    gdef: GenericTypeDef

    def apply(self, env: TypeEnvironment) -> None:
        env.register_generic_type(name=self.name, gdef=self.gdef)


@dataclass(frozen=True, slots=True)
class AliasFact:
    """Journaled :meth:`TypeEnvironment.register_alias` call."""

    name: str
    target_expr: TypeExpr
    type_params: tuple[str, ...]

    def apply(self, env: TypeEnvironment) -> None:
        # Structural == on syntax type nodes: skipping an identical one keeps a frozen alias.
        if env.has_alias_registration(self.name, self.target_expr, self.type_params):
            return
        env.register_alias(
            name=self.name, target_expr=self.target_expr, type_params=self.type_params
        )


@dataclass(frozen=True, slots=True)
class ConstructorSignatureFact:
    """Journaled :meth:`TypeEnvironment.register_constructor_signature` call."""

    sig: ConstructorSignature

    def apply(self, env: TypeEnvironment) -> None:
        env.register_constructor_signature(sig=self.sig)


@dataclass(frozen=True, slots=True)
class ConstructorFieldKindsFact:
    """Journaled :meth:`TypeEnvironment.register_constructor_field_kinds` call.

    ``module_id`` is always the resolved owner module (never ``None``), so
    replay never re-derives it from the replaying environment.
    """

    owner_name: str
    fields: tuple[tuple[str, ParamZone], ...]
    scope_path: ScopePath
    module_id: ModuleId
    decl_id: int | None

    def apply(self, env: TypeEnvironment) -> None:
        env.register_constructor_field_kinds(
            owner_name=self.owner_name,
            fields=self.fields,
            scope_path=self.scope_path,
            module_id=self.module_id,
            decl_id=self.decl_id,
        )


EnvironmentFact = (
    BindingTypeFact
    | FunctionSignatureFact
    | FunctionSignatureByNodeIdFact
    | ExternNodeIdFact
    | TypeFact
    | GenericTypeFact
    | AliasFact
    | ConstructorSignatureFact
    | ConstructorFieldKindsFact
)


@dataclass(frozen=True, slots=True)
class EnvironmentFacts:
    """Ordered journal of a ``TypeEnvironment``'s own-facts mutator calls.

    Data only, so it serializes under the artifact allow-list.
    :meth:`TypeEnvironment.replay` applies each entry in order to reproduce
    the recorded mutations on another environment.
    """

    entries: tuple[EnvironmentFact, ...] = ()


# ---------------------------------------------------------------------------
# CheckedModuleImage — data-only CheckedModule for cache persistence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckedModuleImage:
    """Data-only image of a ``CheckedModule``, ready to persist and rehydrate.

    Every ``CheckedModule`` field except ``resolved``, ``type_env``,
    ``import_env``, and ``source_text`` — those are recovered from the
    current ``ResolvedModule`` on rehydration — plus ``interface`` (this
    module's closed type contribution) and ``environment_facts`` (its
    environment's own-facts journal). Built by :meth:`CheckedModule.image`;
    turned back into an equivalent ``CheckedModule`` by :meth:`rehydrate`.
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
    is_test_constructor_refs: dict[int, ConstructorRef]
    let_matched_types: dict[int, Type]
    pattern_binding_refs: dict[int, BindingRef]
    pattern_constructor_refs: dict[int, ConstructorRef]
    pattern_constructor_owners: dict[int, NominalId]
    method_selections: dict[int, MethodDef]
    explicit_builtin_targets: dict[int, Type]

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
        env.begin_own_facts()
        env.replay(self.environment_facts)
        env.seal()
        return CheckedModule(
            resolved=resolved_module.resolved,
            node_types=self.node_types,
            contract_specs=self.contract_specs,
            call_sites=self.call_sites,
            warnings=self.warnings,
            type_env=env,
            function_signatures=self.function_signatures,
            cast_specs=self.cast_specs,
            argument_bindings=self.argument_bindings,
            pattern_classifications=self.pattern_classifications,
            partial_calls=self.partial_calls,
            published_signatures=self.published_signatures,
            published_binding_types=self.published_binding_types,
            module_id=self.module_id,
            import_env=resolved_module.import_env,
            source_text=resolved_module.source_text,
            slot_resolution=self.slot_resolution,
            slot_constructor_refs=self.slot_constructor_refs,
            is_test_constructor_refs=self.is_test_constructor_refs,
            let_matched_types=self.let_matched_types,
            pattern_binding_refs=self.pattern_binding_refs,
            pattern_constructor_refs=self.pattern_constructor_refs,
            pattern_constructor_owners=self.pattern_constructor_owners,
            method_selections=self.method_selections,
            explicit_builtin_targets=self.explicit_builtin_targets,
        )


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

    - ``program_type_table`` maps ``(ModuleId, name)`` to the fully-built
      ``Type`` objects stamped with their owning ``module_id``.  Built once by
      the program pre-pass; shared (read-only) across all per-module envs.
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
    these fields. They are absent only on the transient, shell-collection-only
    environments the whole-program type pre-pass builds in Step A
    (``typecheck/program.py::_build_program_type_table``) to register every
    module's declaration headers before any body is resolved; those
    environments are never queried beyond that registration.
    """

    def __init__(
        self,
        *,
        program_type_table: Mapping[DeclKey, Type] | None = None,
        program_generic_table: Mapping[DeclKey, GenericTypeDef] | None = None,
        program_alias_table: Mapping[DeclKey, GenericAliasDef] | None = None,
        program_alias_keys: frozenset[DeclKey] | None = None,
        program_alias_resolver: Callable[[DeclKey, SourceSpan | None], Type | None] | None = None,
        program_ctor_sig_table: Mapping[DeclKey, ConstructorSignature] | None = None,
        program_ctor_field_kinds_table: Mapping[DeclKey, tuple[tuple[str, ParamZone], ...]]
        | None = None,
        import_env: ImportEnv | None = None,
        local_scope_paths: frozenset[ScopePath] = frozenset(),
        scope_nodes: Mapping[ScopePath, ScopeNode] | None = None,
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
        # Alias syntax is retained for diagnostics and cycle detection, while the
        # resolved template preserves the nominal identities selected when the
        # alias was declared (including across REPL supersession).
        self._alias_targets: dict[str, TypeExpr] = {}
        self._resolved_aliases: dict[str, GenericAliasDef] = {}
        # Binding node_id → Type (populated as declarations are checked).
        self._binding_types: PersistentDict[int, Type] = PersistentDict()
        # Root-scope function signatures by unqualified name: a compatibility
        # map for standalone callers that only know an unqualified spelling;
        # resolved calls use declaration ids.
        self._function_signatures: dict[str, FunctionSignature] = {}
        # Generic type definitions — name → GenericTypeDef.
        self._generic_types: dict[str, GenericTypeDef] = {}
        # Constructor signatures — ((module, scope path, owner), variant) → signature.
        self._constructor_sigs: dict[DeclKey, ConstructorSignature] = {}
        # Alias type-params — name → tuple of type-param names.
        self._alias_type_params: dict[str, tuple[str, ...]] = {}
        # Node-id-keyed function signatures — decl_node_id → FunctionSignature.
        # Program environments receive explicit headers from the whole-program
        # pre-pass, then closed unannotated candidates from import-SCC inference,
        # so _check_declared_name_call can look up the correct cross-module callee
        # by globally unique decl_node_id rather than by bare name (which would
        # collide when modules define different same-named functions).
        self._function_signatures_by_node_id: dict[int, FunctionSignature] = {}
        # Declaration node_ids of ``extern def``s, keyed by the same globally-unique
        # decl_node_id as ``_function_signatures_by_node_id``.  Populated by
        # ``_preregister_funcdef`` (this module's own externs) and by the program
        # function-signature pre-pass seeding (imported externs).  Consulted by
        # ``_check_declared_name_call`` to decide whether a declared-name call
        # site is an extern call site to record.
        self._extern_node_ids: set[int] = set()
        # Constructor field-kinds registry — ((module, scope path, owner), variant)
        # → ordered (field_name, ParamZone) pairs. Populated by _TypeBuilder
        # and consumed without encoding declaration paths into strings.
        self._constructor_field_kinds: dict[DeclKey, tuple[tuple[str, ParamZone], ...]] = {}
        self._constructor_field_kinds_by_decl_id: dict[int, tuple[tuple[str, ParamZone], ...]] = {}
        # Cross-module constructor field-kinds table keyed by declaration identity.
        self._program_ctor_field_kinds_table: (
            Mapping[DeclKey, tuple[tuple[str, ParamZone], ...]] | None
        ) = program_ctor_field_kinds_table
        # Program context: None in module path.
        self._program_type_table: Mapping[DeclKey, Type] | None = program_type_table
        # Cross-module generic type definitions: (ModuleId, name) → GenericTypeDef.
        # Populated by the program type pre-pass for qualified generic constructor calls.
        self._program_generic_table: Mapping[DeclKey, GenericTypeDef] | None = (
            None
            if program_generic_table is None
            else (
                program_generic_table
                if all(len(key) == 3 for key in program_generic_table)
                else {
                    key if len(key) == 3 else (key[0], (), key[1]): value
                    for key, value in program_generic_table.items()
                }
            )
        )
        # Cross-module parameterized type aliases: (ModuleId, name) → GenericAliasDef.
        self._program_alias_table: Mapping[DeclKey, GenericAliasDef] = (
            {}
            if program_alias_table is None
            else (
                program_alias_table
                if all(len(key) == 3 for key in program_alias_table)
                else {
                    key if len(key) == 3 else (key[0], (), key[1]): value
                    for key, value in program_alias_table.items()
                }
            )
        )
        self._program_alias_keys: frozenset[DeclKey] = (
            program_alias_keys if program_alias_keys is not None else frozenset()
        )
        self._program_alias_resolver: Callable[[DeclKey, SourceSpan | None], Type | None] | None = (
            program_alias_resolver
        )
        # Cross-module constructor signatures keyed by declaration identity.
        self._program_ctor_sig_table: Mapping[DeclKey, ConstructorSignature] | None = (
            program_ctor_sig_table
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
        # Own-facts journal, active from begin_own_facts() until seal() snapshots it
        # into _own_facts. unregister_name, freeze_alias, restore_*,
        # remove_binding_types and seed_from never run in that window; not journaled.
        # ``_resolve_name_type`` does write ``_resolved_aliases`` directly (a memo
        # re-derivable from ``_alias_targets``) on a query path, but every declared
        # alias is frozen during program preparation, so that write is unreachable
        # once the journal window opens.
        self._journal: list[EnvironmentFact] | None = None
        self._own_facts: EnvironmentFacts | None = None
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
        # available.  Field/variant names AND zones for constructor-kind
        # registration come from the shared prelude ``TypeDef`` literals (via
        # ``TypeTable.field_kinds``) — the handles themselves carry no shape
        # data.
        for prelude_name, prelude_type in BUILTIN_PRELUDE_TYPES.items():
            self._types[prelude_name] = prelude_type
            if isinstance(prelude_type, RecordType):
                fields = self._type_table.field_kinds(prelude_type)
                self._constructor_field_kinds[(RESERVED_ID, (), prelude_name)] = fields
                self._constructor_field_kinds_by_decl_id[prelude_type.decl_id] = fields
                continue
            if isinstance(prelude_type, ExceptionType):
                continue
            assert isinstance(prelude_type, EnumType)
            for member in self._type_table.enum_members(prelude_type):
                fields = self._type_table.field_kinds(member)
                self._constructor_field_kinds[(RESERVED_ID, (prelude_name,), member.name)] = fields
                self._constructor_field_kinds_by_decl_id[member.decl_id] = fields
        # Exception constructor field kinds are NOT pre-registered here: each
        # exception's own fields honor their declared ``@arg-*`` attribute
        # (stored on its TypeDef as ``field_kinds``, alongside ``fields``),
        # same as a record's fields.  ``get_constructor_field_kinds_for_type``
        # derives the full flattened (base-chain-inherited + own) kinds
        # directly from ``type_table.field_kinds`` on demand instead of a
        # pre-registration step, since that requires no build ordering.

    def module_interface(self) -> ModuleTypeInterface:
        """Export this module's closed type metadata without another header pass."""

        def own[V](table: Mapping[DeclKey, V] | None) -> dict[DeclKey, V]:
            return {key: value for key, value in (table or {}).items() if key[0] == self._module_id}

        return ModuleTypeInterface(
            own(self._program_type_table),
            own(self._program_generic_table),
            own(self._program_alias_table),
            own(self._program_ctor_sig_table),
            own(self._program_ctor_field_kinds_table),
            tuple(td for td in self._type_table.entries() if td.module_id == self._module_id),
        )

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
        """Freeze this environment as checked output.

        Validates the environment first when self-validation is enabled; sealing
        itself — the functional state that gates further mutation and memoization
        (see :attr:`is_sealed`) — always happens, regardless of the flag. Closes
        an active journal into a stable snapshot :meth:`own_facts` returns.
        """
        if self_validation_enabled():
            self.assert_closed()
        if self._journal is not None:
            self._own_facts = EnvironmentFacts(tuple(self._journal))
            self._journal = None
        self._sealed = True

    # --- Own-facts journal ---

    def begin_own_facts(self) -> None:
        """Start recording this environment's own mutator calls into a journal."""
        self._assert_mutable()
        if self._journal is not None:
            raise AssertionError("own-facts journal already active")
        self._journal = []

    def own_facts(self) -> EnvironmentFacts:
        """Return this environment's own-facts journal.

        The sealed snapshot once :meth:`seal` has run, else the active
        journal. Raises if :meth:`begin_own_facts` was never called: such an
        environment (module path, REPL seed) never journaled anything and has
        no own facts to persist.
        """
        if self._own_facts is not None:
            return self._own_facts
        if self._journal is not None:
            return EnvironmentFacts(tuple(self._journal))
        raise AssertionError("own-facts journal was never started")

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
        self._assert_mutable()
        journal, self._journal = self._journal, None
        try:
            for fact in facts.entries:
                fact.apply(self)
        finally:
            self._journal = journal
        if journal is not None:
            journal.extend(facts.entries)

    def _record_fact(self, fact: EnvironmentFact) -> None:
        """Append a fact while the own-facts journal is active."""
        if self._journal is not None:
            self._journal.append(fact)

    # --- Type namespace queries ---

    def get_type(self, name: str) -> Type | None:
        return self._types.get(name)

    def get_type_by_declaration(self, module_id: ModuleId, scope_path: ScopePath) -> Type | None:
        """Look up a nominal declaration by its module and complete scope path."""
        name = "::".join(scope_path)
        if module_id == self._module_id:
            local = self._types.get(name)
            if local is not None:
                return local
        if self._program_type_table is not None:
            return self._program_type_table.get((module_id, scope_path[:-1], scope_path[-1]))
        typedef = self._type_table.get(module_id, scope_path[-1], scope_path[:-1])
        return None if typedef is None else typedef.handle()

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
        imported_qname = self._resolve_import_qname(qualifier, name, span=span, required=False)
        if imported_qname is not None and self._qname_decl_key(imported_qname) == selected_key:
            return
        raise AglTypeError(
            f"Qualifier '{qualifier.render()}' is both a {selected_route} and a module route "
            f"for '{name}'. {qualification_repair_guidance()}",
            span=span,
        )

    def _resolve_import_qname(
        self,
        qualifier: QualifierChain,
        name: str,
        *,
        span: SourceSpan | None,
        required: bool = True,
    ) -> QName | None:
        """Resolve a module route followed by one structured type path."""
        import_env = cast(ImportEnv, self._import_env)
        route = tuple(part for part in qualifier.segments[0].name.split("/"))
        atom_path = (*tuple(segment.name for segment in qualifier.segments[1:]), name)
        atom: NameAtom = atom_path[0] if len(atom_path) == 1 else atom_path
        if not required:
            return try_resolve_qualified_member(
                import_env, route, atom, anchored=qualifier.anchored
            )
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
        """Resolve an inline enum member after applying its owner type.

        ``Source[T]::Member`` applies ``T`` to ``Source``.  An inline member
        captures only the owner parameters used by its fields, so selecting it
        from the instantiated enum yields its concrete record handle directly.
        """
        if not qualifier.segments:
            return None
        owner_segment = qualifier.segments[-1]
        if owner_segment.type_args is None:
            return None
        from agm.agl.syntax.types import AppliedT

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
        owner = self.resolve_type_expr(
            AppliedT(
                name=owner_segment.name,
                args=owner_segment.type_args,
                qualifier=owner_qualifier,
                span=owner_segment.span,
                node_id=owner_segment.node_id,
            ),
            span=span,
            type_vars=type_vars,
        )
        if not isinstance(owner, EnumType):
            raise AglTypeError(f"'{owner_segment.name}' is not a generic enum type.", span=span)
        owner_template = self.source_type_template_qname(
            owner.module_id, owner.name, scope_path=owner.scope_path
        )
        assert owner_template is not None
        member_template = self.source_type_template_qname(
            owner.module_id,
            member,
            scope_path=(*owner.scope_path, owner.name),
        )
        if member_template is None:
            return None
        resolved = substitute(
            member_template.template,
            dict(zip(owner_template.type_params, owner.type_args, strict=True)),
        )
        return resolved if isinstance(resolved, RecordType) else None

    def register_type(self, name: str, typ: Type) -> None:
        self._assert_mutable()
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

        Constructor signatures and field-kind metadata (``_constructor_sigs``/
        ``_constructor_field_kinds``) are deliberately NOT cleared here even
        though they are also name-keyed: unlike the tables above, a caller
        that queries them by owner name and variant is always resolving a
        SPECIFIC declaration's own field/variant it already holds a handle or
        pattern binder for (a record's fields, one enum variant), not asking
        "what does this bare name mean now" — so a variant the newest
        declaration does not redeclare (an enum's dropped variant, a
        redeclared-non-generic record's stale constructor signature that no
        code path reaches once ``_generic_types`` no longer names it generic)
        must stay reachable for an OLDER-typed retained value, while every key
        the newest declaration DOES define is naturally overwritten by its own
        registration regardless.

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
        self._assert_mutable()
        if name in _BUILTIN_FALLBACK_TYPE_NAMES:
            return
        self._types.pop(name, None)
        self._alias_targets.pop(name, None)
        self._resolved_aliases.pop(name, None)
        self._generic_types.pop(name, None)
        self._alias_type_params.pop(name, None)

    def register_alias(
        self, name: str, target_expr: TypeExpr, *, type_params: tuple[str, ...] = ()
    ) -> None:
        """Store the raw TypeExpr for *name*; resolved lazily by resolve_type_expr.

        ``type_params`` must be provided for parameterized type aliases (e.g.
        ``type Wrapper[T] = array[T]``); defaults to ``()`` for plain aliases.
        """
        self._assert_mutable()
        self._alias_targets[name] = target_expr
        self._resolved_aliases.pop(name, None)
        self._alias_type_params[name] = type_params
        self._record_fact(AliasFact(name=name, target_expr=target_expr, type_params=type_params))

    def freeze_alias(self, name: str, template: Type, *, type_params: tuple[str, ...] = ()) -> None:
        """Preserve an alias's resolved template under its declaring identities."""
        self._assert_mutable()
        self._resolved_aliases[name] = GenericAliasDef(type_params=type_params, template=template)

    # --- Generic type registry ---

    def register_generic_type(self, name: str, gdef: GenericTypeDef) -> None:
        """Register a generic type definition under *name*."""
        self._assert_mutable()
        self._generic_types[name] = gdef
        self._record_fact(GenericTypeFact(name=name, gdef=gdef))

    def get_generic_type(self, name: str) -> GenericTypeDef | None:
        """Return the ``GenericTypeDef`` for *name*, or ``None`` if unknown."""
        return self._generic_types.get(name)

    def get_generic_type_by_declaration(
        self, module_id: ModuleId, scope_path: ScopePath
    ) -> GenericTypeDef | None:
        """Look up a generic declaration by its module and complete scope path."""
        name = "::".join(scope_path)
        if module_id == self._module_id:
            local = self._generic_types.get(name)
            if local is not None:
                return local
        if self._program_generic_table is None:
            return None
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

    # --- Constructor signature registry ---

    @staticmethod
    def _constructor_key(module_id: ModuleId, owner_name: str, scope_path: ScopePath) -> DeclKey:
        """Build a constructor key from structured owner identity.

        Every caller supplies the owner's bare name and its scope path
        separately, so no display spelling is ever parsed back into identity.
        """
        return (module_id, scope_path, owner_name)

    def register_constructor_signature(self, sig: ConstructorSignature) -> None:
        """Register a constructor signature under its result type's identity."""
        self._assert_mutable()
        result = sig.result_template
        assert isinstance(result, (RecordType, EnumType))
        key = self._constructor_key(result.module_id, result.name, result.scope_path)
        self._constructor_sigs[key] = sig
        self._record_fact(ConstructorSignatureFact(sig=sig))

    def get_constructor_signature(
        self, owner_name: str, *, scope_path: ScopePath = ()
    ) -> ConstructorSignature | None:
        """Return a local constructor signature by structured owner identity."""
        return self._constructor_sigs.get(
            self._constructor_key(self._module_id, owner_name, scope_path)
        )

    def resolve_named_type(self, name: str, *, span: SourceSpan | None = None) -> Type | None:
        """Resolve a type *name* alias-transparently to a semantic ``Type``.

        Returns the resolved ``Type`` for a record/enum/exception name or an
        alias chain (multi-hop, alias-of-alias) that bottoms out in a named
        type; ``None`` if the name is unknown or names a non-nominal alias
        target (e.g. an alias of ``array[int]``, which has no single name).
        Used for alias-transparent qualifier resolution in qualified
        constructors and ``is`` tests.

        In program context, also searches types exposed bare by ``use`` declarations
        and import tails when the name is not found locally.

        An ambiguity complaint about a name contributed by several routes
        propagates: it names the problem better than the "unknown type" the
        caller would otherwise report, and callers pass *span* so it lands on
        the reference. A name that resolves to something no bare reference can
        denote -- a parameterized alias, say -- is still just ``None``.
        """
        local_name = self._lexical_type_name(name)
        if local_name in self._generic_types:
            return self._generic_types[local_name].template
        if local_name in self._types or local_name in self._alias_targets:
            try:
                return self._resolve_name_type(
                    local_name, span=span, _resolving=frozenset(), lexical=False
                )
            except AglTypeError:
                return None
        # Bare contributions from a root ``use`` and import tails share one
        # resolution rank. A contribution from a nearer named region still
        # shadows the module-root rank. Generic templates remain useful to
        # alias-transparent qualifier checks even though ordinary bare type
        # expressions require arguments.
        resolved = self._bare_type_key(name, span)
        if resolved is None:
            return None
        key, from_region = resolved
        generic = (self._program_generic_table or {}).get(key)
        if generic is not None and from_region:
            return generic.template
        alias = self._program_alias_table.get(key)
        if alias is not None and from_region:
            return alias.template
        try:
            return self._resolve_type_key_as_bare(key, name, span=span)
        except AglTypeError:
            return None

    def type_name_declaration(self, type_expr: NameT | AppliedT) -> DeclKey | None:
        """Return the declaration identity an annotation's type name selects.

        Follows resolution's selection order without resolving, so a host can
        see through the transparent aliases resolution erases: this module's
        own type namespace, then scope-use and import contributions, then a
        qualified name's module route. ``None`` when no route selects one, as
        for a builtin alias target no declaration reaches. *type_expr* must
        belong to a checked annotation.
        """
        name = type_expr.name
        qualifier = type_expr.qualifier
        if qualifier is None:
            local_name = self._lexical_type_name(name)
            if self._has_own_type_name(local_name):
                return (self._module_id, *_split_scoped_type_name(local_name))
            bare = self._bare_type_key(name, None)
            return None if bare is None else bare[0]
        qualified_local_name = self._local_qualified_type_name(qualifier, name)
        if qualified_local_name is not None:
            return (self._module_id, *_split_scoped_type_name(qualified_local_name))
        if qualifier.anchor is None:
            segments = tuple(segment.name for segment in qualifier.segments)
            opened = self._opened_type_key(_type_path_atom((*segments, name)), None)
            if opened is not None:
                return opened
        qname = self._resolve_import_qname(qualifier, name, span=None, required=False)
        return None if qname is None else self._qname_decl_key(qname)

    @staticmethod
    def _qname_decl_key(qname: QName) -> DeclKey:
        atom = qname[1]
        path = (atom,) if isinstance(atom, str) else atom
        return (qname[0], path[:-1], path[-1])

    def _is_program_type_candidate(self, qname: QName) -> bool:
        """Return whether a program-qualified name denotes any type-namespace declaration."""
        key = self._qname_decl_key(qname)
        return (
            (self._program_type_table is not None and key in self._program_type_table)
            or (self._program_generic_table is not None and key in self._program_generic_table)
            or key in self._program_alias_table
            or key in self._program_alias_keys
        )

    def _ensure_program_alias_resolved(self, key: DeclKey, span: SourceSpan | None) -> Type | None:
        """Resolve a program alias lazily while retaining its declaration path."""
        if key not in self._program_alias_keys or self._program_alias_resolver is None:
            return None
        return self._program_alias_resolver(key, span)

    def _resolve_program_qname_as_bare_type(
        self, qname: QName, exposed_name: str, *, span: SourceSpan | None
    ) -> Type | None:
        """Resolve a program-qualified name used as an unapplied type expression."""
        assert self._program_type_table is not None
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
        return self._program_type_table.get(key)

    # --- Function signature table ---

    def register_function_signature(
        self, name: str, sig: FunctionSignature, *, scope_path: ScopePath = ()
    ) -> None:
        """Register a root-scope signature under its unqualified name.

        Non-root registrations (``scope_path`` set) record the fact for replay
        but leave the compatibility map alone; resolved calls to a scoped
        function use its declaration id instead.
        """
        self._assert_mutable()
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
        self._assert_mutable()
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

    def register_extern_node_id(self, node_id: int) -> None:
        """Mark a function declaration ``node_id`` as an ``extern def``.

        Consulted by ``_check_declared_name_call`` so that direct calls to an
        extern (own-module or imported) are recorded as dry-run call sites the
        same way ``ask``/``exec`` calls are.
        """
        self._assert_mutable()
        self._extern_node_ids.add(node_id)
        self._record_fact(ExternNodeIdFact(node_id=node_id))

    def is_extern_node_id(self, node_id: int) -> bool:
        """Return ``True`` if *node_id* names a declared ``extern def``."""
        return node_id in self._extern_node_ids

    # --- Binding type table ---

    def set_binding_type(self, node_id: int, typ: Type) -> None:
        self._assert_mutable()
        self._binding_types[node_id] = typ
        self._record_fact(BindingTypeFact(node_id=node_id, typ=typ))

    def snapshot_binding_types(self) -> PersistentDict[int, Type]:
        """Return a restorable snapshot of transient binding-type metadata."""
        return self._binding_types.fork()

    def restore_binding_types(self, snapshot: PersistentDict[int, Type]) -> None:
        """Restore binding metadata after a disposable checking session."""
        self._assert_mutable()
        self._binding_types = snapshot

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

    def _opened_type_layer_keys(self, name: NameAtom) -> tuple[ScopeNode, set[DeclKey]] | None:
        """Return the nearest region and type identities contributed there."""
        scope = self._scope_nodes.get(self._type_scope)
        if scope is None:
            return None
        resolved = resolve_bare_contribution_layer(
            scope, name, predicate=self._is_type_contribution
        )
        if resolved is None:
            return None
        layer, candidates = resolved
        return layer, {(ref.module_id, ref.scope_path, ref.name) for ref in candidates}

    def _opened_type_key(self, name: NameAtom, span: SourceSpan | None) -> DeclKey | None:
        """Return the unique type declaration contributed to this type region."""
        resolved = self._opened_type_layer_keys(name)
        keys = set() if resolved is None else resolved[1]
        return self._unique_bare_type_key(name, keys, span)

    def _bare_type_key(self, name: str, span: SourceSpan | None) -> tuple[DeclKey, bool] | None:
        """Resolve a bare type across equally ranked root use and import routes.

        Reports alongside the identity whether it came from the nearest
        region's own contributions, which callers need to choose between an
        unapplied and a bare resolution. One layer walk answers both.
        """
        opened = self._opened_type_layer_keys(name)
        opened_keys = set() if opened is None else opened[1]
        keys = set(opened_keys)
        if opened is None or not opened[0].scope_path:
            if self._import_env is not None and (
                self._program_type_table is not None or self._program_generic_table is not None
            ):
                keys.update(
                    self._qname_decl_key(qname)
                    for qname in self._import_env.unqualified.get(name, frozenset())
                    if self._is_program_type_candidate(qname)
                )
        key = self._unique_bare_type_key(name, keys, span)
        return None if key is None else (key, key in opened_keys)

    def _is_type_contribution(self, ref: BindingRef) -> bool:
        """Whether a shared bare contribution refers to a type declaration."""
        key = (ref.module_id, ref.scope_path, ref.name)
        if self._program_type_table is not None and key in self._program_type_table:
            return True
        if self._program_generic_table is not None and key in self._program_generic_table:
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
        assert self._program_type_table is not None
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
    ) -> Type | None:
        """Resolve one declaration identity while rejecting a bare generic."""
        generic = (self._program_generic_table or {}).get(key)
        if generic is not None:
            rendered = _render_type_atom(name)
            raise AglTypeError(
                f"Generic type '{rendered}' requires {len(generic.type_params)} type argument(s); "
                f"use '{rendered}[...]' to apply it.",
                span=span,
            )
        return self._resolve_type_key_as_bare(key, _render_type_atom(name), span=span)

    def _resolve_opened_type(self, name: NameAtom, span: SourceSpan | None) -> Type | None:
        """Resolve a relative type path through scope-use contributions."""
        key = self._opened_type_key(name, span)
        return None if key is None else self._resolve_type_key_unapplied(key, name, span)

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
        generic = (self._program_generic_table or {}).get(key)
        if generic is not None:
            return self.instantiate_from_gdef(key[2], generic, args, span=span)
        self._ensure_program_alias_resolved(key, span)
        alias = self._program_alias_table.get(key)
        if alias is not None:
            return self.instantiate_alias(key[2], alias, args, span=span)
        alias_name = self._own_alias_name_for_key(key)
        if alias_name is not None:
            resolved = self._resolve_local_applied_alias(alias_name, args, span)
            assert resolved is not None
            return resolved
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

    def _resolve_opened_applied_type(
        self, name: NameAtom, args: tuple[Type, ...], span: SourceSpan | None
    ) -> Type | None:
        """Resolve a relative generic path through scope-use contributions."""
        key = self._opened_type_key(name, span)
        return None if key is None else self._resolve_applied_type_key(key, name, args, span)

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
                or (
                    self._program_type_table is not None
                    and (self._module_id, (), candidate) in self._program_type_table
                )
                or (
                    self._program_generic_table is not None
                    and (self._module_id, (), candidate) in self._program_generic_table
                )
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
            or (
                self._program_type_table is not None
                and (self._module_id, (), name) in self._program_type_table
            )
            or (
                self._program_generic_table is not None
                and (self._module_id, (), name) in self._program_generic_table
            )
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
        bases = (
            ((),)
            if qualifier.anchor is QualifierAnchor.CURRENT_MODULE
            else tuple(self._type_scope[:end] for end in range(len(self._type_scope), -1, -1))
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
        bases = (
            ((),)
            if qualifier.anchor is QualifierAnchor.CURRENT_MODULE
            else tuple(self._type_scope[:end] for end in range(len(self._type_scope), -1, -1))
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
            and _type_path_atom((*qualifier.route_segments, name)) in self._import_env.unqualified
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
            val = self.resolve_type_expr(
                type_expr.value, _resolving=_resolving, type_vars=type_vars
            )
            return DictType(value=val)
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
                return self._resolve_qualified_name_type(
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
                opened_atom = _type_path_atom(
                    (*tuple(segment.name for segment in qualifier.segments), type_expr.name)
                )
                opened = self._resolve_opened_applied_type(
                    opened_atom,
                    resolved_args,
                    span=eff_span,
                )
                if opened is not None:
                    opened_key = self._opened_type_key(opened_atom, eff_span)
                    assert opened_key is not None
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
        if self._import_env is None or self._program_generic_table is None:
            raise AglTypeError(
                f"Module qualifier '{rendered}::' cannot be resolved outside of a module graph.",
                span=span,
            )
        qname = self._resolve_import_qname(qualifier, name, span=span)
        assert qname is not None
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
        if self._program_type_table is not None and key in self._program_type_table:
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

    def _resolve_qualified_name_type(
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
            opened_atom = _type_path_atom(
                (*tuple(segment.name for segment in qualifier.segments), name)
            )
            opened = self._resolve_opened_type(opened_atom, span)
            if opened is not None:
                opened_key = self._opened_type_key(opened_atom, span)
                assert opened_key is not None
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
        if self._program_type_table is None or self._import_env is None:
            raise AglTypeError(
                f"Module qualifier '{rendered}::' cannot be resolved outside of a module graph.",
                span=span,
            )

        qname = self._resolve_import_qname(qualifier, name, span=span)
        assert qname is not None
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

    def resolve_constructible_type_by_module_id(
        self, module_id: ModuleId, name: str, *, scope_path: ScopePath = ()
    ) -> RecordType | EnumType | ExceptionType | None:
        """Resolve a program type name or transparent alias to a nominal target.

        Scope tentatively classifies aliases with a named or applied body as
        constructor bindings. This confirms that classification after aliases
        have resolved, without exposing a non-nominal alias to constructor
        lookup.
        """
        template = self.source_type_template_qname(module_id, name, scope_path=scope_path)
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
        *,
        scope_path: ScopePath = (),
    ) -> TypeTemplateMatch | None:
        """Alias-transparently match a checked source type QName to ``concrete``.

        This is the public checked-type boundary for downstream consumers that
        retain source import names. It exposes only an immutable semantic match,
        not the mutable program type/generic/alias registries. Non-generic aliases,
        generic aliases, alias chains, transformed arguments, and ordinary
        nominal declarations all share the exact template matcher.
        """
        template = self.source_type_template_qname(module_id, name, scope_path=scope_path)
        return None if template is None else match_nominal_owner_template(template, concrete)

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
        local_name = "::".join((*scope_path, name))
        local_generic = self._generic_types.get(local_name)
        if local_generic is not None:
            return TypeTemplate(local_generic.template, local_generic.type_params)
        alias_def = self._resolved_aliases.get(local_name)
        if alias_def is not None:
            return TypeTemplate(alias_def.template, alias_def.type_params)
        alias_expr = self._alias_targets.get(local_name)
        if alias_expr is not None:
            type_params = self._alias_type_params.get(local_name, ())
            with self.type_scope(_split_scoped_type_name(local_name)[0]):
                template = self.resolve_type_expr(
                    alias_expr,
                    _resolving=frozenset({local_name}),
                    type_vars=frozenset(type_params),
                )
            return TypeTemplate(template, type_params)
        resolved = self._types.get(local_name)
        if resolved is None:
            return None
        return TypeTemplate(self._selected_builtin_type(local_name, resolved, None))

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
        if self._program_generic_table is not None:
            names.update(
                "::".join((*scope_path, name))
                for module_id, scope_path, name in self._program_generic_table
                if module_id == self._module_id
            )
        if self._program_type_table is not None:
            names.update(
                "::".join((*scope_path, name))
                for module_id, scope_path, name in self._program_type_table
                if module_id == self._module_id
            )
        own_names = frozenset(names)
        if self._sealed:
            self._sealed_own_source_type_names = own_names
        return own_names

    def resolve_enum_owner_form(
        self,
        kind: EnumOwnerFormKind,
        owner_name: str,
        module_qualifier: QualifierChain | None = None,
        *,
        span: SourceSpan | None = None,
    ) -> EnumOwnerForm | None:
        """Resolve one exact enum-owner source form through checked visibility."""
        source_module_id: ModuleId
        source_scope_path: ScopePath
        source_name: str
        expected_qualifier: tuple[str, ...] | None
        if kind in (EnumOwnerFormKind.LOCAL, EnumOwnerFormKind.SELF):
            if owner_name not in self._own_source_type_names():
                return None
            source_module_id = self._module_id
            source_scope_path = ()
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
            qname = type_qnames[0]
            source_module_id, source_scope_path, source_name = self._qname_decl_key(qname)
            expected_qualifier = None
        else:
            # kind is EnumOwnerFormKind.QUALIFIED_IMPORT here (the only
            # remaining kind): its one caller (_check_module_qualified_variant)
            # chooses it exactly when module_qualifier has route segments, and
            # only ever calls with a real import environment present.
            assert self._import_env is not None
            assert module_qualifier is not None and module_qualifier.route_segments
            # A qualified enum spelling must preserve the shared resolver's
            # verdict.  In particular, an unknown route or ambiguity is not a
            # statement that the subject is "not an enum".
            resolved_qname = self._resolve_import_qname(module_qualifier, owner_name, span=span)
            assert resolved_qname is not None
            if not self._is_program_type_candidate(resolved_qname):
                return None
            source_module_id, source_scope_path, source_name = self._qname_decl_key(resolved_qname)
            expected_qualifier = module_qualifier.route_segments
        template = self.source_type_template_qname(
            source_module_id, source_name, scope_path=source_scope_path
        )
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
            for variant in self.type_table.enum_member_names(template.template)
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
                    template = cast(
                        TypeTemplate,
                        self.source_type_template_qname(
                            qname[0], source_name, scope_path=source_scope_path
                        ),
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

    def get_generic_type_from_module(
        self, module_id: ModuleId, name: str, *, scope_path: ScopePath = ()
    ) -> GenericTypeDef | None:
        """Look up a cross-module ``GenericTypeDef`` by structured owner identity."""
        if self._program_generic_table is None:
            return None
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
        if resolved is None or self._program_generic_table is None:
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
            if self._program_generic_table is None:
                return None
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
            opened_atom = _type_path_atom(
                (*tuple(segment.name for segment in qualifier.segments), name)
            )
            opened_key = self._opened_type_key(opened_atom, span)
            if opened_key is not None:
                assert self._program_generic_table is not None
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
        if self._import_env is None or self._program_generic_table is None:
            return None
        qname = self._resolve_import_qname(qualifier, name, span=span, required=False)
        if qname is None:
            return None
        gdef = self._program_generic_table.get(self._qname_decl_key(qname))
        if gdef is None:
            return None
        rendered = qualifier.render()
        qualified_name = f"{rendered}::{name}"
        return qualified_name, gdef

    def get_ctor_sig_from_module(
        self,
        module_id: ModuleId,
        owner_name: str,
        *,
        scope_path: ScopePath = (),
    ) -> ConstructorSignature | None:
        """Look up a cross-module constructor signature by structured owner identity."""
        if self._program_ctor_sig_table is None:
            return None
        return self._program_ctor_sig_table.get(
            self._constructor_key(module_id, owner_name, scope_path)
        )

    def register_constructor_field_kinds(
        self,
        owner_name: str,
        fields: tuple[tuple[str, ParamZone], ...],
        *,
        scope_path: ScopePath = (),
        module_id: ModuleId | None = None,
        decl_id: int | None = None,
    ) -> None:
        """Register ordered field kinds under structured owner identity.

        *module_id* defaults to this environment's own module; a caller
        registering a ``builtin`` declaration passes its own declaring module
        explicitly too, matching the owner every other piece of that
        declaration's state (its ``TypeDef``, its handle) already uses — a
        ``builtin`` declaration belongs to the module that declares it, like
        every other declaration.
        """
        self._assert_mutable()
        owner_module_id = self._module_id if module_id is None else module_id
        key = self._constructor_key(owner_module_id, owner_name, scope_path)
        self._constructor_field_kinds[key] = fields
        if decl_id is not None:
            self._constructor_field_kinds_by_decl_id[decl_id] = fields
        self._record_fact(
            ConstructorFieldKindsFact(
                owner_name=owner_name,
                fields=fields,
                scope_path=scope_path,
                module_id=owner_module_id,
                decl_id=decl_id,
            )
        )

    def get_constructor_field_kinds(
        self,
        owner_name: str,
        *,
        module_id: ModuleId | None = None,
        scope_path: ScopePath = (),
    ) -> tuple[tuple[str, ParamZone], ...] | None:
        """Return the ordered field-kind pairs for a constructor, or ``None`` if unknown.

        First checks the own-module registry; falls back to the cross-module graph table
        when ``module_id`` is provided (used for cross-module constructor calls where the
        owner type carries its declaring module's ``module_id``).  The owner's named
        scope is supplied structurally, exactly as for constructor signatures.
        """
        owner_module_id = self._module_id if module_id is None else module_id
        key = self._constructor_key(owner_module_id, owner_name, scope_path)
        result = self._constructor_field_kinds.get(key)
        if result is not None:
            return result
        if module_id is not None and self._program_ctor_field_kinds_table is not None:
            return self._program_ctor_field_kinds_table.get(key)
        return None

    def get_constructor_field_kinds_for_type(
        self, typ: Type | None, owner_name: str
    ) -> tuple[tuple[str, ParamZone], ...] | None:
        """Return field-kinds for a constructor identified by its resolved owner *typ*.

        ``RecordType``, ``EnumType``, and ``ExceptionType`` all carry their own
        ``module_id``, so the owning module is read directly off the handle —
        no caller-supplied module id is needed.  Exception field kinds are
        derived directly from ``type_table.field_kinds``, which flattens the
        ``extends`` base chain (base kinds first, then own kinds, each
        honoring its declaration's ``@arg-*`` attribute — exactly like a
        record's fields), rather than through the registered-kinds table
        records/enums use, since an exception's kinds are never pre-registered
        (see ``TypeEnvironment.__init__``).
        """
        if isinstance(typ, ExceptionType):
            return self._type_table.field_kinds(typ)
        assert isinstance(typ, RecordType), f"unexpected constructor owner type {typ!r}"
        by_identity = self._constructor_field_kinds_by_decl_id.get(typ.decl_id)
        if by_identity is not None:
            return by_identity
        return self.get_constructor_field_kinds(
            typ.name, module_id=typ.module_id, scope_path=typ.scope_path
        )

    def all_constructor_field_kinds(
        self,
    ) -> list[tuple[DeclKey, tuple[tuple[str, ParamZone], ...]]]:
        """Return all own-module constructor field-kind entries as (key, kinds) pairs."""
        return list(self._constructor_field_kinds.items())

    def all_generic_types(self) -> dict[str, GenericTypeDef]:
        """Return the own-module generic type map (name → GenericTypeDef)."""
        return self._generic_types

    def all_constructor_sigs(self) -> list[tuple[DeclKey, ConstructorSignature]]:
        """Return all own-module constructor signatures as (key, sig) pairs."""
        return list(self._constructor_sigs.items())

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

        *merge_type_table* is ``False`` for a caller whose ``_type_table`` IS
        *other*'s prior seeding target (the SAME shared instance, not a
        separate table merged from it) — a program check that seeds its
        entry module's environment more than once within one ``check_program``
        run. By the second seeding, this table has already registered the
        current entry's own fresh declarations on top of what *other* (a
        snapshot from BEFORE this entry ran) knows, so re-merging *other*'s
        name index would regress a name this entry just redeclared back onto
        its superseded owner; every other table this method copies is
        per-environment state that a second seeding still needs.
        """
        self._assert_mutable()
        if not other.is_sealed:
            raise AssertionError("cannot seed from an unsealed type environment")
        incoming_type_names = (
            {name for name in other._types if name not in _BUILTIN_FALLBACK_TYPE_NAMES}
            | set(other._alias_targets)
            | set(other._generic_types)
            | set(other._alias_type_params)
        )
        incoming_type_names |= {
            "::".join((*scope_path, owner_name))
            for (module_id, scope_path, owner_name) in other._constructor_sigs
            if module_id == self._module_id
        }
        incoming_type_names |= {
            "::".join((*scope_path, owner_name))
            for (module_id, scope_path, owner_name) in other._constructor_field_kinds
            if module_id == self._module_id
        }
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
        self._constructor_sigs.update(other._constructor_sigs)
        self._constructor_field_kinds.update(other._constructor_field_kinds)
        self._constructor_field_kinds_by_decl_id.update(other._constructor_field_kinds_by_decl_id)
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

        *names* normally comes from this entry's OWN declarations (see the
        REPL's promotion bookkeeping). A failed enum redeclaration also adds
        previously retained inline-member names whose namespace metadata the
        new enum cleared, so those survivors are restored in the same staged
        pass. A reserved built-in exception/prelude name can appear only when
        this entry itself wrote a ``builtin`` declaration of that name — the
        reserved names are non-shadowable other than by one. That declaration
        is rolled back exactly like any other unpromoted one below: no
        special-casing is needed, and none is applied, unlike :meth:`seed_from`,
        which instead has to tell a program's own carried-forward declaration
        apart from the canonical default it must not clobber.

        The shared ``type_table`` is keyed by declaration identity, not name,
        so the unpromoted declaration stays registered under its own identity
        exactly as a superseded one does — the link image derives its nominal
        descriptors from this table on every lowering and needs the entry to
        correct the descriptor the failed entry already linked. What does
        need saying is that the declaration never took effect
        (:meth:`TypeTable.orphan`): unlike a superseded declaration, whose
        surviving values keep its members meaningful, an unpromoted one must
        answer no whole-table query about what the session declares. The
        previous declaration's own identity, methods, and base chain were
        never touched by the redeclaration, so restoring it is just
        ``register`` below reclaiming its name.
        """
        self._assert_mutable()
        for name in names:
            self.unregister_name(name)
            scope_path, declared_name = _split_scoped_type_name(name)
            # Read the unpromoted declaration before ``register`` repoints the
            # name index at the survivor; the seeded table resolves this name
            # to the checked entry's own declaration, which is the one being
            # rolled back.
            unpromoted = self._type_table.get(self._module_id, declared_name, scope_path)
            typedef = other._type_table.get(other._module_id, declared_name, scope_path)
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
            if name in other._types:
                self._types[name] = other._types[name]
            if name in other._alias_targets:
                self._alias_targets[name] = other._alias_targets[name]
            if name in other._resolved_aliases:
                self._resolved_aliases[name] = other._resolved_aliases[name]
            if name in other._generic_types:
                self._generic_types[name] = other._generic_types[name]
            if name in other._alias_type_params:
                self._alias_type_params[name] = other._alias_type_params[name]
            for key, sig in other._constructor_sigs.items():
                if key == (other._module_id, scope_path, declared_name):
                    self._constructor_sigs[key] = sig
            for key, kinds in other._constructor_field_kinds.items():
                if key == (other._module_id, scope_path, declared_name):
                    self._constructor_field_kinds[key] = kinds
