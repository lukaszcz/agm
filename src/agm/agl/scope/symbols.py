"""Symbol and scope-tree data types for the AgL resolution pass.

Data model
----------
- ``BindingRef`` — a resolved variable reference: which scope introduced the
  binding, whether it is mutable, and its declaration span.
- ``ConstructorRef`` — metadata about a resolved constructor reference (record
  or enum variant).
- ``PatternSlot`` — scope-created metadata for a shared branch binding whose
  final meaning is selected by type checking.
- ``method_declarations`` — receiver-owning type paths keyed by structured method identities.
- ``ScopeNode`` — a node in the scope tree (one per scope-introducing
  construct).  The root ``ScopeNode`` is always present; nested scopes form a
  tree for visibility analysis.
- ``ModuleResolution`` — the output of the scope pass: the original
  ``Program`` plus side tables.
- ``BuiltinKind`` — enum classifying a built-in Call node.
- ``AglScopeError`` — fatal scope error raised by the resolver.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias as TypingTypeAlias

from agm.agl.diagnostics import AglError
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.semantics.types import EnumType, RecordType, TypeVarType
from agm.agl.syntax.nodes import (
    EnumDef,
    ExceptionDef,
    ExportItem,
    FuncDef,
    ImportItem,
    Program,
    QualifierChain,
    RecordDef,
    TypeAlias,
    UseDecl,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import AppliedT, NameT
from agm.agl.zones import ParamZone

ScopePath = tuple[str, ...]
BareAtom = str | ScopePath
DeclarationKey = tuple[ModuleId, ScopePath, str]
QName: TypingTypeAlias = tuple[ModuleId, BareAtom]
BareRoute: TypingTypeAlias = tuple[ModuleId, ScopePath]


def to_bare_path(atom: BareAtom) -> ScopePath:
    """Normalize a bare atom to its structured path."""
    return (atom,) if isinstance(atom, str) else atom


def to_bare_atom(path: ScopePath) -> BareAtom:
    """Keep a single-segment path compact as a bare name, else structured.

    The one definition of the root-atom-is-a-bare-string representation that
    every scoped-name table shares; :func:`to_bare_path` is its inverse.
    """
    return path[0] if len(path) == 1 else path


def import_item_path(item: ImportItem | ExportItem) -> ScopePath:
    """The selection prefix an import or export item names: scope path plus name."""
    return (*(segment.name for segment in item.scope_path), item.name)


# ---------------------------------------------------------------------------
# BuiltinKind — classification of a built-in Call node
# ---------------------------------------------------------------------------


class BuiltinKind(enum.Enum):
    """Classification of a resolved built-in call.

    Attached to ``Call.node_id`` in ``ModuleResolution.builtin_calls`` when the
    callee is one of the special built-in names.

    ``PRINT``
        ``print(expr)`` — outputs a value; yields ``unit``.
    ``RENDER``
        ``render(expr)`` — renders a value to ``text``.
    ``EXEC``
        ``exec(command, ...)`` — shell execution; yields ``ExecResult`` or
        a context-typed value.
    ``ASK``
        ``ask(prompt, ...)`` — invokes an agent; yields a context-typed value.
    ``ASK_REQUEST``
        ``ask-request[prompt, ...)`` — builds the ``AgentRequest`` that the
        corresponding ``ask`` call would dispatch, without invoking the agent;
        yields an ``AgentRequest`` record.
    ``COPY``
        ``copy(value)`` — deep copy, preserving sharing; yields the same type
        as its argument.
    ``SHALLOW_COPY``
        ``shallow_copy(value)`` — one-level copy; yields the same type as its
        argument.
    """

    PRINT = "PRINT"
    RENDER = "RENDER"
    EXEC = "EXEC"
    ASK = "ASK"
    ASK_REQUEST = "ASK_REQUEST"
    COPY = "COPY"
    SHALLOW_COPY = "SHALLOW_COPY"
    RESOURCE = "RESOURCE"
    RESOURCE_DIR = "RESOURCE_DIR"


class BuiltinStaticKind(enum.Enum):
    """Type-scoped built-ins classified for checking and lowering."""

    SESSION_OPEN = "SESSION_OPEN"
    SESSION_DEFAULT = "SESSION_DEFAULT"


# The single source of truth for the built-in call names and their kinds.
# The resolver classifies calls by this mapping; the checker and any other
# layer that needs the set of built-in names derives it from here.
BUILTIN_CALL_NAMES: dict[str, BuiltinKind] = {
    "print": BuiltinKind.PRINT,
    "render": BuiltinKind.RENDER,
    "exec": BuiltinKind.EXEC,
    "ask": BuiltinKind.ASK,
    "ask-request": BuiltinKind.ASK_REQUEST,
    "copy": BuiltinKind.COPY,
    "shallow-copy": BuiltinKind.SHALLOW_COPY,
    "resource": BuiltinKind.RESOURCE,
    "resource-dir": BuiltinKind.RESOURCE_DIR,
}

# Built-in statics are registered by their owning nominal's declaration path,
# so similarly named user types and enum variants remain ordinary
# declarations. Their final segments therefore remain ordinary names. The
# owning module is not part of the key: whichever standard-library module
# declares the built-in nominal owns its statics, and the host's reserved
# fallback identity owns them when no source declares it.
BUILTIN_TYPE_STATICS: dict[ScopePath, dict[str, BuiltinStaticKind]] = {
    ("Session",): {
        "open": BuiltinStaticKind.SESSION_OPEN,
        "default": BuiltinStaticKind.SESSION_DEFAULT,
    },
}


#: The scope paths of every nominal that owns built-in statics, for callers
#: that have resolved a relative path but not yet its owning module.
BUILTIN_TYPE_STATIC_OWNER_PATHS: frozenset[ScopePath] = frozenset(BUILTIN_TYPE_STATICS)


def builtin_type_static_kind(
    owner_module_id: ModuleId, owner_path: ScopePath, name: str
) -> BuiltinStaticKind | None:
    """Return the static kind registered for an owning nominal identity."""
    if not owner_module_id.owns_standard_builtins:
        return None
    return BUILTIN_TYPE_STATICS.get(owner_path, {}).get(name)


def is_builtin_type_static_owner(owner_module_id: ModuleId, owner_path: ScopePath) -> bool:
    """Return whether a nominal identity owns registered built-in statics."""
    return owner_module_id.owns_standard_builtins and owner_path in BUILTIN_TYPE_STATICS


BUILTIN_CALL_DISPLAY_NAMES: dict[BuiltinKind | BuiltinStaticKind, str] = {
    **{kind: name for name, kind in BUILTIN_CALL_NAMES.items()},
    BuiltinStaticKind.SESSION_OPEN: "Session::open",
    BuiltinStaticKind.SESSION_DEFAULT: "Session::default",
}


def is_qualified_function_member(module_has_identity: bool, scope_path: ScopePath) -> bool:
    """Return whether a function is reachable only through a qualification.

    A builtin spelling is reserved in a bare namespace, which is all a module
    with no module identity of its own has. A named module's members and a
    named scope's members each have an independent qualified namespace, so the
    spelling is free there -- and a named module keeps that namespace whether
    it is the selected entry or one of its library imports.
    """
    return module_has_identity or bool(scope_path)


# ---------------------------------------------------------------------------
# BinderKind — how a binding was introduced
# ---------------------------------------------------------------------------


class BinderKind(enum.Enum):
    """How an immutable (or mutable) binding was introduced.

    Used to phrase a precise ``:=`` rejection message that names the ACTUAL
    binder kind, rather than always blaming ``let``.

    ``let_binding``
        A ``let`` declaration (immutable).
    ``var_binding``
        A ``var`` declaration (mutable).
    ``param_binding``
        A function or lambda parameter (immutable).
    ``catch_binder``
        The binder introduced by a ``catch e`` clause (immutable, branch-local).
    ``pattern_binding``
        A variable introduced by a ``case``/``match`` pattern (immutable).
    ``function_binding``
        A top-level ``def`` declaration (immutable value binding).
    ``builtin_var_binding``
        A mutable, host-backed ``builtin var`` declaration, readable and
        assignable with ``:=``. ``std/config`` bindings are engine settings;
        other standard-library modules may own independent bindings.
    ``constructor_binding``
        A record constructor or enum variant binding (immutable value binding).
    ``loop_var_binding``
        A ``for``-loop iteration variable (immutable, loop-body-local).
    ``pattern_slot``
        A branch-local field-directed pattern binding selected by type checking.
    """

    let_binding = "let_binding"
    var_binding = "var_binding"
    catch_binder = "catch_binder"
    pattern_binding = "pattern_binding"
    function_binding = "function_binding"
    param_binding = "param_binding"
    builtin_var_binding = "builtin_var_binding"
    constructor_binding = "constructor_binding"
    loop_var_binding = "loop_var_binding"
    pattern_slot = "pattern_slot"


# Per-binder phrasing for the ``:=``-on-immutable rejection.  Mutable binder
# kinds (``var_binding``, ``builtin_var_binding``) have no entry: an assignment
# to them is never rejected.
_IMMUTABLE_BINDER_PHRASES: dict[BinderKind, str] = {
    BinderKind.let_binding: "it was declared with 'let'",
    BinderKind.catch_binder: "it is a catch binder",
    BinderKind.pattern_binding: "it is a pattern binding",
    BinderKind.function_binding: "it is a function (def) binding",
    BinderKind.param_binding: "it is a parameter binding",
    BinderKind.constructor_binding: "it is a constructor binding",
    BinderKind.loop_var_binding: "it is a for-loop variable binding",
}


def immutable_binder_phrase(kind: BinderKind) -> str:
    """Return the ``:=``-rejection phrase naming *kind*'s binder."""
    return _IMMUTABLE_BINDER_PHRASES[kind]


def immutable_assignment_message(name: str, kind: BinderKind) -> str:
    """Return the canonical ``:=``-on-immutable rejection message for *name*.

    Type checking is the only caller: a field-directed pattern slot's final
    binding is selected there, so only it can judge an unqualified target.
    The wording lives here beside :func:`immutable_binder_phrase`, which the
    resolver also uses for the cross-module qualified-assignment rejection.
    """
    return (
        f"Cannot assign to '{name}': "
        f"{immutable_binder_phrase(kind)} (immutable). "
        f"Declare with 'var' to make the variable mutable."
    )


def duplicate_binder_message(name: str) -> str:
    """Return the canonical duplicate-pattern-binder rejection for *name*.

    Scope rejects duplicates it can already prove (neither spelling can be a
    bare nullary enum pattern) and checking rejects the rest, so the wording
    lives here and is identical whichever pass reports it.
    """
    return f"Name '{name}' is bound more than once in this pattern."


# ---------------------------------------------------------------------------
# ConstructorRef — metadata about a resolved constructor reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConstructorRef:
    """A resolved constructor reference's canonical record metadata.

    ``owner_name`` and ``owner_path`` identify the record declaration that
    construction produces. Inline enum members use their true scoped record
    path (``Enum::Member``), while standalone records use their declaration
    path. ``owner_decl_node_id`` is that record's nominal identity, including
    the canonical identity of a seeded builtin member.
    ``inline_enum_owner_decl_node_id`` identifies the enum that synthetically
    declared this record; standalone and referenced records leave it unset.
    """

    owner_name: str
    owner_decl_node_id: int
    type_params: tuple[str, ...]
    owner_module_id: ModuleId = ENTRY_ID
    can_match_bare_pattern: bool = False
    owner_path: ScopePath = ()
    is_builtin: bool = False
    inline_enum_owner_decl_node_id: int | None = None

    @classmethod
    def for_member(cls, member: RecordType) -> "ConstructorRef":
        """Build the reference denoting *member*'s own record declaration."""
        return cls(
            owner_name=member.name,
            owner_decl_node_id=member.decl_id,
            type_params=tuple(arg.name for arg in member.type_args if isinstance(arg, TypeVarType)),
            owner_module_id=member.module_id,
            owner_path=member.scope_path,
        )

    def matches(self, enum_type: EnumType, member_name: str) -> bool:
        """Whether this reference denotes *member_name* of *enum_type*."""
        return (
            self.owner_module_id == enum_type.module_id
            and self.owner_path == (*enum_type.scope_path, enum_type.name)
            and self.owner_name == member_name
        )


def dedupe_constructor_candidates(
    candidates: Iterable[ConstructorRef],
) -> tuple[ConstructorRef, ...]:
    """Keep the first occurrence of each distinct candidate, in input order.

    Parser node ids are session-unique in production. Test REPL entries may
    restart their parser's counter, so an id collision only denotes the same
    declaration when its canonical metadata agrees as well -- two candidates
    are the same injected declaration only when they compare equal outright,
    not merely by sharing ``owner_decl_node_id``.
    """
    seen: set[ConstructorRef] = set()
    unique: list[ConstructorRef] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return tuple(unique)


def alias_denotes_constructible_type(
    alias: TypeAlias,
    lookup: Callable[
        [str, QualifierChain | None], RecordDef | EnumDef | ExceptionDef | TypeAlias | None
    ],
) -> bool:
    """Whether *alias* denotes a constructible (non-enum) type.

    A record, exception, or an alias chain that bottoms out at one of those
    has a variant-less constructor; an enum does not (its variants are the
    constructors, not the enum type itself). This follows the alias chain
    (``type B = A`` where ``type A = Color``) through *lookup*, which resolves
    one referenced type name — together with its ``qualifier`` chain, when
    the reference is module-qualified or reaches its target through an import
    — to its declaration, and returns ``False`` only when the chain provably
    ends at an ``EnumDef``.

    *lookup* is scoped to whatever declarations the caller can see (a
    module's own root declarations and import environment, or a whole
    program's public types); a link the callback cannot resolve — no import
    environment available, a container/primitive alias, or an unresolvable
    name — makes the alias presumed constructible, preserving
    today's permissive default. The walk guards against a cyclic chain by
    tracking the ``node_id`` of every declaration visited, not the bare name
    spelling, since a qualified chain can revisit the same name in a
    different module.
    """
    seen = {alias.node_id}
    current = alias
    while True:
        type_expr = current.type_expr
        if not isinstance(type_expr, (NameT, AppliedT)):
            return True
        target = lookup(type_expr.name, type_expr.qualifier)
        if target is None:
            return True
        if target.node_id in seen:
            return True
        seen.add(target.node_id)
        if isinstance(target, EnumDef):
            return False
        if isinstance(target, TypeAlias):
            current = target
            continue
        return True


# ---------------------------------------------------------------------------
# PatternSlot — shared field-directed pattern metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SlotCandidate:
    """Metadata for one source pattern recorded in a :class:`PatternSlot`.

    ``pattern_node_id`` identifies the candidate pattern.
    ``can_match_bare_pattern`` records whether a bare pattern name can
    directly match a known nullary enum variant.
    """

    pattern_node_id: int
    span: SourceSpan
    can_match_bare_pattern: bool = False


@dataclass(frozen=True, slots=True)
class PatternSlot:
    """Parallel metadata for candidates owned by one pattern match site.

    ``match_site_node_id`` identifies the owning case branch or ``let``
    declaration. ``binder_kind`` records the binding kind requested by that
    site when checking selects a candidate. ``alternative`` is an enclosing
    ordinary binding or an outer pattern-slot binding, if one is visible.
    """

    slot_id: int
    name: str
    candidates: tuple[SlotCandidate, ...]
    alternative: BindingRef | None
    match_site_node_id: int
    binder_kind: BinderKind


# ---------------------------------------------------------------------------
# BindingRef — a resolved variable reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BindingRef:
    """A resolved reference to a scope binding.

    ``name``
        The variable name.
    ``mutable``
        ``True`` for ``var`` bindings; ``False`` for all others.
    ``decl_span``
        Source span of the declaration statement (used in error messages).
    ``decl_node_id``
        The ``node_id`` of the declaration node.
    ``kind``
        How the binding was introduced.  Drives the precise ``:=`` rejection
        message so a mutation of a catch binder is not mislabelled as a
        ``let``.
    ``module_id``
        The :class:`~agm.agl.modules.ids.ModuleId` of the module that owns
        this binding: the resolved module's own id for a local binding, and
        the library module's id for a cross-module reference.
    ``scope_path``
        The named scope path that owns this binding. The empty path is the
        module root.
    ``slot_id``
        The :class:`PatternSlot` id when this reference is a field-directed
        pattern slot, or ``None`` for an ordinary resolved binding.
    ``is_builtin``
        Whether the referenced function declaration is host-implemented.
        This provenance survives imports, re-exports, and REPL retention.
    """

    name: str
    mutable: bool
    decl_span: SourceSpan
    decl_node_id: int
    kind: BinderKind
    module_id: ModuleId = ENTRY_ID
    scope_path: ScopePath = ()
    slot_id: int | None = None
    is_builtin: bool = False


# ---------------------------------------------------------------------------
# ScopeNode — one node in the lexical scope tree
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocalUseContribution:
    """A resolved local-scope use contribution retained on its lexical layer.

    ``bindings``/``constructors`` snapshot exactly the bare bindings and
    constructor candidates this contribution exposed once its target's
    members were fully validated (see
    :meth:`_Resolver._validate_local_use_contributions`), mirroring
    :class:`ImportedUseContribution`'s snapshot so both use kinds share the
    same subtract-then-readd retraction protocol on REPL supersession. A
    freshly declared contribution starts with empty snapshots -- they are
    filled in once validation has walked the whole target subtree.
    """

    declaration: UseDecl
    source: ScopeNode
    target: ResolvedUseTarget
    bindings: Mapping[BareAtom, frozenset[BindingRef]] = field(default_factory=dict)
    constructors: Mapping[BareAtom, frozenset[ConstructorRef]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImportedUseContribution:
    """A resolved imported surface retained on its lexical scope layer.

    ``hidden_prefixes`` records the declaration's ``hiding`` clause as
    selection prefixes (see :func:`import_item_path`), so a wildcard-facade
    refresh (:meth:`_Resolver._nearest_bare_contribution_layer`) can skip
    re-adding a name the ``use`` hid instead of reinstating it from the
    import environment.
    """

    target: ResolvedUseTarget
    refreshes_all_members: bool
    members: Mapping[BareAtom, QName]
    scope_routes: Mapping[BareAtom, frozenset[BareRoute]]
    bindings: Mapping[BareAtom, frozenset[BindingRef]]
    constructors: Mapping[BareAtom, frozenset[ConstructorRef]]
    hidden_prefixes: frozenset[ScopePath]


@dataclass(slots=True)
class ScopeNode:
    """A lexical scope in the scope tree.

    Each ``ScopeNode`` tracks:
    - ``members``: static declarations owned by this named scope path.
    - ``bindings``: lexical value bindings introduced *directly* in this scope.
    - ``parent``: the enclosing scope (``None`` for the root scope).
    - ``node_id``: the ``node_id`` of the AST construct that opened this scope.
    - ``bare_contributions``/``bare_constructor_contributions``: selected
      imports snapshotted for this region.

    ``members`` is read freely but written only through the member mutation
    methods on this class.

    Membership is collected before resolution. Lookup walks the lexical binding
    parent chain; member-reference resolution is introduced separately.
    """

    node_id: int
    parent: ScopeNode | None = None
    bindings: dict[str, BindingRef] = field(default_factory=dict)
    scope_path: ScopePath = ()
    members: dict[str, BindingRef] = field(default_factory=dict)
    bare_contributions: dict[BareAtom, set[BindingRef]] = field(default_factory=dict)
    bare_constructor_contributions: dict[BareAtom, set[ConstructorRef]] = field(
        default_factory=dict
    )
    local_use_contributions: list[LocalUseContribution] = field(default_factory=list)
    imported_use_contributions: list[ImportedUseContribution] = field(default_factory=list)

    def lookup(self, name: str) -> BindingRef | None:
        """Search lexical bindings and named-scope members outward."""
        scope: ScopeNode | None = self
        while scope is not None:
            ref = scope.bindings.get(name)
            if ref is None and scope.scope_path:
                ref = scope.members.get(name)
            if ref is not None:
                return ref
            scope = scope.parent
        return None

    def contribute_bare(self, name: BareAtom, ref: BindingRef) -> None:
        """Add one use-site-resolved bare contribution to this region."""
        self.bare_contributions.setdefault(name, set()).add(ref)

    def contribute_bare_constructor(self, name: BareAtom, ref: ConstructorRef) -> None:
        """Add one constructor candidate contributed bare to this region."""
        self.bare_constructor_contributions.setdefault(name, set()).add(ref)

    def contribute_local_use(self, contribution: LocalUseContribution) -> None:
        """Add a local scope use whose source members remain live."""
        self.local_use_contributions.append(contribution)

    def define(self, name: str, ref: BindingRef) -> None:
        """Add *name* → *ref* to this scope's binding table."""
        self.bindings[name] = ref

    def register_member(self, name: str, ref: BindingRef) -> None:
        """Add *name* → *ref* to this scope's member layer.

        This is the only legal way to add a member.
        """
        self.members[name] = ref

    def clear_owned_constructor_members(self, nested_scope_names: frozenset[str]) -> None:
        """Discard an owning type's constructors while retaining nested type members."""
        self.members = {
            name: ref
            for name, ref in self.members.items()
            if ref.kind is not BinderKind.constructor_binding or name in nested_scope_names
        }


def resolve_bare_contribution_layer(
    scope: ScopeNode,
    name: BareAtom,
    *,
    predicate: Callable[[BindingRef], bool] | None = None,
) -> tuple[ScopeNode, set[BindingRef]] | None:
    """Return the nearest region and its bare candidates in one namespace."""
    layer: ScopeNode | None = scope
    while layer is not None:
        stored = layer.bare_contributions.get(name, ())
        selected = set(stored) if predicate is None else {ref for ref in stored if predicate(ref)}
        if selected:
            return layer, selected
        layer = layer.parent
    return None


# ---------------------------------------------------------------------------
# ModuleResolution — output of the scope pass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedUseTarget:
    """Stable semantic identity of one local or imported ``use`` target.

    ``wildcard_facade_origin_node_id`` ties a retained facade use to the
    wildcard declaration that formed it, allowing incremental wildcard
    expansion without adopting modules from another declaration reusing the
    same alias.
    """

    local_path: ScopePath | None = None
    imported_routes: tuple[tuple[ModuleId, ScopePath], ...] = ()
    wildcard_facade_origin_node_id: int | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class ModuleResolution:
    """Output of the scope resolution pass.

    ``program``
        The original ``Program`` AST node (never mutated).
    ``resolution``
        Maps every ``VarRef.node_id`` and every bare-name ``AssignStmt.node_id``
        to the ``BindingRef`` it resolved to. An ``AssignStmt`` with an indexed
        or field target creates no assignment binding: its receiver (and an
        indexed target's index) is resolved as an ordinary expression instead.
    ``builtin_calls``
        Maps every ``Call.node_id`` whose callee is a built-in name
        (``print``/``exec``/``ask``/``ask-request``) to its ``BuiltinKind``.  Calls whose
        callee resolves to a user-defined binding have no entry here.
    ``root_scope``
        The root ``ScopeNode`` (tree root).  Nested scopes are linked via
        ``ScopeNode.parent``.
    ``declared_functions``
        Maps each source-level root ``def`` name to its :class:`FuncDef` node.
        Host-only synthetic entries are excluded. Populated in the pre-pass;
        useful for downstream typecheck and eval.
    ``allows_root_statements``
        Whether this entry is an incremental REPL entry, whose root retains
        executable items instead of enforcing a static module root.
    ``origin_path``
        This module's canonical source file, or ``None`` for a module with no
        backing file (inline sources, REPL entries). Later passes consult it to
        phrase a diagnostic for the host the module actually came from.
    ``declarations``
        Every named declaration keyed by ``(module_id, scope_path, name)``.
        Root declarations use the empty path just like any other scope.
    ``scope_nodes``
        Named scope-region layers keyed by path, rooted at ``()``.
    ``declared_type_names``
        Names of all root-level ``RecordDef`` / ``EnumDef`` / ``TypeAlias``
        declarations.  Used by the existing root-only constructor resolver.
    ``constructor_candidates``
        Maps each constructor name to an ordered tuple of all
        :class:`ConstructorRef` candidates (one per record/enum that declares
        it).  A single entry means the name is unambiguous; two or more mean
        an overload set requiring qualification.
    ``constructor_refs``
        Maps a ``VarRef.node_id`` (or ``Call.node_id`` whose callee was a
        constructor ``VarRef``) to the single :class:`ConstructorRef` it
        resolved to (only present when the candidate set has exactly one entry
        and no nearer non-constructor binding shadows it).
    ``pattern_constructor_candidates`` / ``pattern_constructor_spellings``
        Map bare ``VarPattern`` and constructor-pattern node ids to viable
        candidates. Constructor patterns retain an empty tuple when their
        named owner is unavailable, preventing fallback to an unqualified
        spelling. Candidates are independent of ordinary value bindings; the
        checker selects a bare name's final interpretation from the matched
        occurrence's type and field name. The spelling table preserves each
        immutable occurrence's source name alongside those candidates.
    ``is_test_constructor_candidates``
        Maps unqualified ``is`` test node ids to every visible constructor
        candidate for their source spelling. Typecheck selects by the left
        operand's nominal enum type.
    ``pattern_slots``
        Scope-created field-directed pattern-slot metadata keyed by slot id.
        Branch-body references resolve directly to the shared slot binding.
    ``match_site_pattern_slots``
        Maps each owning case-branch or ``let`` declaration node id to the
        slot ids its pattern created, in creation (outer-to-inner) order. The
        checker selects exactly these after that match site is classified.
    ``method_declarations``
        Maps each method's structured declaration identity to the complete
        scope path of its nominal receiver owner. This is scope's definitive
        receiver classification; later passes consume it without re-deriving
        whether a function is a method.
    ``param_zones``
        The zone of every parameter and field, keyed by ``Param.node_id``, as
        the declaration's ``@arg-*`` attributes resolved it. Typecheck builds
        every ``ParamSpec`` and constructor field list from this table; the
        AST itself carries no zone.
    """

    program: Program
    resolution: dict[int, BindingRef]
    builtin_calls: dict[int, BuiltinKind]
    root_scope: ScopeNode
    builtin_static_calls: dict[int, BuiltinStaticKind] = field(default_factory=dict)
    declarations: dict[DeclarationKey, BindingRef] = field(default_factory=dict)
    scope_nodes: dict[ScopePath, ScopeNode] = field(default_factory=dict)
    declared_functions: dict[str, FuncDef] = field(default_factory=dict)
    allows_root_statements: bool = False
    origin_path: Path | None = None
    declared_type_names: frozenset[str] = frozenset()
    declared_type_paths: frozenset[ScopePath] = frozenset()
    constructor_candidates: dict[str, tuple[ConstructorRef, ...]] = field(default_factory=dict)
    constructor_candidates_by_path: dict[tuple[ScopePath, str], tuple[ConstructorRef, ...]] = field(
        default_factory=dict
    )
    constructor_refs: dict[int, ConstructorRef] = field(default_factory=dict)
    pattern_constructor_candidates: dict[int, tuple[ConstructorRef, ...]] = field(
        default_factory=dict
    )
    pattern_constructor_spellings: dict[int, str] = field(default_factory=dict)
    is_test_constructor_candidates: dict[int, tuple[ConstructorRef, ...]] = field(
        default_factory=dict
    )
    pattern_slots: dict[int, PatternSlot] = field(default_factory=dict)
    match_site_pattern_slots: dict[int, tuple[int, ...]] = field(default_factory=dict)
    method_declarations: dict[DeclarationKey, ScopePath] = field(default_factory=dict)
    use_targets: dict[int, ResolvedUseTarget] = field(default_factory=dict)
    param_zones: dict[int, ParamZone] = field(default_factory=dict)

    def receiver_owner_for(self, module_id: ModuleId, node: FuncDef) -> ScopePath | None:
        """Return scope's receiver classification for *node*, if it has one.

        Every consumer of ``method_declarations`` asks this one question of a
        declaration node, so the structured key is assembled here rather than at
        each call site.
        """
        return self.method_declarations.get(
            (module_id, tuple(segment.name for segment in node.scope_path), node.name)
        )


# ---------------------------------------------------------------------------
# AglScopeError — fatal scope error
# ---------------------------------------------------------------------------


class AglScopeError(AglError):
    """A fatal name-resolution error.

    Raised by the scope resolver on the first static scope violation
    (first-error abort policy).  Carries an optional ``SourceSpan`` for
    precise source location.
    """
