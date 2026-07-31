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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar

from agm.agl.diagnostics import AglError, Diagnostic
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.semantics.types import EnumType
from agm.agl.syntax.nodes import (
    AgentDecl,
    EnumDef,
    ExceptionDef,
    FuncDef,
    ImportItem,
    Program,
    QualifierChain,
    RecordDef,
    TypeAlias,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import AppliedT, ImportMode, NameT

ScopePath = tuple[str, ...]
BareAtom = str | ScopePath
AgentKey = tuple[ScopePath, str]
DeclarationKey = tuple[ModuleId, ScopePath, str]

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
    ``PARSE_JSON``
        ``parse_json(text)`` — parses a ``text`` value as strict JSON; yields
        ``json``.
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
    PARSE_JSON = "PARSE_JSON"
    COPY = "COPY"
    SHALLOW_COPY = "SHALLOW_COPY"


# The single source of truth for the built-in call names and their kinds.
# The resolver classifies calls by this mapping; the checker and any other
# layer that needs the set of built-in names derives it from here.
BUILTIN_CALL_NAMES: dict[str, BuiltinKind] = {
    "print": BuiltinKind.PRINT,
    "render": BuiltinKind.RENDER,
    "exec": BuiltinKind.EXEC,
    "ask": BuiltinKind.ASK,
    "ask-request": BuiltinKind.ASK_REQUEST,
    "parse_json": BuiltinKind.PARSE_JSON,
    "copy": BuiltinKind.COPY,
    "shallow_copy": BuiltinKind.SHALLOW_COPY,
}


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
        A ``param`` declaration or function/lambda parameter (immutable).
    ``catch_binder``
        The binder introduced by a ``catch e`` clause (immutable, branch-local).
    ``pattern_binding``
        A variable introduced by a ``case``/``match`` pattern (immutable).
    ``function_binding``
        A top-level ``def`` declaration (immutable value binding).
    ``agent_binding``
        An ``agent`` declaration (immutable value binding of type ``agent``).
    ``builtin_var_binding``
        A ``builtin var`` declaration (mutable, engine-backed setting; readable
        and assignable with ``:=``).
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
    agent_binding = "agent_binding"
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
    BinderKind.agent_binding: "it is an agent binding",
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
    """A resolved constructor reference's owner metadata.

    ``owner_name``
        The record or enum TYPE name.
    ``variant``
        The enum variant name; ``None`` for a record constructor.
    ``owner_decl_node_id``
        The ``node_id`` of the ``RecordDef`` / ``EnumDef`` that declares this
        constructor.
    ``type_params``
        The owner's declared type parameters (empty tuple if non-generic).
    ``owner_module_id``
        Semantic owner module, retained so same-named constructors from
        different modules remain distinguishable in resolver side tables.
    ``can_match_bare_pattern``
        Whether a bare pattern name can directly match this candidate. This is
        true precisely for a known nullary enum variant, and is retained for
        scope's field-directed duplicate-binder decision.
    """

    owner_name: str
    variant: str | None
    owner_decl_node_id: int
    type_params: tuple[str, ...]
    owner_module_id: ModuleId = ENTRY_ID
    can_match_bare_pattern: bool = False
    owner_path: ScopePath = ()

    def matches(self, enum_type: EnumType, variant: str) -> bool:
        """Whether this reference denotes *variant* of *enum_type*.

        Constructor identity is structured: same-named constructors from
        different modules or scope paths are distinct, so module, owner path,
        enum name, and variant spelling must all agree.
        """
        return (
            self.owner_module_id == enum_type.module_id
            and self.owner_path == enum_type.scope_path
            and self.owner_name == enum_type.name
            and self.variant == variant
        )


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
        this binding.  For module resolution (``resolve()``) and all
        local bindings, this is always :data:`~agm.agl.modules.ids.ENTRY_ID`.
        For cross-module resolution via ``resolve_program()``, cross-module
        references carry the owning library module's id.
    ``scope_path``
        The named scope path that owns this binding. The empty path is the
        module root.
    ``slot_id``
        The :class:`PatternSlot` id when this reference is a field-directed
        pattern slot, or ``None`` for an ordinary resolved binding.
    """

    name: str
    mutable: bool
    decl_span: SourceSpan
    decl_node_id: int
    kind: BinderKind
    module_id: ModuleId = ENTRY_ID
    scope_path: ScopePath = ()
    slot_id: int | None = None


# ---------------------------------------------------------------------------
# LocalOpenSelection — a local ``open``'s target and selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocalOpenSelection:
    """A local ``open``'s target scope path and its using/hiding/rename selection.

    Recorded on the ``ScopeNode`` that contains the ``open`` and consulted
    live at every later bare reference, rather than snapshotted once at the
    ``open`` itself: a local scope's target may still be gaining members --
    header placement forces the ``open`` to precede its own scope's binders
    -- so ``mode``/``items`` (mirroring ``OpenDecl``) are reapplied against
    the target's current member set each time, via
    :func:`apply_open_selection`. Set semantics on the field that holds these
    (``ScopeNode.opened_local_scopes``) make a repeated identical ``open`` a
    no-op rather than an accumulating duplicate.
    """

    scope_path: ScopePath
    mode: ImportMode
    items: tuple[ImportItem, ...]


# ---------------------------------------------------------------------------
# ScopeNode — one node in the lexical scope tree
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ScopeNode:
    """A lexical scope in the scope tree.

    Each ``ScopeNode`` tracks:
    - ``members``: static declarations owned by this named scope path.
    - ``bindings``: lexical value bindings introduced *directly* in this scope.
    - ``parent``: the enclosing scope (``None`` for the root scope).
    - ``node_id``: the ``node_id`` of the AST construct that opened this scope.
    - ``bare_contributions``/``bare_constructor_contributions``: a cross-module
      ``open``'s selection, snapshotted eagerly -- an imported module's public
      members are complete before the walk starts, so there is nothing to
      defer.
    - ``opened_local_scopes``: every local ``open`` recorded here, target path
      together with its full selection. A local scope's members are resolved
      live against the scope tree (see :func:`resolve_bare_contribution` and
      :func:`resolve_bare_constructor_contribution`), never snapshotted, so a
      member registered after the ``open`` -- header placement requires the
      ``open`` to precede its own scope's binders -- is reached exactly like
      one registered before it.

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
    opened_local_scopes: set[LocalOpenSelection] = field(default_factory=set)

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

    def define(self, name: str, ref: BindingRef) -> None:
        """Add *name* → *ref* to this scope's binding table."""
        self.bindings[name] = ref


_T = TypeVar("_T")


def apply_open_selection(
    mode: ImportMode, items: tuple[ImportItem, ...], target: Mapping[BareAtom, _T]
) -> dict[BareAtom, list[_T]]:
    """Apply one ``open``'s using/hiding/rename selection to *target*.

    Defines wildcard, ``using`` (with optional rename), and ``hiding``
    selection in exactly one place, shared by the eager cross-module ``open``
    path -- whose *target* is a module's complete public member set -- and
    the live local-``open`` path -- whose *target* is one scope's current
    member set, materialized by :func:`local_scope_bindings`. Existence of a
    ``using``/``hiding`` item is the caller's concern: the eager path
    validates it against a complete target immediately, while the live path
    simply omits an item nothing yet matches, leaving it to be resolved (or
    not) by a later reference.

    Each exposed atom maps to *every* value it draws from *target*, not just
    the last: two ``using`` items renamed to the same exposed atom (or, for
    ``ImportMode.ALL``, two members already colliding under *target*) is a
    genuine ambiguity, deferred to the atom's first use rather than reported
    here or silently resolved by overwrite.
    """
    if mode is ImportMode.ALL:
        return {atom: [value] for atom, value in target.items()}

    def relative_path(atom: BareAtom) -> ScopePath:
        return (atom,) if isinstance(atom, str) else atom

    selected_paths: set[ScopePath] = set()
    for item in items:
        prefix = (*tuple(segment.name for segment in item.scope_path), item.name)
        if any(relative_path(atom)[: len(prefix)] == prefix for atom in target):
            selected_paths.add(prefix)
    if mode is ImportMode.USING:
        result: dict[BareAtom, list[_T]] = {}
        for item in items:
            prefix = (*tuple(segment.name for segment in item.scope_path), item.name)
            for atom, value in target.items():
                path = relative_path(atom)
                if path[: len(prefix)] != prefix:
                    continue
                exposed_path = (
                    (item.rename, *path[len(prefix) :]) if item.rename is not None else path
                )
                exposed: BareAtom = exposed_path[0] if len(exposed_path) == 1 else exposed_path
                result.setdefault(exposed, []).append(value)
        return result
    return {
        atom: [value]
        for atom, value in target.items()
        if not any(relative_path(atom)[: len(prefix)] == prefix for prefix in selected_paths)
    }


def local_scope_bindings(
    scope_nodes: Mapping[ScopePath, ScopeNode], scope_path: ScopePath
) -> dict[BareAtom, BindingRef]:
    """Materialize *scope_path*'s current member subtree, relative to itself.

    Every member declared anywhere under *scope_path*, keyed by its path
    relative to it. Recomputed from *scope_nodes* on every call rather than
    cached: mid-resolution, a member registered after the ``open`` that
    targets *scope_path* becomes visible the moment the walk defines it;
    once resolution has finished, the computation simply reads a completed
    tree.
    """
    members: dict[BareAtom, BindingRef] = {}
    for path, node in scope_nodes.items():
        if path[: len(scope_path)] != scope_path:
            continue
        for name, ref in node.members.items():
            relative = (*path[len(scope_path) :], name)
            atom: BareAtom = relative[0] if len(relative) == 1 else relative
            members[atom] = ref
    return members


def resolve_bare_contribution(
    scope: ScopeNode, name: BareAtom, scope_nodes: Mapping[ScopePath, ScopeNode]
) -> set[BindingRef] | None:
    """Return the nearest region's bare candidates for *name*, live where needed.

    Walks *scope* outward. At each layer, a cross-module ``open``'s eager
    ``bare_contributions`` snapshot and every recorded local ``open``'s live
    selection (reapplied against *scope_nodes*'s current member set) merge;
    the first layer with any candidate wins, matching nearest-layer
    precedence. Shared by the resolver's own bare-reference resolution,
    mid-walk, and by any later pass over a completed ``ModuleResolution``
    (e.g. bare type-name resolution), so an ``open``'s contribution reads
    identically wherever it is consulted.
    """
    layer: ScopeNode | None = scope
    while layer is not None:
        candidates = set(layer.bare_contributions.get(name, ()))
        for selection in layer.opened_local_scopes:
            target = local_scope_bindings(scope_nodes, selection.scope_path)
            selected = apply_open_selection(selection.mode, selection.items, target)
            candidates.update(selected.get(name, ()))
        if candidates:
            return candidates
        layer = layer.parent
    return None


def resolve_bare_constructor_contribution(
    scope: ScopeNode,
    name: BareAtom,
    scope_nodes: Mapping[ScopePath, ScopeNode],
    declaring_candidates: Callable[[str, BindingRef], tuple[ConstructorRef, ...]],
) -> set[ConstructorRef] | None:
    """Return the nearest region's constructor candidates for *name*, live where needed.

    Mirrors :func:`resolve_bare_contribution`: a cross-module ``open``'s
    eager ``bare_constructor_contributions`` snapshot and each local
    ``open``'s live selection merge at the same layer. A local candidate's
    constructor identity comes from *declaring_candidates*, called on its
    resolved binding -- the same structured lookup an ordinary reference
    already uses -- rather than a separate constructor snapshot.
    """
    layer: ScopeNode | None = scope
    while layer is not None:
        candidates = set(layer.bare_constructor_contributions.get(name, ()))
        for selection in layer.opened_local_scopes:
            target = local_scope_bindings(scope_nodes, selection.scope_path)
            selected = apply_open_selection(selection.mode, selection.items, target)
            last_segment = name if isinstance(name, str) else name[-1]
            for ref in selected.get(name, ()):
                if ref.kind is BinderKind.constructor_binding:
                    candidates.update(declaring_candidates(last_segment, ref))
        if candidates:
            return candidates
        layer = layer.parent
    return None


# ---------------------------------------------------------------------------
# ModuleResolution — output of the scope pass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModuleResolution:
    """Output of the scope resolution pass.

    ``program``
        The original ``Program`` AST node (never mutated).
    ``resolution``
        Maps every ``VarRef.node_id`` and every bare-name ``AssignStmt.node_id``
        to the ``BindingRef`` it resolved to. An ``AssignStmt`` with an indexed
        target has no entry: its object and index are resolved as ordinary
        expressions instead.
    ``builtin_calls``
        Maps every ``Call.node_id`` whose callee is a built-in name
        (``print``/``exec``/``ask``/``ask-request``) to its ``BuiltinKind``.  Calls whose
        callee resolves to a user-defined binding have no entry here.
    ``root_scope``
        The root ``ScopeNode`` (tree root).  Nested scopes are linked via
        ``ScopeNode.parent``.
    ``declared_agents``
        Maps each declared agent's ``(scope_path, name)`` identity to its
        :class:`AgentDecl` node. Always populated by the resolver.
    ``declared_functions``
        Maps each top-level ``def`` name to its :class:`FuncDef` node.
        Populated in the pre-pass; useful for downstream typecheck and eval.
    ``program_name``
        The source-declared program name from a ``program NAME`` declaration,
        or ``None`` when undeclared.
    ``warnings``
        Non-fatal scope-pass diagnostics (severity ``"warning"``), e.g. an
        agent that is declared but never referenced.  Empty by default.
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
    ``pattern_constructor_candidates``
        Maps bare ``VarPattern`` and constructor-pattern node ids to viable
        candidates. Constructor patterns retain an empty tuple when their
        named owner is unavailable, preventing fallback to an unqualified
        spelling. Candidates are independent of ordinary value bindings; the
        checker selects a bare name's final interpretation from the matched
        occurrence's type and field name.
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
    """

    program: Program
    resolution: dict[int, BindingRef]
    builtin_calls: dict[int, BuiltinKind]
    root_scope: ScopeNode
    declarations: dict[DeclarationKey, BindingRef] = field(default_factory=dict)
    scope_nodes: dict[ScopePath, ScopeNode] = field(default_factory=dict)
    declared_agents: dict[AgentKey, AgentDecl] = field(default_factory=dict)
    declared_functions: dict[str, FuncDef] = field(default_factory=dict)
    program_name: str | None = None
    warnings: tuple[Diagnostic, ...] = ()
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
    pattern_slots: dict[int, PatternSlot] = field(default_factory=dict)
    match_site_pattern_slots: dict[int, tuple[int, ...]] = field(default_factory=dict)
    method_declarations: dict[DeclarationKey, ScopePath] = field(default_factory=dict)

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
