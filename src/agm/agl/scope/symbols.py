"""Symbol and scope-tree data types for the AgL resolution pass.

Data model
----------
- ``BindingRef`` — a resolved variable reference: which scope introduced the
  binding, whether it is mutable, and its declaration span.
- ``ConstructorRef`` — metadata about a resolved constructor reference (record
  or enum variant).
- ``PatternSlot`` — scope-created metadata for a shared branch binding whose
  final meaning is selected by type checking.
- ``method_declarations`` — receiver owners keyed by structured method identities.
- ``ScopeNode`` — a node in the scope tree (one per scope-introducing
  construct).  The root ``ScopeNode`` is always present; nested scopes form a
  tree for visibility analysis.
- ``ModuleResolution`` — the output of the scope pass: the original
  ``Program`` plus side tables.
- ``BuiltinKind`` — enum classifying a built-in Call node.
- ``AglScopeError`` — fatal scope error raised by the resolver; structured
  subclasses distinguish an ambiguous spelling (``AmbiguousQualificationError``,
  ``AmbiguousConstructorError``) from an unresolved one (``UnknownQualifierError``,
  ``UnknownMemberError``) and a name declared twice (``DuplicateDeclarationError``).
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeAlias as TypingTypeAlias

from agm.agl.attributes import ProgramOptionSpec
from agm.agl.diagnostics import AglError, dollar_spacing_hint
from agm.agl.infix import Fixity
from agm.agl.modules.ids import (
    ENTRY_ID,
    ModuleId,
    Reader,
    render_route_member,
    spell_declaration,
    spell_scope_path,
)
from agm.agl.semantics.external_names import ExternalName
from agm.agl.semantics.types import EnumType, ExceptionType, RecordType, TypeVarType
from agm.agl.syntax.nodes import (
    AttributeKeyedArg,
    EnumDef,
    ExceptionDef,
    ExportItem,
    FuncDef,
    ImportItem,
    Program,
    RecordDef,
    TypeAlias,
    UseDecl,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import AppliedT, NameT, TypeExpr, named_builtin_type
from agm.agl.zones import ParamZone

ScopePath = tuple[str, ...]
BareAtom = str | ScopePath
DeclarationKey = tuple[ModuleId, ScopePath, str]
QName: TypingTypeAlias = tuple[ModuleId, BareAtom]


@dataclass(frozen=True, slots=True)
class DeclarationSelection:
    """A spelling selecting the declaration *key* at its full path."""

    key: DeclarationKey


@dataclass(frozen=True, slots=True)
class OwnerMemberSelection:
    """A qualified spelling selecting inline member *member* of type *owner*.

    *owner* is the type its last qualifier segment selects: an enum, or an
    alias whose target enum's member it projects.
    """

    owner: DeclarationKey
    member: str


TypeSelection: TypingTypeAlias = DeclarationSelection | OwnerMemberSelection
"""What scope selected for one spelling (``ModuleResolution.owner_declarations``)."""

BUILTIN_METHOD_RECEIVER_NAMES: frozenset[str] = frozenset(
    {"array", "dict", "text", "json", "int", "decimal", "bool"}
)


@dataclass(frozen=True, slots=True)
class BuiltinMethodReceiver:
    """A builtin method receiver and the method type parameters its head binds.

    ``parameters`` are the indices, among the method's first ``arity`` type
    parameter slots (its head's), of the receiver's type arguments in the
    builtin's order: an array's element, a dict's key and value -- a
    ``dict[text, V]`` receiver has its value's alone, matching only
    text-keyed dicts.
    """

    name: str
    parameters: tuple[int, ...] = ()
    arity: int = 0


@dataclass(frozen=True, slots=True)
class ReceiverOwner:
    """The nominal declaration or builtin scope a method receiver extends.

    ``module_id`` and ``scope_path`` identify a nominal receiver type where it
    was declared, or a builtin receiver scope in the method's declaring module;
    ``builtin`` is set exactly for the latter. The method itself remains keyed
    by its declaring module and scope path in ``method_declarations``.
    """

    module_id: ModuleId
    scope_path: ScopePath
    builtin: BuiltinMethodReceiver | None = None


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
        ``shallow-copy(value)`` — one-level copy; yields the same type as its
        argument.
    ``PARSE``
        ``std/value::parse[T](value)`` — same conversion as ``value as T``
        (see ``semantics.type_table.parse_classification`` for the one
        divergence), raising ``ValueParseError`` on failure instead of
        ``CastError``.
    ``TRY_PARSE``
        ``std/value::try-parse[T](value)`` — ``parse`` wrapped in a
        ``Result[T, ValueParseError]``.

    ``PARSE``/``TRY_PARSE`` are classified only via
    :data:`NON_RESERVED_BUILTIN_CALL_NAMES` (see :func:`builtin_call_kind`):
    unlike every other member here, their bare spelling is an ordinary,
    shadowable name, so a standard-library module (``std/json``, ``std/toml``)
    may declare its own ``parse``/``try-parse`` without collision.
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
    PARSE = "PARSE"
    TRY_PARSE = "TRY_PARSE"


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

# Built-in call names classified only once their reference is already known to
# be a ``builtin def`` (see ``BuiltinKind.PARSE``'s docstring) — NOT part of
# ``_RESERVED_NAMES`` in ``scope/resolver.py``, so a plain function of the
# same bare name (``std/json::parse``, a user's own ``parse``) resolves
# normally instead of being hijacked. Use :func:`builtin_call_kind` to look up
# either map once a name's ``builtin def`` status is already established.
NON_RESERVED_BUILTIN_CALL_NAMES: dict[str, BuiltinKind] = {
    "parse": BuiltinKind.PARSE,
    "try-parse": BuiltinKind.TRY_PARSE,
}


def builtin_call_kind(name: str) -> BuiltinKind | None:
    """Return *name*'s ``BuiltinKind``, reserved or not, or ``None``.

    For a caller that already knows its reference is a ``builtin def`` (a
    resolved binding, or a declaration being validated) and only needs the
    kind that spelling denotes — never for deciding whether a bare name is
    reserved, which stays ``BUILTIN_CALL_NAMES``-only (see
    ``scope/resolver.py::_RESERVED_NAMES``).
    """
    return BUILTIN_CALL_NAMES.get(name, NON_RESERVED_BUILTIN_CALL_NAMES.get(name))


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
    **{kind: name for name, kind in NON_RESERVED_BUILTIN_CALL_NAMES.items()},
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


def immutable_assignment_message(name: str, kind: BinderKind, *, cross_module: bool = False) -> str:
    """Return the canonical ``:=``-on-immutable rejection message for *name*.

    Type checking judges every target scope selects, since a field-directed
    pattern slot's final binding is selected there. *cross_module* drops the
    "declare with 'var'" hint: an importer cannot change how another module
    declared its own binding.
    """
    hint = "" if cross_module else " Declare with 'var' to make the variable mutable."
    return f"Cannot assign to '{name}': {_IMMUTABLE_BINDER_PHRASES[kind]} (immutable).{hint}"


def undefined_name_message(name: str) -> str:
    """Return the canonical undefined-name rejection message for *name*.

    Appends :func:`~agm.agl.diagnostics.dollar_spacing_hint` when *name* looks
    like a verbatim literal written without a space (``exec$ date``).
    """
    hint = dollar_spacing_hint(name) or ""
    return f"'{name}' is not defined.{hint}"


def duplicate_binder_message(name: str) -> str:
    """Return the canonical duplicate-pattern-binder rejection for *name*.

    Scope rejects duplicates it can already prove (neither spelling can be a
    bare nullary enum pattern) and checking rejects the rest, so the wording
    lives here and is identical whichever pass reports it.
    """
    return f"Name '{name}' is bound more than once in this pattern."


def qualification_repair_guidance() -> str:
    """Return the common, source-level repairs for a qualifier ambiguity or clash."""
    return (
        "Use a :: anchor to select the current module, hiding to remove a conflicting member, "
        "a longer suffix or a /-anchored path to select a module, or as to give one import "
        "a distinct name."
    )


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

    A type alias's constructor names the alias itself: ``owner_decl_node_id``
    is the alias declaration, and typecheck follows its checked template.
    ``member`` is set on an alias of an enum and names the member that
    ``Alias::Member`` selects from that template.
    """

    owner_name: str
    owner_decl_node_id: int
    type_params: tuple[str, ...]
    owner_module_id: ModuleId = ENTRY_ID
    can_match_bare_pattern: bool = False
    owner_path: ScopePath = ()
    is_builtin: bool = False
    inline_enum_owner_decl_node_id: int | None = None
    member: str | None = None

    @classmethod
    def for_nominal(cls, nominal: RecordType | ExceptionType) -> "ConstructorRef":
        """Build the reference denoting *nominal*'s own declaration."""
        return cls(
            owner_name=nominal.name,
            owner_decl_node_id=nominal.decl_id,
            type_params=()
            if isinstance(nominal, ExceptionType)
            else tuple(arg.name for arg in nominal.type_args if isinstance(arg, TypeVarType)),
            owner_module_id=nominal.module_id,
            owner_path=nominal.scope_path,
        )

    @classmethod
    def for_alias(
        cls, alias: TypeAlias, module_id: ModuleId, owner_path: ScopePath
    ) -> "ConstructorRef":
        """Build the constructor reference of *alias*, declared at *owner_path*."""
        return cls(
            owner_name=alias.name,
            owner_decl_node_id=alias.node_id,
            type_params=alias.type_params,
            owner_module_id=module_id,
            owner_path=owner_path,
        )

    def matches(self, enum_type: EnumType, member_name: str) -> bool:
        """Whether this reference denotes *member_name* of *enum_type*."""
        return (
            self.owner_module_id == enum_type.module_id
            and self.owner_path == (*enum_type.scope_path, enum_type.name)
            and self.owner_name == member_name
        )

    @property
    def qname(self) -> QName:
        """This reference's declaration path, as a :data:`QName`."""
        return self.owner_module_id, to_bare_atom((*self.owner_path, self.owner_name))

    @property
    def selected_qname(self) -> QName:
        """The path this reference is selected at: an alias's member beneath the alias."""
        if self.member is None:
            return self.qname
        return self.owner_module_id, to_bare_atom((*self.owner_path, self.owner_name, self.member))


def passes_parameters(type_params: tuple[str, ...], target: AppliedT) -> bool:
    """Whether applied *target*'s arguments are exactly *type_params*, in order."""
    return type_params == tuple(
        arg.name if isinstance(arg, NameT) and arg.qualifier is None else None
        for arg in target.args
    )


@dataclass(frozen=True, slots=True)
class TypeTarget:
    """The precise declaration identity an alias's target names.

    A ``QName`` alone is a path, not an identity: a REPL entry can redeclare
    a path, reusing the same ``QName`` for an unrelated later declaration.
    ``decl_node_id`` is the declaration a retained alias actually resolved --
    :attr:`TypeOwner.decl_node_id` of the owner at ``qname`` when the alias
    was declared -- so a later redeclaration at the same path is never
    mistaken for the one the alias still names.
    """

    qname: QName
    decl_node_id: int


@dataclass(frozen=True, slots=True)
class TypeOwner:
    """What a type path selects when it qualifies a constructor, by declaration identity.

    ``decl_node_id`` is this owner's own declaration identity -- the record,
    exception, enum, or alias node's ``node_id`` -- so another owner that
    targets this path (:class:`TypeTarget`) can tell a later redeclaration at
    the same path apart from the declaration it originally resolved.
    ``constructor`` is the constructor the owner's bare spelling names -- a
    record's or exception's own, or an alias's -- and ``None`` for an enum or
    a structural alias. ``names`` holds, for a record or exception target, the
    target's own name and every alias name on the chain leading to it, and for
    an alias presumed constructible its own name; it is empty for an enum or
    structural target, whose owner constructs nothing itself. ``members`` maps
    the names an enum target's scope declares -- its inline members -- to their
    own constructors. ``referenced`` holds the names of the resolved members an
    enum target only references: they keep their own paths, so the owner selects
    none of them, but spelling one is a focused error rather than an unknown
    name. ``alias`` is an alias path's declaration. ``injected`` holds, for an
    enum declaration only, its referenced members' record constructors, whose
    names a root enum injects bare. ``hidden`` holds, for an alias, the
    paths of its target's inline members that the alias's target spelling
    cannot reach where the alias is declared, since a ``hiding`` removed
    them: each a member's name alone, left out of ``members``. ``target`` holds, for an alias
    with a nominal target, the target's identity (:class:`TypeTarget`): what a
    REPL entry retains and never re-selects, however later entries redeclare
    or import around it. ``own_path_referenced`` holds, for
    an enum target, each of ``referenced``'s names that is declared directly
    beneath the enum's own path (``enum Box = ... | Box::Item``): such a name
    selects like a declared member wherever the enum is spelled, through an
    alias too (``B::Item`` with ``type B = Box``). ``arity`` is the number of
    type parameters the declaration itself takes. ``builtin`` is, for an
    alias of a built-in type (directly or through aliases), that type in
    terms of the alias's own type parameters. ``stands_for`` is, for an alias, the type
    expression it stands for where it is declared: its target expression, or
    the argument there of an alias standing for one of its type parameters
    (``Id[Base]`` with ``type Id[T] = T`` stands for ``Base``), or what any
    other generic alias that is a type of its own stands for with its
    arguments there (``P[int]`` with ``type P[T] = Plain`` stands for
    ``Plain``).
    """

    constructor: ConstructorRef | None
    decl_node_id: int
    names: frozenset[str] = frozenset()
    members: Mapping[str, ConstructorRef] = field(default_factory=dict)
    referenced: frozenset[str] = frozenset()
    alias: TypeAlias | None = None
    injected: tuple[ConstructorRef, ...] = ()
    hidden: frozenset[ScopePath] = frozenset()
    target: TypeTarget | None = None
    own_path_referenced: frozenset[str] = frozenset()
    arity: int = 0
    builtin: TypeExpr | None = None
    stands_for: TypeExpr | None = None

    @property
    def renames(self) -> bool:
        """Whether an alias is another name for its target: it passes its type parameters through.

        By the type name it :attr:`stands_for`: ``type A = m::B``, ``type
        A[T] = m::B[T]``, ``type A = Id[m::B]`` and ``type A = P[int]`` with
        ``type P[T] = m::B`` rename ``m::B``; ``type A = m::B[int]`` and
        ``type A[T] = m::B`` are types of their own.
        """
        alias, target = self.alias, self.stands_for
        if isinstance(target, NameT):
            return alias is not None and not alias.type_params
        return (
            alias is not None
            and isinstance(target, AppliedT)
            and passes_parameters(alias.type_params, target)
        )

    @property
    def applies(self) -> bool:
        """Whether an alias applies the type name it stands for to other arguments than its own.

        A built-in's name included: ``type A = m::B[int]`` and ``type A =
        array[int]`` do; ``type A[T] = array[T]`` passes its own parameters
        through, and ``type A = text`` applies nothing.
        """
        alias, target = self.alias, self.stands_for
        if alias is None or target is None:
            return False
        named = target if isinstance(target, (NameT, AppliedT)) else named_builtin_type(target)
        return isinstance(named, AppliedT) and not passes_parameters(alias.type_params, named)

    @property
    def constructs(self) -> bool:
        """Whether the owner qualifies constructors: its own, or its enum members'."""
        return bool(self.names or self.members or self.referenced)

    @property
    def constructible(self) -> bool:
        """Whether the owner's own bare spelling names a constructor.

        True for a record, exception, or constructible alias; false for an
        enum or a structural target.
        """
        return self.constructor is not None and bool(self.names)

    def select(self, name: str, written: str) -> ConstructorRef | None:
        """Return the constructor ``Owner::name`` selects, with the owner spelled *written*.

        A record or exception is also qualified by the owner's written
        spelling, which covers a ``use`` rename. An enum's inline member is
        selected only through an alias, which keeps its own constructor and
        records the member's record name.
        """
        member = self.members.get(name)
        if member is not None and self.constructor is not None:
            return self._selected_member(self.constructor, member)
        if self.names and (name in self.names or name == written):
            return self.constructor
        return None

    def alias_members(self) -> dict[str, ConstructorRef]:
        """An alias's inline members by name, each as ``Alias::name`` selects it."""
        constructor = self.constructor
        if constructor is None:
            return {}
        return {
            name: self._selected_member(constructor, member)
            for name, member in self.members.items()
        }

    @staticmethod
    def _selected_member(constructor: ConstructorRef, member: ConstructorRef) -> ConstructorRef:
        """Inline *member* as the alias whose constructor is *constructor* selects it."""
        return replace(constructor, member=member.owner_name)


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

    ``match_site_node_id`` identifies the owning case branch.
    ``alternative`` is what the name reads outside the pattern -- an
    enclosing binding, an outer pattern-slot binding, or a declaration --
    if anything, and ``alternative_constructor`` the constructor it names.
    ``outside_ambiguity`` is what that reading selects when it is ambiguous.
    """

    slot_id: int
    name: str
    candidates: tuple[SlotCandidate, ...]
    alternative: BindingRef | None
    match_site_node_id: int
    alternative_constructor: ConstructorRef | None = None
    outside_ambiguity: Ambiguity | None = None


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
    ``is_method``
        Whether the function declaration has a ``self`` receiver.
    ``is_param``
        Whether a ``let``/``var`` binding carries the ``@param`` attribute.
        This provenance survives imports, re-exports, and REPL retention, so
        a ``@config`` target check never needs a whole-program node-id set.
    ``is_variant_member``
        Whether this binding is a bare-exposed enum variant's constructor,
        injected as a value/pattern convenience by variant expansion (never
        by a use's or import's own ``members``). Such a binding never
        contributes a type, whatever else shares its declaration route.
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
    is_method: bool = False
    is_param: bool = False
    is_variant_member: bool = False

    @property
    def contributes_a_type(self) -> bool:
        """Whether this binding is eligible to contribute a type.

        The one authoritative check for ``is_variant_member``'s invariant,
        shared by every contribution predicate that must honour it.
        """
        return not self.is_variant_member


# ---------------------------------------------------------------------------
# DeclInfo — pre-pass declaration metadata for cross-module BindingRefs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeclInfo:
    """One declaration's metadata, keyed by ``(module_id, name)`` in a whole-program pre-pass.

    Built before any module's body resolves (see
    :func:`~agm.agl.scope.program.resolve_program`), so a cross-module
    :class:`BindingRef` can be built without waiting on the owning module's
    own resolution to finish.
    """

    decl_node_id: int
    decl_span: SourceSpan
    kind: BinderKind
    is_builtin: bool = False
    is_method: bool = False
    is_param: bool = False


# ---------------------------------------------------------------------------
# ScopeNode — one node in the lexical scope tree
# ---------------------------------------------------------------------------


class ContributionLayer(enum.Enum):
    """Which layer made a candidate visible, recorded where it is contributed.

    ``DECLARED`` is this module's own lexical declaration; ``USE`` a ``use``
    declaration's contribution (local or region-scoped); ``IMPORTED`` an
    import tail's (root or region-scoped) or a module route's. An ambiguity's
    origin kinds are read off these tags (:func:`contribution_origin`), never
    guessed from where a candidate happens to sit.
    """

    DECLARED = enum.auto()
    USE = enum.auto()
    IMPORTED = enum.auto()


Layers: TypingTypeAlias = frozenset[ContributionLayer]
"""Every layer contributing one candidate: one declaration reached several ways is one."""


def add_layers[K](table: dict[K, Layers], key: K, layers: Iterable[ContributionLayer]) -> None:
    """Record that *layers* also contribute *key* to *table*."""
    table[key] = table.get(key, frozenset()).union(layers)


def layered(layers: Layers) -> tuple[ContributionLayer, ...]:
    """Return *layers* in declaration order, for a deterministic walk."""
    return tuple(layer for layer in ContributionLayer if layer in layers)


@dataclass(slots=True)
class ScopeNode:
    """A lexical scope in the scope tree.

    Each ``ScopeNode`` tracks:
    - ``members``: static declarations owned by this named scope path.
    - ``bindings``: lexical value bindings introduced *directly* in this scope.
    - ``parent``: the enclosing scope (``None`` for the root scope).
    - ``node_id``: the ``node_id`` of the AST construct that opened this scope.
    - ``bare_contributions``: selected imports snapshotted for this region.
      Each entry carries every :class:`ContributionLayer` contributing it,
      recorded where contributed.
    - ``uses``: the ``use`` declarations written in this region, in order.
      What one exposes is read where it is written whenever it is used.

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
    bare_contributions: dict[BareAtom, dict[BindingRef, Layers]] = field(default_factory=dict)
    uses: list[UseDecl] = field(default_factory=list)

    def lookup(self, name: str) -> BindingRef | None:
        """Search this layer's and its parents' lexical bindings outward."""
        scope: ScopeNode | None = self
        while scope is not None:
            ref = scope.bindings.get(name)
            if ref is not None:
                return ref
            scope = scope.parent
        return None

    def entry_copy(self) -> "ScopeNode":
        """Copy this layer into a REPL entry's own image.

        ``bindings`` are shared by reference, since they are read-only during
        resolve, while the bare table and uses are copied so the entry's own
        contributions never mutate the session's record. ``members`` starts
        shared too, but ``_build_scope_nodes`` immediately replaces it with a
        fresh dict, re-registering each retained member one at a time, so
        this entry's own registrations never mutate it either.
        """
        return ScopeNode(
            node_id=self.node_id,
            parent=self.parent,
            bindings=self.bindings,
            scope_path=self.scope_path,
            members=self.members,
            bare_contributions={
                atom: dict(sources) for atom, sources in self.bare_contributions.items()
            },
            uses=list(self.uses),
        )

    def contribute_bare(self, name: BareAtom, ref: BindingRef, layer: ContributionLayer) -> None:
        """Add one use-site-resolved bare contribution to this region, tagged with its *layer*."""
        add_layers(self.bare_contributions.setdefault(name, {}), ref, (layer,))

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


def anchored_layers(
    scope_nodes: Mapping[ScopePath, ScopeNode], step: ScopePath, path: ScopePath
) -> Iterator[tuple[ScopeNode, BareAtom]]:
    """Yield each layer anchored at or above *step*, outermost first, with *path* relative to it."""
    layer: ScopeNode | None = scope_nodes[step]
    chain: list[ScopeNode] = []
    while layer is not None:
        chain.append(layer)
        layer = layer.parent
    for anchored in reversed(chain):
        yield anchored, to_bare_atom(path[len(anchored.scope_path) :])


def binding_qname(ref: BindingRef) -> QName:
    """Return the declaring module and atom that *ref* names."""
    return ref.module_id, to_bare_atom((*ref.scope_path, ref.name))


def declaration_qname(key: DeclarationKey) -> QName:
    """Return the full path declaration *key* names."""
    module_id, scope_path, name = key
    return module_id, to_bare_atom((*scope_path, name))


def qname_declaration(qname: QName) -> DeclarationKey:
    """Return the declaration identity a full path *qname* names directly."""
    module_id, atom = qname
    path = to_bare_path(atom)
    return (module_id, path[:-1], path[-1])


def relative_under(atom: BareAtom, target: ScopePath) -> ScopePath | None:
    """Return *atom*'s path relative to *target*, or ``None`` if it is not under it.

    The one definition of the prefix test every reading of a contributed
    surface performs when it re-spells it relative to a path.
    """
    path = to_bare_path(atom)
    return path[len(target) :] if path[: len(target)] == target else None


def atom_under_prefix(atom: BareAtom, prefix: ScopePath) -> bool:
    """Whether *atom* falls under a selection *prefix* (a use tail or hiding item)."""
    return relative_under(atom, prefix) is not None


# ---------------------------------------------------------------------------
# ModuleResolution — output of the scope pass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AttributeFacts:
    """The typed facts one module's recognized declaration attributes carry.

    Built by ``scope.attributes.recognize_attributes`` and carried whole on
    :class:`ModuleResolution`, so a new kind of recognized fact costs one more
    table here rather than another field on every layer between.

    ``param_zones``
        The zone of every parameter and field, keyed by ``Param.node_id``, as
        the declaration's ``@arg-*`` attributes resolved it. Typecheck builds
        every ``ParamSpec`` and constructor field list from this table; the
        AST itself carries no zone.
    ``extern_names``
        The Python companion name of every ``extern def``, keyed by
        ``FuncDef.node_id``: its ``@extern-name`` argument, or its declared
        name verbatim. Two externs of one module never share an entry value.
        Lowering and companion resolution read this table.
    ``program_options``
        The command-line presentation of every ``program def`` parameter,
        keyed by ``Param.node_id``, as its ``@opt-*`` attributes describe it.
        Every program parameter has an entry; the host builds a program's CLI
        from these rather than from declared names.
    ``params``
        The host presentation of every static ``let`` or ``var`` marked
        ``@param``, keyed by its binding node id. Bindings without the marker
        have no entry.
    ``command_registrations``
        The command path every ``program def`` registers itself as through
        ``@command``, keyed by ``FuncDef.node_id``. Only a program carrying
        ``@command`` has an entry; the package domain merges these into a
        manifest's command table, pairing each with the program's ``@doc``.
    ``docs``
        The ``@doc`` text of every declaration carrying one, parameters and
        fields included, keyed by that declaration's node id.
    ``external_names``
        The ``@name``/``@json-name`` spellings of every field, enum member,
        and record declaration carrying one, keyed by that declaration's node
        id. Typecheck stores them on the type table.
    ``program_configs``
        The raw ``key = value`` entries of every ``program def``'s ``@config``
        attribute, keyed by ``FuncDef.node_id``. Keys resolve through the
        ordinary resolution tables, like any other reference. A program
        without ``@config`` has no entry.
    """

    param_zones: dict[int, ParamZone] = field(default_factory=dict)
    extern_names: dict[int, str] = field(default_factory=dict)
    program_options: dict[int, ProgramOptionSpec] = field(default_factory=dict)
    params: dict[int, ProgramOptionSpec] = field(default_factory=dict)
    command_registrations: dict[int, str] = field(default_factory=dict)
    docs: dict[int, str] = field(default_factory=dict)
    external_names: dict[int, ExternalName] = field(default_factory=dict)
    program_configs: dict[int, tuple[AttributeKeyedArg, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModuleResolution:
    """Output of the scope resolution pass.

    ``program``
        The module's ``Program`` with every operator chain grouped: the parsed
        one itself when it applies no user operator.
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
    ``static_root``
        Whether this module enforces the static-root binding rule: false for
        a REPL entry (whose root retains executable items instead of
        enforcing a static module root) or one carrying a host-synthesized
        entry (an inline command); true for every other module, including a
        file with its own ``program def``. Governs relaxed forward-reference
        order for module-level bindings and the narrow-var/constant-initializer
        rules.
    ``origin_path``
        This module's canonical source file, or ``None`` for a module with no
        backing file (inline sources, REPL entries). Later passes consult it to
        phrase a diagnostic for the host the module actually came from.
    ``declarations``
        Every named declaration keyed by ``(module_id, scope_path, name)``.
        Root declarations use the empty path just like any other scope.
    ``scope_nodes``
        Named scope-region layers keyed by path, rooted at ``()``.
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
        named owner is unavailable, so an unqualified spelling never
        substitutes for it. Candidates are independent of ordinary value bindings; the
        checker selects a bare name's final interpretation from the matched
        occurrence's type and field name.
    ``is_test_constructor_candidates``
        Maps ``is`` test node ids to every visible constructor candidate for
        their source spelling, empty as for constructor patterns when a
        qualified owner is unavailable. Typecheck selects by the left
        operand's nominal type.
    ``scope_qualified_spellings``
        Qualified pattern and ``is`` test node ids whose qualifier names a
        local plain scope or is a module qualifier. Their constructor
        selection is complete: typecheck rejects a spelling no published
        constructor fits against the matched type and never reads the
        qualifier as a type owner or module route.
    ``pattern_slots``
        Scope-created field-directed pattern-slot metadata keyed by slot id.
        Branch-body references resolve directly to the shared slot binding.
    ``match_site_pattern_slots``
        Maps each owning case-branch or ``let`` declaration node id to the
        slot ids its pattern created, in creation (outer-to-inner) order. The
        checker selects exactly these after that match site is classified.
    ``method_declarations``
        Maps each method's structured declaration identity to its resolved
        :class:`ReceiverOwner`. This is scope's definitive receiver
        classification; later passes consume it without re-deriving whether a
        function is a method.
    ``reachable_declarations``
        Declaration identities this module declares or reaches through imports.
        Import contributions already reflect route selection and ``hiding``;
        bare-only ``use`` declarations do not contribute identities.
    ``attributes``
        The typed facts this module's declaration attributes carry —
        parameter zones, extern companion names, program-parameter command-line
        presentation, and documentation text. See
        :class:`~agm.agl.scope.attributes.AttributeFacts`, which describes each
        table; typecheck, lowering, and the host read them from there.
    ``type_owners``
        The :class:`TypeOwner` each type the module declares resolved to, keyed
        by type path: a REPL session retains an entry's for later entries, and
        a later resolution reusing the module reads its aliases' reached members.
    ``owner_declarations``
        Maps a node id to what scope selected for it (:data:`TypeSelection`)
        -- the one verdict every position shares: a qualified chain's node id
        (the ``QualifierChain`` on a ``VarRef``/``NameTarget``/``NameT``/
        ``AppliedT``/``ConstructorPattern``/``IsTest``/``VariantRef``) to what
        its whole path selects, and a bare type name's node (``NameT``/
        ``AppliedT``, catch clause, ``exception ... extends`` declaration) to
        the type it selects. A chain selecting an inline member of the type
        its last qualifier segment selects records that owner
        (:class:`OwnerMemberSelection`); any other selection records the
        declaration (:class:`DeclarationSelection`). Typecheck and the REPL
        read every named type and owner from here, never re-resolving a name.
    ``replaced_uses``
        Maps each retained ``use`` a REPL entry replaces to the entry's uses
        replacing it: those at the same scope path renaming to its name, or,
        for a use renaming nothing, renaming nothing and reaching the same
        scopes and types -- unless spelled through the retained use's name.
    ``declared_segments``
        The segments of every path the module declares -- a REPL session's
        retained ones included -- the :class:`~agm.agl.modules.ids.Reader`
        data a diagnostic spelling another module's declaration anchors by.
    ``scope_entity_kinds``, ``import_decl_scope_paths``, ``type_declarations``,
    ``scoped_constructor_candidates``, ``injected_constructors``
        What each own full path declares (a scope, a type or an ordinary
        declaration), each import declaration's region, each type declaration
        with the scope path it is declared in, the constructors each scoped
        spelling names, and the enum members injected bare at each step: with
        the tables above, what a later resolution reusing the module reads
        its paths through, as at an alias's declaration.
    ``fixities``
        The fixity this module declares for each operator name: what an
        operator selecting one of the module's declarations of that name
        groups by.
    """

    program: Program
    resolution: dict[int, BindingRef]
    builtin_calls: dict[int, BuiltinKind]
    root_scope: ScopeNode
    builtin_static_calls: dict[int, BuiltinStaticKind] = field(default_factory=dict)
    declarations: dict[DeclarationKey, BindingRef] = field(default_factory=dict)
    scope_nodes: dict[ScopePath, ScopeNode] = field(default_factory=dict)
    static_root: bool = False
    origin_path: Path | None = None
    declared_type_paths: frozenset[ScopePath] = frozenset()
    constructor_candidates: dict[str, tuple[ConstructorRef, ...]] = field(default_factory=dict)
    constructor_refs: dict[int, ConstructorRef] = field(default_factory=dict)
    pattern_constructor_candidates: dict[int, tuple[ConstructorRef, ...]] = field(
        default_factory=dict
    )
    is_test_constructor_candidates: dict[int, tuple[ConstructorRef, ...]] = field(
        default_factory=dict
    )
    scope_qualified_spellings: frozenset[int] = frozenset()
    pattern_slots: dict[int, PatternSlot] = field(default_factory=dict)
    match_site_pattern_slots: dict[int, tuple[int, ...]] = field(default_factory=dict)
    method_declarations: dict[DeclarationKey, ReceiverOwner] = field(default_factory=dict)
    reachable_declarations: frozenset[DeclarationKey] = frozenset()
    attributes: AttributeFacts = field(default_factory=AttributeFacts)
    type_owners: dict[ScopePath, TypeOwner] = field(default_factory=dict)
    owner_declarations: dict[int, TypeSelection] = field(default_factory=dict)
    replaced_uses: dict[int, frozenset[int]] = field(default_factory=dict)
    declared_segments: frozenset[str] = frozenset()
    scope_entity_kinds: dict[DeclarationKey, str] = field(default_factory=dict)
    import_decl_scope_paths: dict[int, ScopePath] = field(default_factory=dict)
    type_declarations: tuple[
        tuple[RecordDef | EnumDef | ExceptionDef | TypeAlias, ScopePath], ...
    ] = ()
    scoped_constructor_candidates: dict[tuple[ScopePath, str], tuple[ConstructorRef, ...]] = field(
        default_factory=dict
    )
    injected_constructors: dict[tuple[ScopePath, str], tuple[ConstructorRef, ...]] = field(
        default_factory=dict
    )
    fixities: dict[str, Fixity] = field(default_factory=dict)

    def receiver_owner_for(self, module_id: ModuleId, node: FuncDef) -> ReceiverOwner | None:
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


class NoVisibleConstructorError(AglScopeError):
    """A bare pattern or ``is`` spelling that no visible constructor has."""


@dataclass(frozen=True, slots=True)
class ImportedModuleOrigin:
    """An ambiguity candidate an import -- a root or region-scoped tail -- contributes.

    ``declaration`` is the selected candidate's own ``(module, path)``; the
    module it was imported from is ``declaration[0]``.
    """

    declaration: QName


@dataclass(frozen=True, slots=True)
class UseDeclarationOrigin:
    """An ambiguity candidate a ``use`` declaration contributes.

    ``declaration`` names the contributed declaration itself -- not the
    ``use`` node that contributed it.
    """

    declaration: QName


@dataclass(frozen=True, slots=True)
class DeclaredOrigin:
    """An ambiguity candidate the reading module declares and sees directly."""

    declaration: QName


QualificationOrigin: TypingTypeAlias = ImportedModuleOrigin | UseDeclarationOrigin | DeclaredOrigin

_LAYER_ORIGINS: Mapping[ContributionLayer, Callable[[QName], QualificationOrigin]] = {
    ContributionLayer.DECLARED: DeclaredOrigin,
    ContributionLayer.USE: UseDeclarationOrigin,
    ContributionLayer.IMPORTED: ImportedModuleOrigin,
}


def contribution_origin(declaration: QName, layer: ContributionLayer) -> QualificationOrigin:
    """Return the ambiguity origin of candidate *declaration*, which *layer* made visible."""
    return _LAYER_ORIGINS[layer](declaration)


def contribution_origins(declaration: QName, layers: Layers) -> tuple[QualificationOrigin, ...]:
    """Return one origin of *declaration* per layer in *layers*, in layer order."""
    return tuple(contribution_origin(declaration, layer) for layer in layered(layers))


_ORIGIN_KIND_LABELS: tuple[tuple[type, str], ...] = (
    (ImportedModuleOrigin, "imported"),
    (UseDeclarationOrigin, "use-contributed"),
    (DeclaredOrigin, "locally declared"),
)


def _origin_spelling(origin: QualificationOrigin, *, reader: Reader) -> str:
    """Spell *origin*'s declaration the way *reader* would type it."""
    module, path = origin.declaration
    return spell_declaration(module, to_bare_path(path), reader=reader)


def _join_list(items: Sequence[str], *, comma_pair: bool) -> str:
    """Join *items* as an English list.

    *comma_pair* forces a comma before the final "and" even at two items,
    for joining whole clauses rather than a clause's own word list.
    """
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]}, and {items[1]}" if comma_pair else f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _render_ambiguity(
    spelling: str, origins: Iterable[QualificationOrigin], *, reader: Reader
) -> tuple[tuple[QualificationOrigin, ...], str]:
    """Order *origins* and state which declarations *spelling* selects.

    Deduplicates *origins* and orders them by kind, then spelling, so the
    result is deterministic regardless of hash seed. Each declaration is
    spelled the way *reader* would type it
    (:func:`~agm.agl.modules.ids.spell_declaration`).
    """

    def sort_key(origin: QualificationOrigin) -> tuple[int, str]:
        rank = next(
            index
            for index, (kind, _label) in enumerate(_ORIGIN_KIND_LABELS)
            if isinstance(origin, kind)
        )
        return (rank, _origin_spelling(origin, reader=reader))

    ordered = tuple(sorted(set(origins), key=sort_key))
    clauses = [
        f"{label} "
        + _join_list(
            [
                _origin_spelling(origin, reader=reader)
                for origin in ordered
                if isinstance(origin, kind)
            ],
            comma_pair=False,
        )
        for kind, label in _ORIGIN_KIND_LABELS
        if any(isinstance(origin, kind) for origin in ordered)
    ]
    return ordered, f"'{spelling}' is ambiguous: it selects {_join_list(clauses, comma_pair=True)}."


class AmbiguousQualificationError(AglScopeError):
    """A qualified or bare spelling that selects more than one declaration.

    ``origins`` are the contributing declarations -- imported, ``use``
    -contributed, or directly declared -- kept as structured data -- never as
    message text -- so a caller can tell one kind of collision from another.
    Built through :meth:`for_origins`, which renders the verdict with
    :func:`_render_ambiguity`, as :class:`AmbiguousConstructorError`'s
    builder does.
    """

    def __init__(
        self,
        message: str,
        *,
        spelling: str,
        origins: tuple[QualificationOrigin, ...],
        span: SourceSpan,
    ) -> None:
        super().__init__(message, span=span)
        self.spelling = spelling
        self.origins = origins

    @staticmethod
    def for_origins(
        route: tuple[str, ...],
        member_path: tuple[str, ...],
        origins: Iterable[QualificationOrigin],
        *,
        anchored: bool = False,
        span: SourceSpan,
        reader: Reader,
    ) -> AmbiguousQualificationError:
        """Build from *origins*, ending with the source-level repairs.

        *route* and *member_path* spell the ambiguous reference structurally
        instead of as a pre-rendered string: an empty *route* is a bare name,
        spelled by *member_path* alone; a non-empty *route* is a module route
        (:func:`~agm.agl.modules.ids.render_route_member`).
        """
        spelling = (
            render_route_member(route, member_path, anchored=anchored)
            if route
            else spell_scope_path(member_path)
        )
        ordered, statement = _render_ambiguity(spelling, origins, reader=reader)
        return AmbiguousQualificationError(
            f"{statement} {qualification_repair_guidance()}",
            spelling=spelling,
            origins=ordered,
            span=span,
        )


class AmbiguousConstructorError(AmbiguousQualificationError):
    """A constructor spelling several distinct types declare or contribute.

    Denotes the same verdict as :class:`AmbiguousQualificationError` -- one
    spelling selects more than one declaration -- found by matching a bare
    or module-surface constructor spelling against every visible type,
    rather than by a qualified route. ``repair`` is one qualified spelling
    that selects a single candidate, written through the same module
    qualifier when the ambiguous spelling has one.
    """

    def __init__(
        self,
        message: str,
        *,
        spelling: str,
        origins: tuple[QualificationOrigin, ...],
        repair: str,
        span: SourceSpan,
    ) -> None:
        super().__init__(message, spelling=spelling, origins=origins, span=span)
        self.repair = repair

    @staticmethod
    def for_constructor_origins(
        spelling: str,
        origins: Iterable[QualificationOrigin],
        *,
        repair: str,
        span: SourceSpan,
        reader: Reader,
    ) -> AmbiguousConstructorError:
        """Build from *origins*, ending with *repair*, a spelling that selects one candidate."""
        ordered, statement = _render_ambiguity(spelling, origins, reader=reader)
        return AmbiguousConstructorError(
            f"{statement} Qualify the reference, e.g. '{repair}'.",
            spelling=spelling,
            origins=ordered,
            repair=repair,
            span=span,
        )


@dataclass(frozen=True, slots=True)
class Ambiguity:
    """An ambiguous bare spelling's verdict kept as data, to report wherever it is read.

    ``origins`` are what it selects; ``repair`` is set when they are all
    constructors (:class:`AmbiguousConstructorError`).
    """

    origins: tuple[QualificationOrigin, ...]
    repair: str | None

    @staticmethod
    def of(error: AmbiguousQualificationError) -> Ambiguity:
        """The verdict *error* reports."""
        return Ambiguity(
            error.origins, error.repair if isinstance(error, AmbiguousConstructorError) else None
        )

    def error(self, name: str, span: SourceSpan, *, reader: Reader) -> AmbiguousQualificationError:
        """The error for bare *name*, read at *span*, selecting these origins."""
        if self.repair is None:
            return AmbiguousQualificationError.for_origins(
                (), (name,), self.origins, span=span, reader=reader
            )
        return AmbiguousConstructorError.for_constructor_origins(
            name, self.origins, repair=self.repair, span=span, reader=reader
        )


class DuplicateDeclarationError(AglScopeError):
    """A module declaring, or re-exporting, one name twice at one path.

    *name* spells the path as written. *reexports* spells each distinct
    declaration a re-export forwards under that name, when a re-export
    conflict is the duplicate.
    """

    def __init__(
        self, name: str, *, reexports: tuple[str, ...] = (), span: SourceSpan | None
    ) -> None:
        detail = f": it re-exports {_join_list(reexports, comma_pair=False)}" if reexports else ""
        super().__init__(f"Name '{name}' is already declared in this scope{detail}.", span=span)
        self.name = name
        self.reexports = reexports


def _spelling_hint(spelling: str) -> str:
    """Return the dollar-spacing hint for *spelling*'s last segment, or nothing."""
    return dollar_spacing_hint(spelling.rsplit("::", 1)[-1]) or ""


class MissRepair(enum.Enum):
    """The concrete repair a miss verdict suggests, chosen where the miss is decided."""

    NOT_EXPORTED = "Its module exports no declaration of that name."
    IMPORT_MODULE = "Import its module before using it."


def _repair_text(repair: MissRepair | None) -> str:
    """Return *repair*'s sentence, space-led, or nothing."""
    return "" if repair is None else f" {repair.value}"


class UnknownQualifierError(AglScopeError):
    """A qualifier naming no module route, scope region, or type owner.

    *qualifier* is spelled as written; every miss shares this one message,
    with *repair* when the decision knows one, and :func:`_spelling_hint`.
    """

    def __init__(
        self, qualifier: str, *, span: SourceSpan | None, repair: MissRepair | None = None
    ) -> None:
        super().__init__(
            f"Unknown qualifier '{qualifier}': no imported module route, scope, or type "
            f"is named so here.{_repair_text(repair)}{_spelling_hint(qualifier)}",
            span=span,
        )
        self.qualifier = qualifier
        self.repair = repair


class UnknownMemberError(AglScopeError):
    """A qualifier that resolves, but whose full spelling selects no member.

    *spelling* is the qualified name as written; every miss shares this one
    message, with *repair* when the decision knows one, and
    :func:`_spelling_hint`.
    """

    def __init__(
        self, spelling: str, *, span: SourceSpan | None, repair: MissRepair | None = None
    ) -> None:
        super().__init__(
            f"'{spelling}' names no visible member.{_repair_text(repair)}"
            f"{_spelling_hint(spelling)}",
            span=span,
        )
        self.spelling = spelling
        self.repair = repair


class SpacedQualifierError(AglScopeError):
    """A member reference whitespace split from its qualifier; *repair* is the tight spelling."""

    def __init__(self, member: str, repair: str, *, span: SourceSpan | None) -> None:
        super().__init__(
            f"Whitespace before '::{member}' makes this a call with a self-reference, "
            f"not a qualifier. Write '{repair}' without whitespace.",
            span=span,
        )
        self.repair = repair


class TypeArgumentsError(AglScopeError):
    """A qualifier segment carrying type arguments its full path does not take.

    A segment carries them written (``Box[int]``) or by selecting an applied
    alias (``IntBox``). It takes them only when it selects a generic type
    whose inline member the next segment selects, and exactly as many as that
    type declares; a declaration's path takes none, but a builtin receiver's
    bare generic head. *segment* is the segment as spelled; *arity* the
    owner's type parameter count, ``None`` when the segment selects no such
    owner.
    """

    def __init__(self, segment: str, arity: int | None, *, span: SourceSpan) -> None:
        reason = (
            "it selects no generic type owning the next segment as an inline member"
            if arity is None
            else f"its type takes {arity} type argument(s)"
        )
        super().__init__(
            f"Type arguments cannot be applied to segment '{segment}': {reason}.", span=span
        )
        self.segment = segment
        self.arity = arity
